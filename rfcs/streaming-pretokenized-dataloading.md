# RFC: Streaming dataloading for pretokenized SFT datasets at trillion-token scale

- **Status**: Draft
- **Author**: @avigyabb
- **Related work**: #1927 (pretokenized ingestion, merged), #1933 (cloud paths, draft)

## Summary

Add bounded-memory dataloading to `SFTTrainer` for pretokenized datasets in cloud storage
(S3 required, GCS nice-to-have), up to trillion-token scale. Two approaches carry the
proposal, deployed in sequence behind the same library and loader entry point:

1. **Disk-based (map-style)** — download the store to controller NVMe once, then train from
   a memory-mapped arrow `Dataset` instead of a materialized `list[dict]`. Zero trainer
   changes; full sampler and resume semantics; covers datasets up to one node's disk.
2. **HF `datasets` streaming (`IterableDataset`)** — stream shards from S3 through
   `StatefulDataLoader` workers. Bounded by neither RAM nor disk; requires an
   offline-shuffled, sharded layout and a small set of documented trainer-semantics
   changes.

Because both are HuggingFace `datasets` behind `StatefulDataLoader`, streaming is a flag
flip on top of the disk-based change, not a second implementation. Other options (Ray Data,
Mosaic, bespoke streamer, Megatron `.bin`/`.idx`) are dispatched concisely at the end; the
bespoke streamer remains the measured-fallback escape hatch.

## Requirements

1. Scale to trillion-token datasets with bounded-memory dataloading.
2. Load pretokenized datasets from cloud storage (S3 required, GCS good-to-have).
3. Prefetch upcoming data (download, decode, collation) overlapped with the training step.
4. Efficient resume from a checkpoint: restart at sample N in seconds, without replaying or
   re-downloading the prefix.

### Non-requirements

- Online tokenization at this scale (data is pretokenized offline; see #1927 for the schema:
  unpadded `input_ids` + full-sequence 0/1 `loss_mask`).
- Bit-identical batch boundaries when resuming under a *different* topology
  (`num_workers`, DP size). The bar is: deterministic given a fixed config, and no sample is
  repeated or skipped across a same-config resume.
- Multi-epoch semantics in streaming mode. At 1T tokens, runs are sub-epoch; epoch-derived
  step counts (`num_epochs`) do not apply there. (The disk-based path keeps epochs.)

## Background: current architecture

`SFTTrainer` runs a **single controller** (a Ray task) that owns dataloading:

```
list[dict] (fully materialized)          ← dataset (map-style)
  + sampler (random / DataMixingSampler) ← index generator
  → StatefulDataLoader                   ← batching, worker processes, checkpoint state
  → collator (padding or FFD packing)    → TrainingInputBatch
  → WorkerDispatch                       → shards the global batch to DP workers
```

Checkpoint/resume persists `StatefulDataLoader.state_dict()` into `data.pt`. #1927 added
pretokenized ingestion into the materialized list; #1933 adds cloud download of whole stores.
Neither scales past what fits in controller memory: at 1T tokens (~1B rows, ~4 TB of int32
token data before Python object overhead), full materialization is off the table.

## Throughput analysis: why the controller model survives

Training, not I/O, is the bottleneck by ~2 orders of magnitude. A healthy multi-node SFT run
consumes ~0.5–1M tokens/s; at 4 bytes/token that is **2–4 MB/s of sustained ingest**, versus
&gt;200 MB/s from a single S3 prefetch stream and multi-GB/s from local NVMe. Consequences:

- The existing single-controller load→collate→dispatch model needs no re-architecture.
- Distributed reading frameworks solve a problem we do not have.
- The hard problems are **bounded memory**, **shuffle quality without a global shuffle**, and
  **O(1) resume** — not bandwidth.

## Data layout guidance

The offline pretokenization pipeline should produce:

1. **Fixed-size shards** (~128–512 MB parquet, ~100k–500k rows each) — never one giant file.
2. **Offline shuffle at write time** (global, or at minimum shard-internal). Load-bearing
   for the **streaming** path only: with rows pre-shuffled, the online loader shuffles just
   *shard order* (seeded, per pass) and reads sequentially within a shard, eliminating online
   shuffle buffers — the single largest source of complexity and checkpoint-state pain in
   streaming loaders. The disk-based path shuffles via its sampler and does not depend on it.

## Approach 1 — Disk-based: NVMe download + memory-mapped map-style dataset

"Doesn't fit in memory" does not imply "must stream": HF's *non-streaming* `Dataset` is
arrow data memory-mapped from disk, not loaded into RAM. Today's ingestion path already
builds that arrow-backed `Dataset` — and then throws the memory mapping away by
materializing `list[dict]`. The materialization, not the format, is what doesn't scale.

### Architecture

```
S3 store ──(one-time, shard-parallel download to controller NVMe)──▶ local parquet shards
  → arrow conversion (per-shard, cached)      ← load_dataset(..., streaming=False)
  → Dataset (memory-mapped, zero rows in RAM)
  → set_transform(normalize_row)              ← lazy per-row decode, reusing #1927 validation
  → sampler (random / DataMixingSampler / custom)   ← unchanged
  → StatefulDataLoader(num_workers=N, prefetch_factor=k)
  → collator (padding / FFD packing) → WorkerDispatch   ← unchanged
```

Only the **controller** needs the local copy — workers receive dispatched batches — so the
disk requirement is one node's NVMe. Single-digit TB covers a large fraction of real
workloads before true streaming is needed.

### Prefetching and the I/O path

There is no application-level prefetcher to build; "prefetch" is three existing layers
stacked:

1. **Bulk download (one-time, before training)** — shard-parallel ranged GETs from #1933's
   whole-store path. With enough parallel streams S3 saturates the instance NIC, so a
   single-digit-TB store lands in tens of minutes to low hours depending on NIC class. A
   random sampler needs the whole dataset resident before step 1, so this phase is not
   overlapped with training; it *is* amortizable — the local copy survives across runs on the
   same node, so only fresh nodes pay it. It can also be taken **offline entirely**: run the
   download (and optionally the arrow conversion) as a pre-step outside the training job —
   e.g. at the end of the pretokenization pipeline, in node provisioning, or as a separate
   job that warms the NVMe before the trainer launches — so training starts against an
   already-populated local copy and pays zero download time.
2. **Dataloader worker prefetch** — `StatefulDataLoader` keeps `num_workers ×
   prefetch_factor` batches in flight. Workers index the mmap'd dataset, take the page-fault
   and decode cost in their own processes, and hand ready batches to the main process. This
   is the layer that overlaps I/O + decode + collation with the training step (Requirement 3),
   exactly as it does today.
3. **mmap + OS page cache** — a random access pulls in the pages holding that record batch;
   the hot set stays cached. Random sampling defeats sequential readahead, but NVMe random
   reads (~100 µs, multi-GB/s aggregate) leave 3+ orders of magnitude of headroom over the
   2–4 MB/s the trainer consumes.

**Disk sizing caveat**: `load_dataset` keeps the downloaded parquet *and* writes an
uncompressed arrow cache, so peak footprint is roughly parquet + arrow (~1.5–2.5× token
bytes). Mitigation if it matters: convert shard-at-a-time and delete parquet as arrow lands,
or point the arrow cache and download dir at the same NVMe volume sized for both.

### What it preserves

- **Zero trainer changes.** Samplers (`random`, `DataMixingSampler`, custom/curriculum),
  `len()`/epoch derivation, tail-batch padding, `_log_dataset_stats` — all keep working
  verbatim, because the dataset is still map-style.
- **Exact, trivial resume**: sampler position via the existing `data.pt` /
  `StatefulDataLoader.state_dict()` flow, unchanged.
- **True global shuffle** every epoch, not shard-order approximation.

### Limits (why it is the first rung, not the destination)

- Dataset size capped by a single node's disk — ~4 TB of tokens (≈1T) is the practical
  ceiling, and the arrow-cache multiplier eats into it.
- The download is re-paid on every fresh controller node — painful under preemption and
  autoscaling.
- Pins the controller to fat-disk node types.

### Scaling past instance NVMe: network block storage (EBS)

The disk ceiling and the re-download cost can both be pushed back by swapping instance NVMe
for network-attached block storage (EBS on AWS; equivalents elsewhere):

- **Capacity**: gp3 volumes go to 16 TiB each and can be striped (RAID-0) for more, so the
  "one node's disk" ceiling becomes a provisioning knob rather than a hardware limit.
- **Persistence fixes the operational weakness**: an EBS volume outlives the instance. On
  preemption or node replacement, reattach the volume to the new controller instead of
  re-downloading — the download becomes truly once per *dataset*, not per node, and composes
  with the offline pre-warm above (populate a volume once, snapshot it, attach clones to any
  future controller).
- **Performance cost is real but survivable**: EBS random reads are ~0.5–1 ms versus
  ~100 µs on NVMe, and throughput is capped per volume (gp3: 1,000 MB/s, 16k IOPS baseline;
  io2 goes higher for more money). That is a ~10× latency and ~10× throughput haircut
  against local NVMe — but the trainer consumes 2–4 MB/s, so even EBS leaves 2+ orders of
  magnitude of headroom. The latency hit lands on dataloader workers during page faults,
  which is exactly the cost `num_workers × prefetch_factor` of runway exists to hide; if
  stalls appear, more workers is the first knob.
- **One sharp edge**: volumes restored from a snapshot are lazily hydrated — cold blocks are
  fetched from S3 on first touch, so the first pass over a fresh clone can crawl. Either
  pre-warm (fast snapshot restore, or a sequential read of the dataset files) or accept a
  slow first epoch.

Net: EBS trades steady-state I/O margin we aren't using for capacity and persistence we
want. It stretches Approach 1 meaningfully — but past ~10s of TiB, volume cost and
management overhead grow while streaming's cost stays flat, which is where the decision rule
below tips.

## Approach 2 — HF `datasets` streaming (`IterableDataset`)

Correcting an earlier scoping note ("don't think it supports cloud storage"): it does.
`load_dataset("parquet", data_files=..., streaming=True, storage_options=...)` is
fsspec-based, so `s3://` works via `s3fs` (already a repo dependency) and `gs://` via `gcsfs`.

### Architecture

```
shard paths/globs → load_dataset(..., streaming=True, storage_options=...)   ← fsspec/s3fs
  → .shuffle(seed, buffer_size)            ← shard-order shuffle (+ small buffer)
  → .map(normalize_row)                    ← reusing #1927 row validation
  → [interleave_datasets(probabilities=…)] ← multi-store mixing
  → StatefulDataLoader(num_workers=N, prefetch_factor=k)
  → collator → WorkerDispatch              ← unchanged
```

### Why it fits

- **Composes with what we have.** An `IterableDataset` is a valid dataset for
  `StatefulDataLoader`; collators, FFD packing, dispatch, and the `data.pt` checkpoint flow
  are unchanged. This is a dataset swap, not a dataloader replacement — and it is the *same*
  swap as Approach 1 with `streaming=True`.
- **Prefetch overlap for free.** Shards are split across dataloader workers; each worker
  streams its shards sequentially (buffered fsspec reads, parquet row-group at a time), so
  download + decode + collate run in worker processes while the GPU trains, with
  `num_workers × prefetch_factor` batches of runway. Requirement 3 without new machinery.
- **Resume is already wired.** `IterableDataset` implements `state_dict()`/`load_state_dict()`
  (datasets ≥ 2.18), and `StatefulDataLoader` checkpoints/restores per-worker dataset state
  through its existing protocol. Resume lands at (shard, offset) and skips forward within one
  shard — cheap when shards are modest.
- **Multi-store mixing**: `interleave_datasets(streams, probabilities=weights)` replaces
  `DataMixingSampler` (which is map-style and cannot survive streaming).

### Known limitations, accepted

- Python-level decode throughput is unimpressive — irrelevant at 2–4 MB/s required ingest.
- Resume-by-skip within a shard costs one partial shard read.
- fsspec creds are used directly rather than the repo's custom S3 refresh layer.
- Exact determinism across restarts requires the same `num_workers` (shard→worker assignment
  is implicit state) — consistent with the stated non-requirement.

### Trainer integration changes (streaming mode only)

| Today (and Approach 1) | Streaming mode |
|---|---|
| dataset is map-style, materialized or mmap'd | an iterable; nothing materialized |
| `steps_per_epoch = len(dataloader)`; `num_epochs` supported | no `len()`; **`num_steps` required** |
| `_log_dataset_stats` scans all rows | samples a prefix |
| `sampler="random"` shuffle over indices | `.shuffle(seed, buffer_size)` on the stream |
| `DataMixingSampler` for multi-store weighting | `interleave_datasets(probabilities=...)` |
| `sampler="custom"` (curriculum etc.) | unsupported (documented; see note below) |
| tail-batch padding (`drop_last=False`) | moot: `num_steps`-bounded, never hits a tail |
| eval sets materialized | unchanged (eval is small) |
| `data.pt` = dataloader `state_dict` | unchanged mechanism; state is now shard cursors |

**Custom-sampler note.** The existing sampler API produces arbitrary integer indices into a
map-style dataset and therefore cannot drive an `IterableDataset`. A future streaming planner
would instead produce an ordered plan of shard IDs (optionally contiguous row ranges) from
a discovered shard list, partitioned and streamed by dataloader workers, with FFD packing
remaining online. Arbitrary sparse sample-level plans are intentionally out of scope: they
would need a per-row sidecar index and cause read amplification by opening many shards for a
few rows.

## Tradeoffs: disk-based vs streaming

| | Approach 1 — disk-based | Approach 2 — HF streaming |
|---|---|---|
| Scale ceiling | one node's NVMe (~4 TB tokens); stretchable to 10s of TiB with EBS at a latency/cost premium | unbounded |
| Time-to-first-batch | tens of minutes–hours (download + convert), amortized per node | seconds–minutes |
| Trainer changes | none | `num_steps`, sampler/mixing/stats changes per table above |
| Sampler support | full: random, `DataMixingSampler`, custom/curriculum | shard-order shuffle + `interleave_datasets`; custom unsupported |
| Shuffle quality | true global shuffle per epoch | offline shuffle + shard-order shuffle (approximation) |
| Resume | exact sampler position, O(1), unchanged mechanism | (shard, offset) cursors + partial-shard skip |
| Epochs / `len()` | preserved | gone; `num_steps` only |
| Steady-state I/O | NVMe random reads via mmap + page cache | sequential S3 reads in dataloader workers |
| Sensitivity to layout guidance | needs shards; offline shuffle optional | offline shuffle is load-bearing |
| Preemption / autoscaling cost | re-download on every fresh controller node | none beyond a partial shard |
| Node requirements | fat-disk controller | any controller |

### Decision checklist

The table compresses to five operational questions. Answer them for a given workload and the
choice usually falls out:

1. **Is the controller node stable/long-lived, or preemptible/recycled? Is a shared
   filesystem available?** Disk-based's one real weakness is the per-node download, so node
   churn is what decides its fate. Stable controllers amortize the download away;
   preemptible/recycled controllers re-pay it on every replacement, which compounds into
   streaming territory. Two middle grounds neutralize churn: a persistent EBS volume that
   reattaches to the replacement node (see the EBS section above), or a shared filesystem
   (FSx/NFS) already holding the store — with either, disk-based keeps all its semantic
   advantages with no re-download cost, and the case for streaming weakens considerably.
2. **Is the dataset ~1T flat, or growing — and will controller storage be provisioned ahead
   of that growth?** Budget concretely: 1T tokens ≈ 4 TB raw; with parquet + arrow double
   residency and working cushion, plan **~5–8 TB** of controller disk. If the dataset is
   flat, that is a one-time provisioning decision and disk-based is comfortable. If it grows,
   disk-based means perpetually re-provisioning (bigger NVMe node types, or growing/striping
   EBS) and re-downloading on each refresh of the copy — a treadmill. Streaming's footprint
   is flat regardless of dataset size, so a growing dataset is the strongest single argument
   for it.
3. **Do you need to select individual rows on the fly** — exact per-batch source ratios,
   dynamically picking specific examples, curriculum over samples — **or is steering at
   ~100k-row shard granularity (with any fixed ordering baked in offline) enough?** Row-level
   dynamic selection requires random access into a map-style dataset: that is disk-based
   only. Streaming's `interleave_datasets` holds mixing ratios in expectation, not exactly
   per batch, and custom samplers don't survive streaming at all (a shard/range planner is
   the documented future direction, not row-level). If shard-granular steering with
   offline-baked order suffices, both approaches qualify and this question drops out.
4. **How often is the pretokenized store refreshed?** Every refresh invalidates disk-based's
   local copies — each controller re-downloads and re-converts before the next run. A store
   refreshed quarterly makes that negligible; one refreshed weekly (rolling data updates,
   iterated tokenization) makes the download cost recurring rather than one-time, and
   streaming — which always reads the current store directly with zero staging — starts to
   dominate even below the disk ceiling.
5. **How many concurrent runs consume the same store?** Streaming scales out for free: N
   runs are N independent S3 readers, no copies anywhere. Disk-based needs the store local to
   each run's controller — either one copy per controller node (cost and download time ×N),
   controllers colocated on shared-disk nodes, a shared filesystem mount, or EBS snapshot
   clones fanned out to each controller. One or two long-lived runs favor disk-based; a fleet
   of short experimental runs over the same 1T store favors streaming.

**Net rule of thumb**: disk-based wins when the answers are *stable nodes (or shared
disk/EBS), flat dataset that fits ~5–8 TB, row-level control needed, infrequent refresh, few
concurrent runs*. Streaming wins as answers flip — and the more of them flip, the clearer it
gets. Since both are the same library behind the same entry point, the cutover is a config
flag, not a migration, so the choice can be made per-workload rather than once.

## Other options considered (rejected — kept brief)

- **Ray Data** — no sample-exact checkpoint/resume of a streaming iterator (Requirement 4
  would have to be rebuilt on top); built to parallelize per-row compute we moved offline;
  competes with training actors for cluster resources.
- **Mosaic `StreamingDataset`** — best-engineered tool in this space (S3/GCS native,
  manifest-driven deterministic order, instant mid-epoch resume, bounded shard cache), but
  rejected as a dependency per team preference. Its design is the reference for the bespoke
  fallback. Its per-rank model would run degenerate (world=1) under SkyRL's controller
  dispatch anyway.
- **Bespoke shard streamer (fallback, specified not built)** — with the layout guidance in
  place it is small (~400–600 lines): discovered shard list → seeded shard permutation per
  pass → background prefetcher of the next K shards into a bounded queue → sequential row
  iterator;
  resume state is four integers `(pass, shard_perm_seed, shard_index, row_offset)`. Same
  interface as Approach 2 (iterable + `state_dict` protocol feeding `StatefulDataLoader`).
  Built only if HF streaming misses measured targets — e.g. S3 tail-latency stalls worker
  prefetch cannot hide. Tractable precisely because the offline shuffle removes shuffle
  buffers, where homegrown loaders historically go wrong.
- **Megatron indexed dataset (`.bin`/`.idx`)** — strongest resume story (a step counter) and
  zero loader code, but it does not stream from S3 by itself: it needs a full local copy
  (Approach 1's costs with a less flexible format) or filesystem infrastructure
  (mountpoint-s3, FSx for Lustre) this RFC cannot assume, plus format converters for the
  #1927 per-sample `input_ids`/`loss_mask` schema. Revisit if the fleet standardizes on such
  mounts.

## Open questions

1. `interleave_datasets` checkpointability for multi-store mixing needs verification at
   current `datasets` pin (single-store resume is confirmed supported).
2. Loss-mask storage: `uint8` vs bit-packed in shards — 8× size difference on one column;
   decide in the layout guidance.
3. Interaction with `use_sequence_packing` (FFD packs per global batch): unchanged
   mechanically, but packing efficiency under streaming order should be spot-checked.
4. Disk-based path: avoid parquet + arrow double residency on NVMe (per-shard convert-then-
   delete vs sizing the volume for both)?
