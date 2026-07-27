# [RFC] Trillion-token SFT dataloading via row-level random access over cloud storage

# Summary

This RFC proposes row-level random-access dataloading for SkyRL SFT over pretokenized
datasets in cloud storage (S3). The trainer's sampler emits row indices exactly as it does
for a local dataset; the loader resolves each sampled index to the parquet row group
containing it via a footer-derived index, fetches exactly those row groups with coalesced
concurrent ranged reads, and the standard dataloader workers prefetch upcoming batches while
the GPU trains. The dataset is never downloaded or materialized: controller memory is a small
fixed cache plus one index, at any dataset size. Measured end-to-end, exposed data time is
**0.8 ms per 3.6 s training step** — statistically identical to training from local
memory-mapped files.

# Motivation

SFT over a ~1T-token pretokenized dataset cannot materialize rows in memory (≈4 TB of token
data before Python object overhead) and sits at or past the ceiling of a single node's disk
(≈5–8 TB on disk with format double-residency), which makes both the current materializing
loader and a download-to-NVMe approach fragile at this scale: every fresh controller node
and every dataset refresh re-pays a multi-hour download.

Throughput is not the constraint — a healthy multi-node SFT run consumes 2–4 MB/s of token
data against >200 MB/s available from S3 — the constraints are bounded memory, sampling
quality, and fast exact resume. Sequential shard-streaming designs satisfy the first and
third but give up row-level sampling semantics: exact per-batch mixing ratios across
sources, custom/curriculum samplers, true global shuffle, and `len()`/epoch accounting. This
RFC targets workloads that need those semantics at cloud scale.

# Goals

1. **Bounded-memory dataloading at trillion-token scale:** train against cloud-hosted
   pretokenized stores with controller memory independent of dataset size.
2. **Preserve map-style sampling semantics:** random sampling with true global shuffle,
   exact per-batch source mixing, custom samplers, `len()`/epochs — unchanged trainer
   behavior. [^1]
3. **Fully overlap data loading with the training step:** exposed per-step data time within
   noise of local-disk training, using the existing dataloader worker machinery.
4. **Fast, exact resume:** restart at the exact next sample in seconds via the existing
   `StatefulDataLoader` checkpoint flow, with no replay and no re-download.
5. **Writer-agnostic ingestion:** consume parquet produced by any pipeline (Ray Data, Spark,
   pyarrow) with no format conversion and no required sidecar files.

# Non-Goals

1. **Reactive sampling policies** (choosing the next row from the current step's training
   signal): prefetching requires the sampler's future indices to be knowable; staleness can
   never be less than the prefetch horizon. This holds for any remote-storage design; reactive
   policies require a local copy (Option 1 below).
2. **Minimizing read amplification:** this design deliberately pays ~`row_group_size`×
   amplification to buy row-level access (analysis below). Workloads content with shard-order
   sampling should use a sequential streaming mode instead.
3. **Bit-identical batch order across `dataloader_num_workers` changes:** the multiprocessing
   iterator consumes the seeded generator differently than the single-process one, so the
   shuffle order is deterministic per (seed, num_workers) but differs across worker counts.
   This matches existing SkyRL behavior. [^2]

# Functional Requirements

1. Resolve any global row index to its (shard, row group, row offset) in O(log n) with an
   index built from parquet footers only — no data reads, no manifest required.
2. Fetch all row groups needed by a batch with coalesced, concurrent ranged reads (no
   per-row-group round-trips), overlapped with the training step by dataloader workers.
3. Return rows byte-identical to the local (mmap) loader, including `max_length` truncation
   and lazy normalization to the trainer's internal schema.
4. Fail fast with the offending global row index on malformed rows (length mismatch, empty
   loss window). Cloud stores are assumed pre-filtered: a map-style dataset's length is fixed
   before sampling begins, so rows cannot be silently dropped at read time.
5. Checkpoint/resume through the existing `data.pt` / `StatefulDataLoader.state_dict()` flow
   with no new state format.

# Design

## Data layout requirements

The pretokenization pipeline (e.g. Ray Data `write_parquet`) keeps writing parquet to S3 with
three writer settings — no format change:

1. `row_group_size ≈ 128` (`arrow_parquet_args={"row_group_size": 128}`). The row group is
   parquet's smallest fetchable unit; at ~128 rows it is a few hundred KB, so a randomly
   sampled row costs a small ranged read rather than a multi-MB column chunk. The design's
   economics depend on this setting.
2. Explicit compression (`"compression": "zstd"`). Default writer configurations have been
   observed to produce uncompressed files at ~2.5× the size, doubling storage and every read.
3. Pre-filtered rows (non-empty loss window per row), per Functional Requirement 4.

An optional `manifest.json` (per-shard row counts, tokenizer/chat-template identifiers)
skips the startup footer scan and records provenance that no scan can derive; it is an
accelerator, never a gate.

## Overall loader flow

Startup (once per run):

1. List `*.parquet` under the store prefix (hidden files skipped).
2. Read each shard's footer (metadata-only ranged read): per-row-group row counts, schema
   check, exact token totals. No data bytes are read.
3. Build cumulative row-count arrays: global index → (shard, row group, offset) resolves via
   two binary searches.

Per batch (inside dataloader workers, `prefetch_factor × num_workers` batches ahead of
consumption):

1. Resolve the batch's row indices; group by (shard, row group); dedupe (one fetch per
   distinct row group regardless of how many sampled rows share it).
2. Per shard, issue one `read_row_groups(missing, use_threads=True)` on a pyarrow-native
   filesystem with `pre_buffer=True`: the byte ranges of all needed row groups are coalesced
   and fetched concurrently in C++ (no GIL, no per-group round-trips). Shards fetch in
   parallel via a small thread pool.
3. Cache decoded row groups in a bounded LRU (zero-copy slices of the coalesced read).
4. Slice the requested rows in sampler order; validate; apply the same lazy normalization
   transform as the local loader.

Everything above the dataset boundary — samplers, plan generation, `StatefulDataLoader`,
collators (including FFD packing), dispatch, checkpointing — is unchanged.

## Overlap analysis

Data loading is hidden whenever

```
(row_groups_touched_per_batch × S3_latency) / io_concurrency ≤ num_workers × step_time
```

`row_groups_touched ≈ batch_rows` under a shuffled sampler. Both sides scale with batch size
and neither scales with dataset size, so measured margins carry to 1T unchanged. Measured
per-batch fetch is ~1.5 s against a 3.6 s step: one worker sustains, two give 2.4× headroom;
`pyarrow.set_io_thread_count(16)` and `num_workers=2–4` give ~5–10×. `prefetch_factor` (torch
default 2) provides tail-latency runway and should be exposed in `SFTConfig`.

## Benchmarks

Real end-to-end training: FSDP, Qwen2.5-0.5B on 1×L4, batch 64×256 tokens (~3.6 s
compute/step), `prefetch_factor=2` where workers are enabled. All rows use the same 100k-row
store (local disk for the first two rows, S3 for the rest), except the final row, which runs
the proposed design against a **1-billion-row store** (10,000 shards, 7.8M row groups,
~1.2 TB on S3) fabricated by replicating the benchmark shard server-side — the access
pattern, index size, shard scatter, and shuffle permutation of a ~1T-token dataset.

| Configuration | Startup (data prep) | Exposed `timing/data_loading` | Step mean | Controller data memory |
|---|---|---|---|---|
| Previous loader: fully materialized `list[dict]`, local, workers=0 | 19.6 s | 2.5 ms | 3.57 s | O(dataset): 2.2 GB here, ×(workers+1) with workers; ≈4 TB+ at 1T |
| Local mmap dataset, workers=2 | 0.3 s | 0.8 ms | 3.63 s | O(page cache), reclaimable |
| Naive per-group S3 fetches (fsspec), workers=0 | ~0 s | 17,113 ms | 20.7 s | O(LRU) ≈ tens of MB |
| Naive per-group S3 fetches (fsspec), workers=2 | ~0 s | 5,107 ms | 8.8 s | O(LRU) |
| **Proposed: coalesced concurrent S3 fetch, workers=2** | **~0 s** | **0.8 ms** | **3.60 s** | **O(LRU)** |
| **Proposed, 1B-row / 10k-shard store, workers=2 (60-step soak)** | **223 s footer scan**[^3] | **0.8 ms mean, 1.2 ms max** | **3.65 s** | **O(LRU) + 65 MB index** |

The step mean is invariant (~3.6 s) across every configuration whose data path is off the
critical path — the previous materialized loader achieves it too, but only by paying startup
and memory that scale with the dataset (its 19.6 s / 2.2 GB at 100k rows extrapolate to hours
and terabytes at 1T, which is the scaling wall motivating this RFC). The 1B-row soak
confirms the scale-invariance claim directly: exposed data time is identical to the
100k-row case to the millisecond (max 1.2 ms across all steps — no step ever waited on S3,
including the earliest batches where every shard handle was cold), with only the predicted
one-time costs growing: the footer scan (223 s, 32-way parallel; ~0 with a manifest) and a
3.9 s first fetch containing the `randperm(1B)`. Row content is parity-tested byte-identical
across all three loaders, and identical-seed training reproduces the materialized path's
loss curve to the third decimal.

## Cost analysis at 1T tokens

1. **Read amplification:** each sampled row fetches its full ~128-row group ⇒ ~128× useful
   bytes; a full 1T-token run reads ~500 TB from S3 against 4 TB consumed. Same-region
   bandwidth is free; GET request charges are on the order of $1–2k per run; sustained
   NIC/decode load is ~100–500 MB/s at realistic batch sizes, fully hidden per the benchmarks.
2. **Startup footer scan:** ~1B rows at rg128 ⇒ ~8M row-group entries ≈ 2–3 GB of footer
   bytes over ~10k shards. Must be parallelized (~1–2 min at 32-way; currently sequential) and
   the derived index cached locally; a manifest makes it ~zero.
3. **Shuffle setup:** `randperm(1B)` ≈ 8 GB controller RAM and tens of seconds, once per run.
4. **Hardening before multi-week runs:** LRU cap on per-shard `ParquetFile` handles (parsed
   footers ~300 KB each at rg128); soak coverage for S3 credential rotation (pyarrow-native
   S3 uses the AWS SDK chain, not SkyRL's fsspec refresh layer) and tail-latency behavior.

# Other Design Options Considered

<details><summary><b>Option 1: Download to controller NVMe/EBS, memory-mapped map-style dataset (no streaming)</b></summary>

Download the store once to controller-local storage and train from memory-mapped arrow files.
Implemented and benchmarked: identical semantics and identical steady-state performance
(0.8 ms exposed, 3.63 s steps), and it additionally supports *reactive* sampling (Non-Goal 1)
since all rows are local.

Rejected as the primary option at 1T:

1. The dataset sits at the ceiling of one node's storage (~5–8 TB on disk with parquet+arrow
   double residency), pinning the controller to fat-disk node types.
2. The multi-hour download is re-paid on every fresh controller node (preemption,
   autoscaling) and on every dataset refresh; a growing dataset makes this a provisioning
   treadmill. EBS/shared-FS persistence mitigates node churn but not refresh or growth.

| | This RFC (S3 row-level) | Option 1 (disk mmap) |
|---|---|---|
| Scale ceiling | none | ~one node's disk (~1T tokens practical) |
| Time to first batch | seconds–minutes | hours per fresh node (download + convert) |
| Dataset refresh / growth | reads current store; flat footprint | full re-download; re-provisioning |
| Sampling semantics | full, plan-ahead only | full, including reactive |
| Steady-state reads | ~128× amplified S3, fully hidden | local NVMe, fully hidden |

Both options share the loader entry point, so the choice is a config flag per workload; below
disk scale with stable nodes and frozen data, Option 1 is the recommendation.

</details>

<details><summary><b>Option 2: MosaicML StreamingDataset</b></summary>

The strongest off-the-shelf tool in this space: S3/GCS-native, manifest-driven deterministic
sample order, instant mid-epoch resume, bounded local shard cache, production-proven at
multi-week scale, ~1× read amplification (sequential shard reads).

Rejected:

1. **Format migration:** requires converting the 1T-token parquet store to Mosaic's MDS
   format — a full rewrite plus a second copy to maintain — versus consuming the parquet the
   pipeline already writes (Goal 5).
2. **Sampling model:** shard-order shuffle with bounded windows; exact per-batch mixing,
   custom samplers, and true global random access are outside its model — the requirement
   motivating this RFC (Goal 2).
3. **Architectural mismatch:** built around per-GPU-rank dataloaders; under SkyRL's
   central-controller dispatch it runs degenerate (world size 1), keeping the dependency and
   its separate resume model while discarding its distributed machinery.
4. **Dependency:** a training-critical third-party dependency versus ~500 lines on pyarrow
   (already a dependency).

| | This RFC (S3 row-level) | Option 2 (Mosaic) |
|---|---|---|
| Data format | existing parquet | MDS conversion + second copy |
| Sampling | row-level, exact mixing, custom samplers | shard-order + windowed shuffle |
| Read amplification | ~128× (hidden; ~$1–2k/run) | ~1× |
| Resume | sampler cursor via existing flow | Mosaic cursor (excellent) |
| Maturity | benchmarked; soak pending | production-proven |

If post-soak S3 request costs or tails prove problematic *and* sampling needs relax to
shard-order, a Mosaic-style sequential mode behind the same interface is the natural
fallback.

</details>

# Validation Plan

1. Land the two scoped hardening items: parallel footer scan; `ParquetFile` handle LRU.
2. **Virtual-1T test** (no 1T of data required): S3 server-side-copy one rg128 shard to ~10k
   keys ⇒ the loader indexes ~1B rows with fully realistic access patterns. Measure startup
   (footer scan, `randperm`), then soak 200+ steps at workers=2/prefetch=2 tracking exposed
   data time mean *and* max against the 0.8 ms bar, including a credential-rotation window.
3. Pilot on a slice of the real store (validates schema, compression, and row-group settings
   end-to-end).

# Open Questions

1. Column projection in `__getitems__` (skip pass-through columns such as VLM tensors for
   text batches) — free bandwidth win for wide schemas.
2. Expose `dataloader_prefetch_factor` and pyarrow io thread count in `SFTConfig`?
3. Handle-cache sizing: fixed LRU vs memory-budgeted (footer size varies with row-group
   count).
4. Is an arrow-IPC + row-offset-index store variant (exact-row byte ranges, ~1×
   amplification, uncompressed) worth specifying for amplification-sensitive deployments?

[^1]: Sampling policies must be *plan-ahead-able* (deterministic given state — seeded random
order, precomputed curricula). See Non-Goal 1 for the reactive case.

[^2]: Verified empirically: order is deterministic and resumable for a fixed
(seed, num_workers); changing worker counts mid-run changes the permutation, consistent with
the existing trainer's behavior.

[^3]: Footer parsing at this scale also showed a ~16 GB transient RSS peak (10k parsed
footers held during the parallel scan) — acceptable on a 121 GB controller, eliminated
entirely by the manifest fast path or a chunked scan; tracked in the validation plan.
