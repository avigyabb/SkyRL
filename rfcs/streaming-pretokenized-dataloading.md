# RFC: Streaming dataloading for pretokenized SFT datasets at trillion-token scale

- **Status**: Draft
- **Author**: @avigyabb
- **Related work**: #1927 (pretokenized ingestion, merged), #1933 (cloud paths, draft)

## Summary

Add a streaming mode to `SFTTrainer` so pretokenized datasets in cloud storage (S3 required,
GCS nice-to-have) can be trained on at trillion-token scale without materializing the dataset
in memory. **Proposal: use HuggingFace `datasets` streaming (`IterableDataset`) as the loader,
keep torchdata's `StatefulDataLoader` and everything downstream of it unchanged, and require
an offline-shuffled, sharded, manifest-described data layout from the pretokenization
pipeline.** A bespoke shard streamer is specified as a fallback behind the same interface, to
be built only if the HF path fails measured throughput/resume targets.

## Requirements

1. Scale to trillion-token datasets with streaming (bounded-memory) dataloading.
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
- Multi-epoch semantics. At 1T tokens, runs are sub-epoch; epoch-derived step counts
  (`num_epochs`) do not apply in streaming mode.

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
&gt;200 MB/s from a single S3 prefetch stream. Consequences:

- The existing single-controller load→collate→dispatch model needs no re-architecture.
- Distributed reading frameworks solve a problem we do not have.
- The hard problems are **bounded memory**, **shuffle quality without a global shuffle**, and
  **O(1) resume** — not bandwidth.

## Data layout contract (engine-independent)

Required of the offline pretokenization pipeline regardless of loader choice:

1. **Fixed-size shards** (~128–512 MB parquet, ~100k–500k rows each) — never one giant file.
2. **Offline shuffle at write time** (global, or at minimum shard-internal). This is the load-
   bearing decision: with rows pre-shuffled, the online loader only shuffles *shard order*
   (seeded, per pass) and reads sequentially within a shard. That eliminates online shuffle
   buffers — the single largest source of complexity and checkpoint-state pain in streaming
   loaders.
3. **A manifest** alongside the shards (`shard → row_count`, schema, totals). Avoids
   list-then-open-footers over tens of thousands of objects at startup, and gives O(1)
   "global sample N → (shard i, offset j)" seeks.

This is the substance of what Mosaic's MDS format provides, expressed as parquet + one JSON
file.

## Options considered

### Option A — Ray Data

*Streaming execution over S3, distributed decode, shuffle windows; we are already a Ray shop.*

Rejected:

- **No sample-exact checkpoint/resume of a streaming iterator.** Requirement 4 would have to
  be rebuilt on top — which is the hard part of rolling our own, while also inheriting Ray
  Data's execution model.
- Designed to parallelize per-row *compute* (tokenization, decode). We moved that offline; the
  remaining work is I/O at 2–4 MB/s, which one process trivially sustains.
- Competes for cluster resources with training actors; incompatible with the
  `StatefulDataLoader`-based checkpoint format; iterator semantics don't compose with the
  existing trainer loop.

### Option B — HuggingFace `datasets` streaming (`IterableDataset`) ✅ proposed

Correcting the scoping note ("don't think it supports cloud storage"): it does.
`load_dataset("parquet", data_files=..., streaming=True, storage_options=...)` is
fsspec-based, so `s3://` works via `s3fs` (already a repo dependency) and `gs://` via `gcsfs`.

Why it fits:

- **Composes with what we have.** An `IterableDataset` is a valid dataset for
  `StatefulDataLoader`; collators, FFD packing, dispatch, and the `data.pt` checkpoint flow
  are unchanged. This is a dataset swap, not a dataloader replacement.
- **Resume is already wired.** `IterableDataset` implements `state_dict()`/`load_state_dict()`
  (datasets ≥ 2.18), and `StatefulDataLoader` checkpoints/restores per-worker dataset state
  through its existing protocol. Resume lands at (shard, offset) and skips forward within one
  shard — cheap when shards are modest.
- **Prefetch overlap for free.** Shards split across dataloader workers
  (`num_workers`, `prefetch_factor`); download + decode + collate run in worker processes
  while the GPU trains. Requirement 3 without new machinery.
- **Shuffling**: `.shuffle(seed, buffer_size)` shuffles shard order + a small buffer; with the
  offline-shuffled layout the buffer is belt-and-suspenders rather than load-bearing.
- **Multi-store mixing**: `interleave_datasets(streams, probabilities=weights)` replaces
  `DataMixingSampler` (which is map-style and cannot survive streaming).

Known limitations, accepted:

- Python-level decode throughput is unimpressive — irrelevant at 2–4 MB/s required ingest.
- Resume-by-skip within a shard costs one partial shard read.
- fsspec creds are used directly rather than the repo's custom S3 refresh layer.
- Exact determinism across restarts requires the same `num_workers` (shard→worker assignment
  is implicit state) — consistent with the stated non-requirement.

### Option C — Mosaic `StreamingDataset`

Correcting the scoping note: it *does* support S3/GCS natively and is actively maintained
(Databricks). It is the best-engineered tool in this space: manifest-driven deterministic
sample order, instant mid-epoch resume from a cursor, bounded local shard cache with
eviction, tuned shuffle algorithms.

Rejected as a dependency (per team preference), but **its design is the reference** for
Option D: index/manifest, shard-order shuffle, resume as a cursor rather than a replay,
bounded shard cache. Also note its model is per-rank dataloaders; under SkyRL's controller
dispatch we would run it degenerate (world=1), forgoing its main distributed machinery — a
sign it is more tool than we need.

### Option D — Roll our own (fallback, specified but not built)

With the data layout contract in place, a bespoke streamer is small (~400–600 lines):

```
manifest → seeded shard permutation per pass
        → background prefetcher: next K shards into a bounded queue (K ≈ 2–4)
        → sequential row iterator → existing collators
resume state: (pass, shard_perm_seed, shard_index, row_offset)   # four integers, O(1) seek
```

It exists in this RFC as a *designed fallback behind the same interface* (iterable +
`state_dict` protocol feeding `StatefulDataLoader`), to be built only if Option B misses
measured targets — e.g. S3 tail-latency stalls HF's worker prefetch cannot hide, or
resume-by-skip proving too slow. Framing the prefetch budget in **shards** (hundreds of MB in
flight), not batches, is deliberate: a slow GET must be hidden behind a shard's worth of
runway, not one batch's.

The reason DIY is tractable *here* is the offline shuffle: without it, we would be rebuilding
shuffle buffers and their checkpoint semantics, which is where homegrown loaders historically
go wrong.

## Decision

**Option B now, Option D as the escape hatch, Option C as design reference, Option A ruled
out.** The interface seam (anything iterable implementing the `state_dict` protocol) keeps the
Phase-2 swap from being a redesign, and Phase 1 produces the throughput numbers that decide
whether Phase 2 ever gets built.

## Plan

**Phase 0 — layout contract**: document and enforce the shard/manifest/offline-shuffle
requirements in the pretokenization pipeline docs; add a manifest writer/validator helper.

**Phase 1 — HF streaming mode** behind `pretokenized_dataset_streaming=true`:

- `load_dataset("parquet", data_files=<manifest globs>, streaming=True, storage_options=...)`
  → `.map(normalize_row)` (reusing #1927's row validation) → `StatefulDataLoader` with
  `num_workers=N`, `prefetch_factor=k`.
- Benchmark on a real S3 store: sustained tokens/s vs training consumption, stall counts,
  resume latency at various run depths.

**Phase 2 (conditional) — bespoke shard streamer** per Option D, same interface, informed by
Phase-1 measurements.

## Trainer integration changes (same for Phase 1 and 2)

| Today | Streaming mode |
|---|---|
| `load_dataset()` returns a materialized list | returns an iterable; nothing materialized |
| `steps_per_epoch = len(dataloader)`; `num_epochs` supported | no `len()`; **`num_steps` required** |
| `_log_dataset_stats` scans all rows | reads manifest totals / samples a prefix |
| `sampler="random"` shuffle over indices | `.shuffle(seed, buffer_size)` on the stream |
| `DataMixingSampler` for multi-store weighting | `interleave_datasets(probabilities=...)` |
| `sampler="custom"` (curriculum etc.) | unsupported in streaming mode (documented) |
| tail-batch padding (`drop_last=False`) | moot: `num_steps`-bounded, never hits a tail |
| eval sets materialized | unchanged (eval is small) |
| `data.pt` = dataloader `state_dict` | unchanged mechanism; state is now shard cursors |

## Open questions

1. Manifest format: adopt/parallel the pretokenized store schema from #1927, or a sidecar
   `index.json` per store? (Leaning sidecar `index.json`, written by the pretokenization job.)
2. Should streaming mode *require* the manifest, or degrade to listing + estimating? (Leaning
   require: at 30k+ shards, listing-based startup and stats are exactly what we want to ban.)
3. `interleave_datasets` checkpointability for multi-store mixing needs verification at
   current `datasets` pin (single-store resume is confirmed supported).
4. Loss-mask storage: `uint8` vs bit-packed in shards — 8× size difference on one column;
   decide in the layout contract.
5. Interaction with `use_sequence_packing` (FFD packs per global batch): unchanged
   mechanically, but packing efficiency under streaming order should be spot-checked.
