# Cold start (vLLM + Megatron)

Startup-time work for NovaSky-AI/SkyRL#1954. This page is the handoff for validating the
branch on a real cluster: what changed, how to turn each piece on, what to measure, and
what is known to be wrong or untested. Read `inference.md` (Startup / JIT caches) and
`backends/megatron.md` (HF import cache) for the mechanics.

## What is in the branch

| piece | knob | default | status |
|---|---|---|---|
| Startup phase timers | none, always on | on | validated (L4) |
| Engine startup overlapped with worker spawn / model init | `trainer.placement.overlap_worker_spawn` | on | validated colocated 8xH100 (Qwen3-30B-A3B) and non-colocated 4+4 (Qwen2.5-14B) |
| Engines launched before prompt tokenization (`startup/datasets` overlaps engine boot) | none, always on | on | validated 8xH100 |
| Concurrent policy/ref/critic actor spawn; all ranks of a group created at once | part of `RayPPOTrainer.create_actor_groups` / `PPORayActorGroup._initiate_actors` | on | validated 8xH100 (process-group phase 29-36s -> ~1s) |
| Trainer spawn in a thread from before the engine launch (`spawn_actor_groups`) | `trainer.placement.overlap_worker_spawn` | on | validated 8xH100 |
| Colocated model build + offload while the engines boot, engine workers hold `init_device` on a startup barrier (`inference_servers/startup_barrier.py`) | `trainer.placement.overlap_model_init` | on | validated 8xH100: 256.8s -> 214.2s cold, KV cache identical |
| sonicloader mirror (compile cache, optional weights) | `generator.inference_engine.sonic_mirror` (+ `sonic_publish_on_startup`, `sonic_stream_weights`) | off | validated cache path (L4, `serve` entrypoint); **weight streaming untested** (needs `libsonicgpu.so`) |
| vLLM dummy initial weights | `generator.inference_engine.dummy_initial_weights` | off | validated (L4, logprob gap unchanged) |
| Skip step-0 sync | `trainer.skip_initial_weight_sync` | off | validated non-colocated 4+4 (Qwen2.5-14B); colocated allowed via a level-1 first sleep, validated 8xH100 |
| Megatron HF import cache | `trainer.{policy,ref}.megatron_config.hf_import_cache_dir` | off | validated TP=1 (L4); **TP/PP>1 reload untested** |
| JIT cache env forwarding | `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT` on the driver | n/a | forwarded to all Ray workers |

Upstream already has the vLLM compile cache on by default (#2167) and the per-device AOT
cache path (#2183); nothing here duplicates those.

## Reading the numbers

Every run logs one line before the first step:

```
Startup timings: startup/inference_engines=..., startup/spawn_workers=..., startup/inference_engines_ready_wait=...,
startup/init_models=..., startup/init_weight_sync_state=..., startup/sync_weights=..., startup/total=...
```

and the same keys go to the tracker as `timing/startup/*` at step 0 (`commit=False`, merged into the
first row). Meaning:

- `inference_engines` — launching the engines (returns once the server actors are constructed and
  their URLs are resolved; not health). ~25-30s on H100 (uv re-exec + vLLM imports). If this is
  minutes, the overlap did not happen; see the race note under "Known gaps and traps".
- `datasets` — prompt tokenization (train + eval). Runs while the engines start.
- `spawn_workers` — policy/ref/critic Ray actors up, backend imported, process groups joined. The spawn
  starts in a thread before the engine launch (`spawn_actor_groups`); this phase is the residual wait
  once the trainer is constructed, so it is ~0 when `inference_engines + datasets` covers it.
- `init_models` — HF -> Megatron import (or cache load), optimizer build, offload. Colocated with
  `overlap_model_init` (default): runs while the engines boot; their workers wait on the startup barrier
  before `init_device`, so the KV cache is still sized against a free GPU. With it off: after the engines
  are slept.
- `inference_engines_ready_wait` — engine startup not hidden behind the two above. Zero means fully overlapped.
- `sync_weights` — the step-0 weight sync.
- `total` — wall clock from entrypoint construction (includes tokenizer + dataset load).

Correctness check for anything touching weights: `policy/minibatch_rollout_logprobs_abs_diff_max` /
`_mean` at step 1. Healthy is ~0.3-0.5 max / ~0.015 mean on bf16; a wrong-weights engine shows 10+ / 2+.

Per-worker detail on rank 0 of each Megatron group: `megatron model build (HF import) took Xs` or
`megatron model build took Xs, HF import cache load took Ys`. vLLM's own `Model loading took` and
`init engine ... (compilation: ...)` lines are in the infra log.

## L4 reference numbers (Qwen2.5-0.5B, colocated, TP=1, warm compile cache)

| run | engines | spawn / build | init_models | total |
|---|---|---|---|---|
| upstream order (baseline) | 60.7 | 54.1 (build_models) | - | 151.2 |
| overlap + concurrent spawn | 19.1 (+19.5 wait) | 25.6 | 8.3 | 118.4 |
| dummy vLLM weights | 65.3 | 55.2 | - | 157.6 (noise at 0.5B) |
| HF import cache warm | 62.2 | 54.0 | - | 157.6 (import is 0.8s at 0.5B) |

Cold vs warm compile on the same box: `compilation 15.2s -> 0.16s`, engine init `24.9s -> 6.6s` (sonic
mirror pull with all local caches wiped). Fixed per-worker cost: ~5s `uv run` re-exec + ~25s importing
the Megatron worker module (~14s for the vLLM server actor).

## 8xH100 results (2026-09-21, one node, Megatron, vLLM 0.28)

One step of gsm8k (batch 8, 2 samples, 128 tokens), `trainer.logger=console`. "cold" = `~/.cache/vllm
~/.triton ~/.cache/flashinfer /tmp/torchinductor_*` and sonic's `/mnt/local_storage/sonic_vllm_cache` wiped
before the run; the HF checkpoint stays in page cache (1.9TB RAM), so vLLM's own load is 5-7s and the
HF->Megatron import 12s (30B) / 8s (14B). Logprob gap = step-1 `policy/minibatch_rollout_logprobs_abs_diff_max /
_mean`; healthy here is 1.1-2.9 / 0.020-0.029 (30B MoE) and 0.25-0.37 / 0.010-0.013 (14B dense). `max` is
noisy, `mean` is the signal; every run below is in band. Three trees: **as pushed** (this PR before the fixes
below), **+fixes** (race fix, engines before tokenization, all ranks spawned at once), **+fixes+spawn** (trainer
spawn threaded from before the engine launch, colocated skip-sync via a level-1 first sleep, sonic patches).

### Colocated Qwen3-30B-A3B (TP=8/EP=8/ETP=1, one engine tp=8, `gpu_memory_utilization=0.7`)

Megatron optimizer state CPU-offloaded (`optimizer_config_kwargs`: `optimizer_cpu_offload`,
`optimizer_offload_fraction=1.0`, `use_precision_aware_optimizer`, `decoupled_weight_decay`): with DP=1 the
distributed optimizer has nothing to shard across and the 30B OOMs at `optim_step` otherwise. That adds ~40s
to `init_models` uniformly; on >=2 nodes it goes away.

| run | tree | engines | datasets | spawn | ready_wait | init_models | sync_w | **total** | vLLM compile | vLLM load | gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline cold | as pushed | 171.3 (race lost) | - | 54.1 | 0.2 | 85.4 | 14.9 | **356.8** | 50.2 | 6.3 | 1.48 / 0.020 |
| baseline warm | as pushed | 29.1 | - | 61.4 | 24.2 | 83.6 | 15.2 | **244.7** | 0.8 | 6.3 | 2.48 / 0.029 |
| overlap_worker_spawn=false cold | as pushed | 187.5 | - | 58.4 | 0.0 | 82.2 | 15.8 | 374.4 | 51.9 | 6.3 | 2.27 / 0.025 |
| hf_import_cache miss warm | as pushed | 27.4 | - | 58.6 | 23.1 | 194.0 (save 108.5) | 14.7 | 349.1 | 0.8 | 6.2 | 2.95 / 0.024 |
| dummy + hf_import_cache hit warm | as pushed | 26.5 | - | 57.1 | 19.8 | 141.1 (load 73.5) | 17.1 | 293.1 | 0.3 | 0.9 | 2.14 / 0.025 |
| dummy + hf_import_cache hit cold | as pushed | 174.4 (race lost) | - | 56.9 | 0.2 | 97.9 (load 26.7) | 16.9 | 377.1 | 50.5 | 0.6 | 1.61 / 0.022 |
| baseline cold | +fixes | 24.2 | 19.6 | 41.8 | 81.2 | 85.0 | 13.1 | **282.5** | 50.4 | 6.3 | 1.11 / 0.021 |
| baseline warm | +fixes | 28.0 | 19.5 | 35.0 | 23.2 | 83.3 | 13.3 | **213.2** | 0.8 | 6.3 | 2.33 / 0.021 |
| dummy_initial_weights cold | +fixes | 30.5 | 19.3 | 33.6 | 97.3 | 84.5 | 13.5 | 290.1 | 50.4 | 2.1 | 1.43 / 0.021 |
| dummy_initial_weights warm | +fixes | 29.8 | 19.3 | 37.3 | 17.5 | 83.9 | 12.7 | 211.4 | 0.8 | 0.8 | 1.39 / 0.022 |
| distributed_executor_backend=mp warm | +fixes | 21.9 | 19.6 | 28.6 | 76.0 | 85.7 | 15.4 | 258.5 | 51.5 (cache miss) | 6.0 | 1.89 / 0.020 |
| sonic publish boot, cold | +fixes+spawn | 31.9 | 20.7 | 0.0 | 134.5 | 83.9 | 14.0 | 299.9 | 49.6 | 7.3 | 2.63 / 0.021 |
| **sonic restore + skip-sync, cold** | +fixes+spawn | 38.1 | 20.2 | 0.0 | 87.2 | 85.9 | 1.8 (2 wakes) | **256.8** | **0.33** | 20.2 (incl. S3 pull) | 1.66 / 0.022 |
| sonic restore + skip-sync, warm | +fixes+spawn | 35.0 | 20.2 | 0.0 | 59.7 | 84.4 | 3.0 (2 wakes) | 223.4 | 0.34 | 7.2 | 2.04 / 0.020 |
| **+ overlap_model_init (build during engine boot), cold** | +fixes+spawn+overlap | 34.4 | 20.4 | 0.0 | 50.9 | 84.5 (overlapped) | 1.2 | **214.2** | 0.32 | 7.0 | 1.59 / 0.021 |
| + overlap_model_init, warm | +fixes+spawn+overlap | 31.4 | 20.8 | 0.0 | 42.6 | 85.1 (overlapped) | 1.2 | **206.3** | 0.33 | 7.0 | 1.63 / 0.021 |

### Non-colocated Qwen2.5-14B-Instruct (trainer 4 GPUs TP=4, one engine tp=4, `gpu_memory_utilization=0.8`)

| run | tree | engines | datasets | spawn | ready_wait | init_models | sync_w | **total** | vLLM compile | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline cold | as pushed | 19.1 | - | 46.0 | 0.2 | 94.0 | 1.0 | **191.2** | 32.6 | 0.34 / 0.013 |
| baseline warm | as pushed | 79.9 (race lost) | - | 47.4 | 0.2 | 92.7 | 1.0 | 250.2 | 0.7 | 0.26 / 0.010 |
| baseline cold | +fixes | 23.5 | 19.5 | 25.1 | 0.2 | 93.4 | 1.0 | **178.3** | 33.6 | 0.37 / 0.012 |
| baseline warm | +fixes | 21.2 | 19.4 | 29.1 | 0.2 | 92.9 | 1.0 | 173.3 | 0.7 | 0.25 / 0.012 |
| skip_initial_weight_sync cold | +fixes | 21.8 | 19.4 | 37.0 | 0.2 | 88.0 | 0.0 | 175.8 | 32.6 | 0.25 / 0.012 |
| skip_initial_weight_sync warm | +fixes | 21.9 | 19.5 | 31.5 | 0.2 | 88.5 | 0.0 | 171.2 | 0.7 | 0.34 / 0.013 |
| sonic publish boot, no optimizer offload, cold | +fixes+spawn | 23.1 | 20.1 | 0.0 | 68.4 | 24.2 | 1.0 | 148.5 | 32.0 | 0.27 / 0.012 |
| **sonic restore + skip-sync, no offload, cold** | +fixes+spawn | 22.2 | 20.2 | 0.0 | 36.4 | 24.7 | 0.0 | **115.6** | **0.27** | 0.27 / 0.010 |
| sonic restore + skip-sync, no offload, warm | +fixes+spawn | 23.8 | 20.9 | 0.0 | 20.1 | 24.0 | 0.0 | **102.1** | 0.29 | 0.28 / 0.011 |

### What the numbers say

- **Cold start: colocated 356.8s -> 214.2s (-40%), non-colocated 191.2s -> 115.6s (-40%)**, with the
  logprob gap unchanged on every run.
- `overlap_model_init` moves the whole `init_models` (85s here) inside the engine boot: the entrypoint
  creates a `StartupBarrier` actor, the vLLM workers hold `init_device` on it (8 of 8 were waiting when it
  was released), and the trainer builds + offloads meanwhile. vLLM's free-memory check then sees 70.5 GiB
  (vs 74.0 in the load-after-sleep order: the trainer's post-offload allocator/NCCL residual), and the KV
  cache came out **identical** (1,852,256 tokens). `inference_engines_ready_wait` (51s) is now the engines
  finishing their post-barrier boot; with a ~45s build (no optimizer offload, >=2 nodes) the release lands
  before the engines need it and the total tracks engine boot alone (~165s here).
- The startup overlap as pushed was **racy**: 3 of 8 overlap-on runs (both colocated cold runs and the
  non-colocated warm run) had `inference_engines` swallow the whole engine startup (171s, 174s, 80s vs
  24-30s), silently degrading to the sequential order. Fixed by resolving the URLs before `start`.
- After the fixes the colocated critical path is engine boot -> sleep -> `init_models` -> sync; driver
  work (`datasets`, `spawn_workers`) is fully hidden. Sonic removes the 50s compile from engine boot
  (`compilation: 50s -> 0.33s` on a wiped node), the level-1 first sleep removes the 13s step-0 sync. What
  is left after `overlap_model_init`: the engine boot itself (~145s to healthy on a cold node: engine-actor
  construction ~25-35s, vLLM's Ray executor spawning its 8 workers through a second uv re-exec ~36s, load,
  profiling, CUDA graphs) plus ~25s Ray/entrypoint overhead outside the phases; the trainer build only
  shows when it outlasts the engines' pre-`init_device` boot (it does here by ~50s because of the
  single-node optimizer offload).
- Non-colocated was **trainer-bound** as pushed: engine boot, cold or warm, is hidden behind
  `datasets + spawn + init_models`, so sonic/dummy/mp cannot move it. Threading the spawn (`spawn_workers`
  0.0) and dropping the optimizer offload the 14B does not need on 4 dedicated GPUs (`init_models` 93 -> 24)
  made the engine boot the critical path again, and then the sonic restore paid (`ready_wait` 68 -> 36).
- `dummy_initial_weights` is a wash here (vLLM load 6s -> 1-2s, hidden behind the rest of engine boot);
  it pays where the engine's checkpoint read is slow (S3/NFS, no page cache), not on warm local NVMe.
- `hf_import_cache_dir` is a **net loss** here: the HF import is 12s, the cache save 108s (57GB to NFS) and
  the cache load 27-74s. It only pays where the HF import itself is slow. Say so before turning it on.
- `distributed_executor_backend=mp` cut the executor spawn (36s -> 26s) but missed the compile cache
  (51.5s on a warm node); it needs its own cache validation before it can be recommended.
- `gpu_memory_utilization=0.85` colocated 30B **fails in both startup orders** at the post-training
  `wake_up(kv_cache)` (cumem OOM: 9 GiB weights + ~55 GiB KV do not fit next to the trainer's post-offload
  residual); vLLM's startup check passed (74/79 GiB free >= 67.3). The overlap's KV cost is 1.7% (2,422,176 vs
  2,462,896 tokens) from the trainer CUDA contexts present during profiling (1.2% at 0.7). Changing the
  utilization also changes vLLM's compile-cache key (50s compile on a warm node).
- `overlap_worker_spawn=false` (the upstream order) costs 18s over the as-pushed baseline on cold and 92s
  over the fixed tree.
- On an already-**warm** colocated node sonic is a small net loss (223.4s vs 213.2s for the fixed tree without
  it): vLLM's own local compile cache already hits, and the sonic path adds ~20s to engine boot (plugin
  import in the actor, mirror manifest checks, the post-start publish) while the skipped sync saves ~10s.
  Sonic's value is the fresh node; for warm restarts leave it off or set `sonic_publish_on_startup=false`.

## What to run on a large cluster

Pick a model where the weight paths are minutes, not seconds (e.g. Qwen3-30B-A3B or a 70B dense,
TP>=4), Megatron strategy, and run each configuration once cold (fresh nodes or wiped `~/.cache/vllm`,
`~/.triton`, `~/.cache/flashinfer`, `/tmp/torchinductor_*`) and once warm:

1. **Baseline**: defaults (overlap on). Also `trainer.placement.overlap_worker_spawn=false` once to get the
   old sequential order for the same node — that difference is the overlap win.
2. **Non-colocated overlap**: `trainer.placement.colocate_all=false` with engines and trainer on separate
   GPUs. Expect `inference_engines_ready_wait` near zero when `spawn_workers + init_models` exceeds engine
   startup. Watch that the router only starts after the engines are healthy (it waits for backend health;
   `VLLMRouter.url` is known before start).
3. **Dummy initial weights**: `generator.inference_engine.dummy_initial_weights=true`. Expect vLLM
   `Model loading took` to drop to ~1-2s per engine; `sync_weights` unchanged. Check the logprob gap.
   Not for Gemma-3 (its `normalizer` buffer is never synced) and not with MTP drafters the trainer does not own.
4. **Skip initial sync** (non-colocated only): `trainer.skip_initial_weight_sync=true`. Expect
   `sync_weights` ~0. Check the logprob gap carefully: this assumes vLLM's loader and Megatron-Bridge's
   export agree byte-for-byte, which fused or quantized layers may not.
5. **HF import cache**: `trainer.policy.megatron_config.hf_import_cache_dir=<shared fs path>` (and the
   `trainer.ref.megatron_config` twin). First run logs `hf import cache miss` and `HF import cache save
   took`; second run `hit` and `HF import cache load took`. Then change TP or PP and rerun: the cache must
   still hit and load (torch_dist is layout-agnostic) — this reshard case is untested.
6. **Sonic mirror** (needs the `sonic-loader` package in the engine env, AWS creds, `AWS_REGION` exported
   on the driver): `generator.inference_engine.sonic_mirror=s3://bucket/prefix/`. First boot publishes the
   compile-cache tar (`sonic: published artifacts to ...` with `cache_bytes`), a boot on a fresh node
   should show `compilation: <1s`. `sonic_stream_weights=true` additionally publishes per-rank shards and
   streams them on later boots — needs `libsonicgpu.so` (wheel or `cargo build`) and a multi-ENI host to
   beat local disk; once weights are published under a digest every boot with that config streams them.

## Known gaps and traps

- `skip_initial_weight_sync` with colocated placement takes the engines' first sleep at level 1
  (weights backed up to CPU memory, ~7.6GB/GPU for the 30B at TP=8; the first `wake_up(["weights"])`
  restores them) instead of level 2, which discards them (measured gap 17 max when the sync was skipped
  over a level-2 sleep). Later sleeps stay at level 2. It saves the step-0 sync (13s at 30B here) at
  the cost of one CPU round trip of the weights.
- Non-colocated runs on this node are trainer-bound: engine boot (cold or warm) is fully hidden behind
  `datasets + spawn_workers + init_models`, so sonic, dummy weights and the executor backend cannot
  move the total there until the trainer side shrinks (`init_models` first; a 14B on 4 dedicated GPUs
  does not need the optimizer CPU offload).
- `overlap_worker_spawn` puts trainer CUDA contexts on the shared GPUs during vLLM's memory profiling.
  KV cache was identical on the L4 at `gpu_memory_utilization=0.4`; re-check at 0.8+ on large models
  (vLLM raises if free memory at startup is below the requested utilization).
- The engine launch phase blocks on engine actor construction (~25-30s on H100: uv re-exec + vLLM
  imports) so the router/client get URLs. Decoupling the client from server URLs would let the trainer
  spawn start earlier still; on colocated runs it would not change the total, which is engine-bound.
- Fixed here: `ServerGroup.start(blocking=False)` used to submit `get_server_info` just before `start`
  and rely on it running first. The engine build runs synchronously inside the async `start` and holds
  the actor's event loop until the engine is healthy, so when the info RPC landed behind it the launch
  phase silently became the whole engine startup (171s and 174s on the two as-pushed cold runs vs 24-30s)
  and the overlap degraded to the sequential order. The infos are now resolved before `start` is submitted.
- Fixed here: `PPORayActorGroup._initiate_actors` created rank 0, blocked on its address (rank 0's
  re-exec + imports, ~25s), then created ranks 1..N-1, whose own ~25s then sat inside "Initializing
  process group" (29-36s observed). All ranks are created at once and rank 0's address is pushed to
  the others with `set_master_addr_port` before the process group is initialized (~1s now).
- The HF import cache keys Hub ids by name only (no revision); local paths by weight-file names/sizes/mtimes.
- sonic's `.sonic_cache_pulled` marker under `VLLM_CACHE_ROOT` is **per node and digest-agnostic** (an empty
  file; `_maybe_pull_cache` returns as soon as it exists): once any model's cache was pulled on a node, no
  other model's cache is pulled there. On this node a 14B boot's marker made the following 30B boot skip
  its pull and recompile (49.9s) although its cache was in the mirror. Nodes that boot more than one
  engine config need the marker keyed by digest (sonic fix) or removed between models. sonic INFO logs
  are dropped under vLLM's logger config (warnings still show).
- `sonic-loader` must be visible to the **engine worker processes**, not just the driver. `.venv` is
  gitignored, so Ray's `working_dir` upload excludes it and every worker's `uv run` re-exec materializes
  its own project venv inside the working-dir snapshot (`/tmp/ray/session_*/runtime_resources/
  working_dir_files/_ray_pkg_*/.venv`) from `uv.lock`. A wheel added to the driver venv with `uv pip
  install` is not in the lock, so with TP>1 vLLM's `RayWorkerProc`s fail with ``Load format `sonic` is
  not supported`` (the loader registers on plugin import; the L4 validation at TP=1 ran the in-process
  executor and never saw this). Export `UV_PROJECT_ENVIRONMENT=<repo>/.venv` on the driver
  (`prepare_runtime_environment` forwards it) so the workers resolve to the driver venv, or add the
  wheel to the project dependencies. The same snapshot-venv materialization is a one-off cost on the
  first worker start per node/snapshot and is part of what a "fresh node" cold start pays.
- vLLM >= 0.28 validates `model_loader_extra_config` keys in `DefaultModelLoader.__init__`, which
  `SonicLoader` inherits, so sonic's `mirror` / `capture` keys fail with ``Unexpected extra config keys
  for load format sonic`` in every worker. `patches/vllm/patch_sonic_loader_extra_config.py` (imported by
  `NewInferenceWorkerWrap`, so it is applied in each worker process) validates with the sonic keys held
  back and hands the loader its full config; no-op for other formats.
- sonic's `push_artifacts` stamps the **weights** manifest even when it uploaded no shards (a
  cache-only publish, `sonic_stream_weights=false`), and its loader streams whenever that manifest
  exists, so the next boot streamed from an empty prefix and died with ``no shard files for rank N``.
  The same patch module makes `sonic_stream_weights=false` authoritative on the consume side: the
  loader restores the compile cache and loads from HF regardless of the mirror's weights manifest
  (`model_loader_extra_config["stream"]`). Streaming (`sonic_stream_weights=true`) is unchanged.
- Megatron GPU runs need `NVTE_FLASH_ATTN=0`; on Anyscale workspaces the TE cudnn lookup needs the pip
  NVIDIA libs registered with ldconfig incl. unversioned `lib*.so` symlinks (see `troubleshooting.mdx`).

## Tests

```bash
uv run --isolated --extra dev --extra fsdp pytest \
  tests/backends/skyrl_train/inference_servers/test_setup_deferred_ready.py \
  tests/backends/skyrl_train/inference_servers/test_sonic_mirror.py \
  tests/backends/skyrl_train/inference_servers/test_initial_weights_knobs.py \
  tests/backends/skyrl_train/workers/megatron/test_hf_import_cache.py \
  tests/train/test_rl_callbacks.py tests/train/test_trainer.py tests/train/test_fully_async_trainer.py
```

GPU coverage is by the runs above; there is no GPU CI test for the new knobs yet.
