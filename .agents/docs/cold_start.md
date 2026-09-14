# Cold start (vLLM + Megatron)

Startup-time work for NovaSky-AI/SkyRL#1954. This page is the handoff for validating the
branch on a real cluster: what changed, how to turn each piece on, what to measure, and
what is known to be wrong or untested. Read `inference.md` (Startup / JIT caches) and
`backends/megatron.md` (HF import cache) for the mechanics.

## What is in the branch

| piece | knob | default | status |
|---|---|---|---|
| Startup phase timers | none, always on | on | validated (L4) |
| Engine startup overlapped with worker spawn / model init | `trainer.placement.overlap_worker_spawn` | on | validated colocated on 1 GPU; **non-colocated untested** |
| Concurrent policy/ref/critic actor spawn | part of `RayPPOTrainer.create_actor_groups` | on | validated (L4) |
| sonicloader mirror (compile cache, optional weights) | `generator.inference_engine.sonic_mirror` (+ `sonic_publish_on_startup`, `sonic_stream_weights`) | off | validated cache path (L4, `serve` entrypoint); **weight streaming untested** (needs `libsonicgpu.so`) |
| vLLM dummy initial weights | `generator.inference_engine.dummy_initial_weights` | off | validated (L4, logprob gap unchanged) |
| Skip step-0 sync | `trainer.skip_initial_weight_sync` | off | **non-colocated only**, never measured (no 2-GPU box) |
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

- `inference_engines` — launching the engines (returns once the server actors are constructed; not health).
- `spawn_workers` — policy/ref/critic Ray actors up, backend imported, process groups joined. Runs while engines start.
- `init_models` — HF -> Megatron import (or cache load), optimizer build. Colocated: after the engines are slept.
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

- `skip_initial_weight_sync` with colocated placement is rejected on purpose: colocated engines sleep at
  level 2 after startup and lose their weights; measured gap 17 max when forced.
- `overlap_worker_spawn` puts trainer CUDA contexts on the shared GPUs during vLLM's memory profiling.
  KV cache was identical on the L4 at `gpu_memory_utilization=0.4`; re-check at 0.8+ on large models
  (vLLM raises if free memory at startup is below the requested utilization).
- The engine launch phase still blocks ~19s per engine actor construction (`ServerGroup.get_server_infos_nowait`
  waits for the actor to exist so the router/client get URLs). Decoupling the client from server URLs would
  let the trainer spawn start earlier still.
- The HF import cache keys Hub ids by name only (no revision); local paths by weight-file names/sizes/mtimes.
- sonic's `.sonic_cache_pulled` marker under `VLLM_CACHE_ROOT` skips the pull on later boots on that node;
  sonic INFO logs are dropped under vLLM's logger config (warnings still show).
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
