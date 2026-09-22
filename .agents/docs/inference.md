# Inference

For training-to-inference weight transfer (`NewInferenceWorkerWrap`, broadcast vs. CUDA IPC, lifecycle), see [`weight_sync.md`](weight_sync.md).

## Architecture

- Key abstractions: `RemoteInferenceClient` , `ServerGroup`, `VLLMServerActor`, `VLLMRouter`
- `RemoteInferenceClient` interacts with HTTP endpoints: 
    - **Data plane**: Interact with router for completions requests.
    - **Control plane**: Fan-out to individual server URLs for weight sync, pause/resume.
- Shared inference interfaces and types live in `inference_servers/base.py` (`InferenceEngineInterface`, `InferenceEngineInput`/`Output`, `ConversationType`); shared helpers (`build_engine_runtime_env`, `get_sampling_params_for_backend`) live in `inference_servers/engine_utils.py`.

## vLLM Router

- `VLLMRouter` in `skyrl/backends/skyrl_train/inference_servers/vllm_router.py` wraps a child process running `vllm-router`. 

## PD Disaggregation

Prefill-Decode disaggregation:
- **Config**: `enable_pd=true` and `num_prefill` (decode engines = `num_engines - num_prefill`). Requires a
  `kv_transfer_config` with a supported P2P transfer connector (see below).
- **Server groups**: Separate prefill and decode `ServerGroup`s, one per engine.
- **P2P transfer connectors** (`get_pd_p2p_connector_name` in `inference_servers/utils.py`): two flavors are
  supported, either bare or wrapped in a vLLM `MultiConnector` (alongside store connectors such as
  `MooncakeStoreConnector` for KV offloading):
    - `NixlConnector` — pull-based default (NIXL side channel).
    - `MooncakeConnector` — push-based (bootstrap-server handshake). The router runs in `kv_connector=mooncake` mode and
      is given each prefill server's mooncake bootstrap server port. The `MultiKVConnectorPromMetrics.observe` monkeypatch
      (`patches/vllm/patch_multi_connector_stats.py`) is applied to avoid an assert-crash when a child connector
      reports stats without Prometheus metrics.
- **Role-specific engine kwargs**: `prefill_init_kwargs` and `decode_init_kwargs` are per-role pass-through vLLM engine
  kwargs (e.g. a different `all2all_backend`, or per-role `kv_role`) applied on top of the base args by
  `get_pd_cli_args`. They are **mutually exclusive** with `engine_init_kwargs` and require `enable_pd=true`; when used,
  both must be set and each must carry its own `kv_transfer_config` (enforced in `validate_inference_engine_cfg`).
- **Troubleshooting — configuring Mooncake GPU memory registration.** `WITH_NVIDIA_PEERMEM` selects how Mooncake pins
  KV-cache GPU memory. Decide with:

  ```
  lsmod | grep nvidia_peermem                          # is peer-memory available?
  cat /sys/class/infiniband/*/ports/1/link_layer       # InfiniBand | Ethernet (= RoCE)
  ```

  1. **`nvidia_peermem` loaded** (IB or RoCE alike) → leave the default. Mooncake registers via `ibv_reg_mr()`.
  2. **`nvidia_peermem` absent or unloadable** — module not shipped, or the container lacks `CAP_SYS_MODULE`
     (e.g. GKE/COS nodes, driver 580+) → set **`WITH_NVIDIA_PEERMEM=0`** to use the dmabuf path
     (`cuMemGetHandleForAddressRange` + `ibv_reg_dmabuf_mr`). Requires a GPU/driver with dma-buf support.
  3. **Got it wrong?** Symptom is a *silent hang, not an error*: hundreds of
     `Failed to register memory ...: Bad address [14]` at engine startup, servers still report healthy, then the first
     request stalls forever with `Memory region not registered by any active device(s)`.


## Key Config Knobs

All under `generator.inference_engine.*`:
- `enforce_eager` (bool, default false): With `enforce_eager=false`, there can be more mismatch between inference logprobs and trainer logprobs. It is recommended to use off policy correction methods like Truncated Importance Sampling (see `docs/content/docs/algorithms/off_policy_correction.mdx` for details) to prevent logprobs drift. With `enforce_eager=false` every engine start runs torch.compile + CUDA graph capture; see "Startup / JIT caches" below.
- `gpu_memory_utilization` (float, default 0.8)
- `max_num_batched_tokens` (int, default 8192)
- `max_num_seqs` (int, default 1024)
- `enable_prefix_caching` (bool, default true)
- `enable_chunked_prefill` (bool, default true)
- `distributed_executor_backend` ("ray" or "mp")
- `engine_init_kwargs` (dict, pass-through to vLLM EngineArgs)

## Startup / JIT caches

Startup phases are timed and logged once as `timing/startup/*` (inference engine launch, `spawn_workers`,
`init_models`, the residual `inference_engines_ready_wait`, `init_weight_sync_state`, the initial `sync_weights`,
and `startup/total` wall clock since the entrypoint started). Look there first before optimizing cold start.

**Where the fixed cost is (L4, Qwen2.5-0.5B, colocated, warm caches).** Every Ray worker pays the uv runtime-env
re-exec (~5s) plus the backend imports (~25s for the Megatron worker module with TransformerEngine, ~14s for the
vLLM server actor) before it does anything; the HF -> Megatron import of a 0.5B model is <1s and vLLM's engine
init is ~7s. So `RayPPOTrainer.build_models` is split into `create_actor_groups` (spawns the policy/ref/critic
groups concurrently in threads, each blocking only on its own actors) and `init_models`; the entrypoint runs the
spawn while the engines start in both placements, and in colocated mode waits for the engines, sleeps them, then
loads the models (`placement.overlap_worker_spawn=false` restores the sequential order).

**Overlapped init (non-colocated).** `_setup_trainer` builds the inference client with `wait_for_ready=False`:
`create_inference_servers` launches the engines, reads their URLs from the actors' constructor state
(`ServerGroup.get_server_infos_nowait`), and constructs but does not start the router (the router waits for
its backends to pass health checks, so it can only start once they are up; its port is reserved in
`__init__`, so the proxy URL is known before then). The training workers are then built while the engines
load and compile, and `InferenceServerSetup.wait_until_ready()` resolves the start refs and starts the router
before `train()`. The URLs are resolved (`ray.get` on `get_server_info`) *before* `start` is submitted: the
engine build runs synchronously inside the async `start` and holds the actor's event loop, so an info RPC
queued behind it waits for the whole engine startup. `_setup_trainer` launches the engines first and builds
the datasets (`BasePPOExp.train_dataset` / `eval_dataset` are lazy properties) behind them, timed as
`startup/datasets`. Colocated runs (`placement.overlap_model_init`, default on) build and offload the training models while the engines boot: the entrypoint creates a job-scoped `StartupBarrier` actor before launching the engines, `patch_startup_barrier` (applied from `NewInferenceWorkerWrap`) holds every vLLM worker's `init_device` until the entrypoint releases it after `init_models`, so profiling and the KV cache see the same free GPU as the load-after-sleep order; with it off the engines must be healthy and slept before the models load. Entrypoints that call `get_inference_client()` directly (`serve`, `main_generate`)
still get fully-ready engines.

vLLM's torch.compile cache (`~/.cache/vllm/torch_compile_cache`) is **enabled** by default (#2167) so warm starts
skip Inductor compilation; `prepare_runtime_environment` forwards a driver-side `VLLM_DISABLE_COMPILE_CACHE` to the
workers for debugging cold compiles. The AOT artifact directory is patched to include the device index
(`patches/vllm/patch_compile_cache_device_path.py`, #2183) so engines on different GPUs never share kernels. See the
cache-key writeup on #2167 for what invalidates each of the two cache trees; the Dynamo/backend tree keys on
absolute source paths and so misses on every `uv run --isolated` re-exec. CUDA graph capture is not cacheable by
vLLM and always runs when `enforce_eager=false`.

**sonicloader mirror (`generator.inference_engine.sonic_mirror`).** Optional integration with
[anyscale/sonicloader](https://github.com/anyscale/sonicloader), which keeps a per-model "mirror" in S3 with two
independently addressed parts: pre-sharded per-rank weights (streamed S3 -> GPU at ~26-30 GB/s on multi-ENI
hosts) and a tar of the compile caches (`VLLM_CACHE_ROOT` incl. Inductor + Triton, and FlashInfer's JIT dir).
Their measurements on 8xH100 put the cache at 57-91% of the boot-time saving: CUDA graph capture re-JITs Triton
kernels per capture shape, so a restored Triton cache turns capture into pure graph recording. With the knob set,
`build_vllm_cli_args` passes `load_format=sonic` + `model_loader_extra_config={mirror, capture}`; the sonic vLLM
plugin (entry point `vllm.general_plugins`, auto-registered when `sonic-loader` is installed) restores the cache
before compiling, streams weights if published, else loads from HF and stages shards. After the engines are
healthy the entrypoint calls `push_artifacts` on every server (`RemoteInferenceClient.push_sonic_artifacts` ->
`/collective_rpc`; the method lives on `NewInferenceWorkerWrap` and delegates to `SonicWorkerExtension`, because
vLLM accepts one `--worker-extension-cls`). Publishing is digest-keyed and idempotent; `sonic_publish_on_startup=false`
consumes only. By default only the compile cache is published (`capture=false`): the cache path is pure
Python (boto3 + tar) and works from a source checkout, while weight streaming (`sonic_stream_weights=true`) needs
the Rust engine `libsonicgpu.so` and, once weights are published under a digest, every boot with that engine
config and `sonic_stream_weights=true` takes the stream path and fails without it (with the knob off the loader
always restores the cache and loads from HF, whatever the mirror holds — see
`patches/vllm/patch_sonic_loader_extra_config.py`). Constraints: S3 only; `sonic-loader` must be in the engine env; incompatible with
`fp8_weight_sync_mode` (both set `load_format`); the cache digest folds in `VllmConfig.compute_hash()` + tp/pp/dp/ep,
gpu arch, torch and cuda, so any engine-config change misses cleanly. The #2183 device-index AOT dirs are per GPU
index, so publish from a node where every engine has started so the tar carries every `rank_*_dev*` directory.

**Initial weights (`dummy_initial_weights`, `trainer.skip_initial_weight_sync`).** vLLM loads the checkpoint and
the trainer then overwrites every weight in the step-0 sync, so one of the two loads is redundant.
`generator.inference_engine.dummy_initial_weights=true` drops the engine's read (`load_format=dummy`; the sync
supplies the weights, NeMo RL's default) and `trainer.skip_initial_weight_sync=true` drops the sync instead, on a
fresh **non-colocated** run whose engines loaded the same checkpoint (`RayPPOTrainer._initial_weight_sync`).
Colocated engines then take their first sleep at level 1 (weights backed up to CPU memory, restored by the first
wake) rather than level 2, which discards them (measured: skipping the sync over a level-2 sleep left a
rollout-vs-trainer logprob gap of 17 max / 2.5 mean instead of 0.5 / ~0); later sleeps stay at level 2. They are mutually exclusive and validated in `validate_inference_engine_cfg`. Dummy weights
are wrong for models with buffers the sync never writes (Gemma-3 `normalizer`), and skipping the sync assumes the
vLLM loader and the trainer export agree byte-for-byte, which quantized or fused layers may not.

## Placement
- Colocated: vLLM and training workers (FSDP/Megatron) are placed on the same set of GPUs. We offload/backload each component as needed. During weight syncing, model weights from vLLM as well as model weights from the training workers remain on GPU
- Non-colocated: vLLM and training workers (FSDP/Megatron) are placed on a different set of GPUs. This reduces the number of available GPUs per component by half, but is in fact the preferred setup for agentic RL with SkyRL. This is because non-colocated setups allow for asynchronous training, where training and inference can progress together. Inference is typically dominated by a long tail of stragglers, and is also typically the time consuming component, and thus using half the number of GPUs doesn't affect inference time for a batch as much.
