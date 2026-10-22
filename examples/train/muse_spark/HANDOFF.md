# Muse Spark Open in SkyRL: status and next steps

Handoff for an agent picking this up. Branch: `muse-spark-open` (on `avigyabb/SkyRL`), based on `main` @ `68a4f44f`.
Goal: support `meta-models/Muse-Spark-Open-816B-A42B` (gated HF repo) following
`.agents/docs/new_model_support.md` (Stage 0 roundtrip correctness → Stage 1 GSM8K → Stage 2 DAPO + R3 +
checkpointing → Stage 3 perf/quant). Read that guide, `.agents/docs/backends/megatron.md`, and
`skyrl/backends/skyrl_train/patches/megatron/README.md` (Muse Spark section) first.

## Where things stand (2026-10-16)

| Stage | Status |
|---|---|
| 0: 4-layer slice, Megatron vs vLLM | **Pass.** Logprob diff 0.0376, post-sync common prefix 0.0139 (`muse-spark-open-4layer_h100_tp2_ep4` row in `test_megatron_models.py`). Megatron vs HF on the slice: 0.028. |
| 0: slice LoRA | Merged LoRA rows **pass** (0.038 / 0.040). Adapter (unmerged) rows are skipped: vLLM has no MoE LoRA for Muse ("post-expert norm does not support MoE LoRA"). |
| 0: full model | Megatron alone is coherent at TP8/EP32 on 4 nodes. Megatron vs vLLM forward: 0.0128–0.0137. The *colocated* sync OOMs even on 64 B200s (see "Memory"), so Stage 1 runs non-colocated. |
| 1: GSM8K LoRA, full model | **Rollouts correct after sync**: step 1 reward 0.89, pass@8 0.97 (`muse-gsm8k-lora-11`). The run died before the first optimizer step because the cluster autoscaler removed 6 of the 8 B200 nodes (an infra change, not a bug). **No loss or learning curve yet.** |
| 1: full FT | Not started. Needs more HBM than 8 nodes (fp32 master + Adam ≈ 9.8 TB), so it needs CPU offload or more nodes. |
| 2, 3 | Not started. |

## What is on the branch

Megatron side (`skyrl/backends/skyrl_train/patches/megatron/muse_spark/`). This is NVIDIA-NeMo/Megatron-Bridge#6483
re-expressed on the pinned Bridge / megatron-core:
- `blocks.py`: HybridModel layers with pattern `*E` × 62 (each HF block becomes an attention layer plus a MoE layer).
  - Attention: parameter-free QK RMSNorm, a per-head output norm, and the sigmoid output gate.
  - Residual gates live in the BDA slots.
  - MoE: fp32 sigmoid router with a pre-sigmoid `output_multiplier`; latent MoE with pre/post expert norms.
- `provider.py`: `MuseSparkModelProvider(HybridModelProvider)` with embedding RMSNorm and logit multiplier + softcap mixins.
- `bridge.py`: HF↔Megatron mappings.
  - `FP32_HF_PARAM_SUFFIXES` keeps the router weight and bias fp32 on the wire.
  - A `_merge_lora_adapter_weights` override splits the fused Q/K/V/gate LoRA delta.
- Registered in `workers/megatron/model_bridges.py`.
- `distributed/megatron/fused_lm_head.py` raises on softcapping, which the fused head doesn't support.

HF config: `patches/transformers/muse_spark_config.py` lets the pinned transformers load the config. It is registered in
`skyrl/backends/skyrl_train/__init__.py`.

vLLM side. These patches target the vLLM wheel built from vllm-project/vllm#61879:
- `patches/vllm/model_module_hooks.py`: `on_model_module_load()` defers per-model patches until vLLM imports that model.
  This fixed a duplicate `fused_rms_norm_gated` CustomOp, caused by importing glm5next in every worker.
  `patch_glm5next_lora_packing.py` now uses it.
- `patches/vllm/patch_muse_spark_reload.py` fixes three upstream bugs under in-place (layerwise) reload, which is how
  every SkyRL weight sync loads:
  1. The fused QKVO loader bypassed `param.weight_loader`, so attention updates were silently dropped.
  2. The router `output_multiplier` was folded only once, so a reloaded router ran unscaled. It now re-folds when the
     weight content changes.
  3. `materialize_coefficients` re-registers the residual `alpha` / `beta` buffers as new tensors. The CUDA graphs
     captured at boot (`enforce_eager=False`) keep reading the freed addresses. **This was the "garbage generations
     after sync" bug.** It only shows with CUDA graphs on, and every tensor still compares equal. The fix copies the
     new values in place.

  It is installed from `inference_servers/new_inference_worker_wrap.py`.

Weight sync / config:
- `weight_sync/sources.py`: `MegatronWeightSource` keeps the fp32 tensors in `FP32_HF_PARAM_SUFFIXES` as fp32.
- `config.py` adds `generator.inference_engine.weight_transfer_packed` (default true). `weight_senders.py` passes it to
  the NCCL init. Muse sets it false (see "Memory").

Tests:
- `tests/backends/skyrl_train/patches/megatron/test_muse_spark_bridge.py`: CPU; provider translation + mapping coverage.
  Passes.
- `test_megatron_models.py` / `test_megatron_lora_models.py`: Muse rows, using the slice path from
  `SKYRL_MUSE_SPARK_4LAYER`.
- `tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/muse_spark_{vs_hf,export_check,full_sanity}.py`: dev
  harnesses, not pytest.

Example: `examples/train/muse_spark/run_gsm8k_muse_spark_lora_8node.sh`. Its comments record the layout and memory
history.

Dev scripts: `examples/train/muse_spark/dev/` holds copies of the scripts used outside the repo. They hardcode
`/shared/hackskyrl/muse_spark` paths.
- `build_vllm.sh`: builds the vLLM wheel.
- `make_venv.sh`: builds the torch-2.14 venv, with `overrides.txt` and `reqs_nvidia_extra.txt`.
- `make_slice.py`: makes the N-layer slice.
- `run_roundtrip.sh`, `run_full_roundtrip.sh`: run the roundtrip tests as Ray jobs.
- `launch_multinode.py`: runs torchrun on N whole nodes.
- `vllm_*`: vLLM smoke and reload harnesses. `vllm_chunked_reload_test.py` emulates SkyRL's sync and reports logprob
  drift and which tensors changed or moved storage. Env: `EAGER=0` turns on CUDA graphs, plus `PP`, `GMU`,
  `SCALE_PATTERN`, `CAST_BF16`, `PACKED_SIM`.
- `hf_reference.py`, `test_muse_full_local.py`: reference and full-model checks.

## Environment (as of 2026-10-16; may not survive)

- Cluster: Anyscale, 8xB200 workers plus a head node. The head GPUs are used by an external workload, so run GPU work
  as `ray job submit`. Example:
  ```bash
  ray job submit --submission-id <id> \
    --runtime-env-json '{"py_executable": "/shared/hackskyrl/muse_spark/venv214/.venv/bin/python",
      "env_vars": {"FLASHINFER_WORKSPACE_BASE": "/tmp/fi_ws_214", "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"}}' \
    -- bash -c 'cd /shared/hackskyrl/muse_spark/SkyRL && RAY_OVERRIDE_JOB_RUNTIME_ENV=1 \
      PYTHON_CMD=/shared/hackskyrl/muse_spark/venv214/.venv/bin/python DATA_DIR=/shared/hackskyrl/data/gsm8k \
      LOGGER=wandb bash examples/train/muse_spark/run_gsm8k_muse_spark_lora_8node.sh trainer.max_training_steps=20'
  ```
  Ray jobs run from the rsynced copy `/shared/hackskyrl/muse_spark/SkyRL`. Rsync the repo there first:
  `rsync -a --exclude .git <repo>/ /shared/hackskyrl/muse_spark/SkyRL/`.
- Artifacts on `/shared/hackskyrl/muse_spark/`:
  - Full checkpoint `Muse-Spark-Open-816B-A42B`.
  - Slice `Muse-Spark-Open-4layer` (109 GB, local only; **not on HF yet**).
  - vLLM wheel `wheels/vllm-0.32.0rc1.dev241+g6cc692b5a*.whl`.
  - torch-2.14 ABI wheels in `abi/out` (flash-attn, mamba-ssm, causal-conv1d, TE-torch).
  - `venv214/` (the working venv), `hf-venv/` (transformers main, for HF reference), and the `gcc13` toolchain.
- The vLLM build needed:
  - GCC 13 (conda).
  - DeepGEMM repointed from private `Inferact/DeepGEMM` to `vllm-project/DeepGEMM@6ce1c5e`.
  - elfutils headers via `CPATH`.
  - No Rust frontend (its dependency is private).
  - Git forced to HTTP/1.1, because GitHub and HF HTTP/2 streams reset on this network.
- `uv run --isolated` cannot resolve this stack (torch 2.14 + the vLLM wheel), so use `venv214`. The CI path needs the
  wheel published plus a SkyRL dependency bump.
- vLLM server logs for training runs: `/tmp/skyrl-logs/infra-*.log` on the node running the server.
- HF: the environment's default `HF_TOKEN` belongs to someone else. Do not use it to request access or publish. Use the
  user's own token (account `avigyabasnet`), and never write it into files.

## Memory / layout notes (full model)

- bf16 weights are ~1.63 TB.
  - Megatron on 4 nodes at TP8/EP32 holds ~51 GB of frozen base per GPU.
  - vLLM runs as 1 engine at TP8 × PP4 on the other 4 nodes, also ~51 GB of weights per GPU, with
    `gpu_memory_utilization=0.4`.
- vLLM TP16 across nodes hangs in gloo's `is_in_the_same_node` barrier on this cluster. Use PP across nodes instead.
- Packed NCCL sync: receive buffers are floored at the largest parameter (the fused 256-expert `gate_up_proj`,
  ~16 GiB). They are double-buffered outside `gpu_memory_utilization`, and that OOMed inside `update_weights`.
- The trainer only checks the update RPC after its broadcast, so a vLLM-side error shows up as a ~36 min NCCL-watchdog
  hang. **When a sync "hangs", look at the vLLM worker logs first.**
- The unpacked sync takes ~100 s.
- `use_expandable_segments=true` on the engine avoids allocator fragmentation.

## Next steps

1. **Finish Stage 1 LoRA.** Get 8 B200 nodes back and rerun the command above. Check:
   - `minibatch_rollout_logprobs_abs_diff_mean`: should be ~0.01–0.05. The broken runs showed 6–12.
   - `policy_entropy`.
   - Reward trend over 20+ steps.

   Step time was ~110 s for rollout + forward before the train step. With only 2 nodes, the slice can stand in to
   exercise training metrics (non-colocated, Megatron 1 node / vLLM 1 node).
2. **Regression test for reload under CUDA graphs.** The roundtrip tests and harnesses ran eager, which is why bug 3
   escaped. Add an `enforce_eager=False` variant, or graph-mode coverage, to the Muse roundtrip row.
3. **Upstream issues to file** (draft them; the user posts PR comments themselves):
   - vLLM#61879:
     - (a) QKVO loader bypasses `param.weight_loader` under layerwise reload.
     - (b) Router multiplier folded once.
     - (c) Residual alpha/beta re-registered, which breaks CUDA graphs after reload.
     - (d) Private deps: Inferact/DeepGEMM and the Rust frontend's llm-multimodal.
     - (e) No MoE LoRA.
     - (f) Note: layerwise reload holds references to the tensors passed to `load_weights` until the layer completes, so
       callers must not reuse those buffers.
   - SkyRL:
     - Sender hides vLLM `update_weights` errors behind the NCCL watchdog.
     - The packed receive buffer lives outside the gmu budget.
4. **Before merging upstream:** decide with the user on:
   - Hosting the 4-layer slice on HF (private or gated?).
   - Publishing the vLLM and torch-2.14 wheels (skyrl-wheels release).
   - Bumping deps per `.agents/docs/dependency_bumps.md`.

   Then wire the slice into GPU CI.
5. **Stage 1 full FT:** optimizer CPU offload plus more nodes, or decide it's out of scope.
6. **Stage 2:** a DAPO script, R3 (routing replay: check the router indices match between vLLM and Megatron for the
   fp32 sigmoid router), and checkpoint save/resume.
7. **Stage 3:** sync time (packed path with a bounded buffer, or bucketing large expert tensors), FP8 (Megatron
   `fused_lm_head` lacks softcap).
