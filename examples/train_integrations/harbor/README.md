## Harbor Integration

RL training with [Harbor](https://github.com/laude-institute/harbor) as the environment and reward source. See the [full documentation](https://docs.skyrl.ai/docs/harbor) for details.

### Structure

```
examples/train_integrations/harbor/
  harbor_generator.py              # HarborGenerator: bridges SkyRL <-> Harbor
  dataset.py                       # HarborTaskDataset: loads task directory paths
  prepare_harbor_dataset.py        # Downloads + extracts datasets from HuggingFace
  harbor_trial_config/
    default.yaml                   # Harbor TrialConfig template
  entrypoints/
    main_harbor.py                 # Full training entrypoint
    main_harbor_generate.py        # Generation-only debug entrypoint
  run_codecontest.sh               # Code contest training (Qwen3-8B)
  run_harbor_gen.sh                # Debug generation-only
```

### Quick Start

```bash
cd SkyRL

# 1. Set credentials
export WANDB_API_KEY=your_wandb_api_key
# Pick your sandbox provider:
export DAYTONA_API_KEY=your_daytona_api_key
# export MODAL_TOKEN_ID=your_modal_token_id
# export MODAL_TOKEN_SECRET=your_modal_token_secret

# 2. Prepare dataset
uv run examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset open-thoughts/CodeContests
uv run examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset open-thoughts/OpenThoughts-TB-dev

# 3. Launch training
bash examples/train_integrations/harbor/run_codecontest.sh
```

### Throughput tuning for large MoE agents (GLM-5.3-Flash)

Synchronous agent RL steps are usually dominated by waiting on rollouts, not by training: the step
waits for the slowest trajectory while most engines sit idle, and decode throughput collapses at the
low concurrency that sandbox waits and the long tail leave. Measured on one 8xB200 node with
GLM-5.3-Flash (vLLM 0.32 serving the real checkpoint; trainer on 64-expert layer slices extrapolated to
45 layers, 32k context, full fine-tune, BF16, selective `[core_attn, moe]` recompute):

| Lever | Measured effect |
|---|---|
| Trainer settings below vs SkyRL defaults | 4.4x trainer tokens/s (~1.0k -> ~4.5k tok/s/GPU); `empty_cuda_cache=false` alone is ~3x |
| vLLM DP8 + expert parallel + MTP drafting (k=3) vs TP8 | 2.2x / 1.6x / 1.25x output tok/s at 16 / 64 / 256 concurrent requests |
| Decode vs concurrency (TP8) | 91 tok/s/GPU at 16 concurrent requests, 980 at 256: keep many trajectories in flight |

**Trainer (Megatron).** Exact up to summation order (the tuned BF16 loss tracks the default to
<0.03%). FP8 is left off: ~6% more, but it widens the trainer/vLLM logprob mismatch.

```bash
# Kernel opt-ins; SkyRL forwards these to the Megatron workers (skyrl/train/utils/utils.py).
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 SKYRL_FUSED_LOGPROB_TRITON_ROWS=1 SKYRL_MOE_DEDUP_A2A=1 \
  SKYRL_MHC_FUSED_POST_BDA=1 SKYRL_KDA_CONV_BACKEND=mix SKYRL_KDA_INTRACARD_CP=1 SKYRL_KDA_GATHER_ONCE=1 \
  SKYRL_KDA_FANOUT_GRAD=1 SKYRL_DSA_HEAD_FULL=1 SKYRL_DSA_HEAD_FULL_FAST_PERMUTE=1 SKYRL_DSA_HEAD_FULL_ZIGZAG=1 \
  SKYRL_DSA_KPOOL_FUSED_INDEXER=1 SKYRL_DSA_INDEXER_TP_SHARD=1 SKYRL_DSA_KPOOL_SHARD_STRIPE=256 \
  SKYRL_DSA_CUDNN_BWD_ALL_ROWS=1 SKYRL_DSA_TOPK_PRESORTED=1 SKYRL_DSA_TOPK_SCATTER_COMPACT=1 \
  SKYRL_MLA_NOPE_NO_COPY=1 SKYRL_MLA_STRIDED_EINSUM=1
```

```bash
  trainer.policy.megatron_config.empty_cuda_cache=false \
  trainer.policy.megatron_config.transformer_config_kwargs.moe_permute_fusion=true \
  trainer.policy.megatron_config.transformer_config_kwargs.bias_activation_fusion=true \
  trainer.policy.megatron_config.transformer_config_kwargs.dsa_kernel_backend=cudnn \
  trainer.fused_lm_head_logprob=true trainer.logprobs_chunk_size=8192 \
```

**Rollouts.**

```bash
  # Data-parallel attention + expert-parallel MoE per engine. GLM-5.3-Flash's linear-attention
  # state caps the sequences per DP rank (~700-900 on B200), so lower max_num_seqs from 1024.
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.data_parallel_size=8 \
  generator.inference_engine.expert_parallel_size=8 \
  generator.inference_engine.max_num_seqs=256 \
  # Stop waiting for the slowest groups: schedule ~10% more groups than you need and train on the
  # first 90% to finish (the rest are cancelled and loss-masked).
  generator.straggler_cutoff.min_group_fraction=0.9 \
  generator.straggler_cutoff.grace_period_s=60 \
```

MTP drafting: `trainer.mtp.enabled=true` trains the drafter alongside the policy
([spec decode docs](https://docs.skyrl.ai/docs/examples/spec_decode)), but the GLM-5.3-Flash bridge does
not build MTP heads yet (`mtp_num_layers=None`). The checkpoint's own head can draft as-is through
`generator.inference_engine.engine_init_kwargs.speculative_config.method=mtp` and
`generator.inference_engine.engine_init_kwargs.speculative_config.num_speculative_tokens=3`: sampling stays exact (rejection sampling against the policy) and only acceptance decays as the policy
drifts. Not yet validated together with SkyRL's weight sync.

Keep enough trajectories in flight to fill the engines: a step with few rollouts per GPU (e.g. 32
rollouts on 32 GPUs) decodes at <100 tok/s/GPU. Use more rollouts per step, fewer GPUs, or fully
async training (`run_codecontest_fully_async.sh`), which removes the step barrier altogether.
