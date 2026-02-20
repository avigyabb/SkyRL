---
description: SkyRL training pipeline — data flow, Megatron integration, GSPO loss, checkpointing, and optimizer behavior. Consult when debugging training crashes, loss anomalies, or metric questions.
globs: skyrl-train/**
---

# Training Pipeline

## Data Flow

```
Dataset prompts
  → vLLM inference (TP=4, EP=1, DP=1, 2 engines)
    → Agent rollout loop (BiomniCodeActAgent, max 50 iterations)
      → Reward computation (gt + rubric + format, max 7)
        → Megatron forward/backward pass (GSPO loss)
          → CPU-offloaded optimizer step
            → Checkpoint save (every 4 steps, synchronous)
```

## Megatron Parallelism (Current Config)

| Dimension | Value | File |
|-----------|-------|------|
| Tensor Parallel | 2 | `run_biomni_qwen30ba3b_rubric_gspo_tis.sh` |
| Pipeline Parallel | 1 | same |
| Context Parallel | 4 | same |
| Expert Parallel | 8 | same |
| Expert Tensor Parallel | derived | same |

## GSPO (Group Sequence Policy Optimization)

**File**: `skyrl_train/utils/ppo_utils.py`, function `gspo_policy_loss` (~line 596)

Key mechanism: sequence-level importance sampling replaces per-token ratios.

```
log_importance_weights = masked_mean(log_probs - old_log_probs, mask, dim=-1)
log_token_iw = log_probs - log_probs.detach() + log_importance_weights.detach()
ratio = exp(clamp(log_token_iw, max=10))
```

Current clipping bounds are very tight: `eps_clip_low=3e-4`, `eps_clip_high=4e-4`.

**Expected behavior**: `ppo_clip_ratio > 0` from step 1. The sequence-level mean log-ratio redistributed across tokens creates per-token ratios that deviate from 1.0 even before any optimizer update. This is inherent to the GSPO formulation and not a bug.

## Optimizer

- Fully CPU-offloaded (`optimizer_cpu_offload=true`, `optimizer_offload_fraction=1.0`)
- Overlap D2H/H2D transfers enabled
- Precision-aware optimizer enabled
- GPU OOMs in `optimizer_step` / `get_grad_norm_fp32` are typically transient memory fragmentation, not sustained pressure

## Checkpointing

**Files**: `skyrl_train/distributed/megatron/megatron_strategy.py` (save), `skyrl_train/trainer.py` (orchestration + cleanup, ~line 1077), `skyrl_train/utils/trainer_utils.py` (cleanup_old_checkpoints, ~line 124)

- Uses `AsyncCallsQueue(persistent=True)` to avoid `os.fork()` issues with Ray
- Despite the async queue, **actual saves are synchronous** (`async_sharded_save=False`, line ~181)
- All Megatron workers serialize sharded state dicts through `/dev/shm` → requires `--shm-size=64g`
- Checkpoint dir structure: `{ckpt_path}/global_step_N/` containing `policy/`, `critic/` (if used), `trainer_state.json`
- Resume tracking: `{ckpt_path}/latest_ckpt_global_step.txt` — written atomically after all saves succeed; `resume_mode=latest` reads this
- Built-in cleanup via `cleanup_old_checkpoints()`: keeps the N most recent checkpoints per `max_ckpts_to_keep` (default: -1 = keep all). Does NOT support keeping every Nth step — for that, use manual cleanup (see monitor-training skill)
- Current config: `ckpt_interval=4`, `max_ckpts_to_keep=-1`
- Checkpoint path: `/mnt/biomni_filestore/models/skyrlagent/biomni-training-qwen3-30b-a3b-skyrlagent-gspo/biomni-training-qwen3-30b-a3b-8gpus-rubric-gspo-no-tis-eps3e4-4e4/`
- Checkpoint interval: 4 steps
- Checkpoint path: configured via `trainer.ckpt_path`
- `resume_mode=latest` is **critical** — must be set for autoretry to resume from checkpoint instead of restarting from scratch

## Overlong Rollout Filter

Configured in the YAML (`overlong_filter_enabled: true, overlong_filter_threshold: 40000`). Masks out loss for rollouts that are BOTH >40000 tokens AND have ft_reward == 0. Prevents degenerate long non-compliant outputs from polluting the policy gradient.

## Key Metrics

| Metric | Source | Notes |
|--------|--------|-------|
| `policy_loss` (pg) | `megatron_worker.py` ~line 372 | GSPO surrogate loss |
| `ppo_clip_ratio` | `megatron_worker.py` ~line 374 | Fraction of tokens clipped |
| `policy_entropy` (ent) | `megatron_worker.py` ~line 375 | Action entropy |
| `raw_grad_norm` | `megatron_worker.py` ~line 382 | Post-clipping gradient norm |
| `response_length` | `megatron_worker.py` ~line 385 | Per-micro-batch response length |
| `avg_final_rewards` | `trainer.py` ~line 765 | Mean return across batch |
| `avg_response_length` | `trainer.py` ~line 756 | Mean response length across batch |
| `avg_raw_reward` | `trainer.py` ~line 700 | Pre-advantage reward |
| Heavy sample logs | Every `log_heavy_freq=8` steps | Detailed trajectory + reward breakdowns |

## TIS (Truncated Importance Sampling)

Currently **disabled** (`use_tis=false`). Planned for future use to correct rollout-vs-trainer logprob mismatch. Supports `token` and `sequence` modes; `sequence` mode is recommended for GSPO.
