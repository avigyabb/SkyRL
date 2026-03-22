#!/usr/bin/env bash
# DEBUG: Biomni CodeAct 30B-A3B with GSPO + dual_clip + Self-Distillation (SDFT)
# Small batch for quick iteration. Same prompt lengths as production for memory estimation.
# Combines:
#   - GSPO sequence-level importance sampling (handles MoE routing discrepancy)
#   - DAPO dual_clip lower bound (maintains gradient for negative-advantage trajectories)
#   - Relaxed eps_clip 2e-3/3e-3 (~7x original, absorbs Megatron CP numerical noise)
#   - seq_mean_token_sum_norm reduction (fixes length bias for variable-length trajectories)
#   - Cosine LR scheduler (prevents late-training entropy collapse)
#   - SDFT: EMA teacher + full-vocab reverse KL(student||teacher) auxiliary loss

set -euo pipefail
set -x

ulimit -c 0

export PYTHONUNBUFFERED=1
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
: "${OPENAI_API_KEY:=sc}"
export OPENAI_API_KEY

export NCCL_TIMEOUT=28800
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=enp0s19
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=LOC

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export RAY_grpc_keepalive_time_ms=60000
export RAY_grpc_keepalive_timeout_ms=600000

export WANDB_API_KEY="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"

export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_DISABLE_COMPILE_CACHE=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

export UV_CACHE_DIR=/mnt/biomni_filestore/uv_cache
export XDG_CACHE_HOME=$UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT=/mnt/biomni_filestore/venvs/skyrl-agent
export HOME=/workspace

export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export UV_HTTP_TIMEOUT=1800
export BIOMNI_RUNTIME_URL="http://10.138.0.3:8000"

export BIOMNI_CRITIC_MODEL="${BIOMNI_CRITIC_MODEL:-claude-sonnet-4-5}"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "WARNING: ANTHROPIC_API_KEY is not set. Rubric evaluation will fail."
fi

# ── Paths ──
PROJECT_NAME="debug-gspo-dualclip-sdft"
EXPERIMENT_NAME="debug-gspo-dualclip-sdft-reinforce"

DATA_PATH="/mnt/local/biomni/skyrl-data"
TRAIN_FILE="$DATA_PATH/train_freeform.parquet"
VAL_FILE="$DATA_PATH/val_freeform.parquet"

CKPT_PATH="/mnt/biomni_filestore/models/skyrlagent"
MODEL_NAME="/mnt/biomni_filestore/model_weights/biomni-r1-30b-a3b-sft-v0/global_step_46"

# ── Parallelism ──
NNODES=1
NUM_GPUS_PER_NODE=8

MEGATRON_TP=2
MEGATRON_PP=1
MEGATRON_CP=4
MEGATRON_EP=8
MEGATRON_ETP=1

INFER_TP=4
INFER_EP=1
INFER_DP=1
NUM_INFERENCE_ENGINES=$((NUM_GPUS_PER_NODE / (INFER_TP * INFER_DP)))

# ── RL / optimization (GSPO + dual_clip) — DEBUG: small batch ──
TRAIN_BATCH_SIZE=2
MINI_BATCH_SIZE=2
LR=1e-6

# Relaxed ~7x from GSPO paper's 3e-4/4e-4 to absorb Megatron CP numerical noise
# (measured sequence-level noise σ ≈ 2.9e-4, so 2e-3 ≈ 7σ → ~0% spurious clipping)
EPS_LOW="2e-3"
EPS_HIGH="3e-3"

USE_KL_LOSS=false
KL_LOSS_COEF=0.0

# ── Lengths ──
MAX_PROMPT_LENGTH=32768
MAX_RESPONSE_LENGTH=4096
VLLM_MAX_MODEL_LEN=35000

# ── Agent config ──
AGENT_TASK_YAML="$(cd "$(dirname "$0")" && pwd)/../run_biomni/biomni_codeact_rubric_rl_qwen30ba3b_sdft_debug.yaml"

SKYRL_AGENT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
pushd "$SKYRL_AGENT_DIR" >/dev/null

LOGGER="['console','wandb']"

VENV_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
CUDNN_PATH="$($VENV_PYTHON -c 'import inspect, nvidia.cudnn as c, os; print(os.path.dirname(inspect.getfile(c)))' 2>/dev/null || echo '')"
if [ -n "$CUDNN_PATH" ]; then
  export CPATH="$CUDNN_PATH/include:${CPATH:-}"
  export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"
fi

PYTHONUNBUFFERED=1 uv run --frozen --extra skyrl-train --env-file ~/SkyRL/skyrl-agent/examples/run_biomni/.env.biomni \
  -m skyrl_agent.integrations.skyrl_train.skyrl_train_main \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$VAL_FILE']" \
  trainer.strategy=megatron \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=0 \
  trainer.placement.critic_num_gpus_per_node=0 \
  trainer.algorithm.advantage_estimator="rloo" \
  trainer.algorithm.policy_loss_type="gspo_dual_clip" \
  trainer.algorithm.clip_ratio_c=10.0 \
  trainer.algorithm.loss_reduction="seq_mean_token_sum_norm" \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.eps_clip_low=$EPS_LOW \
  trainer.algorithm.eps_clip_high=$EPS_HIGH \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.kl_loss_coef=$KL_LOSS_COEF \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.algorithm.use_tis=false \
  trainer.algorithm.tis_imp_ratio_cap=2.0 \
  +trainer.algorithm.tis_mode=sequence \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.scheduler=cosine_with_min_lr \
  '+trainer.policy.optimizer_config.scheduler_specific_kwargs={min_lr: 1e-7}' \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0 \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=true \
  trainer.gradient_checkpointing=true \
  trainer.epochs=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.inference_engine_tensor_parallel_size=$INFER_TP \
  generator.inference_engine_expert_parallel_size=$INFER_EP \
  generator.inference_engine_data_parallel_size=$INFER_DP \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.gpu_memory_utilization=0.20 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.max_input_length=$MAX_PROMPT_LENGTH \
  +generator.engine_init_kwargs.max_model_len=$VLLM_MAX_MODEL_LEN \
  generator.max_num_seqs=256 \
  generator.enforce_eager=true \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=999 \
  trainer.ckpt_path="/tmp/debug_gspo_dualclip_sdft_ckpt" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$EXPERIMENT_NAME" \
  trainer.logger="['console']" \
  trainer.resume_mode=none \
  trainer.flash_attn=true \
  trainer.use_sample_packing=true \
  +generator.task="$AGENT_TASK_YAML" \
  +trainer.sdft_enabled=true \
  +trainer.sdft_ema_decay=0.99 \
  +trainer.sdft_loss_coef=0.8 \
  +trainer.sdft_kl_estimator="reinforce" \
  "$@"

popd >/dev/null
