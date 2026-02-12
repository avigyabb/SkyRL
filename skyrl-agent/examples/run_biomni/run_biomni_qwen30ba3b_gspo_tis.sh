#!/usr/bin/env bash
set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
: "${OPENAI_API_KEY:=sc}"
export OPENAI_API_KEY

export NCCL_TIMEOUT=28800
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# Help with CUDA memory fragmentation during weight sync
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_DISABLE_COMPILE_CACHE=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

export UV_CACHE_DIR=/dfs/scratch1/lansong/uv_cache
export XDG_CACHE_HOME=$UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT=/dfs/scratch1/lansong/venvs/skyrl-agent
export HOME=/dfs/scratch1/lansong
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export UV_HTTP_TIMEOUT=1800
export BIOMNI_RUNTIME_URL="http://172.24.75.90:8000"

# export VLLM_USE_V1=0

PROJECT_NAME="biomni-training-qwen3-30b-a3b-skyrlagent-gspo"
EXPERIMENT_NAME="biomni-training-qwen3-30b-a3b-16gpus-gspo-tis-eps3e4-4e4"

DATA_PATH="/dfs/scratch1/lansong/BioAgentOS/biomni_env_screen/data/rl_data/skyrl_agent"
TRAIN_FILE="$DATA_PATH/train.parquet"
VAL_FILE="$DATA_PATH/val.parquet"

CKPT_PATH="/dfs/scratch1/lansong/models/skyrlagent"
# Local fine-tuned model
MODEL_NAME="/dfs/scratch1/lansong/models/qwen/biomni-r1-30b-a3b-sft-v0/global_step_46"

# -----------------------------
# Cluster / parallelism
# -----------------------------
NNODES=2
NUM_GPUS_PER_NODE=8
NUM_GPUS_TOTAL=$((NNODES * NUM_GPUS_PER_NODE))

# Megatron parallelism (reasonable starting point for 32 GPUs on MoE)
MEGATRON_TP=4
MEGATRON_PP=1
MEGATRON_CP=4
MEGATRON_EP=16
MEGATRON_ETP=1

# vLLM inference parallelism (EP = DP * TP constraint)
# Simplified: DP=1 to avoid NCCL conflicts between multiple engines with internal DP
INFER_TP=8
INFER_EP=1
INFER_DP=1
NUM_INFERENCE_ENGINES=$((NUM_GPUS_TOTAL / (INFER_TP * INFER_DP)))  # 16 / (4 * 1) = 4 engines

# -----------------------------
# RL / optimization knobs
# -----------------------------
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
LR=1e-6

# GSPO clipping (GSPO-scale, not GRPO-scale)
EPS_LOW="3e-4"
EPS_HIGH="4e-4"

# No KL by default (GSPO author suggested trying KL=0 in VERL thread)
USE_KL_LOSS=false
KL_LOSS_COEF=0.0

# TIS (for rollout-vs-train logprob mismatch)
USE_TIS=true
TIS_IMP_RATIO_CAP=2.0
# Recommended for GSPO: sequence-level TIS to avoid token-wise variance (requires code diff below)
TIS_MODE="sequence"   # token|sequence

# Lengths
MAX_PROMPT_LENGTH=49152
MAX_RESPONSE_LENGTH=4096
# vLLM model len = prompt + response (+ small buffer)
VLLM_MAX_MODEL_LEN=55000

TEMPERATURE=1.0
TOP_P=1.0
FLASH_ATTN=true

AGENT_TASK_YAML="$(cd "$(dirname "$0")" && pwd)/../run_biomni/biomni_codeact_rl_qwen30ba3b_gspo.yaml"

SKYRL_AGENT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
pushd "$SKYRL_AGENT_DIR" >/dev/null

LOGGER="['console','wandb']"

PYTHONUNBUFFERED=1 uv run --extra skyrl-train --env-file /dfs/scratch1/lansong/SkyRLV1/skyrl-agent/examples/run_biomni/.env.biomni \
  -m skyrl_agent.integrations.skyrl_train.skyrl_train_main \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$VAL_FILE']" \
  trainer.strategy=megatron \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=0 \
  trainer.placement.critic_num_gpus_per_node=0 \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="gspo" \
  trainer.algorithm.loss_reduction="sequence_mean" \
  trainer.algorithm.eps_clip_low=$EPS_LOW \
  trainer.algorithm.eps_clip_high=$EPS_HIGH \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.kl_loss_coef=$KL_LOSS_COEF \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.algorithm.use_tis=$USE_TIS \
  trainer.algorithm.tis_imp_ratio_cap=$TIS_IMP_RATIO_CAP \
  +trainer.algorithm.tis_mode=$TIS_MODE \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.policy.optimizer_config.lr=$LR \
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
  generator.gpu_memory_utilization=0.5 \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.sampling_params.logprobs=0 \
  generator.max_input_length=$MAX_PROMPT_LENGTH \
  +generator.engine_init_kwargs.max_model_len=$VLLM_MAX_MODEL_LEN \
  generator.enforce_eager=true \
  trainer.eval_before_train=true \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=4 \
  trainer.ckpt_path="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$EXPERIMENT_NAME" \
  trainer.logger="$LOGGER" \
  trainer.resume_mode=latest \
  trainer.flash_attn=$FLASH_ATTN \
  trainer.use_sample_packing=true \
  +generator.task="$AGENT_TASK_YAML" \
  $@

popd >/dev/null
