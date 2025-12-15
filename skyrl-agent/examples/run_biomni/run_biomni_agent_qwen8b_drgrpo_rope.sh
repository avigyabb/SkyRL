#!/usr/bin/env bash

set -euo pipefail
set -x

# Basic environment setup
export PYTHONUNBUFFERED=1
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
: "${OPENAI_API_KEY:=sc}"
export OPENAI_API_KEY

# NCCL timeouts/debug (optional)
export NCCL_TIMEOUT=28800
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

export FLASHINFER_DISABLE_VERSION_CHECK=1

# export VLLM_USE_V1=0
export VLLM_DISABLE_COMPILE_CACHE=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
# export VLLM_ALLREDUCE_USE_SYMM_MEM=0

export UV_CACHE_DIR=/dfs/scratch1/lansong/uv_cache
export XDG_CACHE_HOME=$UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT=/dfs/scratch1/lansong/venvs/skyrl-agent
export HOME=/dfs/scratch1/lansong
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export UV_HTTP_TIMEOUT=1800
export BIOMNI_RUNTIME_URL="http://172.24.75.90:8000"

# -----------------------------
# User-configurable paths
# -----------------------------
PROJECT_NAME="biomni-training-qwen3-8b-skyrlagent-drgrpo"
EXPERIMENT_NAME="biomni-training-qwen3-8b-32bsz-temp1.0-clip-0.28-48turn-skyrlagent-drgrpo-rope"

DATA_PATH="/dfs/scratch1/lansong/BioAgentOS/biomni_env_screen/data/rl_data/skyrl_agent"
TRAIN_FILE="$DATA_PATH/train.parquet"
VAL_FILE="$DATA_PATH/val.parquet"
SFT_MODEL_PATH="/dfs/scratch1/lansong/models/qwen/qwen3-8b-sft-full-v1/global_step_208"
CKPT_PATH="/dfs/scratch1/lansong/models/skyrlagent"


# RUNTIME_HOSTPORT="172.24.75.90:8000"

# -----------------------------
# Training hyperparameters
# -----------------------------
BATCH_SIZE=32
# MAX_NUM_ITERS=48
NUM_TRAJ=8
# MAX_PARALLEL_AGENTS=128
SAVE_FREQ=8

USE_KL_LOSS=False
KL_LOSS_COEF=0.000

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28

# Parallelism
GPU_MEM_UTIL=0.75
TP_SIZE=1
SP_SIZE=4
NUM_GPUS_PER_NODE=8
NNODES=1

TEMPERATURE=1.0
TOP_P=1.0

# -----------------------------
# Agent task config
# -----------------------------
AGENT_TASK_YAML="$(cd "$(dirname "$0")" && pwd)/../run_biomni/biomni_codeact_rl_qwen8b.yaml"

# -----------------------------
# Run
# -----------------------------
# Ensure uv runs in the skyrl-agent project directory so --extra skyrl-train is resolvable
SKYRL_AGENT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
pushd "$SKYRL_AGENT_DIR" >/dev/null

LOGGER="['console','wandb']"

# Set logger: enable wandb only if WANDB_API_KEY is available
# LOGGER="['console']"
# if [ -n "${WANDB_API_KEY:-}" ]; then
#   LOGGER="['console','wandb']"
# fi

# If local SFT checkpoint isn't available on Ray workers, fall back to HF repo id.
: "${HF_MODEL_ID:=Qwen/Qwen3-8B}"
if [ -d "$SFT_MODEL_PATH" ]; then
  MODEL_PATH="$SFT_MODEL_PATH"
else
  MODEL_PATH="$HF_MODEL_ID"
fi

# # Ensure a project venv and install torch inside it (flash-attn builds in this venv)
# if [ ! -d "$SKYRL_AGENT_DIR/.venv" ]; then
#   (cd "$SKYRL_AGENT_DIR" && uv venv .venv)
# fi
# source "$SKYRL_AGENT_DIR/.venv/bin/activate"

# # GPU default: CUDA 12.1 wheels. Override for CPU with:
# #   export PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# : "${PYTORCH_INDEX_URL:=https://download.pytorch.org/whl/cu121}"
# uv pip install --index-url "$PYTORCH_INDEX_URL" "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"
# uv pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"


PYTHONUNBUFFERED=1 uv run --extra skyrl-train --env-file /dfs/scratch1/lansong/SkyRLV1/skyrl-agent/examples/run_biomni/.env.biomni -m skyrl_agent.integrations.skyrl_train.skyrl_train_main \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$VAL_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.kl_loss_coef=$KL_LOSS_COEF \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.algorithm.loss_reduction="seq_mean_token_sum_norm" \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.sequence_parallel_size=$SP_SIZE \
  trainer.policy.megatron_config.tensor_model_parallel_size=1 \
  trainer.gradient_checkpointing=true \
  trainer.strategy=fsdp2 \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=0 \
  trainer.placement.critic_num_gpus_per_node=0 \
  trainer.epochs=1 \
  trainer.train_batch_size=$BATCH_SIZE \
  trainer.policy_mini_batch_size=$BATCH_SIZE \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.max_prompt_length=45056 \
  trainer.eval_before_train=true \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=$SAVE_FREQ \
  trainer.ckpt_path="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$EXPERIMENT_NAME" \
  trainer.logger="$LOGGER" \
  trainer.resume_mode=none \
  +trainer.policy.model.override_config.max_position_embeddings=49152 \
  +trainer.policy.model.override_config.rope_scaling.rope_type=yarn \
  +trainer.policy.model.override_config.rope_scaling.factor=1.5 \
  +trainer.policy.model.override_config.rope_scaling.original_max_position_embeddings=32768 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.n_samples_per_prompt=$NUM_TRAJ \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  generator.num_inference_engines=$((NUM_GPUS_PER_NODE * NNODES / TP_SIZE)) \
  generator.gpu_memory_utilization=$GPU_MEM_UTIL \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.sampling_params.max_generate_length=4096 \
  generator.max_input_length=45056 \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.policy.fsdp_config.reshard_after_forward=true \
  +generator.task="$AGENT_TASK_YAML" \
  +generator.engine_init_kwargs.rope_scaling.rope_type=yarn \
  +generator.engine_init_kwargs.rope_scaling.factor=1.5 \
  +generator.engine_init_kwargs.rope_scaling.original_max_position_embeddings=32768 \
  +generator.engine_init_kwargs.max_model_len=49152 \
  +generator.use_log_heavy=true \
  +generator.log_heavy_freq=8 \
  $@


#   generator.num_inference_engines=$((NUM_GPUS_PER_NODE * NNODES / TP_SIZE)) \
#   trainer.export_path="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME/exports"
#   trainer.resume_mode=from_path
#   trainer.resume_path="/dfs/scratch0/lansong/models/qwen/biomni-training-qwen3-8b-grpo/biomni-training-qwen3-8b-32bsz-temp0.6-clip-0.28-32turn-grpo-reward2/global_step_4"


# for yarn, needs sglang
  # '+generator.engine_init_kwargs.json_model_override_args="{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":1.5,\"original_max_position_embeddings\":32768},\"max_position_embeddings\":49152}"' \
  # +trainer.policy.model.override_config.max_position_embeddings=49152 \
  # +trainer.policy.model.override_config.rope_scaling.rope_type=yarn \
  # +trainer.policy.model.override_config.rope_scaling.factor=1.5 \
  # +trainer.policy.model.override_config.rope_scaling.original_max_position_embeddings=32768 \

popd >/dev/null
