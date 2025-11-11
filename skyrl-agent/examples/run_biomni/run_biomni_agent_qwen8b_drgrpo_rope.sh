#!/usr/bin/env bash

# Colocated GRPO training with skyrl-train + skyrl-agent for Qwen3-8B on Biomni tasks
set -euo pipefail
set -x

# Basic environment setup
export PYTHONUNBUFFERED=1
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1

# NCCL timeouts/debug (optional)
export NCCL_TIMEOUT=28800
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# -----------------------------
# User-configurable paths
# -----------------------------
PROJECT_NAME="biomni-training-qwen3-8b-grpo"
EXPERIMENT_NAME="biomni-training-qwen3-8b-32bsz-temp0.6-clip-0.28-32turn-grpo"

DATA_PATH="/home/ray/default/screen_design_rl"
TRAIN_FILE="$DATA_PATH/train.parquet"
VAL_FILE="$DATA_PATH/test.parquet"
SFT_MODEL_PATH="/dfs/scratch0/lansong/models/qwen/qwen3-8b-sft-v1/global_step_66"
CKPT_PATH="/dfs/scratch1/lansong/models/qwen"

# If using a remote inference runtime (recommended for large TP):
# Specify as host:port WITHOUT the http:// prefix (skyrl-train adds it)
RUNTIME_HOSTPORT="172.24.75.232:8000"

# -----------------------------
# Training hyperparameters
# -----------------------------
BATCH_SIZE=32
MAX_NUM_ITERS=32
NUM_TRAJ=8
MAX_PARALLEL_AGENTS=128
SAVE_FREQ=4

USE_KL_LOSS=true
KL_LOSS_COEF=0.001

ENTROPY_COEFF=0.001   # not directly exposed; left here for parity
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28

# Parallelism
GPU_MEM_UTIL=0.8
TP_SIZE=2                 # tensor parallel for inference engine
SP_SIZE=2                 # sequence parallel for policy & ref
NUM_GPUS_PER_NODE=8
NNODES=1

TEMPERATURE=0.6
TOP_P=0.95

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

# If local SFT checkpoint isn't available on Ray workers, fall back to an HF repo id.
# Override with: export HF_MODEL_ID="namespace/repo"
: "${HF_MODEL_ID:=Qwen/Qwen2.5-7B-Instruct}"
if [ -d "$SFT_MODEL_PATH" ]; then
  MODEL_PATH="$SFT_MODEL_PATH"
else
  MODEL_PATH="$HF_MODEL_ID"
fi

# Ensure a project venv and install torch inside it (flash-attn builds in this venv)
if [ ! -d "$SKYRL_AGENT_DIR/.venv" ]; then
  (cd "$SKYRL_AGENT_DIR" && uv venv .venv)
fi
source "$SKYRL_AGENT_DIR/.venv/bin/activate"

# GPU default: CUDA 12.1 wheels. Override for CPU with:
#   export PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
: "${PYTORCH_INDEX_URL:=https://download.pytorch.org/whl/cu121}"
uv pip install --index-url "$PYTORCH_INDEX_URL" "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"

uv run --no-build-isolation --extra skyrl-train -m skyrl_agent.integrations.skyrl_train.skyrl_train_main \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$VAL_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.kl_loss_coef=$KL_LOSS_COEF \
  trainer.algorithm.loss_reduction="seq_mean_token_sum_norm" \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.sequence_parallel_size=$SP_SIZE \
  trainer.ref.sequence_parallel_size=$SP_SIZE \
  trainer.gradient_checkpointing=true \
  trainer.strategy=fsdp2 \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.epochs=100 \
  trainer.train_batch_size=$BATCH_SIZE \
  trainer.policy_mini_batch_size=$BATCH_SIZE \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.max_prompt_length=31744 \
  trainer.eval_before_train=true \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=$SAVE_FREQ \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$EXPERIMENT_NAME" \
  trainer.ckpt_path="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME" \
  trainer.export_path="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME/exports" \
  trainer.logger="['console','wandb']" \
  trainer.resume_mode=from_path \
  trainer.resume_path="/dfs/scratch0/lansong/models/qwen/biomni-training-qwen3-8b-grpo/biomni-training-qwen3-8b-32bsz-temp0.6-clip-0.28-32turn-grpo-reward2/global_step_4" \
  generator.backend=vllm \
  generator.run_engines_locally=false \
  generator.remote_inference_engine_urls="['$RUNTIME_HOSTPORT']" \
  generator.inference_engine_tensor_parallel_size=$NUM_GPUS_PER_NODE \
  generator.gpu_memory_utilization=$GPU_MEM_UTIL \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.sampling_params.max_generate_length=3072 \
  +generator.task="$AGENT_TASK_YAML" \
  $@

popd >/dev/null

