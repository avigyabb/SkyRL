#!/usr/bin/env bash
# Biomni CodeAct training with GSPO + SDFT via Tinker backend (Qwen3-30B-A3B)
#
# Key differences from the Megatron-based SDFT script:
#   - Uses Tinker LoRA training (JAX) instead of Megatron full-parameter training
#   - Frozen teacher (no EMA) — teacher is the base model without LoRA updates
#   - Reinforce KL estimator (memory-free, no full-vocab logit materialization)
#   - No tensor/sequence/context parallelism config — Tinker handles sharding internally

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export RUST_BACKTRACE=1

export UV_CACHE_DIR=/mnt/biomni_filestore/uv_cache
export XDG_CACHE_HOME=$UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT=/mnt/biomni_filestore/venvs/skyrl-agent
export UV_HTTP_TIMEOUT=1800

ENV_FILE="$(cd "$(dirname "$0")" && pwd)/.env.biomni"
if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
else
  echo "ERROR: Missing env file: $ENV_FILE" >&2; exit 1
fi

# -----------------------------
# LLM Critic Configuration
# -----------------------------
export BIOMNI_CRITIC_MODEL="${BIOMNI_CRITIC_MODEL:-claude-sonnet-4-5}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-sk-placeholder}"

# -----------------------------
# User-configurable paths
# -----------------------------
PROJECT_NAME="biomni-tinker-qwen3-30b-a3b-rubric-gspo-sdft"
EXPERIMENT_NAME="biomni-tinker-qwen3-30b-a3b-rubric-gspo-sdft-reinforce"

DATA_PATH="/mnt/local/biomni/skyrl-data"
TRAIN_FILE="$DATA_PATH/train_freeform.parquet"
VAL_FILE="$DATA_PATH/val_freeform.parquet"

MODEL_NAME="Qwen/Qwen3-30B-A3B"

TASK_YAML="$(cd "$(dirname "$0")" && pwd)/biomni_codeact_rubric_rl_qwen30ba3b_gspo_sdft_tinker.yaml"
SKYRL_AGENT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
pushd "$SKYRL_AGENT_DIR" >/dev/null

OUTPUT_DIR="${OUTPUT_DIR:-$HOME/skyrlagent/$PROJECT_NAME/$EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "[log] Writing run output to: $LOG_FILE"

# -----------------------------
# RL / optimization knobs
# -----------------------------
BATCH_SIZE=16
LR=1e-6
LORA_RANK=32
GROUP_SIZE=8
MAX_STEPS=200
SAVE_EVERY=4
EVAL_EVERY=10

# SDFT configuration (frozen teacher, reinforce KL estimator)
SDFT_LOSS_COEF=1.0

uv run --isolated --extra tinker \
  -m skyrl_agent.integrations.tinker.tinker_train \
  model_name="$MODEL_NAME" \
  skyrl_agent_task_yaml="$TASK_YAML" \
  dataset_file="$TRAIN_FILE" \
  eval_dataset_file="$VAL_FILE" \
  batch_size=$BATCH_SIZE \
  eval_batch_size=128 \
  learning_rate=$LR \
  lora_rank=$LORA_RANK \
  max_steps=$MAX_STEPS \
  save_every=$SAVE_EVERY \
  eval_every=$EVAL_EVERY \
  loss_fn="ppo" \
  group_size=$GROUP_SIZE \
  normalize_advantages=true \
  sdft_enabled=true \
  sdft_loss_coef=$SDFT_LOSS_COEF \
  wandb_project="$PROJECT_NAME" \
  wandb_name="$EXPERIMENT_NAME" \
  log_dir="$OUTPUT_DIR" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

popd >/dev/null
