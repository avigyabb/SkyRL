#!/usr/bin/env bash
set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export RUST_BACKTRACE=1

export UV_CACHE_DIR=/mnt/biomni_filestore/uv_cache
export XDG_CACHE_HOME=$UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT=/mnt/biomni_filestore/venvs/skyrl-agent
export UV_HTTP_TIMEOUT=1800

export BIOMNI_CRITIC_MODEL="${BIOMNI_CRITIC_MODEL:-claude-sonnet-4-5}"
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "WARNING: ANTHROPIC_API_KEY is not set. Rubric evaluation will fail."
fi

PROJECT_NAME="biomni-tinker-qwen3-30b-a3b-rubric"
EXPERIMENT_NAME="biomni-tinker-qwen3-30b-a3b-rubric-gspo"

DATA_PATH="/mnt/biomni_filestore/biomni"
TRAIN_FILE="$DATA_PATH/train.parquet"
VAL_FILE="$DATA_PATH/val.parquet"
MODEL_NAME="/mnt/biomni_filestore/model_weights/biomni-r1-30b-a3b-sft-v0/global_step_92"

TASK_YAML="$(cd "$(dirname "$0")" && pwd)/biomni_codeact_rubric_rl_qwen30ba3b_tinker.yaml"
ENV_FILE="$(cd "$(dirname "$0")" && pwd)/.env.biomni"
SKYRL_AGENT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
pushd "$SKYRL_AGENT_DIR" >/dev/null

OUTPUT_DIR="/mnt/biomni_filestore/models/skyrlagent/$PROJECT_NAME/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR"
LOG_DIR="/mnt/biomni_filestore/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "[log] Writing run output to: $LOG_FILE"

uv run --isolated --extra tinker --env-file "$ENV_FILE" \
  -m skyrl_agent.integrations.tinker.tinker_train \
  model_name="$MODEL_NAME" \
  skyrl_agent_task_yaml="$TASK_YAML" \
  dataset_file="$TRAIN_FILE" \
  eval_dataset_file="$VAL_FILE" \
  batch_size=64 \
  eval_batch_size=128 \
  learning_rate=1e-6 \
  lora_rank=32 \
  max_steps=200 \
  save_every=2 \
  eval_every=10 \
  loss_fn="ppo" \
  group_size=8 \
  normalize_advantages=true \
  wandb_project="$PROJECT_NAME" \
  wandb_name="$EXPERIMENT_NAME" \
  log_dir="$OUTPUT_DIR" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

popd >/dev/null
