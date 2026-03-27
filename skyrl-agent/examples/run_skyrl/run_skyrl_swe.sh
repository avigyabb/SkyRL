#!/bin/bash
set -x

# Increase file descriptor limit for many concurrent runtime connections
ulimit -n 65536

# =============================================================================
# Tinker RL Training for SWE-Bench Task
# =============================================================================
# Converted from SkyRL-Train backend to Tinker backend.
# Uses Tinker's LoRA training with PPO loss and GRPO advantages.
# =============================================================================

# Data paths
DATA_DIR="/mnt/local/shared_storage/datasets/r2e-all"
TRAIN_DATA="${DATA_DIR}/train.parquet"
VAL_DATA="${DATA_DIR}/validation.parquet"

# Tinker API key
export TINKER_API_KEY="tml-8WIYS48Cbwk4f9p2IZodpi3p3DCtjF17wjh0Yr55jNnseVvomSxrmk1aLerMRYQAEAAAA"

# Wandb API key (get yours from https://wandb.ai/authorize - must be 40 chars)
export WANDB_API_KEY="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"  # Set this to your 40-char wandb key

# OpenHands Remote Runtime (for SWE-bench sandboxes)
export ALLHANDS_API_KEY="sandbox-remote"
export SANDBOX_REMOTE_RUNTIME_API_URL="${SANDBOX_REMOTE_RUNTIME_API_URL:-http://10.138.0.4:3000}"

# Model configuration
MODEL="Qwen/Qwen3-32B"
LORA_RANK="${LORA_RANK:-16}"

# Training hyperparameters
BATCH_SIZE=32
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
MAX_STEPS="${MAX_STEPS:-200}"
SAVE_EVERY="${SAVE_EVERY:-2}"
EVAL_EVERY="${EVAL_EVERY:-10}"

# RL configuration
LOSS_FN="${LOSS_FN:-ppo}"
GROUP_SIZE="${GROUP_SIZE:-4}"  # Should match num_trajectories in YAML
NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-false}"

# Logging
WANDB_PROJECT="${WANDB_PROJECT:-skyagent-32b-r2e-tinker}"
WANDB_NAME="${WANDB_NAME:-skyagent-tinker-32b-r2e-4500-loop-tool}"
seed=1

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/ckpts}"
mkdir -p "$OUTPUT_DIR"

# Task configuration
TASK_YAML="./examples/run_tinker/tinker_swe.yaml"

echo "================================================"
echo "Tinker RL Training Configuration - SWE-Bench"
echo "================================================"
echo "Model: $MODEL"
echo "Train Dataset: $TRAIN_DATA"
echo "Val Dataset: $VAL_DATA"
echo "Task YAML: $TASK_YAML"
echo "Batch Size: $BATCH_SIZE"
echo "Group Size (GRPO): $GROUP_SIZE"
echo "Max Steps: $MAX_STEPS"
echo "Output: $OUTPUT_DIR"
echo "================================================"

# Run training
uv run --isolated --extra tinker -m skyrl_agent.integrations.tinker.tinker_train \
    model_name="$MODEL" \
    skyrl_agent_task_yaml="$TASK_YAML" \
    dataset_file="$TRAIN_DATA" \
    eval_dataset_file="$VAL_DATA" \
    batch_size="$BATCH_SIZE" \
    learning_rate="$LEARNING_RATE" \
    lora_rank="$LORA_RANK" \
    seed="$seed" \
    max_steps="$MAX_STEPS" \
    save_every="$SAVE_EVERY" \
    eval_every="$EVAL_EVERY" \
    loss_fn="$LOSS_FN" \
    group_size="$GROUP_SIZE" \
    normalize_advantages="$NORMALIZE_ADVANTAGES" \
    wandb_project="$WANDB_PROJECT" \
    wandb_name="$WANDB_NAME" \
    log_dir="$OUTPUT_DIR" \
    "$@"

echo "================================================"
echo "Training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "================================================"
