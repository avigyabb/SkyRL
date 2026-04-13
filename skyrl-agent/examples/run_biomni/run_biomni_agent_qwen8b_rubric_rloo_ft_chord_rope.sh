#!/usr/bin/env bash
# Biomni CodeAct training with RLOO + dual_clip + FORMAT-ONLY CHORD correction auxiliary SFT loss.
# Only generates corrections for format failures (ft_reward=0), not rubric/reasoning issues.
# Based on run_biomni_agent_qwen8b_rubric_rloo_chord_rope.sh with:
#   - correction_mode=format_only in YAML (no rubric corrections)
#   - correction_loss_mu=1.0 (stronger signal since format corrections are fewer but targeted)
#   - New experiment name for fresh training from step 0

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
export BIOMNI_RUNTIME_URL="http://10.138.0.4:8000"

export RAY_grpc_keepalive_time_ms=60000
export RAY_grpc_keepalive_timeout_ms=600000

export BIOMNI_CRITIC_MODEL="${BIOMNI_CRITIC_MODEL:-claude-sonnet-4-5}"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "WARNING: ANTHROPIC_API_KEY is not set. Rubric evaluation and corrections will fail."
  echo "Please set ANTHROPIC_API_KEY before running."
fi

PROJECT_NAME="biomni-training-qwen3-8b-skyrlagent-rubric-drgrpo"
EXPERIMENT_NAME="biomni-training-qwen3-8b-32bsz-temp1.0-clip-0.28-48turn-skyrlagent-rubric-rloo-ft-chord-rope-ft-gating"

DATA_PATH="/mnt/local/biomni/skyrl-data"
TRAIN_FILE="$DATA_PATH/train_freeform.parquet"
VAL_FILE="$DATA_PATH/val_freeform.parquet"
SFT_MODEL_PATH="/mnt/biomni_filestore/model_weights/qwen3-8b-sft-full-v1/global_step_104"
CKPT_PATH="/mnt/biomni_filestore/models/skyrlagent"

BATCH_SIZE=16
NUM_TRAJ=5
SAVE_FREQ=8

USE_KL_LOSS=False
KL_LOSS_COEF=0.000

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28

FLASH_ATTN=true

TP_SIZE=1
SP_SIZE=4
NUM_GPUS_PER_NODE=8
NNODES=1

TEMPERATURE=1.0
TOP_P=1.0

AGENT_TASK_YAML="$(cd "$(dirname "$0")" && pwd)/../run_biomni/biomni_codeact_rubric_rl_qwen8b_ft_chord.yaml"

SKYRL_AGENT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
pushd "$SKYRL_AGENT_DIR" >/dev/null

LOGGER="['console','wandb']"

: "${HF_MODEL_ID:=Qwen/Qwen3-8B}"
if [ -d "$SFT_MODEL_PATH" ]; then
  MODEL_PATH="$SFT_MODEL_PATH"
else
  MODEL_PATH="$HF_MODEL_ID"
fi

VENV_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
CUDNN_PATH="$($VENV_PYTHON -c 'import inspect, nvidia.cudnn as c, os; print(os.path.dirname(inspect.getfile(c)))' 2>/dev/null || echo '')"
if [ -n "$CUDNN_PATH" ]; then
  export CPATH="$CUDNN_PATH/include:${CPATH:-}"
  export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"
fi

PYTHONUNBUFFERED=1 uv run --frozen --extra skyrl-train --env-file ~/SkyRL/skyrl-agent/examples/run_biomni/.env.biomni -m skyrl_agent.integrations.skyrl_train.skyrl_train_main \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$VAL_FILE']" \
  trainer.algorithm.advantage_estimator="rloo" \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.kl_loss_coef=$KL_LOSS_COEF \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.algorithm.loss_reduction="seq_mean_token_sum_norm" \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.optimizer_config.scheduler=cosine_with_min_lr \
  'trainer.policy.optimizer_config.scheduler_specific_kwargs={min_lr: 1e-7}' \
  trainer.policy.sequence_parallel_size=$SP_SIZE \
  trainer.policy.megatron_config.tensor_model_parallel_size=1 \
  trainer.gradient_checkpointing=true \
  trainer.strategy=fsdp2 \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=0 \
  trainer.placement.critic_num_gpus_per_node=0 \
  trainer.epochs=8 \
  trainer.train_batch_size=$BATCH_SIZE \
  trainer.policy_mini_batch_size=$BATCH_SIZE \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.max_prompt_length=32768 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=$SAVE_FREQ \
  trainer.ckpt_path="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$EXPERIMENT_NAME" \
  trainer.logger="$LOGGER" \
  trainer.resume_mode=latest \
  trainer.gradient_checkpointing_use_reentrant=true \
  trainer.flash_attn=$FLASH_ATTN \
  trainer.use_sample_packing=true \
  +trainer.use_correction_loss=true \
  +trainer.correction_loss_mu=1.0 \
  +trainer.policy.model.override_config.max_position_embeddings=49152 \
  +trainer.policy.model.override_config.rope_scaling.rope_type=yarn \
  +trainer.policy.model.override_config.rope_scaling.factor=1.5 \
  +trainer.policy.model.override_config.rope_scaling.original_max_position_embeddings=32768 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.n_samples_per_prompt=$NUM_TRAJ \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  generator.num_inference_engines=$((NUM_GPUS_PER_NODE * NNODES / TP_SIZE)) \
  generator.gpu_memory_utilization=0.35 \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.sampling_params.max_generate_length=4096 \
  generator.max_input_length=32768 \
  generator.max_num_seqs=256 \
  generator.enforce_eager=true \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.policy.fsdp_config.reshard_after_forward=true \
  +generator.task="$AGENT_TASK_YAML" \
  +generator.engine_init_kwargs.rope_scaling.rope_type=yarn \
  +generator.engine_init_kwargs.rope_scaling.factor=1.5 \
  +generator.engine_init_kwargs.rope_scaling.original_max_position_embeddings=32768 \
  +generator.engine_init_kwargs.max_model_len=35000 \
  $@

popd >/dev/null
