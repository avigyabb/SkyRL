set -x

# Qwen3.5-9B DAPO with an NVFP4 rollout (W4A4 on Blackwell FP4 tensor cores), three activation-scale arms:
#   ARM=w4a16       weight-only NVFP4 rollout (no activation quantization)
#   ARM=global      W4A4, ONE static activation amax for every layer (SKYRL_NVFP4_INPUT_AMAX, default 64)
#   ARM=calibrated  W4A4, a per-layer amax measured by the trainer on its own forward passes
#                   (SKYRL_NVFP4_CALIBRATE_INPUT=1); no hand-tuned constant
#
# STEPS=5 ARM=calibrated bash examples/train/fp4/run_nvfp4_w4a4_calibration_qwen35_9b.sh
#
# Knobs for the calibrated arm: SKYRL_NVFP4_CALIB_DECAY (default 0.9, forgets a stale max slowly),
# SKYRL_NVFP4_CALIB_MARGIN (default 1.0, headroom multiplier). The first sync happens before the trainer
# has seen data and uses SKYRL_NVFP4_INPUT_AMAX (default 128) for every layer.

: "${ARM:=calibrated}"
: "${STEPS:=5}"
: "${DATA_DIR:=/shared/nvfp4_recipe_data}"
export DATA_DIR

case "$ARM" in
  w4a16)      unset SKYRL_NVFP4_INPUT_AMAX SKYRL_NVFP4_CALIBRATE_INPUT ;;
  global)     export SKYRL_NVFP4_INPUT_AMAX="${SKYRL_NVFP4_INPUT_AMAX:-64}"; unset SKYRL_NVFP4_CALIBRATE_INPUT ;;
  calibrated) export SKYRL_NVFP4_CALIBRATE_INPUT=1 ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac

LOGGER=console bash "$(dirname "$0")/run_nvfp4_blackwell_qwen35_9b.sh" \
  generator.inference_engine.fp8_weight_sync_mode=nvfp4 \
  trainer.max_training_steps=$STEPS \
  trainer.run_name="nvfp4_w4a4_${ARM}" \
  "$@"
