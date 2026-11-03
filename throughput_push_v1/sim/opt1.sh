#!/bin/bash
# Optimized colocated RL sim: same as rl_sim.sh baseline plus the throughput-push-v1 options.
cd "$(dirname "$0")"
STEPS=${STEPS:-2} NAME=${NAME:-opt1} MLP_CHUNKS=${MLP_CHUNKS:-1} GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5} bash rl_sim_glm53_flash_lora.sh \
  trainer.old_logprobs_from_first_minibatch=true \
  trainer.balance_dp_tokens=true \
  trainer.policy.megatron_config.keep_frozen_weights_on_gpu=true "$@"
