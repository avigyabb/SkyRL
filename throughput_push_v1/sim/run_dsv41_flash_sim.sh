#!/bin/bash
# Compressed colocated RL simulation for DeepSeek-V4.1-Flash on one 8-GPU node.
#
# Same pipeline as a real run (vLLM generation -> multi-turn env -> Megatron log-prob + fwd/bwd +
# optimizer -> weight sync -> offload/backload), on a model slice that fits one node:
#   - model: DeepSeek-V4.1-Flash architecture compressed to 8 layers / 64 experts (see
#     make_dsv41_flash_slice.py), keeping the dense / ratio-2 CSA / ratio-1 CSA layer pattern and the
#     cross-layer KV / indexer / candidate sharing; no Engram, no MTP;
#   - weights: random (trainer: SKYRL_MEGATRON_RANDOM_INIT=1, BF16; vLLM: load_format=dummy, FP8 /
#     MXFP4 experts as served); the slice has no safetensors, so the packed weight-sync buffer is set
#     explicitly (2 GB > the 1.32 GB embedding) instead of being floored from checkpoint headers;
#   - env: gsm8k_multi_turn (a random policy never answers, so every episode runs max_turns);
#   - R3 (rollout routing replay) on by default.
# Usage: NAME=x [TURNS= GEN_LEN= PROMPTS= SAMPLES= R3= EXTRA overrides...] bash run_dsv41_flash_sim.sh
set -x
SKYRL_DIR="${SKYRL_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$SKYRL_DIR"
export PYTHONPATH="$SKYRL_DIR"
NAME="${NAME:?}"
MODEL_PATH="${MODEL_PATH:-/shared/hackskyrl/models/dsv41_flash_l8_e64}"
DATA_DIR="${DATA_DIR:-/shared/hackskyrl/data/gsm8k_multi_turn}"
TURNS="${TURNS:-4}"; GEN_LEN="${GEN_LEN:-2048}"; PROMPTS="${PROMPTS:-32}"; SAMPLES="${SAMPLES:-4}"
MAX_LEN=$(( 512 + TURNS * (GEN_LEN + 256) ))
R3="${R3:-true}"; STEPS="${STEPS:-3}"
export SKYRL_MEGATRON_RANDOM_INIT=1 SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export UV_CACHE_DIR=/shared/hackskyrl/uv_cache UV_PROJECT_ENVIRONMENT=/shared/hackskyrl/uv_venvs/sft_lora_tput
uv run --no-sync --extra megatron python -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=gsm8k_multi_turn generator.max_turns=$TURNS \
  trainer.algorithm.advantage_estimator=grpo trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path="$MODEL_PATH" trainer.strategy=megatron \
  trainer.placement.colocate_all=true trainer.placement.policy_num_gpus_per_node=8 \
  trainer.policy.megatron_config.tensor_model_parallel_size=1 \
  trainer.policy.megatron_config.expert_model_parallel_size=8 \
  trainer.policy.megatron_config.moe_enable_routing_replay=$R3 \
  generator.inference_engine.enable_return_routed_experts=$R3 \
  generator.inference_engine.num_engines=8 generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.engine_init_kwargs.load_format=dummy \
  generator.inference_engine.weight_transfer_threshold_cuda_ipc_GB=2 \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_LEN \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.inference_engine.backend=vllm generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl generator.batched=false \
  generator.sampling_params.max_generate_length=$GEN_LEN generator.n_samples_per_prompt=$SAMPLES \
  trainer.max_prompt_length=$MAX_LEN trainer.algorithm.max_seq_len=$MAX_LEN \
  trainer.train_batch_size=$PROMPTS trainer.policy_mini_batch_size=$PROMPTS \
  trainer.micro_forward_batch_size_per_gpu=1 trainer.micro_train_batch_size_per_gpu=1 \
  trainer.epochs=1 trainer.update_epochs_per_batch=1 trainer.eval_interval=-1 trainer.eval_before_train=false \
  trainer.ckpt_interval=-1 trainer.hf_save_interval=-1 trainer.resume_mode=null \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.logger=console trainer.project_name=throughput_push_v1_sim trainer.run_name=$NAME \
  trainer.max_training_steps=$STEPS "$@"
