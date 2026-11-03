#!/bin/bash
# Compressed colocated RL step for GLM-5.3-Flash (real weights, LoRA) on 2x8 B200.
# Same per-GPU shapes as examples/train/glm5_3_flash/run_dapo_glm5p3_flash_lora_sync_2node.sh,
# small batch so a step is minutes. Usage: NAME=x [VAR=..] bash rl_sim.sh [overrides]
set -x
W=/shared/hackskyrl/tput_v1/SkyRL
cd $W
export PYTHONPATH=$W
NAME=${NAME:?}
MODEL_PATH=${MODEL_PATH:-/shared/hackskyrl/models/glm53_flash_bf16}
DATA_DIR=/shared/hackskyrl/data/dapo
TBS=${TBS:-16}; MBS=${MBS:-16}; NSAMP=${NSAMP:-8}
STEPS=${STEPS:-3}
MAX_PROMPT=${MAX_PROMPT:-2048}; MAX_RESP=${MAX_RESP:-8192}
MAX_MODEL_LEN=$((MAX_PROMPT + MAX_RESP + 512))
TP=${TP:-4}; EP=${EP:-16}; ETP=${ETP:-1}
R3=${R3:-true}
MTPM=${MTPM:-8192}
LORA_RANK=${LORA_RANK:-64}
VLLM_LORA_TARGET_MODULES='["fused_qkv_a_proj", "q_b_proj", "kv_b_proj", "o_proj", "gate_up_proj", "down_proj", "in_proj_qkvbfg_a", "experts"]'
ENGINE_INIT_KWARGS='{"max_model_len": '"$MAX_MODEL_LEN"', "kv_cache_dtype": "bfloat16", "lora_target_modules": '"$VLLM_LORA_TARGET_MODULES"', "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "pass_config": {"fuse_allreduce_rms": false}}}'
ROUTER_INIT_KWARGS='{"policy": "round_robin", "queue_size": 8192, "queue_timeout_secs": 1800}'
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=2400
# /data does not exist on these nodes; without a writable dir the LoRA-frozen base weights
# fall back to ~300 GB/node of pinned RAM, which pushes the node past Ray's OOM threshold.
export SKYRL_FROZEN_OFFLOAD_DIR=${SKYRL_FROZEN_OFFLOAD_DIR:-/tmp/skyrl-frozen-offload}
export SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=128
export FLA_TILELANG=0
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export SKYRL_VLLM_START_PORT=${SKYRL_VLLM_START_PORT:-8400}
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1
export UV_CACHE_DIR=/shared/hackskyrl/uv_cache
export UV_PROJECT_ENVIRONMENT=/shared/hackskyrl/uv_venvs/tput_v1
T=trainer.policy.megatron_config.transformer_config_kwargs
uv run --no-sync --extra megatron python -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$DATA_DIR/dapo-math-17k-cleaned.parquet']" data.val_data="['$DATA_DIR/aime-2024-cleaned.parquet']" \
  trainer.strategy=megatron trainer.algorithm.advantage_estimator=grpo trainer.algorithm.policy_loss_type=dual_clip \
  trainer.algorithm.eps_clip_low=0.2 trainer.algorithm.eps_clip_high=0.28 trainer.algorithm.clip_ratio_c=10.0 \
  trainer.algorithm.loss_reduction=token_mean trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.overlong_buffer_len=2048 trainer.algorithm.overlong_buffer_penalty_factor=1.0 generator.apply_overlong_filtering=true \
  trainer.policy.model.path=$MODEL_PATH trainer.policy.language_model_only=true generator.inference_engine.language_model_only=true \
  trainer.placement.colocate_all=true trainer.placement.policy_num_nodes=2 trainer.placement.policy_num_gpus_per_node=8 \
  trainer.policy.megatron_config.tensor_model_parallel_size=$TP trainer.policy.megatron_config.pipeline_model_parallel_size=1 \
  trainer.policy.megatron_config.context_parallel_size=1 trainer.policy.megatron_config.expert_model_parallel_size=$EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$ETP trainer.policy.megatron_config.mtp_num_layers=0 \
  trainer.policy.megatron_config.moe_grouped_gemm=true trainer.policy.megatron_config.moe_token_dispatcher_type=alltoall \
  trainer.policy.megatron_config.moe_router_score_function=sigmoid trainer.policy.megatron_config.moe_router_load_balancing_type=none \
  trainer.policy.megatron_config.moe_enable_routing_replay=$R3 generator.inference_engine.enable_return_routed_experts=$R3 \
  $T.sequence_parallel=true $T.recompute_granularity=${RECOMPUTE_GRANULARITY:-selective} $T.recompute_modules=${RECOMPUTE_MODULES:-[core_attn,moe]} \
  $T.recompute_method=null $T.recompute_num_layers=null $T.mlp_chunks_for_training=${MLP_CHUNKS:-64} \
  $T.gradient_accumulation_fusion=false $T.disable_parameter_transpose_cache=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0 \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=false \
  trainer.policy.model.lora.rank=$LORA_RANK trainer.policy.model.lora.alpha=$LORA_RANK \
  trainer.policy.model.lora.target_modules='[linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj,linear_fc1,linear_fc2,q_proj,k_proj,v_proj,b_proj,f_a_proj,g_a_proj,o_proj]' \
  trainer.policy.megatron_config.lora_config.merge_lora=false trainer.policy.model.lora.share_expert_adapters=false \
  trainer.policy.megatron_config.lora_config.normalize_moe_lora=true generator.inference_engine.scale_moe_lora_rank=true \
  trainer.policy.optimizer_config.lr=1e-5 trainer.policy.optimizer_config.max_grad_norm=1.0 trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.remove_microbatch_padding=true trainer.use_expandable_segments=true trainer.fused_lm_head_logprob=true trainer.logprobs_chunk_size=1024 \
  trainer.max_tokens_per_microbatch=$MTPM trainer.micro_forward_batch_size_per_gpu=1 trainer.micro_train_batch_size_per_gpu=1 \
  trainer.train_batch_size=$TBS trainer.policy_mini_batch_size=$MBS trainer.update_epochs_per_batch=1 trainer.epochs=1 \
  trainer.max_training_steps=$STEPS trainer.max_prompt_length=$MAX_PROMPT \
  trainer.eval_batch_size=16 trainer.eval_before_train=false trainer.eval_interval=-1 trainer.ckpt_interval=-1 trainer.hf_save_interval=-1 \
  trainer.resume_mode=null trainer.ckpt_path=/shared/hackskyrl/tput_v1/ckpt/$NAME \
  generator.inference_engine.backend=vllm generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl generator.inference_engine.distributed_executor_backend=mp \
  generator.inference_engine.num_engines=${NUM_ENGINES:-2} generator.inference_engine.tensor_parallel_size=${ENGINE_TP:-8} \
  generator.inference_engine.max_num_seqs=512 generator.inference_engine.gpu_memory_utilization=${GPU_MEM_UTIL:-0.7} \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" generator.inference_engine.router_init_kwargs="$ROUTER_INIT_KWARGS" \
  generator.sampling_params.max_generate_length=$MAX_RESP generator.sampling_params.temperature=1.0 generator.sampling_params.top_p=1.0 \
  generator.batched=true generator.n_samples_per_prompt=$NSAMP environment.env_class=aime \
  trainer.logger=console trainer.project_name=tput_v1 trainer.run_name=$NAME "$@"
