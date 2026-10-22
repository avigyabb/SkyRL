set -x

# GRPO on GSM8K for Muse Spark Open (816B-A42B) with Megatron + LoRA, on 8 nodes of 8xB200.
#
#   uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
#   export WANDB_API_KEY=<your_key_here>   # or leave LOGGER=console
#   bash examples/train/muse_spark/run_gsm8k_muse_spark_lora_8node.sh
#
# Needs a vLLM build with Muse Spark (vllm-project/vllm#61879). PYTHON_CMD lets you point the
# launcher at an environment that has it instead of the project's `uv run`.
#
# Layout: NOT colocated. The model is ~1.63 TB in bf16. A colocated weight sync needs Megatron's
# shard, its export temporaries (Bridge materializes each layer's fused 256-expert gate_up_proj,
# ~17 GB, on every rank) and vLLM's weights + receive buffers on the same GPU, which OOMed even on
# 64 B200s. Split instead: Megatron on 4 nodes (TP8/EP32: ~51 GB of frozen base per GPU), vLLM on the
# other 4 (one engine, TP8 x PP4: ~51 GB of weights per GPU). vLLM TP across nodes (TP16) hangs in
# its gloo is_in_the_same_node barrier on this cluster, so the engine pipelines across nodes instead.
#
# Why LoRA: a full fine-tune keeps fp32 master + Adam state (~12 B/param, ~9.8 TB) next to bf16
# weights and grads, more HBM than 8 nodes have. LoRA keeps the base frozen.

MODEL_PATH="${MODEL_PATH:-/shared/hackskyrl/muse_spark/Muse-Spark-Open-816B-A42B}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
PYTHON_CMD="${PYTHON_CMD:-uv run --isolated --extra megatron}"
LOGGER="${LOGGER:-console}"  # change to "wandb" to log to wandb

POLICY_NUM_NODES=4
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TP=8
INFERENCE_ENGINE_PP=4

MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
INFERENCE_ENGINE_MAX_MODEL_LEN=2048

# 32 prompts x 8 samples = 256 sequences per step, one optimizer step per batch.
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
MAX_TOKENS_PER_MICROBATCH=4096

USE_KL_LOSS=false
LR=1e-5

# LoRA on Megatron's dense attention names (linear_qkv is the fused Q/K/V/output-gate projection)
# and the routed experts' grouped linears. merge_lora must stay true: vLLM has no MoE LoRA for
# Muse Spark ("post-expert norm does not support MoE LoRA"), so the merged weights are synced.
LORA_RANK=32
LORA_ALPHA=32
MERGE_LORA=true
LORA_TARGET_MODULES='[linear_qkv,linear_proj,linear_fc1,linear_fc2]'

# Megatron mesh: EP carries the 256 experts (8 per GPU); TP covers the ~50B of dense weights.
MEGATRON_TP=8
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=32
MEGATRON_ETP=1

OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

# Muse Spark ships as a vision/audio/language checkpoint; SkyRL trains the language model.
LANGUAGE_MODEL_ONLY=true
# Leave headroom for the weight sync, outside vLLM's memory budget. SkyRL floors the packed NCCL
# buffer at the largest single parameter -- the fused 256-expert gate_up_proj, ~16 GiB -- and vLLM
# double-buffers it, next to the unpacked tensor and the layer being rebuilt. At 0.85, 0.7 and 0.55
# that OOMed inside update_weights (a 24.5 GiB allocation with 24.5 GiB free), and because the trainer
# only checks the update RPC after its broadcast, the sync then hung until the NCCL watchdog (~36 min)
# instead of failing. 0.4 is ~71 GB of a B200: ~51 GB of weights + ~20 GB of KV, >100 GB left free.
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.4}"
ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"'}'

RUN_NAME="muse_spark_gsm8k_lora_r${LORA_RANK}_tp${MEGATRON_TP}_ep${MEGATRON_EP}"

$PYTHON_CMD -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.strategy=megatron \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.loss_reduction="sequence_mean" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_nodes=$POLICY_NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=false \
  trainer.policy.model.lora.rank=$LORA_RANK \
  trainer.policy.model.lora.alpha=$LORA_ALPHA \
  trainer.policy.model.lora.target_modules="$LORA_TARGET_MODULES" \
  trainer.policy.megatron_config.lora_config.merge_lora=$MERGE_LORA \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.remove_microbatch_padding=true \
  trainer.use_expandable_segments=true \
  trainer.logprobs_chunk_size=1024 \
  trainer.max_tokens_per_microbatch=$MAX_TOKENS_PER_MICROBATCH \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.update_epochs_per_batch=1 \
  trainer.epochs=1 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=10 \
  trainer.ckpt_interval=-1 \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/$RUN_NAME" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.distributed_executor_backend="ray" \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.inference_engine.pipeline_parallel_size=$INFERENCE_ENGINE_PP \
  generator.inference_engine.gpu_memory_utilization=$INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION \
  `# The sync's receive allocations vary in size up to ~24.5 GiB per tensor; without expandable` \
  `# segments the caching allocator fragments and reserved memory climbed to 172 GB at 0.4.` \
  generator.inference_engine.use_expandable_segments=true \
  `# One parameter per broadcast. The packed path's receive chunks are floored at the largest` \
  `# parameter (~16 GiB fused experts), live outside gpu_memory_utilization, and OOMed every rank` \
  `# (164 GiB allocated 13 s into the sync); one tensor at a time peaks ~91 GiB at 0.4.` \
  generator.inference_engine.weight_transfer_packed=false \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.batched=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  environment.env_class=gsm8k \
  trainer.logger="$LOGGER" \
  trainer.project_name="muse_spark_gsm8k" \
  trainer.run_name="$RUN_NAME" \
  "$@"
