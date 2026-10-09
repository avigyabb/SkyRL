set -x

# Colocated DAPO with NVFP4 training GEMMs for Qwen3.5-9B-Base (experimental).
# Hardware: 1 node of 8xB200
#
# bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# bash examples/train/fp4/run_nvfp4_blackwell_qwen35_9b.sh
#
# Only the trainer's linear-layer GEMMs run NVFP4 (Transformer Engine NVFP4BlockScaling:
# 1x16 E4M3 scales, random Hadamard transform and stochastic rounding on gradients). Primary weights
# stay BF16 and rollout weights are synced as plain BF16, so the rollout policy is not the quantized
# trainer forward: expect a rollout-vs-trainer logprob gap ~5-10x larger than the FP8 recipes. Token-level
# TIS (below) is REQUIRED: without it a 40-step 9B run learned for 20 steps and then regressed.
#
# Settings that matter (measured on 8xB200, see examples/train/fp4/README.md):
#   - TP=1. At TP=2 the quantized sequence-parallel path made policy_train ~3x slower.
#   - NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1: 1D 1x16 weight scales; ~25% lower train/rollout gap, no cost.
#   - trainer.algorithm.off_policy_correction.tis_ratio_type=token.

MODEL_NAME="Qwen/Qwen3.5-9B-Base"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
LOGGER="${LOGGER:-wandb}"  # change to "console" to print to stdout

# Colocated by default: training and inference share the same GPUs. For a
# disaggregated (non-colocated) run, set COLOCATE_ALL=false and split the GPUs,
# e.g. trainer.placement.policy_num_gpus_per_node=4 with the remaining GPUs
# given to the inference engines via generator.inference_engine.num_engines.
COLOCATE_ALL=${COLOCATE_ALL:-true}

NUM_NODES=1
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=8
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=1

MEGATRON_TP=1
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=1
MEGATRON_ETP=1

# Qwen3.5 goes through the VL bridge (Qwen3VLModel), which packs sequences in its own
# forward and conflicts with SkyRL sample packing; language_model_only routes it to the
# native GPTModel + GDN THD packing path on both the trainer and vLLM.
LANGUAGE_MODEL_ONLY=true

# ---- NVFP4: trainer GEMMs only ----
# fp4 and fp8 are mutually exclusive in Megatron. fp4_param is not supported.
MEGATRON_FP4=e2m1
MEGATRON_FP4_RECIPE=nvfp4
# Forwarded to every Ray worker (TE reads it when it builds the recipe).
export NVTE_NVFP4_DISABLE_2D_QUANTIZATION=${NVTE_NVFP4_DISABLE_2D_QUANTIZATION:-1}

# fla's default TileLang GDN backend aborts in the packed backward on Blackwell (surfaces as
# a CUDA "misaligned address" from the next Triton launch); force the Triton GDN kernels.
# Leave unset on Hopper, where the Triton GDN backward is the broken one:
# https://github.com/fla-org/flash-linear-attention/issues/640#issuecomment-4236520788
export FLA_TILELANG=0

uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="regular" \
  trainer.algorithm.overlong_buffer_len=4096 \
  trainer.algorithm.overlong_buffer_penalty_factor=1.0 \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.off_policy_correction.tis_ratio_type=token \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0 \
  trainer.algorithm.clip_ratio_c=10.0 \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  generator.apply_overlong_filtering=true \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.max_generate_length=8192 \
  generator.sampling_params.logprobs=1 \
  generator.eval_sampling_params.temperature=1.0 \
  generator.eval_sampling_params.top_p=1.0 \
  generator.eval_sampling_params.max_generate_length=8192 \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.ref.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.placement.colocate_all=$COLOCATE_ALL \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.ref.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.ref.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.fp4=$MEGATRON_FP4 \
  trainer.ref.megatron_config.fp4=$MEGATRON_FP4 \
  trainer.policy.megatron_config.fp4_recipe=$MEGATRON_FP4_RECIPE \
  trainer.ref.megatron_config.fp4_recipe=$MEGATRON_FP4_RECIPE \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=16 \
  trainer.epochs=20 \
  trainer.max_training_steps=400 \
  trainer.eval_batch_size=512 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=2048 \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.logger="$LOGGER" \
  trainer.project_name="skyrl_nvfp4" \
  trainer.run_name="nvfp4_blackwell_qwen35_9b" \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=null \
  trainer.max_ckpts_to_keep=3 \
  $@
