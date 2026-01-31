export UV_CACHE_DIR=/dfs/scratch1/lansong/uv_cache
export XDG_CACHE_HOME=$UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT=/dfs/scratch1/lansong/venvs/skyrl-agent
export RAY_RUNTIME_ENV_WORKING_DIR_CACHE_SIZE_GB=64
export HOME=/dfs/scratch1/lansong

# Add CUDNN library path for transformer_engine (libcudnn_graph.so.9)
CUDNN_LIB_PATH=/dfs/scratch1/lansong/venvs/skyrl-agent/lib/python3.12/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH="${CUDNN_LIB_PATH}:${LD_LIBRARY_PATH}"

# Disable flashinfer version check (for vLLM workers)
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_DISABLE_COMPILE_CACHE=1

# Prevent Ray from overriding CUDA_VISIBLE_DEVICES (causes invalid device ordinal errors)
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

# NCCL settings for distributed GPU communication
export NCCL_TIMEOUT=28800
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

# Increase Ray worker registration timeout (default 60s is too short with slow disk/uv)
export RAY_worker_register_timeout_seconds=300

# Use cached HuggingFace models only (avoid network timeouts when multiple workers access HF)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Use ray directly from the venv to avoid uv dependency sync overhead
VENV_BIN=/dfs/scratch1/lansong/venvs/skyrl-agent/bin

$VENV_BIN/ray stop -f
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
$VENV_BIN/ray start --address 172.24.75.90:6379 --num-gpus 8 --num-cpus 128