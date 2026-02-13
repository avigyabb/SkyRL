export UV_CACHE_DIR=/mnt/biomni_filestore/uv_cache
export XDG_CACHE_HOME=$UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT=/mnt/biomni_filestore/venvs/skyrl-agent
export RAY_RUNTIME_ENV_WORKING_DIR_CACHE_SIZE_GB=64
# export HOME=/dfs/scratch1/lansong  # not needed — uses container default

# Add CUDNN headers + library path for transformer_engine builds and runtime
CUDNN_PKG_DIR=/mnt/biomni_filestore/venvs/skyrl-agent/lib/python3.12/site-packages/nvidia/cudnn
ln -sf $CUDNN_PKG_DIR/include/*.h /usr/local/cuda/include/
export LD_LIBRARY_PATH="${CUDNN_PKG_DIR}/lib:${LD_LIBRARY_PATH}"

# Disable flashinfer version check (for vLLM workers)
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_DISABLE_COMPILE_CACHE=1

# Prevent Ray from overriding CUDA_VISIBLE_DEVICES (causes invalid device ordinal errors)
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

# NCCL settings for distributed GPU communication
export NCCL_TIMEOUT=28800
export NCCL_DEBUG=INFO                       # temporarily verbose for multi-node debugging
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=enp0s19            # force NCCL to use the host network (10.138.0.x)
export NCCL_IB_DISABLE=1                     # GCP standard VMs have no InfiniBand
export NCCL_NET_GDR_LEVEL=LOC                # disable GPUDirect RDMA (not avail on standard VMs)

# Increase Ray worker registration timeout (default 60s is too short with slow disk/uv)
export RAY_worker_register_timeout_seconds=300

# Use cached HuggingFace models only (avoid network timeouts when multiple workers access HF)
# Disabled until model is pre-downloaded:
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# Biomni runtime server URL (code-execution backend)
export BIOMNI_RUNTIME_URL="${BIOMNI_RUNTIME_URL:-http://10.138.0.4:8000}"

# Ensure Ray workers can find all package metadata from the shared venv
export PYTHONPATH="/mnt/biomni_filestore/venvs/skyrl-agent/lib/python3.12/site-packages:${PYTHONPATH:-}"

# Use ray directly from the venv to avoid uv dependency sync overhead
VENV_BIN=/mnt/biomni_filestore/venvs/skyrl-agent/bin

$VENV_BIN/ray stop -f
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
$VENV_BIN/ray start --address 10.138.0.3:6379 --num-gpus 8 --num-cpus 128