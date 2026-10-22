#!/usr/bin/env bash
# SkyRL megatron+dev test venv on torch 2.14 with the vllm#61879 (Muse Spark) wheel.
# 1) the current lock's resolution (reqs_213.txt) minus everything torch-coupled, installed --no-deps
#    (re-resolving would pull Megatron-Bridge's own git pins for TE etc.);
# 2) the torch-2.14 set + vLLM wheel, with dependency resolution for vLLM's new requirements.
set -euo pipefail
D=/shared/hackskyrl/muse_spark
V=$D/venv214
cd $V

grep -vE '^(torch==|torchvision|torchaudio|torchcodec|vllm==|flash-attn|flash-mla|flashinfer|mamba-ssm|causal-conv1d|transformer-engine-torch|fast-hadamard-transform|triton|nvidia-|cuda-|tilelang|apache-tvm-ffi|quack-kernels|xgrammar|-e )' reqs_213.txt \
  | grep -vE '^\s' > reqs_filtered.txt

[ -d .venv ] || uv venv --python /home/ray/anaconda3/bin/python3.12 .venv
export VIRTUAL_ENV=$V/.venv
uv pip install --no-deps -r reqs_filtered.txt
uv pip install --index-strategy unsafe-best-match --override $V/overrides.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl/cu130 --extra-index-url https://flashinfer.ai/whl/ \
  "torch==2.14.0" "torchvision==0.29.0" \
  $D/wheels/vllm-*.whl \
  "flashinfer-python==0.7.1" "flashinfer-jit-cache==0.7.1" "flashinfer-cubin==0.7.1" \
  $D/abi/out/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl \
  $D/abi/out/mamba_ssm-2.3.2.post1-cp312-cp312-linux_x86_64.whl \
  $D/abi/out/causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl \
  $D/abi/out/transformer_engine_torch-2.19.0-cp312-cp312-linux_x86_64.whl \
  "transformer-engine==2.19.0" "transformer-engine-cu13==2.19.0" "transformers==5.16.1"
# nvidia-resiliency-ext / nvidia-modelopt / nvidia-cuda-nvdisasm are caught by the nvidia- filter above.
uv pip install --no-deps -r $V/reqs_nvidia_extra.txt "nvidia-cuda-nvdisasm==13.4.92"
uv pip install --no-deps -e $D/SkyRL -e $D/SkyRL/skyrl-gym -e $D/SkyRL/skycap
$V/.venv/bin/python -c "import torch, vllm, transformer_engine.pytorch, megatron.bridge, flash_attn; print(torch.__version__, vllm.__version__)"
