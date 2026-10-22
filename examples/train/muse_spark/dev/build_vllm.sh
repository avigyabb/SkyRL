#!/usr/bin/env bash
# Build a vLLM wheel from vllm-project/vllm#61879 (Muse Spark Open) for torch 2.14 / cu130.
set -euo pipefail
ROOT=/shared/hackskyrl/muse_spark
export RUSTUP_HOME=$ROOT/rust/rustup CARGO_HOME=$ROOT/rust/cargo
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
fi
export PATH=$CARGO_HOME/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
# GitHub HTTP/2 streams from this box get reset mid-clone ("early EOF"); force HTTP/1.1 for git.
# GCC >= 13 for DeepGEMM/DeepSelect C++20 <format> (system has 11.4); runtime uses anaconda libstdc++ 3.4.35.
G=/shared/hackskyrl/muse_spark/gcc13/bin/x86_64-conda-linux-gnu
export CC=$G-gcc CXX=$G-g++ CUDAHOSTCXX=$G-g++
# elfutils headers for DeepGEMM deep_jit (it dlopens libdw at runtime; the system has libdw.so.1).
export CPATH=/shared/hackskyrl/muse_spark/gcc13/include/elfutils-only
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.version GIT_CONFIG_VALUE_0=HTTP/1.1

cd $ROOT
[ -d build-venv ] || uv venv --python 3.12 build-venv
source build-venv/bin/activate
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  -r vllm-src/requirements/build/cuda.txt -r vllm-src/requirements/build/rust.txt

cd vllm-src
export MAX_JOBS=${MAX_JOBS:-100} NVCC_THREADS=2
# SM90 for the H100 CI runners, SM100 for B200.
export TORCH_CUDA_ARCH_LIST="9.0 10.0"
# Rust chat frontend: optional. Its llm-multimodal dep (Inferact/llm-multimodal-internal) is private.
mkdir -p $ROOT/wheels
uv build --wheel --no-build-isolation -o $ROOT/wheels .
ls -la $ROOT/wheels
