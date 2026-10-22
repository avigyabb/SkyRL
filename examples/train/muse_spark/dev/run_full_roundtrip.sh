#!/usr/bin/env bash
# Full-model roundtrip: driver attaches to the cluster; workers run the torch-2.14 venv (py_executable).
cd /shared/hackskyrl/muse_spark/SkyRL
mkdir -p ~/data && ln -sfn /shared/hackskyrl/data/gsm8k ~/data/gsm8k
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export FLASHINFER_WORKSPACE_BASE=/tmp/fi_ws_214
export PATH=/shared/hackskyrl/muse_spark/venv214/.venv/bin:$PATH
python -m pytest -s -v tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_muse_full_local.py
