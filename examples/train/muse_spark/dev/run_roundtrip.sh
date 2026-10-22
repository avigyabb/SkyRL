#!/usr/bin/env bash
# Muse Spark Megatron <-> vLLM roundtrip in the torch-2.14 venv, on a 4-GPU worker allocation.
cd /shared/hackskyrl/muse_spark/SkyRL
export RAY_ADDRESS=local
# The job holds the whole node; a local raylet rejects a CUDA_VISIBLE_DEVICES subset of the node GPUs.
unset CUDA_VISIBLE_DEVICES
# Anyscale turns on event export to its aggregator; a nested local Ray has none and SEGVs on shutdown.
export RAY_enable_core_worker_ray_event_to_aggregator=0 RAY_enable_ray_event=0
# The outer Ray job's runtime env leaks into nested ray.init calls (vLLM's engine core); merge instead of failing.
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
# Tests read ~/data/gsm8k; worker nodes only have it on /shared.
mkdir -p ~/data && ln -sfn /shared/hackskyrl/data/gsm8k ~/data/gsm8k
export SKYRL_MUSE_SPARK_4LAYER=/shared/hackskyrl/muse_spark/Muse-Spark-Open-4layer
export FLASHINFER_WORKSPACE_BASE=/tmp/fi_ws_214
export PATH=/shared/hackskyrl/muse_spark/venv214/.venv/bin:$PATH
python -m pytest -s -v -m "h100 or not h100" "$@"
