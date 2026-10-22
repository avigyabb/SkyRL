"""Full Muse-Spark-Open-816B-A42B Megatron <-> vLLM roundtrip (local only; not part of the PR).

Reuses test_megatron_models.test_logprobs_matching_roundtrip:
Megatron TP8 / EP64 over 64 GPUs (8 x 8xB200), vLLM TP8 x PP8 x 1 engine (~25 GB/GPU each),
colocated. vLLM TP across nodes (TP16) hangs in its gloo is_in_the_same_node barrier on this cluster,
in plain vLLM too, so the engine keeps TP inside a node and pipelines across four.
Run from the repo root as a Ray job attached to the cluster, with the torch-2.14 venv as py_executable.
"""

import pytest

from tests.backends.skyrl_train.gpu.gpu_ci.megatron import test_megatron_models as t

FULL = "/shared/hackskyrl/muse_spark/Muse-Spark-Open-816B-A42B"

_orig_overrides = t._engine_overrides_for_model


def _overrides(model_name, fp8_mode=None):
    out = _orig_overrides(model_name, fp8_mode)
    # The full model needs ~102 GB/GPU at TP16; vLLM sleeps while Megatron runs, so no co-residency.
    out["gpu_memory_utilization"] = 0.8
    return out


_orig_get_cfg = t.get_test_actor_config


def _get_cfg(model_name):
    cfg = _orig_get_cfg(model_name)
    # As the GLM-5.3-Flash recipes do: the colocated sync otherwise OOMed with 115 GB held by the
    # Megatron worker (25 GB of weights) next to vLLM's receive buffers.
    cfg.trainer.use_expandable_segments = True
    return cfg


_orig_create = t.InferenceEngineState.create.__func__


def _create(cls, cfg, *args, **kwargs):
    # In place on the test's cfg (create deep-copies), so the trainer's weight sync sees it too.
    ie = cfg.generator.inference_engine
    # The NCCL sync wakes vLLM weights next to the resident Megatron shard, and Bridge's fused-expert
    # export materializes a layer's full gate_up_proj (~17 GB) on every rank. On 32 GPUs that OOMed
    # (51 + 51 + 17 GB + caches); on 64 both halves drop to ~25 GB/GPU.
    ie.tensor_parallel_size, ie.pipeline_parallel_size, ie.num_engines = 8, 8, 1
    return _orig_create(cls, cfg, *args, **kwargs)


@pytest.mark.asyncio
async def test_muse_spark_full_roundtrip(monkeypatch):
    monkeypatch.setattr(t, "_engine_overrides_for_model", _overrides)
    monkeypatch.setattr(t, "get_test_actor_config", _get_cfg)
    monkeypatch.setattr(t.InferenceEngineState, "create", classmethod(_create))
    await t.test_logprobs_matching_roundtrip(
        tp=8,
        pp=1,
        cp=1,
        ep=64,
        etp=1,
        inference_tp=16,
        num_gpus=64,
        model_name=FULL,
        vllm_threshold=3e-1,
        megatron_threshold=1e-1,
        fp8_mode=None,
        max_generate_length=None,
    )
