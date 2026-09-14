"""
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/inference_servers/test_sonic_mirror.py

Covers the ``generator.inference_engine.sonic_mirror`` knob: the vLLM CLI args it
produces and the config validation around it. sonicloader itself is not required;
its presence is stubbed through ``importlib.util.find_spec``.
"""

import importlib.util

import pytest

from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_inference_engine_cfg

MIRROR = "s3://bucket/sonic-mirror/"


def _cfg(**overrides) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = "Qwen/Qwen2.5-0.5B-Instruct"
    ie_cfg = cfg.generator.inference_engine
    ie_cfg.sonic_mirror = MIRROR
    for key, value in overrides.items():
        setattr(ie_cfg, key, value)
    return cfg


@pytest.fixture
def sonic_installed(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: object() if name == "sonic" else real(name, *a, **k)
    )


def test_cli_args_without_mirror_leave_load_format_alone():
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = "Qwen/Qwen2.5-0.5B-Instruct"
    args = build_vllm_cli_args(cfg)
    assert args.load_format != "sonic"


def test_cli_args_select_sonic_load_format_cache_only_by_default():
    args = build_vllm_cli_args(_cfg())
    assert args.load_format == "sonic"
    assert args.model_loader_extra_config == {"mirror": MIRROR, "capture": False}


def test_cli_args_stage_weights_only_when_streaming_and_publishing():
    args = build_vllm_cli_args(_cfg(sonic_stream_weights=True))
    assert args.model_loader_extra_config == {"mirror": MIRROR, "capture": True}

    args = build_vllm_cli_args(_cfg(sonic_stream_weights=True, sonic_publish_on_startup=False))
    assert args.model_loader_extra_config == {"mirror": MIRROR, "capture": False}


def test_validation_accepts_installed_sonic(sonic_installed):
    validate_inference_engine_cfg(_cfg())


def test_validation_requires_package(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "sonic" else real(name))
    with pytest.raises(ValueError, match="sonic-loader"):
        validate_inference_engine_cfg(_cfg())


def test_validation_requires_s3_prefix(sonic_installed):
    with pytest.raises(ValueError, match="s3://"):
        validate_inference_engine_cfg(_cfg(sonic_mirror="/mnt/shared/mirror"))


def test_validation_rejects_engine_kwargs_clash(sonic_installed):
    with pytest.raises(ValueError, match="load_format"):
        validate_inference_engine_cfg(_cfg(engine_init_kwargs={"load_format": "dummy"}))


def test_validation_rejects_fp8_weight_sync(sonic_installed):
    with pytest.raises(ValueError, match="fp8_weight_sync_mode"):
        validate_inference_engine_cfg(_cfg(fp8_weight_sync_mode="blockwise"))
