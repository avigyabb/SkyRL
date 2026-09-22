"""
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/inference_servers/test_sonic_mirror.py

Covers the ``generator.inference_engine.sonic_mirror`` knob: the vLLM CLI args it
produces and the config validation around it. sonicloader itself is not required;
its presence is stubbed through ``importlib.util.find_spec``.
"""

import importlib.util

import pytest

pytest.importorskip("vllm", reason="build_vllm_cli_args constructs vLLM's EngineArgs")

pytestmark = pytest.mark.vllm

from skyrl.backends.skyrl_train.inference_servers.utils import (  # noqa: E402
    build_vllm_cli_args,
)
from skyrl.train.config import SkyRLTrainConfig  # noqa: E402
from skyrl.train.utils.utils import validate_inference_engine_cfg  # noqa: E402

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
    assert args.model_loader_extra_config == {"mirror": MIRROR, "capture": False, "stream": False}


def test_cli_args_stage_weights_only_when_streaming_and_publishing():
    args = build_vllm_cli_args(_cfg(sonic_stream_weights=True))
    assert args.model_loader_extra_config == {"mirror": MIRROR, "capture": True, "stream": True}

    args = build_vllm_cli_args(_cfg(sonic_stream_weights=True, sonic_publish_on_startup=False))
    assert args.model_loader_extra_config == {"mirror": MIRROR, "capture": False, "stream": True}


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


def test_default_loader_patch_keeps_sonic_extra_config_keys():
    """vLLM >= 0.28 validates model_loader_extra_config keys; the sonic format keeps its own."""
    pytest.importorskip("vllm")
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    import skyrl.backends.skyrl_train.patches.vllm.patch_sonic_loader_extra_config  # noqa: F401  (applies the patch)

    # Other formats are still validated.
    with pytest.raises(ValueError, match="Unexpected extra config keys"):
        DefaultModelLoader(LoadConfig(load_format="auto", model_loader_extra_config={"mirror": "s3://b/p/"}))

    sonic = pytest.importorskip("sonic.adapters.vllm")  # registers the `sonic` format
    from vllm.model_executor.model_loader import get_model_loader

    extra = {"mirror": "s3://b/p/", "capture": False, "stream": False}
    loader = get_model_loader(LoadConfig(load_format="sonic", model_loader_extra_config=dict(extra)))
    assert isinstance(loader, sonic.SonicLoader)
    assert loader.load_config.model_loader_extra_config == extra
    # stream=false is honored on the consume side even if the mirror holds a weights manifest
    assert getattr(sonic.SonicLoader, "_skyrl_no_stream_patched", False)
    # the HF fallback inside the loader dispatches on load_format and rejects "sonic"
    assert loader.load_config.load_format == "auto"
