"""
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/inference_servers/test_initial_weights_knobs.py

Covers ``generator.inference_engine.dummy_initial_weights`` and
``trainer.skip_initial_weight_sync``: the vLLM CLI args and the config validation.
"""

import pytest

from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_inference_engine_cfg


def _cfg() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = "Qwen/Qwen2.5-0.5B-Instruct"
    cfg.trainer.resume_mode = "none"
    return cfg


def test_dummy_initial_weights_selects_dummy_load_format():
    cfg = _cfg()
    assert build_vllm_cli_args(cfg).load_format != "dummy"
    cfg.generator.inference_engine.dummy_initial_weights = True
    assert build_vllm_cli_args(cfg).load_format == "dummy"
    validate_inference_engine_cfg(cfg)


def test_dummy_initial_weights_rejects_lora_adapter_sync():
    cfg = _cfg()
    cfg.generator.inference_engine.dummy_initial_weights = True
    cfg.trainer.strategy = "fsdp"
    cfg.trainer.policy.model.lora.rank = 8
    with pytest.raises(ValueError, match="LoRA"):
        validate_inference_engine_cfg(cfg)


def test_dummy_initial_weights_rejects_simulated_training():
    cfg = _cfg()
    cfg.generator.inference_engine.dummy_initial_weights = True
    cfg.trainer.fully_async.simulate_training = True
    with pytest.raises(ValueError, match="simulate_training"):
        validate_inference_engine_cfg(cfg)


def test_dummy_initial_weights_rejects_user_load_format():
    cfg = _cfg()
    cfg.generator.inference_engine.dummy_initial_weights = True
    cfg.generator.inference_engine.engine_init_kwargs = {"load_format": "safetensors"}
    with pytest.raises(ValueError, match="load_format"):
        validate_inference_engine_cfg(cfg)


def test_skip_initial_weight_sync_accepts_fresh_non_colocated_run():
    cfg = _cfg()
    cfg.trainer.skip_initial_weight_sync = True
    cfg.trainer.placement.colocate_all = False
    validate_inference_engine_cfg(cfg)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: setattr(c.generator.inference_engine, "dummy_initial_weights", True), "dummy_initial_weights"),
        (lambda c: setattr(c.generator.inference_engine, "weight_sync_backend", "delta"), "delta"),
        (lambda c: setattr(c.trainer, "resume_mode", "latest"), "resume_mode"),
        # Colocated engines sleep at level 2 after startup and lose their weights.
        (lambda c: setattr(c.trainer.placement, "colocate_all", True), "colocate_all"),
    ],
)
def test_skip_initial_weight_sync_rejections(mutate, match):
    cfg = _cfg()
    cfg.trainer.skip_initial_weight_sync = True
    cfg.trainer.placement.colocate_all = False
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        validate_inference_engine_cfg(cfg)
