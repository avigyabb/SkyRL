"""
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/inference_servers/test_initial_weights_knobs.py

Covers ``trainer.skip_initial_weight_sync``: the vLLM CLI args and the config validation.
"""

import pytest

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_inference_engine_cfg


def _cfg() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = "Qwen/Qwen2.5-0.5B-Instruct"
    cfg.trainer.resume_mode = "none"
    return cfg


def test_skip_initial_weight_sync_accepts_fresh_non_colocated_run():
    cfg = _cfg()
    cfg.trainer.skip_initial_weight_sync = True
    cfg.trainer.placement.colocate_all = False
    validate_inference_engine_cfg(cfg)


@pytest.mark.parametrize(
    "mutate, match",
    [
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
