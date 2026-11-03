"""The plain Harbor generator's straggler cutoff for synchronous generation batches."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from omegaconf import OmegaConf

pytest.importorskip("harbor")

from examples.train_integrations.harbor import harbor_generator  # noqa: E402
from examples.train_integrations.harbor.entrypoints.main_harbor import (  # noqa: E402
    HARBOR_DEFAULT_CONFIG,
)
from examples.train_integrations.harbor.harbor_generator import (  # noqa: E402
    HarborGenerator,
    StragglerCutoffConfig,
)
from skyrl.train.generators.base import TrajectoryID  # noqa: E402

pytestmark = pytest.mark.integrations


def rewarded() -> SimpleNamespace:
    return SimpleNamespace(
        exception_info=None,
        verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
        agent_result=SimpleNamespace(
            rollout_details=[{"prompt_token_ids": [[1, 2]], "completion_token_ids": [[3]], "logprobs": [[-0.5]]}],
            metadata={"n_episodes": 1},
        ),
        environment_setup=None,
        agent_execution=None,
        verifier=None,
    )


class SleepyTrial:
    """Replaces Harbor's ``Trial``: a task path's trial takes ``delays[path]`` seconds."""

    delays: dict[str, float] = {}
    cancelled: list[str] = []

    def __init__(self, path: str) -> None:
        self.path = path

    @classmethod
    async def create(cls, config: Any) -> "SleepyTrial":
        return cls(str(config.task.path))

    async def run(self) -> SimpleNamespace:
        try:
            await asyncio.sleep(self.delays[self.path])
        except asyncio.CancelledError:
            self.cancelled.append(self.path)
            raise
        return rewarded()


def make_generator(monkeypatch, cutoff: dict | None) -> tuple[HarborGenerator, list[str]]:
    monkeypatch.setattr(harbor_generator, "Trial", SleepyTrial)
    finished_sessions: list[str] = []

    async def finish_session(session_id: str) -> None:
        finished_sessions.append(session_id)

    client = SimpleNamespace(get_endpoint_url=lambda: "http://engine", finish_session=finish_session)
    cfg = {
        "inference_engine": {"served_model_name": "policy", "enable_return_sample_support_set": False},
        "step_wise_trajectories": True,
        "merge_stepwise_output": True,
        "apply_overlong_filtering": False,
        "rate_limit": None,
    }
    if cutoff is not None:
        cfg["straggler_cutoff"] = cutoff
    with open(HARBOR_DEFAULT_CONFIG) as f:
        harbor_cfg = yaml.safe_load(f)
    return HarborGenerator(OmegaConf.create(cfg), harbor_cfg, client, tokenizer=None, max_seq_len=1024), finished_sessions


def batch(paths: list[str], samples: int) -> dict:
    prompts = [p for p in paths for _ in range(samples)]
    ids = [TrajectoryID(instance_id=p, repetition_id=r) for p in paths for r in range(samples)]
    return {"prompts": prompts, "trajectory_ids": ids}


@pytest.mark.asyncio
async def test_cutoff_cancels_and_masks_the_slowest_group(monkeypatch) -> None:
    SleepyTrial.delays = {"fast_a": 0.01, "fast_b": 0.02, "slow": 30.0}
    SleepyTrial.cancelled = []
    generator, finished = make_generator(monkeypatch, {"min_group_fraction": 0.6, "grace_period_s": 0.0})

    out = await asyncio.wait_for(generator.generate(batch(["fast_a", "fast_b", "slow"], 2), disable_tqdm=True), 10)

    # Two of three groups (ceil(0.6 * 3)) finished; the slow group's two trials were cancelled and masked.
    assert sorted(SleepyTrial.cancelled) == ["slow", "slow"]
    assert out["stop_reasons"] == ["complete"] * 4 + ["error"] * 2
    assert out["loss_masks"][4:] == [[0], [0]]
    assert out["rewards"] == [1.0] * 4 + [0.0] * 2
    metrics = out["rollout_metrics"]
    assert metrics["generate/num_straggler_cut_trajectories"] == 2
    assert metrics["generate/num_masked_instances"] == 1
    assert metrics["generate/num_error_trajectories"] == 0
    # Every attempt, including the cancelled ones, released its router session.
    assert len(finished) == 6


@pytest.mark.asyncio
async def test_grace_period_lets_a_late_group_finish(monkeypatch) -> None:
    SleepyTrial.delays = {"fast": 0.01, "late": 0.2}
    SleepyTrial.cancelled = []
    generator, _ = make_generator(monkeypatch, {"min_group_fraction": 0.5, "grace_period_s": 5.0})

    out = await asyncio.wait_for(generator.generate(batch(["fast", "late"], 2), disable_tqdm=True), 10)

    assert SleepyTrial.cancelled == []
    assert out["rewards"] == [1.0] * 4
    assert out["rollout_metrics"]["generate/num_straggler_cut_trajectories"] == 0


@pytest.mark.asyncio
async def test_default_waits_for_every_trajectory(monkeypatch) -> None:
    SleepyTrial.delays = {"fast": 0.01, "slower": 0.3}
    SleepyTrial.cancelled = []
    generator, _ = make_generator(monkeypatch, None)

    out = await asyncio.wait_for(generator.generate(batch(["fast", "slower"], 2), disable_tqdm=True), 10)

    assert SleepyTrial.cancelled == []
    assert out["rewards"] == [1.0] * 4


def test_invalid_fraction_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_group_fraction"):
        StragglerCutoffConfig(min_group_fraction=0.0)
    with pytest.raises(ValueError, match="grace_period_s"):
        StragglerCutoffConfig(min_group_fraction=0.9, grace_period_s=-1)
