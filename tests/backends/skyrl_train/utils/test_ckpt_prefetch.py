"""
uv run --extra dev --extra skyrl-train pytest tests/backends/skyrl_train/utils/test_ckpt_prefetch.py

Staging a cloud checkpoint while the models build. The load-bearing property is that the
worker thread issues no collectives -- Megatron builds on the main thread throughout and a
concurrent barrier on the same process group would deadlock -- so that is asserted directly.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from skyrl.backends.skyrl_train.utils import ckpt_prefetch as cp

CKPT = "s3://bucket/run/global_step_1/policy"
ENTRIES = ["__0_0.distcp", "__1_0.distcp", ".metadata", "metadata.json", "huggingface/"]


@pytest.fixture(autouse=True)
def clean():
    cp.clear_prefetch_state()
    yield
    cp.clear_prefetch_state()


@pytest.fixture
def env(tmp_path):
    """Patch io + dist, and route staging into tmp_path."""
    fake_io = MagicMock()
    fake_io.is_cloud_path.side_effect = lambda p: p.startswith(("s3://", "gs://", "gcs://"))
    fake_io.list_dir.return_value = ENTRIES
    fake_io.isdir.side_effect = lambda p: p.endswith("huggingface")
    fake_dist = MagicMock()
    fake_dist.get_rank.return_value = 0
    with patch.object(cp, "io", fake_io), patch.object(cp, "dist", fake_dist), patch.object(
        cp, "local_dir_for", return_value=str(tmp_path / "stage")
    ):
        yield fake_io, fake_dist, str(tmp_path / "stage")


def test_local_checkpoint_is_not_prefetched(env):
    fake_io, fake_dist, _ = env
    assert cp.start_checkpoint_prefetch("/mnt/local/ckpt/policy", node_local_rank=0) is False
    fake_dist.barrier.assert_not_called()


def test_stages_own_shard_and_metadata_then_joins(env):
    fake_io, fake_dist, stage = env
    assert cp.start_checkpoint_prefetch(CKPT, node_local_rank=0) is True
    assert cp.wait_for_checkpoint_prefetch(CKPT) == stage

    downloaded = [c.args[0] for c in fake_io.download_file.call_args_list]
    assert f"{CKPT}/__0_0.distcp" in downloaded, "rank 0 must fetch its own shard"
    assert f"{CKPT}/__1_0.distcp" not in downloaded, "and only its own"
    assert f"{CKPT}/.metadata" in downloaded and f"{CKPT}/metadata.json" in downloaded
    assert f"{CKPT}/huggingface" not in downloaded, "directories are skipped"


def test_non_local_rank_zero_skips_the_shared_metadata(env):
    fake_io, fake_dist, _ = env
    fake_dist.get_rank.return_value = 1
    cp.start_checkpoint_prefetch(CKPT, node_local_rank=1)
    cp.wait_for_checkpoint_prefetch(CKPT)
    downloaded = [c.args[0] for c in fake_io.download_file.call_args_list]
    assert downloaded == [f"{CKPT}/__1_0.distcp"]


def test_worker_thread_issues_no_collectives(env):
    """The whole design rests on this: barriers happen on the main thread, never in the thread."""
    fake_io, fake_dist, _ = env
    main_thread = __import__("threading").current_thread().name
    seen = []
    fake_dist.barrier.side_effect = lambda *a, **k: seen.append(
        __import__("threading").current_thread().name
    )
    cp.start_checkpoint_prefetch(CKPT, node_local_rank=0)
    cp.wait_for_checkpoint_prefetch(CKPT)
    assert seen == [main_thread], f"barrier ran off the main thread: {seen}"


def test_local_rank_zero_clears_the_staging_dir_before_the_barrier(env, tmp_path):
    fake_io, fake_dist, stage = env
    os.makedirs(stage, exist_ok=True)
    stale = os.path.join(stage, "stale.distcp")
    open(stale, "w").close()
    cp.start_checkpoint_prefetch(CKPT, node_local_rank=0)
    cp.wait_for_checkpoint_prefetch(CKPT)
    assert not os.path.exists(stale)
    fake_dist.barrier.assert_called_once()


def test_wait_returns_none_when_nothing_was_started(env):
    assert cp.wait_for_checkpoint_prefetch(CKPT) is None


def test_failure_reports_none_so_the_loader_falls_back(env):
    fake_io, _, _ = env
    fake_io.download_file.side_effect = RuntimeError("s3 exploded")
    assert cp.start_checkpoint_prefetch(CKPT, node_local_rank=0) is True
    assert cp.wait_for_checkpoint_prefetch(CKPT) is None


def test_second_start_is_a_noop(env):
    fake_io, fake_dist, _ = env
    assert cp.start_checkpoint_prefetch(CKPT, node_local_rank=0) is True
    assert cp.start_checkpoint_prefetch(CKPT, node_local_rank=0) is True
    fake_dist.barrier.assert_called_once()
    cp.wait_for_checkpoint_prefetch(CKPT)


def test_staging_dir_is_stable_and_node_local():
    d1, d2 = cp.local_dir_for("s3://b/a"), cp.local_dir_for("s3://b/a")
    assert d1 == d2 and d1 != cp.local_dir_for("s3://b/other")
    assert d1.startswith(__import__("tempfile").gettempdir())
