"""Startup barrier used to overlap the colocated model build with the engine boot
(``inference_servers/startup_barrier.py``): engine workers wait until the entrypoint releases it,
and proceed immediately when their job has no barrier.

uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/inference_servers/test_startup_barrier.py
"""

import time

import pytest
import ray

from skyrl.backends.skyrl_train.inference_servers import startup_barrier as sb


@pytest.fixture(scope="module")
def ray_ctx():
    started = not ray.is_initialized()
    if started:
        ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    yield
    if started:
        ray.shutdown()


@ray.remote(num_cpus=0)
def _worker_waits(name, timeout_s):
    # Ray worker processes do not see the test process's env, so the name is passed explicitly
    # (in a training run it arrives through the engine actors' runtime env).
    t0 = time.monotonic()
    waited = sb.wait_for_startup_barrier(name=name, timeout_s=timeout_s)
    return waited, time.monotonic() - t0


def _name():
    import os

    return os.environ[sb.BARRIER_ENV_VAR]


def test_workers_block_until_released(ray_ctx):
    barrier = sb.create_startup_barrier()
    try:
        ref = _worker_waits.remote(_name(), 60.0)
        deadline = time.monotonic() + 30
        while ray.get(barrier.num_waiters.remote()) < 1:  # the worker has reached the barrier
            assert time.monotonic() < deadline, "worker never reached the barrier"
            time.sleep(0.1)
        ready, _ = ray.wait([ref], timeout=1.0)
        assert not ready, "worker proceeded before the barrier was released"
        sb.release_startup_barrier(barrier)
        waited, _elapsed = ray.get(ref, timeout=30)
        assert waited is True
        sb.release_startup_barrier(barrier)  # idempotent
    finally:
        ray.kill(barrier)


def test_worker_proceeds_without_barrier(ray_ctx):
    waited, elapsed = ray.get(_worker_waits.remote("skyrl_startup_barrier_missing", 5.0), timeout=30)
    assert waited is False and elapsed < 5.0


def test_worker_gives_up_after_timeout(ray_ctx):
    barrier = sb.create_startup_barrier()
    try:
        waited, elapsed = ray.get(_worker_waits.remote(_name(), 1.0), timeout=30)
        assert waited is True and 1.0 <= elapsed < 10.0
    finally:
        ray.kill(barrier)
