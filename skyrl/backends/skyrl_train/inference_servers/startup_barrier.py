"""Startup barrier: let colocated training workers build and offload their models while the
inference engines boot, without the engines sizing their KV cache against a busy GPU.

Colocated engines and training workers share GPUs. vLLM sizes its KV cache from the memory it
sees during profiling, so trainer tensors resident at that moment would shrink the cache for the
whole run -- which is why the models used to be loaded only after the engines were healthy and
slept. Most of an engine's boot (actor construction, the executor spawning its workers) happens
before it touches GPU memory, though. The entrypoint creates this barrier before launching the
engines, runs ``init_models`` (which ends with the models offloaded to CPU) while they boot, and
releases it; ``patch_startup_barrier`` makes every vLLM worker wait on it right before
``init_device``. Profiling therefore sees the same free GPU it sees today. If the trainer takes
longer than the engines' pre-device boot, the engines wait; if it fails, the entrypoint releases
the barrier so the engines never hang on it, and a worker gives up waiting after
``SKYRL_STARTUP_BARRIER_TIMEOUT_S`` (default 1800s) with a warning.

The barrier is a named Ray actor in a fixed namespace; its name travels to the engine workers in
``SKYRL_STARTUP_BARRIER`` (set on the driver by ``create_startup_barrier``, forwarded into the
engine actors' runtime env by ``build_engine_runtime_env``, inherited by vLLM's EngineCore
subprocess and copied to its Ray workers). It cannot be job-scoped: the EngineCore opens its own
Ray job. A worker without the variable, or whose barrier is gone, proceeds immediately
(non-colocated runs, the ``serve`` / ``main_generate`` entrypoints).
"""

import asyncio
import logging
import os
import uuid
from typing import Optional

import ray

logger = logging.getLogger(__name__)

BARRIER_ENV_VAR = "SKYRL_STARTUP_BARRIER"
BARRIER_NAMESPACE = "skyrl_startup_barrier"
DEFAULT_TIMEOUT_S = float(os.environ.get("SKYRL_STARTUP_BARRIER_TIMEOUT_S", "1800"))


@ray.remote(num_cpus=0)
class StartupBarrier:
    def __init__(self):
        self._released = asyncio.Event()
        self._waiters = 0

    async def wait(self) -> bool:
        self._waiters += 1
        await self._released.wait()
        return True

    async def num_waiters(self) -> int:
        """Workers that have reached the barrier so far (diagnostics / tests)."""
        return self._waiters

    async def release(self) -> None:
        self._released.set()

    async def is_released(self) -> bool:
        return self._released.is_set()


def create_startup_barrier() -> ray.actor.ActorHandle:
    """Driver side: create the barrier and publish its name in ``SKYRL_STARTUP_BARRIER``.

    Call before the engines are launched so the name is in their runtime env.
    """
    name = f"skyrl_startup_barrier_{uuid.uuid4().hex[:12]}"
    handle = StartupBarrier.options(name=name, namespace=BARRIER_NAMESPACE).remote()
    ray.get(handle.is_released.remote())  # constructed and registered before the engines can look it up
    os.environ[BARRIER_ENV_VAR] = name
    return handle


def release_startup_barrier(handle: Optional[ray.actor.ActorHandle]) -> Optional[int]:
    """Driver side: let the engine workers proceed to their CUDA init. Idempotent.

    Returns how many workers had reached the barrier (``None`` without a barrier).
    """
    if handle is None:
        return None
    waiting = ray.get(handle.num_waiters.remote())
    ray.get(handle.release.remote())
    return waiting


def wait_for_startup_barrier(name: Optional[str] = None, timeout_s: float = DEFAULT_TIMEOUT_S) -> bool:
    """Engine worker side: block until the barrier named by ``name`` / ``SKYRL_STARTUP_BARRIER`` is released.

    Returns False without waiting when there is no barrier (variable unset, actor gone, or not
    running under Ray). A timeout is logged and treated as released so a worker never hangs.
    """
    name = name or os.environ.get(BARRIER_ENV_VAR)
    if not name or not ray.is_initialized():
        return False
    try:
        handle = ray.get_actor(name, namespace=BARRIER_NAMESPACE)
    except ValueError:
        return False
    try:
        ray.get(handle.wait.remote(), timeout=timeout_s)
    except ray.exceptions.GetTimeoutError:
        logger.warning(
            "startup barrier not released after %.0fs; proceeding with CUDA init (the KV cache may be "
            "sized against trainer memory)",
            timeout_s,
        )
    return True
