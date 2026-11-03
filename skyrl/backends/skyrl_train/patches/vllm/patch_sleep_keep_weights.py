"""Sleep mode that frees only the KV cache and leaves the model weights on the GPU.

vLLM's default ``cumem`` sleep backend (``vllm/device_allocator/sleep_mode_backend.py``)
copies the weights to a freshly allocated pinned host buffer at level 1 (or discards them at
level 2), and copies them back on ``wake_up(["weights"])``. When the colocated trainer fits on
the GPU next to the engine's weights, that round trip is pure overhead: every step moves the
full per-GPU weight shard down and back up, and the level-1 host copy holds the whole model in
pinned RAM on every node.

With ``SKYRL_VLLM_SLEEP_KEEP_WEIGHTS=1`` (set from
``generator.inference_engine.sleep_keep_weights``), the ``cumem`` backend is replaced in each
vLLM worker process by one whose ``suspend`` discards only the ``kv_cache`` pool, at every
level. Waking ``weights`` is then a no-op and waking ``kv_cache`` remaps the pool; its contents
are undefined, as after a level-1 sleep, and the engine resets the prefix cache on every
level >= 1 sleep. A weight sync writes into the resident weights.
"""

import os

KEEP_WEIGHTS_ENV = "SKYRL_VLLM_SLEEP_KEEP_WEIGHTS"


def _keep_weights_backend_cls():
    from vllm.device_allocator.sleep_mode_backend import CuMemBackend

    class KeepWeightsCuMemBackend(CuMemBackend):
        """``cumem`` backend whose suspend discards only the KV cache."""

        def suspend(self, level: int = 1) -> None:
            from vllm.device_allocator import get_mem_allocator_instance

            self._state = "SUSPENDED"
            get_mem_allocator_instance().discard(("kv_cache",))

    return KeepWeightsCuMemBackend


def apply_sleep_keep_weights_patch() -> None:
    """Route the ``cumem`` sleep backend to the keep-weights variant if the env var is set.

    Must run in the vLLM worker process before its first sleep (the worker creates its
    backend lazily on first use).
    """
    if os.environ.get(KEEP_WEIGHTS_ENV, "0") != "1":
        return
    from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

    SleepModeBackendFactory._registry["cumem"] = _keep_weights_backend_cls
