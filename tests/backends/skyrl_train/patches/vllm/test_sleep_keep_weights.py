"""CPU tests for ``patch_sleep_keep_weights``.

A fake allocator stands in for vLLM's ``CuMemAllocator``, so the backend runs without a GPU.
"""

import pytest


class _FakeAllocator:
    def __init__(self):
        self.calls = []

    def sleep(self, offload_tags=None):
        self.calls.append(("sleep", offload_tags))

    def discard(self, tags):
        self.calls.append(("discard", tags))

    def wake_up(self, tags=None):
        self.calls.append(("wake_up", tags))


@pytest.mark.vllm
class TestSleepKeepWeightsPatch:
    @pytest.fixture(autouse=True)
    def registry(self, monkeypatch):
        pytest.importorskip("vllm")
        from vllm.device_allocator import sleep_mode_backend

        monkeypatch.setattr(
            sleep_mode_backend.SleepModeBackendFactory,
            "_registry",
            dict(sleep_mode_backend.SleepModeBackendFactory._registry),
        )
        allocator = _FakeAllocator()
        monkeypatch.setattr("vllm.device_allocator.get_mem_allocator_instance", lambda: allocator)
        self.allocator = allocator
        self.factory = sleep_mode_backend.SleepModeBackendFactory
        self.default_cls = sleep_mode_backend.CuMemBackend

    def test_noop_without_env(self, monkeypatch):
        from skyrl.backends.skyrl_train.patches.vllm.patch_sleep_keep_weights import (
            KEEP_WEIGHTS_ENV,
            apply_sleep_keep_weights_patch,
        )

        monkeypatch.delenv(KEEP_WEIGHTS_ENV, raising=False)
        apply_sleep_keep_weights_patch()
        assert self.factory.get_backend_class("cumem") is self.default_cls

    @pytest.mark.parametrize("level", [1, 2])
    def test_suspend_discards_only_kv_cache(self, monkeypatch, level):
        from skyrl.backends.skyrl_train.patches.vllm.patch_sleep_keep_weights import (
            KEEP_WEIGHTS_ENV,
            apply_sleep_keep_weights_patch,
        )

        monkeypatch.setenv(KEEP_WEIGHTS_ENV, "1")
        apply_sleep_keep_weights_patch()
        backend_cls = self.factory.get_backend_class("cumem")
        assert issubclass(backend_cls, self.default_cls) and backend_cls is not self.default_cls

        backend = backend_cls()
        backend.suspend(level)
        assert self.allocator.calls == [("discard", ("kv_cache",))]
        assert backend.state() == "SUSPENDED"

        backend.resume(["kv_cache"])
        assert self.allocator.calls[-1] == ("wake_up", ["kv_cache"])
        assert backend.state() == "RUNNING"
