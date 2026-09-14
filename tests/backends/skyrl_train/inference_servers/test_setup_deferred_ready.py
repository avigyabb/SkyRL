"""
uv run --isolated --extra dev pytest tests/backends/skyrl_train/inference_servers/test_setup_deferred_ready.py

Covers the ``wait_for_ready=False`` path of ``create_inference_servers``: the
servers are launched and addressed without waiting for health, the router is
constructed but not started, and ``InferenceServerSetup.wait_until_ready``
resolves both.
"""

from unittest.mock import MagicMock, patch

import pytest

from skyrl.backends.skyrl_train.inference_servers import setup as setup_mod
from skyrl.backends.skyrl_train.inference_servers.common import ServerInfo
from skyrl.backends.skyrl_train.inference_servers.setup import (
    InferenceServerSetup,
    _launch_server_groups,
    _make_router,
)


class FakeRouter:
    def __init__(self):
        self.is_started = False
        self.url = "http://10.0.0.1:30000"

    def start(self):
        self.is_started = True
        return self.url


class FakeGroup:
    def __init__(self, urls):
        self._infos = [ServerInfo(ip=u.split(":")[0], port=int(u.split(":")[1])) for u in urls]
        self.start_refs = [f"ref-{u}" for u in urls]
        self.start_calls = []

    def start(self, blocking=True):
        self.start_calls.append(blocking)
        return self.start_refs

    @property
    def server_infos(self):
        return self._infos

    def get_server_infos_nowait(self):
        return self._infos


@pytest.fixture
def ray_get():
    with patch.object(setup_mod.ray, "get") as m:
        yield m


def test_launch_server_groups_defers_health_wait(ray_get):
    groups = [FakeGroup(["10.0.0.1:8000"]), FakeGroup(["10.0.0.2:8000", "10.0.0.2:8001"])]

    pending, infos = _launch_server_groups(groups, wait_for_ready=False)

    ray_get.assert_not_called()
    assert pending == groups[0].start_refs + groups[1].start_refs
    assert [[i.url for i in g] for g in infos] == [
        ["http://10.0.0.1:8000"],
        ["http://10.0.0.2:8000", "http://10.0.0.2:8001"],
    ]
    assert all(g.start_calls == [False] for g in groups)


def test_launch_server_groups_waits_when_requested(ray_get):
    groups = [FakeGroup(["10.0.0.1:8000"])]

    pending, infos = _launch_server_groups(groups, wait_for_ready=True)

    ray_get.assert_called_once_with(groups[0].start_refs)
    assert pending == []
    assert [i.url for i in infos[0]] == ["http://10.0.0.1:8000"]


def test_make_router_only_starts_when_ready():
    with patch.object(setup_mod, "VLLMRouter", side_effect=lambda *a, **k: FakeRouter()):
        router, url = _make_router(MagicMock(), log_path="/tmp", wait_for_ready=False)
        assert not router.is_started
        assert url == router.url

        router, url = _make_router(MagicMock(), log_path="/tmp", wait_for_ready=True)
        assert router.is_started
        assert url == router.url


def test_wait_until_ready_resolves_refs_then_starts_router(ray_get):
    router = FakeRouter()
    setup = InferenceServerSetup(
        proxy_url=router.url,
        server_urls=["http://10.0.0.1:8000"],
        router=router,
        pending_start_refs=["ref-a", "ref-b"],
    )

    setup.wait_until_ready()

    ray_get.assert_called_once_with(["ref-a", "ref-b"])
    assert setup.pending_start_refs == []
    assert router.is_started

    # Second call is a no-op.
    setup.wait_until_ready()
    ray_get.assert_called_once()


def test_wait_until_ready_noop_for_external_or_awaited_setup(ray_get):
    started = FakeRouter()
    started.start()
    for setup in (
        InferenceServerSetup(proxy_url="http://proxy", server_urls=["http://s"]),
        InferenceServerSetup(proxy_url=started.url, server_urls=["http://s"], router=started),
    ):
        setup.wait_until_ready()
    ray_get.assert_not_called()
