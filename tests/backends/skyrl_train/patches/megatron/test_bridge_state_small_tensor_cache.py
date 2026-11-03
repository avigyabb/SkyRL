"""``patch_bridge_state_small_tensor_cache.py``: the backport of Megatron-Bridge #6456.

The tests are upstream's ``TestSafeTensorsStateSourceCache`` (tests/unit_tests/models/hf_pretrained/
test_state_dict.py), run against the pinned class with the patch applied. The only difference is
that the backport creates the cache on first use instead of in ``__init__``.
"""

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("megatron.bridge")

from megatron.bridge.models.hf_pretrained.state import (  # noqa: E402
    SafeTensorsStateSource,
)

from skyrl.backends.skyrl_train.patches.megatron.patch_bridge_state_small_tensor_cache import (  # noqa: E402
    apply_bridge_state_small_tensor_cache_patch,
)


@pytest.fixture(autouse=True)
def _patched():
    apply_bridge_state_small_tensor_cache_patch()
    assert hasattr(SafeTensorsStateSource, "_get_handle") or SafeTensorsStateSource.load_tensors.__module__.endswith(
        "patch_bridge_state_small_tensor_cache"
    )


def _write_checkpoint(root: Path, tensors: dict, filename: str = "model-00001-of-00001.safetensors") -> None:
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, root / filename)
    index = {"metadata": {}, "weight_map": {k: filename for k in tensors}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))


def _count_safe_open(monkeypatch):
    import safetensors

    calls = {"n": 0}
    real = safetensors.safe_open

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", counting)
    return calls


def test_small_cache_returns_independent_tensors(tmp_path):
    _write_checkpoint(tmp_path, {"w.scale": torch.tensor([1.0]), "w.weight": torch.ones(4, 4)})
    source = SafeTensorsStateSource(tmp_path)

    first = source.load_tensors(["w.scale"])["w.scale"]
    first.zero_()
    second = source.load_tensors(["w.scale"])["w.scale"]
    assert second.item() == 1.0
    second.fill_(7.0)
    assert source.load_tensors(["w.scale"])["w.scale"].item() == 1.0


def test_retry_refreshes_incomplete_shard(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    shard = "model-00001-of-00001.safetensors"
    _write_checkpoint(tmp_path, {"text.other": torch.zeros(2)}, shard)
    index = {"metadata": {}, "weight_map": {"text.other": shard, "text.weight": shard}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    def replace_shard(_seconds):
        complete = tmp_path / "complete.safetensors"
        save_file({"text.other": torch.zeros(2), "text.weight": torch.ones(2)}, complete)
        complete.replace(tmp_path / shard)

    monkeypatch.setattr("time.sleep", replace_shard)
    source = SafeTensorsStateSource(tmp_path, key_prefix="text.")
    loaded = source.load_tensors(["weight"])
    assert torch.equal(loaded["weight"], torch.ones(2))


@pytest.mark.parametrize("cache_small,reuse_handles", [("1", "1"), ("1", "0"), ("0", "1"), ("0", "0")])
def test_cache_and_handle_switches(tmp_path, monkeypatch, cache_small, reuse_handles):
    _write_checkpoint(tmp_path, {"w.scale": torch.tensor([2.0]), "w.weight": torch.ones(8, 8)})
    monkeypatch.setenv("MBRIDGE_STATE_CACHE_SMALL_TENSORS", cache_small)
    monkeypatch.setenv("MBRIDGE_STATE_REUSE_HANDLES", reuse_handles)
    calls = _count_safe_open(monkeypatch)
    source = SafeTensorsStateSource(tmp_path)

    for _ in range(3):
        assert source.load_tensors(["w.scale"])["w.scale"].item() == 2.0
    opens_after_scales = calls["n"]
    if cache_small == "1":
        assert opens_after_scales == 1  # first read opens the shard, the two cache hits do not
    elif reuse_handles == "1":
        assert opens_after_scales == 1  # one handle, reused for all three reads
    else:
        assert opens_after_scales == 3  # baseline behaviour: one safe_open per load_tensors

    for _ in range(2):
        assert source.load_tensors(["w.weight"]).keys() == {"w.weight"}
    extra_opens = calls["n"] - opens_after_scales
    expected = {("1", "1"): 0, ("1", "0"): 1, ("0", "1"): 0, ("0", "0"): 2}
    assert extra_opens == expected[(cache_small, reuse_handles)]


def test_cache_threshold_and_instance_isolation(tmp_path, monkeypatch):
    _write_checkpoint(tmp_path, {"small": torch.ones(4), "big": torch.ones(16)})
    monkeypatch.setenv("MBRIDGE_STATE_SMALL_TENSOR_NUMEL", "4")
    monkeypatch.setenv("MBRIDGE_STATE_REUSE_HANDLES", "0")
    calls = _count_safe_open(monkeypatch)

    source = SafeTensorsStateSource(tmp_path)
    source.load_tensors(["small", "big"])
    assert set(source._small_tensor_cache) == {"small"}  # inclusive threshold: 4 <= 4 cached, 16 is not
    source.load_tensors(["small"])
    assert calls["n"] == 1  # served from the cache
    source.load_tensors(["big"])
    assert calls["n"] == 2  # re-read from the shard

    other = SafeTensorsStateSource(tmp_path)
    assert other.__dict__.get("_small_tensor_cache", {}) == {}
    other.load_tensors(["small"])
    assert calls["n"] == 3  # a new instance starts cold
