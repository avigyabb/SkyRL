"""
uv run --isolated --extra dev pytest tests/backends/skyrl_train/workers/megatron/test_hf_import_cache.py

Pure-core tests for the HF -> Megatron import cache: keying, addressing and the
hit/miss decision. Saving and loading run inside Megatron workers and are covered
by GPU runs.
"""

import json
import os

from skyrl.backends.skyrl_train.workers.megatron import hf_import_cache as hic


def _factors(model_path: str, **overrides):
    base = dict(
        model_path=model_path,
        bf16=True,
        language_model_only=False,
        enable_mtp=False,
        model_config_kwargs={},
        transformer_config_kwargs={"recompute_granularity": None},
    )
    base.update(overrides)
    return hic.import_factors(**base)


def test_fingerprint_tracks_weight_and_index_files_only(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"x" * 10)
    (ckpt / "config.json").write_text("{}")
    (ckpt / "README.md").write_text("ignored")

    fp = hic.checkpoint_fingerprint(str(ckpt))
    assert [entry[0] for entry in fp["files"]] == ["config.json", "model.safetensors"]

    before = hic.digest(_factors(str(ckpt)))
    (ckpt / "model.safetensors").write_bytes(b"y" * 11)
    assert hic.digest(_factors(str(ckpt))) != before

    assert hic.checkpoint_fingerprint("Qwen/Qwen2.5-0.5B-Instruct") == {"id": "Qwen/Qwen2.5-0.5B-Instruct"}


def test_digest_changes_with_shape_affecting_knobs_and_is_stable():
    a = hic.digest(_factors("Qwen/Qwen2.5-0.5B-Instruct"))
    assert a == hic.digest(_factors("Qwen/Qwen2.5-0.5B-Instruct"))
    assert a != hic.digest(_factors("Qwen/Qwen2.5-0.5B-Instruct", enable_mtp=True))
    assert a != hic.digest(_factors("Qwen/Qwen2.5-0.5B-Instruct", model_config_kwargs={"num_layers": 2}))
    assert a != hic.digest(_factors("Qwen/Qwen2.5-0.5B-Instruct", bf16=False))


def test_cache_path_uses_model_slug_and_digest(tmp_path):
    factors = _factors("Qwen/Qwen2.5-0.5B-Instruct")
    path = hic.cache_path(str(tmp_path), "Qwen/Qwen2.5-0.5B-Instruct", factors)
    assert path == os.path.join(str(tmp_path), "qwen__qwen2.5-0.5b-instruct", hic.digest(factors))

    local = hic.cache_path(str(tmp_path), "/mnt/models/My Model/", factors)
    assert os.path.basename(os.path.dirname(local)) == "my-model"


def test_resolve_hits_only_on_complete_marker(tmp_path):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    factors = _factors(model)
    miss = hic.resolve(str(tmp_path), model, factors)
    assert not miss.hit

    os.makedirs(miss.path)
    (tmp_path / "unrelated").mkdir()
    assert not hic.resolve(str(tmp_path), model, factors).hit  # directory without marker: a partial write

    with open(os.path.join(miss.path, hic.COMPLETE_MARKER), "w") as f:
        json.dump({"digest": hic.digest(factors)}, f)
    hit = hic.resolve(str(tmp_path), model, factors)
    assert hit.hit and hit.path == miss.path
