"""
CPU tests for pretokenized SFT dataset ingestion.

uv run --extra dev pytest tests/train/test_sft_pretokenized.py -v
"""

import os
import shutil
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from skyrl.train.config.sft_config import SFTConfig, validate_sft_cfg
from skyrl.train.dataset.pretokenized import load_from_pretokenized
from skyrl.train.sft_trainer import SFTTrainer, collate_sft_batch


def _rows():
    """Rows in the pretokenized input schema: input_ids + full-sequence loss_mask."""
    return [
        # Instruction-following style: loss on the trailing response tokens.
        {"input_ids": [1, 2, 3, 4, 5], "loss_mask": [0, 0, 0, 1, 1]},
        # Single-token response.
        {"input_ids": [6, 7, 8], "loss_mask": [0, 0, 1]},
    ]


def _vlm_rows():
    """VLM rows: image tensors stored as nested lists (parquet-compatible)."""
    return [
        {
            "input_ids": [1, 2, 3, 4],
            "loss_mask": [0, 0, 1, 1],
            "pixel_values": [[0.1] * 8] * 4,
            "image_grid_thw": [[1, 2, 2]],
        },
        {
            "input_ids": [5, 6, 7],
            "loss_mask": [0, 1, 1],
            "pixel_values": [[0.2] * 8] * 4,
            "image_grid_thw": [[1, 2, 2]],
        },
    ]


def _assert_normalized(rows):
    for row in rows:
        assert row["attention_mask"] == [1] * len(row["input_ids"])
        assert 0 < row["num_actions"] <= len(row["input_ids"])
        assert len(row["loss_mask"]) == row["num_actions"]


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def test_load_parquet_file(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(_rows()).to_parquet(path)

    rows = load_from_pretokenized(path)
    assert len(rows) == 2
    assert rows[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert rows[0]["num_actions"] == 2
    assert rows[0]["loss_mask"] == [1, 1]
    _assert_normalized(rows)


def test_load_jsonl_file(tmp_path):
    path = str(tmp_path / "data.jsonl")
    Dataset.from_list(_rows()).to_json(path)

    rows = load_from_pretokenized(path)
    assert len(rows) == 2
    _assert_normalized(rows)


def test_load_save_to_disk_dir(tmp_path):
    path = str(tmp_path / "hf_dataset")
    Dataset.from_list(_rows()).save_to_disk(path)

    rows = load_from_pretokenized(path)
    assert len(rows) == 2
    _assert_normalized(rows)


def test_load_arrow_file(tmp_path):
    saved = tmp_path / "hf_dataset"
    Dataset.from_list(_rows()).save_to_disk(str(saved))
    arrow_files = [f for f in os.listdir(saved) if f.endswith(".arrow")]
    assert arrow_files
    path = str(tmp_path / "data.arrow")
    shutil.copy(saved / arrow_files[0], path)

    rows = load_from_pretokenized(path)
    assert len(rows) == 2
    _assert_normalized(rows)


def test_load_directory_of_parquet_shards(tmp_path):
    data_dir = tmp_path / "shards"
    data_dir.mkdir()
    Dataset.from_list(_rows()).to_parquet(str(data_dir / "shard-00000.parquet"))
    Dataset.from_list(_rows()).to_parquet(str(data_dir / "shard-00001.parquet"))

    rows = load_from_pretokenized(str(data_dir))
    assert len(rows) == 4


def test_load_directory_of_jsonl_shards(tmp_path):
    data_dir = tmp_path / "shards"
    data_dir.mkdir()
    Dataset.from_list(_rows()).to_json(str(data_dir / "shard-00000.jsonl"))
    Dataset.from_list(_rows()).to_json(str(data_dir / "shard-00001.jsonl"))

    rows = load_from_pretokenized(str(data_dir))
    assert len(rows) == 4
    _assert_normalized(rows)


def test_load_directory_of_arrow_shards(tmp_path):
    saved = tmp_path / "hf_dataset"
    Dataset.from_list(_rows()).save_to_disk(str(saved))
    arrow_files = [f for f in os.listdir(saved) if f.endswith(".arrow")]
    assert arrow_files

    data_dir = tmp_path / "shards"
    data_dir.mkdir()
    shutil.copy(saved / arrow_files[0], data_dir / "shard-00000.arrow")
    shutil.copy(saved / arrow_files[0], data_dir / "shard-00001.arrow")

    rows = load_from_pretokenized(str(data_dir))
    assert len(rows) == 4
    _assert_normalized(rows)


def test_hidden_files_and_dirs_skipped(tmp_path):
    data_dir = tmp_path / "shards"
    data_dir.mkdir()
    Dataset.from_list(_rows()).to_parquet(str(data_dir / "shard-00000.parquet"))
    # Stale Jupyter checkpoint copy (would silently duplicate rows) and a
    # macOS AppleDouble sidecar (not valid parquet, would crash the load).
    checkpoints = data_dir / ".ipynb_checkpoints"
    checkpoints.mkdir()
    Dataset.from_list(_rows()).to_parquet(str(checkpoints / "shard-00000.parquet"))
    (data_dir / "._shard-00000.parquet").write_bytes(b"\x00\x05\x16\x07not parquet")

    rows = load_from_pretokenized(str(data_dir))
    assert len(rows) == 2


def test_mixed_formats_in_directory_raises(tmp_path):
    data_dir = tmp_path / "mixed"
    data_dir.mkdir()
    Dataset.from_list(_rows()).to_parquet(str(data_dir / "a.parquet"))
    Dataset.from_list(_rows()).to_json(str(data_dir / "b.jsonl"))

    with pytest.raises(ValueError, match="mix of data formats"):
        load_from_pretokenized(str(data_dir))


def test_empty_directory_raises(tmp_path):
    with pytest.raises(ValueError, match="No supported data files"):
        load_from_pretokenized(str(tmp_path))


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_from_pretokenized(str(tmp_path / "nope.parquet"))


# ---------------------------------------------------------------------------
# Schema validation / normalization
# ---------------------------------------------------------------------------


def test_multi_turn_loss_mask_preserves_interior_zeros(tmp_path):
    path = str(tmp_path / "data.parquet")
    # Conversational data: assistant turns at positions 2 and 4, user turn between.
    Dataset.from_list([{"input_ids": [1, 2, 3, 4, 5], "loss_mask": [0, 0, 1, 0, 1]}]).to_parquet(path)

    rows = load_from_pretokenized(path)
    assert rows[0]["num_actions"] == 3
    assert rows[0]["loss_mask"] == [1, 0, 1]


def test_all_zero_loss_mask_row_dropped(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(
        [
            {"input_ids": [1, 2, 3], "loss_mask": [0, 0, 0]},
            {"input_ids": [1, 2, 3], "loss_mask": [0, 0, 1]},
        ]
    ).to_parquet(path)

    rows = load_from_pretokenized(path)
    assert len(rows) == 1


def test_all_rows_dropped_raises(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3], "loss_mask": [0, 0, 0]}]).to_parquet(path)

    with pytest.raises(ValueError, match="0 usable examples"):
        load_from_pretokenized(path)


def test_missing_loss_mask_raises(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3]}]).to_parquet(path)

    with pytest.raises(ValueError, match="missing required 'loss_mask'"):
        load_from_pretokenized(path)


def test_labels_column_rejected(tmp_path):
    path = str(tmp_path / "data.parquet")
    # HF-style labels are not a supported loss target; loss_mask is required.
    Dataset.from_list([{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}]).to_parquet(path)

    with pytest.raises(ValueError, match="missing required 'loss_mask'"):
        load_from_pretokenized(path)


def test_num_actions_column_rejected(tmp_path):
    path = str(tmp_path / "data.parquet")
    # num_actions alone is not a loss target; the full-sequence loss_mask is required.
    Dataset.from_list([{"input_ids": [1, 2, 3], "num_actions": 2}]).to_parquet(path)

    with pytest.raises(ValueError, match="missing required 'loss_mask'"):
        load_from_pretokenized(path)


def test_window_form_loss_mask_rejected(tmp_path):
    path = str(tmp_path / "data.parquet")
    # Window-form mask (len == 2 != len(input_ids) == 4) is no longer accepted.
    Dataset.from_list([{"input_ids": [1, 2, 3, 4], "loss_mask": [1, 1]}]).to_parquet(path)

    with pytest.raises(ValueError, match="full-sequence"):
        load_from_pretokenized(path)


def test_non_binary_loss_mask_rejected(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3], "loss_mask": [0, 2, 1]}]).to_parquet(path)

    with pytest.raises(ValueError, match="only 0s and 1s"):
        load_from_pretokenized(path)


def test_redundant_num_actions_dropped_when_loss_mask_present(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3, 4], "loss_mask": [0, 0, 1, 1], "num_actions": 3}]).to_parquet(path)

    rows = load_from_pretokenized(path)
    # num_actions is always re-inferred from the mask, never read from the store.
    assert rows[0]["num_actions"] == 2


def test_missing_input_ids_raises(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"loss_mask": [1], "other": "x"}]).to_parquet(path)

    with pytest.raises(ValueError, match="input_ids"):
        load_from_pretokenized(path)


def test_padded_attention_mask_raises(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(
        [{"input_ids": [0, 0, 1, 2], "attention_mask": [0, 0, 1, 1], "loss_mask": [0, 0, 0, 1]}]
    ).to_parquet(path)

    with pytest.raises(ValueError, match="unpadded"):
        load_from_pretokenized(path)


# ---------------------------------------------------------------------------
# max_length truncation
# ---------------------------------------------------------------------------


def test_max_length_truncates_action_window(tmp_path):
    path = str(tmp_path / "data.parquet")
    # 3 prompt tokens + 4 response tokens; max_length=5 keeps 2 response tokens.
    Dataset.from_list([{"input_ids": [1, 2, 3, 4, 5, 6, 7], "loss_mask": [0, 0, 0, 1, 1, 1, 1]}]).to_parquet(path)

    rows = load_from_pretokenized(path, max_length=5)
    assert rows[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert rows[0]["num_actions"] == 2
    assert rows[0]["loss_mask"] == [1, 1]


def test_max_length_drops_fully_truncated_response(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(
        [
            # 4 prompt tokens; truncation to 3 removes the whole response.
            {"input_ids": [1, 2, 3, 4, 5], "loss_mask": [0, 0, 0, 0, 1]},
            {"input_ids": [1, 2], "loss_mask": [0, 1]},
        ]
    ).to_parquet(path)

    rows = load_from_pretokenized(path, max_length=3)
    assert len(rows) == 1
    assert rows[0]["input_ids"] == [1, 2]


# ---------------------------------------------------------------------------
# VLM rows
# ---------------------------------------------------------------------------


def test_vlm_rows_pass_through(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(_vlm_rows()).to_parquet(path)

    rows = load_from_pretokenized(path)
    assert len(rows) == 2
    _assert_normalized(rows)
    assert rows[0]["pixel_values"] == [[0.1] * 8] * 4
    assert rows[0]["image_grid_thw"] == [[1, 2, 2]]


def test_vlm_rows_collate_to_tensor_lists(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(_vlm_rows()).to_parquet(path)
    rows = load_from_pretokenized(path)

    batch = collate_sft_batch(rows, SimpleNamespace(pad_token_id=0))
    assert batch["sequences"].shape == (2, 4)
    assert len(batch["pixel_values"]) == 2
    assert torch.as_tensor(rows[0]["pixel_values"]).shape == batch["pixel_values"][0].shape
    assert batch["image_grid_thw"][0].tolist() == [[1, 2, 2]]


def test_vlm_row_over_max_length_dropped_not_truncated(tmp_path):
    path = str(tmp_path / "data.parquet")
    long_vlm = {
        "input_ids": [1, 2, 3, 4, 5, 6],
        "loss_mask": [0, 0, 0, 1, 1, 1],
        "pixel_values": [[0.1] * 8] * 4,
        "image_grid_thw": [[1, 2, 2]],
    }
    Dataset.from_list([long_vlm] + _vlm_rows()).to_parquet(path)

    rows = load_from_pretokenized(path, max_length=4)
    # The long VLM row is dropped (never truncated); the short ones survive intact.
    assert len(rows) == 2
    assert all(len(r["input_ids"]) <= 4 for r in rows)


def test_vlm_row_missing_grid_raises(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list([{"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "pixel_values": [[0.1] * 8] * 4}]).to_parquet(
        path
    )

    with pytest.raises(ValueError, match="both"):
        load_from_pretokenized(path)


def test_mixed_text_and_vlm_store_drops_null_image_columns(tmp_path):
    path = str(tmp_path / "data.parquet")
    # A store mixing text and VLM rows materializes the image columns as None
    # on text rows; those keys must not leak into the normalized text rows.
    Dataset.from_list(_vlm_rows() + [{"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]}]).to_parquet(path)

    rows = load_from_pretokenized(path)
    assert len(rows) == 3
    text_rows = [r for r in rows if "pixel_values" not in r]
    assert len(text_rows) == 1


# ---------------------------------------------------------------------------
# Trainer / config integration
# ---------------------------------------------------------------------------


def test_trainer_load_dataset_routes_to_pretokenized(tmp_path):
    train_path = str(tmp_path / "train.parquet")
    eval_path = str(tmp_path / "eval.parquet")
    Dataset.from_list(_rows()).to_parquet(train_path)
    Dataset.from_list(_rows()[:1]).to_parquet(eval_path)

    cfg = SFTConfig(
        pretokenized_dataset_paths=[train_path],
        eval_pretokenized_dataset_paths=[eval_path],
        enable_ray_gpu_monitor=False,
        disable_cache=True,
    )
    validate_sft_cfg(cfg)
    trainer = SFTTrainer(cfg)

    # No tokenizer / workers needed: the pretokenized path never tokenizes.
    tokenized, dataset_lengths = trainer.load_dataset()
    assert len(tokenized) == 2
    assert dataset_lengths == [2]
    eval_sets = trainer.load_eval_datasets()
    assert len(eval_sets) == 1
    eval_name, eval_tokenized = eval_sets[0]
    assert eval_name == "eval.parquet"
    assert len(eval_tokenized) == 1


def test_trainer_concatenates_multiple_pretokenized_stores(tmp_path):
    path_a = str(tmp_path / "store_a.parquet")
    path_b = str(tmp_path / "store_b.parquet")
    Dataset.from_list(_rows()).to_parquet(path_a)
    Dataset.from_list(_rows()[:1]).to_parquet(path_b)

    cfg = SFTConfig(
        pretokenized_dataset_paths=[path_a, path_b],
        enable_ray_gpu_monitor=False,
        disable_cache=True,
    )
    validate_sft_cfg(cfg)
    # Equal mixing weights are defaulted for the random sampler.
    assert cfg.train_dataset_weights == [0.5, 0.5]

    trainer = SFTTrainer(cfg)
    tokenized, dataset_lengths = trainer.load_dataset()
    assert len(tokenized) == 3
    assert dataset_lengths == [2, 1]

    # Multiple sources -> DataMixingSampler configured from the store lengths.
    sampler = trainer.build_train_sampler(tokenized, dataset_lengths)
    assert sampler is not None
    assert len(list(iter(sampler))) > 0


def test_multiple_pretokenized_eval_stores_named_by_basename(tmp_path):
    path_a = str(tmp_path / "eval_a.parquet")
    path_b = str(tmp_path / "eval_b.parquet")
    Dataset.from_list(_rows()).to_parquet(path_a)
    Dataset.from_list(_rows()[:1]).to_parquet(path_b)

    cfg = SFTConfig(
        pretokenized_dataset_paths=[path_a],
        eval_pretokenized_dataset_paths=[path_a, path_b],
        enable_ray_gpu_monitor=False,
        disable_cache=True,
    )
    validate_sft_cfg(cfg)
    assert cfg.eval_dataset_names == ["eval_a.parquet", "eval_b.parquet"]

    trainer = SFTTrainer(cfg)
    eval_sets = trainer.load_eval_datasets()
    assert [(name, len(rows)) for name, rows in eval_sets] == [("eval_a.parquet", 2), ("eval_b.parquet", 1)]


def test_validate_cfg_rejects_pretokenized_with_train_datasets():
    cfg = SFTConfig(
        pretokenized_dataset_paths=["/data/train"],
        train_datasets=["yahma/alpaca-cleaned"],
        train_dataset_splits=["train[:100]"],
    )
    with pytest.raises(ValueError, match="only one of pretokenized_dataset_paths"):
        validate_sft_cfg(cfg)


def test_validate_cfg_rejects_colliding_default_eval_names():
    cfg = SFTConfig(
        pretokenized_dataset_paths=["/data/train"],
        eval_pretokenized_dataset_paths=["/data/a/eval", "/data/b/eval"],
    )
    with pytest.raises(ValueError, match="collide"):
        validate_sft_cfg(cfg)

    # Explicit names disambiguate.
    cfg = SFTConfig(
        pretokenized_dataset_paths=["/data/train"],
        eval_pretokenized_dataset_paths=["/data/a/eval", "/data/b/eval"],
        eval_dataset_names=["eval_a", "eval_b"],
    )
    validate_sft_cfg(cfg)


def test_validate_cfg_weights_validated_against_pretokenized_paths():
    cfg = SFTConfig(
        pretokenized_dataset_paths=["/data/a", "/data/b"],
        train_dataset_weights=[0.9],
    )
    with pytest.raises(ValueError, match="one weight per entry of pretokenized_dataset_paths"):
        validate_sft_cfg(cfg)

    cfg = SFTConfig(
        pretokenized_dataset_paths=["/data/a", "/data/b"],
        train_dataset_weights=[0.9, 0.1],
    )
    validate_sft_cfg(cfg)
    assert cfg.train_dataset_weights == [0.9, 0.1]


def test_pretokenized_rows_collate(tmp_path):
    path = str(tmp_path / "data.parquet")
    Dataset.from_list(_rows()).to_parquet(path)
    rows = load_from_pretokenized(path)

    batch = collate_sft_batch(rows, SimpleNamespace(pad_token_id=0))
    assert batch["sequences"].shape == (2, 5)
    assert batch["loss_mask"].shape == (2, 2)
    assert batch.metadata["response_length"] == 2


def test_validate_cfg_accepts_pretokenized_eval_only():
    cfg = SFTConfig(
        pretokenized_dataset_paths=["/data/train"],
        eval_pretokenized_dataset_paths=["/data/eval"],
        eval_interval=10,
    )
    validate_sft_cfg(cfg)


def test_validate_cfg_eval_interval_requires_some_eval_dataset():
    cfg = SFTConfig(eval_interval=10)
    with pytest.raises(ValueError, match="eval_interval"):
        validate_sft_cfg(cfg)


# ---------------------------------------------------------------------------
# RowGroupPretokenizedDataset (cloud row-group fetch, tested over local files)
# ---------------------------------------------------------------------------


def _write_rowgroup_shard(path, rows, row_group_size=4):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "input_ids": [r["input_ids"] for r in rows],
            "loss_mask": [r["loss_mask"] for r in rows],
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)


def _rowgroup_store(tmp_path, num_shards=3, rows_per_shard=25):
    """Multi-shard store with small row groups; row i's ids start with i."""
    store = tmp_path / "rg_store"
    store.mkdir()
    idx = 0
    for shard in range(num_shards):
        rows = []
        for _ in range(rows_per_shard):
            seq_len = 4 + (idx % 5)
            rows.append(
                {
                    "input_ids": [idx] * seq_len,
                    "loss_mask": [0] * (seq_len // 2) + [1] * (seq_len - seq_len // 2),
                }
            )
            idx += 1
        _write_rowgroup_shard(str(store / f"shard-{shard:05d}.parquet"), rows)
    return str(store), idx


def test_rowgroup_parity_with_mmap_loader(tmp_path):
    from skyrl.train.dataset.pretokenized import RowGroupPretokenizedDataset

    store, total = _rowgroup_store(tmp_path)
    mmap_ds = load_from_pretokenized(store, max_length=7)
    rg_ds = RowGroupPretokenizedDataset(store, max_length=7)

    assert len(rg_ds) == len(mmap_ds) == total
    for i in range(total):
        assert rg_ds[i] == mmap_ds[i], f"row {i} mismatch"


def test_rowgroup_getitems_preserves_requested_order(tmp_path):
    from skyrl.train.dataset.pretokenized import RowGroupPretokenizedDataset

    store, total = _rowgroup_store(tmp_path)
    rg_ds = RowGroupPretokenizedDataset(store)
    indices = [50, 3, 74, 3, 26, 0]
    rows = rg_ds.__getitems__(indices)
    # Row identity is recoverable from the ids payload (row i's ids are all i).
    assert [r["input_ids"][0] for r in rows] == indices


def test_rowgroup_cache_avoids_refetch(tmp_path, monkeypatch):
    from skyrl.train.dataset.pretokenized import RowGroupPretokenizedDataset

    store, _ = _rowgroup_store(tmp_path)
    rg_ds = RowGroupPretokenizedDataset(store)
    reads = {"n": 0}
    original = rg_ds._parquet_file

    def counting(shard):
        pf = original(shard)

        class _Wrap:
            def read_row_group(self, rg, **kw):
                reads["n"] += 1
                return pf.read_row_group(rg, **kw)

        return _Wrap()

    monkeypatch.setattr(rg_ds, "_parquet_file", counting)
    rg_ds.__getitems__([0, 1, 2, 3])  # same row group (row_group_size=4)
    assert reads["n"] == 1
    rg_ds.__getitems__([0, 1, 2, 3])  # served from LRU cache
    assert reads["n"] == 1


def test_rowgroup_missing_loss_mask_raises_at_construction(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from skyrl.train.dataset.pretokenized import RowGroupPretokenizedDataset

    store = tmp_path / "bad_store"
    store.mkdir()
    pq.write_table(pa.table({"input_ids": [[1, 2, 3]]}), str(store / "shard.parquet"))
    with pytest.raises(ValueError, match="missing required 'loss_mask'"):
        RowGroupPretokenizedDataset(str(store))


def test_rowgroup_empty_loss_window_raises_at_access(tmp_path):
    from skyrl.train.dataset.pretokenized import RowGroupPretokenizedDataset

    store = tmp_path / "unfiltered"
    store.mkdir()
    _write_rowgroup_shard(
        str(store / "shard.parquet"),
        [{"input_ids": [1, 2, 3], "loss_mask": [0, 0, 0]}],
    )
    rg_ds = RowGroupPretokenizedDataset(str(store))
    with pytest.raises(ValueError, match="pre-filtered"):
        rg_ds[0]


def test_rowgroup_cloud_uri_routing(tmp_path, monkeypatch):
    import fsspec

    from skyrl.train.dataset import pretokenized as ptk

    store, total = _rowgroup_store(tmp_path)
    real_url_to_fs = fsspec.core.url_to_fs

    def fake_url_to_fs(url, **kw):
        assert url.startswith("s3://")
        return real_url_to_fs(store)

    monkeypatch.setattr(fsspec.core, "url_to_fs", fake_url_to_fs)
    ds = load_from_pretokenized("s3://bucket/prefix/train")
    assert isinstance(ds, ptk.RowGroupPretokenizedDataset)
    assert len(ds) == total


def test_rowgroup_dataloader_resume_and_spawn_workers(tmp_path):
    from torchdata.stateful_dataloader import StatefulDataLoader

    from skyrl.train.dataset.pretokenized import RowGroupPretokenizedDataset

    store, total = _rowgroup_store(tmp_path)
    rg_ds = RowGroupPretokenizedDataset(store)

    dl = StatefulDataLoader(rg_ds, batch_size=8, shuffle=True, collate_fn=list, num_workers=0)
    it = iter(dl)
    for _ in range(3):
        next(it)
    state = dl.state_dict()
    expected = next(it)
    dl2 = StatefulDataLoader(rg_ds, batch_size=8, shuffle=True, collate_fn=list, num_workers=0)
    dl2.load_state_dict(state)
    resumed = next(iter(dl2))
    assert [r["input_ids"] for r in expected] == [r["input_ids"] for r in resumed]

    spawn_dl = StatefulDataLoader(
        rg_ds,
        batch_size=8,
        sampler=list(range(32)),
        collate_fn=list,
        num_workers=2,
        multiprocessing_context="spawn",
    )
    assert sum(len(b) for b in spawn_dl) == 32
