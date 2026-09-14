"""Cache the HF -> Megatron weight import as a Megatron distributed checkpoint.

Megatron-Bridge's ``load_weights_hf_to_megatron`` has every rank read the full HF
tensors and convert them on CPU at every startup. The converted weights are the
same for every run of a given checkpoint + model config, so the first run saves the
built model's ``sharded_state_dict`` under ``hf_import_cache_dir`` and later runs
load that instead. Megatron's ``torch_dist`` format is parallelism-agnostic, so a
cache written at one TP/PP layout loads into another.

Layout: ``<root>/<model-slug>/<digest>/`` with a ``.skyrl_hf_import_complete``
marker holding the factor map that produced the digest. The digest covers the
checkpoint fingerprint, the library versions that define the conversion, and the
config knobs that change parameter shapes. Writers save into a sibling temp
directory and rename it into place, so concurrent jobs never see a partial cache.
"""

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

COMPLETE_MARKER = ".skyrl_hf_import_complete"
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_META_FILES = ("config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json")
_VERSIONED_PACKAGES = ("megatron-core", "megatron-bridge", "transformers", "torch")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def checkpoint_fingerprint(model_path: str) -> Dict[str, Any]:
    """Identify the checkpoint bytes the import reads.

    A local directory is fingerprinted by the name, size and mtime of its weight
    files and index/config files. Anything else (a Hub id) is identified by name
    only: Hub snapshots are content-addressed by revision, and re-resolving the
    revision here would need network access in every worker.
    """
    if not os.path.isdir(model_path):
        return {"id": model_path}
    entries: List[List[Any]] = []
    for name in sorted(os.listdir(model_path)):
        if not (name.endswith(_WEIGHT_SUFFIXES) or name in _META_FILES):
            continue
        st = os.stat(os.path.join(model_path, name))
        entries.append([name, st.st_size, int(st.st_mtime)])
    return {"dir": os.path.realpath(model_path), "files": entries}


def import_factors(
    *,
    model_path: str,
    bf16: bool,
    language_model_only: bool,
    enable_mtp: bool,
    model_config_kwargs: Dict[str, Any],
    transformer_config_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Everything that changes the converted weights or their shapes."""
    return {
        "checkpoint": checkpoint_fingerprint(model_path),
        "versions": {name: _package_version(name) for name in _VERSIONED_PACKAGES},
        "bf16": bf16,
        "language_model_only": language_model_only,
        "enable_mtp": enable_mtp,
        "model_config_kwargs": model_config_kwargs,
        "transformer_config_kwargs": transformer_config_kwargs,
    }


def digest(factors: Dict[str, Any]) -> str:
    canonical = json.dumps(factors, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _slug(model_path: str) -> str:
    text = model_path.strip().rstrip("/").lower()
    if os.path.isabs(text):
        text = os.path.basename(text)
    text = text.replace("/", "__")
    return re.sub(r"[^a-z0-9._-]+", "-", text).strip("-") or "model"


def cache_path(root: str, model_path: str, factors: Dict[str, Any]) -> str:
    return os.path.join(root, _slug(model_path), digest(factors))


@dataclass
class HFImportCache:
    path: str
    hit: bool
    factors: Dict[str, Any]


def resolve(root: str, model_path: str, factors: Dict[str, Any]) -> HFImportCache:
    """Locate the cache entry and decide hit/miss consistently across ranks.

    Rank 0 checks the marker (a shared filesystem may race with a writer, and
    ranks must not disagree on whether to run the HF import) and broadcasts.
    """
    path = cache_path(root, model_path, factors)
    decision = [os.path.isfile(os.path.join(path, COMPLETE_MARKER))]
    if dist.is_initialized():
        dist.broadcast_object_list(decision, src=0)
    return HFImportCache(path=path, hit=decision[0], factors=factors)


def _unwrap(module: nn.Module) -> nn.Module:
    while hasattr(module, "module"):
        module = module.module
    return module


def _base_model(actor_module: List[nn.Module]) -> nn.Module:
    assert len(actor_module) == 1, "Megatron virtual pipeline parallel is not yet supported"
    return _unwrap(actor_module[0])


def _dp_group():
    import megatron.core.parallel_state as mpu

    return mpu.get_data_parallel_group(with_context_parallel=True)


def load_into(actor_module: List[nn.Module], path: str) -> None:
    """Load the cached converted weights into the built (randomly initialized) model."""
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.serialization import (
        get_default_load_sharded_strategy,
    )
    from megatron.core.dist_checkpointing.strategies.fully_parallel import (
        FullyParallelLoadStrategyWrapper,
    )

    model = _base_model(actor_module)
    sharded_state_dict = {"model": model.sharded_state_dict()}
    strategy = FullyParallelLoadStrategyWrapper(get_default_load_sharded_strategy(path), _dp_group())
    state_dict = dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=path, sharded_strategy=strategy
    )
    model.load_state_dict(state_dict["model"], strict=True)


def save_from(actor_module: List[nn.Module], path: str, factors: Dict[str, Any]) -> None:
    """Write the converted weights of the just-imported model as a cache entry.

    All ranks write into one temp directory chosen by rank 0; rank 0 then stamps the
    marker and renames it into place. If another job completed the same entry in the
    meantime, the temp directory is dropped and theirs is kept.
    """
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.serialization import (
        get_default_save_sharded_strategy,
    )
    from megatron.core.dist_checkpointing.strategies.fully_parallel import (
        FullyParallelSaveStrategyWrapper,
    )

    is_rank_0 = not dist.is_initialized() or dist.get_rank() == 0
    tmp = [f"{path}.tmp-{uuid.uuid4().hex[:8]}"]
    if dist.is_initialized():
        dist.broadcast_object_list(tmp, src=0)
    tmp_path = tmp[0]
    if is_rank_0:
        os.makedirs(tmp_path, exist_ok=True)
    _barrier()

    model = _base_model(actor_module)
    strategy = FullyParallelSaveStrategyWrapper(get_default_save_sharded_strategy("torch_dist"), _dp_group())
    dist_checkpointing.save(
        sharded_state_dict={"model": model.sharded_state_dict()},
        checkpoint_dir=tmp_path,
        sharded_strategy=strategy,
        validate_access_integrity=True,
    )
    _barrier()

    if is_rank_0:
        with open(os.path.join(tmp_path, COMPLETE_MARKER), "w") as f:
            json.dump({"digest": digest(factors), "factors": factors}, f, indent=2, default=str)
        if os.path.isfile(os.path.join(path, COMPLETE_MARKER)):
            shutil.rmtree(tmp_path, ignore_errors=True)
            logger.info(f"HF import cache {path} was completed by another writer; dropped {tmp_path}")
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            os.rename(tmp_path, path)
    _barrier()


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def describe(cache: Optional[HFImportCache]) -> str:
    if cache is None:
        return "hf import cache: disabled"
    return f"hf import cache {'hit' if cache.hit else 'miss'}: {cache.path}"


__all__ = [
    "COMPLETE_MARKER",
    "HFImportCache",
    "cache_path",
    "checkpoint_fingerprint",
    "describe",
    "digest",
    "import_factors",
    "load_into",
    "resolve",
    "save_from",
]

# torch is imported for type completeness of callers that pass modules; keep the
# module importable without a GPU.
_ = torch
