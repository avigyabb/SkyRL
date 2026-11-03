"""Backport of NVIDIA-NeMo/Megatron-Bridge#6456: cache small tensors and reuse ``safe_open`` handles in
``SafeTensorsStateSource``.

Exporting a quantized-checkpoint model (FP8 / MXFP4: DeepSeek-V4 / V4.1, Kimi K2.5 / K3, gpt-oss) to HF
-- every trainer -> vLLM weight sync -- re-quantizes each weight in the layout of its checkpoint
``*.scale`` tensor, and the bridge fetches those scales one key at a time:
``hf_state_dict[key]`` -> ``load_tensors([key])`` -> one ``safe_open`` of the shard (header parse) per
scale, on every sync. On DeepSeek-V4.1-Flash that is 46,412 opens at ~4.2 ms each, ~190 s per sync
summed over the exporting ranks; upstream measured 657-698 s -> 238-254 s per export on DeepSeek-V4-Pro.

The pinned Bridge (``c4067f50``) predates the fix (merged 2026-10-15 as ``f20da0c4``). This applies the
same change -- per-instance cache of tensors with at most 16M elements, reused handles, a stale handle
dropped before each retry, and the same three environment switches (``MBRIDGE_STATE_CACHE_SMALL_TENSORS``,
``MBRIDGE_STATE_REUSE_HANDLES``, ``MBRIDGE_STATE_SMALL_TENSOR_NUMEL``) -- to the pinned class.

Cost: the cached scales stay in host memory -- each rank keeps the scales of the weights it exports
(DeepSeek-V4.1-Flash: 23.6 GB of scales in the checkpoint, 17.4 GB of them routed experts, so ~2-3 GB
per rank at EP8). ``MBRIDGE_STATE_CACHE_SMALL_TENSORS=0`` turns the cache off.

Remove when the Megatron-Bridge pin includes ``f20da0c4``; the patch is skipped once the class has
upstream's ``_get_handle``.
"""

import os
import time
from collections import defaultdict
from glob import glob as file_glob
from pathlib import Path
from typing import Dict, List

import torch
from loguru import logger

_APPLIED = False


def _ensure_state(self) -> None:
    # Upstream sets these in __init__; instances built before the patch, or by subclasses that skip
    # __init__, get them on first use.
    if "_small_tensor_cache" in self.__dict__:
        return
    self._small_tensor_cache: Dict[str, torch.Tensor] = {}
    self._open_handles: Dict[str, object] = {}
    self._cache_small = os.environ.get("MBRIDGE_STATE_CACHE_SMALL_TENSORS", "1") == "1"
    self._reuse_handles = os.environ.get("MBRIDGE_STATE_REUSE_HANDLES", "1") == "1"
    self._small_numel = int(os.environ.get("MBRIDGE_STATE_SMALL_TENSOR_NUMEL", str(16 * 1024 * 1024)))


def _get_handle(self, file_path: Path):
    """Return a safe_open handle for ``file_path``, reused across calls when enabled."""
    from safetensors import safe_open

    if not self._reuse_handles:
        return safe_open(file_path, framework="pt", device="cpu")
    key = str(file_path)
    handle = self._open_handles.get(key)
    if handle is None:
        handle = safe_open(file_path, framework="pt", device="cpu")
        self._open_handles[key] = handle
    return handle


def _maybe_cache(self, key: str, tensor: torch.Tensor) -> None:
    # Keep a private copy: callers may modify the tensors they get back in place.
    if self._cache_small and tensor.numel() <= self._small_numel:
        self._small_tensor_cache[key] = tensor.clone()


def _invalidate_handle(self, file_path: Path) -> None:
    """Drop a reused handle so the next access re-reads the shard header (e.g. after a retry)."""
    self._open_handles.pop(str(file_path), None)


def load_tensors(self, keys_to_load: List[str]) -> Dict[str, torch.Tensor]:
    """The pinned ``SafeTensorsStateSource.load_tensors`` with upstream #6456's hunks."""
    if not keys_to_load:
        return {}
    _ensure_state(self)

    from safetensors import safe_open

    loaded_tensors = {}
    remaining_keys = set(keys_to_load)
    key_to_filename_map = self.key_to_filename_map

    if self._cache_small:
        for key in list(remaining_keys):
            cached = self._small_tensor_cache.get(key)
            if cached is not None:
                loaded_tensors[key] = cached.clone()
                remaining_keys.discard(key)
        if not remaining_keys:
            return loaded_tensors

    max_retries = 3

    # A selected subtree is authoritative: do not fall back to unrelated
    # shards for absent keys, including when a selected shard is incomplete.
    if self.key_prefix is not None:
        for key in keys_to_load:
            if key not in key_to_filename_map:
                raise KeyError(key)

    if key_to_filename_map:
        file_to_keys_map = defaultdict(list)
        for key in list(remaining_keys):
            if key in key_to_filename_map:
                filename = key_to_filename_map[key]
                file_to_keys_map[filename].append(key)

        for filename, keys_in_file in file_to_keys_map.items():
            file_path = (
                self._download_file(filename)
                if self._lazy_hub and not Path(self.model_name_or_path).is_dir()
                else self.path / filename
            )
            if file_path.exists():
                for attempt in range(max_retries):
                    f = _get_handle(self, file_path)  # may be shared across calls; not closed here
                    file_keys = set(f.keys())
                    for key in keys_in_file:
                        source_key = (self.key_prefix or "") + key
                        if source_key in file_keys and key not in loaded_tensors:
                            loaded_tensors[key] = f.get_tensor(source_key)
                            _maybe_cache(self, key, loaded_tensors[key])
                            remaining_keys.discard(key)
                    still_missing = [k for k in keys_in_file if k in remaining_keys]
                    if not still_missing:
                        break
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries}: {len(still_missing)} keys "
                            f"not found in {filename}, retrying after delay..."
                        )
                        # the shard may be replaced meanwhile; do not retry on a stale header
                        _invalidate_handle(self, file_path)
                        time.sleep(1.0 * (attempt + 1))

    if remaining_keys and self.key_prefix is None:
        safetensor_files = file_glob(str(self.path / "*.safetensors"))
        if not safetensor_files and not key_to_filename_map and not loaded_tensors:
            raise FileNotFoundError(
                f"No .safetensors files found in {self.model_name_or_path} to load keys: {remaining_keys}"
            )
        for safetensor_file_path in safetensor_files:
            if not remaining_keys:
                break
            with safe_open(safetensor_file_path, framework="pt", device="cpu") as f:
                current_file_keys = f.keys()
                for key in list(remaining_keys):
                    if key in current_file_keys:
                        loaded_tensors[key] = f.get_tensor(key)
                        remaining_keys.remove(key)

    if remaining_keys:
        raise KeyError(f"Keys not found in safetensors from {self.model_name_or_path}: {remaining_keys}")

    return loaded_tensors


def apply_bridge_state_small_tensor_cache_patch() -> bool:
    """Install the backport on ``SafeTensorsStateSource``. Idempotent; a no-op on a Bridge with #6456."""
    global _APPLIED
    if _APPLIED:
        return True
    try:
        from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource
    except ImportError:
        return False
    if hasattr(SafeTensorsStateSource, "_get_handle"):
        return False  # upstream already has it
    SafeTensorsStateSource.load_tensors = load_tensors
    _APPLIED = True
    logger.info("Megatron-Bridge SafeTensorsStateSource: small-tensor cache + reused safe_open handles (#6456)")
    return True
