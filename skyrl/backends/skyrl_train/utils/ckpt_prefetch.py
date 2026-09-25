"""Download a cloud checkpoint's shards while the trainer is still building its models.

On a resume from S3/GCS, ``_load_dist_checkpoint_from_cloud`` pulls each rank's shard only
once ``load_checkpoints`` runs -- after the engines are healthy and the models are built --
so the transfer sits alone on the critical path. Nothing about it needs the model: it is
bytes to local disk. Started right after the actor groups come up, it runs underneath the
Megatron build and the optimizer allocation instead (measured cover on a 14B resume: ~90s of
download against ~140s of ``build_models``).

**The worker thread must not touch the process group.** Megatron's build issues collectives on
the main thread throughout, and a concurrent ``dist.barrier()`` from a second thread on the
same group deadlocks. So the two collectives the download needs are both hoisted onto the main
thread: :func:`start_checkpoint_prefetch` barriers *before* the thread starts (after local rank
0 has made the directory) and the loader barriers *after* joining it. The thread itself only
issues object GETs.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import threading
from typing import Dict, Optional

import torch.distributed as dist
from loguru import logger

from skyrl.backends.skyrl_train.utils.io import io

# ``__<rank>_<n>.distcp`` -- one shard file per rank, as Megatron writes them.
SHARD_FILE_PATTERN = re.compile(r"__(\d+)_\d+\.distcp$")


def local_dir_for(ckpt_dir: str) -> str:
    """Node-local staging directory for a cloud checkpoint. All ranks on a node share it."""
    dir_hash = hashlib.md5(ckpt_dir.encode()).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f"skyrl_ckpt_load_{dir_hash}")


class _Prefetch:
    def __init__(self, local_dir: str) -> None:
        self.local_dir = local_dir
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None


_IN_FLIGHT: Dict[str, _Prefetch] = {}
_LOCK = threading.Lock()


def _download_shards(ckpt_dir: str, local_dir: str, global_rank: int, node_local_rank: int, state: _Prefetch) -> None:
    """Runs on the worker thread. Object GETs only -- no collectives."""
    try:
        entries = io.list_dir(ckpt_dir)
        for entry in entries:
            name = entry.rstrip("/").split("/")[-1]
            if not name:
                continue
            match = SHARD_FILE_PATTERN.search(name)
            cloud_entry = ckpt_dir.rstrip("/") + "/" + name
            if match:
                # Every rank fetches exactly its own shard, so a node downloads one full copy.
                if int(match.group(1)) == global_rank:
                    io.download_file(cloud_entry, os.path.join(local_dir, name))
            elif node_local_rank == 0 and not io.isdir(cloud_entry):
                io.download_file(cloud_entry, os.path.join(local_dir, name))
    except BaseException as e:  # noqa: BLE001 - surfaced on join, where the loader can fall back
        state.error = e


def start_checkpoint_prefetch(ckpt_dir: str, node_local_rank: int) -> bool:
    """Begin staging ``ckpt_dir`` locally. Call from the main thread on every rank.

    Returns False when there is nothing to do (local checkpoint, or already started), in which
    case the loader downloads inline exactly as before.
    """
    if not io.is_cloud_path(ckpt_dir):
        return False
    with _LOCK:
        if ckpt_dir in _IN_FLIGHT:
            return True

    local_dir = local_dir_for(ckpt_dir)
    # Directory setup and its barrier stay on the main thread: local rank 0 clears the
    # staging dir, and no rank may start writing into it before that finishes.
    if node_local_rank == 0:
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        os.makedirs(local_dir)
    dist.barrier()

    state = _Prefetch(local_dir)
    state.thread = threading.Thread(
        target=_download_shards,
        args=(ckpt_dir, local_dir, dist.get_rank(), node_local_rank, state),
        name="skyrl-ckpt-prefetch",
        daemon=True,
    )
    with _LOCK:
        _IN_FLIGHT[ckpt_dir] = state
    state.thread.start()
    logger.info(f"Prefetching checkpoint shards from {ckpt_dir} into {local_dir} while the models build")
    return True


def wait_for_checkpoint_prefetch(ckpt_dir: str) -> Optional[str]:
    """Join a prefetch started earlier and return its staging directory.

    ``None`` means there is no usable prefetch and the caller should download inline: either
    none was started, or the worker thread failed (already logged). A failed prefetch leaves
    the staging directory behind, so the caller is responsible for recreating it.
    """
    with _LOCK:
        state = _IN_FLIGHT.pop(ckpt_dir, None)
    if state is None:
        return None
    if state.thread is not None:
        state.thread.join()
    if state.error is not None:
        logger.warning(f"Checkpoint prefetch for {ckpt_dir} failed ({state.error!r}); downloading inline instead.")
        return None
    return state.local_dir


def clear_prefetch_state() -> None:
    """Drop bookkeeping without joining. For tests."""
    with _LOCK:
        _IN_FLIGHT.clear()
