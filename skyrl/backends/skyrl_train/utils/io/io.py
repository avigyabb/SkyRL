"""
File I/O utilities for handling both local filesystem and cloud storage (S3/GCS).

This module provides a unified interface for file operations that works with:
- Local filesystem paths
- S3 paths (s3://bucket/path)
- Google Cloud Storage paths (gs://bucket/path or gcs://bucket/path)

Uses fsspec for cloud storage abstraction.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from typing import Optional

import fsspec
from loguru import logger

from .s3fs import ClientError, call_with_s3_retry, get_s3_fs, s3_refresh_if_expiring


def is_cloud_path(path: str) -> bool:
    """Check if the given path is a cloud storage path."""
    return path.startswith(("s3://", "gs://", "gcs://"))


def _get_filesystem(path: str):
    """Get the appropriate filesystem for the given path."""
    if not is_cloud_path(path):
        return fsspec.filesystem("file")

    proto = path.split("://", 1)[0]
    if proto == "s3":
        fs = get_s3_fs()
        s3_refresh_if_expiring(fs)
        return fs
    return fsspec.filesystem(proto)


def open_file(path: str, mode: str = "rb"):
    """Open a file using fsspec, works with both local and cloud paths."""
    if not is_cloud_path(path):
        return fsspec.open(path, mode)

    fs = _get_filesystem(path)
    norm = fs._strip_protocol(path)
    try:
        return fs.open(norm, mode)
    except ClientError as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code")
        if code in {"ExpiredToken", "ExpiredTokenException", "RequestExpired"} and hasattr(fs, "connect"):
            try:
                fs.connect(refresh=True)
            except Exception:
                pass
            return fs.open(norm, mode)
        raise


def makedirs(path: str, exist_ok: bool = True) -> None:
    """Create directories. Only applies to local filesystem paths."""
    if not is_cloud_path(path):
        os.makedirs(path, exist_ok=exist_ok)


def exists(path: str) -> bool:
    """Check if a file or directory exists."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        return call_with_s3_retry(fs, fs.exists, path)
    return fs.exists(path)


def isdir(path: str) -> bool:
    """Check if path is a directory."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        return call_with_s3_retry(fs, fs.isdir, path)
    return fs.isdir(path)


def list_dir(path: str) -> list[str]:
    """List contents of a directory."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        return call_with_s3_retry(fs, fs.ls, path, detail=False)
    return fs.ls(path, detail=False)


def remove(path: str) -> None:
    """Remove a file or directory."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        if call_with_s3_retry(fs, fs.isdir, path):
            call_with_s3_retry(fs, fs.rm, path, recursive=True)
        else:
            call_with_s3_retry(fs, fs.rm, path)
        return
    if fs.isdir(path):
        fs.rm(path, recursive=True)
    else:
        fs.rm(path)


# fsspec downloads a single object on one stream. For the multi-GiB shards of a
# distributed checkpoint that leaves most of the link idle: measured on one 48.15 GiB
# shard from S3 on an 8xH100 node, fsspec `get_file` ran at 0.33 GiB/s (144.1s) while
# s5cmd's multipart transfer ran at 0.93 GiB/s (51.5s), 2.8x faster. Issuing concurrent
# ranged GETs through fsspec instead was measured *slower* than the single stream
# (0.21 GiB/s), so the win comes from s5cmd's transfer machinery, not from concurrency
# per se. s5cmd ships with the `aws` extra; when it is missing we fall back silently.
_S5CMD_DISABLED = os.environ.get("SKYRL_DISABLE_S5CMD", "") not in ("", "0", "false", "False")
# Below this, process spawn dominates and the two paths are indistinguishable.
_S5CMD_MIN_BYTES = 64 << 20


def _find_s5cmd() -> Optional[str]:
    """Locate the s5cmd binary, or None. Workers re-exec into the project venv, so
    check it explicitly rather than trusting PATH."""
    exe = shutil.which("s5cmd")
    if exe:
        return exe
    candidate = os.path.join(sys.prefix, "bin", "s5cmd")
    return candidate if os.access(candidate, os.X_OK) else None


def _s5cmd_download(cloud_path: str, local_path: str) -> bool:
    """Download one S3 object with s5cmd. Returns False to fall back to fsspec.

    s5cmd resolves credentials through the AWS SDK's default chain, the same one
    botocore/s3fs use, so it inherits instance/web-identity roles without extra config.
    """
    if _S5CMD_DISABLED:
        return False
    exe = _find_s5cmd()
    if exe is None:
        return False
    try:
        size = _get_filesystem(cloud_path).info(_get_filesystem(cloud_path)._strip_protocol(cloud_path))["size"]
    except Exception:  # noqa: BLE001 - a stat failure is fsspec's problem to report, not ours
        return False
    if size < _S5CMD_MIN_BYTES:
        return False

    concurrency = os.environ.get("SKYRL_S5CMD_CONCURRENCY", "16")
    numworkers = os.environ.get("SKYRL_S5CMD_NUMWORKERS", "32")
    cmd = [exe, "--numworkers", numworkers, "cp", "--concurrency", concurrency, cloud_path, local_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as e:
        logger.warning(f"s5cmd could not be launched ({e}); falling back to fsspec for {cloud_path}")
        return False
    if proc.returncode != 0:
        # A partial file would be indistinguishable from a good one downstream.
        if os.path.exists(local_path):
            os.remove(local_path)
        logger.warning(
            f"s5cmd exited {proc.returncode} for {cloud_path}; falling back to fsspec. "
            f"stderr: {proc.stderr.strip()[:500]}"
        )
        return False
    if not os.path.exists(local_path) or os.path.getsize(local_path) != size:
        got = os.path.getsize(local_path) if os.path.exists(local_path) else 0
        if os.path.exists(local_path):
            os.remove(local_path)
        logger.warning(f"s5cmd wrote {got} bytes for {cloud_path}, expected {size}; falling back to fsspec")
        return False
    return True


def download_file(cloud_path: str, local_path: str) -> None:
    """Download a single file from cloud storage to local storage.

    Args:
        cloud_path: Source file path in cloud storage (s3://, gs://, gcs://).
        local_path: Destination path on the local filesystem.
    """
    if not is_cloud_path(cloud_path):
        raise ValueError(f"Source must be a cloud path, got: {cloud_path}")

    parent = os.path.dirname(os.path.abspath(local_path))
    os.makedirs(parent, exist_ok=True)

    if cloud_path.startswith("s3://") and _s5cmd_download(cloud_path, local_path):
        return

    fs = _get_filesystem(cloud_path)
    remote = fs._strip_protocol(cloud_path)
    if cloud_path.startswith("s3://"):
        call_with_s3_retry(fs, fs.get_file, remote, local_path)
    else:
        fs.get_file(cloud_path, local_path)


def upload_directory(local_path: str, cloud_path: str) -> None:
    """Upload a local directory to cloud storage."""
    if not is_cloud_path(cloud_path):
        raise ValueError(f"Destination must be a cloud path, got: {cloud_path}")

    fs = _get_filesystem(cloud_path)
    # NOTE (sumanthrh): While uploading files in a directory `src` to an existing directory `dst` with fsspec,
    # we need to ensure that the file path ends in a trailing slash. otherwise, fsspec will create a subdirectory
    # `dst/src` instead of directly syncing contents of `src` into the root `dst` directory
    local_path = os.path.join(local_path, "")
    if cloud_path.startswith("s3://"):
        call_with_s3_retry(fs, fs.put, local_path, fs._strip_protocol(cloud_path), recursive=True)
    else:
        fs.put(local_path, cloud_path, recursive=True)
    logger.info(f"Uploaded {local_path} to {cloud_path}")


def download_directory(cloud_path: str, local_path: str) -> None:
    """Download a cloud directory to local storage."""
    if not is_cloud_path(cloud_path):
        raise ValueError(f"Source must be a cloud path, got: {cloud_path}")

    fs = _get_filesystem(cloud_path)
    if cloud_path.startswith("s3://"):
        call_with_s3_retry(fs, fs.get, fs._strip_protocol(cloud_path), local_path, recursive=True)
    else:
        fs.get(cloud_path, local_path, recursive=True)
    logger.info(f"Downloaded {cloud_path} to {local_path}")


@contextmanager
def local_work_dir(output_path: str):
    """
    Context manager that provides a local working directory.

    For local paths, returns the path directly.
    For cloud paths, creates a temporary directory and uploads content at the end.

    Args:
        output_path: The final destination path (local or cloud)

    Yields:
        str: Local directory path to work with

    Example:
        with local_work_dir("s3://bucket/model") as work_dir:
            # Save files to work_dir
            model.save_pretrained(work_dir)
            # Files are automatically uploaded to s3://bucket/model at context exit
    """
    if is_cloud_path(output_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                yield temp_dir
            finally:
                # Upload everything from temp_dir to cloud path
                upload_directory(temp_dir, output_path)
                logger.info(f"Uploaded directory contents to {output_path}")
    else:
        # For local paths, ensure directory exists and use it directly
        makedirs(output_path, exist_ok=True)
        yield output_path


@contextmanager
def local_read_dir(input_path: str):
    """
    Context manager that provides a local directory with content from input_path.

    For local paths, returns the path directly.
    For cloud paths, downloads content to a temporary directory.

    Args:
        input_path: The source path (local or cloud)

    Yields:
        str: Local directory path containing the content

    Example:
        with local_read_dir("s3://bucket/model") as read_dir:
            # Load files from read_dir
            model = AutoModel.from_pretrained(read_dir)
    """
    if is_cloud_path(input_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download everything from cloud path to temp_dir
            download_directory(input_path, temp_dir)
            logger.info(f"Downloaded directory contents from {input_path}")
            yield temp_dir
    else:
        # For local paths, use directly (but check it exists)
        if not exists(input_path):
            raise FileNotFoundError(f"Path does not exist: {input_path}")
        yield input_path


@contextmanager
def local_read_files(input_dir, filenames):
    """
    Context manager that provides a local directory with specific files from input_dir.

    For local paths, returns the path directly.
    For cloud paths, downloads specified files to a temporary directory.

    Args:
        input_dir: The source directory (local or cloud)
        filenames: List of filenames to download

    Yields:
        str: Local directory path containing the requested files

    Example:
        with local_read_files("s3://bucket/model", ["config.json", "pytorch_model.bin"]) as read_dir:
            # Load files from read_dir
            config_path = os.path.join(read_dir, "config.json")
            model_path = os.path.join(read_dir, "pytorch_model.bin")
    """
    if is_cloud_path(input_dir):
        with tempfile.TemporaryDirectory() as temp_dir:
            for name in filenames:
                src = input_dir.rstrip("/") + "/" + name
                if exists(src):
                    download_file(src, os.path.join(temp_dir, name))
            yield temp_dir
    else:
        if not exists(input_dir):
            raise FileNotFoundError(f"Path does not exist: {input_dir}")
        yield input_dir
