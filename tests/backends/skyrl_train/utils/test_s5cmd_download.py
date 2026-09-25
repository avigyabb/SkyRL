"""
uv run --extra dev --extra skyrl-train pytest tests/backends/skyrl_train/utils/test_s5cmd_download.py

``io.download_file`` pulls S3 objects with s5cmd when it is available. These cover the
decision and the failure handling, not the transfer: every path that declines must fall
back to fsspec, and every path that fails must not leave a partial file behind (a short
file is indistinguishable from a good one to the checkpoint loader).
"""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from skyrl.backends.skyrl_train.utils.io import io as io_mod

S3 = "s3://bucket/ckpt/__0_0.distcp"
BIG = io_mod._S5CMD_MIN_BYTES * 4


@pytest.fixture
def fake_fs():
    """Stub the filesystem so size lookups do not touch the network."""
    fs = MagicMock()
    fs._strip_protocol.side_effect = lambda p: p.replace("s3://", "")
    fs.info.return_value = {"size": BIG}
    with patch.object(io_mod, "_get_filesystem", return_value=fs):
        yield fs


def completed(returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


def test_downloads_with_s5cmd_and_reports_success(tmp_path, fake_fs):
    dest = tmp_path / "shard"

    def fake_run(cmd, **kwargs):
        dest.write_bytes(b"x" * BIG)
        return completed()

    with (
        patch.object(io_mod, "_find_s5cmd", return_value="/bin/s5cmd"),
        patch.object(io_mod.subprocess, "run", side_effect=fake_run) as run,
    ):
        assert io_mod._s5cmd_download(S3, str(dest)) is True
    assert S3 in run.call_args[0][0] and str(dest) in run.call_args[0][0]


def test_declines_when_s5cmd_is_absent(tmp_path, fake_fs):
    with patch.object(io_mod, "_find_s5cmd", return_value=None):
        assert io_mod._s5cmd_download(S3, str(tmp_path / "s")) is False


def test_declines_for_small_objects(tmp_path, fake_fs):
    fake_fs.info.return_value = {"size": 1024}
    with patch.object(io_mod, "_find_s5cmd", return_value="/bin/s5cmd"):
        assert io_mod._s5cmd_download(S3, str(tmp_path / "s")) is False


def test_declines_when_disabled(tmp_path, fake_fs):
    with patch.object(io_mod, "_S5CMD_DISABLED", True), patch.object(io_mod, "_find_s5cmd", return_value="/bin/s5cmd"):
        assert io_mod._s5cmd_download(S3, str(tmp_path / "s")) is False


def test_declines_when_size_lookup_fails(tmp_path, fake_fs):
    fake_fs.info.side_effect = OSError("no such key")
    with patch.object(io_mod, "_find_s5cmd", return_value="/bin/s5cmd"):
        assert io_mod._s5cmd_download(S3, str(tmp_path / "s")) is False


def test_nonzero_exit_removes_the_partial_file(tmp_path, fake_fs):
    dest = tmp_path / "shard"

    def fake_run(cmd, **kwargs):
        dest.write_bytes(b"partial")
        return completed(returncode=1, stderr="boom")

    with (
        patch.object(io_mod, "_find_s5cmd", return_value="/bin/s5cmd"),
        patch.object(io_mod.subprocess, "run", side_effect=fake_run),
    ):
        assert io_mod._s5cmd_download(S3, str(dest)) is False
    assert not dest.exists(), "a partial download must not survive for the loader to read"


def test_short_file_is_rejected_and_removed(tmp_path, fake_fs):
    dest = tmp_path / "shard"

    def fake_run(cmd, **kwargs):
        dest.write_bytes(b"x" * (BIG // 2))  # exit 0 but truncated
        return completed()

    with (
        patch.object(io_mod, "_find_s5cmd", return_value="/bin/s5cmd"),
        patch.object(io_mod.subprocess, "run", side_effect=fake_run),
    ):
        assert io_mod._s5cmd_download(S3, str(dest)) is False
    assert not dest.exists()


def test_launch_failure_falls_back(tmp_path, fake_fs):
    with (
        patch.object(io_mod, "_find_s5cmd", return_value="/bin/s5cmd"),
        patch.object(io_mod.subprocess, "run", side_effect=OSError("exec format error")),
    ):
        assert io_mod._s5cmd_download(S3, str(tmp_path / "s")) is False


def test_download_file_falls_back_to_fsspec_when_s5cmd_declines(tmp_path, fake_fs):
    dest = tmp_path / "shard"
    with (
        patch.object(io_mod, "_s5cmd_download", return_value=False),
        patch.object(io_mod, "call_with_s3_retry") as retry,
    ):
        io_mod.download_file(S3, str(dest))
    retry.assert_called_once()


def test_download_file_skips_fsspec_when_s5cmd_succeeds(tmp_path, fake_fs):
    with (
        patch.object(io_mod, "_s5cmd_download", return_value=True),
        patch.object(io_mod, "call_with_s3_retry") as retry,
    ):
        io_mod.download_file(S3, str(tmp_path / "shard"))
    retry.assert_not_called()


def test_gcs_never_routes_through_s5cmd(tmp_path, fake_fs):
    # s5cmd is S3-only; gs:// must keep using fsspec.
    with patch.object(io_mod, "_s5cmd_download") as s5:
        io_mod.download_file("gs://bucket/o", str(tmp_path / "o"))
    s5.assert_not_called()


def test_find_s5cmd_uses_the_venv_when_not_on_path():
    # Workers re-exec into the project venv, where s5cmd is installed but PATH may not cover it.
    with patch.object(io_mod.shutil, "which", return_value=None), patch.object(io_mod.os, "access", return_value=True):
        found = io_mod._find_s5cmd()
    assert found and found.endswith(os.path.join("bin", "s5cmd"))
