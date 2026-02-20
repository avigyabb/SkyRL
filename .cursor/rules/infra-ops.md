---
description: Infrastructure and operations knowledge — Docker, Ray, runtime server, autoretry, and common failure modes. Consult when launching, restarting, or debugging infrastructure issues.
globs:
---

# Infrastructure & Operations

## Docker Container: skyrl-train

```bash
sudo docker run -d \
  --runtime=nvidia --gpus all \
  --shm-size=512g \
  --network=host \
  --name skyrl-train \
  --user root \
  -v /home/ryan/SkyRL:/workspace/SkyRL \
  -v /mnt:/mnt \
  novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 sleep infinity
```

**Critical flags**:
- `--shm-size=512g` — Megatron workers serialize large model shards (~60GB for 30B-A3B) through `/dev/shm` during checkpointing. 64g was insufficient and caused `SIGBUS` crashes. The host has 1.5TB RAM, so 512g is safe.
- `--user root` — **NEVER omit this.** The NFS filestore (`/mnt/biomni_filestore`) and UV cache have root-owned files. Without `--user root`, the container runs as `ray` (uid=1000) and all file operations on NFS fail with `Permission denied`.

`tmux` is not in the base image — install after container creation:
```bash
docker exec skyrl-train bash -c 'apt-get update -qq && apt-get install -y -qq tmux < /dev/null'
```

## Ray

- Single-node cluster inside `skyrl-train` container
- Started via `examples/run_biomni/start_ray_head.sh`
- Session storage at `/tmp/ray/session_*` — stale sessions accumulate and should be cleaned before starting new runs
- Verify with: `ray status` (using the venv Python at `/mnt/biomni_filestore/venvs/skyrl-agent/bin/ray`)

## Biomni Runtime Server

- Docker container: `biomni_exec_service` (on host, not inside skyrl-train)
- Provides code execution endpoint at `http://localhost:8000`
- Health check: `curl http://localhost:8000/healthcheck`
- Managed by: `~/BioAgentOS/biomni_env_screen/restart_server.sh` (auto-restarts on crash)
- Logs: `~/BioAgentOS/biomni_runtime.log`
- Runs in host tmux session 0 (`tmux a -t 0` or alias `tx 0`)

## Autoretry Wrapper

**File**: `examples/run_biomni/run_biomni_qwen30ba3b_gspo_tis_autoretry.sh`

- Infinite retry loop by default
- Restarts Ray head between retries (unless `--no-ray-restart`)
- Detects rapid crashes (<120s) and warns
- Relies on `resume_mode=latest` in the training script
- Wrapper log: `logs/autoretry_wrapper.log`
- Search `"Attempt #"` in the training log to count retries
- Search `"Training exited with code"` for crash details

## Common Failure Modes

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| `ENOSPC` / `SIGBUS` at checkpoint | `/dev/shm` too small | Recreate container with `--shm-size=64g` |
| `SIGBUS` → core dump → disk full cascade | Core dumps (40GB+) fill root partition, causing subsequent checkpoint saves to silently fail (only metadata `common.pt` written). `latest_ckpt_global_step.txt` never created → training restarts from scratch. | Add `ulimit -c 0` to training script. Check `du -sh /tmp/ray/*/runtime_resources/working_dir_files/*/core` for stale core dumps. |
| Checkpoint save "succeeds" but only 412KB | Disk full during `dist_checkpointing.save()` — model weight shards silently fail, only `common.pt` metadata written | Free disk space, verify checkpoint size after saves. Check `df -h /` in container. |
| CUDA OOM in `optimizer_step` | Transient memory fragmentation | Autoretry handles this; if persistent, investigate memory pressure |
| CUDA OOM in `get_grad_norm_fp32` | Same as above | Same as above |
| Ray fails to start | Stale `/tmp/ray/session_*` | `rm -rf /tmp/ray/session_*` |
| Ray `/tmp/ray` fills disk (47GB+) | `working_dir_files` accumulates across restarts when `--no-ray-restart` is used | Clean stale ray packages: `du -sh /tmp/ray/*/runtime_resources/working_dir_files/*` |
| Runtime server unreachable (HTTP errors in rollout) | `biomni_exec_service` container down | Restart via tmux session 0 |
| Training restarts but no progress | `resume_mode=none` or `latest_ckpt_global_step.txt` missing | Check for missing tracking file, change to `resume_mode=latest` |
| Rapid crash loop (<120s per attempt) | Config/code bug, not transient | Stop autoretry, read the actual error, trace through code |

## Debugging vs Production Training

When iterating on a fix (e.g., checkpoint issues, config bugs), use **debug mode** — NOT production mode:

| | Debug Mode | Production Mode |
|---|---|---|
| **Autoretry** | NO — run the training script directly | YES — use autoretry wrapper |
| **Log file** | New, separate file (e.g. `debug_ckpt_test_1.log`) | Main training log |
| **Config** | Reduced batch/traj/iters for fast iteration | Full batch_size=16, traj=5, max_iter=50 |
| **Checkpoints** | `ckpt_interval=1` for quick testing | `ckpt_interval=4` (normal) |
| **After debugging** | Revert ALL config changes, delete ALL debug checkpoints | — |

**Debug launch (no autoretry)** — must include UV env vars + PYTHONPATH:
```bash
docker exec skyrl-train tmux new-session -d -s training \
  "export UV_CACHE_DIR=/mnt/biomni_filestore/uv_cache && \
   export XDG_CACHE_HOME=/mnt/biomni_filestore/uv_cache && \
   export UV_PROJECT_ENVIRONMENT=/mnt/biomni_filestore/venvs/skyrl-agent && \
   export PYTHONPATH=/workspace/SkyRL/skyrl-train:/workspace/SkyRL/skyrl-agent:/mnt/biomni_filestore/venvs/skyrl-agent/lib/python3.12/site-packages:\${PYTHONPATH:-} && \
   bash /workspace/SkyRL/skyrl-agent/examples/run_biomni/run_biomni_qwen30ba3b_rubric_gspo_tis.sh \
     &> /workspace/SkyRL/skyrl-agent/logs/debug_<description>.log; exec bash"
```

**Why the env vars**: `start_ray_head.sh` sets `UV_PROJECT_ENVIRONMENT` etc. but those exports die with that `docker exec` shell. `PYTHONPATH` must include the workspace source dirs (`skyrl-train/`, `skyrl-agent/`) because `multiprocessing.spawn` children (used by PyTorch/Megatron) don't inherit `uv run`'s dynamic path additions.

**Critical: Stale editable finders.** Ray's `working_dir` mechanism causes `uv` to write editable install finders (`.pth`/`_finder.py` in venv site-packages) pointing to Ray session-specific paths (`/tmp/ray/session_*/runtime_resources/...`). When Ray sessions are cleaned up, these paths become stale and the finders fail silently, blocking imports. After cleaning Ray sessions, check for and remove stale finders:
```bash
docker exec skyrl-train bash -c 'rm -f /mnt/biomni_filestore/venvs/skyrl-agent/lib/python3.12/site-packages/__editable__*skyrl_train* /mnt/biomni_filestore/venvs/skyrl-agent/lib/python3.12/site-packages/__editable__*skyrl_gym* /mnt/biomni_filestore/venvs/skyrl-agent/lib/python3.12/site-packages/__editable__*skyagent*'
```

## Pre-Launch Cleanup (ALWAYS do before any launch)

```bash
# 1. Clean stale Ray sessions
docker exec skyrl-train bash -c 'rm -rf /tmp/ray/session_*'

# 2. Clean core dumps (can be 40GB+ each)
docker exec skyrl-train bash -c 'find /tmp/ray -name "core" -delete 2>/dev/null'

# 3. Remove stale editable finders (Ray working_dir creates these with session-specific paths)
docker exec skyrl-train bash -c 'rm -f /mnt/biomni_filestore/venvs/skyrl-agent/lib/python3.12/site-packages/__editable__*skyrl_train* /mnt/biomni_filestore/venvs/skyrl-agent/lib/python3.12/site-packages/__editable__*skyrl_gym* /mnt/biomni_filestore/venvs/skyrl-agent/lib/python3.12/site-packages/__editable__*skyagent*'

# 4. Check disk space
docker exec skyrl-train df -h /dev/shm /tmp /
df -h /mnt/biomni_filestore
```

## Filesystem Layout

| Path | Contents | Concern |
|------|----------|---------|
| `/dev/shm` (in container) | Shared memory for tensor serialization | Must be 512g (was 64g, caused SIGBUS) |
| `/tmp/ray/` (in container) | Ray session storage, logs | Clean stale sessions |
| `/mnt/biomni_filestore/` | NFS: checkpoints, venvs, datasets | Check disk space for checkpoint growth |
| `/workspace/SkyRL/` (in container) | Bind-mount of `/home/ryan/SkyRL` | Working directory |
| `~/BioAgentOS/biomni_runtime.log` | Runtime server logs | Rotate before each training run |
