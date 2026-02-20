---
description: SkyRL repository architecture, layout, and key conventions. Read this first when working in this repo.
globs:
---

# SkyRL Architecture

## Repository Structure

- `skyrl-agent/` — Agent logic, rollout orchestration, reward computation, task definitions. Has its own `pyproject.toml` with extras for `skyrl-train`.
- `skyrl-train/` — Training framework built on Megatron-LM. Handles distributed training, policy loss computation, checkpointing, optimizer management.
- `skyrl-gym/` — Environment interfaces (not actively used in current Biomni training).
- `skyrl-tx/` — Data transformation utilities.

## Current Training Setup (GCP VM)

- **Hardware**: Single node, 8x H200 GPUs
- **Docker container**: `skyrl-train` (image `novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8`)
  - Workspace at `/workspace/SkyRL` (bind-mounted from `/home/ryan/SkyRL`)
  - NFS storage at `/mnt/biomni_filestore` (bind-mounted from `/mnt`)
  - Must use `--shm-size=64g` (Megatron workers need large `/dev/shm` for tensor serialization during checkpointing)
- **Ray**: Single-node cluster started inside the Docker container
- **Biomni runtime server**: Separate Docker container `biomni_exec_service` on the host, providing code execution at `http://localhost:8000`. Managed by `~/BioAgentOS/biomni_env_screen/restart_server.sh`.

## Package Management

- Uses `uv` as the Python package manager/runner
- Training launched via `uv run --frozen --extra skyrl-train`
- Entry point: `skyrl_agent.integrations.skyrl_train.skyrl_train_main`

## Key Conventions

- Config is Hydra/OmegaConf-based, passed as CLI overrides to the training script
- Chat template (`biomni_qwen3.jinja`) is shared between vLLM generation and training tokenization — changes affect both paths
- Secrets and env vars go in `.env.biomni` (not gitignored on purpose — needed for Ray worker processes)
- Logs are in `skyrl-agent/logs/` (gitignored via `*.log` pattern)
- `.cursor/` directory is NOT in `.gitignore` — these rules are part of the repo knowledge

## Maintenance Notes

- `biomni_codeact_agent.py` line 3 has a hardcoded Stanford `sys.path.append` — harmless but should be cleaned up
- The `.gemini/config.yaml` configures Gemini for code review on PRs
