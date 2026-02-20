---
description: HARD RULE — Protects the working environment from accidental damage. ALWAYS applies. Violation of these rules has previously cost a full day of training progress.
alwaysApply: true
---

# GOLDEN ENVIRONMENT PROTECTION — HARD RULE

## NEVER modify these files or directories

The following are part of the **golden working environment**. They were set up once via `setup_venv.sh` and must NEVER be modified, deleted, moved, chown'd, or overwritten by the agent:

- `/mnt/biomni_filestore/venvs/skyrl-agent/` — the shared venv (NFS-mounted, persists across container recreations)
- `/mnt/biomni_filestore/uv_cache/` — the UV package cache
- Any `__editable__*.pth` or `__editable__*_finder.py` files in the venv's `site-packages/`
- Any `.dist-info/` directories in the venv's `site-packages/`

**No exceptions.** Do not delete, edit, chown, chmod, or rm -rf any of these. Do not run `uv sync`, `uv pip install`, `pip install`, or any package manager command against this venv unless the user explicitly asks you to.

## If something breaks, RESET — don't patch

If training fails with import errors, permission errors, or environment issues:

1. **STOP.** Do not attempt to fix the environment by modifying files.
2. **Identify what YOU changed** that caused the breakage. Check your own recent actions first.
3. **Revert YOUR changes.** If you modified config files, revert them. If you deleted files, that's already irreversible damage — tell the user immediately.
4. **If the venv is corrupted**, the ONLY approved fix is: ask the user, then re-run `setup_venv.sh` inside the container. Nothing else.
5. **If the container needs recreation**, follow the EXACT command in `infra-ops.md` — copy it, don't retype from memory. Then re-run `setup_venv.sh`.

## What you ARE allowed to modify

- Training config files (`run_biomni_*.sh`, `*.yaml`) — these are version-controlled and easily reverted
- Training log files
- Checkpoint directories (delete debug checkpoints, etc.)
- Ray session directories (`/tmp/ray/session_*`) — cleanup before launches is fine
- Core dump files — cleanup is fine
- Files in the workspace source (`/workspace/SkyRL/` or `/home/ryan/SkyRL/`) — code changes are fine

## Container recreation checklist

When recreating the `skyrl-train` container, use the EXACT command from `infra-ops.md`. Critical flags that must NEVER be omitted:
- `--user root`
- `--shm-size=512g`
- `sudo` prefix

After recreation: install tmux, then run `setup_venv.sh`, then proceed with Ray + training.

## Historical context

On 2026-02-19, the agent deleted editable install files from the venv and ran chown on the UV cache, corrupting the environment. This caused a full day of wasted training time debugging `ModuleNotFoundError: No module named 'skyrl_train'` — an error that did not exist before the agent's modifications. The fix required re-running `setup_venv.sh` from scratch.
