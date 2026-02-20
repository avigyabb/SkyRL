---
description: Mandatory operational discipline. ALWAYS follow these rules when launching, restarting, debugging, or monitoring training. Never improvise — consult the documented skills and rules.
alwaysApply: true
---

# Operational Discipline — MANDATORY

## Always Follow the Documented Procedures

When performing ANY infrastructure or training operation:

1. **Read the relevant skill** (`launch-training`, `monitor-training`) BEFORE executing commands
2. **Read the relevant `.cursor/rules/`** (`infra-ops.md`, `training-pipeline.md`) for config details
3. **Follow the pre-launch checklist** in `launch-training` — every item, no skipping
4. **Never improvise** Docker commands, launch flags, or config from memory

## Debug vs Production

- **Debug mode** (testing a fix): no autoretry, separate log file, reduced config, **10–20 min monitoring sleep**. See `launch-training` skill → "Debug Mode"
- **Production mode** (real training): autoretry wrapper, main log, full config, 1 hour monitoring sleep
- After debug succeeds: revert ALL config changes, delete ALL debug checkpoints, clean stale Ray/dumps, then relaunch production

## Pre-Launch Cleanup (EVERY launch)

Always clean before starting:
```bash
docker exec skyrl-train bash -c 'rm -rf /tmp/ray/session_*'
docker exec skyrl-train bash -c 'find /tmp/ray -name "core" -delete 2>/dev/null'
```

## Container Recreation Essentials

When recreating the `skyrl-train` container, NEVER omit:
- `--user root` (NFS/UV cache are root-owned)
- `--shm-size=512g` (Megatron checkpoint needs ~60GB in `/dev/shm`)
- `sudo` prefix (required for GPU runtime access)

The exact command is in `infra-ops.md` and `launch-training` skill. Copy it, don't retype it.
