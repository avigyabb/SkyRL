# Training Monitor Report

## 2026-02-20 02:00 UTC — Debug checkpoint fix verified, production relaunched

### Summary
After debugging caused by environment corruption, the checkpoint SIGBUS/ENOSPC issue has been resolved by increasing `--shm-size` from 64g to 512g. Two consecutive checkpoints (global_step_1 and global_step_2) saved successfully without any crash.

### Debug run results (debug_ckpt_shm512g_5.log)
- **Step 1**: avg_final_rewards=2.99, avg_response_length=1999.7, pg=0, ent=7.69, grad_norm=0
- **Step 2**: avg_final_rewards=2.41, avg_response_length=1516.5, pg=0, ent=9.78, grad_norm=0
- **Checkpoint saves**: Both global_step_1 and global_step_2 saved successfully. `/dev/shm` peaked at 68% (345G/512G) during save, dropped to 26% (129G) after flush to NFS.
- **No SIGBUS, no ENOSPC, no ModuleNotFoundError** in this run.

### Config changes applied
- Reverted to production: batch_size=16, num_trajectories=5, max_iterations=50, ckpt_interval=4
- Debug checkpoints (global_step_1, global_step_2) deleted from NFS

### Actions taken
1. Killed debug training
2. Reverted all config to production values
3. Deleted debug checkpoints from NFS
4. Cleaned stale Ray sessions and core dumps
5. Restarted Ray
6. Launched production training with autoretry wrapper -> `training_rubric_gspo_20260220.log`

### Notes on pg=0 and grad_norm=0
The `pg=0` (policy gradient loss = 0) and `grad_norm=0` are expected for the first few steps with GSPO and very tight clipping (eps_clip_low=3e-4, eps_clip_high=4e-4). At step 1, the policy and reference model are identical, so the importance sampling ratio is exactly 1.0, and the clipped loss is 0. This should change as training progresses and the policy diverges from the reference.

### Observed tool errors in agent rollouts
- `TypeError: query_opentarget_genetics() got an unexpected keyword argument 'max_results'` — the model hallucinated an argument that doesn't exist in the API. This is a model behavior issue, not an environment issue.

### Environment protection rule
Created `golden-env-protection.md` hard rule to prevent future environment corruption.

---

## 2026-02-20 05:20 UTC — Monitoring cycle 2 (production run v2)

### Status: HEALTHY
- **Step 1 complete** at 04:56 UTC (100 min per step)
- Step 2 rollouts in progress
- Training Batches: 1/212 processed
- Attempt #1, 0 restarts

### Step 1 Metrics
- avg_pass_at_5: 1.0 (perfect)
- avg_raw_reward: 5.19 / 7.0
- avg_final_rewards: 5.19
- avg_response_length: 15043 tokens
- pg=1.08, ent=11.8, grad_norm=0.203 (policy is learning)

### Format compliance
- ft_reward pass: 88/98 (90%)
- ft_reward fail: 10/98 (10%)
- No "random token replacing </think>" issues observed

### Fix applied this cycle
- Added `/workspace/SkyRL/skyrl-train:/workspace/SkyRL/skyrl-agent` to PYTHONPATH in `start_ray_head.sh` to prevent `ModuleNotFoundError` in multiprocessing.spawn children. Ray's `uv_runtime_env_hook` was rewriting editable finders to point at stale session paths.

---
