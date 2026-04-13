# Full CHORD Correction Analysis and Training Launch

All correction/reward logic lives in `skyrl-agent/skyrl_agent/tasks/biomni_rubric_reward_adapter.py`. Launch scripts and YAMLs live in `skyrl-agent/examples/run_biomni/`.

## Phase 1: Stop current ft-chord training [COMPLETED]

Kill the running training process (tmux session or direct PID), then `ray stop` to free GPU resources. The ft-chord step-16 checkpoint is preserved on the filestore but will not be used further.

## Phase 2: Enhance correction logging (debug-only) [COMPLETED]

Added verbose logging gated behind a module-level `_CHORD_DEBUG_CORRECTIONS` flag in `biomni_rubric_reward_adapter.py`:

- **When `True`**: log ALL corrections with their originals and append a JSON record per trajectory to `/tmp/chord_corrections_dump.jsonl`
- **When `False` (production)**: keep current behavior (1 random sample logged per trajectory)

The flag is hardcoded (not env var) because Ray workers don't inherit env vars from launch scripts.

## Phase 3: Create test rollout script [COMPLETED]

Created `run_biomni_chord_correction_test.sh` and `biomni_codeact_rubric_rl_qwen8b_chord_test.yaml`:
- Model: Base SFT at `/mnt/biomni_filestore/model_weights/qwen3-8b-sft-full-v1/global_step_104`
- Batch: `BATCH_SIZE=8`, `NUM_TRAJ=2` (16 total samples)
- No checkpointing, no wandb, `log_heavy_freq=1`

## Phase 4: Run test rollout and analyze corrections [COMPLETED → RE-RUNNING]

Run the test script. After first rollout + training step completes, kill the process and analyze:

1. Parse `/tmp/chord_corrections_dump.jsonl` for all correction pairs
2. For each (original, corrected) pair, check:
   - Does the corrected `<think>` block only reference prior observations? (no future leakage)
   - Does the corrected `<execute>` block avoid looping API calls?
   - Does the correction preserve the agent's existing methodology where it was working?
   - Does the correction align with the agent system prompt?
   - Is the correction generalizable (no task-specific hardcoded knowledge)?
3. Document any problematic patterns

**IMPORTANT**: Read EVERY problematic correction in full, not just summaries.

### Round 1 Findings (pre-prompt-fix)
- **API call loops**: Corrections generated `for gene in genes: query_opentarget(gene)` patterns
- **Future information leakage**: `<think>` blocks referencing observations from later turns
- **Fabricated data**: Correction #16 preserved hardcoded variant scores from a degenerated original trajectory
- **Methodology changes**: Some corrections changed strategy beyond fixing the identified problem

### Prompt Fixes Applied (Round 2)
1. **Guideline #2 (LIMIT EXTERNAL API CALLS)**: Added explicit instruction to batch multiple queries into a single `advanced_web_search` prompt instead of looping
2. **New Guideline #5 (NO FABRICATED DATA)**: Added rule to not preserve fabricated data, hardcoded scores, or gibberish from degenerated original trajectories
3. **Format correction prompt**: Added caveat to STYLE REMINDER about cleaning up fabricated content
4. **Debug flag**: Hardcoded `_CHORD_DEBUG_CORRECTIONS = True` (no more env var issues)

## Phase 5: Iterate on correction prompt until satisfactory [IN PROGRESS]

Loop between modifying the correction prompt and re-running the test. Key areas to watch:

- **Leaked future info**: `<think>` blocks referencing observations from later turns
- **Discouraged tool use**: corrections that remove or avoid web searches / database queries
- **API call loops**: corrections introducing `for gene in genes: api_call(gene)` patterns → should batch into single `advanced_web_search`
- **Over-specificity**: corrections that inject task-specific domain knowledge instead of fixing methodology
- **Fabricated data**: corrections preserving hardcoded scores/data from degenerated trajectories
- **Style drift**: corrections written as external reviewer rather than first-person agent

Once corrections are clean across the test trajectories, proceed to formal training.

## Phase 6: Prepare formal training config

Create a new launch script based on the existing full CHORD script:

- **Experiment name**: Descriptive, e.g. `biomni-qwen3-8b-rloo-full-chord-v1` (finalize with user)
- **Model**: SFT base (`/mnt/biomni_filestore/model_weights/qwen3-8b-sft-full-v1/global_step_104`)
- **resume_mode**: `none` (fresh optimizer, new experiment)
- **correction_loss_mu**: `0.8`
- **Production settings**: `BATCH_SIZE=16`, `NUM_TRAJ=5`, wandb logging, `SAVE_FREQ=8`
- **Tasks**: All tasks (remove any `TARGET_TASKS` filter if present)
- **Debug flag**: Set `_CHORD_DEBUG_CORRECTIONS = False` before launch

## Phase 7: Launch formal training

Use the `/launch-training` skill to:

1. Verify exec service health
2. Launch the new training script in tmux
3. Verify startup (Ray init, vLLM engines, first rollout begins)
4. Monitor initial steps for anomalies
