---
description: Biomni CodeAct agent loop, reward system (gt + rubric + format), and format validation rules. Consult when debugging reward issues, agent behavior, or format compliance.
globs: skyrl-agent/**
---

# Agent & Reward System

## Biomni CodeAct Agent

**File**: `skyrl_agent/agents/biomni_codeact/biomni_codeact_agent.py`

The agent runs a multi-turn loop:
1. Generate assistant response (via vLLM)
2. Extract code from `<execute>` blocks
3. Send code to Biomni runtime server (`http://10.138.0.3:8000`) for execution
4. Append observation (stdout/stderr) to conversation
5. Repeat until agent produces `<solution>` block or hits limits

**Limits**:
- Max iterations: 50 (from YAML `generator.max_iterations`)
- Max prompt length: 40960 tokens (from YAML `generator.max_prompt_length`)
- Max response length: 4096 tokens per generation (from YAML `generator.sampling_params.max_tokens`)

### Context Overflow Sentinel

When `len(input_ids) > max_prompt_len`, `_llm_generate()` returns `_CONTEXT_OVERFLOW_SENTINEL` (a string constant, NOT model output). The `run()` loop appends a `[CONTEXT_LIMIT]` message and breaks cleanly. This is graceful — does not cause OOM.

## Reward Structure

**File**: `skyrl_agent/tasks/biomni_rubric_reward_adapter.py`

```
total_reward = gt_reward + rubric_reward + ft_reward    (max 7)
```

| Component | Range | Source |
|-----------|-------|--------|
| `gt_reward` | 0 or 1 | Task-specific ground truth check (`task.reward()`) |
| `rubric_reward` | 0–5 | LLM critic (Claude) evaluation, normalized from 0–50 scale |
| `ft_reward` | 0 or 1 | Format validation via `_valid_block()` |

### LLM Critic

- Uses `ChatAnthropic` (Claude) with structured output (`CriticMetrics`)
- Rubric dimensions: output_grading (0–20), methodology_knowhow (0–10), code_data_handling (0–10), reasoning_coherence (0–10)
- Total raw score 0–50, normalized to 0–5 for reward

### Task Mapping

Tasks are registered in `_task_mapping` from modules in `skyrl_agent/agents/biomni_codeact/task/`:
- `screen_design`
- `screen_gene_retrieval`
- `crispr_delivery`
- `rare_disease_diagnosis`
- `gwas_causal_gene`
- `gwas_variant_prioritization`
- `patient_gene_detection`
- `lab_bench`

## Format Validation Rules

**File**: `biomni_rubric_reward_adapter.py`, function `_validate_format._valid_block` (~line 202)

Every assistant message is checked. If ANY message fails, `ft_reward = 0` for the entire trajectory.

| Rule | Check | Failure Log Pattern |
|------|-------|-------------------|
| 1 | Starts with `<think>` | `"not start with <think>"` |
| 2 | Exactly one `<think>` and one `</think>` | `"not exactly one <think>"` |
| 3 | Ends with `</execute>` or `</solution>` | `"not end with </execute> or </solution>"` |
| 4 | At least one action block after `</think>` | `"no action tag after </think>"` |
| 5 | Ending tag matches the outer (first) action type | `"outer is <execute> but doesn't end with"` |
| 6 | Only ONE outer action block (nested OK, sequential not) | `"multiple outer.*blocks"` |
| 7 | Last message → `<solution>` outer; others → `<execute>` outer | `"is_last but outer is <execute>"` / `"not is_last but outer is <solution>"` |
| 8 | No `<think>`/`</think>` after the think block | `"<think> or </think> in after_think"` |

**Rule 2 and random token corruption**: Rule 2 failures ("not exactly one `<think>` and one `</think>`") are often caused by the model producing a random token (e.g., a CJK character) in place of the `</think>` closing tag. This was a known issue on A100 GPUs due to inference numerical inaccuracies; H200 GPUs should reduce the frequency. Track Rule 2 failure count per batch over training iterations — increasing frequency may indicate inference precision issues or the model learning malformed patterns.

## Agent Configuration (Current YAML)

**File**: `examples/run_biomni/biomni_codeact_rubric_rl_qwen30ba3b_gspo.yaml`

| Setting | Value | Notes |
|---------|-------|-------|
| `num_trajectories` | 5 | Rollouts per prompt |
| `max_iterations` | 50 | Agent loop iterations |
| `max_prompt_length` | 40960 | Context window budget |
| `max_tokens` | 4096 | Per-generation cap |
| `temperature` | 1.0 | Sampling temperature |
| `top_p` | 1.0 | Nucleus sampling (effectively disabled) |
| `log_heavy_freq` | 8 | Steps between detailed sample logs |
| `overlong_filter_threshold` | 40000 | Token threshold for overlong filter |
| `qwen3_enable_thinking` | true | Enable `<think>` mode |
| `qwen3_acc_thinking` | true | Accumulate thinking tokens |
| `remove_think_tokens` | false | Keep think tokens in training data |
| `max_parallel_agents` | 128 | Concurrent rollouts |

## Runtime Client & Code Execution

**File**: `biomni_codeact_agent.py`, class `BiomniRuntimeClient` (~line 131)

- Executes code via HTTP POST to `http://10.138.0.3:8000/execute`
- Default execution timeout: 600s per code block (configurable)
- HTTP request timeout: execution timeout + 5s
- On timeout: raises `RuntimeError("Code execution timed out...")`, caught by the agent loop and wrapped as `[runtime-error] ...`
- On session not found (404): automatically creates a new session and retries
- Observations are wrapped as `<observation>output</observation>` and appended as user messages
- The `[runtime-error]` prefix in observations is a catch-all for any exception during `execute()`, including timeouts — so grepping `runtime-error` double-counts timeout errors
- To monitor runtime health, read actual observation blocks from the log and look for any error patterns. Record new patterns as they're discovered (see monitor-training skill "Known Runtime Error Patterns").

## Trajectory Processing

**File**: `skyrl_agent/agents/biomni_codeact/biomni_codeact_runner.py`

`BiomniCodeActTrajectory` extends `BaseTrajectory`. It:
1. Converts dataset prompts to structured messages via `_messages_to_prompt()`
2. Uses the shared `gen_chat_template` (from `biomni_qwen3.jinja`) for both generation and post-processing tokenization
3. This template consistency is critical — if the jinja template is modified, both rollout and training tokenization change
