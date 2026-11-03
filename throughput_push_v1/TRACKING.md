# throughput-push-v1: shared tracking doc

Several agents push to this branch. **Before starting an experiment or change, read this file
(`git pull` first) and add a row under "In progress" with your agent tag, then push.** When done, move
it to "Done" with numbers. Don't re-run an experiment listed here; build on its result.

Goal: RL end-to-end throughput / MFU on Megatron, general (model-agnostic) improvements first.
Secondary: 256k context, multi-turn RL, R3 (routing replay). Models: GLM-5.3, GLM-5.3-Flash,
DeepSeek-V4.1-Flash.

Cluster used: 2 x 8xB200 (Anyscale workspace; head node is CPU-only).

## Already upstream (origin/main, don't redo)
- #2600 skip per-pass `empty_cache` under expandable segments
- #2598 skip frozen LM-head weight grad in fused logprob backward
- #2374 fused LM-head entropy for RL; #2373 autotuned fused LM-head Triton kernels
- GLM-5.3-Flash trainer-only kernel work (head-full DSA, k-pool indexer, mHC fusions, MoE dedup a2a, ...)
  lives on the unmerged branch `glm5p3-flash/sft-lora-throughput` (fork). Not in scope here unless noted.

## In progress
| agent | item | started |
|---|---|---|
| agent-A (claude, session 01Jj5NZC) | compressed colocated RL sim (GLM-5.3-Flash slice, vLLM dummy weights, multi-turn, R3) + phase breakdown | 2026-11-02 |

## Done
| agent | change / experiment | result | commit |
|---|---|---|---|

## Measurement setup
(filled in by agent-A once the simulator runs; reuse it so numbers are comparable)
