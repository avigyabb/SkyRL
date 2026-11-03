# throughput-push-v1: shared tracking doc

Several agents push to this branch. **Before starting an experiment or change, read this file
(`git pull` first) and add a row under "In progress" with your agent tag, then push.** When done, move
it to "Done" with numbers. Don't re-run an experiment listed here; build on its result.

Goal: RL end-to-end throughput / MFU on Megatron, general (model-agnostic) improvements first.
Secondary: 256k context, multi-turn RL, R3 (routing replay). Models: GLM-5.3, GLM-5.3-Flash,
DeepSeek-V4.1-Flash.

Cluster used: 2 x 8xB200 (Anyscale workspace; head node is CPU-only). agent-B: the Ray cluster it uses shows one 8xB200 GPU node,
so compare agent-A / agent-B numbers only when the setup is noted as the same.

## Already upstream (origin/main, don't redo)
- #2600 skip per-pass `empty_cache` under expandable segments
- #2598 skip frozen LM-head weight grad in fused logprob backward
- #2374 fused LM-head entropy for RL; #2373 autotuned fused LM-head Triton kernels
- GLM-5.3-Flash trainer-only kernel work (head-full DSA, k-pool indexer, mHC fusions, MoE dedup a2a, ...)
  lives on the unmerged branch `glm5p3-flash/sft-lora-throughput` (fork). Not in scope here unless noted.

## In progress
| agent | item | started |
|---|---|---|
| agent-A (claude, session 01Jj5NZC) | compressed colocated RL sim (setup below): baseline breakdown done; A/B of the colocation options below pending | 2026-11-02 |
| agent-A | R3 + `mlp_chunks_for_training>1` + `moe` recompute crash in GLM-5.3-Flash's chunked MoE (`mcore_ext/mhc_transformer_layer.py`): fix written, verifying grads on a 4-layer slice | 2026-11-02 |
| agent-A | GPU A/B of `trainer.old_logprobs_from_first_minibatch`, `generator.inference_engine.sleep_keep_weights`, `megatron_config.keep_frozen_weights_on_gpu` (all committed, opt-in, CPU-tested) | 2026-11-02 |
| agent-B (claude, session 01Jj5NZE) | compressed colocated RL sim for **DeepSeek-V4.1-Flash** (multi-turn, R3) + phase breakdown and general fixes it finds. Slice: 8 layers / 64 experts, real layer roles, no Engram / MTP, BF16 (`throughput_push_v1/sim/make_dsv41_flash_slice.py`); env `gsm8k_multi_turn`; vLLM `load_format=dummy` (`throughput_push_v1/sim/run_dsv41_flash_sim.sh`, not yet validated) | 2026-11-02 |

## Done
| agent | change / experiment | result | commit |
|---|---|---|---|
| agent-B | merged branch `rl-agent-throughput` here: GLM-5.3-Flash trainer kernels (formerly only on `glm5p3-flash/sft-lora-throughput`), all opt-in `SKYRL_*` flags; flags + per-change numbers in `skyrl/backends/skyrl_train/patches/megatron/README.md` and commit messages | GLM-5.3-Flash 64-expert slices, 1x8xB200: 256k LoRA SFT ~11.7k -> ~41.9k tok/s (TP4); 32k full FT ~1.0k -> ~5.2k tok/s/GPU BF16 (~6.0k FP8) vs SkyRL defaults | merge of `rl-agent-throughput` |
| agent-B | k-pool indexer reshaped to merge with upstream: `mcore_ext/dsa_kpool.py` = megatron-core `main` @ 631e1463 verbatim + #8299; SkyRL speedups in `mcore_ext/dsa_kpool_fast.py` at upstream's function seams | bitwise-identical selections vs before (synthetic, captured training call, TP4 sharded+striped); same speed | ff517e2e |
| agent-B | APEX-free gradient-accumulation fusion for native `Column/RowParallelLinear` (`SKYRL_GRAD_ACCUM_FUSION_TE_WGRAD=1`): `fused_weight_gradient_mlp_cuda` stand-in backed by megatron-core's own `_wgrad_gemm` (TE `general_gemm`); no upstream PR | exact (rel. err 4.7e-7 vs fp64, TP2+SP); neutral on GLM-5.3-Flash (big linears are TE); lets runs with native trainable linears use the fusion instead of erroring | c09a6f30 |
| agent-B | open upstream PRs from this work: NovaSky-AI/SkyRL#2601 (head-full DSA), #2606 (MoE dedup all-to-all, 1.8% of a 32k full-FT step) | -- | -- |
| agent-B | Harbor (agent RL) straggler cutoff, opt-in `generator.straggler_cutoff` | unit tests only; not measured live | 4028da84 |
| agent-B | vLLM decode layouts, GLM-5.3-Flash real ckpt, 8xB200, 16 concurrent: TP8 91 tok/s/GPU; DP8+EP+MTP3 204; **DP4xTP2+EP+MTP4 220** (2.4x); `max_num_seqs<=256` needed under DP+EP | measured | config only |
| agent-B | 32k full-FT recompute: `[core_attn,moe_act]` = no-MoE-recompute speed (-40 ms/layer vs `[core_attn,moe]`) for +4.4 GB/layer at 32k tok/GPU | measured | config only |
| agent-B | dead ends: TileLang DSA backend (2.5x slower than cuDNN at head-full), cuDNN DSA bwd `block_tile`, overlap_grad_reduce / overlap_param_gather (asserts in SkyRL loop), micro-batch 2 (no per-layer gain), Triton short-conv bwd, in-place LoRA add | -- | -- |

## Measurement setup
agent-A RL sim (`throughput_push_v1/sim/rl_sim_glm53_flash_lora.sh`): real GLM-5.3-Flash bf16 weights, LoRA r64
(merge_lora=false), colocated on 2x8 B200, Megatron TP4/EP16, vLLM 2 engines x TP8, R3 on, DAPO-17k prompts,
16 prompts x 8 samples (one mini-batch), 8k max response, `max_tokens_per_microbatch=8192`, recompute
`[core_attn,moe]`, `mlp_chunks_for_training=64`. Startup is ~25 min (cold weight load); a step is minutes.

Findings (agent-A, baseline, step 1 = 901 s):
- `generate` 170 s (128 seqs, mean 3.9k tokens), `fwd_logprobs_values_reward` **689 s**, `policy_train` crashed at 32 s.
- The 689 s is almost all the first backload of the LoRA-frozen base (~37 GB/GPU) from the file-backed offload
  (`SKYRL_FROZEN_OFFLOAD_DIR`); the forward itself is ~10 s of compute for ~520k tokens.
- `/data` does not exist on these nodes, so by default the frozen offload falls back to pinned RAM:
  ~300 GB/node on top of vLLM's level-1 sleep backup (~620 GB/node). Ray's memory monitor then kills the
  Megatron workers (node at 1.69 of 1.76 TB). Point `SKYRL_FROZEN_OFFLOAD_DIR` at local disk or keep frozen weights on GPU.
- R3 + `mlp_chunks_for_training=64` + `moe` recompute (the GLM-5.3-Flash recipe settings with R3 on) crashes in
  backward: `RuntimeError: Size does not match ... index [2099, 8] ... self [32, 288]` (`router_replay.get_replay_topk`).

