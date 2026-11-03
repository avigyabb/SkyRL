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
| agent-A | `sleep_keep_weights` at 256k (trainer peak there is 84 GB, so the engine's 78 GB of weights may fit beside it) | 2026-11-03 |
| agent-B (claude, session 01Jj5NZE) | compressed colocated RL sim for **DeepSeek-V4.1-Flash** (multi-turn, R3) + phase breakdown and general fixes it finds. Slice: 8 layers / 64 experts, real layer roles, no Engram / MTP, BF16 (`throughput_push_v1/sim/make_dsv41_flash_slice.py`); env `gsm8k_multi_turn`; vLLM `load_format=dummy` (`throughput_push_v1/sim/run_dsv41_flash_sim.sh`, not yet validated) | 2026-11-02 |

## Done
| agent | change / experiment | result | commit |
|---|---|---|---|
| agent-A | cost of R3: opt2 with R3 off (same batches) | steps 2/3 218 / 251 s vs 225 / 259 with R3; generation unchanged (156 / 172 vs 156 / 174); training fwd/bwd 39 / 53 vs 43 / 59 s (R3 replay ~10% of training, ~3% of the step). Rollout/train logprob diff 0.018 without R3 vs 0.003 with | -- |
| agent-A | idea, not done: LoRA adapter sync is ~18 s/step; memory mode spends 10.9 s in vLLM building the adapter from 73k tensors (per-expert keys) and 7.0 s sending; disk mode now logs its breakdown (8b850d0d) | -- | -- |
| agent-A | **256k-context RL** (`throughput_push_v1/sim/long256.sh`: opt2 config, `longprompt_262144`, 4 prompts x 4 samples of ~262k tokens, 1k responses, R3 on) | with selective recompute `[core_attn,moe_act]` the first 262k micro-batch dies in an MoE collective (`ncclUnhandledCudaError`, with or without `SKYRL_MOE_DEDUP_A2A`, 4 or 16 MLP chunks: OOM inside NCCL). With **full recompute** (next row) + 16 MLP chunks: step 2 **149 s** (generate 66, fwd/bwd 54 for 4.2M tokens = **4527 tok/s/GPU**, sync 19), peak 84 GB, no padding micro-batches | config + 69dad0c5 |
| agent-A | full activation recompute for GLM-5.3-Flash: SkyRL used to downgrade `recompute_granularity=full` to selective for mHC models. Now honored (model built without recompute, config switched to full after build) | slice grads: 0.7168 / 0.7141 (full, 1 / 64 chunks, R3) vs 0.7182 none; makes 256k RL fit (row above) | 69dad0c5 |
| agent-A | ablation: opt1b config with `mlp_chunks_for_training=64` | step 1 forward/backward **1142 s** vs 141 s with 1 chunk (opt1b, incl. one-time backload): chunking at 8k is most of the training speedup, DP balancing the rest. GLM-5.3-Flash recipes still set 64; startup warning added | warning commit below |
| agent-A | **opt3** = opt2 + `max_tokens_per_microbatch=16384` | steps 2/3: **218 / 255 s** (vs opt2 225 / 259); train 37 / 58 s (vs 43 / 60); 785 / 699 tok/s/GPU; peak 127-130 GB | config |
| agent-A | **opt2** = opt1b + agent-B's GLM-5.3-Flash kernel flags (`throughput_push_v1/sim/opt2.sh`: head-full DSA, cuDNN DSA, fused k-pool indexer, mHC/KDA fusions, MoE dedup a2a, TF32 override), BF16, recompute `[core_attn,moe_act]`, `moe_permute_fusion`, `bias_activation_fusion`, `lora.sync_mode=memory` | steps 2/3: **225 / 259 s** (vs opt1b 263 / 303, baseline 2103); train 43 / 60 s (vs 81 / 111); ~705 tok/s/GPU; peak 90 GB (vs 111). Generation 156-174 s is now ~70% of the step. `sync_mode=memory` 17-19 s, no faster than disk here (export 4.9 s, send 7.0 s, vLLM register 10.9 s) | config |
| agent-A | torch.profiler, opt1b training step (rank 0, 64 seqs): GPU busy 69%; elementwise 34% (stock DSA masking), FP32 SIMT sgemm 19% (k-pool indexer einsum on CUDA cores), MoE SendRecv 19%, SP all-gather 8.5%; optimizer step 0.18 s | motivated opt2 | -- |
| agent-A | **opt1b** = baseline + `trainer.old_logprobs_from_first_minibatch` + `trainer.balance_dp_tokens` + `megatron_config.keep_frozen_weights_on_gpu` + `mlp_chunks_for_training=1` + vLLM `gpu_memory_utilization=0.5` (KV peaks at 6.6% of 44 GB here; at 0.65 the resident frozen weights OOM the engine wake). `sleep_keep_weights` does not fit here (trainer peak 111 GB + engine weights 78 GB) | steps 1-3: **331 / 263 / 303 s** vs baseline 2502 / 2103 s; train 141 / 81 / 111 s vs 1751 / 1437; logprob fwd 0 vs 483-559; backload 0.05 s vs 42-46; 376-391 tok/s/GPU vs 19-22; train/rollout logprob diff 0.003-0.005 | config (`throughput_push_v1/sim/opt1.sh`) |
| agent-A | R3 + `mlp_chunks_for_training>1` + `moe` recompute crashed in backward (GLM-5.3-Flash mHC `chunked_mlp`): each chunk is checkpointed by the MoE layer and recompute popped the whole micro-batch's routes. Fixed: `chunked_mlp` checkpoints chunks itself with routes bound to the call | slice repro: grad norm 0.7159 (fixed, 64 chunks + recompute) vs 0.7170/0.7161 no recompute, 0.7154 unchunked; old code raises | ffa706b8 |
| agent-A | timers `timing/colocate_<model>_{backload,offload}`, `timing/inference_engine_sleep` | -- | a462bc7c |
| agent-A | baseline sim (2 steps) | step 2: **2103 s** = generate 157 + engine sleep 3.5 + logprob fwd 483 (backload 42) + train 1437 + sync 20; 21.7 tok/s/GPU; trainer peak 111 GB; DP token imbalance 60.5k vs 142.8k tokens per rank | -- |
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
- R3 + `mlp_chunks_for_training=64` + `moe` recompute (the GLM-5.3-Flash recipe settings with R3 on) crashed in
  backward: `RuntimeError: Size does not match ... index [2099, 8] ... self [32, 288]` (fixed in ffa706b8).
- During training GPUs draw ~265 W of 1000 W: latency-bound. Suspects: DP token imbalance (fixed by
  `trainer.balance_dp_tokens`, 58ee4b48) and `mlp_chunks_for_training=64` at 8k (64 cross-node all-to-alls of ~32
  tokens per MoE layer per micro-batch).
- Opt-in options added (all committed): `trainer.old_logprobs_from_first_minibatch` (04add505),
  `trainer.balance_dp_tokens` (58ee4b48), `generator.inference_engine.sleep_keep_weights` and
  `megatron_config.keep_frozen_weights_on_gpu` (c417c3b9).

