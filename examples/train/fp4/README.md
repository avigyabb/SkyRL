# NVFP4 RL training (experimental)

`run_nvfp4_blackwell_qwen35_9b.sh` and `run_nvfp4_blackwell_qwen35_35b_a3b.sh` train Qwen3.5 with DAPO on
8xB200 using Transformer Engine's NVFP4 recipe for the Megatron linear-layer GEMMs
(`megatron_config.fp4=e2m1`, `fp4_recipe=nvfp4`). Primary weights stay BF16.

**Bottom line (measured on 8xB200, TE 2.19, vLLM 0.30):** NVFP4 training *works* -- it tracks the BF16 and MXFP8
reward curves when token-level TIS is on -- but it does **not** beat MXFP8 on speed at these model sizes, and its
rollout-vs-trainer logprob gap is 4-5x larger. Use it to study FP4 RL, not (yet) to save time.

## Required settings

| Setting | Why |
| --- | --- |
| `tensor_model_parallel_size=1` | At TP=2 the quantized sequence-parallel path made `policy_train` ~3x slower (57-73 s vs 18 s on 9B). |
| `NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1` | 1D 1x16 weight scales (the finer layout). ~25% lower train/rollout gap, same speed. |
| `trainer.algorithm.off_policy_correction.tis_ratio_type=token` | Without it a 40-step 9B run learned for 20 steps and then regressed (see below). |
| Blackwell (SM100+) | TE's NVFP4 GEMMs. `fp4` and `fp8` are mutually exclusive; `fp4_param=true` is rejected. |

`NVTE_NVFP4_*` and `NVTE_BACKWARD_OVERRIDE` are forwarded from the launching shell to every Ray worker.
Sequence packing pads each sub-sequence to 32 tokens per TP/CP shard (Megatron's `get_fp4_align_size`).

## Results

Qwen3.5-9B-Base, DAPO, 8xB200, colocated, TP=1, 40 steps, one seed each (treat gaps below ~0.05 as noise).
Mean training reward per 10-step block:

| trainer / rollout | s1-10 | s11-20 | s21-30 | s31-40 | rollout-trainer logprob gap |
| --- | --- | --- | --- | --- | --- |
| BF16 / BF16 | -0.644 | -0.399 | -0.245 | -0.177 | 0.006 |
| MXFP8 / BF16 | -0.651 | -0.465 | -0.287 | -0.253 | 0.018 |
| NVFP4 / BF16, no TIS | -0.655 | -0.387 | -0.334 | **-0.499** | 0.064 |
| NVFP4 / BF16, token TIS | -0.627 | -0.421 | -0.253 | -0.203 | 0.067 |

Steady-state step timing (steps 2-4, seconds). Training is a minority of each step: generation is ~55 s.

| 9B | `policy_train` | peak mem (GB) |
| --- | --- | --- |
| BF16, TP=2 | ~25-45 (noisy) | 57 |
| MXFP8, TP=1 | 18.4 | 115 |
| NVFP4, TP=1 | 17.8 | 109 |
| MXFP8, TP=2 | 23 | 64 |
| NVFP4, TP=2 | 57-73 | 61 |

(BF16 was not run at TP=1 with micro-batch 2; at micro-batch 4 it was ~32-48 s and 138 GB, against MXFP8 24-40 s / 150 GB and NVFP4 17-35 s / 145 GB.)

Micro-batch 4, disabling RHT/stochastic rounding, and keeping the first/last layers in BF16 did not change speed or
the gap. 35B-A3B (TP=1, EP=8): NVFP4 `policy_train` ~38-70 s vs MXFP8 ~39-50 s, and generation is slower (~95 s vs
~80 s) because the MXFP8 recipe also serves an FP8 rollout.

## NVFP4 rollout weight sync (`fp8_weight_sync_mode=nvfp4`, opt-in)

Serves the rollout from the trainer's own NVFP4 weights (vLLM compressed-tensors, weight-only W4A16), quantized with
TE's quantizer so rollout weights equal the trainer's GEMM operands. Fused projections (q/k/v, gate/up, the GDN
in-projections) share one global scale, as vLLM requires. Dense Qwen3.5 only; MoE experts raise `NotImplementedError`.

* It halves the gap (0.065 -> 0.035) but the rollout policy is the quantized model, so rewards start lower
  (about -0.8 vs -0.6 at step 1).
* It is **not faster**: weight-only NVFP4 decodes ~14% faster than BF16 in isolation (10.3k -> 11.8k tok/s on
  Qwen3.5-9B, 64 concurrent), but in RL generation stays ~55 s and each weight sync costs ~5 s more than BF16.
* `SKYRL_NVFP4_INPUT_AMAX=<float>` (experimental) adds a static activation scale for W4A4 serving. Generation was
  ~10% faster than BF16 (44-57 s) with the same gap, but the value is not calibrated: `64` and `256` behave alike,
  `16` overflows (gap 0.15, reward -1.7). Do not use it without calibrating per model.

## Not covered

No NVFP4 parameter storage (`fp4_param`), no NVFP4 MoE weight sync, no activation calibration, no run longer than
40 steps, one seed per row.
