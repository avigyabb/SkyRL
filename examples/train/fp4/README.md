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
in-projections) share one global scale, as vLLM requires. Routed experts are quantized per expert (gate and up
jointly, since vLLM keeps one global scale per expert for the fused `w13`) and shipped batched over the expert
dimension; the per-expert global scales are loaded one expert at a time because vLLM's per-tensor scale loader
indexes a single expert. Qwen3.5 dense and MoE.

* It halves the gap (0.065 -> 0.035) but the rollout policy is the quantized model, so rewards start lower
  (about -0.8 vs -0.6 at step 1).
* It is **not faster**: weight-only NVFP4 decodes ~14% faster than BF16 in isolation (10.3k -> 11.8k tok/s on
  Qwen3.5-9B, 64 concurrent), but in RL generation stays ~55 s and each weight sync costs ~5 s more than BF16.
* `SKYRL_NVFP4_INPUT_AMAX=<float>` (experimental) adds a static activation scale for W4A4 serving. Generation was
  ~10% faster than BF16 (44-57 s) with the same gap, but the value is not calibrated: `64` and `256` behave alike,
  `16` overflows (gap 0.15, reward -1.7). Do not use it without calibrating per model.
* `SKYRL_NVFP4_CALIBRATE_INPUT=1` (experimental, dense linears, Megatron, PP=1) replaces that constant with a
  per-layer amax the trainer measures on its own forward passes (forward pre-hooks on the TE linears, norm replayed
  for the fused `LayerNormLinear`, max-reduced across ranks, mapped to HF weight names through the bridge). Each sync
  ships one `input_global_scale` per layer; the first sync, before any data, uses `SKYRL_NVFP4_INPUT_AMAX`
  (default 128). Knobs: `SKYRL_NVFP4_CALIB_DECAY` (0.9, how slowly a stale peak is forgotten),
  `SKYRL_NVFP4_CALIB_MARGIN` (1.0). `SKYRL_NVFP4_CALIB_DUMP=<dir>` writes the measured table as JSON.
  Routed-expert inputs are not calibrated yet (they still ship 1.0).

### 35B-A3B (MoE), TP=1, EP=8, 4 steps, one seed

NVFP4 trainer, BF16 rollout vs NVFP4 wire (steady-state steps 2-4):

| rollout | logprob gap | `generate` (s) | `sync_weights` (s) | reward s1-s4 |
| --- | --- | --- | --- | --- |
| BF16 | 0.062-0.076 | 89-101 | ~17 | -0.446 -0.690 -0.702 -0.385 |
| NVFP4 wire | 0.037-0.049 | 75-86 | ~29 | -0.461 -0.611 -0.723 -0.432 |

Generation is ~15% faster and the gap is about halved, but the sync costs ~12 s more per step, so a step is not
faster overall. The per-expert TE casts account for only ~3 s of that; I did not profile the rest. A vectorized torch cast was tried and dropped: it differs from TE on ~0.1% of codes and scales
and saved almost nothing.

### Disaggregated: 8 training GPUs + 4 rollout GPUs (35B-A3B, 4 steps, one seed)

`COLOCATE_ALL=false`, EP=8 on the trainer, `num_engines=4`. Here generation is the bottleneck, so the faster NVFP4
decode outweighs the extra sync. Steady state (steps 3-4):

| rollout | `generate` (s) | `sync_weights` (s) | `step` (s) | logprob gap |
| --- | --- | --- | --- | --- |
| BF16 | 118-119 | 6.4-7.7 | 174-175 | 0.052-0.065 |
| NVFP4 wire | 98 | 16.3-16.9 | 162-164 | 0.038-0.044 |

About 7% faster per step. Trainer memory is unchanged (peak ~130 GB), and squeezing the 35B trainer onto 4 GPUs
(EP=4) ran out of memory in `optim_step` (165 GB): NVFP4 does not shrink training state, so the trainer GPU count
is set by the BF16 weights and optimizer, not by the rollout format.

### Rollout engine capacity

vLLM engine on one B200, Qwen3.5-35B-A3B-Base, `gpu_memory_utilization=0.9`, 8k context (dummy weights; only the
footprint matters):

| rollout weights | weight memory | KV cache | KV tokens |
| --- | --- | --- | --- |
| BF16 | 64.7 GiB | 94 GiB | 2.74M |
| NVFP4 | 19.7 GiB (3.3x smaller) | 138 GiB | 4.0M (+47%) |

The training side gets no such saving: primary weights, FP32 masters and optimizer state stay as they were.

## Not covered

No NVFP4 parameter storage (`fp4_param`), no activation calibration, only 4 steps of any 35B run, no run longer than
40 steps, one seed per row.


### W4A4 activation scales: global constant vs trainer-calibrated (Qwen3.5-9B, 6 steps, 3 seeds)

`run_nvfp4_w4a4_calibration_qwen35_9b.sh` (`ARM=global|calibrated|w4a16`). Measured per-layer amax spans ~2 to ~350
(median ~47); the largest are the late-layer MLP down-projections (layer 31: 346), which a single constant of 64 clips.

| arm | seeds | mean logprob gap | mean worst-token gap | `generate` (s) | mean reward |
| --- | --- | --- | --- | --- | --- |
| global amax 64 | 42, 1, 2 | 0.0390 | 14.5 | 48.7 | -0.93 |
| calibrated | 42, 1, 2 | 0.0376 | 8.3 | 48.7 | -0.90 |

* The worst-token gap is about halved in all three seeds (7.3 / 9.2 / 8.5 vs 15.2 / 14.2 / 14.2).
* The mean gap is lower in every seed but by only ~3.6% (paired differences 0.003 / 0.0003 / 0.0009), which is within
  noise. Generation time and reward do not separate. Do not read this as a speed or reward win.
* Margin sweep (seed 42, one run each): 0.7 clips and the gap more than doubles (0.091); 1.0 gives 0.040; 1.5 gives
  0.041 but the worst-token gap returns to the global arm's (14.4). Keep the margin at 1.0.
