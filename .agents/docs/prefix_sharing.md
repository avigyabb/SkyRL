# Prefix-shared training forward (Megatron)

`trainer.prefix_sharing=true` makes the Megatron policy/ref forward compute each shared token
prefix in a micro-batch once. GRPO replicates every prompt `n_samples_per_prompt` times and
step-wise agentic trajectories replicate long histories; in the standard THD packing path every
replica is a separate packed sequence and the prompt goes through every layer once per replica.

## What it does

1. **Layout** (`distributed/megatron/prefix_sharing.py::build_prefix_shared_layout`, CPU): rows of a
   micro-batch are inserted into a radix tree over their valid tokens and emitted in DFS order as a
   single packed token stream with explicit per-token positions. At a branch point the shared
   segment ends one token early and each child starts with its own copy of the branching token, so
   every packed position has exactly one next-token target. `row_to_packed[b, c]` says which packed
   position predicts `sequences[b, c+1]`. `prefix_sharing_min_shared_tokens` stops the tree from
   splitting segments over a handful of tokens two responses happen to share.
2. **Attention** (`prefix_shared_attention`): a token attends causally to its own segment and fully to
   all ancestor segments. It runs as two flash-attn varlen calls, causal within segments and
   non-causal branch -> gathered ancestor K/V, merged through the log-sum-exp. The backward feeds the
   merged output/LSE to both flash-attn backward kernels, which makes `P_ij = exp(s_ij - lse_i)`
   the true probability over the union of both regions; dK/dV of the ancestors accumulate over all
   branches with `index_add_`. No new kernel is involved.
3. **Patches** (`install_prefix_sharing_patches`): `TEDotProductAttention.forward` routes to the
   attention op when `packed_seq_params` carries a layout; Megatron's THD `apply_rotary_pos_emb`
   gathers frequencies per token from `cu_seqlens.prefix_position_ids` (an attribute on the tensor
   Megatron already passes, so it survives activation recompute).
4. **Loss** (`megatron_model_wrapper.py`): per-token log-probs and entropy are computed in the packed
   layout and scattered to `[B, S-1]` through `row_to_packed`; a shared position feeds several rows
   and autograd sums their gradients into it, which is exactly the per-replica sum.

## Requirements and fallbacks

Megatron strategy, `remove_microbatch_padding=true`, `context_parallel_size=1`, no
`moe_enable_routing_replay` (checked in `validate_cfg`). Micro-batches with controller-side
`sub_seq_lengths`, MTP draft loss, VLM inputs or router replay silently use plain THD packing.
Rows only share inside a micro-batch, so `micro_train_batch_size_per_gpu` and
`micro_forward_batch_size_per_gpu` should be a multiple of `generator.n_samples_per_prompt`
(rows arrive grouped by prompt; `MeshDispatch` shards contiguously).

## Numerics contract (IsoExec-style entry)

* Everything outside attention is the same per-token computation as unshared THD packing.
* Branch tokens' attention output is the fp32 LSE-weighted merge of two bf16 flash-attn partial
  results, rounded once to bf16, instead of one flash-attn result. Measured on Qwen2.5-1.5B and
  Qwen3-30B-A3B with random tokens (`|logprob| ~ 12`): mean |delta logprob| ~ 0.02, p99 ~ 0.08,
  max ~ 0.2 (about 1 bf16 ulp of the logits); repacking the same rows into different micro-batches
  gives 0. Gradient norms agree to about 1 percent at these batch sizes. Kernel-level parity tests
  (`tests/backends/skyrl_train/gpu/gpu_ci/test_prefix_sharing.py`) compare forward and backward
  against flash-attn on the replicated rows.
* Training/inference log-prob matching therefore inherits this bf16-level deviation on top of the
  usual sampler/trainer kernel mismatch; record `prefix_sharing` in any execution contract that
  tracks trainer-side kernels.

## Metrics

`prefix_sharing/token_ratio` (packed tokens / unshared tokens the trunk would have processed) and
`prefix_sharing/branches` per micro-batch.

## Benchmarks

* `skyrl/benchmarks/bench_prefix_attention.py`: single GPU, attention op vs flash-attn on the
  replicated rows.
* `skyrl/benchmarks/bench_prefix_sharing.py`: Ray + Megatron worker A/B on one loaded model
  (`--check` compares log-probs and grad norms with `lr=0`; timing of `forward_backward` and
  `forward`). Results for Qwen3-30B-A3B are in the PR / X-post write-up.

## Results (2026-09-23, Qwen3-30B-A3B-Base, 8xH100 80GB, TP=8 / EP=8 / ETP=1, bf16, full recompute)

One mini-batch of 2 prompts x G samples, synthetic random-token rows with P prompt and R response
tokens. Baseline: unshared THD packing with the largest micro-batch within 64k tokens. Shared: two
prompt groups per micro-batch for G=8, one for G=16. Best of 3 after warmup; optimizer step excluded.

| P / R / G | rows | trunk tokens unshared -> shared | fwd+bwd baseline | fwd+bwd shared | speedup | fwd baseline | fwd shared | speedup |
|---|---|---|---|---|---|---|---|---|
| 1024 / 2048 / 8 | 16 | 49,152 -> 34,830 (0.71x) | 2.75 s | 2.08 s | 1.32x | 0.84 s | 0.67 s | 1.26x |
| 4096 / 1024 / 8 | 16 | 81,920 -> 24,590 (0.30x) | 5.26 s | 1.58 s | 3.33x | 1.49 s | 0.51 s | 2.90x |
| 4096 / 2048 / 16 | 32 | 196,608 -> 73,758 (0.38x) | 13.66 s | 5.27 s | 2.59x | 2.89 s | 1.23 s | 2.36x |
| 8192 / 1024 / 16 | 32 | 294,912 -> 49,182 (0.17x) | 20.38 s | 3.79 s | 5.38x | 3.97 s | 0.94 s | 4.20x |
| 16384 / 512 / 16 | 32 | 540,672 -> 49,182 (0.09x) | 43.02 s | 4.68 s | 9.20x | 7.45 s | 1.03 s | 7.24x |

Numerics on the same weights (lr=0): shared-vs-unshared per-token log-prob deviation mean 0.05 /
p99 0.25 on |logprob| ~ 12, identical to re-running the unshared path with a different micro-batch
split on this MoE (routing shifts with the token set). Grad norms agree to 0.01-2.7 percent, within
the same repacking control. On dense Qwen2.5-1.5B (repacking is exactly deterministic) the shared
path deviates by 0.021 mean, the same as switching the baseline attention kernel from flash-attn
to cuDNN (0.022), and grad norms agree to 0.016 percent.

Attention op alone (1 H100, Qwen3-30B-A3B TP=8 head shard, fwd+bwd vs flash-attn on replicated
rows): 0.74x at 1k/2k/8, 1.36x at 4k/1k/8, 1.27x at 4k/2k/16, 2.49x at 8k/1k/16, 5.35x at 16k/512/16.
