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
