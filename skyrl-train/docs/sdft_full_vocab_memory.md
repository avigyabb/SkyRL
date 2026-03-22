# SDFT Full-Vocab KL: Memory Optimization Approaches

## Problem

The full-vocab SDFT KL estimator (`sdft_kl_estimator="full_vocab"`) computes `KL(student || teacher)` using the complete vocabulary distribution via TP-sharded logits. Each teacher logits tensor is `[1, num_actions, vocab_size/TP]` -- approximately **1.4 GB** per sequence (with vocab_size=151936, TP=2, num_actions~10K, bf16).

With `mini_batch_size=16` and `micro_train_batch_size_per_gpu=1`, all 16 teacher logits are pre-computed and attached to `micro_buffer` before `forward_backward_mini_batch` is called. This creates a **peak memory of 16 x 1.4 GB = 22.4 GB** on top of the existing training memory.

The `reinforce` estimator avoids this by only storing per-token log-probs (~80 KB per sequence), but has higher variance.

## Current Workaround

Use `sdft_kl_estimator="reinforce"` for production. Peak SDFT memory = 16 x 80 KB = 1.3 MB.

## Approach A: Lazy Computation Inside loss_func

Compute teacher logits on-demand inside `loss_func` (the closure in `forward_backward_mini_batch`) rather than pre-computing all of them.

**How it works:**
- Instead of attaching the full teacher logits tensor to each micro-buffer entry, attach only the lightweight teacher input data (prompt IDs, attention mask, num_actions -- ~KB).
- Inside `loss_func`, when the SDFT loss is needed for the current micro-batch, run the EMA teacher forward right there.
- The teacher logits tensor (~1.4 GB) exists only during that micro-batch's loss computation and is freed immediately after.

**Peak memory:** 1 x 1.4 GB (only the current micro-batch's teacher logits).

**Complexity:** Moderate (~30 lines). The main challenge is that `loss_func` doesn't have access to the EMA weights or the model's forward method -- these would need to be threaded through as closure variables or stored on `self`. The EMA weight swap (`_ema_weights()` context manager) would need to happen inside the Megatron pipeline scheduler, which may conflict with the scheduler's assumptions about model state.

**Key code location:** `megatron_model_wrapper.py`, `loss_func` closure inside `forward_backward_mini_batch` (line ~297).

## Approach C: Sub-Mini-Batch Splitting

Split the mini-batch into smaller sub-batches. For each sub-batch: compute teacher logits, run training forward-backward, then free teacher logits before the next sub-batch.

**How it works:**
```python
sub_batch_size = 2  # process 2 micro-batches at a time
for start in range(0, len(micro_buffer), sub_batch_size):
    sub_buffer = micro_buffer[start:start+sub_batch_size]
    
    # Compute teacher for just these 2
    for mb, tmb in zip(sub_buffer, teacher_micro_batches[start:...]):
        with torch.no_grad(), self._ema_weights():
            logits = self.model.forward_teacher_logits([tmb], ...)
        mb["sdft_teacher_logits"] = logits[0].detach()
    
    # Train on just these 2 (gradients accumulate across sub-batches)
    self.model.forward_backward_mini_batch(sub_buffer, ...)
    # data.pop() in loss_func frees teacher logits for these 2

# Single optimizer step after all sub-batches
```

**Peak memory:** sub_batch_size x 1.4 GB (e.g., 2.8 GB with sub_batch_size=2).

**Complexity:** Moderate (~50 lines). Gradients accumulate correctly across sub-batches because the optimizer step happens after all sub-batches. The main challenge is integrating with the existing `micro_batches_per_mini_batch` logic in `megatron_worker.py` -- the current code uses this to control when to call `forward_backward_mini_batch` and `optimizer_step`. Splitting further requires careful handling of:
- The existing gradient accumulation counter
- The `zero_grad_buffer` calls
- The metrics aggregation across sub-batches
- The `data.pop("sdft_teacher_logits")` in loss_func to free each tensor after use

**Key code location:** `megatron_worker.py`, `ppo_train` method (line ~420), the `if len(micro_buffer) == micro_batches_per_mini_batch:` block.

## Comparison

| Approach | Peak SDFT Memory | Code Complexity | Risk |
|----------|-----------------|-----------------|------|
| Reinforce estimator | 1.3 MB | None (config change) | Higher variance KL gradient |
| Approach A (lazy) | 1.4 GB | Moderate | EMA swap inside pipeline scheduler |
| Approach C (sub-batch) | N x 1.4 GB | Moderate | Gradient accumulation integration |
| Current full_vocab | 22.4 GB | Already implemented | OOM on production batch sizes |
