"""Triton row kernels for the fused LM-head log-prob (``SKYRL_FUSED_LOGPROB_TRITON=1``).

The stock chunk loop upcasts each ``[rows, V//TP]`` logits chunk to fp32 and runs max, subtract,
exp, sum and gather as separate passes, then recomputes the whole log-softmax (with its
all-reduces) in backward. Here:

- forward reads the chunk once: per-row local max, sum of ``exp(x - max)`` and the target logit;
  the cross-rank combine runs on ``[rows]`` tensors only, and the global log-sum-exp is saved;
- backward writes ``(one_hot(target) - exp(x - lse)) * grad`` in the logits dtype in one pass
  over the recomputed logits, with no collectives.

Same math as ``_compute_distributed_log_softmax`` up to summation order.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_stats_kernel(
    logits_ptr, target_ptr, max_ptr, sumexp_ptr, chosen_ptr, n_cols, stride_row, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride_row
    offs = tl.arange(0, BLOCK)
    m = tl.full([BLOCK], float("-inf"), tl.float32)
    s = tl.zeros([BLOCK], tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + offs
        x = tl.load(base + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, x)
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_new))
        s = s * alpha + tl.where(x == float("-inf"), 0.0, tl.exp(x - m_new))
        m = m_new
    row_max = tl.max(m, axis=0)
    row_sum = tl.sum(tl.where(m == float("-inf"), 0.0, s * tl.exp(m - row_max)), axis=0)
    t = tl.load(target_ptr + row)
    tl.store(max_ptr + row, row_max)
    tl.store(sumexp_ptr + row, row_sum)
    tl.store(chosen_ptr + row, tl.load(base + t).to(tl.float32))


@triton.jit
def _grad_logits_kernel(
    logits_ptr, out_ptr, lse_ptr, grad_ptr, target_ptr, valid_ptr, n_cols, stride_row, stride_out,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lse = tl.load(lse_ptr + row)
    g = tl.load(grad_ptr + row).to(tl.float32)
    t = tl.load(target_ptr + row)
    valid = tl.load(valid_ptr + row)
    offs = tl.arange(0, BLOCK)
    for start in range(0, n_cols, BLOCK):
        cols = start + offs
        mask = cols < n_cols
        x = tl.load(logits_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
        grad = -tl.exp(x - lse) * g
        grad = tl.where((cols == t) & valid, grad + g, grad)
        tl.store(out_ptr + row * stride_out + cols, grad.to(out_ptr.dtype.element_ty), mask=mask)


_BLOCK = 2048


def row_stats(logits: torch.Tensor, local_target: torch.Tensor):
    """Per row of ``logits [rows, V]``: (max, sum exp(x - max), x[target]) in fp32."""
    rows, n_cols = logits.shape
    out = torch.empty((3, rows), dtype=torch.float32, device=logits.device)
    _row_stats_kernel[(rows,)](
        logits, local_target, out[0], out[1], out[2], n_cols, logits.stride(0), BLOCK=_BLOCK, num_warps=8
    )
    return out[0], out[1], out[2]


def grad_logits_(
    logits: torch.Tensor,
    lse: torch.Tensor,
    grad: torch.Tensor,
    local_target: torch.Tensor,
    valid: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """``out = (one_hot(target) * valid - exp(logits - lse)) * grad``; ``out`` may alias ``logits``."""
    rows, n_cols = logits.shape
    _grad_logits_kernel[(rows,)](
        logits, out, lse, grad, local_target, valid, n_cols, logits.stride(0), out.stride(0), BLOCK=_BLOCK,
        num_warps=8,
    )
    return out
