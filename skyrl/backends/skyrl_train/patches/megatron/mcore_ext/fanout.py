"""Hand one tensor to several consumers and sum their gradients in a single pass.

When a tensor feeds ``n`` autograd consumers, the engine accumulates its gradient with ``n - 1``
pairwise adds, each reading two full tensors and writing one (``3 (n - 1)`` passes over memory).
:func:`fanout` returns ``n`` aliases of the input, one per consumer; its backward sums the ``n``
gradients with one Triton kernel (``n + 1`` passes). Values are the same up to summation order.
"""

import torch
import triton
import triton.language as tl

_MAX = 8


@triton.jit
def _sum_kernel(out, p0, p1, p2, p3, p4, p5, p6, p7, numel, N: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    acc = tl.load(p0 + offs, mask=mask).to(tl.float32)
    if N > 1:
        acc += tl.load(p1 + offs, mask=mask).to(tl.float32)
    if N > 2:
        acc += tl.load(p2 + offs, mask=mask).to(tl.float32)
    if N > 3:
        acc += tl.load(p3 + offs, mask=mask).to(tl.float32)
    if N > 4:
        acc += tl.load(p4 + offs, mask=mask).to(tl.float32)
    if N > 5:
        acc += tl.load(p5 + offs, mask=mask).to(tl.float32)
    if N > 6:
        acc += tl.load(p6 + offs, mask=mask).to(tl.float32)
    if N > 7:
        acc += tl.load(p7 + offs, mask=mask).to(tl.float32)
    tl.store(out + offs, acc.to(out.dtype.element_ty), mask=mask)


def sum_tensors(tensors):
    """Elementwise sum of up to 8 same-shape, same-dtype tensors in one kernel."""
    tensors = [t.contiguous() for t in tensors]
    if len(tensors) == 1:
        return tensors[0]
    if len(tensors) > _MAX:
        return sum_tensors([sum_tensors(tensors[:_MAX])] + tensors[_MAX:])
    out = torch.empty_like(tensors[0])
    numel = out.numel()
    ptrs = tensors + [tensors[0]] * (_MAX - len(tensors))
    block = 4096
    _sum_kernel[(triton.cdiv(numel, block),)](out, *ptrs, numel, N=len(tensors), BLOCK=block)
    return out


class _Fanout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        return tuple(x.view_as(x) for _ in range(n))

    @staticmethod
    def backward(ctx, *grads):
        grads = [g for g in grads if g is not None]
        if not grads:
            return None, None
        return sum_tensors(grads), None


def fanout(x: torch.Tensor, n: int):
    """``n`` aliases of ``x`` whose gradients are summed in one pass."""
    return _Fanout.apply(x, n)
