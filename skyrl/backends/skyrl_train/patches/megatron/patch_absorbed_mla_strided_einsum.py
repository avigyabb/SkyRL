"""Opt-in: absorbed-MLA per-head projections as strided batched GEMMs, without layout copies.

``AbsorbedMLASelfAttention`` absorbs ``k_up`` into the query (``einsum("...nd,ndk->...nk")``) and
applies ``v_up`` to the attention output (``einsum("...nc,ndc->...nd")``). ``torch.einsum`` runs
these as a ``bmm`` over heads on a head-major copy of the input and returns a head-major result,
which the module then makes ``.contiguous()``; the backward copies the gradient head-major again.
Each copy is a strided permute of a ``[tokens, heads, 512]`` tensor at ~1/3 of copy bandwidth.

``SKYRL_MLA_STRIDED_EINSUM=1`` gives the ``absorbed_mla`` module an ``einsum`` that, for these two
equations, calls ``bmm`` on token-major views (cuBLAS takes the head as the batch stride and the
token as the leading dimension) and writes the result, and the input gradient, straight into
token-major tensors. Same GEMMs and values, no copies. Other equations use ``torch.einsum``.
"""

import os

import torch
from loguru import logger

_APPLIED = False
_EQUATIONS = {"...nd,ndk->...nk": False, "...nc,ndc->...nd": True}  # value: weight given as [n, out, in]


class _PerHeadMatmul(torch.autograd.Function):
    """``out[m, n, :] = x[m, n, :] @ w[n]`` for token-major ``x`` ``[m, n, d]`` and ``w`` ``[n, d, k]``."""

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        out = x.new_empty(x.size(0), x.size(1), w.size(2))
        torch.bmm(x.transpose(0, 1), w, out=out.transpose(0, 1))
        return out

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        grad = grad.contiguous()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(x)
            torch.bmm(grad.transpose(0, 1), w.transpose(1, 2), out=dx.transpose(0, 1))
        if ctx.needs_input_grad[1]:
            dw = torch.bmm(x.transpose(0, 1).transpose(1, 2), grad.transpose(0, 1))
        return dx, dw


class _StridedEinsumTorch:
    """``torch`` stand-in for ``absorbed_mla`` whose ``einsum`` handles the absorption equations."""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        return getattr(self._base, name)

    def einsum(self, equation, *operands):
        transposed = _EQUATIONS.get(equation.replace(" ", ""))
        if transposed is None or len(operands) != 2:
            return self._base.einsum(equation, *operands)
        x, w = operands
        if transposed:
            w = w.transpose(1, 2)
        if w.dim() != 3 or x.dim() < 2 or x.shape[-2:] != (w.size(0), w.size(1)) or not x.is_contiguous():
            return self._base.einsum(equation, *operands)
        out = _PerHeadMatmul.apply(x.view(-1, x.size(-2), x.size(-1)), w)
        return out.view(*x.shape[:-1], w.size(2))


def apply_absorbed_mla_strided_einsum_patch() -> None:
    global _APPLIED
    if _APPLIED or os.environ.get("SKYRL_MLA_STRIDED_EINSUM", "0").lower() not in ("1", "true"):
        return
    from megatron.core.transformer.experimental_attention_variant import absorbed_mla

    absorbed_mla.torch = _StridedEinsumTorch(absorbed_mla.torch)
    _APPLIED = True
    logger.info("MLA: absorbed per-head projections run as strided batched GEMMs")
