"""Fused mHC mapping projection + RMS factor (``SKYRL_MHC_FUSED_PROJ_RMS=1``).

``proj = x @ W.T`` (FP32 accumulate, TF32 like the cuBLAS path) and ``r = rsqrt(mean(x^2) + eps)``
over the activation-dtype ``x [M, K]`` in one read, without an FP32 copy of ``x``. The backward is
``dx = dproj @ W - dr * r^3 / K * x`` in one more pass, plus ``dW = dproj.T @ x`` (one more read of
``x``, FP32 partial sums over row splits) when ``W`` is trainable (full fine-tuning; with LoRA it is
frozen). Only ``x`` and ``r`` are saved, so no checkpoint recompute is needed either.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(X, W, P, R, M, K, N, eps, stride_x, stride_w, stride_p, BM: tl.constexpr, BK: tl.constexpr, NP: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    row_ok = rows < M
    cols_n = tl.arange(0, NP)
    acc = tl.zeros([BM, NP], tl.float32)
    ss = tl.zeros([BM], tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        k_ok = ks < K
        x = tl.load(X + rows[:, None].to(tl.int64) * stride_x + ks[None, :], mask=row_ok[:, None] & k_ok[None, :], other=0.0).to(tl.float32)
        w = tl.load(W + cols_n[:, None] * stride_w + ks[None, :], mask=(cols_n[:, None] < N) & k_ok[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w), input_precision="tf32")
        ss += tl.sum(x * x, axis=1)
    tl.store(P + rows[:, None].to(tl.int64) * stride_p + cols_n[None, :], acc, mask=row_ok[:, None] & (cols_n[None, :] < N))
    tl.store(R + rows, tl.rsqrt(ss / K + eps), mask=row_ok)


@triton.jit
def _bwd_kernel(DP, W, X, COEF, DX, M, K, N, stride_dp, stride_w, stride_x, stride_dx, BM: tl.constexpr, BK: tl.constexpr, NP: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    ks = tl.program_id(1) * BK + tl.arange(0, BK)
    row_ok, k_ok = rows < M, ks < K
    cols_n = tl.arange(0, NP)
    dp = tl.load(DP + rows[:, None].to(tl.int64) * stride_dp + cols_n[None, :], mask=row_ok[:, None] & (cols_n[None, :] < N), other=0.0)
    w = tl.load(W + cols_n[:, None] * stride_w + ks[None, :], mask=(cols_n[:, None] < N) & k_ok[None, :], other=0.0)
    x = tl.load(X + rows[:, None].to(tl.int64) * stride_x + ks[None, :], mask=row_ok[:, None] & k_ok[None, :], other=0.0).to(tl.float32)
    coef = tl.load(COEF + rows, mask=row_ok, other=0.0)
    dx = tl.dot(dp, w, input_precision="tf32") + coef[:, None] * x
    tl.store(DX + rows[:, None].to(tl.int64) * stride_dx + ks[None, :], dx.to(DX.dtype.element_ty), mask=row_ok[:, None] & k_ok[None, :])


@triton.jit
def _dw_kernel(DP, X, DWP, M, K, N, rows_per_split, stride_dp, stride_x, BM: tl.constexpr, BK: tl.constexpr, NP: tl.constexpr):
    """Partial ``dW[N, K] = dP.T @ X`` over this program's row split, written to ``DWP[split]``."""
    ks = tl.program_id(0) * BK + tl.arange(0, BK)
    split = tl.program_id(1)
    k_ok = ks < K
    cols_n = tl.arange(0, NP)
    acc = tl.zeros([NP, BK], tl.float32)
    lo = split * rows_per_split
    for m0 in range(lo, lo + rows_per_split, BM):
        rows = m0 + tl.arange(0, BM)
        row_ok = rows < M
        dp = tl.load(DP + rows[:, None].to(tl.int64) * stride_dp + cols_n[None, :], mask=row_ok[:, None] & (cols_n[None, :] < N), other=0.0)
        x = tl.load(X + rows[:, None].to(tl.int64) * stride_x + ks[None, :], mask=row_ok[:, None] & k_ok[None, :], other=0.0).to(tl.float32)
        acc += tl.dot(tl.trans(dp), x, input_precision="tf32")
    out = DWP + split.to(tl.int64) * N * K + cols_n[:, None] * K + ks[None, :]
    tl.store(out, acc, mask=(cols_n[:, None] < N) & k_ok[None, :])


_BM, _BK = 128, 64
_DW_SPLITS = 32


class FusedProjRms(torch.autograd.Function):
    """``(x @ W.T, rsqrt(mean(x^2) + eps))`` in FP32 for ``x [M, K]``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        x = x.contiguous()
        weight_dtype = weight.dtype
        weight = weight.float().contiguous()
        M, K = x.shape
        N = weight.shape[0]
        proj = torch.empty((M, N), dtype=torch.float32, device=x.device)
        r = torch.empty((M,), dtype=torch.float32, device=x.device)
        NP = max(16, triton.next_power_of_2(N))
        _fwd_kernel[(triton.cdiv(M, _BM),)](
            x, weight, proj, r, M, K, N, eps, x.stride(0), weight.stride(0), proj.stride(0), BM=_BM, BK=_BK, NP=NP,
            num_warps=4,
        )
        ctx.save_for_backward(x, weight, r)
        ctx.weight_dtype = weight_dtype
        return proj, r.unsqueeze(-1)

    @staticmethod
    def backward(ctx, dproj: torch.Tensor, dr: torch.Tensor):
        x, weight, r = ctx.saved_tensors
        M, K = x.shape
        N = weight.shape[0]
        dproj = dproj.float().contiguous()
        # d rsqrt(mean(x^2) + eps) / dx = -r^3 * x / K
        coef = (-dr.float().reshape(-1) * r.pow(3) / K).contiguous()
        dx = torch.empty_like(x)
        NP = max(16, triton.next_power_of_2(N))
        _bwd_kernel[(triton.cdiv(M, _BM), triton.cdiv(K, _BK))](
            dproj, weight, x, coef, dx, M, K, N, dproj.stride(0), weight.stride(0), x.stride(0), dx.stride(0),
            BM=_BM, BK=_BK, NP=NP, num_warps=4,
        )
        dw = None
        if ctx.needs_input_grad[1]:
            rows_per_split = triton.cdiv(triton.cdiv(M, _DW_SPLITS), _BM) * _BM
            splits = triton.cdiv(M, rows_per_split)
            partial = torch.empty((splits, N, K), dtype=torch.float32, device=x.device)
            _dw_kernel[(triton.cdiv(K, _BK), splits)](
                dproj, x, partial, M, K, N, rows_per_split, dproj.stride(0), x.stride(0), BM=_BM, BK=_BK, NP=NP,
                num_warps=4,
            )
            dw = partial.sum(0).to(ctx.weight_dtype)
        return dx, dw, None


# Kept for callers of the LoRA-era name.
FrozenProjRms = FusedProjRms
