"""SkyRL-only speedups for the k-pool DSA indexer, layered on upstream's code in ``dsa_kpool``.

``dsa_kpool`` is a verbatim copy of megatron-core's pooled indexer (NVIDIA/Megatron-LM#7522 and
follow-ups, plus #8299's query sharding). Everything SkyRL adds lives here and attaches at its
function boundaries -- module globals that ``fused_qk_topk_kpool`` looks up at call time -- so a
megatron-core bump only changes where ``_base`` points:

- ``_kpool_topk_from_score_chunks``:

  - ``SKYRL_DSA_KPOOL_FUSED_INDEXER=1``: score pools with a fused Triton kernel (BF16 inputs, FP32
    accumulation) that skips tiles outside every row's causal bounds, and select over the prefix of
    pools a chunk's rows can see. Not bitwise equal to upstream's FP32 einsum: near-ties can select
    differently. Inputs it does not handle (explicit masks without varlen bounds, non-BF16 queries,
    head dims that are not a power of two) use upstream's path.
  - ``SKYRL_DSA_TOPK_PRESORTED=1``: emit each row's pool ids in ascending order (-1 first), so the
    expanded token indices are ascending once the cuDNN path compacts out the -1 slots and its
    sort-by-index can be skipped (``patch_dsa_cudnn_fast_paths``).

- ``SKYRL_DSA_KPOOL_SHARD_STRIPE=n`` with a ``query_shard_group`` (call ``fused_qk_topk_kpool`` from
  here): causal scoring work grows with the query position, so #8299's contiguous row slices leave
  the last rank with ~2x the average work. Rank ``r`` instead scores the stripes ``r``, ``r + shard``,
  ``r + 2 * shard``, ... of ``n`` rows (in ``_kpool_topk_from_score_chunks``), and the all-gather
  (``_all_gather_kpool_query_shards``) puts the pool ids back in row order. Every step of the
  selection is per query row, so the result is the same.

- ``SKYRL_DSA_KPOOL_SCORE_CHUNK_ELEMS``: FP32 score elements per query chunk (upstream's
  ``_KPOOL_SCORE_CHUNK_BYTES`` / 4, 2 GiB by default).

When the pinned megatron-core has the pooled indexer: point ``_base`` at
``megatron.core.transformer.experimental_attention_variant.dsa`` and delete ``dsa_kpool.py``.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from skyrl.backends.skyrl_train.patches.megatron.mcore_ext import dsa_kpool as _base

_FUSED_INDEXER = os.environ.get("SKYRL_DSA_KPOOL_FUSED_INDEXER", "0").lower() in ("1", "true")
_PRESORT = os.environ.get("SKYRL_DSA_TOPK_PRESORTED", "0").lower() in ("1", "true")
_SHARD_STRIPE = int(os.environ.get("SKYRL_DSA_KPOOL_SHARD_STRIPE", "0"))
if os.environ.get("SKYRL_DSA_KPOOL_SCORE_CHUNK_ELEMS"):
    _base._KPOOL_SCORE_CHUNK_BYTES = 4 * int(os.environ["SKYRL_DSA_KPOOL_SCORE_CHUNK_ELEMS"])

_upstream_topk_from_score_chunks = _base._kpool_topk_from_score_chunks
_upstream_all_gather_query_shards = _base._all_gather_kpool_query_shards

# (shard_size, shard_rank, stripe) while a striped ``fused_qk_topk_kpool`` call is running.
_stripe = None


@triton.jit
def _kpool_score_kernel(
    Q,
    K,
    W,
    PS,
    PE,
    PX,
    OUT,
    S,
    NP,
    stride_qs,
    stride_qh,
    stride_ks,
    stride_ws,
    stride_os,
    H: tl.constexpr,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BN: tl.constexpr,
    RELU: tl.constexpr,
    HAS_PX: tl.constexpr,
):
    """OUT[q, p] = sum_h W[q, h] * relu(Q[q, h] . K[p]) for PS[q] <= p < PE[q] (and PX[p]), else -inf."""
    q0 = tl.program_id(0) * BQ
    n0 = tl.program_id(1) * BN
    offs_q = q0 + tl.arange(0, BQ)
    offs_n = n0 + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    q_ok = offs_q < S
    ps = tl.load(PS + offs_q, mask=q_ok, other=2147483647)
    pe = tl.load(PE + offs_q, mask=q_ok, other=0)
    out_ptrs = OUT + offs_q[:, None] * stride_os + offs_n[None, :]
    out_ok = q_ok[:, None] & (offs_n[None, :] < NP)
    acc = tl.full([BQ, BN], float("-inf"), tl.float32)
    # Skip tiles outside every row's bounds (the non-causal half).
    if (n0 < tl.max(pe)) & (n0 + BN > tl.min(ps)):
        k = tl.load(K + offs_n[:, None] * stride_ks + offs_d[None, :], mask=(offs_n < NP)[:, None], other=0.0)
        kt = tl.trans(k)
        acc = tl.zeros([BQ, BN], tl.float32)
        for h in range(H):
            qh = tl.load(
                Q + offs_q[:, None] * stride_qs + h * stride_qh + offs_d[None, :], mask=q_ok[:, None], other=0.0
            )
            s = tl.dot(qh, kt)
            if RELU:
                s = tl.maximum(s, 0.0)
            wh = tl.load(W + offs_q * stride_ws + h, mask=q_ok, other=0.0)
            acc += s * wh[:, None]
        valid = (offs_n[None, :] >= ps[:, None]) & (offs_n[None, :] < pe[:, None])
        if HAS_PX:
            px = tl.load(PX + offs_n, mask=offs_n < NP, other=0)
            valid = valid & (px[None, :] != 0)
        acc = tl.where(valid, acc, float("-inf"))
    tl.store(out_ptrs, acc, mask=out_ok)


@torch.no_grad()
def _fused_topk(q, weights, k_pooled, pool_exists, varlen_starts, varlen_ends, key_positions, budget, use_relu, q_lo, q_hi):
    """Pool selection for query rows ``[q_lo, q_hi)`` with scores from ``_kpool_score_kernel``.

    Pool ``p`` is valid for a query when ``varlen_starts <= key_positions[p] < varlen_ends``;
    ``key_positions`` (pool end positions) is increasing, so the bounds become pool-index ranges
    ``[ps, pe)``, which the kernel uses to skip the non-causal tiles. Pools at or past a chunk's
    largest ``pe`` are -inf for every row in it, so a chunk scores and selects over that prefix only
    (at least ``select_k`` wide).
    """
    sq, batch, n_heads, _ = q.shape
    num_pools = k_pooled.size(0)
    select_k = min(budget, num_pools)
    pool_topk = torch.full((batch, q_hi - q_lo, budget), -1, dtype=torch.int64, device=q.device)
    k_pos = key_positions.contiguous()
    if pool_exists is not None:
        # Per-document pools include placeholders (``pool_token_base < 0``) whose clamped positions
        # break the increasing order ``searchsorted`` needs. Give each the position of the last real
        # pool before it; the kernel masks the placeholders out.
        k_pos = torch.cummax(torch.where(pool_exists, k_pos, torch.full_like(k_pos, -1)), dim=0).values
        exists = pool_exists.to(torch.int8).contiguous()
    else:
        exists = k_pos  # unused (HAS_PX=False)
    pool_starts = torch.searchsorted(k_pos, varlen_starts[q_lo:q_hi].contiguous()).to(torch.int32)
    pool_ends = torch.searchsorted(k_pos, varlen_ends[q_lo:q_hi].contiguous()).to(torch.int32)
    chunk = max(1, min(q_hi - q_lo, _base._KPOOL_SCORE_CHUNK_BYTES // (4 * max(1, num_pools))))
    bq, bn = 64, 128
    starts = list(range(q_lo, q_hi, chunk))
    chunk_ends = torch.stack([pool_ends[q0 - q_lo : min(q_hi, q0 + chunk) - q_lo].max() for q0 in starts])
    widths = [min(num_pools, max(select_k, int(w))) for w in chunk_ends.tolist()]
    for b in range(batch):
        k_b = k_pooled[:, b].to(torch.bfloat16).contiguous()
        w_b = weights[:, b].float().contiguous()
        q_b = q[:, b].contiguous()
        for q0, width in zip(starts, widths):
            q1 = min(q_hi, q0 + chunk)
            ps, pe = pool_starts[q0 - q_lo : q1 - q_lo], pool_ends[q0 - q_lo : q1 - q_lo]
            scores = torch.empty((q1 - q0, width), dtype=torch.float32, device=q.device)
            q_c, w_c = q_b[q0:q1], w_b[q0:q1]
            grid = (triton.cdiv(q1 - q0, bq), triton.cdiv(width, bn))
            _kpool_score_kernel[grid](
                q_c,
                k_b,
                w_c,
                ps,
                pe,
                exists,
                scores,
                q1 - q0,
                width,
                q_c.stride(0),
                q_c.stride(1),
                k_b.stride(0),
                w_c.stride(0),
                scores.stride(0),
                H=n_heads,
                D=q.size(-1),
                BQ=bq,
                BN=bn,
                RELU=use_relu,
                HAS_PX=pool_exists is not None,
            )
            topk_scores, ids = scores.topk(select_k, dim=-1)
            del scores
            pool_topk[b, q0 - q_lo : q1 - q_lo, :select_k] = ids.masked_fill_(topk_scores == float("-inf"), -1)
    return pool_topk


def _stripe_rows(sq: int, shard_size: int, shard_rank: int, stripe: int, device) -> torch.Tensor:
    """Query rows of rank ``shard_rank``'s stripes ``r, r + shard, ...`` (``stripe`` rows each), in order."""
    n_stripes = sq // (shard_size * stripe)
    first = (torch.arange(n_stripes, device=device) * shard_size + shard_rank) * stripe
    return (first.unsqueeze(1) + torch.arange(stripe, device=device)).reshape(-1)


def _kpool_topk_from_score_chunks(
    q: torch.Tensor,
    k_pooled: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    pool_size: int,
    pool_exists: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
    pool_token_base: torch.Tensor,
    sk: int,
    use_relu: bool,
    query_start: int = 0,
    query_end: Optional[int] = None,
) -> torch.Tensor:
    """Upstream's ``_kpool_topk_from_score_chunks`` with the fused scorer, presorting and striping."""
    if _stripe is not None:
        # This rank's stripes instead of its contiguous slice; same row count (sq divides evenly).
        rows = _stripe_rows(q.size(0), *_stripe, q.device)
        q, weights = q.index_select(0, rows), weights.index_select(0, rows)
        varlen_starts, varlen_ends = varlen_starts.index_select(0, rows), varlen_ends.index_select(0, rows)
        query_start, query_end = 0, rows.numel()
    query_end = q.size(0) if query_end is None else query_end
    budget = index_topk // pool_size
    dim = q.size(-1)
    fused = (
        _FUSED_INDEXER
        and varlen_starts is not None
        and q.dtype == torch.bfloat16
        and dim >= 16
        and dim & (dim - 1) == 0
        and k_pooled.size(0) > 0
        and budget > 0
        and query_end > query_start
    )
    if fused:
        pool_topk = _fused_topk(
            q, weights, k_pooled, pool_exists, varlen_starts, varlen_ends, key_positions, budget, use_relu,
            query_start, query_end,
        )
    else:
        pool_topk = _upstream_topk_from_score_chunks(
            q=q,
            k_pooled=k_pooled,
            weights=weights,
            index_topk=index_topk,
            pool_size=pool_size,
            pool_exists=pool_exists,
            mask=mask,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
            pool_token_base=pool_token_base,
            sk=sk,
            use_relu=use_relu,
            query_start=query_start,
            query_end=query_end,
        )
    if _PRESORT:
        # Pool ids fit in int32, which sorts faster.
        pool_topk = pool_topk.to(torch.int32).sort(dim=-1).values.to(torch.int64)
    return pool_topk


def _all_gather_kpool_query_shards(local_pool_topk, sq, rows_per_shard, group):
    """Upstream's all-gather; under striping, also put the gathered rows back in query order."""
    gathered = _upstream_all_gather_query_shards(local_pool_topk, sq, rows_per_shard, group)
    if _stripe is None:
        return gathered
    shard_size, _, stripe = _stripe
    batch, _, budget = gathered.shape
    # Gathered order is [shard, stripe index, row in stripe]; query order is [stripe index, shard, row].
    gathered = gathered.view(batch, shard_size, sq // (shard_size * stripe), stripe, budget).transpose(1, 2)
    return gathered.reshape(batch, sq, budget)


if _FUSED_INDEXER or _PRESORT or _SHARD_STRIPE > 0:
    _base._kpool_topk_from_score_chunks = _kpool_topk_from_score_chunks
if _SHARD_STRIPE > 0:
    _base._all_gather_kpool_query_shards = _all_gather_kpool_query_shards


def fused_qk_topk_kpool(q, k, weights, index_topk, pool_size, gate_score, ape, *args, **kwargs):
    """Upstream's ``fused_qk_topk_kpool``, with striped query sharding (see module doc)."""
    global _stripe
    group = kwargs.get("query_shard_group")
    shard_size = group.size() if group is not None else 1
    striped = (
        _SHARD_STRIPE > 0
        and shard_size > 1
        and not args
        and kwargs.get("mask") is None
        and kwargs.get("varlen_starts") is not None
        and kwargs.get("varlen_ends") is not None
        and q.size(0) % (shard_size * _SHARD_STRIPE) == 0
    )
    if not striped:
        return _base.fused_qk_topk_kpool(q, k, weights, index_topk, pool_size, gate_score, ape, *args, **kwargs)
    _stripe = (shard_size, group.rank(), _SHARD_STRIPE)
    try:
        return _base.fused_qk_topk_kpool(q, k, weights, index_topk, pool_size, gate_score, ape, **kwargs)
    finally:
        _stripe = None
