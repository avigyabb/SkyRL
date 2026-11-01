"""DSA k-pool indexer kernels, vendored from NVIDIA/Megatron-LM#7522.

GLM-5.3-Flash's DSA layers use a *k-pool* compressed indexer: keys are pooled in groups of
``index_kpool`` consecutive tokens, the indexer scores and selects ``index_topk / index_kpool``
pools, the selected pools are expanded back to token indices, and the query's own incomplete
tail pool is always appended. Without it the token-level indexer is exact only up to
``dsa_indexer_topk`` (2048 for this checkpoint), which caps training at short sequences.

The pinned megatron-core does not have this yet -- NVIDIA/Megatron-LM#7522 ("Add KPool DSA
indexer, NoPE MLA support, and FP8 precision controls") is still open. The functions below are
copied **verbatim** from that PR's branch (HollowMan6/Megatron-LM @ beb4be3a8,
``megatron/core/transformer/experimental_attention_variant/dsa.py`` lines 743-993) so they can
be diffed against it mechanically::

    git diff <this file> <megatron>/core/transformer/experimental_attention_variant/dsa.py

Everything they depend on -- ``_compute_index_scores``, ``hadamard_transform``, ``dsa_masking``
-- already exists in the pinned megatron-core with identical signatures, so this module needs
no patching of megatron itself.

One deliberate deviation: ``fused_qk_topk_kpool`` scores and selects in query chunks (see the
comment there), since the verbatim version materializes O(sq^2) per-head FP32 scores and OOMs
at 32k context. #7522 has since chunked as well (b27efd8, a fixed 256 MiB cap); the chunk loop
here follows its structure with a configurable, larger cap.

DELETE THIS MODULE once the megatron-core pin includes #7522.
"""

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from megatron.core.transformer.experimental_attention_variant import dsa_masking
from megatron.core.transformer.experimental_attention_variant.dsa import (
    _compute_index_scores,
)

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


# Upper bound on elements of one query chunk's per-head FP32 index scores (default 2 GiB).
_KPOOL_SCORE_CHUNK_ELEMS = int(os.environ.get("SKYRL_DSA_KPOOL_SCORE_CHUNK_ELEMS", 1 << 29))

# Opt-in: emit each row's selected pool ids in ascending order (-1 first), so the expanded token
# indices are ascending once the cuDNN path compacts out the -1 slots and its sort-by-index becomes
# an identity (see patch_dsa_cudnn_fast_paths). Sorting [rows, budget] pool ids per rank, before the
# gather, is ~4x fewer elements than sorting the gathered [rows, index_topk] token ids.
_KPOOL_PRESORT = os.environ.get("SKYRL_DSA_TOPK_PRESORTED", "0").lower() in ("1", "true")


def _presort(pool_ids: torch.Tensor) -> torch.Tensor:
    return pool_ids.sort(dim=-1).values if _KPOOL_PRESORT else pool_ids
# SkyRL: score pools with a fused Triton kernel (BF16 inputs, FP32 accumulation, causal tiles only)
# instead of the FP32 einsum. Not bitwise equal: near-ties in the pool selection can differ.
# SkyRL: rows per stripe when the indexer's query rows are split across a group (0: contiguous slices).
_KPOOL_SHARD_STRIPE = int(os.environ.get("SKYRL_DSA_KPOOL_SHARD_STRIPE", "0"))
_KPOOL_FUSED_INDEXER = os.environ.get("SKYRL_DSA_KPOOL_FUSED_INDEXER", "0").lower() in ("1", "true")


def _kpool_fp8_input(x: torch.Tensor) -> torch.Tensor:
    """Match the indexer's FP32 Hadamard, BF16 rounding, and E4M3 power-of-two scale."""
    if not x.numel():
        return x.float()
    assert hadamard_transform is not None, "fast_hadamard_transform is required for FP8 KPool."
    x = hadamard_transform(x.float(), scale=x.shape[-1] ** -0.5).to(torch.bfloat16).float()
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / 448.0)))
    return (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale


def _kpool_compress_keys(k: torch.Tensor, gate_score: torch.Tensor, ape: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Softmax-weighted pool keys, accumulated in FP32 and returned in BF16.

    Keys and gates are [tokens, batch, head_dim]; ape is [pool_size, head_dim].
    Only complete pools are compressed. Query-local tails are appended separately.
    """
    seqlen, bsz, head_dim = k.shape
    assert head_dim == ape.shape[1], f"head_dim {head_dim} != ape dim1 {ape.shape[1]}"
    num_pools = seqlen // pool_size
    # Drop the trailing incomplete pool from compression; its tokens are appended
    # later via append_tail_to_topk (always_select_tail).
    usable = num_pools * pool_size
    # [num_pools, pool_size, batch, head_dim]
    k_p = k[:usable].reshape(num_pools, pool_size, bsz, head_dim)
    # gate_score: [seqlen, batch, head_dim] -> [num_pools, pool_size, batch, head_dim]
    if gate_score is not None:
        g = gate_score[:usable].reshape(num_pools, pool_size, bsz, head_dim).float()
    else:
        g = torch.zeros((num_pools, pool_size, bsz, head_dim), dtype=torch.float32, device=k.device)

    # Per-dim softmax across the pool's slots: score[slot] = gate_score[slot] + ape[slot].
    # ape: [pool_size, head_dim] -> broadcast over (num_pools, batch).
    ape_f = ape.to(dtype=torch.float32, device=k.device)  # [pool_size, head_dim]
    score = g + ape_f.unsqueeze(0).unsqueeze(2)  # [num_pools, pool_size, batch, head_dim]
    # Numerically-stable per-dim softmax over dim=1 (the pool slot dim).
    score_max = score.max(dim=1, keepdim=True).values
    prob = torch.exp(score - score_max)
    # weighted sum of k over pool slots: [num_pools, batch, head_dim]. Keep the
    # numerator 4D ([num_pools, 1, batch, head_dim]) so it broadcasts cleanly
    # against the 4D denom ([num_pools, 1, 1, head_dim]); a 3D numerator would
    # left-pad and produce a spurious extra (num_pools) dimension.
    k_f = k_p.float()
    k_pooled = (prob * k_f).sum(dim=1, keepdim=True) / prob.sum(dim=1, keepdim=True).clamp(min=1e-12)
    k_pooled = k_pooled.squeeze(1)  # [num_pools, batch, head_dim]
    return k_pooled.to(torch.bfloat16)


def _expand_pools_to_tokens(
    pool_ids: torch.Tensor,
    pool_valid: torch.Tensor,
    topk_tokens: int,
    pool_size: int,
    pool_token_base: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expand fixed-width pool IDs to token IDs, preserving -1 padding."""
    assert pool_ids.ndim == 2 and pool_valid.shape == pool_ids.shape
    assert pool_ids.shape[1] * pool_size == topk_tokens
    if pool_token_base is None:
        starts = pool_ids * pool_size
    elif pool_token_base.numel():
        starts = pool_token_base[pool_ids.clamp(min=0)]
    else:
        starts = torch.zeros_like(pool_ids)
    offsets = torch.arange(pool_size, device=pool_ids.device)
    tokens = starts.unsqueeze(-1) + offsets
    tokens = tokens.masked_fill(~pool_valid.unsqueeze(-1), -1)
    return tokens.reshape(pool_ids.shape[0], topk_tokens).to(torch.int32)


def _append_tail_to_topk(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
    tail_start_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Append each query's incomplete causal pool, in global token coordinates."""
    tail_count = seq_lens.to(torch.int32).remainder(pool_size)
    tail_start = (
        seq_lens.to(torch.int32) - tail_count if tail_start_override is None else tail_start_override.to(torch.int32)
    )
    offsets = torch.arange(pool_size - 1, device=topk_result.device)
    tail = tail_start[:, None] + offsets
    tail = tail.masked_fill(offsets >= tail_count[:, None], -1).to(topk_result.dtype)
    return torch.cat((topk_result, tail), dim=-1)


def _kpool_compress_keys_per_seg(
    k: torch.Tensor,
    gate_score: Optional[torch.Tensor],
    ape: torch.Tensor,
    pool_size: int,
    cu_seqlens_kv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress complete pools within each packed segment.

    Return pooled keys and their global starting token indices. Segment boundaries
    need not be multiples of pool_size; a pool must never span two documents.
    """
    cu = cu_seqlens_kv.to(device=k.device, dtype=torch.int64)
    n_seg = int(cu.numel()) - 1
    pooled_parts = []
    base_parts = []
    for i in range(n_seg):
        s = int(cu[i])
        e = int(cu[i + 1])
        seg_len = e - s
        if seg_len <= 0:
            continue
        k_seg = k[s:e]
        gate_seg = gate_score[s:e] if gate_score is not None else None
        k_pooled_seg = _kpool_compress_keys(k_seg, gate_seg, ape, pool_size)
        # [num_pools_seg, b, d]
        n_pools_seg = k_pooled_seg.size(0)
        pooled_parts.append(k_pooled_seg)
        # pool j (local) of this segment starts at global token s + j*pool_size.
        seg_bases = torch.arange(n_pools_seg, device=k.device, dtype=torch.int64) * pool_size + s
        base_parts.append(seg_bases)
    k_pooled_global = torch.cat(pooled_parts, dim=0)
    pool_token_base = torch.cat(base_parts, dim=0)
    return k_pooled_global, pool_token_base


@torch.no_grad()
def _kpool_topk_in_query_chunks(
    q: torch.Tensor,
    weights: torch.Tensor,
    k_pooled: torch.Tensor,
    budget: int,
    select_k: int,
    chunk: int,
    q_lo: int,
    q_hi: int,
    use_relu: bool,
    mask: Optional[torch.Tensor],
    v_starts: Optional[torch.Tensor],
    v_ends: Optional[torch.Tensor],
    k_pos: Optional[torch.Tensor],
) -> torch.Tensor:
    """``_compute_index_scores`` + masking + top-k, one query chunk at a time.

    Scores query rows ``[q_lo, q_hi)`` and returns their ``[batch, q_hi - q_lo, budget]`` pool ids,
    -1 for unused slots. Bitwise the same as the one-shot
    path: the same FP32 einsum, ReLU, head weighting and head sum, with ReLU and weighting in place
    so a chunk holds one FP32 score buffer instead of two or three (the selection is discrete, so
    no autograd graph is needed), ``k_pooled`` cast to FP32 once, and the output preallocated.
    """
    batch = q.shape[1]
    pool_topk = torch.full((batch, q_hi - q_lo, budget), -1, dtype=torch.int64, device=q.device)
    if select_k == 0:
        return pool_topk
    if _KPOOL_FUSED_INDEXER and v_starts is not None:
        fused = _kpool_topk_fused(
            q, weights, k_pooled, pool_topk, select_k, q_lo, q_hi, use_relu, v_starts, v_ends, k_pos
        )
        if fused is not None:
            return fused
    k_fp32 = k_pooled.float()
    for q0 in range(q_lo, q_hi, chunk):
        q1 = min(q_hi, q0 + chunk)
        scores = torch.einsum("sbhd,tbd->sbht", q[q0:q1].float(), k_fp32)
        if use_relu:
            scores.relu_()
        scores.mul_(weights[q0:q1].unsqueeze(-1))
        # [chunk, batch, num_pools] -> [batch, chunk, num_pools]
        scores = scores.sum(dim=2).transpose(0, 1)
        if v_starts is not None:
            scores = dsa_masking.apply_starts_ends_mask_to_scores(scores, v_starts[q0:q1], v_ends[q0:q1], k_pos)
        elif mask is not None:
            assert mask.dtype == scores.dtype, "Mask dtype must match index scores dtype"
            scores.add_(mask[..., q0:q1, :])
        topk_scores, topk_ids = scores.topk(select_k, dim=-1)
        del scores
        pool_topk[:, q0 - q_lo : q1 - q_lo, :select_k] = topk_ids.masked_fill_(topk_scores == float("-inf"), -1)
    return pool_topk


@triton.jit
def _kpool_score_kernel(
    Q,
    K,
    W,
    PS,
    PE,
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
):
    """OUT[q, p] = sum_h W[q, h] * relu(Q[q, h] . K[p]) for PS[q] <= p < PE[q], else -inf."""
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
        acc = tl.where(valid, acc, float("-inf"))
    tl.store(out_ptrs, acc, mask=out_ok)


@torch.no_grad()
def _kpool_topk_fused(q, weights, k_pooled, pool_topk, select_k, q_lo, q_hi, use_relu, v_starts, v_ends, k_pos):
    """``_kpool_topk_in_query_chunks`` with the scores from ``_kpool_score_kernel``.

    Pool ``p`` is valid for a query when ``v_starts <= k_pos[p] < v_ends``; ``k_pos`` (pool end
    positions) is increasing, so the bounds become pool-index ranges ``[ps, pe)``, which the kernel
    uses to skip the non-causal tiles. Scores use BF16 inputs with FP32 accumulation instead of the
    FP32 einsum, so near-ties can select differently.
    """
    sq, batch, n_heads, dim = q.shape
    if q.dtype != torch.bfloat16 or dim & (dim - 1) or dim < 16:
        return None
    k_pos = k_pos.contiguous()
    pool_starts = torch.searchsorted(k_pos, v_starts[q_lo:q_hi].contiguous()).to(torch.int32)
    pool_ends = torch.searchsorted(k_pos, v_ends[q_lo:q_hi].contiguous()).to(torch.int32)
    num_pools = k_pooled.size(0)
    chunk = max(1, min(q_hi - q_lo, _KPOOL_SCORE_CHUNK_ELEMS // max(1, num_pools)))
    bq, bn = 64, 128
    # Pools at or past a chunk's largest ``pe`` are -inf for every row in it, so score and select
    # over that prefix only (at least ``select_k`` wide): causal chunks scan ~half the pools.
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
                scores,
                q1 - q0,
                width,
                q_c.stride(0),
                q_c.stride(1),
                k_b.stride(0),
                w_c.stride(0),
                scores.stride(0),
                H=n_heads,
                D=dim,
                BQ=bq,
                BN=bn,
                RELU=use_relu,
            )
            topk_scores, ids = scores.topk(select_k, dim=-1)
            del scores
            pool_topk[b, q0 - q_lo : q1 - q_lo, :select_k] = ids.masked_fill_(topk_scores == float("-inf"), -1)
    return pool_topk


def fused_qk_topk_kpool(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    pool_size: int,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    use_relu: bool = True,
    always_select_tail: bool = True,
    fp8_indexer: bool = False,
    query_shard_group=None,
):
    """Select complete causal pools and append each query's incomplete tail.

    q is [queries, batch, heads, dim], k/gate_score are [keys, batch, dim],
    and weights are [queries, batch, heads]. Packed bounds use global token
    coordinates. Output indices are [batch, queries, index_topk + pool_size - 1]
    when always_select_tail is enabled, with -1 for unused slots.
    """
    sk = k.size(0)

    # Packed pools restart at each document boundary.
    use_per_seg = cu_seqlens_kv is not None and cu_seqlens_kv.numel() >= 2
    if use_per_seg:
        k_pooled, pool_token_base = _kpool_compress_keys_per_seg(k, gate_score, ape, pool_size, cu_seqlens_kv)
        num_pools = k_pooled.size(0)
    else:
        num_pools = sk // pool_size
        k_pooled = _kpool_compress_keys(k, gate_score, ape, pool_size)
        pool_token_base = torch.arange(num_pools, device=k.device, dtype=torch.int64) * pool_size

    if fp8_indexer:
        q, k_pooled = _kpool_fp8_input(q), _kpool_fp8_input(k_pooled)

    # A pool is causal only when its final token is within the query's bounds.
    pool_positions = pool_token_base + (pool_size - 1)
    eff_key_positions = key_positions[pool_positions] if key_positions is not None else pool_positions
    v_starts, v_ends, k_pos = dsa_masking.normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=eff_key_positions,
        sk=num_pools,
        device=q.device,
    )

    # Keep the selection width fixed, including when fewer causal pools exist.
    budget = index_topk // pool_size
    select_k = min(budget, num_pools)

    # SkyRL deviation from #7522: score and select in query chunks. _compute_index_scores
    # materializes FP32 [sq, batch, heads, num_pools] per-head scores (32 GiB at sq=32k for
    # GLM-5.3-Flash's 32 heads) and the full [batch, sq, num_pools] matrix is O(sq^2), so past one
    # chunk only the [batch, sq, budget] selection is kept. Top-k is per query row, so this is
    # exact. Full scores (with autograd) are returned only when one chunk covers every query.
    sq, batch, n_heads = q.shape[0], q.shape[1], q.shape[2]
    chunk = max(1, min(sq, _KPOOL_SCORE_CHUNK_ELEMS // max(1, batch * n_heads * num_pools)))
    # SkyRL: with ``query_shard_group`` (inputs identical on every rank of the group, as for the
    # tensor-parallel group, which all see the gathered sequence), each rank selects pools only
    # for its contiguous slice of queries and the selections are all-gathered. The scoring is
    # O(sq * num_pools) and otherwise replicated on every rank; per query row nothing changes.
    shard_size = query_shard_group.size() if query_shard_group is not None else 1
    if chunk >= sq and shard_size == 1:
        index_scores = _compute_index_scores(q, weights, k_pooled, use_relu=use_relu)
        if v_starts is not None:
            index_scores = dsa_masking.apply_starts_ends_mask_to_scores(index_scores, v_starts, v_ends, k_pos)
        elif mask is not None:
            assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
            index_scores = index_scores + mask
        pool_topk = torch.full((batch, sq, budget), -1, dtype=torch.int64, device=q.device)
        if select_k > 0:
            topk_scores, topk_ids = index_scores.topk(select_k, dim=-1)
            # [batch, seqlen_q, select_k] -> mask invalid pools
            pool_topk[..., :select_k] = topk_ids.masked_fill(topk_scores == float("-inf"), -1)
    else:
        index_scores = None
        rows_per_rank = -(-sq // shard_size)
        stripe = _KPOOL_SHARD_STRIPE
        striped = shard_size > 1 and stripe > 0 and v_starts is not None and sq % (shard_size * stripe) == 0
        if striped:
            # SkyRL: causal scoring work grows with the query position, so contiguous row slices
            # leave the last rank with ~2x the average work. Rank r takes the stripes r, r + shard,
            # r + 2 * shard, ... of ``stripe`` rows instead; the gather below restores row order.
            rank = query_shard_group.rank()
            n_stripes = sq // (shard_size * stripe)
            rows = (
                (torch.arange(n_stripes, device=q.device) * shard_size + rank).unsqueeze(1) * stripe
                + torch.arange(stripe, device=q.device)
            ).reshape(-1)
            pool_topk = _kpool_topk_in_query_chunks(
                q.index_select(0, rows),
                weights.index_select(0, rows),
                k_pooled,
                budget,
                select_k,
                chunk,
                0,
                rows.numel(),
                use_relu,
                None,
                v_starts.index_select(0, rows),
                v_ends.index_select(0, rows),
                k_pos,
            )
        else:
            q_lo = min(sq, query_shard_group.rank() * rows_per_rank) if shard_size > 1 else 0
            q_hi = min(sq, q_lo + rows_per_rank)
            pool_topk = _kpool_topk_in_query_chunks(
                q, weights, k_pooled, budget, select_k, chunk, q_lo, q_hi, use_relu, mask, v_starts, v_ends, k_pos
            )
        if striped:
            local = _presort(pool_topk.to(torch.int32)).contiguous()
            gathered = torch.empty((shard_size * batch, rows_per_rank, budget), dtype=torch.int32, device=q.device)
            torch.distributed.all_gather_into_tensor(gathered, local, group=query_shard_group)
            # [shard, batch, n_stripes, stripe, budget] -> [batch, n_stripes, shard, stripe, budget]
            gathered = gathered.view(shard_size, batch, n_stripes, stripe, budget).permute(1, 2, 0, 3, 4)
            pool_topk = gathered.reshape(batch, sq, budget).to(torch.int64)
        elif shard_size > 1:
            # Pool ids fit in int32; pad the (possibly short) last slice so shards are equal-sized.
            local = _presort(pool_topk.to(torch.int32))
            if local.shape[1] < rows_per_rank:
                local = torch.nn.functional.pad(local, (0, 0, 0, rows_per_rank - local.shape[1]), value=-1)
            # Concatenated (not stacked) output shape: the one every backend accepts.
            gathered = torch.empty((shard_size * batch, rows_per_rank, budget), dtype=torch.int32, device=q.device)
            torch.distributed.all_gather_into_tensor(gathered, local.contiguous(), group=query_shard_group)
            gathered = gathered.view(shard_size, batch, rows_per_rank, budget)
            pool_topk = gathered.permute(1, 0, 2, 3).reshape(batch, shard_size * rows_per_rank, budget)[:, :sq]
            pool_topk = pool_topk.to(torch.int64)

    # Expand [batch * queries, pools] to a fixed token budget.
    rows = pool_topk.shape[0] * pool_topk.shape[1]
    pool_flat = pool_topk.reshape(rows, -1)
    pool_valid = pool_flat >= 0
    # Clamp invalid ids to 0 for the arithmetic, restore -1 via the where mask.
    safe_pool = pool_flat.clamp(min=0)
    token_topk = _expand_pools_to_tokens(
        safe_pool,
        pool_valid,
        index_topk,
        pool_size,
        pool_token_base=pool_token_base if use_per_seg else None,
    )
    # token_topk is [rows, index_topk]; reshape back to [batch, seqlen_q, index_topk].
    token_topk = token_topk.reshape(pool_topk.shape[0], pool_topk.shape[1], index_topk)

    if always_select_tail:
        # Pool phase is query-local, never the final length of the packed sample.
        sq, batch = q.shape[:2]
        ends = v_ends if v_ends is not None else torch.arange(1, sq + 1, device=q.device)
        if v_starts is not None:
            starts = v_starts
        elif cu_seqlens_kv is not None:
            cu = cu_seqlens_kv.to(device=q.device, dtype=torch.int64)
            starts = cu[torch.searchsorted(cu[1:], ends - 1, right=True)]
        else:
            starts = torch.zeros_like(ends)
        lengths = ends - starts
        tail_starts = ends - lengths.remainder(pool_size)
        token_topk = _append_tail_to_topk(
            token_topk.reshape(rows, -1),
            lengths.expand(batch, -1).reshape(-1),
            pool_size,
            tail_start_override=tail_starts.expand(batch, -1).reshape(-1),
        ).reshape(batch, sq, -1)

    return index_scores, token_topk
