"""K-pool DSA indexer: a verbatim copy of megatron-core's, plus NVIDIA/Megatron-LM#8299.

The pinned megatron-core predates the pooled indexer. This module copies it unchanged from
NVIDIA/Megatron-LM ``main`` @ ``631e146322af`` (``megatron/core/transformer/experimental_attention_variant/dsa.py``:
``_compute_index_scores`` and the block from ``_kpool_fp8_input`` through ``fused_qk_topk_kpool``,
i.e. #7522 and its follow-ups), with the k-pool hunks of the open NVIDIA/Megatron-LM#8299 applied
(``query_shard_group``: split the indexer's query rows across the tensor-parallel group).

Do not edit the code below: keep it byte-identical to upstream so a megatron-core bump deletes this
module instead of merging it. SkyRL-only speedups live in ``dsa_kpool_fast.py`` and hook the
function boundaries here (``_kpool_topk_from_score_chunks``, ``fused_qk_topk_kpool``).
Formatting of the copied code follows upstream, not SkyRL's style.
"""

# fmt: off
# ruff: noqa
from typing import Optional, Tuple

import torch
from megatron.core.transformer.experimental_attention_variant import dsa_masking
from megatron.core.transformer.experimental_attention_variant.dsa import rotate_activation

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


def _compute_index_scores(
    q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor, use_relu: bool = True
) -> torch.Tensor:
    """
    Perform index scoring with BF16/FP32 inputs.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py#L254-L274
    This implementation accepts BF16/FP32 inputs for the `fp8_index` scoring logic:
        1. Compute attention scores: q @ k^T;
        2. Optionally apply ReLU activation (DeepSeek V3.2 only; disabled for GLM5);
        3. Weight by attention weights;
        4. Sum across attention heads.

    Args:
        q: BF16/FP32 [seqlen_q, batch, index_n_heads, index_head_dim], the query tensor.
        weights: BF16/FP32 [seqlen_q, batch, index_n_heads], the attention weights.
        k: BF16/FP32 [seqlen_k, batch, index_head_dim], the key tensor.

    Returns:
        index_scores: FP32 [batch, seqlen_q, seqlen_k], the index scores.
    """
    sq, batch, n_heads, head_dim = q.shape
    sk = k.size(0)
    if sq == 0:
        return torch.empty(batch, 0, sk, dtype=torch.float32, device=q.device)
    k_fp32 = k.float()

    # Chunk over seqlen_q to avoid materializing the full [sq, batch, heads, sk]
    # fp32 tensor. Target 256 MiB per chunk to leave room for the output.
    bytes_per_token = batch * n_heads * sk * 4
    chunk_size = min(sq, max(1, 256 * 1024 * 1024 // max(1, bytes_per_token)))
    needs_grad = torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, weights, k))
    if needs_grad:
        score_chunks = []
    else:
        index_scores = torch.empty(sq, batch, sk, dtype=torch.float32, device=q.device)

    for start in range(0, sq, chunk_size):
        end = min(start + chunk_size, sq)
        # [chunk, batch, heads, sk]
        scores = torch.einsum('sbhd,tbd->sbht', q[start:end].float(), k_fp32)
        if use_relu:
            scores.relu_()
        # Top-k selection runs without gradients and can reuse the score buffer. When autograd
        # is active, collect chunks so the returned scores retain their computation graph.
        head_weights = weights[start:end].unsqueeze(-1)
        if needs_grad:
            scores = scores * head_weights
        else:
            scores.mul_(head_weights)
        score_chunk = scores.sum(dim=2)
        if needs_grad:
            score_chunks.append(score_chunk)
        else:
            index_scores[start:end] = score_chunk
        del scores

    # Transpose to [batch, seqlen_q, seqlen_k].
    if needs_grad:
        index_scores = torch.cat(score_chunks, dim=0)
    index_scores = index_scores.transpose(0, 1)

    return index_scores



def _kpool_fp8_input(x: torch.Tensor) -> torch.Tensor:
    """Match the indexer's FP32 Hadamard, BF16 rounding, and E4M3 power-of-two scale."""
    if not x.numel():
        return x.float()
    assert hadamard_transform is not None, "fast_hadamard_transform is required for FP8 KPool."
    x = hadamard_transform(x.float(), scale=x.shape[-1] ** -0.5).to(torch.bfloat16).float()
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / 448.0)))
    return (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale


def _kpool_compress_keys(
    k: torch.Tensor, gate_score: torch.Tensor, ape: torch.Tensor, pool_size: int
) -> torch.Tensor:
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
    k_pooled = (prob * k_f).sum(dim=1, keepdim=True) / prob.sum(dim=1, keepdim=True).clamp(
        min=1e-12
    )
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
    tail_count_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Append each query's incomplete causal pool, in global token coordinates."""
    tail_count = (
        seq_lens.to(torch.int32).remainder(pool_size)
        if tail_count_override is None
        else tail_count_override.to(torch.int32)
    )
    tail_start = (
        seq_lens.to(torch.int32) - tail_count
        if tail_start_override is None
        else tail_start_override.to(torch.int32)
    )
    offsets = torch.arange(pool_size - 1, device=topk_result.device)
    tail = tail_start[:, None] + offsets
    tail = tail.masked_fill(offsets >= tail_count[:, None], -1).to(topk_result.dtype)
    return torch.cat((topk_result, tail), dim=-1)


def _fill_kpool_left_boundary(
    token_topk: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, pool_size: int
) -> torch.Tensor:
    """Use unused expanded-pool slots for a left partial pool from an explicit mask."""
    rows, width = token_topk.shape
    complete_start = ((starts + pool_size - 1) // pool_size) * pool_size
    complete_end = (ends // pool_size) * pool_size
    has_complete_pool = complete_start < complete_end
    left_end = torch.where(starts < ends, torch.minimum(complete_start, ends), starts)
    left_count = (left_end - starts).clamp_min(0)
    left_count = torch.minimum(
        left_count,
        torch.where(
            has_complete_pool,
            torch.full_like(left_count, pool_size - 1),
            torch.full_like(left_count, pool_size),
        ),
    )

    offsets = torch.arange(pool_size, device=token_topk.device, dtype=torch.int32)
    left_tokens = starts[:, None].to(torch.int32) + offsets
    left_valid = offsets[None, :] < left_count[:, None]

    invalid = token_topk < 0
    invalid_positions = torch.where(
        invalid,
        torch.arange(width, device=token_topk.device, dtype=torch.int32)[None, :],
        torch.full((rows, width), width, device=token_topk.device, dtype=torch.int32),
    )
    fill_positions = invalid_positions.topk(k=min(pool_size, width), dim=-1, largest=False).values
    fill_positions_safe = fill_positions.clamp_max(width - 1)
    fill_valid = left_valid[:, : fill_positions.size(1)] & (fill_positions < width)
    current = token_topk.gather(-1, fill_positions_safe)
    values = torch.where(fill_valid, left_tokens[:, : fill_positions.size(1)], current)
    return token_topk.scatter(-1, fill_positions_safe, values.to(token_topk.dtype))


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
    sk, bsz, head_dim = k.shape
    n_seg = cu.numel() - 1
    max_pools = sk // pool_size
    if max_pools == 0:
        return (
            k.new_empty((0, bsz, head_dim), dtype=torch.bfloat16),
            torch.empty(0, device=k.device, dtype=torch.int64),
        )

    # Enumerate a fixed upper bound of complete pools and map each pool to its
    # segment. Invalid slots stay allocated and are excluded from top-k below.
    pools_per_segment = torch.div(cu[1:] - cu[:-1], pool_size, rounding_mode="floor")
    pool_offsets = torch.cat(
        (torch.zeros(1, device=k.device, dtype=torch.int64), pools_per_segment.cumsum(0))
    )
    pool_ids = torch.arange(max_pools, device=k.device, dtype=torch.int64)
    segment_ids = torch.searchsorted(pool_offsets[1:], pool_ids, right=True).clamp_max(n_seg - 1)
    pool_token_base = cu[segment_ids] + (pool_ids - pool_offsets[segment_ids]) * pool_size
    pool_valid = pool_ids < pool_offsets[-1]
    token_positions = pool_token_base.unsqueeze(-1) + torch.arange(pool_size, device=k.device)
    token_positions = token_positions.clamp(0, sk - 1).reshape(-1)
    pooled_gate = gate_score[token_positions] if gate_score is not None else None
    k_pooled = _kpool_compress_keys(k[token_positions], pooled_gate, ape, pool_size)
    pool_token_base = pool_token_base.masked_fill(~pool_valid, -1)
    return k_pooled, pool_token_base


def _pool_validity_from_token_mask(
    mask: torch.Tensor, pool_token_base: torch.Tensor, pool_size: int, sk: int
) -> torch.Tensor:
    """Map token validity to complete-pool validity for KPool scoring."""
    if mask.ndim not in (2, 3) or mask.shape[-1] != sk:
        raise ValueError(
            "KPool attention mask must have shape [sq, sk] or [batch, sq, sk], "
            f"got {tuple(mask.shape)} for sk={sk}"
        )
    token_positions = pool_token_base.clamp_min(0).unsqueeze(-1) + torch.arange(
        pool_size, device=mask.device
    )
    token_positions = token_positions.clamp_max(sk - 1)
    if mask.ndim == 2:
        token_valid = torch.gather(
            torch.isfinite(mask).unsqueeze(1).expand(-1, pool_token_base.numel(), -1),
            -1,
            token_positions.unsqueeze(0).expand(mask.size(0), -1, -1),
        )
    else:
        token_valid = torch.gather(
            torch.isfinite(mask).unsqueeze(-2).expand(-1, -1, pool_token_base.numel(), -1),
            -1,
            token_positions.view(1, 1, -1, pool_size).expand(mask.size(0), mask.size(1), -1, -1),
        )
    pool_valid = token_valid.all(dim=-1)
    return pool_valid & (pool_token_base >= 0)


def _query_bounds_from_token_mask(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return global valid key bounds for each query row of an explicit mask."""
    valid = torch.isfinite(mask)
    sk = mask.size(-1)
    key_positions = torch.arange(sk, device=mask.device, dtype=torch.int64)
    starts = key_positions.masked_fill(~valid, sk).amin(dim=-1)
    ends = key_positions.masked_fill(~valid, -1).amax(dim=-1) + 1
    has_valid = valid.any(dim=-1)
    starts = starts.masked_fill(~has_valid, 0)
    ends = ends.masked_fill(~has_valid, 0)
    return starts, ends


def _expand_query_bounds_for_batch(bounds: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """Normalize explicit-mask bounds to [batch, seqlen] without materializing masks."""
    if bounds.ndim == 1:
        if bounds.numel() != seqlen:
            raise ValueError(f"Expected {seqlen} query bounds, got {bounds.numel()}.")
        return bounds.unsqueeze(0).expand(batch, -1)
    if bounds.ndim == 2 and bounds.shape == (batch, seqlen):
        return bounds
    raise ValueError(
        f"Expected query bounds with shape [{seqlen}] or [{batch}, {seqlen}], "
        f"got {tuple(bounds.shape)}."
    )


_KPOOL_SCORE_CHUNK_BYTES = 2 * 1024 * 1024 * 1024


@torch.no_grad()
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
    """Compute KPool scores and selection in query chunks.

    KPool selection is discrete and runs under ``no_grad``. Only the fixed-width pool IDs
    are retained, not the full ``[batch, query, pool]`` score tensor.

    Only query rows ``[query_start, query_end)`` are scored, so the result is
    ``[batch, query_end - query_start, index_topk // pool_size]``. Each row's selection
    depends only on that row and the keys, so it is the same as the matching rows of a
    call that scores every query.
    """
    sq, batch, n_heads, _ = q.shape
    if query_end is None:
        query_end = sq
    num_rows = query_end - query_start
    num_pools = k_pooled.size(0)
    budget = index_topk // pool_size
    pool_topk = torch.full((batch, num_rows, budget), -1, dtype=torch.int64, device=q.device)
    if num_rows == 0 or budget == 0 or num_pools == 0:
        return pool_topk

    bytes_per_token = batch * n_heads * num_pools * 4
    chunk_size = min(num_rows, max(1, _KPOOL_SCORE_CHUNK_BYTES // max(1, bytes_per_token)))
    select_k = min(budget, num_pools)
    k_fp32 = k_pooled.float()
    pool_exists_mask = pool_exists.view(1, 1, -1) if pool_exists is not None else None

    for start in range(query_start, query_end, chunk_size):
        end = min(start + chunk_size, query_end)
        scores = torch.einsum('sbhd,tbd->sbht', q[start:end].float(), k_fp32)
        if use_relu:
            scores.relu_()
        scores.mul_(weights[start:end].unsqueeze(-1))
        score_chunk = scores.sum(dim=2).transpose(0, 1)
        del scores

        if varlen_starts is not None:
            valid = dsa_masking.build_valid_mask_from_starts_ends(
                varlen_starts[start:end], varlen_ends[start:end], key_positions
            )
            score_chunk.masked_fill_(~valid.unsqueeze(0), float('-inf'))
        elif mask is not None:
            mask_chunk = mask[start:end] if mask.ndim == 2 else mask[:, start:end]
            pool_valid = _pool_validity_from_token_mask(mask_chunk, pool_token_base, pool_size, sk)
            score_chunk.masked_fill_(~pool_valid, float('-inf'))
        if pool_exists_mask is not None:
            score_chunk.masked_fill_(~pool_exists_mask, float('-inf'))

        topk_scores, topk_indices = score_chunk.topk(select_k, dim=-1)
        topk_indices.masked_fill_(topk_scores == float('-inf'), -1)
        pool_topk[:, start - query_start : end - query_start, :select_k] = topk_indices

    return pool_topk


def _kpool_query_shard_bounds(sq: int, shard_size: int, shard_rank: int) -> Tuple[int, int, int]:
    """Return ``(query_start, query_end, rows_per_shard)`` for one rank's contiguous query slice.

    Every shard owns ``ceil(sq / shard_size)`` rows except that trailing shards are clipped
    at ``sq`` (and can be empty when ``sq < shard_size``).
    """
    rows_per_shard = -(-sq // shard_size)
    query_start = min(sq, shard_rank * rows_per_shard)
    query_end = min(sq, query_start + rows_per_shard)
    return query_start, query_end, rows_per_shard


def _all_gather_kpool_query_shards(
    local_pool_topk: torch.Tensor,
    sq: int,
    rows_per_shard: int,
    group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """All-gather per-rank KPool selections into the full ``[batch, sq, budget]`` tensor.

    Args:
        local_pool_topk: This rank's ``[batch, rows, budget]`` pool IDs, ``rows <= rows_per_shard``.
        sq: Total number of query rows.
        rows_per_shard: Rows owned by every shard before clipping at ``sq``.
        group: Process group the query rows are split across, in rank order.

    Returns:
        ``[batch, sq, budget]`` int64 pool IDs, -1 for unused slots.
    """
    group_size = group.size()
    batch, local_rows, budget = local_pool_topk.shape
    # Pool IDs fit in int32, which halves the all-gather volume.
    local = local_pool_topk.to(torch.int32)
    if local_rows < rows_per_shard:
        local = torch.nn.functional.pad(local, (0, 0, 0, rows_per_shard - local_rows), value=-1)
    gathered = torch.empty(
        (group_size * batch, rows_per_shard, budget), dtype=torch.int32, device=local.device
    )
    torch.distributed.all_gather_into_tensor(gathered, local.contiguous(), group=group)
    gathered = gathered.view(group_size, batch, rows_per_shard, budget).transpose(0, 1)
    return gathered.reshape(batch, group_size * rows_per_shard, budget)[:, :sq].to(torch.int64)


def _mask_topk_tokens_with_token_mask(
    topk_indices: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Remove expanded KPool tokens that are invalid in an explicit token mask."""
    token_valid = torch.isfinite(mask)
    if token_valid.ndim == 2:
        token_valid = token_valid.unsqueeze(0)
    if token_valid.size(0) == 1 and topk_indices.size(0) != 1:
        token_valid = token_valid.expand(topk_indices.size(0), -1, -1)
    token_ids = topk_indices.clamp_min(0).to(torch.long)
    selected_valid = torch.gather(token_valid, -1, token_ids)
    return topk_indices.masked_fill((topk_indices < 0) | ~selected_valid, -1)


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
    rotate_activation_enabled: bool = False,
    return_index_scores: bool = True,
    query_shard_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """Select complete causal pools and append each query's incomplete tail.

    q is [queries, batch, heads, dim], k/gate_score are [keys, batch, dim],
    and weights are [queries, batch, heads]. Packed bounds use global token
    coordinates. Output indices are [batch, queries, index_topk + pool_size - 1]
    when always_select_tail is enabled, with -1 for unused slots.
    Set return_index_scores=False to select in query chunks without retaining scores.

    query_shard_group splits the O(queries * pools) scoring across a process group whose
    ranks all hold identical inputs (e.g. the tensor-parallel group, whose ranks see the
    same gathered sequence). Each rank scores and selects pools only for its contiguous
    slice of query rows; the int32 pool IDs are then all-gathered. Every rank returns the
    same indices as an unsharded call. Requires return_index_scores=False.
    """
    shard_size = query_shard_group.size() if query_shard_group is not None else 1
    if shard_size > 1 and return_index_scores:
        raise ValueError("query_shard_group requires return_index_scores=False.")
    sk = k.size(0)

    # Packed pools restart at each document boundary.
    use_per_seg = cu_seqlens_kv is not None and cu_seqlens_kv.numel() >= 2
    if use_per_seg:
        k_pooled, pool_token_base = _kpool_compress_keys_per_seg(
            k, gate_score, ape, pool_size, cu_seqlens_kv
        )
        num_pools = k_pooled.size(0)
    else:
        num_pools = sk // pool_size
        k_pooled = _kpool_compress_keys(k, gate_score, ape, pool_size)
        pool_token_base = torch.arange(num_pools, device=k.device, dtype=torch.int64) * pool_size

    if fp8_indexer:
        q, k_pooled = _kpool_fp8_input(q), _kpool_fp8_input(k_pooled)
    elif rotate_activation_enabled:
        q = rotate_activation(q)
        if k_pooled.numel():
            k_pooled = rotate_activation(k_pooled)
    # A pool is causal only when its final token is within the query's bounds.
    pool_exists = pool_token_base >= 0
    pool_positions = pool_token_base.clamp_min(0) + (pool_size - 1)
    eff_key_positions = (
        key_positions[pool_positions] if key_positions is not None else pool_positions
    )
    v_starts, v_ends, k_pos = dsa_masking.normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=eff_key_positions,
        sk=num_pools,
        device=q.device,
    )
    if return_index_scores:
        index_scores = _compute_index_scores(q, weights, k_pooled, use_relu=use_relu)
        if v_starts is not None:
            index_scores = dsa_masking.apply_starts_ends_mask_to_scores(
                index_scores, v_starts, v_ends, k_pos
            )
        elif mask is not None:
            assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
            pool_exists = pool_exists & _pool_validity_from_token_mask(
                mask, pool_token_base, pool_size, sk
            )
        index_scores = index_scores.masked_fill(~pool_exists, float("-inf"))

        # Keep the selection width fixed, including when fewer causal pools exist.
        budget = index_topk // pool_size
        select_k = min(budget, num_pools)
        if select_k > 0:
            topk_scores, pool_topk = index_scores.topk(select_k, dim=-1)
            # [batch, seqlen_q, select_k] -> mask invalid pools
            pool_topk = pool_topk.masked_fill(topk_scores == float("-inf"), -1)
        else:
            pool_topk = torch.empty(
                index_scores.shape[:-1] + (0,), dtype=torch.int64, device=index_scores.device
            )
        if pool_topk.shape[-1] < budget:
            pad = torch.full(
                index_scores.shape[:-1] + (budget - pool_topk.shape[-1],),
                -1,
                dtype=torch.int64,
                device=index_scores.device,
            )
            pool_topk = torch.cat([pool_topk, pad], dim=-1)
    else:
        if mask is not None:
            assert mask.dtype == torch.float32, "mask dtype must be float32"
        index_scores = None
        query_start, query_end, rows_per_shard = _kpool_query_shard_bounds(
            q.size(0), shard_size, query_shard_group.rank() if shard_size > 1 else 0
        )
        pool_topk = _kpool_topk_from_score_chunks(
            q=q,
            k_pooled=k_pooled,
            weights=weights,
            index_topk=index_topk,
            pool_size=pool_size,
            pool_exists=pool_exists if use_per_seg else None,
            mask=mask,
            varlen_starts=v_starts,
            varlen_ends=v_ends,
            key_positions=k_pos,
            pool_token_base=pool_token_base,
            sk=sk,
            use_relu=use_relu,
            query_start=query_start,
            query_end=query_end,
        )
        if shard_size > 1:
            pool_topk = _all_gather_kpool_query_shards(
                pool_topk, q.size(0), rows_per_shard, query_shard_group
            )

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
        if v_ends is not None:
            ends = v_ends
        elif mask is not None:
            mask_starts, ends = _query_bounds_from_token_mask(mask)
        else:
            ends = torch.arange(1, sq + 1, device=q.device)
        if v_starts is not None:
            starts = v_starts
        elif mask is not None:
            starts = mask_starts
        elif cu_seqlens_kv is not None:
            cu = cu_seqlens_kv.to(device=q.device, dtype=torch.int64)
            starts = cu[torch.searchsorted(cu[1:], ends - 1, right=True)]
        else:
            starts = torch.zeros_like(ends)
        lengths = ends - starts
        if mask is not None and v_starts is None:
            # Explicit masks use globally aligned pools. Fill a leading partial pool into
            # unused expanded-pool slots, then append only the trailing global-grid tail.
            starts = _expand_query_bounds_for_batch(starts, batch, sq).to(
                device=token_topk.device, dtype=torch.int64
            )
            ends = _expand_query_bounds_for_batch(ends, batch, sq).to(
                device=token_topk.device, dtype=torch.int64
            )
            token_topk = _fill_kpool_left_boundary(
                token_topk.reshape(rows, -1), starts.reshape(-1), ends.reshape(-1), pool_size
            ).reshape(batch, sq, -1)
            complete_end = (ends // pool_size) * pool_size
            tail_starts = complete_end
            tail_counts = (ends - tail_starts).clamp_min(0).clamp_max(pool_size - 1)
        else:
            starts = _expand_query_bounds_for_batch(starts, batch, sq)
            ends = _expand_query_bounds_for_batch(ends, batch, sq)
            lengths = ends - starts
            tail_starts = ends - lengths.remainder(pool_size)
            tail_counts = None
        token_topk = _append_tail_to_topk(
            token_topk.reshape(rows, -1),
            lengths.reshape(-1),
            pool_size,
            tail_start_override=tail_starts.reshape(-1),
            tail_count_override=tail_counts.reshape(-1) if tail_counts is not None else None,
        ).reshape(batch, sq, -1)

    if mask is not None:
        token_topk = _mask_topk_tokens_with_token_mask(token_topk, mask)

    return index_scores, token_topk
