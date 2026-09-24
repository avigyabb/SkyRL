"""Prefix-shared training forward for the Megatron backend.

GRPO-style RL replicates every prompt ``n_samples_per_prompt`` times in the training batch, and
step-wise / multi-turn agentic trajectories replicate long shared histories. In the standard THD
sample-packing path every replica is a separate packed sequence, so the prompt (or history) goes
through every linear, MoE and attention layer once per replica.

For a causal LM the hidden state of a token depends only on the tokens (and positions) before it,
so identical token prefixes have identical activations. This module computes each shared prefix
exactly once:

* :func:`build_prefix_shared_layout` folds the rows of a micro-batch into a radix tree of token
  segments and emits them in DFS order as one packed token stream, with explicit per-token
  position ids and a ``row -> packed`` index map used to scatter per-token log-probs back into the
  ``[batch, seq]`` layout the loss code expects.
* :func:`prefix_shared_attention` is the "two-region" attention: causal flash-attention inside
  every segment plus a non-causal cross-attention from every branch onto its ancestor segments,
  merged through the log-sum-exp (forward) and fed back through both flash-attention backward
  kernels with the merged output/LSE (backward). Both regions run the stock flash-attn varlen
  CUDA kernels, so no new kernel has to be trusted; the branch->ancestor region reads a gathered
  copy of the ancestor K/V.
* :func:`install_prefix_sharing_patches` routes Megatron's TE attention module and its THD RoPE
  helper to the layout when a micro-batch carries one.

**Numerics contract.** Everything outside attention is bit-identical to the unshared packed path
for the same token (same weights, same position). Attention output for branch tokens is the
LSE-weighted merge of two flash-attention partial results computed in fp32 and rounded once to
the activation dtype, so per-token log-probs differ from the single-kernel path by bf16-level
noise (~1e-3 relative; see ``tests/backends/skyrl_train/gpu/gpu_ci/test_prefix_sharing.py`` for
the measured envelope). Gradients are the exact sum of the per-replica gradients up to the same
rounding. Record this deviation wherever training/inference execution contracts are tracked.

Requirements: ``trainer.remove_microbatch_padding=True``, Megatron backend, context parallel
size 1, no router replay, no MTP loss, no VLM inputs. Rows that cannot be shared fall back to a
one-segment-per-row layout inside the same code path, so mixed batches are fine.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from megatron.core.packed_seq_params import PackedSeqParams

__all__ = [
    "PrefixSharedLayout",
    "PrefixSharedPackedSeqParams",
    "build_prefix_shared_layout",
    "prefix_shared_attention",
    "reference_tree_attention",
    "install_prefix_sharing_patches",
    "scatter_packed_to_rows",
    "prefix_shared_logprobs_from_logits",
    "prefix_shared_logprobs_from_hidden",
    "prefix_shared_entropy_from_logits",
    "prefix_shared_entropy_from_hidden",
]


# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------


@dataclass
class PrefixSharedLayout:
    """A micro-batch folded into a DFS-ordered tree of token segments.

    All index tensors are ``int64`` except the flash-attn ``cu_seqlens`` (``int32``). Every
    segment is a contiguous span of the packed stream; a segment attends causally to itself and
    fully to every ancestor segment. The tail alignment padding (if any) is emitted as one extra
    root segment so attention writes finite values for it.
    """

    tokens: torch.Tensor  # [T] packed token ids
    position_ids: torch.Tensor  # [T] per-token RoPE position
    packed_targets: torch.Tensor  # [T] next-token target per packed position (0 where none)
    row_to_packed: torch.Tensor  # [B, S-1] packed index whose logits predict sequences[b, c+1]; -1 = none
    seg_start: torch.Tensor  # [N]
    seg_len: torch.Tensor  # [N]
    seg_parent: torch.Tensor  # [N] index into segments, -1 for roots
    cu_seqlens: torch.Tensor  # [N+1] int32, region 1 (causal within segment)
    q_index: torch.Tensor  # [Tq2] packed indices of branch tokens, region 2 queries
    cu_seqlens_q2: torch.Tensor  # [nb+1] int32
    kv_index: torch.Tensor  # [Tkv2] packed indices of ancestor tokens, region 2 keys/values
    cu_seqlens_kv2: torch.Tensor  # [nb+1] int32
    total_tokens: int
    real_tokens: int  # tokens before alignment padding
    max_seqlen: int  # longest segment (region 1)
    max_seqlen_q2: int
    max_seqlen_kv2: int
    max_position: int  # max position id + 1 (RoPE table length)
    num_rows: int
    row_tokens: int  # sum of valid tokens over rows (= unshared packed size)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_branches(self) -> bool:
        return int(self.cu_seqlens_q2.numel()) > 1

    def to(self, device) -> "PrefixSharedLayout":
        kwargs = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            kwargs[f.name] = v.to(device=device, non_blocking=True) if torch.is_tensor(v) else v
        return PrefixSharedLayout(**kwargs)

    def packed_seq_params(self) -> "PrefixSharedPackedSeqParams":
        """Megatron ``PackedSeqParams`` view of this layout.

        ``max_seqlen_*`` is the RoPE table length (largest position + 1), which is what
        ``GPTModel`` reads it for; attention never sees these values because
        :func:`install_prefix_sharing_patches` intercepts it. ``cu_seqlens_q_padded`` carries the
        per-token position ids as an attribute so the RoPE patch can find them from the argument
        Megatron already passes (this survives activation recompute, unlike a context manager).
        """
        cu = self.cu_seqlens
        cu.prefix_position_ids = self.position_ids  # type: ignore[attr-defined]
        return PrefixSharedPackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=self.max_position,
            max_seqlen_kv=self.max_position,
            total_tokens=self.total_tokens,
            prefix_layout=self,
        )


@dataclass
class PrefixSharedPackedSeqParams(PackedSeqParams):
    """``PackedSeqParams`` plus the layout. The extra field is not a base-class dataclass field,
    so Megatron's TE wrapper never forwards it to TransformerEngine."""

    prefix_layout: Optional[PrefixSharedLayout] = None


class _Node:
    __slots__ = ("ref", "a", "b", "parent", "children", "rows", "seg")

    def __init__(self, ref: int, a: int, b: int, parent: Optional["_Node"]):
        self.ref = ref  # a row whose valid tokens [a, b) spell this node
        self.a = a
        self.b = b
        self.parent = parent
        self.children: List["_Node"] = []
        self.rows: List[int] = []  # rows whose last valid token is the last token of this node
        self.seg = -1  # emitted segment index, -1 if the emission is empty


def _lcp(a: torch.Tensor, b: torch.Tensor) -> int:
    n = min(a.numel(), b.numel())
    if n == 0:
        return 0
    neq = (a[:n] != b[:n]).nonzero()
    return n if neq.numel() == 0 else int(neq[0])


def build_prefix_shared_layout(
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    align_size: int = 1,
    min_shared_tokens: int = 1,
) -> PrefixSharedLayout:
    """Fold ``[B, S]`` left/right-padded rows into a prefix-shared packed layout.

    Args:
        sequences: ``[B, S]`` token ids (CPU or GPU; the tree is built on the CPU).
        attention_mask: ``[B, S]`` valid-token mask. Valid tokens must be contiguous per row; the
            i-th valid token has position ``i`` (matches SkyRL's ``cumsum(mask) - 1`` positions).
        align_size: total packed length is padded up to a multiple of this (TP/FP8 alignment).
        min_shared_tokens: do not split an existing segment to share fewer than this many tokens
            (avoids fragmenting the stream over a handful of tokens two responses happen to share).

    The branching token trick: when rows diverge after a common prefix of length ``L``, the
    hidden state at position ``L-1`` is shared but its next-token target differs per row. The
    shared segment therefore ends at ``L-2`` and every child segment starts with its own copy of
    token ``L-1`` (one duplicated token per branch), so each packed position has exactly one
    target and the log-prob scatter is a plain gather.
    """
    if sequences.dim() != 2:
        raise ValueError(f"sequences must be [B, S], got {tuple(sequences.shape)}")
    seqs = sequences.detach().to("cpu", torch.int64)
    mask = attention_mask.detach().to("cpu", torch.bool)
    B, S = seqs.shape
    min_shared_tokens = max(1, int(min_shared_tokens))

    rows: List[torch.Tensor] = [seqs[r][mask[r]] for r in range(B)]
    row_lens = [int(t.numel()) for t in rows]
    top: List[_Node] = []
    all_nodes: List[_Node] = []

    def new_node(ref: int, a: int, b: int, parent: Optional[_Node]) -> _Node:
        n = _Node(ref, a, b, parent)
        all_nodes.append(n)
        return n

    for r in range(B):
        t = rows[r]
        L = row_lens[r]
        if L == 0:
            continue
        pos = 0
        parent: Optional[_Node] = None
        children = top
        while True:
            best: Optional[_Node] = None
            best_lcp = 0
            for c in children:
                lcp = _lcp(t[pos : c.b - c.a + pos], rows[c.ref][c.a : c.b])
                if lcp > best_lcp:
                    best, best_lcp = c, lcp
            if best is None:
                leaf = new_node(r, pos, L, parent)
                children.append(leaf)
                leaf.rows.append(r)
                break
            clen = best.b - best.a
            if best_lcp == clen:
                pos += clen
                if pos == L:
                    best.rows.append(r)
                    break
                parent, children = best, best.children
                continue
            if best_lcp < min_shared_tokens:
                leaf = new_node(r, pos, L, parent)
                children.append(leaf)
                leaf.rows.append(r)
                break
            # Split ``best`` at best_lcp: head keeps [a, a+lcp), tail takes the rest + children.
            tail = new_node(best.ref, best.a + best_lcp, best.b, best)
            tail.children = best.children
            for ch in tail.children:
                ch.parent = tail
            tail.rows = best.rows
            best.rows = []
            best.b = best.a + best_lcp
            best.children = [tail]
            if pos + best_lcp == L:
                best.rows.append(r)
            else:
                leaf = new_node(r, pos + best_lcp, L, best)
                best.children.append(leaf)
                leaf.rows.append(r)
            break

    # ---- emit segments in DFS preorder -------------------------------------------------
    seg_tokens: List[torch.Tensor] = []
    seg_targets: List[torch.Tensor] = []
    seg_pos: List[torch.Tensor] = []
    seg_start: List[int] = []
    seg_len: List[int] = []
    seg_parent: List[int] = []
    seg_estart: List[int] = []
    order: List[_Node] = []
    offset = 0
    stack = list(reversed(top))
    while stack:
        n = stack.pop()
        order.append(n)
        has_parent = n.parent is not None
        has_children = len(n.children) > 0
        e_start = n.a - (1 if has_parent else 0)
        e_end = n.b - (1 if has_children else 0)
        length = e_end - e_start
        if length > 0:
            ref = rows[n.ref]
            n.seg = len(seg_start)
            seg_tokens.append(ref[e_start:e_end])
            tgt = ref[e_start + 1 : e_end + 1]
            if tgt.numel() < length:  # leaf: the last position has no target
                tgt = torch.cat([tgt, torch.zeros(length - tgt.numel(), dtype=torch.int64)])
            seg_targets.append(tgt)
            seg_pos.append(torch.arange(e_start, e_end, dtype=torch.int64))
            seg_start.append(offset)
            seg_len.append(length)
            seg_estart.append(e_start)
            offset += length
        # parent segment index is resolved after emission (nearest emitted ancestor)
        stack.extend(reversed(n.children))

    real_tokens = offset
    total = int(math.ceil(real_tokens / align_size) * align_size) if align_size > 1 else real_tokens
    total = max(total, align_size if real_tokens == 0 else total)
    pad = total - real_tokens

    def nearest_emitted_ancestor(n: _Node) -> int:
        p = n.parent
        while p is not None:
            if p.seg >= 0:
                return p.seg
            p = p.parent
        return -1

    for n in order:
        if n.seg >= 0:
            seg_parent.append(nearest_emitted_ancestor(n))

    # region 2: every emitted segment with an emitted ancestor
    q_index_parts: List[torch.Tensor] = []
    kv_index_parts: List[torch.Tensor] = []
    cu_q2 = [0]
    cu_kv2 = [0]
    max_q2 = 0
    max_kv2 = 0
    for n in order:
        if n.seg < 0 or seg_parent[n.seg] < 0:
            continue
        s = n.seg
        q_index_parts.append(torch.arange(seg_start[s], seg_start[s] + seg_len[s], dtype=torch.int64))
        chain: List[int] = []
        p = seg_parent[s]
        while p >= 0:
            chain.append(p)
            p = seg_parent[p]
        chain.reverse()
        kv_len = 0
        for a in chain:
            kv_index_parts.append(torch.arange(seg_start[a], seg_start[a] + seg_len[a], dtype=torch.int64))
            kv_len += seg_len[a]
        cu_q2.append(cu_q2[-1] + seg_len[s])
        cu_kv2.append(cu_kv2[-1] + kv_len)
        max_q2 = max(max_q2, seg_len[s])
        max_kv2 = max(max_kv2, kv_len)

    if pad > 0:
        seg_tokens.append(torch.zeros(pad, dtype=torch.int64))
        seg_targets.append(torch.zeros(pad, dtype=torch.int64))
        seg_pos.append(torch.zeros(pad, dtype=torch.int64))
        seg_start.append(real_tokens)
        seg_len.append(pad)
        seg_parent.append(-1)

    tokens = torch.cat(seg_tokens) if seg_tokens else torch.zeros(0, dtype=torch.int64)
    targets = torch.cat(seg_targets) if seg_targets else torch.zeros(0, dtype=torch.int64)
    position_ids = torch.cat(seg_pos) if seg_pos else torch.zeros(0, dtype=torch.int64)
    cu = torch.zeros(len(seg_start) + 1, dtype=torch.int32)
    if seg_len:
        cu[1:] = torch.cumsum(torch.tensor(seg_len, dtype=torch.int64), 0).to(torch.int32)

    # ---- row -> packed map --------------------------------------------------------------
    row_to_packed = torch.full((B, max(S - 1, 0)), -1, dtype=torch.int64)
    for n in order:
        for r in n.rows:
            L = row_lens[r]
            need = L - 1  # positions 0..L-2 predict tokens 1..L-1
            if need <= 0:
                continue
            map_valid = torch.full((need,), -1, dtype=torch.int64)
            m: Optional[_Node] = n
            while m is not None:
                if m.seg >= 0:
                    s = m.seg
                    e_start = seg_estart[s]
                    e_end = min(e_start + seg_len[s], need)
                    if e_end > e_start:
                        map_valid[e_start:e_end] = torch.arange(
                            seg_start[s], seg_start[s] + (e_end - e_start), dtype=torch.int64
                        )
                m = m.parent
            cols = mask[r].nonzero().squeeze(1)[:need]
            row_to_packed[r, cols] = map_valid

    n_shared_rows = sum(len(n.rows) for n in all_nodes if n.parent is not None) if all_nodes else 0
    stats = {
        "rows": B,
        "row_tokens": int(sum(row_lens)),
        "packed_tokens": real_tokens,
        "segments": len(seg_start) - (1 if pad > 0 else 0),
        "branches": len(cu_q2) - 1,
        "rows_in_shared_subtrees": n_shared_rows,
        "max_depth": _tree_depth(top),
    }
    return PrefixSharedLayout(
        tokens=tokens,
        position_ids=position_ids,
        packed_targets=targets,
        row_to_packed=row_to_packed,
        seg_start=torch.tensor(seg_start, dtype=torch.int64),
        seg_len=torch.tensor(seg_len, dtype=torch.int64),
        seg_parent=torch.tensor(seg_parent, dtype=torch.int64),
        cu_seqlens=cu,
        q_index=torch.cat(q_index_parts) if q_index_parts else torch.zeros(0, dtype=torch.int64),
        cu_seqlens_q2=torch.tensor(cu_q2, dtype=torch.int32),
        kv_index=torch.cat(kv_index_parts) if kv_index_parts else torch.zeros(0, dtype=torch.int64),
        cu_seqlens_kv2=torch.tensor(cu_kv2, dtype=torch.int32),
        total_tokens=total,
        real_tokens=real_tokens,
        max_seqlen=max(seg_len) if seg_len else 1,
        max_seqlen_q2=max_q2,
        max_seqlen_kv2=max_kv2,
        max_position=int(position_ids.max().item()) + 1 if position_ids.numel() else 1,
        num_rows=B,
        row_tokens=int(sum(row_lens)),
        stats=stats,
    )


def _tree_depth(top: List[_Node]) -> int:
    best = 0
    stack = [(n, 1) for n in top]
    while stack:
        n, d = stack.pop()
        best = max(best, d)
        stack.extend((c, d + 1) for c in n.children)
    return best


# --------------------------------------------------------------------------------------
# Two-region attention on flash-attn varlen primitives
# --------------------------------------------------------------------------------------


def _fa():
    from flash_attn import flash_attn_interface as fai

    return fai


class _PrefixSharedAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, softmax_scale, deterministic, cu, max_seqlen, q_index, cu_q2, kv_index, cu_kv2, max_q2, max_kv2
    ):
        fai = _fa()
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        # Region 1: causal self-attention within every segment (including the pad tail).
        o1, lse1 = fai._flash_attn_varlen_forward(
            q, k, v, cu, cu, max_seqlen, max_seqlen, 0.0, softmax_scale, True, -1, -1, 0.0, None, False
        )[:2]
        if q_index.numel() == 0:
            ctx.save_for_backward(q, k, v, o1, lse1, cu, q_index, cu_q2, kv_index, cu_kv2)
            ctx.meta = (softmax_scale, deterministic, max_seqlen, max_q2, max_kv2)
            return o1
        # Region 2: branch tokens attend (non-causally) to their ancestors' K/V.
        q2 = q.index_select(0, q_index)
        k2 = k.index_select(0, kv_index)
        v2 = v.index_select(0, kv_index)
        o2, lse2 = fai._flash_attn_varlen_forward(
            q2, k2, v2, cu_q2, cu_kv2, max_q2, max_kv2, 0.0, softmax_scale, False, -1, -1, 0.0, None, False
        )[:2]
        # Merge in fp32: o = (o1*e^lse1 + o2*e^lse2) / (e^lse1 + e^lse2), lse = logaddexp.
        lse1_b = lse1[:, q_index]  # [H, Tq2]
        lse = torch.logaddexp(lse1_b, lse2)
        w1 = torch.exp(lse1_b - lse).t().unsqueeze(-1)  # [Tq2, H, 1]
        w2 = torch.exp(lse2 - lse).t().unsqueeze(-1)
        o_b = (o1.index_select(0, q_index).float() * w1 + o2.float() * w2).to(q.dtype)
        o = o1.index_copy(0, q_index, o_b)
        lse_full = lse1.index_copy(1, q_index, lse)
        ctx.save_for_backward(q, k, v, o, lse_full, cu, q_index, cu_q2, kv_index, cu_kv2)
        ctx.meta = (softmax_scale, deterministic, max_seqlen, max_q2, max_kv2)
        return o

    @staticmethod
    def backward(ctx, do):
        fai = _fa()
        q, k, v, o, lse, cu, q_index, cu_q2, kv_index, cu_kv2 = ctx.saved_tensors
        softmax_scale, deterministic, max_seqlen, max_q2, max_kv2 = ctx.meta
        do = do.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        # Region 1 backward against the *merged* output / LSE: P_ij = exp(s_ij - lse_i) is then
        # the true softmax probability over the union of both regions, so dS is exact.
        fai._flash_attn_varlen_backward(
            do,
            q,
            k,
            v,
            o,
            lse,
            dq,
            dk,
            dv,
            cu,
            cu,
            max_seqlen,
            max_seqlen,
            0.0,
            softmax_scale,
            True,
            -1,
            -1,
            0.0,
            None,
            deterministic,
            None,
        )
        if q_index.numel() > 0:
            q2 = q.index_select(0, q_index)
            k2 = k.index_select(0, kv_index)
            v2 = v.index_select(0, kv_index)
            o2 = o.index_select(0, q_index)
            do2 = do.index_select(0, q_index)
            lse2 = lse[:, q_index].contiguous()
            dq2 = torch.empty_like(q2)
            dk2 = torch.empty_like(k2)
            dv2 = torch.empty_like(v2)
            fai._flash_attn_varlen_backward(
                do2,
                q2,
                k2,
                v2,
                o2,
                lse2,
                dq2,
                dk2,
                dv2,
                cu_q2,
                cu_kv2,
                max_q2,
                max_kv2,
                0.0,
                softmax_scale,
                False,
                -1,
                -1,
                0.0,
                None,
                deterministic,
                None,
            )
            dq.index_add_(0, q_index, dq2)
            dk.index_add_(0, kv_index, dk2)
            dv.index_add_(0, kv_index, dv2)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def prefix_shared_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: PrefixSharedLayout,
    softmax_scale: Optional[float] = None,
    deterministic: bool = False,
) -> torch.Tensor:
    """Tree-masked attention over a :class:`PrefixSharedLayout`.

    ``q``: ``[T, H, D]``, ``k``/``v``: ``[T, Hkv, D]`` (GQA allowed), all in the packed order of the
    layout. Returns ``[T, H, D]``. Position ``i`` attends to positions ``j <= i`` of its own segment
    and to every token of its ancestor segments, which is exactly the causal mask each original
    row would have seen.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    return _PrefixSharedAttention.apply(
        q,
        k,
        v,
        float(softmax_scale),
        bool(deterministic),
        layout.cu_seqlens,
        int(layout.max_seqlen),
        layout.q_index,
        layout.cu_seqlens_q2,
        layout.kv_index,
        layout.cu_seqlens_kv2,
        int(layout.max_seqlen_q2),
        int(layout.max_seqlen_kv2),
    )


def tree_attention_mask(layout: PrefixSharedLayout, device=None) -> torch.Tensor:
    """Dense ``[T, T]`` boolean mask (query i may attend key j) for tests/reference."""
    T = layout.total_tokens
    seg_start = layout.seg_start.tolist()
    seg_len = layout.seg_len.tolist()
    seg_parent = layout.seg_parent.tolist()
    allowed = torch.zeros(T, T, dtype=torch.bool)
    for s, (st, ln) in enumerate(zip(seg_start, seg_len)):
        idx = torch.arange(st, st + ln)
        allowed[st : st + ln, st : st + ln] = torch.tril(torch.ones(ln, ln, dtype=torch.bool))
        p = seg_parent[s]
        while p >= 0:
            allowed[idx.unsqueeze(1), torch.arange(seg_start[p], seg_start[p] + seg_len[p]).unsqueeze(0)] = True
            p = seg_parent[p]
    return allowed.to(device) if device is not None else allowed


def reference_tree_attention(q, k, v, layout: PrefixSharedLayout, softmax_scale=None) -> torch.Tensor:
    """fp32 dense reference for :func:`prefix_shared_attention`."""
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    H, Hkv = q.shape[1], k.shape[1]
    rep = H // Hkv
    kf = k.float().repeat_interleave(rep, dim=1)
    vf = v.float().repeat_interleave(rep, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.float(), kf) * softmax_scale
    mask = tree_attention_mask(layout, q.device)
    s = s.masked_fill(~mask.unsqueeze(0), float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("hqk,khd->qhd", p, vf)


# --------------------------------------------------------------------------------------
# Megatron / TE patches
# --------------------------------------------------------------------------------------

_PATCHED = False


def install_prefix_sharing_patches() -> None:
    """Route Megatron's attention + THD RoPE to the layout when a micro-batch carries one.

    Idempotent. Both patches are pass-through unless the ``packed_seq_params`` is a
    :class:`PrefixSharedPackedSeqParams` (attention) or the ``cu_seqlens`` tensor carries
    ``prefix_position_ids`` (RoPE), so unshared micro-batches behave exactly as before.
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.models.common.embeddings import rope_utils

    orig_te_forward = TEDotProductAttention.forward

    def te_forward(
        self, query, key, value, attention_mask, attn_mask_type, attention_bias=None, packed_seq_params=None, **kwargs
    ):
        layout = getattr(packed_seq_params, "prefix_layout", None)
        if layout is None:
            return orig_te_forward(
                self,
                query,
                key,
                value,
                attention_mask,
                attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                **kwargs,
            )
        if attention_bias is not None:
            raise NotImplementedError("prefix sharing does not support attention_bias")
        if getattr(self.config, "window_size", None) is not None:
            raise NotImplementedError("prefix sharing does not support sliding-window attention")
        if self.training and float(getattr(self.config, "attention_dropout", 0.0) or 0.0) > 0.0:
            raise NotImplementedError("prefix sharing requires attention_dropout=0")
        if query.dim() != 3:
            raise ValueError(f"expected THD [T, H, D] query for prefix sharing, got {tuple(query.shape)}")
        scale = getattr(self.config, "softmax_scale", None)
        if scale is None:
            scale = 1.0 / math.sqrt(query.shape[-1])
        deterministic = bool(getattr(self.config, "deterministic_mode", False))
        out = prefix_shared_attention(query, key, value, layout, softmax_scale=scale, deterministic=deterministic)
        return out.reshape(out.shape[0], -1)

    TEDotProductAttention.forward = te_forward

    orig_apply = rope_utils.apply_rotary_pos_emb

    def apply_rotary_pos_emb(
        t,
        freqs,
        config,
        cu_seqlens=None,
        mscale=1.0,
        cp_group=None,
        mla_rotary_interleaved=False,
        inverse=False,
        mla_output_remove_interleaving=False,
        max_seqlen=None,
        **kwargs,
    ):
        pos = getattr(cu_seqlens, "prefix_position_ids", None) if cu_seqlens is not None else None
        if pos is None:
            return orig_apply(
                t,
                freqs,
                config,
                cu_seqlens=cu_seqlens,
                mscale=mscale,
                cp_group=cp_group,
                mla_rotary_interleaved=mla_rotary_interleaved,
                inverse=inverse,
                mla_output_remove_interleaving=mla_output_remove_interleaving,
                max_seqlen=max_seqlen,
                **kwargs,
            )
        # t: [T, H, D] (thd). Gather one frequency row per token from the [max_pos, 1, 1, D] table.
        freqs_packed = freqs.index_select(0, pos.clamp(max=freqs.shape[0] - 1))
        if mla_rotary_interleaved is None:
            mla_rotary_interleaved = getattr(config, "multi_latent_attention", False)
        fused = getattr(rope_utils, "fused_apply_rotary_pos_emb", None)
        if (
            getattr(config, "apply_rope_fusion", False)
            and fused is not None
            and mscale == 1.0
            and not mla_rotary_interleaved
            and not inverse
        ):
            return fused(t.unsqueeze(1), freqs_packed, interleaved=config.rotary_interleaved).squeeze(1)
        return rope_utils._apply_rotary_pos_emb_bshd(
            t.unsqueeze(1),
            freqs_packed,
            rotary_interleaved=config.rotary_interleaved,
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
            inverse=inverse,
            mla_output_remove_interleaving=mla_output_remove_interleaving,
        ).squeeze(1)

    rope_utils.apply_rotary_pos_emb = apply_rotary_pos_emb
    import importlib
    import sys

    for mod_name in list(sys.modules):
        if not mod_name.startswith("megatron."):
            continue
        mod = sys.modules.get(mod_name)
        if mod is not None and getattr(mod, "apply_rotary_pos_emb", None) is orig_apply:
            mod.apply_rotary_pos_emb = apply_rotary_pos_emb
    # Modules imported later pick the patched name up from rope_utils; make sure the attention
    # module is loaded now so the loop above covered it.
    importlib.import_module("megatron.core.transformer.attention")
    attn_mod = sys.modules["megatron.core.transformer.attention"]
    if getattr(attn_mod, "apply_rotary_pos_emb", None) is orig_apply:
        attn_mod.apply_rotary_pos_emb = apply_rotary_pos_emb


# --------------------------------------------------------------------------------------
# Log-probs / entropy in the shared layout
# --------------------------------------------------------------------------------------


def scatter_packed_to_rows(values: torch.Tensor, row_to_packed: torch.Tensor) -> torch.Tensor:
    """``[T]`` per-packed-token values -> ``[B, S-1]`` rows (0 where a row has no token).

    A shared packed position feeds several rows; autograd's gather backward accumulates the rows'
    gradients into it, which is exactly the sum the unshared replicas would have produced.
    """
    valid = row_to_packed >= 0
    gathered = values[row_to_packed.clamp(min=0)]
    return torch.where(valid, gathered, torch.zeros((), dtype=gathered.dtype, device=gathered.device))


def _packed_logprobs(
    vocab_parallel_logits, targets, vocab_start_index, vocab_end_index, group, inference_only, chunk_size
):
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        ChunkedDistributedLogprob,
        DistributedLogprob,
    )

    seq_len = vocab_parallel_logits.shape[1]
    if chunk_size is not None and chunk_size < seq_len:
        probs = ChunkedDistributedLogprob.apply(
            vocab_parallel_logits, targets, vocab_start_index, vocab_end_index, chunk_size, group, inference_only
        )
    else:
        probs = DistributedLogprob.apply(
            vocab_parallel_logits, targets, vocab_start_index, vocab_end_index, group, inference_only
        )
    return probs.squeeze(0).contiguous()


def prefix_shared_logprobs_from_logits(
    vocab_parallel_logits: torch.Tensor,
    layout: PrefixSharedLayout,
    vocab_start_index: int,
    vocab_end_index: int,
    group,
    inference_only: bool = False,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """``[1, T, V/TP]`` packed logits -> ``[B, S-1]`` per-token log-probs of the next token."""
    probs = _packed_logprobs(
        vocab_parallel_logits,
        layout.packed_targets.unsqueeze(0),
        vocab_start_index,
        vocab_end_index,
        group,
        inference_only,
        chunk_size,
    )
    return scatter_packed_to_rows(probs, layout.row_to_packed)


def prefix_shared_logprobs_from_hidden(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    layout: PrefixSharedLayout,
    vocab_start_index: int,
    vocab_end_index: int,
    group,
    inference_only: bool = False,
    chunk_size: Optional[int] = None,
    temperature: float = 1.0,
    fused_backend: str = "torch",
) -> torch.Tensor:
    """Fused-LM-head variant of :func:`prefix_shared_logprobs_from_logits` (``hidden`` ``[1, T, H]``)."""
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        _fused_lm_head_logprob_apply,
    )

    if temperature != 1.0:
        lm_head_weight = lm_head_weight / temperature
    seq_len = hidden.shape[1]
    eff_chunk = chunk_size if (chunk_size is not None and chunk_size < seq_len) else seq_len
    probs = (
        _fused_lm_head_logprob_apply(
            fused_backend,
            hidden,
            lm_head_weight,
            layout.packed_targets.unsqueeze(0),
            vocab_start_index,
            vocab_end_index,
            eff_chunk,
            group,
            inference_only,
        )
        .squeeze(0)
        .contiguous()
    )
    return scatter_packed_to_rows(probs, layout.row_to_packed)


def _packed_action_weights(layout: PrefixSharedLayout, num_actions: int, loss_mask, dtype, device) -> torch.Tensor:
    """Per-packed-token weight = number of (row, action position) pairs it stands in for."""
    B, Sm1 = layout.row_to_packed.shape
    action_weights = torch.zeros((B, Sm1), dtype=dtype, device=device)
    if loss_mask is None:
        action_weights[:, -num_actions:] = 1.0
    else:
        action_weights[:, -num_actions:] = loss_mask.to(device=device, dtype=dtype)
    r2p = layout.row_to_packed.to(device)
    valid = r2p >= 0
    packed_weights = torch.zeros((layout.total_tokens,), dtype=dtype, device=device)
    packed_weights.index_add_(0, r2p[valid], action_weights[valid])
    return packed_weights


def prefix_shared_entropy_from_logits(
    vocab_parallel_logits: torch.Tensor,
    layout: PrefixSharedLayout,
    num_actions: int,
    loss_mask,
    chunk_size: Optional[int] = None,
    chunk_memory_mb: int = 512,
):
    """Action-token entropy on packed TP-sharded logits. Returns (metric, loss term) like
    :func:`model_utils.vocab_parallel_entropy_packed_sequences` (context parallel size 1)."""
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        vocab_parallel_entropy_weighted_sum,
    )

    weights = _packed_action_weights(
        layout, num_actions, loss_mask, vocab_parallel_logits.dtype, vocab_parallel_logits.device
    )
    entropy_sum = vocab_parallel_entropy_weighted_sum(
        vocab_parallel_logits, weights, chunk_size=chunk_size, chunk_memory_mb=chunk_memory_mb
    )
    count = weights.sum().clamp(min=1.0)
    return (entropy_sum.detach() / count), entropy_sum / count


def prefix_shared_entropy_from_hidden(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    layout: PrefixSharedLayout,
    num_actions: int,
    loss_mask,
    tp_group,
    chunk_size: Optional[int] = None,
    temperature: float = 1.0,
):
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        _fused_vocab_parallel_entropy_from_hidden,
    )

    entropy_tokens = _fused_vocab_parallel_entropy_from_hidden(
        hidden, lm_head_weight, tp_group, chunk_size=chunk_size, temperature=temperature
    ).squeeze(0)
    weights = _packed_action_weights(layout, num_actions, loss_mask, entropy_tokens.dtype, entropy_tokens.device)
    entropy_sum = (entropy_tokens * weights).sum()
    count = weights.sum().clamp(min=1.0)
    return (entropy_sum.detach() / count), entropy_sum / count
