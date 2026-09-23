"""Unit tests for the prefix-shared training forward (layout + two-region attention).

Run (single GPU, no Ray needed):
    uv run --isolated --extra dev --extra megatron pytest \
        tests/backends/skyrl_train/gpu/gpu_ci/test_prefix_sharing.py -x -q
"""

import math

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.prefix_sharing import (
    build_prefix_shared_layout,
    prefix_shared_attention,
    reference_tree_attention,
    scatter_packed_to_rows,
    tree_attention_mask,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _make_rows(B, S, prompt_len, resp_lens, n_groups, seed=0, vocab=1000, left_pad=None):
    """GRPO-shaped rows: n_groups prompts x (B // n_groups) responses, left-padded prompt, right-padded response."""
    g = torch.Generator().manual_seed(seed)
    per = B // n_groups
    seqs = torch.zeros(B, S, dtype=torch.long)
    mask = torch.zeros(B, S, dtype=torch.bool)
    for gi in range(n_groups):
        prompt = torch.randint(1, vocab, (prompt_len,), generator=g)
        for j in range(per):
            r = gi * per + j
            L_resp = resp_lens[r % len(resp_lens)]
            resp = torch.randint(1, vocab, (L_resp,), generator=g)
            lp = (S - prompt_len - max(resp_lens)) if left_pad is None else left_pad
            seqs[r, lp : lp + prompt_len] = prompt
            seqs[r, lp + prompt_len : lp + prompt_len + L_resp] = resp
            mask[r, lp : lp + prompt_len + L_resp] = True
    return seqs, mask


def _check_layout_matches_rows(layout, seqs, mask):
    """Every row position must map to a packed token with the same token id, position and
    next-token target, and the packed token must see exactly the row's prefix."""
    B, S = seqs.shape
    allowed = tree_attention_mask(layout)
    for r in range(B):
        toks = seqs[r][mask[r]]
        cols = mask[r].nonzero().squeeze(1)
        L = toks.numel()
        for t in range(L - 1):
            j = int(layout.row_to_packed[r, cols[t]])
            assert j >= 0, (r, t)
            assert int(layout.tokens[j]) == int(toks[t])
            assert int(layout.position_ids[j]) == t
            assert int(layout.packed_targets[j]) == int(toks[t + 1])
            vis = allowed[j].nonzero().squeeze(1)
            seen = sorted((int(layout.position_ids[i]), int(layout.tokens[i])) for i in vis.tolist())
            assert seen == [(i, int(toks[i])) for i in range(t + 1)], (r, t)
        # columns beyond the row's valid range map nowhere
        invalid = torch.ones(S - 1, dtype=torch.bool)
        invalid[cols[: L - 1]] = False
        assert (layout.row_to_packed[r][invalid] == -1).all()


@pytest.mark.parametrize("min_shared", [1, 16])
def test_layout_grpo_groups(min_shared):
    B, S = 8, 96
    seqs, mask = _make_rows(B, S, prompt_len=40, resp_lens=[20, 33, 7, 41], n_groups=2)
    layout = build_prefix_shared_layout(seqs, mask, align_size=8, min_shared_tokens=min_shared)
    assert layout.total_tokens % 8 == 0
    _check_layout_matches_rows(layout, seqs, mask)
    # two prompts shared across 4 rows each: packed size ~ 2*(P-1) + sum(1 + resp)
    assert layout.real_tokens < layout.row_tokens
    assert layout.stats["branches"] >= 8


def test_layout_multiturn_chain_and_prefix_rows():
    """Row k is a strict prefix of row k+1 (step-wise trajectories), plus an unrelated row."""
    g = torch.Generator().manual_seed(1)
    base = torch.randint(1, 500, (60,), generator=g)
    S = 64
    rows = [base[:20], base[:35], base[:60], torch.randint(1, 500, (30,), generator=g)]
    seqs = torch.zeros(len(rows), S, dtype=torch.long)
    mask = torch.zeros(len(rows), S, dtype=torch.bool)
    for r, t in enumerate(rows):  # right-padded
        seqs[r, : t.numel()] = t
        mask[r, : t.numel()] = True
    layout = build_prefix_shared_layout(seqs, mask, align_size=4, min_shared_tokens=1)
    _check_layout_matches_rows(layout, seqs, mask)
    assert layout.stats["max_depth"] >= 3
    # nothing is replicated beyond the branching tokens
    assert layout.real_tokens <= 60 + 30 + 4


def test_layout_no_sharing_is_plain_packing():
    g = torch.Generator().manual_seed(2)
    seqs = torch.randint(1, 500, (3, 20), generator=g)
    mask = torch.ones(3, 20, dtype=torch.bool)
    layout = build_prefix_shared_layout(seqs, mask, align_size=1)
    assert not layout.has_branches
    assert layout.real_tokens == 60
    _check_layout_matches_rows(layout, seqs, mask)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("H,Hkv,D", [(8, 2, 128), (4, 4, 64)])
def test_two_region_attention_matches_dense_reference(dtype, H, Hkv, D):
    torch.manual_seed(0)
    B, S = 8, 200
    seqs, mask = _make_rows(B, S, prompt_len=70, resp_lens=[50, 61, 17, 90], n_groups=2, seed=3)
    layout = build_prefix_shared_layout(seqs, mask, align_size=16).to("cuda")
    T = layout.total_tokens
    q = torch.randn(T, H, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(T, Hkv, D, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(T, Hkv, D, device="cuda", dtype=dtype, requires_grad=True)

    out = prefix_shared_attention(q, k, v, layout)
    ref = reference_tree_attention(q.detach(), k.detach(), v.detach(), layout)
    tol = 2e-2 if dtype == torch.bfloat16 else 8e-3
    assert torch.allclose(out.float(), ref, atol=tol, rtol=tol), (out.float() - ref).abs().max()

    do = torch.randn_like(out)
    out.backward(do)
    q32 = q.detach().float().requires_grad_(True)
    k32 = k.detach().float().requires_grad_(True)
    v32 = v.detach().float().requires_grad_(True)
    reference_tree_attention(q32, k32, v32, layout).backward(do.float())
    for a, b, name in ((q.grad, q32.grad, "dq"), (k.grad, k32.grad, "dk"), (v.grad, v32.grad, "dv")):
        err = (a.float() - b).abs().max().item()
        scale = b.abs().max().item()
        assert err <= tol * max(1.0, scale), (name, err, scale)


def test_two_region_attention_matches_unshared_flash_attention():
    """Sharing must reproduce what flash-attn computes on the replicated (unshared) sequences."""
    from flash_attn import flash_attn_interface as fai

    torch.manual_seed(0)
    H, Hkv, D = 8, 2, 128
    B, S = 6, 300
    seqs, mask = _make_rows(B, S, prompt_len=150, resp_lens=[100, 120, 60], n_groups=2, seed=5)
    layout = build_prefix_shared_layout(seqs, mask, align_size=8).to("cuda")
    T = layout.total_tokens
    # Per-(position, token) K/V/Q so that shared tokens have identical projections in both layouts.
    # Emulate "activations are a function of the row prefix" by hashing (row-group, position).
    # Simplest faithful construction: draw per packed token, then expand to rows via row_to_packed.
    q_p = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
    k_p = torch.randn(T, Hkv, D, device="cuda", dtype=torch.bfloat16)
    v_p = torch.randn(T, Hkv, D, device="cuda", dtype=torch.bfloat16)
    out_shared = prefix_shared_attention(q_p, k_p, v_p, layout)

    # Unshared: build each row's full sequence from the packed tokens it maps to (+ the last token).
    r2p = layout.row_to_packed.cpu()
    for r in range(B):
        cols = mask[r].nonzero().squeeze(1)
        L = cols.numel()
        idx = r2p[r, cols[: L - 1]]
        assert (idx >= 0).all()
        # the last valid token of a row is never a query anyone else needs; skip it.
        q_r, k_r, v_r = q_p[idx], k_p[idx], v_p[idx]
        cu = torch.tensor([0, L - 1], device="cuda", dtype=torch.int32)
        out_r = fai.flash_attn_varlen_func(q_r, k_r, v_r, cu, cu, L - 1, L - 1, causal=True)
        err = (out_r.float() - out_shared[idx].float()).abs().max().item()
        assert err < 2e-2, (r, err)


def test_scatter_packed_to_rows_accumulates_shared_grads():
    vals = torch.arange(5, dtype=torch.float32, requires_grad=True)
    r2p = torch.tensor([[0, 1, 2, -1], [0, 1, 3, 4]])
    rows = scatter_packed_to_rows(vals, r2p)
    assert rows.tolist() == [[0, 1, 2, 0], [0, 1, 3, 4]]
    rows.sum().backward()
    assert vals.grad.tolist() == [2, 2, 1, 1, 1]


def test_attention_scale_argument():
    torch.manual_seed(0)
    seqs, mask = _make_rows(4, 64, prompt_len=30, resp_lens=[20, 10], n_groups=1, seed=7)
    layout = build_prefix_shared_layout(seqs, mask, align_size=8).to("cuda")
    T = layout.total_tokens
    q = torch.randn(T, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(T, 4, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(T, 4, 64, device="cuda", dtype=torch.bfloat16)
    scale = 0.5 / math.sqrt(64)
    out = prefix_shared_attention(q, k, v, layout, softmax_scale=scale)
    ref = reference_tree_attention(q, k, v, layout, softmax_scale=scale)
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_two_region_backward_matches_unshared_flash_attention_grads():
    """dq of branch tokens must equal FA2's dq on the replicated rows; dk/dv of a shared prefix token
    must equal the *sum* of FA2's dk/dv over the G replicas (same kernels, so only accumulation-order
    noise remains)."""
    from flash_attn import flash_attn_interface as fai

    torch.manual_seed(1)
    H, Hkv, D = 8, 2, 128
    B, S = 6, 300
    seqs, mask = _make_rows(B, S, prompt_len=150, resp_lens=[100, 120, 60], n_groups=2, seed=11)
    layout = build_prefix_shared_layout(seqs, mask, align_size=8).to("cuda")
    T = layout.total_tokens
    q_p = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k_p = torch.randn(T, Hkv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v_p = torch.randn(T, Hkv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    do_p = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
    out = prefix_shared_attention(q_p, k_p, v_p, layout)
    out.backward(do_p)

    dq_ref = torch.zeros_like(q_p)
    dk_ref = torch.zeros_like(k_p)
    dv_ref = torch.zeros_like(v_p)
    covered_mask = torch.ones(T, dtype=torch.bool, device="cuda")  # True = dO not yet delivered
    r2p = layout.row_to_packed.cpu()
    all_idx = []
    for r in range(B):
        cols = mask[r].nonzero().squeeze(1)
        L = cols.numel()
        # positions 0..L-2 via the map; the row's last token sits right after position L-2 in its leaf segment
        idx = torch.cat([r2p[r, cols[: L - 1]], r2p[r, cols[L - 2]].view(1) + 1]).cuda()
        all_idx.append(idx)
        q_r = q_p.detach()[idx].requires_grad_(True)
        k_r = k_p.detach()[idx].requires_grad_(True)
        v_r = v_p.detach()[idx].requires_grad_(True)
        cu = torch.tensor([0, L], device="cuda", dtype=torch.int32)
        # The shared layout consumes a shared token's attention output once, so its upstream dO must
        # reach exactly one replica; zero it for rows that already covered the token.
        do_r = do_p[idx].clone()
        do_r[~covered_mask[idx]] = 0
        covered_mask[idx] = False
        o_r = fai.flash_attn_varlen_func(q_r, k_r, v_r, cu, cu, L, L, causal=True)
        o_r.backward(do_r)
        dq_ref.index_add_(0, idx, q_r.grad)
        dk_ref.index_add_(0, idx, k_r.grad)
        dv_ref.index_add_(0, idx, v_r.grad)

    covered = torch.cat(all_idx).unique()
    for got, ref, name, tol in (
        (q_p.grad, dq_ref, "dq", 3e-2),
        (k_p.grad, dk_ref, "dk", 3e-2),
        (v_p.grad, dv_ref, "dv", 3e-2),
    ):
        g, rf = got[covered].float(), ref[covered].float()
        err = (g - rf).abs().max().item()
        assert err <= tol * max(1.0, rf.abs().max().item()), (name, err, rf.abs().max().item())
