"""Single-GPU micro-benchmark: two-region prefix-shared attention vs flash-attn on replicated rows.

For one GRPO group (prompt P, G responses of R tokens) it times forward+backward of
``prefix_shared_attention`` over the shared layout against ``flash_attn_varlen_func`` (causal) over
the G unshared sequences, for a Qwen3-30B-A3B-like head configuration (TP=8 shard: 4 query heads,
1 KV head, head_dim 128) and a dense-model shard (TP=8 of 64 heads: 8 q heads, 1 KV head).

Usage:
    PYTHONPATH=. .venv/bin/python skyrl/benchmarks/bench_prefix_attention.py
"""

import argparse
import time

import torch

from skyrl.backends.skyrl_train.distributed.megatron.prefix_sharing import (
    build_prefix_shared_layout,
    prefix_shared_attention,
)


def make_rows(P, R, G, vocab=1000, seed=0):
    gen = torch.Generator().manual_seed(seed)
    S = P + R
    seqs = torch.zeros(G, S, dtype=torch.long)
    mask = torch.ones(G, S, dtype=torch.bool)
    prompt = torch.randint(1, vocab, (P,), generator=gen)
    for j in range(G):
        seqs[j, :P] = prompt
        seqs[j, P:] = torch.randint(1, vocab, (R,), generator=gen)
    return seqs, mask


def bench(fn, reps=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def causal_flops(lens, H, D):
    # QK^T and PV, causal half, forward only: 4 * L^2/2 * H * D per sequence
    return sum(4 * (L * L / 2) * H * D for L in lens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shapes", nargs="+", default=["1024:2048:8", "4096:1024:8", "4096:2048:16", "8192:1024:16", "16384:512:16"]
    )
    ap.add_argument("--heads", nargs="+", default=["4:1:128", "8:1:128"], help="H:Hkv:D per TP shard")
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    from flash_attn import flash_attn_interface as fai

    print(
        "| heads (H:Hkv:D) | P | R | G | unshared FA2 fwd+bwd | shared two-region fwd+bwd | speedup | attention FLOP ratio |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for hs in args.heads:
        H, Hkv, D = (int(x) for x in hs.split(":"))
        for shp in args.shapes:
            P, R, G = (int(x) for x in shp.split(":"))
            seqs, mask = make_rows(P, R, G)
            layout = build_prefix_shared_layout(seqs, mask, align_size=8).to("cuda")
            T = layout.total_tokens
            q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(T, Hkv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn(T, Hkv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            do = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)

            def shared():
                out = prefix_shared_attention(q, k, v, layout)
                out.backward(do)

            L = P + R
            Tu = G * L
            qu = torch.randn(Tu, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            ku = torch.randn(Tu, Hkv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            vu = torch.randn(Tu, Hkv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            dou = torch.randn(Tu, H, D, device="cuda", dtype=torch.bfloat16)
            cu = torch.arange(0, Tu + 1, L, device="cuda", dtype=torch.int32)

            def unshared():
                out = fai.flash_attn_varlen_func(qu, ku, vu, cu, cu, L, L, causal=True)
                out.backward(dou)

            t_s = bench(shared, args.reps)
            t_u = bench(unshared, args.reps)
            f_u = causal_flops([L] * G, H, D)
            f_s = causal_flops([P - 1] + [R + 1] * G, H, D) + G * 4 * (R + 1) * (P - 1) * H * D
            print(
                f"| {hs} | {P} | {R} | {G} | {t_u * 1e3:.2f} ms | {t_s * 1e3:.2f} ms | {t_u / t_s:.2f}x | {f_s / f_u:.2f} |",
                flush=True,
            )
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
