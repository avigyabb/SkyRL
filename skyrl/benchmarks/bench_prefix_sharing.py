"""End-to-end A/B of the prefix-shared training forward on a real Megatron policy worker.

For every ``P:R:G`` shape (prompt tokens, response tokens, samples per prompt) the script builds a
synthetic GRPO mini-batch of ``--groups`` prompts x ``G`` samples, then on the *same loaded model*:

  1. correctness: per-token log-probs from ``forward`` (unshared vs shared) and the gradient norm of
     one PPO ``forward_backward`` + ``optim_step`` (lr=0, so weights never move) in both modes;
  2. timing: wall time of the policy update (``forward_backward`` over the mini-batch) and of the
     log-prob recompute (``forward``) in both modes.

Rows reach the worker in ``--baseline-rows`` / ``--shared-rows`` row chunks (one Megatron
micro-batch each); ``auto`` gives the baseline the same token budget per micro-batch as one shared
group so both modes see comparable activation memory.

Usage (8xH100, Qwen3-30B-A3B, TP=8/EP=8):
    PYTHONPATH=. .venv/bin/python skyrl/benchmarks/bench_prefix_sharing.py \
        --model /path/to/Qwen3-30B-A3B --gpus 8 --tp 8 --ep 8 --etp 1 --offload-optimizer \
        --shapes 1024:2048:8 4096:1024:8 4096:2048:16 8192:1024:16 --groups 2 --reps 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import ray
import torch


def parse_shape(s: str):
    p, r, g = (int(x) for x in s.split(":"))
    return p, r, g


def make_grpo_batch(n_groups: int, G: int, P: int, R: int, vocab: int, seed: int):
    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

    gen = torch.Generator().manual_seed(seed)
    B, S = n_groups * G, P + R
    seqs = torch.zeros(B, S, dtype=torch.long)
    mask = torch.zeros(B, S, dtype=torch.long)
    resp_mask = torch.zeros(B, R, dtype=torch.long)
    for gi in range(n_groups):
        prompt = torch.randint(1, vocab, (P,), generator=gen)
        for j in range(G):
            r = gi * G + j
            rl = R  # full-length responses so the per-sample logprob lists cover all R action slots
            seqs[r, :P] = prompt
            seqs[r, P : P + rl] = torch.randint(1, vocab, (rl,), generator=gen)
            mask[r, : P + rl] = 1
            resp_mask[r, :rl] = 1
    adv = torch.randn(B, R, generator=gen) * resp_mask
    batch = TrainingInputBatch(
        {
            "sequences": seqs,
            "attention_mask": mask,
            "action_log_probs": torch.zeros(B, R),
            "base_action_log_probs": torch.zeros(B, R),
            "rollout_logprobs": torch.zeros(B, R),
            "values": torch.zeros(B, R),
            "returns": torch.zeros(B, R),
            "advantages": adv,
            "loss_mask": resp_mask.clone(),
            "response_mask": resp_mask.clone(),
        }
    )
    batch.metadata = {"response_length": R}
    return batch


def chunks(batch, rows: int):
    B = batch["sequences"].shape[0]
    out = []
    for s in range(0, B, rows):
        out.append(batch.slice(s, min(B, s + rows)))
    return out


def run_forward(group, batch, rows: int):
    lps = []
    for c in chunks(batch, rows):
        out = ray.get(group.async_run_ray_method("mesh", "forward", data=c))
        lps.append(torch.tensor([o["logprobs"] for o in out[0].loss_fn_outputs], dtype=torch.float32))
    return torch.cat(lps, dim=0)


def run_forward_backward(group, batch, rows: int) -> Dict[str, float]:
    metrics: Dict[str, List[float]] = {}
    for c in chunks(batch, rows):
        outs = ray.get(group.async_run_ray_method("mesh", "forward_backward", data=c))
        for k, v in outs[0].metrics.items():
            if isinstance(v, (int, float)):
                metrics.setdefault(k, []).append(float(v))
    out = {k: sum(v) / len(v) for k, v in metrics.items()}
    for k in ("policy_loss", "final_loss"):  # pre-scaled sums over the mini-batch -> add over calls
        if k in metrics:
            out[k] = sum(metrics[k])
    return out


def optim_step(group):
    gns = ray.get(group.async_run_ray_method("pass_through", "optim_step"))
    return gns[0]


def set_mode(group, shared: bool, min_shared: int):
    ray.get(group.async_run_ray_method("pass_through", "set_prefix_sharing", shared, min_shared))


def sync_time(fn):
    t0 = time.perf_counter()
    res = fn()
    return res, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--etp", type=int, default=None)
    ap.add_argument("--offload-optimizer", action="store_true")
    ap.add_argument("--shapes", nargs="+", default=["1024:2048:8", "4096:1024:8"])
    ap.add_argument("--groups", type=int, default=2, help="prompts per mini-batch")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--baseline-rows", default="auto", help="rows per forward_backward call, baseline")
    ap.add_argument("--shared-rows", default="G", help="rows per forward_backward call, shared (G = one group)")
    ap.add_argument("--min-shared", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--check", action="store_true", help="run the log-prob / grad-norm A/B (all shapes)")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument(
        "--no-optim-in-timing", action="store_true", help="skip optim_step between timed reps (grads just accumulate)"
    )
    ap.add_argument("--fused-lm-head", action="store_true")
    ap.add_argument("--out", default=None, help="JSONL results path")
    ap.add_argument("--save-logprobs", default=None, help="torch.save the A/B log-prob tensors per shape here (.pt)")
    ap.add_argument("--cudnn-attn", action="store_true", help="baseline attention via TE cuDNN fused attention")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, repo)
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.utils.utils import prepare_runtime_environment, validate_cfg
    from tests.backends.skyrl_train.gpu.utils import init_worker_with_type

    shapes = [parse_shape(s) for s in args.shapes]
    max_g = max(g for _, _, g in shapes)

    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.policy.model.path = args.model
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = args.gpus
    cfg.trainer.placement.ref_num_gpus_per_node = args.gpus
    mc = cfg.trainer.policy.megatron_config
    mc.tensor_model_parallel_size = args.tp
    mc.pipeline_model_parallel_size = 1
    mc.expert_model_parallel_size = args.ep
    if args.etp is not None:
        mc.expert_tensor_parallel_size = args.etp
    if args.offload_optimizer:
        mc.optimizer_config_kwargs.update(
            {
                "optimizer_cpu_offload": True,
                "optimizer_offload_fraction": 1.0,
                "use_precision_aware_optimizer": True,
                "decoupled_weight_decay": True,
            }
        )
    cfg.trainer.policy.optimizer_config.lr = 0.0  # weights never move -> both modes see identical params
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.advantage_estimator = "grpo"
    cfg.trainer.remove_microbatch_padding = True
    cfg.trainer.prefix_sharing = True
    cfg.trainer.prefix_sharing_min_shared_tokens = args.min_shared
    cfg.trainer.fused_lm_head_logprob = args.fused_lm_head
    cfg.trainer.flash_attn = not args.cudnn_attn
    cfg.trainer.micro_train_batch_size_per_gpu = max_g * args.groups
    cfg.trainer.micro_forward_batch_size_per_gpu = max_g * args.groups
    cfg.trainer.train_batch_size = max_g * args.groups
    cfg.trainer.policy_mini_batch_size = max_g * args.groups
    cfg.generator.n_samples_per_prompt = max_g
    validate_cfg(cfg)

    env = prepare_runtime_environment(cfg)
    env["PYTHONPATH"] = repo + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    for k in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "LD_LIBRARY_PATH", "HF_HOME", "HF_HUB_OFFLINE"):
        if k in os.environ:
            env[k] = os.environ[k]
    if args.cudnn_attn:
        env["NVTE_FLASH_ATTN"] = "0"
        env["NVTE_FUSED_ATTN"] = "1"
    saved = {}
    ray.init(runtime_env={"env_vars": env, "py_executable": sys.executable}, log_to_driver=True)

    t0 = time.perf_counter()
    group = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=args.gpus, cfg=cfg)
    print(f"[bench] model ready in {time.perf_counter() - t0:.1f}s", flush=True)

    results = []
    out_f = open(args.out, "a") if args.out else None

    def emit(rec):
        results.append(rec)
        print("[bench] " + json.dumps(rec), flush=True)
        if out_f:
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()

    for P, R, G in shapes:
        batch = make_grpo_batch(args.groups, G, P, R, args.vocab, seed=P * 31 + R * 7 + G)
        row_tokens = int(batch["attention_mask"].sum())
        shared_tokens_est = args.groups * (P - 1) + int(batch["response_mask"].sum()) + args.groups * G
        ratio = shared_tokens_est / row_tokens
        shared_rows = G if args.shared_rows == "G" else int(args.shared_rows)
        if args.baseline_rows == "auto":
            baseline_rows = max(1, int(round(shared_rows * ratio)))
        else:
            baseline_rows = int(args.baseline_rows)
        info = dict(
            P=P,
            R=R,
            G=G,
            groups=args.groups,
            rows=int(batch["sequences"].shape[0]),
            row_tokens=row_tokens,
            shared_tokens_est=shared_tokens_est,
            token_ratio_est=round(ratio, 4),
            baseline_rows=baseline_rows,
            shared_rows=shared_rows,
        )
        print(f"[bench] shape {info}", flush=True)

        if args.check:

            def lp_stats(a, b):
                m = batch["response_mask"].bool()
                d = (a - b).abs()[m]
                return dict(
                    max_abs_diff=d.max().item(),
                    mean_abs_diff=d.mean().item(),
                    p99_abs_diff=d.float().quantile(0.99).item(),
                    mean_abs_logprob=a[m].abs().mean().item(),
                )

            set_mode(group, False, args.min_shared)
            lp0 = run_forward(group, batch, baseline_rows)
            lp0b = (
                run_forward(group, batch, shared_rows) if shared_rows != baseline_rows else lp0
            )  # same math, other packing
            set_mode(group, True, args.min_shared)
            lp1 = run_forward(group, batch, shared_rows)
            emit(
                dict(
                    kind="check_logprobs",
                    **info,
                    shared_vs_baseline=lp_stats(lp0, lp1),
                    noise_floor_baseline_repack=lp_stats(lp0, lp0b),
                )
            )
            saved[(P, R, G)] = dict(
                lp_baseline=lp0, lp_baseline_repack=lp0b, lp_shared=lp1, response_mask=batch["response_mask"].clone()
            )
            # PPO update with old logprobs = baseline forward logprobs (ratio ~ 1 -> gradients flow)
            batch["action_log_probs"] = lp0.clone()
            batch["rollout_logprobs"] = lp0.clone()
            set_mode(group, False, args.min_shared)
            met0 = run_forward_backward(group, batch, baseline_rows)
            gn0 = optim_step(group)
            if shared_rows != baseline_rows:
                met0b = run_forward_backward(group, batch, shared_rows)
                gn0b = optim_step(group)
            else:
                met0b, gn0b = met0, gn0
            # each mode uses its own recomputed old log-probs, as recompute_old_logprobs_per_minibatch does
            batch["action_log_probs"] = lp1.clone()
            batch["rollout_logprobs"] = lp1.clone()
            set_mode(group, True, args.min_shared)
            met1 = run_forward_backward(group, batch, shared_rows)
            gn1 = optim_step(group)
            batch["action_log_probs"] = lp0.clone()
            batch["rollout_logprobs"] = lp0.clone()
            emit(
                dict(
                    kind="check_grads",
                    **info,
                    grad_norm=dict(baseline=gn0, baseline_repack=gn0b, shared=gn1),
                    grad_norm_rel_diff=dict(
                        shared_vs_baseline=abs(gn0 - gn1) / max(abs(gn0), 1e-12),
                        noise_floor_baseline_repack=abs(gn0 - gn0b) / max(abs(gn0), 1e-12),
                    ),
                    policy_loss=dict(
                        baseline=met0.get("policy_loss"),
                        baseline_repack=met0b.get("policy_loss"),
                        shared=met1.get("policy_loss"),
                    ),
                    entropy=dict(
                        baseline=met0.get("policy_entropy"),
                        baseline_repack=met0b.get("policy_entropy"),
                        shared=met1.get("policy_entropy"),
                    ),
                    token_ratio_measured=met1.get("prefix_sharing/token_ratio"),
                )
            )
            saved[(P, R, G)].update(grad_norm=dict(baseline=gn0, baseline_repack=gn0b, shared=gn1))
            if args.save_logprobs:
                torch.save(saved, args.save_logprobs)

        if args.no_timing:
            continue
        for shared in (False, True):
            rows = shared_rows if shared else baseline_rows
            set_mode(group, shared, args.min_shared)
            # warmup
            run_forward_backward(group, batch, rows)
            optim_step(group)
            run_forward(group, batch, rows)
            fb_times, fw_times, tok_ratio = [], [], None
            for _ in range(args.reps):
                met, dt = sync_time(lambda: run_forward_backward(group, batch, rows))
                fb_times.append(dt)
                tok_ratio = met.get("prefix_sharing/token_ratio", 1.0)
                if not args.no_optim_in_timing:
                    optim_step(group)
                _, dt = sync_time(lambda: run_forward(group, batch, rows))
                fw_times.append(dt)
            if args.no_optim_in_timing:
                optim_step(group)  # clear the accumulated grads once per mode
            emit(
                dict(
                    kind="timing",
                    mode="shared" if shared else "baseline",
                    **info,
                    rows_per_call=rows,
                    fwd_bwd_s=min(fb_times),
                    fwd_bwd_mean_s=sum(fb_times) / len(fb_times),
                    fwd_s=min(fw_times),
                    fwd_mean_s=sum(fw_times) / len(fw_times),
                    token_ratio_measured=tok_ratio,
                    fwd_bwd_row_tokens_per_s=row_tokens / min(fb_times),
                )
            )

    # summary table
    print(
        "\n| P | R | G | tokens (unshared -> shared) | fwd+bwd baseline | fwd+bwd shared | speedup | fwd baseline | fwd shared | speedup |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    by = {}
    for r in results:
        if r["kind"] == "timing":
            by.setdefault((r["P"], r["R"], r["G"]), {})[r["mode"]] = r
    for (P, R, G), d in by.items():
        b, s = d.get("baseline"), d.get("shared")
        if not (b and s):
            continue
        print(
            f"| {P} | {R} | {G} | {b['row_tokens']} -> {int(b['row_tokens'] * (s['token_ratio_measured'] or 1))} "
            f"| {b['fwd_bwd_s']:.2f}s | {s['fwd_bwd_s']:.2f}s | {b['fwd_bwd_s'] / s['fwd_bwd_s']:.2f}x "
            f"| {b['fwd_s']:.2f}s | {s['fwd_s']:.2f}s | {b['fwd_s'] / s['fwd_s']:.2f}x |"
        )
    if out_f:
        out_f.close()
    ray.shutdown()


if __name__ == "__main__":
    main()
