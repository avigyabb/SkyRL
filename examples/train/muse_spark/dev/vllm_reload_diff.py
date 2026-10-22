"""Which tensors does vLLM's in-place reload of the same checkpoint change? (Muse Spark debug)

Usage: VLLM_ALLOW_INSECURE_SERIALIZATION=1 python vllm_reload_diff.py <model> <tp>
"""

import sys

import torch
from vllm import LLM


def snapshot(worker):
    model = worker.model_runner.model
    out = {}
    for mod_name, mod in model.named_modules():
        for attr, val in list(vars(mod).items()):
            if isinstance(val, torch.Tensor) and attr not in ("_parameters", "_buffers"):
                out[f"{mod_name}.{attr} (attr)"] = val
        for name, p in mod.named_parameters(recurse=False):
            out[f"{mod_name}.{name}"] = p
        for name, b in mod.named_buffers(recurse=False):
            out[f"{mod_name}.{name} (buf)"] = b
    snap = {}
    snap["__ptrs__"] = {k: v.data_ptr() for k, v in out.items() if isinstance(v, torch.Tensor)}
    for k, v in out.items():
        if not v.is_floating_point() or "kv_cache" in k:
            continue
        v = v.detach()
        if v.numel() < 2**26:
            snap[k] = v.float().cpu().clone()
        else:  # fingerprint big tensors without an fp32 copy
            snap[k] = torch.stack(
                [
                    torch.linalg.vector_norm(v, dtype=torch.float32),
                    v.view(-1)[:: max(1, v.numel() // 4096)].float().sum(),
                ]
            ).cpu()
    return snap


def take(worker):
    worker._muse_snap = snapshot(worker)
    return len(worker._muse_snap)


def compare(worker):
    after = snapshot(worker)
    rows = []
    before_ptrs, after_ptrs = worker._muse_snap["__ptrs__"], after.pop("__ptrs__")
    for k, p in before_ptrs.items():
        if k in after_ptrs and after_ptrs[k] != p:
            rows.append((k, "PTR moved"))
    for k, v in worker._muse_snap.items():
        if k == "__ptrs__":
            continue
        w = after.get(k)
        if w is None:
            rows.append((k, "MISSING after"))
        elif v.shape != w.shape:
            rows.append((k, f"shape {tuple(v.shape)} -> {tuple(w.shape)}"))
        else:
            d = (v - w).abs().max().item()
            if d > 0:
                rows.append((k, f"max|d|={d:.4g} ref|max|={v.abs().max().item():.4g}"))
    return rows


def main():
    model, tp = sys.argv[1], int(sys.argv[2])
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        language_model_only=True,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    print("SNAP tensors:", llm.collective_rpc(take)[0], flush=True)
    llm.collective_rpc("reload_weights", kwargs={"weights_path": model})
    rows = llm.collective_rpc(compare)[0]
    print(f"CHANGED {len(rows)} tensors on rank 0", flush=True)
    for k, msg in rows[:200]:
        print(f"CHANGED {k}: {msg}", flush=True)


if __name__ == "__main__":
    main()
