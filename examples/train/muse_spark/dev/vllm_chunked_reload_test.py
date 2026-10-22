"""Emulate SkyRL's chunked weight sync inside vLLM: layerwise reload with one load_weights call per
tensor, language-model tensors only (what Megatron exports). Reports the logprob shift and which
tensors changed. Usage:
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 python vllm_chunked_reload_test.py <model> <tp> [worker_ext]
"""

import json
import os
import sys

import torch
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(__file__))
from vllm_reload_diff import compare, take  # noqa: E402

TEXT = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."


def score(llm):
    out = llm.generate([TEXT], SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))[0]
    return [next(iter(d.values())).logprob for d in out.prompt_logprobs[1:]]


def chunked_reload(worker, model_dir):
    from safetensors import safe_open
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
    )

    model = worker.model_runner.model
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    names = [n for n in index if n.startswith(("model.language_model.", "lm_head."))]
    max_layer = int(os.environ.get("MAX_LAYER", "-1"))
    if max_layer >= 0:
        import re as _re

        names = [n for n in names if (m := _re.search(r"\.layers\.(\d+)\.", n)) and int(m.group(1)) <= max_layer]
    mem_log = []
    last_layer = None
    initialize_layerwise_reload(model)
    for name in names:
        with safe_open(os.path.join(model_dir, index[name]), "pt", device="cpu") as f:
            tensor = f.get_tensor(name).cuda()
        scale_pat = os.environ.get("SCALE_PATTERN")  # e.g. "self_attn." to perturb those tensors
        if scale_pat and scale_pat in name:
            tensor = tensor * 0.5
        if os.environ.get("CAST_BF16") == "1":  # what MegatronWeightSource does to every tensor
            tensor = tensor.to(torch.bfloat16)
        if os.environ.get("PACKED_SIM") == "1":
            # Emulate packed_nccl_broadcast_consumer: copy into one of two rotating uint8 buffers and
            # hand load_weights a view into it (the buffer is reused two chunks later).
            buf_idx = len(mem_log) % 2
            nbytes = tensor.numel() * tensor.element_size()
            bufs = worker.__dict__.setdefault("_sim_bufs", [None, None])
            if bufs[buf_idx] is None or bufs[buf_idx].numel() < nbytes:
                bufs[buf_idx] = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            bufs[buf_idx][:nbytes].copy_(tensor.contiguous().view(-1).view(torch.uint8))
            view = bufs[buf_idx][:nbytes].view(tensor.dtype).view(tensor.shape)
            del tensor
            model.load_weights([(name, view)])
            mem_log.append(("chunk", 0))  # advance the rotation per tensor
            continue
        model.load_weights([(name, tensor)])
        del tensor
        import re as _re

        lm = _re.search(r"\.layers\.(\d+)\.", name)
        layer_id = int(lm.group(1)) if lm else None
        if layer_id != last_layer:
            mem_log.append((last_layer, round(torch.cuda.memory_allocated() / 2**30, 1)))
            last_layer = layer_id
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    pending = []
    for lname, layer in model.named_modules():
        info = get_layerwise_info(layer)
        if info.can_load() and info.load_numel_total and info.load_numel < info.load_numel_total:
            pending.append(f"{lname} [{type(layer).__name__}] {info.load_numel}/{info.load_numel_total}")
    peak = torch.cuda.max_memory_allocated() / 2**30
    finalize_layerwise_reload(model, worker.model_runner.vllm_config.model_config)
    mem_log.append((last_layer, round(torch.cuda.memory_allocated() / 2**30, 1)))
    return len(names), peak, pending, mem_log


def main():
    model, tp = sys.argv[1], int(sys.argv[2])
    ext = sys.argv[3] if len(sys.argv) > 3 else None
    pp = int(os.environ.get("PP", "1"))
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        language_model_only=True,
        max_model_len=4096,
        distributed_executor_backend="ray" if pp > 1 else None,
        gpu_memory_utilization=float(os.environ.get("GMU", "0.8")),
        enforce_eager=os.environ.get("EAGER", "1") == "1",
        **({"worker_extension_cls": ext} if ext else {}),
    )
    before = score(llm)
    llm.collective_rpc(take)
    torch.cuda.reset_peak_memory_stats()
    results = llm.collective_rpc(chunked_reload, args=(model,))
    n, peak, pending, mem_log = results[0]
    print("MEMLOG rank0 (layer done, GiB allocated):", mem_log, flush=True)
    print(
        f"CHUNKED tensors sent: {n}, rank0 peak allocated {peak:.1f} GiB, {len(pending)} layers incomplete before finalize",
        flush=True,
    )
    for line in pending[:40]:
        print("PENDING", line, flush=True)
    for rank, rows in enumerate(llm.collective_rpc(compare)):
        print(f"CHANGED rank{rank}: {len(rows)} tensors", flush=True)
        for k, msg in rows[:40]:
            print(f"CHANGED rank{rank} {k}: {msg}", flush=True)
    after = score(llm)
    diff = [abs(a - b) for a, b in zip(before, after)]
    print(
        f"CHUNKED before={sum(before)/len(before):.4f} after={sum(after)/len(after):.4f} "
        f"|diff| mean={sum(diff)/len(diff):.4f} max={max(diff):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
