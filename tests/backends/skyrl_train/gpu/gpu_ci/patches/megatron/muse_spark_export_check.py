"""Megatron -> HF export of the Muse Spark bridge vs the source checkpoint, tensor by tensor.

Development harness, not a pytest. Loads the HF checkpoint into Megatron, streams it back out
through ``AutoBridge.export_hf_weights`` and compares every exported tensor with the safetensors
original (on rank 0).

    torchrun --nproc-per-node 4 muse_spark_export_check.py <model_dir> <tp> <ep>
"""

import json
import os
import sys

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from safetensors import safe_open

import skyrl.backends.skyrl_train  # noqa: F401  registers the muse_spark HF config
import skyrl.backends.skyrl_train.workers.megatron.model_bridges  # noqa: F401


def main():
    model_dir, tp, ep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    bridge = AutoBridge.from_hf_pretrained(model_dir, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = tp
    provider.expert_model_parallel_size = ep
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = tp > 1
    provider.pipeline_dtype = torch.bfloat16
    provider.attention_backend = "fused"
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    provider.initialize_model_parallel(seed=0)
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    rank0 = dist.get_rank() == 0

    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    bad, n = [], 0
    max_tensors = int(os.environ.get("MAX_TENSORS", "0"))
    order_out = os.environ.get("ORDER_OUT")
    order = []
    for name, tensor in bridge.export_hf_weights(model, cpu=True):
        if max_tensors and n >= max_tensors:
            break
        n += 1 if not rank0 else 0
        if not rank0:
            continue
        n += 1
        order.append(f"{name}\t{tensor.dtype}\t{tuple(tensor.shape)}")
        with safe_open(os.path.join(model_dir, index[name]), "pt") as f:
            ref = f.get_tensor(name)
        if ref.shape != tensor.shape:
            bad.append(f"{name}: shape {tuple(tensor.shape)} vs ref {tuple(ref.shape)}")
            continue
        d = (tensor.float() - ref.float()).abs().max().item()
        if d > 0:
            bad.append(
                f"{name}: max|d|={d:.4g} ref|max|={ref.float().abs().max().item():.4g} "
                f"dtype {tensor.dtype} vs {ref.dtype}"
            )
    if rank0 and order_out:
        with open(order_out, "w") as f:
            f.write("\n".join(order) + "\n")
    if rank0:
        print(f"EXPORTED {n} tensors, {len(bad)} differ", flush=True)
        for line in bad[:100]:
            print("DIFF", line, flush=True)


if __name__ == "__main__":
    main()
