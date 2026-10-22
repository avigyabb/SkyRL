"""Compare the Muse Spark Megatron forward (patches/megatron/muse_spark) to HF reference logprobs.

Development harness, not a pytest: the HF reference needs transformers >= 5.21 (``muse_spark``
modeling), which the SkyRL env doesn't have, so it is produced separately and passed in.

    torchrun --nproc-per-node 4 muse_spark_vs_hf.py <model_dir> <hf_reference.pt> <tp> <ep>
"""

import sys

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region

import skyrl.backends.skyrl_train  # noqa: F401  registers the muse_spark HF config
import skyrl.backends.skyrl_train.workers.megatron.model_bridges  # noqa: F401


def main():
    model_dir, ref_path, tp, ep = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
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
    model = provider.provide_distributed_model(wrap_with_ddp=False)[0].eval()

    ref = torch.load(ref_path)
    diffs = []
    for ids, ref_lp in zip(ref["input_ids"], ref["logprobs"]):
        seq = ids.cuda()[None]
        # Sequence parallel splits the sequence over TP ranks.
        pad = (-seq.shape[1]) % tp
        seq_p = torch.nn.functional.pad(seq, (0, pad), value=0)
        pos = torch.arange(seq_p.shape[1], device="cuda")[None]
        with torch.no_grad():
            logits = model(seq_p, pos, None)
            logits = gather_from_tensor_model_parallel_region(logits)[:, : seq.shape[1]].float()
        lp = torch.log_softmax(logits[0, :-1], dim=-1).gather(-1, seq[0, 1:, None])[:, 0].cpu()
        diff = (lp - ref_lp).abs()
        diffs.append(diff)
        if dist.get_rank() == 0:
            print(
                f"len={seq.shape[1]} megatron={lp.mean():.4f} hf={ref_lp.mean():.4f} "
                f"|diff| mean={diff.mean():.4f} max={diff.max():.4f}",
                flush=True,
            )
    if dist.get_rank() == 0:
        print(f"OVERALL |dlogprob| mean = {torch.cat(diffs).mean():.4f}", flush=True)
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
