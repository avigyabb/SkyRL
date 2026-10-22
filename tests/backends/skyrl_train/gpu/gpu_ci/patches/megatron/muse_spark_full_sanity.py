"""Full Muse-Spark-Open-816B-A42B Megatron sanity check: natural-text logprobs and a greedy continuation.

HF transformers can't hold the full model on one node, so this checks coherence instead of a
logprob diff: a correct port scores fluent text far above uniform (ln 202048 = 12.2 nats) and
continues a prompt sensibly. Development harness, not a pytest.

    torchrun --nnodes N --nproc-per-node 8 ... muse_spark_full_sanity.py <model_dir> <tp> <ep>
"""

import sys

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from transformers import AutoTokenizer

import skyrl.backends.skyrl_train  # noqa: F401  registers the muse_spark HF config
import skyrl.backends.skyrl_train.workers.megatron.model_bridges  # noqa: F401

TEXTS = [
    "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome.",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May? Natalia sold 48/2 = 24 clips in May. "
    "Natalia sold 48+24 = 72 clips altogether in April and May.",
]
PROMPT = "Question: What is 17 * 3? Answer:"


def logits_for(model, seq, tp):
    pad = (-seq.shape[1]) % tp
    seq_p = torch.nn.functional.pad(seq, (0, pad), value=0)
    pos = torch.arange(seq_p.shape[1], device="cuda")[None]
    with torch.no_grad():
        logits = model(seq_p, pos, None)
    return gather_from_tensor_model_parallel_region(logits)[:, : seq.shape[1]].float()


def main():
    model_dir, tp, ep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tok = AutoTokenizer.from_pretrained(model_dir)
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
    rank0 = dist.get_rank() == 0

    for text in TEXTS:
        seq = tok(text, return_tensors="pt").input_ids.cuda()
        logits = logits_for(model, seq, tp)
        lp = torch.log_softmax(logits[0, :-1], dim=-1).gather(-1, seq[0, 1:, None])[:, 0]
        if rank0:
            print(f"len={seq.shape[1]} mean logprob={lp.mean():.4f} (uniform = -12.2)", flush=True)

    seq = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    for _ in range(16):
        next_id = logits_for(model, seq, tp)[0, -1].argmax()
        seq = torch.cat([seq, next_id.view(1, 1)], dim=1)
    if rank0:
        print(f"GREEDY: {tok.decode(seq[0])!r}", flush=True)
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
