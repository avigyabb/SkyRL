"""Repro for R3 + chunked MoE (mlp_chunks_for_training > 1) + `moe` activation recompute on a GLM-5.3-Flash slice.

Random-init 4-layer / 64-expert slice, TP4 / EP8 on 8 GPUs, token-based micro-batches (DP ranks get
uneven counts, so padding micro-batches appear). Prints the policy loss and the first optimizer step's
grad norm; compare `--recompute none` against `--recompute [core_attn,moe]` (same seed).

Run from the repo root with the Ray cluster up:
  python throughput_push_v1/sim/repro_r3_chunked_moe_recompute.py --mlp-chunks 64 --zero-mask-row 0
"""

import argparse
import os

import ray
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type

MODEL = "/shared/hackskyrl/models/glm53_flash_cfg_l4_e64"
LAYERS, TOPK, EXPERTS = 4, 8, 64

p = argparse.ArgumentParser()
p.add_argument("--zero-mask-row", type=int, default=-1, help="sample index whose loss_mask is all zero")
p.add_argument("--lora", action="store_true")
p.add_argument("--recompute", default="[core_attn,moe]")
p.add_argument("--max-tokens", type=int, default=2048)
p.add_argument("--mlp-chunks", type=int, default=1)
args = p.parse_args()
torch.manual_seed(1234)

cfg = SkyRLTrainConfig()
cfg.trainer.strategy = "megatron"
cfg.trainer.policy.model.path = MODEL
cfg.trainer.policy.language_model_only = True
cfg.trainer.remove_microbatch_padding = True
cfg.trainer.max_tokens_per_microbatch = args.max_tokens
cfg.trainer.micro_train_batch_size_per_gpu = 1
cfg.trainer.micro_forward_batch_size_per_gpu = 1
cfg.trainer.fused_lm_head_logprob = True
cfg.trainer.placement.policy_num_gpus_per_node = 8
mc = cfg.trainer.policy.megatron_config
mc.tensor_model_parallel_size = 4
mc.expert_model_parallel_size = 8
mc.expert_tensor_parallel_size = 1
mc.mtp_num_layers = 0
mc.moe_grouped_gemm = True
mc.moe_token_dispatcher_type = "alltoall"
mc.moe_router_score_function = "sigmoid"
mc.moe_router_load_balancing_type = "none"
mc.moe_enable_routing_replay = True
tk = mc.transformer_config_kwargs
tk["sequence_parallel"] = True
if args.recompute == "full":
    tk["recompute_granularity"] = "full"
    tk["recompute_method"] = "uniform"
    tk["recompute_num_layers"] = 1
elif args.recompute != "none":
    tk["recompute_granularity"] = "selective"
    tk["recompute_method"] = None
    tk["recompute_num_layers"] = None
    tk["recompute_modules"] = [m for m in args.recompute.strip("[]").split(",") if m]
tk["gradient_accumulation_fusion"] = False
tk["mlp_chunks_for_training"] = args.mlp_chunks
if args.lora:
    cfg.trainer.policy.model.lora.rank = 16
    cfg.trainer.policy.model.lora.alpha = 16
    cfg.trainer.policy.model.lora.target_modules = [
        "linear_q_down_proj",
        "linear_q_up_proj",
        "linear_kv_down_proj",
        "linear_kv_up_proj",
        "linear_proj",
        "linear_fc1",
        "linear_fc2",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]

# 2 DP ranks; lengths chosen so the ranks get different microbatch counts (-> padding microbatches)
# and one long sample sits alone in a microbatch.
lengths = [int(x) for x in os.environ.get("LENGTHS", "1800,1700,300,400,1900,200,350,450").split(",")]
prompt_len = 64
num = len(lengths)
max_len = max(lengths)
seq = torch.zeros((num, max_len), dtype=torch.long)
attn = torch.zeros((num, max_len), dtype=torch.long)
for i, n in enumerate(lengths):  # left-padded prompt + right-padded response (SkyRL layout)
    seq[i, :n] = torch.randint(10, 1000, (n,))
    attn[i, :n] = 1
num_actions = max_len - prompt_len
loss_mask = attn[:, prompt_len:].float().clone()
if args.zero_mask_row >= 0:
    loss_mask[args.zero_mask_row] = 0
segs = [torch.randint(0, EXPERTS, (n, LAYERS, TOPK), dtype=torch.int16) for n in lengths]
segs = [torch.sort(s, dim=-1).values for s in segs]
for s in segs:  # distinct experts per token
    s.copy_((s[..., :1] + torch.arange(TOPK, dtype=torch.int16)) % EXPERTS)
g = torch.Generator().manual_seed(0)
data = TrainingInputBatch(
    {
        "sequences": seq,
        "attention_mask": attn,
        "response_mask": attn[:, prompt_len:],
        "loss_mask": loss_mask,
        "rollout_logprobs": -torch.rand((num, num_actions), generator=g),
        "action_log_probs": -torch.rand((num, num_actions), generator=g),
        "base_action_log_probs": -torch.rand((num, num_actions), generator=g),
        "advantages": torch.randn((num, num_actions), generator=g),
        "rollout_expert_indices": PackedTensor.from_segments(segs),
        "router_padding_mask": ~attn.bool(),
    }
)
data.metadata = {"response_length": num_actions}

env = {
    "SKYRL_MEGATRON_RANDOM_INIT": "1",
    "FLA_TILELANG": "0",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_CUMEM_ENABLE": "0",
    "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
    "PYTHONPATH": os.environ["PYTHONPATH"],
    "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda"),
}
ray.init(
    address="auto", runtime_env={"env_vars": env, "py_executable": "/shared/hackskyrl/uv_venvs/tput_v1/bin/python"}
)
group = init_worker_with_type("policy", num_gpus_per_node=8, cfg=cfg)
out = ray.get(group.async_run_ray_method("mesh", "forward_backward", data=data))
gn = ray.get(group.async_run_ray_method("pass_through", "optim_step"))
print("RESULT grad_norm", gn[0])
print(
    "RESULT ok policy_loss",
    out[0].metrics.get("policy_loss"),
    "microbatches",
    out[0].metrics.get("num_microbatches"),
    "padding",
    out[0].metrics.get("num_padding_microbatches"),
)
