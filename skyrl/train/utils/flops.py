"""Analytic training FLOPs per token, for model FLOPs utilization (MFU).

Counts model FLOPs only: one forward and one backward per token, no activation recompute.
A matmul with ``n`` weight parameters costs ``2n`` FLOPs per token forward and ``2n`` for the
input gradient; full fine-tuning adds ``2n`` for the weight gradient, LoRA (frozen base) does
not. Attention cores have no weights, so their backward is always ~2x their forward.

Handles dense / MoE MLPs, standard (GQA) attention, MLA, DeepSeek sparse attention (top-k keys
per query, plus the k-pool indexer, which only runs forward), and KDA linear attention, as laid
out by HF configs such as GLM-5.3-Flash (``glm5_next``). Small terms (norms, routers, mHC mixing,
LoRA adapters) are left out.
"""

import json
import os
from typing import Any

# Dense (non-sparse) BF16 tensor-core peak per GPU, FLOP/s.
PEAK_FLOPS_BF16 = {
    "B200": 2.25e15,
    "B300": 2.25e15,
    "H100": 0.989e15,
    "H200": 0.989e15,
}

# Chunk length of the chunked KDA / gated delta rule kernels.
_KDA_CHUNK = 64


def load_text_config(model_path: str) -> dict[str, Any]:
    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)
    return config.get("text_config", config)


def _mean_causal_keys(seq_len: int, cap: int | None = None) -> float:
    """Mean number of keys a causal query attends to over positions 1..seq_len, capped at ``cap``."""
    if cap is None or cap >= seq_len:
        return (seq_len + 1) / 2
    # Positions 1..cap attend to p keys, the rest to cap.
    return (cap * (cap + 1) / 2 + (seq_len - cap) * cap) / seq_len


def _mlp_params(cfg: dict[str, Any], layer: int) -> int:
    hidden = cfg["hidden_size"]
    mlp_types = cfg.get("mlp_layer_types")
    if mlp_types is not None:
        is_moe = mlp_types[layer] == "sparse"
    else:
        is_moe = cfg.get("n_routed_experts") is not None and layer >= cfg.get("first_k_dense_replace", 0)
    if not is_moe:
        return 3 * hidden * cfg["intermediate_size"]
    experts = cfg["num_experts_per_tok"] + cfg.get("n_shared_experts", 0)
    return experts * 3 * hidden * cfg["moe_intermediate_size"]


def _kda_flops(cfg: dict[str, Any]) -> tuple[int, float]:
    """(projection params, core forward FLOPs per token) of one KDA layer."""
    hidden = cfg["hidden_size"]
    lin = cfg["linear_attn_config"]
    heads, head_dim = lin["num_heads"], lin["head_dim"]
    inner = heads * head_dim
    # q/k/v and o projections, low-rank forget/output gates (hidden -> head_dim -> inner), beta.
    params = 4 * hidden * inner + 2 * (hidden * head_dim + head_dim * inner) + hidden * heads
    # Per head: state read and update (2 * dk * dv each) and the intra-chunk terms.
    core = heads * (4 * head_dim * head_dim + 6 * _KDA_CHUNK * head_dim)
    return params, core


def _mla_dsa_flops(cfg: dict[str, Any], seq_len: int) -> tuple[int, float, float]:
    """(projection params, attention core fwd FLOPs, indexer fwd FLOPs) per token of one MLA layer."""
    hidden, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    qk_dim = cfg.get("qk_head_dim") or cfg["qk_nope_head_dim"] + cfg.get("qk_rope_head_dim", 0)
    v_dim = cfg["v_head_dim"]
    q_lora, kv_lora = cfg.get("q_lora_rank"), cfg["kv_lora_rank"]
    rope = cfg.get("qk_rope_head_dim") or 0
    q_params = hidden * q_lora + q_lora * heads * qk_dim if q_lora else hidden * heads * qk_dim
    kv_params = hidden * (kv_lora + rope) + kv_lora * heads * (cfg["qk_nope_head_dim"] + v_dim)
    params = q_params + kv_params + heads * v_dim * hidden

    topk = cfg.get("index_topk")
    keys = _mean_causal_keys(seq_len, topk)
    core = 2 * heads * (qk_dim + v_dim) * keys

    indexer = 0.0
    if topk is not None:
        idx_heads, idx_dim = cfg["index_n_heads"], cfg["index_head_dim"]
        pool = cfg.get("index_kpool", 1) or 1
        params += (q_lora or hidden) * idx_heads * idx_dim + hidden * idx_dim + hidden * idx_heads
        indexer = 2 * idx_heads * idx_dim * _mean_causal_keys(seq_len) / pool
    return params, core, indexer


def _gqa_flops(cfg: dict[str, Any], seq_len: int) -> tuple[int, float]:
    hidden, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads", heads)
    head_dim = cfg.get("head_dim") or hidden // heads
    params = hidden * heads * head_dim * 2 + hidden * kv_heads * head_dim * 2
    core = 4 * heads * head_dim * _mean_causal_keys(seq_len, cfg.get("sliding_window"))
    return params, core


def train_flops_per_token(cfg: dict[str, Any], seq_len: int, lora: bool) -> float:
    """Model FLOPs to train on one token of a ``seq_len``-token sequence (forward + backward)."""
    num_layers = cfg["num_hidden_layers"]
    layer_types = cfg.get("layer_types") or ["full_attention"] * num_layers

    matmul_params = cfg["hidden_size"] * cfg["vocab_size"]  # LM head
    attn_core = 0.0
    fwd_only = 0.0
    for layer in range(num_layers):
        matmul_params += _mlp_params(cfg, layer)
        if layer_types[layer] == "linear_attention":
            params, core = _kda_flops(cfg)
        elif cfg.get("kv_lora_rank"):
            params, core, indexer = _mla_dsa_flops(cfg, seq_len)
            fwd_only += indexer
        else:
            params, core = _gqa_flops(cfg, seq_len)
        matmul_params += params
        attn_core += core

    # fwd 2n, input grad 2n, weight grad 2n (full FT only).
    matmul = (4 if lora else 6) * matmul_params
    return matmul + 3 * attn_core + fwd_only


def peak_flops_per_gpu(gpu_name: str) -> float:
    """BF16 dense peak for a GPU name such as ``NVIDIA B200``; overridable via ``SKYRL_MFU_PEAK_TFLOPS``."""
    override = os.environ.get("SKYRL_MFU_PEAK_TFLOPS")
    if override:
        return float(override) * 1e12
    for key, peak in PEAK_FLOPS_BF16.items():
        if key in gpu_name:
            return peak
    raise ValueError(f"Unknown peak FLOP/s for GPU {gpu_name!r}; set SKYRL_MFU_PEAK_TFLOPS")
