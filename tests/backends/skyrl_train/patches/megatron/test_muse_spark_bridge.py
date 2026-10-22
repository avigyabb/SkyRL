"""CPU test for ``patches/megatron/muse_spark``: HF config -> provider translation and the mapping set.

Uses a 4-layer config shaped like Muse-Spark-Open-816B-A42B's (same schedules, smaller widths). No
weights are read: ``to_megatron_provider(load_weights=False)`` only parses the config.
"""

import json
import math

import pytest

pytest.importorskip("megatron.core", reason="requires the megatron extra")

pytestmark = pytest.mark.megatron

LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention", "sliding_attention"]
ROPE_THETA = [500000.0, 0, 500000.0, 500000.0]


def _config():
    text = {
        "model_type": "muse_spark_text",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 200000,
        "eos_token_id": 200001,
        "pad_token_id": 200018,
        "final_logit_softcapping": 20.0,
        "head_dim": 64,
        "hidden_act": "silu",
        "hidden_size": 512,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "layer_rope_theta": ROPE_THETA,
        "layer_types": LAYER_TYPES,
        "max_position_embeddings": 4096,
        "mlp_layer_types": ["sparse"] * 4,
        "moe_hidden_size": 256,
        "moe_intermediate_size": 128,
        "num_attention_heads": 8,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "num_local_experts": 8,
        "num_shared_experts": 0,
        "output_multiplier": 0.17677669529663687,
        "post_norm_eps": 1e-08,
        "qk_scale_factor": 5.4730064863875,
        "residual_gate_temperature": 0.3,
        "rms_norm_eps": 1e-05,
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
        "routed_scaling_factor": 1.0,
        "sliding_window": 2048,
        "tie_word_embeddings": False,
        "vocab_size": 202048,
    }
    return {
        "architectures": ["MuseSparkForConditionalGeneration"],
        "model_type": "muse_spark",
        "dtype": "bfloat16",
        "image_token_id": 200092,
        "video_token_id": 200091,
        "audio_token_id": 200038,
        "out_hidden_size": 6144,
        "projector_hidden_size": 4096,
        "projector_hidden_act": "gelu",
        "text_config": text,
        "vision_config": {"model_type": "muse_spark_vision", "hidden_size": 1536, "merge_size": 2},
        "audio_config": {"model_type": "muse_spark_audio", "out_hidden_size": 512},
    }


@pytest.fixture
def bridge(tmp_path):
    from megatron.bridge import AutoBridge

    import skyrl.backends.skyrl_train  # noqa: F401  registers the HF config
    import skyrl.backends.skyrl_train.workers.megatron.model_bridges  # noqa: F401

    (tmp_path / "config.json").write_text(json.dumps(_config()))
    return AutoBridge.from_hf_pretrained(str(tmp_path))


def test_provider_translation(bridge):
    from skyrl.backends.skyrl_train.patches.megatron.muse_spark.bridge import (
        MuseSparkBridge,
    )
    from skyrl.backends.skyrl_train.patches.megatron.muse_spark.provider import (
        MuseSparkModelProvider,
    )

    assert isinstance(bridge._model_bridge, MuseSparkBridge)
    provider = bridge.to_megatron_provider(load_weights=False)
    assert isinstance(provider, MuseSparkModelProvider)
    provider.finalize()

    # One attention + one MoE Hybrid layer per HF block.
    assert provider.hybrid_layer_pattern == "*E*E*E*E"
    assert provider.num_layers == 8
    # Schedules are indexed by Hybrid layer: MoE positions are 0.
    assert provider.window_attn_skip_freq == [1, 0, 0, 0, 1, 0, 1, 0]
    assert provider.no_rope_freq == [0, 0, 1, 0, 0, 0, 0, 0]
    assert provider.window_size == (2047, 0)
    assert provider.softmax_scale == pytest.approx(5.4730064863875 / math.sqrt(64))
    assert provider.moe_latent_size == 256
    assert provider.moe_ffn_hidden_size == 128
    assert provider.moe_router_bias_update_rate == 0.0
    assert provider.layernorm_zero_centered_gamma
    assert provider.final_logit_softcapping == 20.0


def test_mapping_covers_every_language_tensor(bridge):
    registry = bridge._model_bridge.mapping_registry()
    hf_names = set()
    for mapping in registry.mappings:
        hf = mapping.hf_param
        hf_names.update(hf.values() if isinstance(hf, dict) else [hf])
    per_layer = {
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.gate_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_residual_gate.gate",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "post_feedforward_residual_gate.gate",
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
        "mlp.pre_expert_proj.weight",
        "mlp.pre_expert_norm.weight",
        "mlp.post_expert_proj.weight",
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "mlp.experts.post_expert_norm.weight",
    }
    expected = {"lm_head.weight", "model.language_model.embed_tokens.weight", "model.language_model.norm.weight"}
    expected |= {f"model.language_model.layers.{i}.{name}" for i in range(4) for name in per_layer}
    assert expected <= hf_names, sorted(expected - hf_names)
