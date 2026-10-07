"""NVFP4 wire: fusion-group buffering, tensor naming, engine config. TE's cast is stubbed (CPU)."""

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.quantization_utils import (
    resolve_auto_wire_format,
    wire_to_engine_quantization,
)
from skyrl.backends.skyrl_train.weight_sync.fp8 import vllm_format
from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import NVFP4, WIRE_FORMATS
from skyrl.backends.skyrl_train.weight_sync.fp8.vllm_format import (
    get_serialized_fp8_quantization_config,
    iter_serialized_nvfp4_tensors,
    resolve_serialized_fp8_config,
)


class _Cfg:
    model_type = "qwen3_5_text"
    layer_types = ["linear_attention", "full_attention"]


@pytest.fixture
def config(monkeypatch):
    calls = []

    def fake_cast(w, two_d=None):
        calls.append(tuple(w.shape))
        rows, cols = w.shape
        return (
            torch.zeros(rows, cols // 2, dtype=torch.uint8),
            torch.zeros(rows, cols // 16, dtype=torch.float8_e4m3fn),
            torch.tensor([float(rows)]),
        )

    monkeypatch.setattr(vllm_format, "nvfp4_cast", fake_cast)
    return resolve_serialized_fp8_config(NVFP4, _Cfg()), calls


def _w(rows, cols=256):
    return torch.zeros(rows, cols, dtype=torch.bfloat16)


def test_wire_registered_and_maps_to_compressed_tensors():
    assert NVFP4 in WIRE_FORMATS
    assert wire_to_engine_quantization(NVFP4) == "compressed-tensors"
    assert resolve_auto_wire_format("mxfp8", fp4_enabled=True) == NVFP4


def test_qkv_group_quantized_together_with_shared_scale(config):
    config, calls = config
    p = "model.layers.3.self_attn."
    stream = [
        (p + "q_proj.weight", _w(512)),
        (p + "k_proj.weight", _w(128)),
        (p + "v_proj.weight", _w(128)),
        ("model.layers.3.input_layernorm.weight", torch.zeros(256)),
    ]
    out = dict(iter_serialized_nvfp4_tensors(iter(stream), config))
    assert calls == [(768, 256)]  # one cast over the fused rows
    for n, rows in (("q_proj", 512), ("k_proj", 128), ("v_proj", 128)):
        assert out[p + n + ".weight_packed"].shape == (rows, 128)
        assert out[p + n + ".weight_scale"].shape == (rows, 16)
        assert out[p + n + ".weight_global_scale"].item() == 768.0
        assert p + n + ".weight" not in out
    assert "model.layers.3.input_layernorm.weight" in out


def test_ungrouped_weight_quantized_alone(config):
    config, _ = config
    name = "model.layers.0.self_attn.o_proj.weight"
    out = dict(iter_serialized_nvfp4_tensors(iter([(name, _w(256))]), config))
    assert set(out) == {
        "model.layers.0.self_attn.o_proj.weight_packed",
        "model.layers.0.self_attn.o_proj.weight_scale",
        "model.layers.0.self_attn.o_proj.weight_global_scale",
    }


def test_incomplete_group_is_an_error(config):
    config, _ = config
    stream = [("model.layers.0.mlp.gate_proj.weight", _w(512))]
    with pytest.raises(ValueError, match="incomplete fused groups"):
        list(iter_serialized_nvfp4_tensors(iter(stream), config))


def test_moe_experts_rejected(config):
    config, _ = config
    with pytest.raises(NotImplementedError, match="MoE"):
        list(
            iter_serialized_nvfp4_tensors(
                iter([("model.layers.0.mlp.experts.down_proj", torch.zeros(2, 4, 4))]), config
            )
        )


def test_unaligned_weight_stays_bf16(config):
    config, _ = config
    name = "model.layers.0.self_attn.o_proj.weight"  # cols % 128 != 0 -> Marlin cannot take it
    out = dict(iter_serialized_nvfp4_tensors(iter([(name, _w(256, cols=192))]), config))
    assert set(out) == {name}


def test_quantization_config_is_weight_only_nvfp4():
    q = get_serialized_fp8_quantization_config(ignored_layers=["x"], wire_format=NVFP4)
    w = q["config_groups"]["group_0"]
    assert q["format"] == "nvfp4-pack-quantized" and q["ignore"] == ["x"]
    assert w["weights"]["strategy"] == "tensor_group" and w["weights"]["group_size"] == 16
    assert "input_activations" not in w


def test_model_without_nvfp4_support_rejected():
    class Glm:
        model_type = "glm_moe_dsa"

    with pytest.raises(ValueError):
        resolve_serialized_fp8_config(NVFP4, Glm())


def test_groups_in_one_layer_are_not_mixed(config):
    """qkv, gate/up and the GDN in-projections share a layer prefix but are separate fused modules."""
    config, calls = config
    p = "model.layers.5."
    stream = [
        (p + "self_attn.q_proj.weight", _w(512)),
        (p + "mlp.gate_proj.weight", _w(1024)),
        (p + "self_attn.k_proj.weight", _w(128)),
        (p + "mlp.up_proj.weight", _w(1024)),
        (p + "self_attn.v_proj.weight", _w(128)),
        (p + "linear_attn.in_proj_qkv.weight", _w(2048)),
        (p + "linear_attn.in_proj_z.weight", _w(1024)),
    ]
    out = dict(iter_serialized_nvfp4_tensors(iter(stream), config))
    assert sorted(calls) == sorted([(2048, 256), (768, 256), (3072, 256)])
    assert out[p + "mlp.gate_proj.weight_global_scale"].item() == 2048.0
    assert out[p + "mlp.up_proj.weight_global_scale"].item() == 2048.0
    assert out[p + "self_attn.q_proj.weight_global_scale"].item() == 768.0
    assert out[p + "linear_attn.in_proj_z.weight_global_scale"].item() == 3072.0


def test_static_input_scale_emitted_when_amax_set(monkeypatch, config):
    config, _ = config
    from dataclasses import replace

    cfg = replace(config, nvfp4_input_amax=64.0)
    name = "model.layers.0.self_attn.o_proj.weight"
    out = dict(iter_serialized_nvfp4_tensors(iter([(name, _w(256))]), cfg))
    assert out["model.layers.0.self_attn.o_proj.input_global_scale"].item() == pytest.approx(448 * 6 / 64.0)
    plain = dict(iter_serialized_nvfp4_tensors(iter([(name, _w(256))]), config))
    assert not any(k.endswith(".input_global_scale") for k in plain)
    q = get_serialized_fp8_quantization_config(wire_format=NVFP4, nvfp4_static_input=True)
    assert "input_activations" in q["config_groups"]["group_0"]
