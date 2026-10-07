from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
    _load_batched_moe_fp8_tensor,
)
from skyrl.backends.skyrl_train.weight_sync.fp8 import (
    SKYRL_BATCHED_MOE_FP8_PREFIX,
)


def test_batched_moe_tensor_uses_one_full_expert_loader_call():
    calls = []

    def weight_loader(param, loaded_weight, weight_name, *, shard_id, expert_id, return_success):
        calls.append((param, loaded_weight, weight_name, shard_id, expert_id, return_success))
        return True

    weight_loader.supports_moe_loading = True
    param = torch.nn.Parameter(torch.empty(3, 8, 4), requires_grad=False)
    param.weight_loader = weight_loader
    target_name = "model.layers.0.mlp.experts.w13_weight"
    loaded_weight = torch.randn(3, 4, 4)
    wire_name = f"{SKYRL_BATCHED_MOE_FP8_PREFIX}model.layers.0.mlp.experts.gate_proj.weight"

    loaded = _load_batched_moe_fp8_tensor(
        SimpleNamespace(),
        {target_name: param},
        wire_name,
        loaded_weight,
    )

    assert loaded
    assert len(calls) == 1
    assert calls[0][1] is loaded_weight
    assert calls[0][2:] == (target_name, "w1", 0, True)


def test_batched_moe_scale_maps_to_fused_scale_parameter():
    calls = []

    def weight_loader(param, loaded_weight, weight_name, *, shard_id, expert_id, return_success):
        calls.append((weight_name, shard_id, tuple(loaded_weight.shape)))
        return True

    weight_loader.supports_moe_loading = True
    param = torch.nn.Parameter(torch.empty(2, 6, 3), requires_grad=False)
    param.weight_loader = weight_loader
    target_name = "language_model.model.layers.2.mlp.experts.w13_weight_scale_inv"
    loaded_weight = torch.randn(2, 3, 3)
    mapper = SimpleNamespace(
        apply_list=lambda names: [names[0].replace("model.language_model.", "language_model.model.", 1)]
    )
    model = SimpleNamespace(hf_to_vllm_mapper=mapper)
    wire_name = f"{SKYRL_BATCHED_MOE_FP8_PREFIX}" "model.language_model.layers.2.mlp.experts.up_proj.weight_scale_inv"

    assert _load_batched_moe_fp8_tensor(model, {target_name: param}, wire_name, loaded_weight)
    assert calls == [(target_name, "w3", (2, 3, 3))]


def test_batched_moe_resolves_routed_experts_nesting():
    """vLLM 0.26: expert params live on MoERunner's RoutedExperts submodule."""

    calls = []

    def weight_loader(param, loaded_weight, weight_name, *, shard_id, expert_id, return_success):
        calls.append((weight_name, shard_id, expert_id))
        return True

    weight_loader.supports_moe_loading = True
    param = torch.nn.Parameter(torch.empty(3, 8, 4), requires_grad=False)
    param.weight_loader = weight_loader
    nested_name = "language_model.model.layers.0.mlp.experts.routed_experts.w13_weight"
    mapper = SimpleNamespace(
        apply_list=lambda names: [names[0].replace("model.language_model.", "language_model.model.", 1)]
    )
    model = SimpleNamespace(hf_to_vllm_mapper=mapper)
    wire_name = f"{SKYRL_BATCHED_MOE_FP8_PREFIX}model.language_model.layers.0.mlp.experts.gate_proj.weight"

    loaded = _load_batched_moe_fp8_tensor(model, {nested_name: param}, wire_name, torch.randn(3, 4, 4))

    assert loaded
    assert calls == [(nested_name, "w1", 0)]


def test_batched_moe_missing_target_error_names_both_candidates():
    wire_name = f"{SKYRL_BATCHED_MOE_FP8_PREFIX}model.layers.0.mlp.experts.down_proj.weight"

    with pytest.raises(ValueError) as excinfo:
        _load_batched_moe_fp8_tensor(SimpleNamespace(), {}, wire_name, torch.randn(2, 4, 4))

    message = str(excinfo.value)
    assert "model.layers.0.mlp.experts.w2_weight" in message
    assert "model.layers.0.mlp.experts.routed_experts.w2_weight" in message


def test_nvfp4_global_scale_loads_one_expert_at_a_time_even_when_all_experts_are_local():
    """vLLM's per-tensor scale loader writes one scalar into one expert's slot, so a ``[E, 1, 1]``
    global scale must be unbound per expert instead of taking the single full-load call."""

    calls = []

    def weight_loader(param, loaded_weight, weight_name, *, shard_id, expert_id, return_success):
        calls.append((weight_name, shard_id, expert_id, tuple(loaded_weight.shape)))
        return True

    weight_loader.supports_moe_loading = True
    param = torch.nn.Parameter(torch.empty(3, 2), requires_grad=False)
    param.weight_loader = weight_loader
    target_name = "model.layers.0.mlp.experts.w13_weight_global_scale"
    wire_name = f"{SKYRL_BATCHED_MOE_FP8_PREFIX}model.layers.0.mlp.experts.up_proj.weight_global_scale"

    assert _load_batched_moe_fp8_tensor(SimpleNamespace(), {target_name: param}, wire_name, torch.ones(3, 1, 1))
    assert calls == [(target_name, "w3", expert, (1, 1)) for expert in range(3)]


def test_nvfp4_packed_and_input_scale_targets():
    calls = []

    def weight_loader(param, loaded_weight, weight_name, *, shard_id, expert_id, return_success):
        calls.append((weight_name, shard_id, expert_id))
        return True

    weight_loader.supports_moe_loading = True
    packed = torch.nn.Parameter(torch.empty(2, 8, 4), requires_grad=False)
    packed.weight_loader = weight_loader
    input_scale = torch.nn.Parameter(torch.empty(2), requires_grad=False)
    input_scale.weight_loader = weight_loader
    params = {
        "model.layers.0.mlp.experts.w2_weight_packed": packed,
        "model.layers.0.mlp.experts.w2_input_global_scale": input_scale,
    }
    base = f"{SKYRL_BATCHED_MOE_FP8_PREFIX}model.layers.0.mlp.experts.down_proj"

    assert _load_batched_moe_fp8_tensor(SimpleNamespace(), params, base + ".weight_packed", torch.zeros(2, 8, 4))
    assert _load_batched_moe_fp8_tensor(SimpleNamespace(), params, base + ".input_global_scale", torch.ones(2, 1, 1))
    assert calls == [
        ("model.layers.0.mlp.experts.w2_weight_packed", "w2", 0),
        ("model.layers.0.mlp.experts.w2_input_global_scale", "w2", 0),
        ("model.layers.0.mlp.experts.w2_input_global_scale", "w2", 1),
    ]
