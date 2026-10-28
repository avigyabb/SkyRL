"""``patch_mhc_recompute_mappings``: the mhc recompute path no longer keeps an fp32 copy of the hidden states.

CPU, one real ``HyperConnectionModule`` (unfused path, torch.compile off); the fused GPU path is checked by
``examples/train/deepseek_v4_flash/dev/check_mhc_recompute.py``.

Run with:
uv run --isolated --extra dev --extra megatron pytest \
    tests/backends/skyrl_train/patches/megatron/test_mhc_recompute_mappings.py
"""

import pytest
import torch

pytestmark = pytest.mark.megatron

S, N, C = 64, 4, 128


def _run(saved_bytes):
    from megatron.core.tensor_parallel.random import CheckpointWithoutOutputManager
    from megatron.core.transformer.hyper_connection import HyperConnectionModule
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.manual_seed(0)
    config = TransformerConfig(
        num_layers=1, hidden_size=C, num_attention_heads=4, enable_mhc_connections=True, mhc_num_residual_streams=N
    )
    module = HyperConnectionModule(config, layer_number=1)
    with torch.no_grad():
        module.mapping_proj.weight.normal_(0, 0.02)
        module.bias.normal_(0, 0.1)
    x = torch.randn(S, 1, N * C, dtype=torch.bfloat16, requires_grad=True)

    def pack(t):
        saved_bytes.append(t.numel() * t.element_size())
        return t

    manager = CheckpointWithoutOutputManager()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        aggregated, h_res, h_post, _ = module._forward_with_checkpoint(x, manager)
        layer_out = aggregated * 1.5  # stand-in for the wrapped layer
    manager.discard_all_outputs_and_register_unified_recompute(layer_out)
    loss = layer_out.float().square().mean() + (h_res.float() * 3).sin().sum() + h_post.float().square().sum()
    loss.backward()
    return [t.detach().clone() for t in (layer_out, h_res, h_post)], [x.grad] + [p.grad for p in module.parameters()]


def test_mhc_recompute_mappings_keeps_grads_and_drops_the_fp32_copy(monkeypatch):
    from megatron.core.transformer.hyper_connection import HyperConnectionModule

    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_mhc_recompute_mappings as mod,
    )

    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    monkeypatch.setattr(
        HyperConnectionModule, "_forward_with_checkpoint", HyperConnectionModule._forward_with_checkpoint
    )
    base_saved, patched_saved = [], []
    base_out, base_grads = _run(base_saved)

    monkeypatch.setattr(mod, "_APPLIED", False)
    assert mod.apply_mhc_recompute_mappings_patch()
    assert mod.apply_mhc_recompute_mappings_patch()  # idempotent: wrapped once
    out, grads = _run(patched_saved)

    for a, b in zip(base_out + base_grads, out + grads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    fp32_copy = S * N * C * 4
    assert sum(base_saved) - sum(patched_saved) >= fp32_copy
