"""CPU tests for the NVFP4 activation calibrator's norm replay and hook bookkeeping."""

import torch

from skyrl.backends.skyrl_train.workers.megatron.quantization.activation_calibrator import (
    _quantizer_input,
    _strip_wrappers,
)


class _FusedNormLinear(torch.nn.Module):
    """Stand-in for TE LayerNormLinear's norm attributes."""

    def __init__(self, zero_centered):
        super().__init__()
        self.layer_norm_weight = torch.nn.Parameter(torch.rand(16) + 0.5)
        self.eps = 1e-6
        self.normalization = "RMSNorm"
        self.zero_centered_gamma = zero_centered


def test_rmsnorm_replay_matches_reference():
    x = torch.randn(4, 16, dtype=torch.bfloat16) * 5
    for zero_centered in (False, True):
        m = _FusedNormLinear(zero_centered)
        gamma = m.layer_norm_weight.float() + (1.0 if zero_centered else 0.0)
        xf = x.float()
        ref = xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + m.eps) * gamma
        torch.testing.assert_close(_quantizer_input(m, x), ref)


def test_plain_linear_input_is_unchanged():
    x = torch.randn(3, 8)
    assert _quantizer_input(torch.nn.Linear(8, 8), x) is x


def test_strip_wrappers():
    assert _strip_wrappers("module.module.decoder.layers.0.linear_qkv") == "decoder.layers.0.linear_qkv"


def _fake_te_linear(in_features, out_features):
    cls = type("Linear", (torch.nn.Linear,), {"__module__": "transformer_engine.pytorch.module.linear"})
    return cls(in_features, out_features)


def test_calibrator_hooks_maps_and_collects_decayed_max():
    from types import SimpleNamespace

    from skyrl.backends.skyrl_train.workers.megatron.quantization.activation_calibrator import (
        ActivationAmaxCalibrator,
    )

    chunk = torch.nn.Module()
    chunk.decoder = torch.nn.Module()
    chunk.decoder.linear_qkv = _fake_te_linear(8, 8)
    chunk.decoder.norm = torch.nn.LayerNorm(8)  # not a TE linear: must not be hooked
    wrapped = torch.nn.Module()
    wrapped.module = chunk

    qkv = SimpleNamespace(
        global_param_name="module.decoder.linear_qkv.weight",
        mapping=SimpleNamespace(hf_param={"q": "model.q_proj.weight", "k": "model.k_proj.weight"}),
    )
    bridge = SimpleNamespace(get_conversion_tasks=lambda _modules: [qkv])
    cal = ActivationAmaxCalibrator(bridge, [wrapped], decay=0.5, margin=2.0)
    assert cal._names == ["decoder.linear_qkv"]

    chunk.decoder.linear_qkv(torch.full((2, 8), -3.0))
    chunk.decoder.linear_qkv(torch.full((2, 8), 1.0))
    out = cal.collect()
    assert out == {"model.q_proj.weight": 6.0, "model.k_proj.weight": 6.0}  # max |x| = 3, times margin 2

    # Next window is quieter: the previous peak decays by 0.5 instead of vanishing or sticking.
    chunk.decoder.linear_qkv(torch.full((2, 8), 1.0))
    assert cal.collect()["model.q_proj.weight"] == 2.0 * max(1.0, 0.5 * 3.0)

    cal.enabled = False
    chunk.decoder.linear_qkv(torch.full((2, 8), 100.0))
    assert cal.collect()["model.q_proj.weight"] == 2.0 * 0.5 * 1.5  # disabled hook records nothing; peak keeps decaying
    cal.remove()
