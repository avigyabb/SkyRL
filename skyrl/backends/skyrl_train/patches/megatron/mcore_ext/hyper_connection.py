"""Standard-RMSNorm input normalization for megatron-core's mHC module.

``HyperConnectionModule`` normalizes the flattened residual streams as ``x / (rms(x) + eps)``
with ``eps`` hard-coded to 1e-6. GLM-5.3-Flash instead uses a standard RMSNorm,
``x * rsqrt(mean(x^2) + rms_norm_eps)``. The two agree for O(1) activations but not for small
residual streams -- this model's embeddings have a per-token rms below ``sqrt(1e-5)``, where the
placement of the epsilon changes the mixing weights materially.

DELETE THIS MODULE once ``TransformerConfig`` carries the input-norm knobs upstream
(``mhc_norm_eps`` / ``mhc_norm_eps_inside_sqrt``, read by ``HyperConnectionModule`` itself).
"""

import os
from functools import partial
from typing import Tuple

import torch
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor
from torch.utils.checkpoint import checkpoint


class RMSNormInputHyperConnectionModule(HyperConnectionModule):
    """mHC module whose input normalization is a standard RMSNorm.

    Reads ``mhc_norm_eps`` from the config, falling back to ``layernorm_epsilon``.
    """

    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__(config, layer_number)
        if config.use_fused_mhc:
            raise NotImplementedError(
                "The fused mHC kernels implement the 1/(rms+eps) input normalization only; "
                "use_fused_mhc is not compatible with mhc_norm_eps_inside_sqrt=True."
            )
        self.norm_eps = getattr(config, "mhc_norm_eps", None) or config.layernorm_epsilon
        if _FUSED_POST_BDA:
            # Only the input norm is incompatible with use_fused_mhc; megatron's Triton
            # ``H_res.T @ residual + H_post * x`` kernel is ~3x faster than the compiled bmm (n=4).
            from megatron.core.fusions.fused_mhc_kernels import fused_h_post_bda

            self._h_post_bda_op = partial(fused_h_post_bda, backend="triton")

    def _projection_and_get_norm(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Projection + standard RMS normalization.

        Args:
            x: [s, b, n*C] - n-stream hidden states
        """
        s, b, nC = x.shape
        if _FUSED_PROJ_RMS:
            from skyrl.backends.skyrl_train.patches.megatron.mcore_ext.mhc_proj_rms import FusedProjRms

            proj, r = FusedProjRms.apply(x.reshape(s * b, nC), self.mapping_proj.weight, self.norm_eps)
            return proj.view(s, b, -1), r.view(s, b, 1)
        # The mHC mapping runs in FP32 (the parameters are kept in FP32 and the activations are
        # upcast in _proj_rms); compute_mappings casts the bounded mixing weights back down.
        # Checkpointed so backward keeps only the activation-dtype input, not its FP32 upcast
        # (2 GiB per mHC site at 32k tokens per rank); the upcast and the math rerun in backward.
        proj, r = checkpoint(
            _proj_rms_compiled if _COMPILE_PROJ_RMS else _proj_rms,
            x.reshape(s * b, nC),
            self.mapping_proj.weight,
            self.norm_eps,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        return proj.view(s, b, -1), r.view(s, b, 1)


def _proj_rms(x: Tensor, weight: Tensor, eps: float) -> Tuple[Tensor, Tensor]:
    """FP32 projection and standard RMS factor ``rsqrt(mean(x^2) + eps)`` of the activation-dtype ``x``."""
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    proj = torch.matmul(x, weight.t())
    r = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return proj, r


# Opt-in: fuse the FP32 upcast, RMS reduction and their backward (eager: ~6 elementwise kernels
# over the [tokens, n*C] stream per mHC site, forward plus recompute plus backward).
_COMPILE_PROJ_RMS = os.environ.get("SKYRL_MHC_COMPILE_PROJ_RMS", "0").lower() in ("1", "true")
_proj_rms_compiled = torch.compile(_proj_rms, dynamic=True)

# Opt-in: one-pass Triton projection + RMS factor, with the mapping weight frozen (LoRA) or trained
# (full fine-tuning); see ``mhc_proj_rms``.
_FUSED_PROJ_RMS = os.environ.get("SKYRL_MHC_FUSED_PROJ_RMS", "0").lower() in ("1", "true")

# Opt-in: megatron's fused Triton h_post_bda (see RMSNormInputHyperConnectionModule.__init__).
_FUSED_POST_BDA = os.environ.get("SKYRL_MHC_FUSED_POST_BDA", "0").lower() in ("1", "true")
