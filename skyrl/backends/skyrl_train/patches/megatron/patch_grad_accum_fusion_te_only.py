"""Opt-in: gradient-accumulation fusion for Transformer Engine linears without APEX.

``gradient_accumulation_fusion=True`` makes weight-gradient GEMMs accumulate straight into the FP32
``main_grad`` instead of writing a separate gradient that is then added to it. Transformer Engine's
linears (including the grouped MoE expert GEMMs) implement this themselves, but megatron-core's
native ``ColumnParallelLinear`` / ``RowParallelLinear`` need APEX's ``fused_weight_gradient_mlp_cuda``
extension and refuse to build without it, so the whole option is unusable on an APEX-free install.

``SKYRL_GRAD_ACCUM_FUSION_TE_ONLY=1``: when that extension is missing, build the native linears
with the option off (their usual unfused weight gradient) and leave it on for everything else.
Same gradients; only where the accumulation happens changes.
"""

import functools
import os

from loguru import logger

_APPLIED = False


def apply_grad_accum_fusion_te_only_patch() -> None:
    global _APPLIED
    if _APPLIED or os.environ.get("SKYRL_GRAD_ACCUM_FUSION_TE_ONLY", "0").lower() not in ("1", "true"):
        return
    from megatron.core.tensor_parallel import layers

    if layers._grad_accum_fusion_available:
        return

    def without_fusion(init):
        @functools.wraps(init)
        def wrapped(self, *args, **kwargs):
            config = kwargs.get("config")
            if config is None or not getattr(config, "gradient_accumulation_fusion", False):
                return init(self, *args, **kwargs)
            config.gradient_accumulation_fusion = False
            try:
                return init(self, *args, **kwargs)
            finally:
                config.gradient_accumulation_fusion = True

        return wrapped

    layers.ColumnParallelLinear.__init__ = without_fusion(layers.ColumnParallelLinear.__init__)
    layers.RowParallelLinear.__init__ = without_fusion(layers.RowParallelLinear.__init__)
    _APPLIED = True
    logger.info("Gradient-accumulation fusion: Transformer Engine linears only (APEX extension not installed)")
