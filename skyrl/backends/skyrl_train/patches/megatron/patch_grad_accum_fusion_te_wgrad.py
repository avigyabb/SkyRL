"""Opt-in: gradient-accumulation fusion without APEX, through Transformer Engine's GEMM.

``gradient_accumulation_fusion=True`` makes weight-gradient GEMMs accumulate straight into the FP32
``main_grad`` instead of writing a separate gradient that is then added to it. Transformer Engine's
linears implement it themselves, but megatron-core's native ``ColumnParallelLinear`` /
``RowParallelLinear`` (and the pipeline-drain embedding wgrad in ``megatron.core.utils``) call APEX's
``fused_weight_gradient_mlp_cuda`` extension and refuse to build without it, so the option is
unusable on an APEX-free install.

megatron-core already has the replacement: ``tensor_parallel.layers._wgrad_gemm`` (TE
``general_gemm``, FP32 output, ``accumulate=True`` adds into ``out``), which its Megatron-FSDP branch
uses in place of the APEX call. ``SKYRL_GRAD_ACCUM_FUSION_TE_WGRAD=1``: when the APEX extension is
missing and Transformer Engine is present, register a ``fused_weight_gradient_mlp_cuda`` stand-in
whose ``wgrad_gemm_accum_fp32`` / ``wgrad_gemm_accum_fp16`` call ``_wgrad_gemm(main_grad,
grad_output, total_input, accumulate=True)``, and mark the fusion available. megatron-core's call
sites are unchanged; with APEX installed this does nothing.

Remove when megatron-core falls back to ``_wgrad_gemm`` without APEX (no upstream PR as of
NVIDIA/Megatron-LM ``main`` @ 631e1463).
"""

import os
import sys
import types

from loguru import logger

_APPLIED = False


def apply_grad_accum_fusion_te_wgrad_patch() -> None:
    global _APPLIED
    if _APPLIED or os.environ.get("SKYRL_GRAD_ACCUM_FUSION_TE_WGRAD", "0").lower() not in ("1", "true"):
        return
    from megatron.core.extensions.transformer_engine import te_general_gemm
    from megatron.core.tensor_parallel import layers

    if layers._grad_accum_fusion_available or te_general_gemm is None:
        return

    def wgrad_gemm_accum(total_input, grad_output, main_grad):
        layers._wgrad_gemm(main_grad, grad_output, total_input, accumulate=True)

    shim = types.ModuleType("fused_weight_gradient_mlp_cuda")
    shim.__doc__ = "SkyRL stand-in for APEX's extension, backed by megatron-core's _wgrad_gemm (TE general_gemm)."
    shim.wgrad_gemm_accum_fp32 = wgrad_gemm_accum
    shim.wgrad_gemm_accum_fp16 = wgrad_gemm_accum
    sys.modules.setdefault("fused_weight_gradient_mlp_cuda", shim)
    layers.fused_weight_gradient_mlp_cuda = sys.modules["fused_weight_gradient_mlp_cuda"]
    layers._grad_accum_fusion_available = True
    _APPLIED = True
    logger.info("Gradient-accumulation fusion: APEX extension missing, using TE general_gemm (_wgrad_gemm)")
