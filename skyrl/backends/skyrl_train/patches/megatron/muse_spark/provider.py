"""Model provider for the Muse Spark Open language model."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from megatron.bridge.models.gemma.modules import extend_instance
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.spec_utils import ModuleSpec
from torch import Tensor, nn

from skyrl.backends.skyrl_train.patches.megatron.muse_spark.blocks import (
    get_muse_spark_hybrid_stack_spec,
)


class MuseSparkEmbeddingMixin(nn.Module):
    """Parameter-free RMSNorm on the token embeddings (HF ``MuseSparkTextNormedEmbedding``)."""

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        embeddings = super().forward(*args, **kwargs)
        return F.rms_norm(embeddings, (embeddings.shape[-1],), eps=self.config.layernorm_epsilon)


class MuseSparkOutputLayerMixin(nn.Module):
    """``softcap * tanh(logits * output_multiplier / softcap)`` on the LM head output."""

    def forward(self, *args: Any, **kwargs: Any) -> tuple[Tensor, Optional[Tensor]]:
        output, bias = super().forward(*args, **kwargs)
        output = output * self.config.output_multiplier
        softcap = self.config.final_logit_softcapping
        if softcap:
            output = softcap * torch.tanh(output / softcap)
        return output, bias


@dataclass
class MuseSparkModelProvider(HybridModelProvider):
    """Megatron configuration and provider for the Muse Spark Open language model.

    Each HF decoder block maps to two Hybrid layers: attention (``*``) then latent MoE (``E``), so
    ``num_layers`` is twice HF's ``num_hidden_layers`` and the per-layer schedules
    (``window_attn_skip_freq``, ``no_rope_freq``) have a ``0`` entry for every MoE layer.

    Only the language model is built. The checkpoint's vision and audio towers stay in the
    inference engine, so training needs ``language_model_only=True``.
    """

    hybrid_stack_spec: Union[ModuleSpec, Callable] = get_muse_spark_hybrid_stack_spec

    # Muse-specific fields read by the blocks and the mixins above.
    post_norm_epsilon: float = 1e-8
    residual_gate_temperature: float = 0.3
    output_multiplier: float = 1.0
    final_logit_softcapping: Optional[float] = None

    def finalize(self) -> None:
        if self.expert_tensor_parallel_size not in (None, 1):
            raise ValueError("Muse Spark requires expert_tensor_parallel_size=1 (shared post-expert norm).")
        if self.transformer_impl != "transformer_engine":
            raise ValueError("Muse Spark uses Transformer Engine attention and grouped experts.")
        if self.hidden_dropout or self.add_bias_linear or self.add_qkv_bias:
            raise ValueError("Muse Spark requires bias-free projections and hidden_dropout=0.")
        if self.moe_apply_probs_on_input or self.moe_expert_capacity_factor is not None:
            raise ValueError("Muse Spark requires dropless routing with probabilities applied after the expert norm.")
        if self.mtp_num_layers:
            raise ValueError("Muse Spark Open has no MTP layers.")
        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> HybridModel:
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        if model.pre_process:
            extend_instance(model.embedding, MuseSparkEmbeddingMixin)
        if model.post_process:
            extend_instance(model.output_layer, MuseSparkOutputLayerMixin)
        return model
