"""Muse Spark Open decoder blocks on megatron-core's HybridModel.

Vendored from NVIDIA-NeMo/Megatron-Bridge#6483
(``megatron/bridge/models/muse_spark_open/modeling_muse_spark_open/blocks.py``), adapted to the
pinned Megatron-Bridge / megatron-core:

- ``MuseSelfAttention`` (parameter-free Q/K norms) is inlined: the pinned Bridge only has it fused
  with Glimmer's post norm as ``MuseGlimmerSelfAttention``.
- ``MuseSparkMoE.preprocess`` takes no ``padding_mask``: the pinned ``MoELayer.preprocess`` doesn't.

Each HF decoder block becomes two Hybrid layers, ``*`` (attention) then ``E`` (MoE). The residual
gates sit in the layers' BDA slots so TransformerLayer's recompute/offload machinery still applies.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.utils import sharded_state_dict_default
from torch import Tensor, nn


class InputScaledRMSNorm(nn.Module):
    """``rms_norm(x * (1 + weight))``: HF's ``MuseSparkTextInputScaledRMSNorm``.

    TE's affine RMSNorm scales after normalizing and cannot express this.
    """

    def __init__(self, hidden_size: int, *, eps: float, config: Any, sequence_parallel: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=config.params_dtype))
        self.weight.sequence_parallel = sequence_parallel
        self.norm = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.norm(hidden_states * (1 + self.weight))


class MuseSparkResidualGate(nn.Module):
    """Post-normalize a branch and mix it into the residual with learned sigmoid gates.

    Implements the callable interface TransformerLayer expects in its BDA slot.
    """

    def __init__(self, config: Any, *, input_scaled_norm: bool) -> None:
        super().__init__()
        self.temperature = config.residual_gate_temperature
        self.gate = nn.Parameter(torch.zeros(config.hidden_size, dtype=torch.float32))
        mark_keep_in_fp32(self.gate)
        self.gate.sequence_parallel = config.sequence_parallel
        if config.perform_initialization:
            nn.init.normal_(self.gate, std=config.init_method_std)
        if input_scaled_norm:
            self.post_norm = InputScaledRMSNorm(
                config.hidden_size,
                eps=config.post_norm_epsilon,
                config=config,
                sequence_parallel=config.sequence_parallel,
            )
        else:
            self.post_norm = nn.RMSNorm(config.hidden_size, eps=config.post_norm_epsilon, elementwise_affine=False)

    def forward(self, training: bool, fused: bool) -> Callable:
        return self._apply_branch

    def _apply_branch(self, output_with_bias: tuple[Tensor, Tensor | None], residual: Tensor, dropout: float) -> Tensor:
        if dropout:
            raise ValueError("Muse Spark residual mixing requires hidden_dropout=0.")
        output, bias = output_with_bias
        if bias is not None:
            output = output + bias
        output = self.post_norm(output)
        gate = self.gate.float() / self.temperature
        beta = torch.sigmoid(gate)
        alpha = (torch.sigmoid(-gate) * (1 + beta)).clamp_min(1e-3).sqrt()
        return (alpha * residual + beta * output).to(residual.dtype)


class MuseSparkSelfAttention(SelfAttention):
    """GQA with parameter-free Q/K norms, a per-head output norm, and the sigmoid output gate.

    ``qk_scale_factor`` is folded into ``softmax_scale`` by the provider; the output gate itself is
    megatron-core's ``attention_output_gate``.
    """

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        # TE RMSNorm always has an affine weight; Muse's Q/K norms have none.
        self.q_layernorm = nn.RMSNorm(
            self.hidden_size_per_attention_head, eps=config.layernorm_epsilon, elementwise_affine=False
        )
        self.k_layernorm = nn.RMSNorm(
            self.hidden_size_per_attention_head, eps=config.layernorm_epsilon, elementwise_affine=False
        )
        self.output_norm = nn.RMSNorm(config.kv_channels, eps=config.layernorm_epsilon, elementwise_affine=False)

    def _apply_output_gate(self, hidden_states: Tensor, gate: Tensor) -> Tensor:
        shape = hidden_states.shape
        hidden_states = self.output_norm(hidden_states.reshape(*shape[:-1], -1, self.config.kv_channels))
        return super()._apply_output_gate(hidden_states.reshape(shape), gate)


class MuseSparkRouter(TopKRouter):
    """Sigmoid top-k routing in fp32, with the model's pre-sigmoid logit scale."""

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.weight.data = self.weight.data.float()
        mark_keep_in_fp32(self.weight)

    def _apply(self, fn: Callable, recurse: bool = True) -> "MuseSparkRouter":
        """Keep the fp32 correction bias across device and dtype moves."""
        bias = self.expert_bias
        result = super()._apply(fn, recurse=recurse)
        if bias is not None:
            # Module._apply replaces buffers, so re-wrap the moved values.
            self.expert_bias = mark_keep_in_fp32(bias.to(device=self.expert_bias.device, dtype=torch.float32))
        return result

    def gating(self, input: Tensor) -> Tensor:
        # TE's gating GEMM needs matching operand dtypes; the router weight stays fp32 as in HF.
        return super().gating(input.to(self.weight.dtype)) * self.config.output_multiplier


class MuseSparkExperts(TEGroupedMLP):
    """Normalize each expert's output before applying its routing weight."""

    def __init__(self, num_local_experts: int, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(num_local_experts, config, *args, **kwargs)
        self.attention_tp_group = kwargs["pg_collection"].tp
        if config.expert_tensor_parallel_size != 1:
            raise ValueError("Muse Spark expert output normalization requires expert_tensor_parallel_size=1.")
        self.post_expert_norm = InputScaledRMSNorm(
            config.moe_latent_size,
            eps=config.post_norm_epsilon,
            config=config,
            sequence_parallel=config.sequence_parallel,
        )

    def forward(
        self, permuted_local_hidden_states: Tensor, tokens_per_expert: Tensor, permuted_probs: Tensor
    ) -> tuple[Tensor, None]:
        # Run the grouped GEMMs unweighted, then norm, then weight.
        output, bias = super().forward(permuted_local_hidden_states, tokens_per_expert, torch.ones_like(permuted_probs))
        if bias is not None:
            raise ValueError("Muse Spark experts require bias-free projections.")
        output = self.post_expert_norm(output)
        return (output * permuted_probs.reshape(-1, 1)).to(output.dtype), None

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None) -> dict:
        state = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        # The norm is shared by all experts, so it replicates over attention TP (and DP), not
        # over the expert TP group TEGroupedMLP uses for its children.
        state.update(
            sharded_state_dict_default(
                self.post_expert_norm,
                f"{prefix}post_expert_norm.",
                sharded_offsets,
                metadata,
                tp_group=self.attention_tp_group,
            )
        )
        return state


class MuseSparkMoE(MoELayer):
    """Latent MoE (``fc1_latent_proj`` / ``fc2_latent_proj``) with a pre-expert RMSNorm."""

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.pre_expert_norm = TENorm(config, config.moe_latent_size, eps=config.layernorm_epsilon)

    def preprocess(self, hidden_states: Tensor, probs: Tensor, routing_map: Tensor) -> tuple[Tensor, Tensor]:
        hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states = self.pre_expert_norm(hidden_states)
        return self.token_dispatcher.dispatch_preprocess(hidden_states, routing_map, probs)


def get_muse_spark_hybrid_stack_spec(config: Any) -> ModuleSpec:
    """Attention (``*``) and MoE (``E``) Hybrid layers for the repeated ``*E`` pattern."""
    experts = partial(
        MuseSparkExperts,
        submodules=GroupedMLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
        ),
    )
    moe = partial(MuseSparkMoE, submodules=MoESubmodules(experts=experts, router=MuseSparkRouter))
    return ModuleSpec(
        module=HybridStack,
        submodules=HybridStackSubmodules(
            attention_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=TENorm,
                    self_attention=ModuleSpec(
                        module=MuseSparkSelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=TEColumnParallelLinear,
                            core_attention=TEDotProductAttention,
                            linear_proj=TERowParallelLinear,
                        ),
                    ),
                    self_attn_bda=ModuleSpec(
                        module=MuseSparkResidualGate, params={"config": config, "input_scaled_norm": False}
                    ),
                ),
            ),
            moe_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    pre_mlp_layernorm=TENorm,
                    mlp=moe,
                    mlp_bda=ModuleSpec(
                        module=MuseSparkResidualGate, params={"config": config, "input_scaled_norm": True}
                    ),
                ),
            ),
        ),
    )
