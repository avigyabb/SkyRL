"""Hugging Face <-> Megatron bridge for the Muse Spark Open language model.

Only the language model is bridged (``model.language_model.*`` and ``lm_head``). The vision and
audio towers of the multimodal checkpoint stay in the inference engine.

The config translation and parameter mappings follow Megatron-Bridge#6483's ``MuseSparkBridge``
(``hf_config_to_model_config`` / ``mapping_registry``), re-expressed on the provider API that
SkyRL's Megatron worker drives.
"""

import math
import re

import torch
import torch.nn.functional as F
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    ReplicatedMapping,
    RMSNorm2ZeroCenteredRMSNormMapping,
    split_qkv_weights,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.muse_glimmer.muse_glimmer_bridge import (
    MuseGlimmerQKVGMapping,
)
from megatron.bridge.peft.lora_merge import LoRAMerge
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.utils import unwrap_model

from skyrl.backends.skyrl_train.patches.megatron.muse_spark.provider import (
    MuseSparkModelProvider,
)

_QKVG_PART = re.compile(r"\.self_attn\.(q_proj|k_proj|v_proj|gate_proj)\.weight$")


def _interleave_moe(values: list[int]) -> list[int]:
    """Per-HF-layer flags -> per-Hybrid-layer flags (``[a, 0]`` for each attention + MoE pair)."""
    return [v for value in values for v in (value, 0)]


@MegatronModelBridge.register_bridge(
    source="MuseSparkForConditionalGeneration",
    target=HybridModel,
    provider=MuseSparkModelProvider,
    model_type="muse_spark",
)
class MuseSparkBridge(MegatronModelBridge):
    """Megatron Bridge for the Muse Spark Open language model.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("meta-models/Muse-Spark-Open-816B-A42B")
        >>> provider = bridge.to_megatron_provider()
    """

    # fp32 in the checkpoint, in Megatron and in vLLM; MegatronWeightSource keeps them fp32 on the
    # wire, since bf16-rounding the router changes which experts tokens go to.
    FP32_HF_PARAM_SUFFIXES = ("mlp.gate.weight", "mlp.gate.e_score_correction_bias")

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MuseSparkModelProvider:
        hf_config = hf_pretrained.config
        text = getattr(hf_config, "text_config", hf_config)
        num_layers = text.num_hidden_layers

        if set(text.mlp_layer_types) != {"sparse"} or getattr(text, "num_shared_experts", 0):
            raise ValueError("Muse Spark: only all-sparse layers without shared experts are supported.")
        rope_base = float(text.rope_parameters["rope_theta"])
        if text.rope_parameters.get("rope_type", "default") != "default" or any(
            theta not in (0, rope_base) for theta in text.layer_rope_theta
        ):
            raise ValueError("Muse Spark: only one RoPE base with per-layer NoPE is supported.")

        params_dtype = self.dtype_from_hf(hf_config, default=torch.bfloat16)
        provider = MuseSparkModelProvider(
            hybrid_layer_pattern="*E" * num_layers,
            num_layers=2 * num_layers,
            hidden_size=text.hidden_size,
            ffn_hidden_size=text.intermediate_size,
            num_attention_heads=text.num_attention_heads,
            num_query_groups=text.num_key_value_heads,
            kv_channels=text.head_dim,
            vocab_size=text.vocab_size,
            should_pad_vocab=False,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text.vocab_size),
            seq_length=text.max_position_embeddings,
            share_embeddings_and_output_weights=bool(
                getattr(hf_config, "tie_word_embeddings", text.tie_word_embeddings)
            ),
            position_embedding_type="rope",
            rotary_base=rope_base,
            normalization="RMSNorm",
            layernorm_epsilon=text.rms_norm_eps,
            # HF's affine norms compute x_hat * (1 + w).
            layernorm_zero_centered_gamma=True,
            post_norm_epsilon=text.post_norm_eps,
            residual_gate_temperature=text.residual_gate_temperature,
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            add_qkv_bias=False,
            hidden_dropout=0.0,
            attention_dropout=float(text.attention_dropout),
            # MuseSparkSelfAttention installs its own parameter-free Q/K norms.
            qk_layernorm=False,
            attention_output_gate=True,
            # HF scales the normalized query by qk_scale_factor on top of 1/sqrt(head_dim).
            softmax_scale=float(text.qk_scale_factor) / math.sqrt(text.head_dim),
            # HF's sliding window counts the query token itself.
            window_size=(int(text.sliding_window) - 1, 0),
            window_attn_skip_freq=_interleave_moe([int(t == "sliding_attention") for t in text.layer_types]),
            no_rope_freq=_interleave_moe([int(theta == 0) for theta in text.layer_rope_theta]),
            num_moe_experts=text.num_local_experts,
            moe_router_topk=text.num_experts_per_tok,
            moe_latent_size=text.moe_hidden_size,
            moe_ffn_hidden_size=text.moe_intermediate_size,
            moe_router_score_function="sigmoid",
            moe_router_dtype="fp32",
            moe_router_enable_expert_bias=True,
            # e_score_correction_bias is a fixed checkpoint value; never update it during RL.
            moe_router_bias_update_rate=0.0,
            moe_router_topk_scaling_factor=float(text.routed_scaling_factor),
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            expert_tensor_parallel_size=1,
            init_method_std=text.initializer_range,
            params_dtype=params_dtype,
            fp16=params_dtype == torch.float16,
            bf16=params_dtype == torch.bfloat16,
            output_multiplier=float(text.output_multiplier),
            final_logit_softcapping=text.final_logit_softcapping,
        )
        return provider

    def _merge_lora_adapter_weights(self, megatron_model, converted_weights_dict, adapter_weights):
        """Merge a ``linear_qkv`` LoRA into the four HF tensors it fuses (q, k, v and the gate).

        Bridge's generic merge recognizes fused QKV and GDN layouts, not QKV + output gate, and
        would add the whole fused delta to ``q_proj``. Build the delta in Megatron's fused layout
        (the adapter's B is TP-gathered here) and split it the way ``MuseGlimmerQKVGMapping`` splits
        the base weight.
        """
        names = list(converted_weights_dict)
        parts = {name: _QKVG_PART.search(name) for name in names}
        if not names or not all(parts.values()) or len(adapter_weights) != 1:
            return super()._merge_lora_adapter_weights(megatron_model, converted_weights_dict, adapter_weights)

        adapter = adapter_weights[0]
        config = unwrap_model(megatron_model)[0].config
        base = converted_weights_dict[names[0]]
        lora_out = adapter.linear_out_weight.weight.to(device=base.device, dtype=torch.float32)
        lora_in = adapter.linear_in_weight.weight.to(device=base.device, dtype=torch.float32)
        zeros = torch.zeros((lora_out.shape[0], lora_in.shape[1]), device=base.device, dtype=torch.float32)
        delta = LoRAMerge().merge(zeros, lora_out, lora_in, adapter.alpha, adapter.dim, tp_group=None)

        query_with_gate, key, value = split_qkv_weights(config, delta)
        head_dim = config.kv_channels
        query, gate = query_with_gate.view(config.num_attention_heads, 2 * head_dim, -1).split(head_dim, dim=1)
        split = {
            "q_proj": query.reshape(-1, delta.shape[-1]),
            "gate_proj": gate.reshape(-1, delta.shape[-1]),
            "k_proj": key,
            "v_proj": value,
        }
        for name, match in parts.items():
            base_weight = converted_weights_dict[name]
            converted_weights_dict[name] = (base_weight.float() + split[match.group(1)]).to(base_weight.dtype)
        return converted_weights_dict

    def mapping_registry(self) -> MegatronMappingRegistry:
        hf_config = self.hf_config
        text_only = not hasattr(hf_config, "text_config")
        text = hf_config if text_only else hf_config.text_config
        prefix = "model" if text_only else "model.language_model"
        mappings = [
            AutoMapping("embedding.word_embeddings.weight", f"{prefix}.embed_tokens.weight"),
            AutoMapping("output_layer.weight", "lm_head.weight"),
            # HF's final norm is a plain affine RMSNorm; Megatron's is zero-centered.
            RMSNorm2ZeroCenteredRMSNormMapping("decoder.final_norm.weight", f"{prefix}.norm.weight"),
        ]
        for index in range(text.num_hidden_layers):
            hf = f"{prefix}.layers.{index}"
            attention = f"decoder.layers.{2 * index}"
            moe = f"decoder.layers.{2 * index + 1}"
            mappings.extend(
                [
                    MuseGlimmerQKVGMapping(
                        f"{attention}.self_attention.linear_qkv.weight",
                        q=f"{hf}.self_attn.q_proj.weight",
                        k=f"{hf}.self_attn.k_proj.weight",
                        v=f"{hf}.self_attn.v_proj.weight",
                        gate=f"{hf}.self_attn.gate_proj.weight",
                    ),
                    AutoMapping(f"{attention}.self_attention.linear_proj.weight", f"{hf}.self_attn.o_proj.weight"),
                    ReplicatedMapping(f"{attention}.input_layernorm.weight", f"{hf}.input_layernorm.weight"),
                    ReplicatedMapping(f"{attention}.self_attn_bda.gate", f"{hf}.post_attention_residual_gate.gate"),
                    ReplicatedMapping(f"{moe}.pre_mlp_layernorm.weight", f"{hf}.pre_feedforward_layernorm.weight"),
                    ReplicatedMapping(f"{moe}.mlp_bda.gate", f"{hf}.post_feedforward_residual_gate.gate"),
                    ReplicatedMapping(f"{moe}.mlp_bda.post_norm.weight", f"{hf}.post_feedforward_layernorm.weight"),
                    ReplicatedMapping(f"{moe}.mlp.router.weight", f"{hf}.mlp.gate.weight"),
                    ReplicatedMapping(f"{moe}.mlp.router.expert_bias", f"{hf}.mlp.gate.e_score_correction_bias"),
                    ReplicatedMapping(f"{moe}.mlp.fc1_latent_proj.weight", f"{hf}.mlp.pre_expert_proj.weight"),
                    ReplicatedMapping(f"{moe}.mlp.pre_expert_norm.weight", f"{hf}.mlp.pre_expert_norm.weight"),
                    ReplicatedMapping(f"{moe}.mlp.fc2_latent_proj.weight", f"{hf}.mlp.post_expert_proj.weight"),
                    ReplicatedMapping(
                        f"{moe}.mlp.experts.post_expert_norm.weight", f"{hf}.mlp.experts.post_expert_norm.weight"
                    ),
                    FusedGatedExpertMapping(f"{moe}.mlp.experts.linear_fc1.weight*", f"{hf}.mlp.experts.gate_up_proj"),
                    FusedExpertMapping(f"{moe}.mlp.experts.linear_fc2.weight*", f"{hf}.mlp.experts.down_proj"),
                ]
            )
        return MegatronMappingRegistry(*mappings)
