"""Serialized wire formats for vLLM FP8 checkpoints: blockwise and MXFP8.

The two wires side by side — every branch in this package is one row of this
table, keyed off ``SerializedFp8Config.wire_format``:

                        blockwise                    mxfp8
  trainer recipe        Float8BlockScaling           MXFP8BlockScaling
  scale granularity     128x128 tiles                1x32 groups (along K)
  scale encoding        FP32 (or power-of-2)         E8M0 biased exponent, uint8
  scale tensor          .weight_scale_inv            .weight_scale
  vLLM quantization     fp8                          compressed-tensors
  cast kernels          blockwise_cast_to_fp8 /      mx_cast_to_fp8 /
                        batched_blockwise_cast...    batched_mx_cast_to_fp8

Sender (``SerializedFp8WeightSource``) and receiver (worker-extension loaders) are
wire-format-agnostic; the format decides only what this serializer emits, the
quantization config injected at engine boot, and the per-model ignore lists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import torch

from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import (
    AUTO_FP8,
    BLOCKWISE_FP8,
    MXFP8,
    NVFP4,
    NVFP4_GLOBAL_SCALE_SUFFIX,
    NVFP4_PACKED_SUFFIX,
    WIRE_FORMATS,
    WIRE_SCALE_SUFFIX,
    ModelFp8Spec,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.quantize import (
    MXFP8_GROUP_SIZE,
    NVFP4_GROUP_SIZE,
    batched_blockwise_cast_to_fp8,
    batched_mx_cast_to_fp8,
    blockwise_cast_to_fp8,
    mx_cast_to_fp8,
    normalize_block_size,
    nvfp4_cast,
    use_power_2_scales_default,
)

__all__ = [  # re-exported so callers keep importing formats from the serializer
    "AUTO_FP8",
    "BLOCKWISE_FP8",
    "MXFP8",
    "NVFP4",
    "WIRE_FORMATS",
]

# Internal wire-format marker for Qwen3.5 MoE tensors that remain batched over
# experts. The receiver strips this marker and routes the tensor directly to
# vLLM's fused-MoE parameter loader instead of the ordinary HF-name loader.
SKYRL_BATCHED_MOE_FP8_PREFIX = "__skyrl_batched_moe_fp8__:"


@dataclass(frozen=True)
class SerializedFp8Config:
    """Configuration for serialized FP8 rollout weight sync.

    ``spec`` is the per-model quantization policy, resolved once from the HF
    config via ``resolve_fp8_spec``; the tensor iterators require it.
    """

    # blockwise-wire parameters; the MXFP8 wire has no equivalents (its group
    # geometry and scale encoding are fixed by the OCP microscaling spec).
    weight_block_size: tuple[int, int] = (128, 128)
    power_2_scale: bool = field(default_factory=use_power_2_scales_default)
    # shared across wires
    spec: ModelFp8Spec | None = None
    wire_format: str = BLOCKWISE_FP8
    # NVFP4 wire only. ``None`` ships weight-only NVFP4 (W4A16). A float is the static activation
    # amax every quantized linear's ``input_global_scale`` is derived from, which makes vLLM quantize
    # activations to FP4 too (W4A4).
    nvfp4_input_amax: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight_block_size", normalize_block_size(self.weight_block_size))
        if type(self.power_2_scale) is not bool:
            raise ValueError(f"power_2_scale must be a bool, got {self.power_2_scale!r}")
        if self.wire_format not in WIRE_FORMATS:
            raise ValueError(f"wire_format must be one of {WIRE_FORMATS}, got {self.wire_format!r}")

    @property
    def is_mxfp8(self) -> bool:
        return self.wire_format == MXFP8

    @property
    def is_nvfp4(self) -> bool:
        return self.wire_format == NVFP4

    def require_spec(self) -> ModelFp8Spec:
        if self.spec is None:
            raise ValueError(
                "SerializedFp8Config.spec is not set; resolve the model spec with "
                "resolve_fp8_spec(hf_config) before serializing weights"
            )
        return self.spec


def resolve_serialized_fp8_config(
    fp8_weight_sync_mode: str | None,
    hf_config: object | None,
) -> SerializedFp8Config | None:
    """Turn ``fp8_weight_sync_mode`` into the sender's serializer config.

    The single place a wire-format name becomes a ``SerializedFp8Config``, so a
    gate that admits one wire and not another cannot diverge between backends:
    every wire in ``WIRE_FORMATS`` is usable end to end, or none is.

    Returns None when FP8 weight sync is off. Raises ``ValueError`` for a wire
    outside ``WIRE_FORMATS`` (``"auto"`` included -- it must already be resolved
    to a concrete wire by then) or for a checkpoint with no registered spec.
    """
    from skyrl.backends.skyrl_train.weight_sync.fp8.models import (
        registered_fp8_spec_names,
        resolve_fp8_spec,
    )

    if fp8_weight_sync_mode is None:
        return None
    if fp8_weight_sync_mode not in WIRE_FORMATS:
        raise ValueError(
            f"Unsupported fp8_weight_sync_mode={fp8_weight_sync_mode!r}. "
            f"Supported values: {', '.join(WIRE_FORMATS)}."
        )
    spec = resolve_fp8_spec(hf_config) if hf_config is not None else None
    if spec is None:
        raise ValueError(
            "FP8 weight sync requires a registered model spec for the configured checkpoint "
            f"(registered specs: {', '.join(registered_fp8_spec_names())})."
        )
    if fp8_weight_sync_mode == NVFP4 and not spec.supports_nvfp4:
        raise ValueError(
            f"The NVFP4 wire is not implemented for the {spec.name!r} model spec; "
            "use fp8_weight_sync_mode='blockwise' or 'mxfp8', or leave it unset for a BF16 rollout."
        )
    nvfp4_input_amax = None
    if fp8_weight_sync_mode == NVFP4:
        raw = os.environ.get("SKYRL_NVFP4_INPUT_AMAX")
        nvfp4_input_amax = float(raw) if raw else None
    return SerializedFp8Config(spec=spec, wire_format=fp8_weight_sync_mode, nvfp4_input_amax=nvfp4_input_amax)


def _mxfp8_group_args(dynamic: bool) -> dict:
    """One compressed-tensors group descriptor for MXFP8.

    vLLM selects its MXFP8 schemes -- dense and fused-MoE alike -- from this
    exact predicate (``CompressedTensorsConfig._is_mxfp8``): group strategy,
    symmetric, group size 32, float, 8 bits, and a uint8 scale dtype.
    """

    return {
        "num_bits": 8,
        "type": "float",
        "strategy": "group",
        "group_size": MXFP8_GROUP_SIZE,
        "symmetric": True,
        "scale_dtype": "uint8",
        "dynamic": dynamic,
    }


def get_serialized_fp8_quantization_config(
    weight_block_size: Sequence[int] = (128, 128),
    ignored_layers: Sequence[str] | None = None,
    wire_format: str = BLOCKWISE_FP8,
    nvfp4_static_input: bool = False,
) -> dict:
    """Return vLLM's Hugging Face quantization config for serialized FP8."""

    if wire_format not in WIRE_FORMATS:
        raise ValueError(f"wire_format must be one of {WIRE_FORMATS}, got {wire_format!r}")

    if wire_format == NVFP4:
        # Weight-only NVFP4 (vLLM's NVFP4A16 scheme: no input_activations group, so no activation
        # scales are needed). With ``nvfp4_static_input`` an input_activations group selects W4A4,
        # whose static ``input_global_scale`` the sender ships per module.
        nvfp4_args = {
            "num_bits": 4,
            "type": "float",
            "strategy": "tensor_group",
            "group_size": NVFP4_GROUP_SIZE,
            "symmetric": True,
            "dynamic": False,
        }
        group = {"targets": ["Linear"], "weights": dict(nvfp4_args)}
        if nvfp4_static_input:
            group["input_activations"] = dict(nvfp4_args)
        return {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {"group_0": group},
            "ignore": list(ignored_layers or ()),
        }

    if wire_format == MXFP8:
        # MXFP8 is served through compressed-tensors, which names the excluded
        # modules "ignore" rather than vLLM-fp8's "ignored_layers".
        return {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": _mxfp8_group_args(dynamic=False),
                    "input_activations": _mxfp8_group_args(dynamic=True),
                }
            },
            "ignore": list(ignored_layers or ()),
        }

    block_m, block_n = normalize_block_size(weight_block_size)
    qconfig = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [block_m, block_n],
    }
    if ignored_layers:
        qconfig["ignored_layers"] = list(ignored_layers)
    return qconfig


def scale_name_for_weight(name: str, wire_format: str = BLOCKWISE_FP8) -> str:
    """Return the scale tensor name paired with a quantized weight.

    Blockwise ships an inverse FP32 scale per 128x128 tile; MXFP8 ships an E8M0
    exponent per 32-element group, which compressed-tensors loads as
    ``weight_scale``.
    """

    if not name.endswith(".weight"):
        raise ValueError(f"FP8 scale can only be derived from .weight tensors: {name}")
    suffix = WIRE_SCALE_SUFFIX[MXFP8] if wire_format == MXFP8 else WIRE_SCALE_SUFFIX[BLOCKWISE_FP8]
    return name[: -len(".weight")] + suffix


def iter_batched_moe_expert_fp8_tensors(
    name: str,
    tensor: torch.Tensor,
    config: SerializedFp8Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Convert a batched expert tensor without expanding expert names.

    The old wire format emitted one weight and one scale tensor for every
    expert/projection pair. Keeping the expert dimension intact reduces each
    routed MoE layer from ``6 * num_experts`` tensors to six and lets vLLM use
    its fused 3D loader.
    """
    moe_spec = config.require_spec().moe_expert_spec(name)
    if moe_spec is None:
        raise ValueError(f"Not a batched MoE expert tensor: {name}")
    if tensor.ndim != 3:
        raise ValueError(f"Batched MoE expert tensor must be 3D, got shape={tuple(tensor.shape)}")
    if moe_spec.split_dim is not None:
        num_projections = len(moe_spec.projections)
        if tensor.shape[moe_spec.split_dim] % num_projections != 0:
            raise ValueError(
                f"Batched MoE tensor dim {moe_spec.split_dim} must split evenly across "
                f"{num_projections} projections, got shape={tuple(tensor.shape)}"
            )
        projection_tensors = torch.chunk(tensor, num_projections, dim=moe_spec.split_dim)
    else:
        projection_tensors = (tensor,)

    for proj, projection_tensor in zip(moe_spec.projections, projection_tensors):
        if config.is_mxfp8:
            q_weight, scale = batched_mx_cast_to_fp8(projection_tensor)
        else:
            q_weight, scale = batched_blockwise_cast_to_fp8(
                projection_tensor,
                config.weight_block_size,
                config.power_2_scale,
            )
        weight_name = f"{SKYRL_BATCHED_MOE_FP8_PREFIX}{moe_spec.experts_base}.{proj.hf_name}.weight"
        yield weight_name, q_weight
        yield scale_name_for_weight(weight_name, config.wire_format), scale


def iter_serialized_fp8_tensors(
    name: str,
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    config: SerializedFp8Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield vLLM checkpoint tensors for one Megatron-exported weight."""

    if config.is_nvfp4:
        raise ValueError("The NVFP4 wire quantizes fused groups together; use iter_serialized_nvfp4_tensors")
    spec = config.require_spec()
    if spec.moe_expert_spec(name) is not None:
        yield from iter_batched_moe_expert_fp8_tensors(name, tensor, config)
        return

    if tensor.ndim == 2 and spec.should_quantize(name, tuple(tensor.shape), config.wire_format):
        if config.is_mxfp8:
            q_weight, scale = mx_cast_to_fp8(tensor)
        else:
            q_weight, scale = blockwise_cast_to_fp8(
                tensor,
                config.weight_block_size,
                config.power_2_scale,
            )
        yield name, q_weight
        yield scale_name_for_weight(name, config.wire_format), scale
        return

    yield name, tensor.to(dtype=target_dtype)


def _nvfp4_group_members(spec: ModelFp8Spec, name: str) -> tuple[tuple[str, str], tuple[str, ...]] | None:
    """Return ``(group_key, member_names)`` if ``name`` belongs to a fused-module group.

    ``group_key`` is the layer prefix *and* the group, because several groups (qkv, gate/up, GDN
    in-projections) share one layer prefix and must not be buffered together.
    """

    for group in spec.nvfp4_fusion_groups:
        for suffix in group:
            if name.endswith(suffix):
                prefix = name[: -len(suffix)]
                return (prefix, group[0]), tuple(prefix + member for member in group)
    return None


NVFP4_INPUT_GLOBAL_SCALE_SUFFIX = ".input_global_scale"


def _nvfp4_tensors_for_group(
    members: Sequence[tuple[str, torch.Tensor]],
    input_amax: float | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Quantize one fused module's weights together so they share a global scale.

    vLLM keeps a single global scale per fused module and silently collapses differing per-shard
    values (rescaling the other shards' weights incorrectly). Quantizing the row-concatenation
    gives every shard TE's own per-tensor amax for the fused weight -- exactly what the trainer's
    fused Megatron linear computes -- and the 16-row 2D blocks never straddle a shard because
    every member's row count is a multiple of 16.
    """

    fused = torch.cat([tensor for _, tensor in members], dim=0) if len(members) > 1 else members[0][1]
    packed, scales, global_scale = nvfp4_cast(fused)
    row = 0
    for name, tensor in members:
        rows = tensor.shape[0]
        base = name[: -len(".weight")]
        yield base + NVFP4_PACKED_SUFFIX, packed[row : row + rows].contiguous()
        yield base + WIRE_SCALE_SUFFIX[NVFP4], scales[row : row + rows].contiguous()
        yield base + NVFP4_GLOBAL_SCALE_SUFFIX, global_scale.clone()
        if input_amax is not None:
            # compressed-tensors stores the divisor 448*6/amax, like the weight global scale.
            yield base + NVFP4_INPUT_GLOBAL_SCALE_SUFFIX, torch.full_like(global_scale, (448.0 * 6.0) / input_amax)
        row += rows


def iter_serialized_nvfp4_tensors(
    stream: Iterator[tuple[str, torch.Tensor]],
    config: SerializedFp8Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield vLLM compressed-tensors NVFP4 tensors for a Megatron-exported weight stream.

    Unlike the per-tensor FP8 iterators this consumes the whole stream: members of a fused-module
    group are held until the group is complete, then quantized together. Members normally arrive
    adjacently, so at most one group is buffered; a stream that ends mid-group is an error.
    """

    spec = config.require_spec()
    pending: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for name, tensor in stream:
        if spec.moe_expert_spec(name) is not None:
            raise NotImplementedError(
                "The NVFP4 wire does not support batched MoE expert tensors yet "
                f"({name!r}); use fp8_weight_sync_mode='mxfp8' for MoE models."
            )
        if not (tensor.ndim == 2 and spec.should_quantize(name, tuple(tensor.shape), NVFP4)):
            yield name, tensor
            continue
        grouped = _nvfp4_group_members(spec, name)
        if grouped is None:
            yield from _nvfp4_tensors_for_group([(name, tensor)], config.nvfp4_input_amax)
            continue
        key, member_names = grouped
        bucket = pending.setdefault(key, {})
        bucket[name] = tensor
        if len(bucket) == len(member_names):
            yield from _nvfp4_tensors_for_group([(m, bucket[m]) for m in member_names], config.nvfp4_input_amax)
            del pending[key]
    if pending:
        incomplete = {key: sorted(bucket) for key, bucket in pending.items()}
        raise ValueError(f"NVFP4 weight stream ended with incomplete fused groups: {incomplete}")
