"""Opt-in: LoRA adapters on row-parallel linears without full-width sequence-parallel collectives.

For a row-parallel base linear under sequence parallelism (attention ``o_proj`` / ``linear_proj``,
MLP ``linear_fc2``), Megatron-Bridge's ``ParallelLinearAdapter`` computes
``x [s, in/TP] -> linear_in (all-reduce) -> [s, r] -> linear_out [s, out/TP] -> all-gather over the
hidden dim -> [s, out] -> scatter to the sequence shard [s/TP, out]``: one full-width all-gather
forward and one backward (the scatter's), each the size of the layer output.

``SKYRL_LORA_ROW_ADAPTER_LOCAL=1`` instead takes the rank's sequence rows of the rank-``r``
activation and multiplies them by the all-gathered (``out x r``, tiny) ``linear_out`` weight:
``scatter([s, r]) @ gather(W_out).T -> [s/TP, out]``. The weight gather's backward reduce-scatter
sums every rank's row contributions into each rank's weight shard, so the result and both
gradients are the same as the stock path up to summation order.

Applies only to plain TP-sharded row adapters (not expert, all-to-all, replicated or shared-expert
external-reduce adapters); everything else runs the stock forward.
"""

import os

from loguru import logger

_APPLIED = False


def apply_lora_row_adapter_local_patch() -> None:
    global _APPLIED
    if _APPLIED or os.environ.get("SKYRL_LORA_ROW_ADAPTER_LOCAL", "0").lower() not in ("1", "true"):
        return
    import torch
    from megatron.bridge.peft import utils
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.tensor_parallel.mappings import (
        gather_from_sequence_parallel_region,
        scatter_to_sequence_parallel_region,
    )

    adapter_cls = utils.ParallelLinearAdapter
    previous_forward = adapter_cls.forward

    def eligible(self) -> bool:
        return (
            self.input_is_parallel
            and not self.disable_sequence_parallel_comm
            and not self.is_expert
            and not self.use_a2a
            and not self.replicate_adapter
            and getattr(self, "_skyrl_external_tp_reduce_scale", 1.0) == 1.0
            and isinstance(self.linear_out, ColumnParallelLinear)
            and self.linear_out.gather_output
            and isinstance(self.dropout, torch.nn.Identity)
            and self.tp_group is not None
            and self.tp_group.size() > 1
        )

    def forward(self, x, *args, **kwargs):
        if not eligible(self):
            return previous_forward(self, x, *args, **kwargs)
        x, _ = self.linear_in(x)
        x = self.activation(x) * (self.alpha / self.dim)
        x = scatter_to_sequence_parallel_region(x, group=self.tp_group)
        weight = gather_from_sequence_parallel_region(
            self.linear_out.weight, tensor_parallel_output_grad=True, group=self.tp_group
        )
        return torch.nn.functional.linear(x, weight)

    adapter_cls.forward = forward
    _APPLIED = True
    logger.info("LoRA: row-parallel adapters compute their sequence shard locally (no full-width gathers)")
