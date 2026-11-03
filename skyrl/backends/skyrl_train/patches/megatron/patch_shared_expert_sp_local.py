"""Opt-in: run the MoE shared expert on each rank's sequence shard with gathered weights.

Under tensor + sequence parallelism the shared expert's ``linear_fc1`` all-gathers the
``[tokens, hidden]`` input and ``linear_fc2`` reduce-scatters the output, forward and backward,
although the routed experts next to it (expert parallel) work on the sequence shard directly.
``SKYRL_SHARED_EXPERT_SP_LOCAL=1`` instead applies the whole expert to the rank's own tokens: the
frozen base weights are all-gathered once and cached, and the (tiny, TP-sharded) LoRA weights are
all-gathered each forward with a reduce-scatter of their gradient -- every rank's partial
gradient from its own tokens summed into each shard, the gradient the TP path computes. Same
FLOPs; the per-layer sequence collectives become weight-sized ones. The base GEMMs run in the
activation dtype (no FP8).

Applies to a shared expert whose ``linear_fc1`` / ``linear_fc2`` are plain TP-sharded linears,
optionally wrapped in a LoRA adapter with identity activation and no dropout; anything else runs
the stock forward.
"""

import os

import torch
import torch.nn.functional as F
from loguru import logger

_APPLIED = False


class _GatherShardedWeight(torch.autograd.Function):
    """All-gather a TP-sharded weight along ``dim``; reduce-scatter its gradient back to the shard."""

    @staticmethod
    def forward(ctx, weight, dim, group):
        ctx.dim, ctx.group = dim, group
        parts = [torch.empty_like(weight) for _ in range(group.size())]
        torch.distributed.all_gather(parts, weight.contiguous(), group=group)
        return torch.cat(parts, dim=dim)

    @staticmethod
    def backward(ctx, grad):
        chunks = [c.contiguous() for c in grad.chunk(ctx.group.size(), dim=ctx.dim)]
        out = torch.empty_like(chunks[0])
        torch.distributed.reduce_scatter(out, chunks, group=ctx.group)
        return out, None, None


def _gather_frozen(weight, dim, group):
    parts = [torch.empty_like(weight) for _ in range(group.size())]
    torch.distributed.all_gather(parts, weight.detach().contiguous(), group=group)
    return torch.cat(parts, dim=dim)


def _glu_row_order(rows_per_rank: int, tp: int, device) -> torch.Tensor:
    """Gathered ``[g0; u0; g1; u1; ...]`` rows -> ``[g0; g1; ...; u0; u1; ...]``."""
    half = rows_per_rank // 2
    rank_base = torch.arange(tp, device=device).unsqueeze(1) * rows_per_rank
    gate = (rank_base + torch.arange(half, device=device)).reshape(-1)
    return torch.cat([gate, gate + half])


class _LocalLinear(torch.nn.Module):
    """Sequence-local stand-in for a (LoRA-wrapped) TP linear of the shared expert."""

    def __init__(self, owner, name, glu_rows):
        super().__init__()
        self._owner = (owner,)  # not registered: the real module stays the checkpointed one
        self._name = name
        self._glu_rows = glu_rows

    def forward(self, x):
        owner = self._owner[0]
        module = owner._modules[self._name]
        base = getattr(module, "to_wrap", module)
        adapter = getattr(module, "adapter", None)
        group = owner._skyrl_tp_group
        column = self._name == "linear_fc1"
        # Column-parallel base: [out/TP, in] sharded on rows. Row-parallel: [out, in/TP] on columns.
        key = (base.weight.data_ptr(), base.weight._version)
        cache = owner._skyrl_weight_cache.get(self._name)
        if cache is None or cache[0] != key:
            full = _gather_frozen(base.weight, 0 if column else 1, group)
            if self._glu_rows is not None:
                full = full.index_select(0, self._glu_rows)
            owner._skyrl_weight_cache[self._name] = cache = (key, full.to(x.dtype))
        out = F.linear(x, cache[1])
        if adapter is not None:
            w_in = adapter.linear_in.weight
            w_out = adapter.linear_out.weight
            # fc1: linear_in [r/TP, in] and linear_out [out/TP, r] row-sharded.
            # fc2: linear_in [r, in/TP] column-sharded, linear_out [out/TP, r] row-sharded.
            w_in = _GatherShardedWeight.apply(w_in, 0 if column else 1, group)
            w_out = _GatherShardedWeight.apply(w_out, 0, group)
            if self._glu_rows is not None:
                w_out = w_out.index_select(0, self._glu_rows)
            h = adapter.activation(F.linear(x, w_in)) * (adapter.alpha / adapter.dim)
            out = out + F.linear(h, w_out)
        return out, None


def apply_shared_expert_sp_local_patch() -> None:
    global _APPLIED
    if _APPLIED or os.environ.get("SKYRL_SHARED_EXPERT_SP_LOCAL", "0").lower() not in ("1", "true"):
        return
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

    stock_forward = SharedExpertMLP.forward

    def eligible(self) -> bool:
        config = self.config
        tp_group = getattr(self, "tp_group", None)
        if tp_group is None or tp_group.size() == 1 or not config.sequence_parallel:
            return False
        if config.add_bias_linear or getattr(config, "moe_shared_expert_overlap", False):
            return False
        for name in ("linear_fc1", "linear_fc2"):
            module = self._modules[name]
            base = getattr(module, "to_wrap", module)
            if getattr(base, "weight", None) is None or base.weight.requires_grad:
                return False
            adapter = getattr(module, "adapter", None)
            if adapter is not None and (
                adapter.is_expert
                or adapter.use_a2a
                or adapter.replicate_adapter
                or adapter.dropout_position != "pre"
                or not isinstance(adapter.dropout, torch.nn.Identity)
                or not isinstance(adapter.activation, torch.nn.Identity)
            ):
                return False
        return True

    def forward(self, hidden_states):
        if not hasattr(self, "_skyrl_sp_local"):
            self._skyrl_sp_local = eligible(self)
            if self._skyrl_sp_local:
                self._skyrl_tp_group = self.tp_group
                self._skyrl_weight_cache = {}
                tp = self.tp_group.size()
                fc1 = getattr(self.linear_fc1, "to_wrap", self.linear_fc1)
                rows = fc1.weight.shape[0]
                glu = _glu_row_order(rows, tp, fc1.weight.device) if self.config.gated_linear_unit else None
                self._skyrl_local = {
                    "linear_fc1": _LocalLinear(self, "linear_fc1", glu),
                    "linear_fc2": _LocalLinear(self, "linear_fc2", None),
                }
        if not self._skyrl_sp_local:
            return stock_forward(self, hidden_states)
        # Instance attributes shadow the registered submodules for MLP.forward only.
        self.__dict__.update(self._skyrl_local)
        try:
            return stock_forward(self, hidden_states)
        finally:
            del self.__dict__["linear_fc1"], self.__dict__["linear_fc2"]

    SharedExpertMLP.forward = forward
    _APPLIED = True
    logger.info("MoE: shared expert runs on the sequence shard with gathered weights")
