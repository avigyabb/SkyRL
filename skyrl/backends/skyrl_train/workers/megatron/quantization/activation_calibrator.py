"""Per-layer activation amax, measured by the trainer, for W4A4 (NVFP4) rollouts.

vLLM's W4A4 NVFP4 scheme quantizes each linear's input with a *static* per-tensor ``input_global_scale``
(= 448 * 6 / amax). One shared amax is wrong for every layer at once: a value that fits the attention
inputs clips the MLP down-projection (whose inputs carry massive outliers), and a value that fits the
down-projection wastes block-scale range everywhere else.

The trainer already runs the policy forward over exactly the tokens the rollout produced, so it can
measure each linear's real input range for free. This module hooks the Transformer Engine linears,
keeps a running max of ``|input|`` per module, reduces it across ranks, and maps each Megatron module to
the Hugging Face weight names the weight sync ships, so the sender can emit one ``input_global_scale``
per layer.

A TE ``LayerNormLinear`` (Megatron's fused ``linear_qkv`` / ``linear_fc1``) quantizes the *normalized*
activation, not the module input the hook sees, so the norm is replayed here to measure the right tensor.
"""

import json
import os
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from loguru import logger

_TE_LINEAR_NAMES = {"Linear", "LayerNormLinear"}


def _is_te_linear(module: torch.nn.Module) -> bool:
    return any(
        cls.__name__ in _TE_LINEAR_NAMES and cls.__module__.startswith("transformer_engine")
        for cls in type(module).__mro__
    )


def _dump(tag: str, payload: dict) -> None:
    """Write a diagnostic to ``$SKYRL_NVFP4_CALIB_DUMP`` (a directory), rank 0 only; worker stdout is not always captured."""
    out_dir = os.environ.get("SKYRL_NVFP4_CALIB_DUMP")
    if not out_dir or (dist.is_initialized() and dist.get_rank() != 0):
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{tag}_{int(time.time())}.json"), "w") as f:
        json.dump(payload, f, indent=1)


def _strip_wrappers(name: str) -> str:
    while name.startswith("module."):
        name = name[len("module.") :]
    return name


@torch.no_grad()
def _quantizer_input(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The tensor the module's GEMM actually quantizes (post-norm for a fused LayerNormLinear)."""
    weight = getattr(module, "layer_norm_weight", None)
    if weight is None:
        return x
    xf = x.float()
    gamma = weight.float() + (1.0 if getattr(module, "zero_centered_gamma", False) else 0.0)
    if getattr(module, "normalization", "LayerNorm") == "RMSNorm":
        return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + module.eps) * gamma
    mean = xf.mean(-1, keepdim=True)
    var = xf.var(-1, unbiased=False, keepdim=True)
    y = (xf - mean) * torch.rsqrt(var + module.eps) * gamma
    bias = getattr(module, "layer_norm_bias", None)
    return y + bias.float() if bias is not None else y


class ActivationAmaxCalibrator:
    """Running per-module input amax over the trainer's forward passes.

    Args:
        decay: Across syncs, ``amax = max(this_window, decay * previous)``: it follows the true max up
            immediately and forgets a stale peak slowly. ``0`` uses only the latest window.
        margin: Multiplier on the reported amax (headroom for tokens not yet seen).
    """

    def __init__(self, bridge, actor_module: List[torch.nn.Module], decay: float = 0.9, margin: float = 1.0):
        self.decay, self.margin = decay, margin
        self.enabled = True
        self._modules: Dict[str, torch.nn.Module] = {}
        for chunk in actor_module:
            for name, module in chunk.named_modules():
                if _is_te_linear(module):
                    self._modules[_strip_wrappers(name)] = module
        self._names = sorted(self._modules)
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self._window = torch.zeros(len(self._names), dtype=torch.float32, device=device)
        self._smoothed = torch.zeros_like(self._window)
        self._index = {name: i for i, name in enumerate(self._names)}
        self._hf_names = self._map_to_hf(bridge, actor_module)
        self._handles = [
            module.register_forward_pre_hook(self._make_hook(self._index[name]))
            for name, module in self._modules.items()
        ]
        logger.info(
            f"NVFP4 activation calibrator: hooked {len(self._names)} TE linears, "
            f"mapped {len(self._hf_names)} to HF weight names; unmapped: "
            f"{[n for n in self._names if n not in self._hf_names][:6]}"
        )
        _dump("init", {"hooked": self._names, "mapped": self._hf_names})

    def _map_to_hf(self, bridge, actor_module) -> Dict[str, List[str]]:
        """Megatron module name -> HF weight names its GEMM serves (fused modules map to several)."""
        mapping: Dict[str, List[str]] = {}
        for task in bridge.get_conversion_tasks(actor_module):
            gname = _strip_wrappers(task.global_param_name)
            if not gname.endswith(".weight") or gname[: -len(".weight")] not in self._index:
                continue
            hf = task.mapping.hf_param
            hf_names = list(hf.values()) if isinstance(hf, dict) else [hf]
            mapping.setdefault(gname[: -len(".weight")], []).extend(hf_names)
        return mapping

    def _make_hook(self, idx: int):
        @torch.no_grad()
        def hook(module, args):
            if not self.enabled or not args or not torch.is_tensor(args[0]):
                return
            amax = _quantizer_input(module, args[0].detach()).abs().amax().float()
            torch.maximum(self._window[idx], amax, out=self._window[idx])

        return hook

    @torch.no_grad()
    def collect(self) -> Dict[str, float]:
        """Reduce the window across ranks, fold it into the smoothed amax and reset it.

        Collective: every rank that exports weights must call this at the same point.
        """
        window = self._window.clone()
        if dist.is_initialized():
            dist.all_reduce(window, op=dist.ReduceOp.MAX)
        self._window.zero_()
        self._smoothed = torch.maximum(window, self._smoothed * self.decay)
        out: Dict[str, float] = {}
        smoothed = (self._smoothed * self.margin).tolist()
        for name, hf_names in self._hf_names.items():
            value = smoothed[self._index[name]]
            if value > 0:
                for hf_name in hf_names:
                    out[hf_name] = value
        _dump("collect", {"amax_by_hf_weight": out})
        if out:
            values = sorted(out.values())
            logger.info(
                f"NVFP4 input amax per layer: n={len(values)} min={values[0]:.3g} "
                f"median={values[len(values) // 2]:.3g} max={values[-1]:.3g}"
            )
        return out

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def make_refresh(calibrator: Optional[ActivationAmaxCalibrator]):
    """Callable the weight source runs before each export to refresh the sender's per-layer amax."""
    if calibrator is None:
        return None
    return calibrator.collect
