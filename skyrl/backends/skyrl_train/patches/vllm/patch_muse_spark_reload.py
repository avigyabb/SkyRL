"""Make vllm-project/vllm#61879's Muse Spark model correct under in-place weight reloads.

Every SkyRL weight sync is an in-place, checkpoint-format reload: ``initialize_layerwise_reload``,
one ``model.load_weights`` call per chunk, ``finalize_layerwise_reload``. Three things in the Muse
Spark model don't survive that; both were found by reloading into a running engine and checking
the logprobs (``/shared/hackskyrl/muse_spark/vllm_{reload,chunked_reload}_test.py``).

1. **Fused Q/K/V/O-gate updates are dropped.** ``MuseSpark1xQKVOGateParallelLinear.load_weights``
   calls ``self.weight_loader(param, ...)``, the class method, instead of ``param.weight_loader``.
   Layerwise reload tracks loading through the per-parameter loader it installs, so it sees nothing
   loaded for these layers ("MuseSpark1xQKVOGateParallelLinear: Failed to load weights"), and
   finalize puts the old kernel tensors back: reloading with ``q_proj`` halved left the model
   bit-identical. Attention weights would never update during training. Upstream bypasses
   ``param.weight_loader`` because ``ColumnParallelLinear`` points it at the shard-unaware
   ``weight_loader_v2``; fixed by pointing it at the shard-aware class loader at construction and
   dispatching through it.

2. **The router multiplier is folded only once.** ``MuseSpark1xMoE.fold_router_output_multiplier``
   multiplies the router weight by ``output_multiplier`` after loading and then clears the
   multiplier, so it is a no-op on later loads and a reloaded router runs unscaled: reloading the
   *unchanged* checkpoint moved logprobs by 0.27 on average. (Leaving it unfolded doesn't work:
   the FlashInfer TRT-LLM MoE path routes from the raw logits and ignores the runtime multiplier.)
   The replacement keeps the multiplier and re-folds whenever the router weight's content changed
   since the last fold. Content, because layerwise reload writes through ``param.data.copy_()``,
   which neither moves the storage nor bumps the tensor version. And since ``load_weights``' own
   ``materialize_residual_path_constants`` runs while the layers still hold reload placeholders,
   it is re-run after ``finalize_layerwise_processing`` (which also refreshes the cached
   residual-gate alpha/beta from the new gate weights).

3. **Refreshing the residual alpha/beta moves their storage.** ``materialize_coefficients``
   re-registers them as new tensors, while the CUDA graphs captured at boot (``enforce_eager=False``,
   SkyRL's default) keep reading the old, now freed, addresses. Every tensor compares equal after
   the reload and generations are still garbage (slice: |logprob diff| 4.1 with graphs, 0 eager).
   The wrapper copies the new values into the existing buffers.

Remove once upstream fixes all three. Check: the two reload tests above report no drift for an
unchanged checkpoint, and a halved ``self_attn.q_proj`` changes the logprobs.
"""

import logging
import sys

import torch

logger = logging.getLogger(__name__)

_PATCHED: list[str] = []


# --- 1. fused Q/K/V/O-gate projection -----------------------------------------------------------


def _wrap_qkvo_init(cls) -> None:
    """Point the fused parameter's own ``weight_loader`` at the shard-aware class loader.

    ``ColumnParallelLinear`` sets ``param.weight_loader`` to the base ``weight_loader_v2`` (no shard
    id), which is why upstream bypasses it. Fixing it at construction lets both the first load and
    layerwise reload (which wraps ``param.weight_loader``) use the shard-aware loader.
    """
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.weight.weight_loader = self.weight_loader

    __init__.__wrapped__ = original_init
    cls.__init__ = __init__


def _qkvo_load_weights(self, weights):
    """``MuseSpark1xQKVOGateParallelLinear.load_weights`` through the per-parameter loader."""
    for name, loaded_weight in weights:
        shard_id = getattr(loaded_weight, "shard_id", None)
        if name == "bias" and self.bias is None:
            continue
        param = getattr(self, name)
        param.weight_loader(param, loaded_weight, shard_id)
        yield name


# --- 2. router multiplier fold -------------------------------------------------------------------


def _weight_state(weight: torch.Tensor) -> tuple[int, float, float]:
    w = weight.detach().double()
    return weight.data_ptr(), w.sum().item(), w.square().sum().item()


def _fold_router_output_multiplier(self) -> None:
    """Fold ``output_multiplier`` into the router weight once per load of that weight."""
    multiplier = self.__dict__.setdefault("_skyrl_router_multiplier", self.router_output_multiplier)
    if multiplier is None:
        return
    weight = self.router.weight
    if weight.is_meta:
        # A layerwise-reload placeholder; the post-finalize pass folds the real weight.
        return
    if getattr(self, "_skyrl_folded_state", None) == _weight_state(weight):
        return
    with torch.no_grad():
        weight.mul_(multiplier)
    self._skyrl_folded_state = _weight_state(weight)
    # The folded weight already carries the multiplier; routing must not apply it again.
    self.router_output_multiplier = None
    self.moe_router.router_output_multiplier = None


def _rematerialize_after_layerwise_reload() -> None:
    """Re-run Muse's post-load constants once a layerwise reload has placed the real weights.

    Every weight-transfer engine finalizes through ``finalize_layerwise_processing`` (looked up as
    a module global in ``layerwise``), so wrapping it there covers all of them.
    """
    from vllm.model_executor.model_loader.reload import layerwise
    from vllm.models.muse_spark.common.layers import materialize_residual_path_constants

    original = layerwise.finalize_layerwise_processing

    def finalize_layerwise_processing(model, model_config):
        result = original(model, model_config)
        if type(model).__module__.startswith("vllm.models.muse_spark"):
            materialize_residual_path_constants(model)
        return result

    finalize_layerwise_processing.__wrapped__ = original
    layerwise.finalize_layerwise_processing = finalize_layerwise_processing


def _materialize_coefficients_in_place(original):
    """Refresh the cached residual coefficients without moving their storage.

    ``materialize_coefficients`` re-registers ``alpha`` / ``beta`` as new tensors. At boot that is
    fine, but on a reload the captured CUDA graphs (``enforce_eager=False``) still read the old
    addresses, which the allocator then hands out to other tensors: generations turn to garbage
    while every parameter compares equal.
    """

    def materialize_coefficients(self):
        old = {name: self._buffers.get(name) for name in ("alpha", "beta")}
        result = original(self)
        for name, tensor in old.items():
            new = self._buffers.get(name)
            if tensor is None or new is None or new is tensor:
                continue
            if tensor.shape != new.shape or tensor.dtype != new.dtype or tensor.device != new.device:
                raise RuntimeError(
                    f"Muse Spark residual {name} changed layout on reload: {tensor.shape} -> {new.shape}"
                )
            tensor.copy_(new)
            self._buffers[name] = tensor
        return result

    materialize_coefficients.__wrapped__ = original
    return materialize_coefficients


# --- install -------------------------------------------------------------------------------------


def _install(_package) -> None:
    from vllm.models.muse_spark.common.layers import (
        MuseSpark1xNormPreservingResidualConnection,
        MuseSpark1xQKVOGateParallelLinear,
    )

    residual = MuseSpark1xNormPreservingResidualConnection
    residual.materialize_coefficients = _materialize_coefficients_in_place(residual.materialize_coefficients)
    _PATCHED.append("MuseSpark1xNormPreservingResidualConnection.materialize_coefficients")
    _wrap_qkvo_init(MuseSpark1xQKVOGateParallelLinear)
    MuseSpark1xQKVOGateParallelLinear.load_weights = _qkvo_load_weights
    _PATCHED.append("MuseSpark1xQKVOGateParallelLinear.load_weights")
    for module_name in ("vllm.models.muse_spark.nvidia.model", "vllm.models.muse_spark.amd.model"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for cls in list(vars(module).values()):
            if isinstance(cls, type) and "fold_router_output_multiplier" in cls.__dict__:
                cls.fold_router_output_multiplier = _fold_router_output_multiplier
                _PATCHED.append(f"{cls.__name__}.fold_router_output_multiplier")
    _rematerialize_after_layerwise_reload()
    _PATCHED.append("layerwise.finalize_layerwise_processing")
    logger.info(f"Muse Spark reload fixes installed: {_PATCHED}")


def apply_muse_spark_reload_patch() -> None:
    """Install when vLLM loads the Muse Spark package; a build without it never triggers."""
    from skyrl.backends.skyrl_train.patches.vllm.model_module_hooks import (
        on_model_module_load,
    )

    on_model_module_load("vllm.models.muse_spark", _install)
