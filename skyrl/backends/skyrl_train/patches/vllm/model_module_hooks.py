"""Run a patch when vLLM's model registry first loads a model's module.

SkyRL's per-model vLLM patches are installed from the worker extension, which every vLLM worker
imports before the model exists. Importing a model package there to patch it has side effects for
every *other* model: each package registers its own CustomOps on import, and two packages
registering the same op name (e.g. ``fused_rms_norm_gated``, from both the KDA models and Muse
Spark) assert. Deferring to the registry's lazy class load patches only the model being served.
"""

import sys
from typing import Callable

_HOOKS: dict[str, list[Callable]] = {}
_INSTALLED = False


def on_model_module_load(module_name: str, callback: Callable) -> None:
    """Call ``callback(module)`` once ``module_name`` is loaded: now if already imported, else on load."""
    if module_name in sys.modules:
        callback(sys.modules[module_name])
        return
    _HOOKS.setdefault(module_name, []).append(callback)
    _install()


def _install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.model_executor.models.registry import _LazyRegisteredModel

    original = _LazyRegisteredModel.load_model_cls

    def load_model_cls(self):
        model_cls = original(self)
        for callback in _HOOKS.pop(self.module_name, []):
            callback(sys.modules[self.module_name])
        return model_cls

    load_model_cls.__wrapped__ = original
    _LazyRegisteredModel.load_model_cls = load_model_cls
    _INSTALLED = True
