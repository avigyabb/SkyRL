"""Let the ``sonic`` load format keep its ``model_loader_extra_config`` keys on vLLM >= 0.28.

``sonicloader``'s ``SonicLoader`` subclasses vLLM's ``DefaultModelLoader`` and reads its
``mirror`` / ``capture`` parameters from ``load_config.model_loader_extra_config``. vLLM 0.28
added a key allow-list to ``DefaultModelLoader.__init__`` (``enable_multithread_load``,
``num_threads``, ``enable_weights_track``) that the sonic keys fail, so every engine worker
dies with ``Unexpected extra config keys for load format sonic`` before loading weights, and
its HF fallback (``DefaultModelLoader.load_weights``) rejects ``load_format="sonic"`` with
``Unknown load_format``. Validate with the sonic keys held back, then hand the loader its full
extra config under ``load_format="auto"``. No-op for
every other load format. Imported (and thereby applied) by ``NewInferenceWorkerWrap`` so it
runs in each vLLM worker process before ``get_model_loader``.
"""

import dataclasses

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

SONIC_LOAD_FORMAT = "sonic"
SONIC_EXTRA_CONFIG_KEYS = frozenset({"mirror", "capture", "stream"})

_original_init = DefaultModelLoader.__init__


def _init_with_sonic_keys(self, load_config, *args, **kwargs):
    extra = load_config.model_loader_extra_config
    if (
        getattr(load_config, "load_format", None) == SONIC_LOAD_FORMAT
        and isinstance(extra, dict)
        and SONIC_EXTRA_CONFIG_KEYS & extra.keys()
    ):
        sanitized = dataclasses.replace(
            load_config,
            model_loader_extra_config={k: v for k, v in extra.items() if k not in SONIC_EXTRA_CONFIG_KEYS},
        )
        _original_init(self, sanitized, *args, **kwargs)
        # The loader keeps sonic's keys, but its own ``load_format`` must be one the default HF
        # path understands: ``DefaultModelLoader.load_weights`` (sonic's fallback when no weights
        # are published) dispatches weight discovery on it and rejects ``sonic``. The registry
        # dispatch on the ``sonic`` name has already happened by the time this runs.
        self.load_config = dataclasses.replace(load_config, load_format="auto")
        return
    _original_init(self, load_config, *args, **kwargs)


if not getattr(DefaultModelLoader, "_skyrl_sonic_extra_config_patched", False):
    DefaultModelLoader.__init__ = _init_with_sonic_keys
    DefaultModelLoader._skyrl_sonic_extra_config_patched = True


def _install_no_stream_guard() -> None:
    """Honor ``model_loader_extra_config["stream"] is False`` in ``SonicLoader.load_weights``.

    sonic streams weights whenever a weights manifest exists under the mirror, and its
    ``push_artifacts`` stamps that manifest even when it uploaded no shards (a cache-only
    publish), so the boot after a cache-only publish would try to stream from an empty prefix
    and fail. With ``stream`` false (SkyRL's ``sonic_stream_weights=false``) restore the compile
    cache and load from HF, whatever the mirror holds.
    """
    try:
        import sonic.adapters.vllm as sonic_vllm
    except ImportError:
        return
    loader_cls = sonic_vllm.SonicLoader
    if getattr(loader_cls, "_skyrl_no_stream_patched", False):
        return
    _original_load_weights = loader_cls.load_weights

    def load_weights(self, model, model_config):
        extra = dict(self.load_config.model_loader_extra_config or {})
        if extra.get("mirror") and extra.get("stream", True) is False:
            from vllm.config import get_current_vllm_config
            from vllm.distributed import get_tensor_model_parallel_rank

            rank = get_tensor_model_parallel_rank()
            _weights, cache = sonic_vllm.artifacts_for(get_current_vllm_config(), extra["mirror"])
            sonic_vllm._maybe_pull_cache(cache, rank)
            DefaultModelLoader.load_weights(self, model, model_config)
            if extra.get("capture", True):
                self._capture(model, rank)
            return
        return _original_load_weights(self, model, model_config)

    loader_cls.load_weights = load_weights
    loader_cls._skyrl_no_stream_patched = True


_install_no_stream_guard()
