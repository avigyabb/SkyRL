"""Register the Muse Spark Open (``model_type="muse_spark"``) config with transformers.

``meta-models/Muse-Spark-Open-816B-A42B`` landed in transformers main
(huggingface/transformers#49644, 5.21.0.dev0), but SkyRL pins ``transformers<=5.16.1`` and
vLLM's Muse Spark branch (vllm-project/vllm#61879) requires ``<5.20``. Without a registered
config, every ``AutoConfig.from_pretrained`` on the checkpoint fails before Megatron-Bridge or
vLLM sees it.

These configs are pass-through: ``config.json`` carries every field (explicit per-layer
``layer_types`` / ``layer_rope_theta`` / ``mlp_layer_types`` schedules), so nothing is
defaulted here, only validated. The shape mirrors vLLM's
``vllm/transformers_utils/configs/muse_spark.py``.

``skyrl.backends.skyrl_train`` calls :func:`register_muse_spark_config` on package import.
Delete this module (and that call) once the transformers pin includes
``transformers.models.muse_spark``; the registration is already a no-op then.
"""

from copy import deepcopy
from typing import Any

from transformers import PretrainedConfig


class MuseSparkTextConfig(PretrainedConfig):
    model_type = "muse_spark_text"
    base_config_key = "text_config"
    has_no_defaults_at_init = True

    def __init__(self, use_cache: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.use_cache = use_cache
        if self.hidden_act != "silu" or self.attention_bias:
            raise ValueError("Muse Spark requires SiLU and bias-free attention")
        for field, choices in (
            ("layer_types", {"full_attention", "sliding_attention"}),
            ("mlp_layer_types", {"sparse", "dense"}),
        ):
            values = getattr(self, field)
            if len(values) != self.num_hidden_layers or not set(values) <= choices:
                raise ValueError(f"Invalid Muse Spark {field}: {values}")
        if len(self.layer_rope_theta) != self.num_hidden_layers:
            raise ValueError("Muse Spark requires one RoPE theta per layer")


class MuseSparkVisionConfig(PretrainedConfig):
    model_type = "muse_spark_vision"
    base_config_key = "vision_config"
    has_no_defaults_at_init = True


class MuseSparkAudioConfig(PretrainedConfig):
    model_type = "muse_spark_audio"
    base_config_key = "audio_config"
    has_no_defaults_at_init = True


class MuseSparkConfig(PretrainedConfig):
    model_type = "muse_spark"
    has_no_defaults_at_init = True
    sub_configs = {
        "text_config": MuseSparkTextConfig,
        "vision_config": MuseSparkVisionConfig,
        "audio_config": MuseSparkAudioConfig,
    }

    def __init__(
        self,
        text_config: dict[str, Any] | MuseSparkTextConfig | None = None,
        vision_config: dict[str, Any] | MuseSparkVisionConfig | None = None,
        audio_config: dict[str, Any] | MuseSparkAudioConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        for name, value in (
            ("text_config", text_config),
            ("vision_config", vision_config),
            ("audio_config", audio_config),
        ):
            config_cls = self.sub_configs[name]
            if isinstance(value, dict):
                value = config_cls(**deepcopy(value))
            if not isinstance(value, config_cls):
                raise ValueError(f"Muse Spark requires {name}")
            setattr(self, name, value)


_CONFIGS = (MuseSparkConfig, MuseSparkTextConfig, MuseSparkVisionConfig, MuseSparkAudioConfig)


def register_muse_spark_config() -> None:
    """Register the configs unless transformers already ships them. Idempotent."""
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    for config_cls in _CONFIGS:
        if config_cls.model_type not in CONFIG_MAPPING:
            AutoConfig.register(config_cls.model_type, config_cls)
