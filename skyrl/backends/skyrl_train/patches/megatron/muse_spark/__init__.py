"""Muse Spark Open (HF ``muse_spark``) for the Megatron backend.

Upstream counterpart: NVIDIA-NeMo/Megatron-Bridge#6483 (``megatron/bridge/models/muse_spark_open``),
which landed after the pinned Megatron-Bridge rev. This package carries its language-model half
on the provider API:

- ``blocks``: the Hybrid attention / latent-MoE layer specs (residual gates, per-head output norm,
  fp32 sigmoid router, per-expert post norm).
- ``provider``: ``MuseSparkModelProvider`` plus the embedding-norm and logit-softcap mixins.
- ``bridge``: ``MuseSparkBridge``, the HF <-> Megatron mapping for
  ``MuseSparkForConditionalGeneration``. Importing it registers the bridge.

Removal plan: ``skyrl/backends/skyrl_train/patches/megatron/README.md``.
"""
