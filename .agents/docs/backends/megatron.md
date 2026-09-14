# Megatron Backend

## Megatron-Bridge

SkyRL uses Megatron-Bridge for HF-to-Megatron model conversion. Installed from git with a pinned rev in `[tool.uv.sources]`.

## Key abstractions
- `MegatronConfig` in `skyrl/train/config.py`
- `MegatronWorker` in `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py`.
- Custom bridges in `skyrl/backends/skyrl_train/workers/megatron/model_bridges.py` (e.g., `GLM47FlashBridge`).

## Parallelism Strategies

For picking TP/PP/EP/CP/SP sizes, invoke the `parallelism-strategies` skill.

Key strategies:
- **Tensor Parallelism (TP)**: Splits layers across GPUs within an NVLink domain. Use TP ≤ GPUs per node. Applicable for non-MoE linear layers.
- **Pipeline Parallelism (PP)**: Splits model layers across nodes. Use for cross-node scaling.
- **Data Parallelism (DP)**: Implicit — `world_size / (TP * PP)`. Each DP rank processes different data.
- **Sequence Parallelism (SP)**: Requires TP > 1. Splits along sequence dimension for LayerNorm/Dropout.
- **Context Parallelism (CP)**: For sequences > 8K tokens. Splits attention computation across GPUs.
- **Expert Parallelism (EP)**: For MoE models. Distributes experts across GPUs.
- **Expert Tensor Parallelism (ETP)**: For MoE models. Tensor parallelism for the expert layers.

Note: Sequence parallelism is auto-enabled when `tensor_model_parallel_size > 1` — there is no separate config field for it.

## Test Requirements

Megatron GPU tests need: `NVTE_FLASH_ATTN=0`

## HF import cache (`trainer.policy.megatron_config.hf_import_cache_dir`)

Megatron-Bridge's `load_weights_hf_to_megatron` has every rank read the full HF tensors and convert them on CPU
at every startup. With `hf_import_cache_dir` set (same field on `trainer.ref.megatron_config`), the first run
imports from HF as usual and then saves the built model's `sharded_state_dict()` as a Megatron `torch_dist`
checkpoint under `<dir>/<model-slug>/<digest>/`; later runs build the provider with `load_weights=False` and
`perform_initialization=False` and `dist_checkpointing.load` the cache instead. Implementation:
`workers/megatron/hf_import_cache.py`, wired from `MegatronWorker._resolve_hf_import_cache` /
`_apply_hf_import_cache` in the policy and ref `init_model`.

- The digest covers the checkpoint fingerprint (local dirs: weight/index/config file names, sizes, mtimes; Hub
  ids: the id only), the megatron-core / megatron-bridge / transformers / torch versions, `bf16`,
  `language_model_only`, `enable_mtp`, and both `model_config_kwargs` and `transformer_config_kwargs`. The
  `torch_dist` format is parallelism-agnostic, so a cache written at one TP/PP layout loads into another.
- Hit/miss is decided on rank 0 and broadcast so all ranks agree; writers save into `<path>.tmp-<id>` and rank 0
  renames it into place after stamping `.skyrl_hf_import_complete`, so concurrent jobs never read a partial entry.
- Not applied (falls back to the HF import) with LoRA (adapters wrap params before the build finishes) or fake-INT4
  QAT (different source checkpoint). Not for cloud paths.
- The worker logs `megatron model build ... took` with the import / cache load / cache save split on rank 0.
