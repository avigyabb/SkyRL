"""Opt-in fast paths around megatron's cuDNN DSA sparse attention.

``SKYRL_DSA_CUDNN_BWD_ALL_ROWS=1``: ``dsa_cudnn_kernels._run_sparse_attention_backward`` defaults to
compacting away query rows with no valid top-k index: it appends a dummy row to q / out / dO / lse /
indices (``torch.cat``), gathers the valid rows (``index_select``) and scatters dq back
(``index_copy_``) -- several copies of ``[tokens, heads, 512]`` tensors per DSA layer.
``FusedSparseAttentionFunc.backward`` never passes ``all_rows_nonempty``, although with
GLM-5.3-Flash's k-pool indexer every query selects at least its own tail pool. Check
``topk_length > 0`` once per call (one host sync) and, when it holds, call the wrapper on the tensors
directly; the cuDNN kernel and its per-row inputs are the same.

``SKYRL_DSA_TOPK_PRESORTED=1``: ``_prepare_attention_topk_indices`` argsorts every row's compacted
top-k by key index. When the indexer already emits ascending indices (the k-pool indexer does under
the same flag, see ``dsa_kpool._presort``), check that in one elementwise pass and skip the sort;
rows that are not ascending fall back to the stock sort.
"""

import os

from loguru import logger

_APPLIED = False


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true")


def apply_dsa_cudnn_fast_paths_patch() -> None:
    global _APPLIED
    if _APPLIED:
        return
    all_rows, presorted = _enabled("SKYRL_DSA_CUDNN_BWD_ALL_ROWS"), _enabled("SKYRL_DSA_TOPK_PRESORTED")
    if not (all_rows or presorted):
        return
    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

    if all_rows:
        run_backward = dsa_cudnn_kernels._run_sparse_attention_backward

        def _run_sparse_attention_backward(**kwargs):
            if not kwargs.get("all_rows_nonempty", False):
                kwargs["all_rows_nonempty"] = bool((kwargs["topk_length"] > 0).all())
            return run_backward(**kwargs)

        dsa_cudnn_kernels._run_sparse_attention_backward = _run_sparse_attention_backward
        logger.info("DSA: cuDNN sparse-attention backward skips row compaction when no row is empty")

    if presorted:
        sort_by_index = dsa_cudnn_kernels._sort_valid_topk_indices_by_index

        def _sort_valid_topk_indices_by_index(topk_indices, topk_length, sk):
            # Compacted input: valid prefix of length topk_length, -1 after it.
            valid = dsa_cudnn_kernels._trailing_positions(topk_indices) < topk_length.unsqueeze(-1)
            ascending = (topk_indices[..., 1:] > topk_indices[..., :-1]) | ~valid[..., 1:]
            if bool(ascending.all()):
                return topk_indices
            return sort_by_index(topk_indices, topk_length, sk)

        dsa_cudnn_kernels._sort_valid_topk_indices_by_index = _sort_valid_topk_indices_by_index
        logger.info("DSA: cuDNN top-k prep skips the sort-by-index when indices are already ascending")
    _APPLIED = True
