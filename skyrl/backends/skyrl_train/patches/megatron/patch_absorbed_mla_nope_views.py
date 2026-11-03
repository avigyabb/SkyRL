"""Opt-in: skip absorbed-MLA's split/concat copies of an empty RoPE part (NoPE MLA).

``AbsorbedMLASelfAttention`` splits ``q`` into ``[qk_head_dim, qk_pos_emb_head_dim]`` and later
concatenates the absorbed query (and the compressed KV) with the RoPE'd part. With
``qk_pos_emb_head_dim == 0`` (GLM-5.3-Flash's NoPE DSA layers) each ``torch.cat`` still copies the
whole ``[tokens, heads, kv_lora_rank]`` query, and the split / cat backward copy its gradient
again -- several GB per DSA layer. ``SKYRL_MLA_NOPE_NO_COPY=1`` gives the ``absorbed_mla`` module a
``torch`` whose ``cat`` returns the only non-empty input and whose ``split`` returns the input itself
next to a detached empty piece; every other attribute is the real ``torch``. Results are the same
tensors; only the copies go away.
"""

import os

import torch
from loguru import logger

_APPLIED = False


class _NoEmptyCopyTorch:
    """``torch`` stand-in whose ``cat`` / ``split`` skip zero-size pieces."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def cat(tensors, dim=0, **kwargs):
        tensors = list(tensors)
        nonempty = [t for t in tensors if t.size(dim) != 0]
        if len(nonempty) == 1 and not kwargs:
            return nonempty[0]
        return torch.cat(tensors, dim=dim, **kwargs)

    @staticmethod
    def split(tensor, split_size_or_sections, dim=0):
        sections = split_size_or_sections
        if isinstance(sections, (list, tuple)) and len(sections) == 2 and 0 in sections:
            full, empty = (0, 1) if sections[1] == 0 else (1, 0)
            if sections[full] == tensor.size(dim):
                shape = list(tensor.shape)
                shape[dim] = 0
                pieces = [None, None]
                pieces[full] = tensor
                pieces[empty] = tensor.new_empty(shape)
                return tuple(pieces)
        return torch.split(tensor, split_size_or_sections, dim=dim)


def apply_absorbed_mla_nope_views_patch() -> None:
    global _APPLIED
    if _APPLIED or os.environ.get("SKYRL_MLA_NOPE_NO_COPY", "0").lower() not in ("1", "true"):
        return
    from megatron.core.transformer.experimental_attention_variant import absorbed_mla

    absorbed_mla.torch = _NoEmptyCopyTorch()
    _APPLIED = True
    logger.info("MLA: absorbed attention skips split/concat copies of an empty RoPE part")
