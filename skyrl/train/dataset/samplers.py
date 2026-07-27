"""Stateful samplers for :class:`~skyrl.train.sft_trainer.SFTTrainer`.

These samplers plug into ``torchdata.stateful_dataloader.StatefulDataLoader``
so the sampling position is captured in the dataloader's ``state_dict`` and
restored on resume. A sampler exposes ``state_dict``/``load_state_dict``; the
``StatefulDataLoader`` fast-forwards to the saved position when iteration
resumes after a checkpoint load.

Core ships :class:`StatefulSequentialSampler` (backing the ``sampler="sequential"``
config option) and :class:`DataMixingSampler` (weighted multi-source mixing, used
by default when multiple ``train_datasets`` are configured with ``sampler="random"``).
The ``sampler="custom"`` path loads a user-supplied sampler from
``SFTConfig.sampler_class_path`` via :func:`import_sampler_class`, instantiating
it as ``ClassName(tokenized, **sampler_kwargs)``. See
``examples/train/sft/curriculum_sampler.py`` for a reference custom sampler.
"""

from __future__ import annotations

import importlib
from typing import Iterator, List, Optional, Sequence

import numpy as np
import torch
from loguru import logger

__all__ = [
    "DataMixingSampler",
    "ShardBlockShuffleSampler",
    "StatefulSequentialSampler",
    "import_sampler_class",
]


def import_sampler_class(class_path: str) -> type:
    """Import a sampler class from a ``module.path.ClassName`` string.

    Args:
        class_path: Fully-qualified import path, e.g.
            ``"examples.train.sft.curriculum_sampler.CurriculumLearningSampler"``.
            The import runs inside a Ray task (which does not inherit the
            driver's ``PYTHONPATH``), so the module must be importable from the
            worker's ``sys.path`` -- e.g. a dotted path from the repo root when
            launching from it.

    Returns:
        The resolved class object.

    Raises:
        ValueError: If ``class_path`` is not a dotted ``module.ClassName`` path.
    """
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"Invalid sampler_class_path '{class_path}'; expected a dotted path like " f"'my_module.MySampler'."
        )
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class StatefulSequentialSampler(torch.utils.data.Sampler[int]):
    """Yield indices ``0..len-1`` in order, resumable across checkpoints.

    Unlike ``torch.utils.data.SequentialSampler``, this tracks an internal
    ``position`` cursor and exposes ``state_dict``/``load_state_dict`` so that
    a ``StatefulDataLoader`` can resume mid-epoch from the exact next index.
    The cursor resets to ``0`` once an epoch is exhausted, so a fresh iterator
    starts over from the beginning.
    """

    def __init__(self, data_source: Sequence):
        self.data_source = data_source
        self.position = 0

    def __iter__(self) -> Iterator[int]:
        while self.position < len(self.data_source):
            idx = self.position
            self.position += 1
            yield idx
        # Reset so the next epoch (a fresh ``iter()``) starts from the top.
        self.position = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def state_dict(self) -> dict:
        return {"position": self.position}

    def load_state_dict(self, state: dict) -> None:
        self.position = state["position"]


class ShardBlockShuffleSampler(torch.utils.data.Sampler[int]):
    """Storage-aligned shuffle for row-group-backed cloud datasets (~1x reads).

    A fully random epoch over a parquet store fetches each row group roughly
    once *per row it contains* (~``row_group_size``x read amplification),
    because a group's rows are sampled at widely separated points and the row
    group cache has long been evicted between them. This sampler instead
    shuffles at three nested levels -- shard order, row-group order within each
    shard, and row order within a sliding window of ``shuffle_window`` row
    groups -- so consecutive indices exhaust a bounded working set of groups
    before moving on. Each row group is fetched once per epoch (per dataloader
    worker process) and every row is still visited exactly once.

    Randomness/locality trade-off: a batch draws from ~``shuffle_window``
    row groups inside one shard neighborhood rather than from the whole
    dataset. Larger windows improve batch diversity and require a larger
    ``row_group_cache_size`` on the dataset (keep ``cache >= window + a few``
    so in-flight batches spanning a window boundary stay resident).

    ``data_source`` must expose ``row_group_blocks()`` (see
    :class:`~skyrl.train.dataset.pretokenized.RowGroupPretokenizedDataset`).
    Plugs into the trainer via ``sampler="custom"`` with
    ``sampler_class_path="skyrl.train.dataset.samplers.ShardBlockShuffleSampler"``.

    Epoch plans are drawn from a persistent seeded generator with the same
    resume contract as :class:`DataMixingSampler`: ``state_dict`` snapshots the
    pre-draw generator state plus the cursor, so a mid-epoch resume re-draws
    the identical plan and later epochs match the uninterrupted run.

    Args:
        data_source: A dataset exposing ``row_group_blocks()``.
        shuffle_window: Number of consecutive row-group blocks whose union is
            row-shuffled together.
        seed: Seed for the persistent plan generator.
    """

    def __init__(self, data_source: Sequence, shuffle_window: int = 16, seed: int = 0):
        if not hasattr(data_source, "row_group_blocks"):
            raise TypeError(
                "ShardBlockShuffleSampler requires a dataset exposing row_group_blocks() "
                f"(e.g. RowGroupPretokenizedDataset); got {type(data_source).__name__}."
            )
        if shuffle_window < 1:
            raise ValueError(f"shuffle_window must be >= 1, got {shuffle_window}.")
        self.data_source = data_source
        self.shuffle_window = shuffle_window
        self.position = 0

        starts, lengths, shards = data_source.row_group_blocks()
        self._block_starts = starts
        self._block_lengths = lengths
        # Blocks arrive in (shard, row group) file order, so each shard's
        # blocks form one contiguous span: [span_bounds[s], span_bounds[s+1]).
        self._shard_span_bounds = np.concatenate([[0], np.cumsum(np.bincount(shards))])

        self._generator = torch.Generator()
        self._generator.manual_seed(seed)
        self._plan: Optional[np.ndarray] = None
        self._plan_gen_state: Optional[torch.Tensor] = None

    def _ensure_plan(self) -> None:
        if self._plan is not None:
            return
        self._plan_gen_state = self._generator.get_state()
        epoch_seed = int(torch.randint(0, 2**62, (1,), generator=self._generator).item())
        rng = np.random.default_rng(epoch_seed)

        num_shards = len(self._shard_span_bounds) - 1
        block_order_parts = []
        for shard in rng.permutation(num_shards):
            lo, hi = int(self._shard_span_bounds[shard]), int(self._shard_span_bounds[shard + 1])
            block_order_parts.append(lo + rng.permutation(hi - lo))
        block_order = np.concatenate(block_order_parts)

        plan = np.concatenate(
            [self._block_starts[b] + np.arange(self._block_lengths[b], dtype=np.int64) for b in block_order]
        )
        # Row-shuffle each window of `shuffle_window` consecutive blocks in
        # place: rows leave in random order but never before their block's
        # window opens nor after it closes, which is what bounds the number of
        # concurrently-live row groups (and hence fetches) to the window size.
        window_rows = np.add.reduceat(
            self._block_lengths[block_order],
            np.arange(0, len(block_order), self.shuffle_window),
        )
        offset = 0
        for rows in window_rows:
            rng.shuffle(plan[offset : offset + int(rows)])
            offset += int(rows)
        self._plan = plan

    def __iter__(self) -> Iterator[int]:
        self._ensure_plan()
        while self.position < len(self._plan):
            idx = int(self._plan[self.position])
            self.position += 1
            yield idx
        self.position = 0
        self._plan = None

    def __len__(self) -> int:
        return len(self.data_source)

    def state_dict(self) -> dict:
        if self._plan is not None:
            generator_state = self._plan_gen_state
        else:
            generator_state = self._generator.get_state()
        return {"position": self.position, "generator_state": generator_state}

    def load_state_dict(self, state: dict) -> None:
        self.position = state["position"]
        self._generator.set_state(state["generator_state"])
        self._plan = None
        self._plan_gen_state = None


class DataMixingSampler(torch.utils.data.Sampler[int]):
    """Weighted multi-source sampler built on ``WeightedRandomSampler``.

    The dataset is a concatenation of sources with sizes ``lengths`` (in order);
    ``weights`` gives a sampling weight per source. Per-example weights are set to
    ``weight_source / size_source`` so each *source* is sampled in proportion to
    its weight independent of its size.

    Each epoch draws a fresh plan of ``num_samples`` indices from a persistent
    generator seeded with ``seed``: exhausting the plan clears it, and the next
    ``iter()`` (the trainer re-creates the dataloader iterator at epoch
    boundaries) re-draws with the advanced generator state. ``state_dict``
    captures the cursor plus the generator state *as of the current plan's
    draw*, so a mid-epoch checkpoint resume reproduces the in-flight plan and
    all subsequent epochs match the uninterrupted run.

    Args:
        data_source: The (concatenated) training dataset; only its length is used.
        lengths: Size of each source, in the order they appear in the dataset.
        weights: Per-source sampling weight (need not sum to 1; relative scale matters).
        num_samples: Number of indices to emit per epoch. Defaults to
            ``len(data_source)``, so ``steps_per_epoch`` matches a single
            concatenated dataset of the same size.
        seed: Seed for the deterministic weighted draw.
        replacement: Whether to sample with replacement (passed through to
            ``WeightedRandomSampler``; mixing across sources generally wants True).
    """

    def __init__(
        self,
        data_source: Sequence,
        lengths: Sequence[int],
        weights: Sequence[float],
        num_samples: Optional[int] = None,
        seed: int = 0,
        replacement: bool = True,
    ):
        self.data_source = data_source
        n = len(data_source)
        if sum(lengths) != n:
            raise ValueError(f"DataMixingSampler: sum(lengths)={sum(lengths)} must equal len(data_source)={n}.")
        if len(weights) != len(lengths):
            raise ValueError(f"DataMixingSampler: weights ({len(weights)}) and lengths ({len(lengths)}) must align.")
        if any(length <= 0 for length in lengths):
            raise ValueError(f"DataMixingSampler: all lengths must be > 0, got {list(lengths)}.")
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(
                f"DataMixingSampler: weights must be non-negative with a positive sum, got {list(weights)}."
            )

        self.num_samples = num_samples if num_samples is not None else n
        if self.num_samples <= 0:
            raise ValueError(f"DataMixingSampler: num_samples must be > 0, got {self.num_samples}.")
        self.position = 0
        self.replacement = replacement

        # Spread each source's weight across its examples so the source-level
        # mixing proportion matches ``weights`` regardless of source size.
        per_example_weights: List[float] = []
        for length, weight in zip(lengths, weights):
            per_example_weights.extend([weight / length] * length)
        self._per_example_weights = per_example_weights

        self._generator = torch.Generator()
        self._generator.manual_seed(seed)
        # The current epoch's plan, drawn lazily. ``_plan_gen_state`` snapshots
        # the generator *before* the draw so a resumed sampler re-draws the
        # identical plan.
        self._plan: Optional[List[int]] = None
        self._plan_gen_state: Optional[torch.Tensor] = None

    def _ensure_plan(self) -> None:
        if self._plan is not None:
            return
        self._plan_gen_state = self._generator.get_state()
        weighted = torch.utils.data.WeightedRandomSampler(
            self._per_example_weights,
            num_samples=self.num_samples,
            replacement=self.replacement,
            generator=self._generator,
        )
        self._plan = list(weighted)

    def __iter__(self) -> Iterator[int]:
        self._ensure_plan()
        while self.position < len(self._plan):
            idx = self._plan[self.position]
            self.position += 1
            yield idx
        # Reset for the next epoch: clear the plan so the next ``iter()``
        # draws fresh indices with the advanced generator state.
        self.position = 0
        self._plan = None

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> dict:
        # Mid-epoch: persist the pre-draw state so resume re-draws the current
        # plan. Between epochs: persist the advanced state so the next epoch's
        # draw matches the uninterrupted run.
        if self._plan is not None:
            generator_state = self._plan_gen_state
        else:
            generator_state = self._generator.get_state()
        return {"position": self.position, "generator_state": generator_state}

    def load_state_dict(self, state: dict) -> None:
        self.position = state["position"]
        if "generator_state" in state:
            self._generator.set_state(state["generator_state"])
        else:
            # Position-only state (no generator state): keep the freshly-seeded
            # generator and resume the cursor best-effort.
            logger.warning(
                "DataMixingSampler: checkpoint has no generator_state; "
                "resuming position only -- the restored plan may differ from the run that saved it."
            )
        self._plan = None
        self._plan_gen_state = None
