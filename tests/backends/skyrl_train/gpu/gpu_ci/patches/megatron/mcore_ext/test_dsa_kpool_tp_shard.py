"""Two-rank check: ``fused_qk_topk_kpool(query_shard_group=...)`` selects exactly what one rank does alone.

With ``SKYRL_DSA_INDEXER_TP_SHARD=1`` every tensor-parallel rank scores only its slice of query rows
(NVIDIA/Megatron-LM#8299, vendored in ``mcore_ext/dsa_kpool.py``) and the pool selections are
all-gathered. Each row is computed from the same inputs either way, so the selected token indices
must be bitwise identical -- including when the query count does not split evenly across ranks (the
last shard is padded before the all-gather), and with SkyRL's striped row assignment
(``SKYRL_DSA_KPOOL_SHARD_STRIPE``, ``mcore_ext/dsa_kpool_fast.py``).

Run with:
uv run --isolated --extra dev --extra megatron pytest -s \
    tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/mcore_ext/test_dsa_kpool_tp_shard.py
"""

import pytest
import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from skyrl.train.utils import get_ray_pg_ready_with_timeout

pytestmark = pytest.mark.megatron

_WORLD = 2
POOL_SIZE = 4
HEAD_DIM = 128
N_HEADS = 4
INDEX_TOPK = 64


@ray.remote(num_gpus=1)
class _Worker:
    def endpoint(self):
        import socket

        from ray.util import get_node_ip_address

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return get_node_ip_address(), sock.getsockname()[1]

    def run(self, rank, master_addr, master_port, seq_lens, score_chunk_rows, stripe):
        import torch.distributed as dist
        from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
            generate_varlen_mask_params_for_positions,
        )

        from skyrl.backends.skyrl_train.patches.megatron.mcore_ext import dsa_kpool, dsa_kpool_fast

        torch.cuda.set_device(0)
        dist.init_process_group("nccl", init_method=f"tcp://{master_addr}:{master_port}", rank=rank, world_size=_WORLD)
        try:
            device = "cuda"
            seqlen = sum(seq_lens)
            gen = torch.Generator(device=device).manual_seed(0)  # identical inputs on both ranks
            k = torch.randn(seqlen, 1, HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen)
            gate = torch.randn(seqlen, 1, HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen)
            ape = torch.randn(POOL_SIZE, HEAD_DIM, device=device, generator=gen)
            q = torch.randn(seqlen, 1, N_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen)
            weights = torch.randn(seqlen, 1, N_HEADS, device=device, generator=gen)
            cu = torch.tensor([0] + seq_lens, device=device).cumsum(0)
            positions = torch.arange(seqlen, device=device)  # global packed-row positions
            starts, ends = generate_varlen_mask_params_for_positions(cu, positions)
            if score_chunk_rows:
                num_pools = sum(n // POOL_SIZE for n in seq_lens)
                dsa_kpool._KPOOL_SCORE_CHUNK_BYTES = 4 * score_chunk_rows * N_HEADS * num_pools
            dsa_kpool_fast._SHARD_STRIPE = stripe
            if stripe:
                dsa_kpool._kpool_topk_from_score_chunks = dsa_kpool_fast._kpool_topk_from_score_chunks
                dsa_kpool._all_gather_kpool_query_shards = dsa_kpool_fast._all_gather_kpool_query_shards

            def select(group):
                return dsa_kpool_fast.fused_qk_topk_kpool(
                    q,
                    k,
                    weights,
                    INDEX_TOPK,
                    POOL_SIZE,
                    gate,
                    ape,
                    varlen_starts=starts,
                    varlen_ends=ends,
                    cu_seqlens_kv=cu,
                    always_select_tail=True,
                    return_index_scores=False,
                    query_shard_group=group,
                )[1]

            return torch.equal(select(None), select(dist.group.WORLD))
        finally:
            dist.destroy_process_group()


@pytest.mark.parametrize(
    "seq_lens,score_chunk_rows,stripe",
    [
        ([1001], None, 0),  # odd total -> ragged last shard
        ([300, 517, 183], 37, 0),  # many small score chunks
        ([512, 256, 256], 37, 64),  # striped row assignment (needs sq % (ranks * stripe) == 0)
    ],
)
def test_tp_sharded_kpool_selection_is_exact(ray_init_fixture, seq_lens, score_chunk_rows, stripe):
    pg = placement_group([{"GPU": _WORLD, "CPU": _WORLD}], strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=30)
    strategy = PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=0)
    workers = [_Worker.options(scheduling_strategy=strategy).remote() for _ in range(_WORLD)]
    master_addr, master_port = ray.get(workers[0].endpoint.remote())
    same = ray.get(
        [w.run.remote(r, master_addr, master_port, seq_lens, score_chunk_rows, stripe) for r, w in enumerate(workers)]
    )
    assert all(same), same
