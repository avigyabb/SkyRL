"""Run a torchrun command on N whole 8-GPU Ray nodes (one torchrun per node).

Usage (as a Ray job entrypoint): python launch_multinode.py <num_nodes> <script> [args...]
Runs from /shared/hackskyrl/muse_spark/SkyRL inside `uv run --isolated --extra megatron`.
"""

import subprocess
import sys

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

REPO = "/shared/hackskyrl/muse_spark/SkyRL"


@ray.remote(num_gpus=8, num_cpus=32)
class Node:
    def ip(self):
        return ray.util.get_node_ip_address()

    def run(self, nnodes, rank, master, args):
        import os

        launcher = os.environ.get("TORCHRUN", "").split() or [
            "uv",
            "run",
            "--isolated",
            "--extra",
            "megatron",
            "torchrun",
        ]
        cmd = [
            *launcher,
            f"--nnodes={nnodes}",
            "--nproc-per-node=8",
            f"--node-rank={rank}",
            f"--master-addr={master}",
            "--master-port=29511",
            *args,
        ]
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        return rank, proc.returncode, proc.stdout[-20000:], proc.stderr[-8000:]


def main():
    nnodes, args = int(sys.argv[1]), sys.argv[2:]
    ray.init()
    pg = placement_group([{"GPU": 8, "CPU": 32}] * nnodes, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    nodes = [
        Node.options(scheduling_strategy=PlacementGroupSchedulingStrategy(pg, placement_group_bundle_index=i)).remote()
        for i in range(nnodes)
    ]
    master = ray.get(nodes[0].ip.remote())
    for rank, code, out, err in ray.get([n.run.remote(nnodes, i, master, args) for i, n in enumerate(nodes)]):
        print(f"===== node {rank} exit={code}\n{out}")
        if code:
            print(f"----- node {rank} stderr\n{err}")


if __name__ == "__main__":
    main()
