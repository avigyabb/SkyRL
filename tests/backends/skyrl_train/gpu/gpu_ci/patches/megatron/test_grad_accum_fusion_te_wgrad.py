"""``patch_grad_accum_fusion_te_wgrad.py``: native Megatron linears with gradient-accumulation fusion, no APEX.

Under torchrun (TP 2, sequence parallel), every rank builds a ``ColumnParallelLinear`` and a
``RowParallelLinear`` with ``gradient_accumulation_fusion=True``. With the patch they build without
APEX's ``fused_weight_gradient_mlp_cuda`` and accumulate their weight gradient into the FP32
``main_grad`` through TE's ``general_gemm``. Two backward passes must sum into ``main_grad`` and match
a float64 reference of ``sum_i dY_i^T X_i``.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from skyrl.train.utils.utils import get_free_port

TOKENS, HIDDEN, FFN = 512, 256, 512


def _check(layer, inputs, grads_out, gather_x, gather_dy):
    """Two backward passes; ``main_grad`` must equal the float64 sum of the local weight gradients.

    Under sequence parallelism a column-parallel layer sees the all-gathered input and a row-parallel
    layer the all-gathered output gradient; ``gather_x`` / ``gather_dy`` rebuild what each one sees.
    """
    layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=torch.float32, device="cuda")
    ref = torch.zeros(layer.weight.shape, dtype=torch.float64, device="cuda")
    for x, dy in zip(inputs, grads_out):
        out, _ = layer(x)
        out.backward(dy)
        x_seen, dy_seen = gather_x(x), gather_dy(dy)
        ref += dy_seen.reshape(-1, dy_seen.size(-1)).double().t() @ x_seen.reshape(-1, x_seen.size(-1)).double()
    err = (layer.weight.main_grad.double() - ref).norm() / ref.norm()
    return bool(err < 1e-2), float(err)


def _distributed_main() -> None:
    from megatron.core import parallel_state
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.tensor_parallel import layers
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    os.environ["SKYRL_GRAD_ACCUM_FUSION_TE_WGRAD"] = "1"
    from skyrl.backends.skyrl_train.patches.megatron.patch_grad_accum_fusion_te_wgrad import (
        apply_grad_accum_fusion_te_wgrad_patch,
    )

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(1234)
    had_apex = layers._grad_accum_fusion_available
    apply_grad_accum_fusion_te_wgrad_patch()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    config = ModelParallelConfig(
        tensor_model_parallel_size=world,
        sequence_parallel=True,
        gradient_accumulation_fusion=True,
        params_dtype=torch.bfloat16,
    )
    init = torch.nn.init.normal_
    col = layers.ColumnParallelLinear(HIDDEN, FFN, config=config, init_method=init, bias=False, gather_output=False)
    row = layers.RowParallelLinear(
        FFN, HIDDEN, config=config, init_method=init, bias=False, input_is_parallel=True, skip_bias_add=True
    )
    torch.manual_seed(rank)
    sp_rows = TOKENS // world
    col_inputs = [torch.randn(sp_rows, 1, HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    col_grads = [torch.randn(TOKENS, 1, FFN // world, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    row_inputs = [torch.randn(TOKENS, 1, FFN // world, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    row_grads = [torch.randn(sp_rows, 1, HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in range(2)]

    def sp_gather(x):
        return gather_from_sequence_parallel_region(x, group=tp_group)

    def same(x):
        return x

    ok_col, err_col = _check(col, col_inputs, col_grads, sp_gather, same)
    ok_row, err_row = _check(row, row_inputs, row_grads, same, sp_gather)
    ok = ok_col and ok_row and layers._grad_accum_fusion_available
    flag = torch.tensor([1.0 if ok else 0.0], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(
            f"RESULT: {'PASS' if flag.item() == 1.0 else 'FAIL'} (TP={world}, apex={had_apex}, "
            f"col rel err {err_col:.2e}, row rel err {err_row:.2e})"
        )
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def test_native_linears_accumulate_into_main_grad_without_apex():
    """Spawn torchrun --nproc_per_node=2 and check both native linears on every rank."""
    if torch.cuda.device_count() < 2:
        pytest.skip(f"needs >= 2 GPUs, have {torch.cuda.device_count()}")
    env = dict(os.environ, MASTER_ADDR="localhost", MASTER_PORT=str(get_free_port()))
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_port",
        env["MASTER_PORT"],
        __file__,
    ]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert "RESULT: PASS" in (res.stdout + res.stderr), f"failed:\n{res.stdout}\n{res.stderr}"


if __name__ == "__main__":
    _distributed_main()
