"""
# Run only vllm tests (requires vllm extra):
uv run --isolated --extra dev --extra vllm --extra deepspeed pytest tests/gpu/test_policy_local_engines_e2e.py -m "vllm"

# Run only sglang tests (requires sglang extra):
uv run --isolated --extra dev --extra sglang --extra deepspeed pytest tests/gpu/test_policy_local_engines_e2e.py -m "sglang"
"""

import pytest
import asyncio
import ray
import hydra
from omegaconf import DictConfig
from collections import defaultdict
from tests.gpu.utils import init_worker_with_type, get_test_prompts, init_inference_engines, run_inference
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.utils.ppo_utils import PolicyLossRegistry, AdvantageEstimatorRegistry
from ray.experimental.collective import create_collective_group
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
    serve,
    wait_for_server_ready,
    shutdown_server,
)
from tests.gpu.utils import init_inference_engines, initialize_ray
import os
from transformers import AutoTokenizer
from tests.gpu.gpu_ci.test_engine_generation import init_remote_inference_servers
import threading
SERVER_PORT = 8123
SERVER_HOST = "127.0.0.1"


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL = "facebook/opt-125m"


def get_test_actor_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

        # Override specific parameters
        cfg.trainer.policy.model.path = MODEL
        cfg.trainer.critic.model.path = ""
        cfg.trainer.placement.policy_num_gpus_per_node = 2
        cfg.generator.async_engine = True
        cfg.generator.num_inference_engines = 1
        cfg.generator.run_engines_locally = True

        return cfg


@pytest.mark.parametrize(
    ("colocate_all", "weight_sync_backend", "strategy", "backend", "tp_size"),
    [
        pytest.param(False, "nccl", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        # pytest.param(True, "nccl", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        # pytest.param(False, "gloo", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        # pytest.param(True, "gloo", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        # pytest.param(False, "nccl", "deepspeed", "vllm", 2, marks=pytest.mark.vllm),
        # pytest.param(True, "nccl", "deepspeed", "vllm", 2, marks=pytest.mark.vllm),
        # pytest.param(False, "nccl", "fsdp2", "vllm", 2, marks=pytest.mark.vllm),
        # pytest.param(True, "nccl", "fsdp2", "vllm", 2, marks=pytest.mark.vllm),
        # # TODO(Charlie): add TP > 1 tests for sglang when we support it
        # pytest.param(False, "nccl", "deepspeed", "sglang", 1, marks=pytest.mark.sglang),
        # pytest.param(True, "nccl", "deepspeed", "sglang", 1, marks=pytest.mark.sglang),
        # pytest.param(False, "nccl", "fsdp2", "sglang", 1, marks=pytest.mark.sglang),
        # pytest.param(True, "nccl", "fsdp2", "sglang", 1, marks=pytest.mark.sglang),
        # pytest.param(False, "gloo", "fsdp", "sglang", 1, marks=pytest.mark.sglang),
        # pytest.param(True, "gloo", "fsdp", "sglang", 1, marks=pytest.mark.sglang),
    ],
    ids=[
        "no_colocate_nccl_fsdp_vllm",
        # "colocate_nccl_fsdp_vllm",
        # "no_colocate_gloo_fsdp_vllm",
        # "colocate_gloo_fsdp_vllm",
        # "no_colocate_nccl_deepspeed_vllm",
        # "colocate_nccl_deepspeed_vllm",
        # "no_colocate_nccl_fsdp2_vllm",
        # "colocate_nccl_fsdp2_vllm",
        # "no_colocate_nccl_deepspeed_sglang",
        # "colocate_nccl_deepspeed_sglang",
        # "no_colocate_nccl_fsdp2_sglang",
        # "colocate_nccl_fsdp2_sglang",
        # "no_colocate_gloo_fsdp_sglang",
        # "colocate_gloo_fsdp_sglang",
    ],
)
def test_policy_local_engines_e2e(colocate_all, weight_sync_backend, strategy, backend, tp_size):
    """
    Tests initalizing the policy actor group and inference engine, syncing weights, and performing generation.
    """
    print(f"Ray version: {ray.__version__}, path: {ray.__file__}")
    try:
        cfg = get_test_actor_config()
        cfg.trainer.placement.colocate_all = colocate_all
        cfg.generator.weight_sync_backend = weight_sync_backend
        cfg.trainer.strategy = strategy
        cfg.generator.backend = backend
        cfg.generator.inference_engine_tensor_parallel_size = tp_size

        # If colocate is True, this will load the engine, sleep, and wake up the engine
        client, pg = init_inference_engines(
            model=MODEL,
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend=backend,
        )

        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=cfg.trainer.placement.colocate_all,
            num_gpus_per_node=cfg.generator.inference_engine_tensor_parallel_size,
            cfg=cfg,
        )

        policy_actor_to_gpu = {}
        gpu_to_inference_engine_actor = {}
        # Query GPU UUIDs via remote methods
        actor_gpu_uuids = ray.get([a.get_gpu_uuid.remote() for a in policy._actor_handlers])
        actor_list = []
        for a, uuid in zip(policy._actor_handlers, actor_gpu_uuids):
            policy_actor_to_gpu[a] = uuid
            actor_list.append(a)
        engine_gpu_uuids = ray.get([e.inference_engine_actor.get_gpu_uuid.remote() for e in client.engines])
        for e, uuid in zip(client.engines, engine_gpu_uuids):
            gpu_to_inference_engine_actor[uuid] = e.inference_engine_actor
            actor_list.append(e.inference_engine_actor)

        create_collective_group(actor_list, backend="nccl")

        gpu_to_req_ref = {} # all req refs will contain dtensors, but some will contain gpu specific tensors
        for policy_actor in policy._actor_handlers:
            weights_req_ref = policy_actor.get_named_weights_gpu.remote()
            # only need one req per gpu since we don't want to ipc the same weights multiple times
            gpu_to_req_ref[policy_actor_to_gpu[policy_actor]] = weights_req_ref
            # possible deadlock here if you ray.get(ref), all the actors need to get_named_weights_gpu so
            # DTensor can be combined

        output_refs = []
        for gpu in gpu_to_req_ref:
            if gpu in gpu_to_inference_engine_actor:
                inference_engine_actor = gpu_to_inference_engine_actor[gpu]
                weights_req_ref = gpu_to_req_ref[gpu] # only send the req on the same gpu as the inference engine actor
                ref = inference_engine_actor.update_named_weights_gpu.remote(weights_req_ref)
                output_refs.append(ref)

        output = ray.get(output_refs)
        print(f"output: {output}")

        # ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        # asyncio.run(client.reset_prefix_cache())
        # ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL), sampling_params))

        print(f"Example output: {outputs['responses'][0]}, {outputs['stop_reasons'][0]}")
    finally:
        AdvantageEstimatorRegistry.reset()
        PolicyLossRegistry.reset()
        ray.shutdown()


@pytest.mark.parametrize(
    ("colocate_all", "weight_sync_backend", "strategy", "backend", "tp_size"),
    [
        pytest.param(False, "nccl", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        # pytest.param(True, "nccl", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
    ],
    ids=[
        "no_colocate_nccl_fsdp_vllm",
        # "colocate_nccl_fsdp_vllm",
    ],
)
def test_policy_remote_engines_e2e(colocate_all, weight_sync_backend, strategy, backend, tp_size):
    """
    Tests initalizing the policy actor group and inference engine, syncing weights, and performing generation.
    """
    print(f"Ray version: {ray.__version__}, path: {ray.__file__}")

    def get_free_port():
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    server_port = None

    try:
        # 1. Initialize InferenceEngineClient client with remote servers
        cfg = get_test_actor_config()
        cfg.generator.backend = backend
        cfg.trainer.placement.colocate_all = False
        cfg.generator.num_inference_engines = 1
        cfg.generator.inference_engine_tensor_parallel_size = tp_size
        
        # Pin policy to GPU 2 to avoid overlapping with remote vLLM GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        initialize_ray(cfg)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        # Launch remote vLLM on GPUs 0,1 (policy is pinned to GPU 2)
        client, remote_server_process = init_remote_inference_servers(tp_size, backend, tokenizer, cfg, MODEL, gpu_ids=[0, 1] if tp_size > 1 else [0])
        # sampling_params = _get_test_sampling_params(backend, cfg)

        # 2. Start HTTP endpoint in background thread using serve function directly
        server_port = get_free_port()

        def run_server():
            serve(client, host=SERVER_HOST, port=server_port, log_level="warning")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait for server to be ready using the helper method
        wait_for_server_ready(host=SERVER_HOST, port=server_port, max_wait_seconds=30)
        base_url = f"http://{SERVER_HOST}:{server_port}/v1"

        # 4. Shutdown server
        shutdown_server(host=SERVER_HOST, port=server_port, max_wait_seconds=5)
        if server_thread.is_alive():
            server_thread.join(timeout=5)

        policy = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=cfg.trainer.placement.colocate_all,
            num_gpus_per_node=1,
            cfg=cfg,
        )

        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        asyncio.run(client.reset_prefix_cache())
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        # sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        # outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL), sampling_params))

        # print(f"Example output: {outputs['responses'][0]}, {outputs['stop_reasons'][0]}")
    finally:
        AdvantageEstimatorRegistry.reset()
        PolicyLossRegistry.reset()
        ray.shutdown()
