import torch
import torch.nn as nn
import torch.distributed
import ray
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download
import os
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from collections import defaultdict
from tqdm import tqdm
from omegaconf import OmegaConf

# NOTE: The following imports are commented out and moved to lazy imports inside methods
# to avoid protobuf serialization issues when Ray pickles the actor class.
# See: "cannot pickle 'google._upb._message.Descriptor' object" error

# from mbridge import AutoBridge
# import megatron.core.parallel_state as mpu
# from megatron.core.optimizer import DistributedOptimizer
# from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

# from skyrl_train.distributed.megatron.optimizer import (
#     init_megatron_optim_config,
#     get_megatron_optimizer,
#     get_megatron_optimizer_param_scheduler,
# )

from contextlib import contextmanager
from skyrl_train.distributed.dispatch import MeshRank
# from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy
# from skyrl_train.distributed.megatron.megatron_utils import freeze_moe_router, print_model_size
from skyrl_train.utils.utils import update_model_config, str_to_torch_dtype, get_physical_gpu_id
from skyrl_train.training_batch import TrainingOutputBatch
from skyrl_train.workers.worker_utils import BatchIterator, reduce_metrics
from skyrl_train.workers.worker import (
    PolicyWorkerBase,
    RefWorkerBase,
    CriticWorkerBase,
)
# from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper
from skyrl_train.utils.profiler import Profiler

# Type hints only - not imported at runtime to avoid protobuf issues
if TYPE_CHECKING:
    from megatron.core.optimizer import DistributedOptimizer
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper


class MegatronWorker:
    def check_te_import(self):
        try:
            import transformer_engine  # noqa: F401
        except ImportError:
            raise ValueError(
                """
                transformer_engine is required for using the megatron backend.
                For single node training follow the instructions in the pyproject.toml file to install transformer_engine.
                For multi node training, please install transformer_engine in the docker image, and set the PYTHONPATH
                to `/home/ray/anaconda3/lib/python3.12/site-packages` or wherever your base installation of
                transformer_engine lives.
            """
            )

    def init_configs(self, model_path, model_config_kwargs, transformer_config_kwargs, flash_attn=False):
        # Lazy import to avoid protobuf serialization issues with Ray
        from mbridge import AutoBridge

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        override_config_kwargs = {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        override_config_kwargs.update(model_config_kwargs.get("model_config", {}))
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)

        # if flash_attn is enabled, we use flash attention backend, otherwise fall back to fused attention backend
        transformer_config_kwargs = OmegaConf.to_container(transformer_config_kwargs, resolve=True)
        transformer_config_kwargs["attention_backend"] = "flash" if flash_attn else "fused"

        bridge = AutoBridge.from_config(hf_config)
        bridge.set_extra_args(**transformer_config_kwargs)
        tf_config = bridge.config
        self.bridge = bridge

        self.hf_config = hf_config
        self.strategy.hf_config = hf_config
        self.tf_config = tf_config
        self.tokenizer = tokenizer

    def make_megatron_module(
        self,
        model_config_kwargs: Dict[str, Any],
        wrap_with_ddp: bool = True,
        ddp_config: Optional[Dict[str, Any]] = None,
    ) -> List[nn.Module]:
        """
        Creates a megatron GPTModel (optionally DDP wrapped) using the bridge.
        """
        model = self.bridge.get_model(
            post_model_creation_callbacks=[],  # don't rely on these since we might switch to Megatron-Bridge
            wrap_with_ddp=wrap_with_ddp,
            ddp_config=ddp_config,
        )
        if model_config_kwargs.get("moe_config", {}).get("freeze_moe_router", False):
            # Lazy import to avoid protobuf serialization issues with Ray
            from skyrl_train.distributed.megatron.megatron_utils import freeze_moe_router
            freeze_moe_router(model)
        return model

    def forward(self, data):
        """
        Override `Worker.forward` to support passing the full mini batch to the MegatronModelWrapper.forward method.
        """
        # Run in micro batches grouped into a single mini-batch
        micro_bsz = self.cfg.trainer.micro_forward_batch_size_per_gpu
        micro_batches = data.chunk(micro_bsz)

        # Build micro-batch dicts expected by policy.forward_mini_batch
        micro_dicts = []
        device = torch.cuda.current_device()
        for micro in micro_batches:
            micro.to(device)
            sequences = micro["sequences"]
            attention_mask = micro["attention_mask"]
            num_actions = micro.metadata["response_length"]
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            micro_dicts.append(
                {
                    "sequences": sequences,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_actions": num_actions,
                }
            )

        self.model.eval()
        seq_len = micro_dicts[0]["sequences"].shape[1]
        mbs = micro_dicts[0]["sequences"].shape[0]
        with torch.no_grad():
            log_probs = self.model.forward(
                micro_batches=micro_dicts,
                seq_len=seq_len,
                micro_batch_size=mbs,
                temperature=self.cfg.generator.sampling_params.temperature,
            )

        log_probs = log_probs.to("cpu")
        output = TrainingOutputBatch({"output": log_probs})
        output.metadata = data.metadata
        return output

    def save_hf_model(self, export_dir: str, tokenizer):
        # Save model in HuggingFace safetensors format
        self.strategy.save_hf_model(
            self.bridge,
            self.model,
            export_dir,
            tokenizer=tokenizer,
        )


class MegatronPolicyWorkerBase(MegatronWorker, PolicyWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.check_te_import()
        self.model: "MegatronModelWrapper" = None
        self.actor_module: List[nn.Module] = None
        self.scheduler: "OptimizerParamScheduler" = None
        self.optimizer: "DistributedOptimizer" = None
        self.profiler: Profiler = None

    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.actor_module, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(
            self.actor_module, self.optimizer, non_blocking, backload_optimizer, backload_model
        )

    def init_worker_process_group(self):
        """
        Override DistributedTorchRayActor.init_worker_process_group to use megatron distributed setup to create the mesh.
        """
        import logging as _logging
        _logging.info(
            f"[PolicyWorker] init_worker_process_group ENTER: rank={self._rank}/{self._world_size}, "
            f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}"
        )

        # Lazy imports to avoid protobuf serialization issues with Ray
        import megatron.core.parallel_state as mpu
        from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy

        if not torch.distributed.is_initialized():
            import datetime
            _nccl_timeout = int(os.environ.get("NCCL_TIMEOUT", 1800))
            _logging.info(f"[PolicyWorker] rank={self._rank} calling torch.distributed.init_process_group(backend='nccl', timeout={_nccl_timeout}s)...")
            torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=_nccl_timeout))
            _logging.info(f"[PolicyWorker] rank={self._rank} init_process_group DONE")
        else:
            _logging.info(f"[PolicyWorker] rank={self._rank} torch.distributed already initialized, skipping")

        self.strategy = MegatronStrategy(
            megatron_config=self.cfg.trainer.policy.megatron_config,
            optimizer_config=self.cfg.trainer.policy.optimizer_config,
            seed=self.cfg.trainer.seed,
        )
        self.strategy.setup_distributed()

        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )
        _logging.info(f"[PolicyWorker] init_worker_process_group DONE: rank={self._rank}, mesh_rank={self.mesh_rank}")

    def init_model(self, model_path, num_training_steps: int = 1e9):
        """
        Initialize the model, optimizer, and scheduler for the policy worker.
        """
        # Lazy imports to avoid protobuf serialization issues with Ray
        from skyrl_train.distributed.megatron.megatron_utils import print_model_size
        from skyrl_train.distributed.megatron.optimizer import (
            init_megatron_optim_config,
            get_megatron_optimizer,
            get_megatron_optimizer_param_scheduler,
        )
        from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper

        # get hf_config and tf_config
        self.init_configs(
            model_path,
            self.cfg.trainer.policy.megatron_config.model_config_kwargs,
            self.cfg.trainer.policy.megatron_config.transformer_config_kwargs,
            flash_attn=self.cfg.trainer.flash_attn,
        )

        # wrap with DDP for training
        self.actor_module = self.make_megatron_module(
            self.cfg.trainer.policy.megatron_config.model_config_kwargs,
            wrap_with_ddp=True,
            ddp_config=self.cfg.trainer.policy.megatron_config.ddp_config,
        )

        if self._local_rank == 0 and not os.path.exists(
            model_path
        ):  # if not local path, try downloading model weights from huggingface
            snapshot_download(model_path)  # will be no-op if already downloaded
        torch.distributed.barrier()

        # load weights
        # NOTE (erictang000): there is currently a bug in mbridge that causes the model to not load correctly if tie_word_embeddings is set
        # see: https://github.com/NVIDIA/Megatron-LM/issues/533#issuecomment-1760193239
        # this is the case for the Qwen2.5-1.5B and 3B models, but not the 7B model
        self.bridge.load_weights(self.actor_module, model_path)

        if self._rank == 0:
            print_model_size(self.actor_module[0])

        # create profiler
        if self.cfg.trainer.policy.megatron_config.torch_profiler_config.enable:
            self.profiler = Profiler(self.cfg.trainer.policy.megatron_config.torch_profiler_config)

        # create optimizer
        optim_config = init_megatron_optim_config(
            self.cfg.trainer.policy.optimizer_config, self.cfg.trainer.policy.megatron_config.optimizer_config_kwargs
        )
        self.optimizer = get_megatron_optimizer(self.actor_module, optim_config)

        self._normalize_mini_batch_size()

        # create scheduler
        self.scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer,
            config=self.cfg.trainer.policy.optimizer_config,
            num_training_steps=num_training_steps,
        )

        # create worker model
        self.model = MegatronModelWrapper(
            config=self.cfg,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.actor_module,
            actor_optimizer=self.optimizer,
            policy_loss_fn=self.policy_loss_fn,
        )

        sdft_enabled = getattr(self.cfg.trainer, 'sdft_enabled', False)
        if sdft_enabled:
            self._init_ema()
            if self._rank == 0:
                ema_mem = sum(p.numel() * p.element_size() for p in self.ema_params.values())
                print(f"[SDFT] Initialized EMA parameters: {ema_mem / 1e9:.2f} GB on rank 0")

    def _init_ema(self):
        """Initialize EMA parameters by cloning the current model's local shards."""
        self.ema_params = {}
        for name, param in self.actor_module[0].module.named_parameters():
            self.ema_params[name] = param.data.clone()

    def _ema_update(self, decay: float):
        """Update EMA: ema = decay * ema + (1 - decay) * student."""
        with torch.no_grad():
            for name, param in self.actor_module[0].module.named_parameters():
                self.ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    @contextmanager
    def _ema_weights(self):
        """Swap student<->EMA via pointer exchange. No memory allocation."""
        for name, param in self.actor_module[0].module.named_parameters():
            param.data, self.ema_params[name] = self.ema_params[name], param.data
        try:
            yield
        finally:
            for name, param in self.actor_module[0].module.named_parameters():
                param.data, self.ema_params[name] = self.ema_params[name], param.data

    def _build_sdft_teacher_micro_batches(self, sdft_data, micro_sample_indices, device):
        """Build teacher micro-batches from SDFT data for EMA forward pass.

        Returns a list aligned with micro_sample_indices (same length).
        Entries are None where sdft_data has no data for that sample.
        """
        teacher_micro_batches = []
        for idx in micro_sample_indices:
            entry = sdft_data[idx]
            if entry is None:
                teacher_micro_batches.append(None)
                continue

            teacher_prompt_ids = entry["teacher_prompt_ids"]
            response_ids = entry["response_ids"]
            full_ids = teacher_prompt_ids + response_ids
            seq_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            attn_mask = torch.ones_like(seq_tensor, dtype=torch.long)
            position_ids = attn_mask.long().cumsum(-1) - 1

            teacher_micro_batches.append({
                "sequences": seq_tensor,
                "attention_mask": attn_mask,
                "position_ids": position_ids,
                "num_actions": entry["num_actions"],
            })

        if not any(mb is not None for mb in teacher_micro_batches):
            return None
        return teacher_micro_batches

    def ppo_train(self, train_data) -> "TrainingOutputBatch":
        """
        Overrides `PolicyWorkerBase.ppo_train` for megatron.

        Since we want megatron to handle gradient accumulation over micro batches, we directly pass mini batches into the
        worker MegatronModelWrapper.forward_backward_mini_batch method.
        """
        dataloader = BatchIterator(
            train_data, sample_batch_size=self.cfg.trainer.micro_train_batch_size_per_gpu, drop_last=False
        )

        micro_batches_per_mini_batch = (
            self.policy_mini_batch_size_per_gpu // self.cfg.trainer.micro_train_batch_size_per_gpu
        )

        status_list = []
        all_metrics = defaultdict(list)
        policy_update_steps = 0

        if self.profiler is not None:
            self.profiler.start()

        sdft_enabled = getattr(self.cfg.trainer, 'sdft_enabled', False)
        sdft_data = train_data.metadata.get("sdft_data") if sdft_enabled else None

        for epoch in range(self.cfg.trainer.update_epochs_per_batch):
            self.optimizer.zero_grad()
            pbar = tqdm(
                dataloader,
                desc=f"Policy Train epoch [{epoch + 1}/{self.cfg.trainer.update_epochs_per_batch}]",
                disable=not self.strategy.is_rank_0(),
            )

            micro_buffer = []
            micro_sample_indices = []
            for local_step, experience in enumerate(pbar):
                experience.to_device(torch.cuda.current_device())
                sequences = experience.sequences
                attention_mask = experience.attention_mask
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)

                micro_buffer.append(
                    {
                        "sequences": sequences,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "num_actions": experience.num_actions,
                        "old_action_log_probs": experience.action_log_probs,
                        "base_action_log_probs": experience.base_action_log_probs,
                        "advantages": experience.advantages,
                        "loss_mask": experience.loss_mask,
                        "rollout_action_logprobs": experience.rollout_logprobs,
                    }
                )
                micro_sample_indices.append(local_step)

                if len(micro_buffer) == micro_batches_per_mini_batch:
                    # run mini-batch forward-backward and then one optimizer step
                    self.model.train()
                    for chunk in self.actor_module:
                        # if use distributed optimizer, zero grad buffer will be handled by optimizer
                        chunk.zero_grad_buffer()
                    seq_len = micro_buffer[0]["sequences"].shape[1]
                    micro_bsz = micro_buffer[0]["sequences"].shape[0]
                    device = micro_buffer[0]["sequences"].device

                    if sdft_data:
                        kl_estimator = getattr(self.cfg.trainer, 'sdft_kl_estimator', 'reinforce')
                        sdft_coeff = getattr(self.cfg.trainer, 'sdft_loss_coef', 1.0)
                        temperature = self.cfg.generator.sampling_params.temperature

                        teacher_micro_batches = self._build_sdft_teacher_micro_batches(
                            sdft_data, micro_sample_indices, device
                        )

                        if teacher_micro_batches:
                            with torch.no_grad():
                                with self._ema_weights():
                                    if kl_estimator == "reinforce":
                                        teacher_outputs = []
                                        for tmb in teacher_micro_batches:
                                            if tmb is None:
                                                teacher_outputs.append(None)
                                                continue
                                            logps = self.model.forward(
                                                [tmb],
                                                seq_len=tmb["sequences"].shape[1],
                                                micro_batch_size=1,
                                                temperature=temperature,
                                            )
                                            teacher_outputs.append(logps)
                                            print(
                                                f"[SDFT DIAG] micro {len(teacher_outputs)-1}: "
                                                f"teacher_seq_len={tmb['sequences'].shape[1]}, "
                                                f"teacher_num_actions={tmb['num_actions']}, "
                                                f"teacher_logps_shape={logps.shape}"
                                            )
                                    else:
                                        teacher_outputs = []
                                        for tmb in teacher_micro_batches:
                                            if tmb is None:
                                                teacher_outputs.append(None)
                                                continue
                                            logits_list = self.model.forward_teacher_logits(
                                                [tmb],
                                                seq_len=tmb["sequences"].shape[1],
                                                micro_batch_size=1,
                                                temperature=temperature,
                                            )
                                            teacher_outputs.append(logits_list[0])

                            for i, mb in enumerate(micro_buffer):
                                if teacher_outputs[i] is None:
                                    continue
                                idx = micro_sample_indices[i]
                                mask = sdft_data[idx]["loss_mask"]
                                mask_tensor = torch.tensor(mask, device=device, dtype=torch.float32).unsqueeze(0)
                                student_na = mb["num_actions"]
                                if mask_tensor.shape[1] < student_na:
                                    mask_tensor = torch.nn.functional.pad(mask_tensor, (0, student_na - mask_tensor.shape[1]), value=0.0)
                                elif mask_tensor.shape[1] > student_na:
                                    mask_tensor = mask_tensor[:, :student_na]
                                mb["sdft_loss_mask"] = mask_tensor
                                mb["sdft_coeff"] = sdft_coeff
                                if kl_estimator == "reinforce":
                                    teacher_logps = teacher_outputs[i].detach()
                                    student_na = mb["num_actions"]
                                    teacher_na = teacher_logps.shape[1]
                                    if teacher_na < student_na:
                                        teacher_logps = torch.nn.functional.pad(
                                            teacher_logps, (0, student_na - teacher_na), value=0.0
                                        )
                                    elif teacher_na > student_na:
                                        teacher_logps = teacher_logps[:, :student_na]
                                    mb["sdft_teacher_logps"] = teacher_logps
                                    print(
                                        f"[SDFT DIAG] attach micro {i}: "
                                        f"student_num_actions={student_na}, "
                                        f"teacher_num_actions_orig={teacher_outputs[i].shape[1]}, "
                                        f"teacher_logps_shape={teacher_logps.shape}, "
                                        f"mask_sum={sum(sdft_data[micro_sample_indices[i]]['loss_mask'])}"
                                    )
                                else:
                                    teacher_logits = teacher_outputs[i].detach()
                                    teacher_na = teacher_logits.shape[1]
                                    student_response_tokens = mb["sequences"][0, -teacher_na:]
                                    teacher_response_tokens = teacher_micro_batches[i]["sequences"][0, -teacher_na:]
                                    tokens_match = (student_response_tokens == teacher_response_tokens).all().item()
                                    mb["sdft_teacher_logits"] = teacher_logits
                                    print(
                                        f"[SDFT DIAG] micro {i}: "
                                        f"teacher_num_actions={teacher_na}, "
                                        f"student_batch_num_actions={mb['num_actions']}, "
                                        f"response_tokens_match={tokens_match}, "
                                        f"teacher_logits_shape={teacher_logits.shape}"
                                    )

                    metrics_list = self.model.forward_backward_mini_batch(
                        micro_batches=micro_buffer,
                        seq_len=seq_len,
                        micro_batch_size=micro_bsz,
                        temperature=self.cfg.generator.sampling_params.temperature,
                    )

                    grad_norm = self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler, name="actor")

                    if sdft_data:
                        ema_decay = getattr(self.cfg.trainer, 'sdft_ema_decay', 0.99)
                        self._ema_update(ema_decay)

                    # within a DP group, metrics are already the same across all workers - we then just all reduce across
                    # the whole world size to get the metrics for the global micro batch
                    for i, metrics in enumerate(metrics_list):
                        status = {
                            "policy_loss": metrics["policy_loss"],
                            "policy_lr": self.optimizer.param_groups[0]["lr"],
                            "ppo_clip_ratio": metrics["ppo_clip_ratio"],
                            "policy_entropy": metrics["policy_entropy"],
                        }
                        if self.cfg.trainer.algorithm.use_kl_loss:
                            status["policy_kl"] = metrics["policy_kl"]
                        if sdft_data:
                            status["sdft_kl"] = metrics.get("sdft_kl", 0.0)

                        # Attach grad norm only for the last micro in the mini-batch
                        if i == len(metrics_list) - 1 and grad_norm is not None:
                            status["raw_grad_norm"] = grad_norm

                        # attach response_length
                        status["response_length"] = micro_buffer[i]["num_actions"]

                        status = self.strategy.all_reduce(status)
                        status_list.append(status)
                        for k, v in status.items():
                            all_metrics[k].append(v)

                    short_status = {
                        "pg": status_list[-1]["policy_loss"],
                        "glen": status_list[-1]["response_length"],
                        "policy_lr": status_list[-1]["policy_lr"],
                        "ent": status_list[-1]["policy_entropy"],
                    }
                    if "raw_grad_norm" in status_list[-1]:
                        short_status["grad_norm"] = status_list[-1]["raw_grad_norm"]
                    if sdft_data and "sdft_kl" in status_list[-1]:
                        short_status["sdft_kl"] = status_list[-1]["sdft_kl"]
                    pbar.set_postfix(short_status)

                    policy_update_steps += 1
                    micro_buffer = []
                    micro_sample_indices = []

            # drop any trailing micros that don't fill a mini-batch (keep behavior consistent)
            micro_buffer = []
            micro_sample_indices = []

        torch.distributed.barrier()
        if self.profiler is not None:
            self.profiler.stop_and_save()
            self.profiler.stop_trace()

        # not needed beyond status logging
        all_metrics.pop("response_length", None)

        status_mean = reduce_metrics(all_metrics)
        status_mean["policy_update_steps"] = policy_update_steps

        output = TrainingOutputBatch()
        output.metadata = {"train_status": status_mean}
        return output

    async def broadcast_to_inference_engines(self, inference_engine_client):
        use_prefix_cache = self.cfg.generator.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()
        per_tensor_param = self.bridge.export_weights(self.actor_module)
        weights_update_request = {"names": [], "dtypes": [], "shapes": [], "extras": []}
        current_size = 0

        for name, param in per_tensor_param:
            # NOTE (erictang000) we do not use bucketed weight updates for megatron here, which means this is not compatible with the FlashRL integration
            # in the future we should improve this to use bucketed weight updates and support FlashRL + megatron for large models
            from torch.multiprocessing.reductions import reduce_tensor

            device = torch.cuda.current_device()
            param = param.to(device, non_blocking=True)
            param = param.to(generator_dtype)
            weight = param.data.clone()
            ipc_handle = reduce_tensor(weight)

            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                current_size += weight.nbytes
                weights_update_request["names"].append(name)
                weights_update_request["dtypes"].append(self.cfg.generator.model_dtype)
                weights_update_request["shapes"].append(param.shape)
                weights_update_request["extras"].append({"ipc_handles": ipc_handles})
                if current_size / (1024**3) > self.cfg.generator.weight_transfer_threshold_cuda_ipc_GB:
                    await inference_engine_client.update_named_weights(weights_update_request)
                    current_size = 0
                    weights_update_request = {"names": [], "dtypes": [], "shapes": [], "extras": []}
                    # force collect any sent tensors if possible to be memory efficient
                    torch.cuda.ipc_collect()

            torch.distributed.barrier()
            torch.cuda.synchronize()

        if len(weights_update_request["names"]) > 0 and torch.distributed.get_rank() == 0:
            await inference_engine_client.update_named_weights(weights_update_request)
            torch.cuda.ipc_collect()
        torch.distributed.barrier()
        torch.cuda.synchronize()

        if cache_reset_task is not None:
            await cache_reset_task
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # this already gets set in the init_model method
        pass


class MegatronRefWorkerBase(MegatronWorker, RefWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.check_te_import()
        self.model: "MegatronModelWrapper" = None
        self.actor_module: List[nn.Module] = None

    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.actor_module, None, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        self.strategy.backload_to_gpu(self.actor_module, None, non_blocking)

    def init_worker_process_group(self):
        """
        Override DistributedTorchRayActor.init_worker_process_group to use megatron distributed setup to create the mesh.
        """
        import logging as _logging
        _logging.info(
            f"[RefWorker] init_worker_process_group ENTER: rank={self._rank}/{self._world_size}, "
            f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}"
        )

        # Lazy imports to avoid protobuf serialization issues with Ray
        import megatron.core.parallel_state as mpu
        from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy

        if not torch.distributed.is_initialized():
            import datetime
            _nccl_timeout = int(os.environ.get("NCCL_TIMEOUT", 1800))
            _logging.info(f"[RefWorker] rank={self._rank} calling torch.distributed.init_process_group(backend='nccl', timeout={_nccl_timeout}s)...")
            torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=_nccl_timeout))
            _logging.info(f"[RefWorker] rank={self._rank} init_process_group DONE")
        else:
            _logging.info(f"[RefWorker] rank={self._rank} torch.distributed already initialized, skipping")

        self.strategy = MegatronStrategy(
            megatron_config=self.cfg.trainer.ref.megatron_config,
            optimizer_config=None,
            seed=self.cfg.trainer.seed,
        )
        self.strategy.setup_distributed()

        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )

    def init_model(self, model_path, num_training_steps: int = 1e9):
        """
        Initialize the model for the ref worker.
        """
        # Lazy imports to avoid protobuf serialization issues with Ray
        from skyrl_train.distributed.megatron.megatron_utils import print_model_size
        from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper

        # get hf_config and tf_config
        self.init_configs(
            model_path,
            self.cfg.trainer.ref.megatron_config.model_config_kwargs,
            self.cfg.trainer.ref.megatron_config.transformer_config_kwargs,
            flash_attn=self.cfg.trainer.flash_attn,
        )

        self.actor_module = self.make_megatron_module(
            self.cfg.trainer.ref.megatron_config.model_config_kwargs, wrap_with_ddp=False, ddp_config=None
        )

        # load weights
        self.bridge.load_weights(self.actor_module, model_path)
        if self._rank == 0:
            print_model_size(self.actor_module[0])

        # create worker model
        self.model = MegatronModelWrapper(
            config=self.cfg, hf_config=self.hf_config, tf_config=self.tf_config, actor_module=self.actor_module
        )

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # this already gets set in the init_model method
        pass


class MegatronCriticWorkerBase(MegatronWorker, CriticWorkerBase):
    def __init__(self, **kwargs):
        raise NotImplementedError()


PolicyWorker = ray.remote(num_gpus=1)(MegatronPolicyWorkerBase)
RefWorker = ray.remote(num_gpus=1)(MegatronRefWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(MegatronCriticWorkerBase)
