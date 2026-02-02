import json
import random
from typing import Any, Dict, TypedDict, List, Optional, Type, Callable
from collections import defaultdict
from abc import ABC, abstractmethod
import os
import copy
from omegaconf import OmegaConf, DictConfig
import pandas as pd
from loguru import logger
from skyrl_agent.tasks.base import BaseTask
from transformers import AutoTokenizer
from dataclasses import dataclass

from skyrl_agent.integrations.base import (
    build_backend,
    build_generator_input,
    build_generator_output,
    _import_object,
    AsyncInferBackend,
)
from skyrl_agent.dispatcher.dispatchers import DISPATCHER_REGISTRY, DispatcherType
from skyrl_agent.config.configuration_utils import TASK_CONFIG_REGISTRY, get_field_from_config, TrajectoryConfig
from skyrl_agent.functional.chat_template import chat_template, chat_template_qwen3_thinking
from .mapping import AGENT_TRAJECTORY_REGISTRY


class CompleterOutput:
    pass


@dataclass
class RuntimeConfig:
    runtime_initializer: Optional[Callable]
    instruction_getter: Callable
    config_builder: Optional[Callable]
    completer: Optional[Callable]
    evaluator: Callable

    @classmethod
    def from_dict(cls, cfg: DictConfig):
        def safe_import(cfg, key):
            try:
                val = cfg.get(key, None)
                return _import_object(val) if val else None
            except AttributeError:
                return None

        runtime_initializer = safe_import(cfg, "initializer")
        config_builder = safe_import(cfg, "config_builder")
        instruction_getter = safe_import(cfg, "instruction_getter")  # If optional; else raise if missing
        completer = safe_import(cfg, "completer")
        evaluator = safe_import(cfg, "evaluator")
        return cls(
            runtime_initializer=runtime_initializer,
            config_builder=config_builder,
            instruction_getter=instruction_getter,
            completer=completer,
            evaluator=evaluator,
        )


@dataclass
class TrajectoryResult(TypedDict):
    instance_id: str
    trajectory_id: str
    messages: List[Dict[str, str]]
    state: Any
    results: Optional[CompleterOutput]
    error: Optional[str]
    finish: bool
    finish_reason: str
    reward: Optional[bool]
    eval_error: Optional[str]


class BaseTrajectory(ABC):
    # Custom chat templates for tokenization in post-processing.
    # Subclasses can override these to use agent-specific templates.
    # If None, the default templates from skyrl_agent.functional.chat_template will be used.
    CHAT_TEMPLATE: Optional[str] = None
    CHAT_TEMPLATE_THINKING: Optional[str] = None

    def __init__(
        self,
        cfg: TrajectoryConfig,
        data: Dict[str, Any],
        infer_engine: AsyncInferBackend,
        tokenizer: AutoTokenizer,
        task: BaseTask,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.data = data
        self.infer_engine = infer_engine
        self.tokenizer = tokenizer
        self.task = task
        self.agent_cls = _import_object(cfg.agent_cls)

        self.result: TrajectoryResult = None

    @abstractmethod
    async def initialize_trajectory(self):
        pass

    @abstractmethod
    async def generate_trajectory(self):
        pass

    @abstractmethod
    async def evaluate_trajectory(self):
        pass


# TODO(csy): also specify whether loss_mask, attention_mask, etc. are needed -- for training or eval
class AgentRunner:
    def __init__(self, cfg: Dict[str, Any], infer_engine: Any, tokenizer: Any) -> None:
        """
        Initialize the CodeActGenerator with the given configuration.

        Args:
            generation_config: Configuration dictionary containing parameters like max_prompt_length, max_response_length, etc.
        """
        self.cfg = cfg

        # infer engine
        self.infer_engine = build_backend(
            cfg.generator.infer_backend, infer_engine=infer_engine, cfg=cfg.generator.backend_config
        )
        self.tokenizer = tokenizer
        self.traj_cls: Type[BaseTrajectory] = _import_object(AGENT_TRAJECTORY_REGISTRY.get(cfg.agent_cls))
        self.task: BaseTask = _import_object(cfg.task)()

        # metadata
        self.trajectories: Dict[str, Dict[str, BaseTrajectory]] = {}

        # Will be set in subclasses
        self.agent_config = None

    @classmethod
    def from_task(cls, task: str, infer_engine: Any, tokenizer: Any):
        # Resolve task name or path
        if os.path.exists(task):
            config_path = task
        elif task in TASK_CONFIG_REGISTRY:
            config_path = TASK_CONFIG_REGISTRY[task]
        else:
            raise ValueError(
                f"Unknown task '{task}'. Must be a YAML path or one of: {list(TASK_CONFIG_REGISTRY.keys())}"
            )

        cfg = OmegaConf.load(config_path)

        return cls(cfg, infer_engine, tokenizer)

    def _get_data(self, content) -> Dict[str, Any]:
        """Process input data into trajectory input."""
        data_cfg = self.cfg.get("data", {})

        # Resolve instance payload
        instance = None
        if data_cfg.get("instance_key"):
            try:
                instance = get_field_from_config(data_cfg.get("instance_key"), content)
            except ValueError:
                instance = None

        # Resolve instance_id; rely on configured key and downstream default to batch_id if missing
        instance_id = None
        if data_cfg.get("instance_id_key"):
            try:
                instance_id = get_field_from_config(data_cfg.get("instance_id_key"), content)
            except ValueError:
                instance_id = None

        # Resolve data_source with default fallback
        data_source = "default"
        if data_cfg.get("data_source_key"):
            try:
                data_source = get_field_from_config(data_cfg.get("data_source_key"), content)
            except ValueError:
                data_source = "default"

        return {
            "instance": instance if instance is not None else content,
            "instance_id": instance_id,
            "data_source": data_source,
        }

    def _initialize_trajectories(self, val_mode: bool = False):
        for batch_id, content in enumerate(self.batch):
            data = self._get_data(content)
            instance_id: str = data["instance_id"] if data["instance_id"] else batch_id
            self.trajectories[instance_id] = {}
            sampling_params = (
                self.cfg.generator.val_config.sampling_params if val_mode else self.cfg.generator.sampling_params
            )
            sampling_params = OmegaConf.to_container(sampling_params, resolve=True)  # e.g. converts ListConfig to list
            num_trajectories = (
                self.cfg.generator.val_config.num_trajectories if val_mode else self.cfg.generator.num_trajectories
            )

            profile_tools_cfg = None
            debug_log_cfg = None
            for path in ("generator.profile_tools", "skyrl_agent.profile_tools"):
                try:
                    profile_tools_cfg = OmegaConf.select(self.cfg, path)
                except Exception:
                    profile_tools_cfg = None
                if profile_tools_cfg is not None:
                    break
            for path in ("generator.debug_log", "skyrl_agent.debug_log"):
                try:
                    debug_log_cfg = OmegaConf.select(self.cfg, path)
                except Exception:
                    debug_log_cfg = None
                if debug_log_cfg is not None:
                    break
            profile_tools = bool(profile_tools_cfg) if profile_tools_cfg is not None else False
            debug_log = bool(debug_log_cfg) if debug_log_cfg is not None else False

            use_log_heavy = self.cfg.get("use_log_heavy", True)
            log_heavy_freq = self.cfg.get("log_heavy_freq", 8)

            for traj_id in range(num_trajectories):
                traj_cfg = TrajectoryConfig(
                    instance_id=instance_id,
                    trajectory_id=traj_id,
                    max_prompt_length=self.cfg.generator.max_prompt_length,
                    sampling_params=sampling_params,
                    vision_is_active=self.cfg.generator.vision_is_active,
                    qwen3_enable_thinking=self.cfg.generator.qwen3_enable_thinking,
                    qwen3_acc_thinking=self.cfg.generator.qwen3_acc_thinking,
                    max_iterations=self.cfg.generator.max_iterations,
                    tools=self.cfg.tools,
                    agent_cls=self.cfg.agent_cls,
                    profile_tools=profile_tools,
                    debug_log=debug_log,
                    use_log_heavy=use_log_heavy,
                    log_heavy_freq=log_heavy_freq,
                )
                traj: BaseTrajectory = self.traj_cls(
                    cfg=traj_cfg,
                    data=data,
                    tokenizer=self.tokenizer,
                    infer_engine=self.infer_engine,
                    task=self.task,
                )
                self.trajectories[instance_id][traj_id] = traj

    def _post_process_results(self, return_tensors=False, val_mode: bool = False, global_step: int = 0) -> Dict[str, Any]:
        """
        Post-process the results to convert them into the appropriate output format.

        Returns:
            A dictionary containing the processed results.
        """
        raw_reward_list = []
        gt_reward_list = []
        ft_reward_list = []
        rubric_reward_list = []
        # Itemized rubric scores
        rubric_output_grading_list = []
        rubric_methodology_list = []
        rubric_code_handling_list = []
        rubric_reasoning_list = []
        all_results = {}
        matched_results = []
        instance_list = []
        error_list = []
        resolved_list = []
        has_finish_action_list = []
        finish_reason_list = []

        num_trajectories = (
            self.cfg.generator.val_config.num_trajectories if val_mode else self.cfg.generator.num_trajectories
        )

        for instance_id in self.trajectories:
            for trajectory_id in self.trajectories[instance_id]:
                all_results.setdefault(instance_id, {})[trajectory_id] = self.trajectories[instance_id][
                    trajectory_id
                ].result

        for batch_idx, content in enumerate(self.batch):
            data = self._get_data(content)
            instance = pd.Series(data["instance"])
            instance_id = data["instance_id"] if data["instance_id"] else batch_idx # this is always batch_idx for biomni workloads
            instance["instance_id"] = instance_id  # safe mutation
            trajectories = all_results.get(instance_id, {})
            matched_results.extend(trajectories.values())
            instance_list.extend([instance] * len(trajectories))

        assert len(matched_results) == num_trajectories * len(
            self.batch
        ), f"Expected number of results {num_trajectories * len(self.batch)}, got {len(matched_results)}"

        # Group results by instance_id for message handling
        results_by_instance = {}
        for i, (instance, result) in enumerate(zip(instance_list, matched_results)):
            instance_id = instance["instance_id"]
            results_by_instance.setdefault(instance_id, []).append((i, result))

        global_fallback_set = None
        for results in results_by_instance.values():
            if all(res.get("messages") for _, res in results):
                global_fallback_set = [copy.deepcopy(res) for _, res in results]
                break
        # get reward before handling empty messages
        for idx, result in enumerate(matched_results):
            reward = result.get("reward", False)
            raw_reward_list.append(reward)
            gt_reward_list.append(result.get("gt_reward", 0.0))
            ft_reward_list.append(result.get("ft_reward", 0.0))
            rubric_reward_list.append(result.get("rubric_reward", 0.0))
            # Collect itemized rubric scores if available
            rubric_details = result.get("rubric_details", {})
            rubric_output_grading_list.append(rubric_details.get("output_grading", 0.0))
            rubric_methodology_list.append(rubric_details.get("methodology_knowhow", 0.0))
            rubric_code_handling_list.append(rubric_details.get("code_data_handling", 0.0))
            rubric_reasoning_list.append(rubric_details.get("reasoning_coherence", 0.0))
        raw_reward = sum(raw_reward_list) / len(raw_reward_list)
        num_empty_messages = sum(1 for res in matched_results if not res.get("messages", []))
        # Handle empty messages by copying from another trajectory of the same instance
        for instance_id, results in results_by_instance.items():
            # Look for a non-empty base result
            fallback = next((res for _, res in results if res.get("messages")), None)
            if not fallback:
                if global_fallback_set:
                    logger.warning(
                        f"[WARN] No local fallback for instance_id {instance_id}, using global fallback set."
                    )
                    for j, (idx, res) in enumerate(results):
                        # Use corresponding global fallback result (same trajectory index)
                        fallback_res = global_fallback_set[j % len(global_fallback_set)]
                        print(f"Empty messages for instance_id {instance_id}, trajectory {idx}. Using global fallback.")
                        for key, value in fallback_res.items():
                            matched_results[idx][key] = copy.deepcopy(value)

                else:
                    logger.error(f"[FATAL] No fallback (local/global) for instance_id {instance_id}. Skipping.")
                    continue
            else:
                for idx, res in results:
                    if not res.get("messages", []):
                        print(f"Empty messages for instance_id {instance_id}, trajectory {idx}. Using local fallback.")
                        for key, value in fallback.items():
                            matched_results[idx][key] = copy.deepcopy(value)

        # Get batch of messages
        all_messages = []
        all_prompts = []
        all_responses = []
        num_turns = []  # assistant-based turns
        for result in matched_results:
            messages = result.get("messages", [])
            all_messages.append(messages)
            # get the response: starting from the first assistant message
            starting_index = 0
            # Count assistant messages as turns to match actual steps
            num_turns.append(sum(1 for m in messages if m.get("role") == "assistant"))
            for i, msg in enumerate(messages):
                if msg["role"] == "assistant":
                    starting_index = i
                    break
            if starting_index == 0:
                # If we don't find an assistant, all messages are prompts and there are no responses
                print(
                    f'ERROR: Found no assistant message. len(messages) == {len(messages)} and roles are {[msg["role"] for msg in messages]}'
                )
                starting_index = len(messages)
            prompt = messages[:starting_index]
            all_prompts.append(prompt)
            response = messages[starting_index:]
            all_responses.append(response)

            error_list.append(result.get("error", None))
            resolved_list.append(result.get("reward", False))
            has_finish_action_list.append(result.get("finish", False))
            finish_reason_list.append(result.get("finish_reason", None))

        # Encode messages, get assitant mask and position ids
        # Use trajectory-class-specific templates if available, otherwise use defaults
        traj_template = getattr(self.traj_cls, 'CHAT_TEMPLATE', None) or chat_template
        traj_template_thinking = getattr(self.traj_cls, 'CHAT_TEMPLATE_THINKING', None) or chat_template_qwen3_thinking
        
        if getattr(self.traj_cls, 'CHAT_TEMPLATE', None) is None:
            logger.warning("[AgentRunner] CHAT_TEMPLATE is None or missing for traj_cls, using the default chat template.")
        if getattr(self.traj_cls, 'CHAT_TEMPLATE_THINKING', None) is None:
            logger.warning("[AgentRunner] CHAT_TEMPLATE_THINKING is None or missing for traj_cls, using the default chat_template_qwen3_thinking.")
        
        prompt_encodings = self.tokenizer.apply_chat_template(
            all_prompts,
            # return_tensors="pt",
            add_generation_prompt=False,
            return_dict=True,
            # padding=True
        )
        prompt_input_ids = prompt_encodings["input_ids"]

        response_encodings = self.tokenizer.apply_chat_template(
            all_responses,
            chat_template=traj_template_thinking if self.cfg.generator.remove_think_tokens else traj_template,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
            return_dict=True,
        )

        response_ids = response_encodings["input_ids"]
        response_assistant_mask = response_encodings["assistant_masks"]

        mask_out_reason = ["CONTEXT_WINDOW_EXCEEDED", "error_runtime", "error_evaluation", "max_iterations_reached"]
        max_response_length = self.cfg.generator.max_prompt_length
        truncated_ids = []
        truncated_masks = []

        for idx, (ids, mask, reason) in enumerate(zip(response_ids, response_assistant_mask, finish_reason_list)):
            # Check if truncation is needed
            if len(ids) > max_response_length:
                # Assert finish_reason correctness
                # assert reason in mask_out_reason, (
                #     f"[Sanity Check Failed] Response length {len(ids)} > max_response_length={max_response_length} "
                #     f"but finish_reason='{reason}' not in mask_out_reason={mask_out_reason}"
                # )
                if reason not in mask_out_reason:
                    logger.warning(
                        f"[WARN] Response length {len(ids)} > max_response_length={max_response_length} "
                        f"but finish_reason='{reason}' not in mask_out_reason={mask_out_reason}"
                    )
                    # modify reason to CONTEXT_WINDOW_EXCEEDED
                    finish_reason_list[idx] = "CONTEXT_WINDOW_EXCEEDED"
                # Truncate tokens and masks
                truncated_ids.append(ids[:max_response_length])
                truncated_masks.append(mask[:max_response_length])
            else:
                # No truncation needed
                truncated_ids.append(ids)
                truncated_masks.append(mask)
        response_ids = truncated_ids
        response_assistant_mask = truncated_masks

        loss_mask = [
            [0] * len(mask) if (reason in mask_out_reason) else mask
            for mask, reason in zip(response_assistant_mask, finish_reason_list)
        ]
        
        # Additional filtering: mask out overlong rollouts with bad formatting
        # Configurable threshold (default: 32768 tokens)
        overlong_threshold = self.cfg.get("overlong_filter_threshold", 32768)
        overlong_filter_enabled = self.cfg.get("overlong_filter_enabled", False)
        num_overlong_filtered = 0
        
        if overlong_filter_enabled:
            for idx, (ids, ft_reward, mask) in enumerate(zip(response_ids, ft_reward_list, loss_mask)):
                response_len = len(ids)
                # If response is overlong AND format reward is 0, mask it out
                if response_len > overlong_threshold and ft_reward == 0.0:
                    loss_mask[idx] = [0] * len(mask)
                    num_overlong_filtered += 1
                    logger.info(
                        f"[Overlong Filter] Masked out rollout {idx}: "
                        f"response_len={response_len} > threshold={overlong_threshold}, ft_reward={ft_reward}"
                    )

        rollout_metrics = {}
        # Compute assistant-based turn average and record metric
        avg_turn_assistant = (sum(num_turns) / len(num_turns)) if len(num_turns) > 0 else 0.0
        rollout_metrics["rollout_metrics/avg_turn_assistant"] = avg_turn_assistant

        # Note: no backward-compat key kept (removed per request)

        # 1. Calculate pass@n using GT reward (strictly task success)
        total_per_instance_gt = defaultdict(int)
        resolved_per_instance_gt = defaultdict(int)
        for instance, reward in zip(instance_list, gt_reward_list):
            instance_id = instance["instance_id"]
            total_per_instance_gt[instance_id] += 1
            if reward > 0:
                resolved_per_instance_gt[instance_id] += 1

        # 2. Calculate num_resolved using total reward (used for training signals)
        total_per_instance = defaultdict(int)
        resolved_per_instance = defaultdict(int)
        for instance, reward in zip(instance_list, resolved_list):
            instance_id = instance["instance_id"]
            total_per_instance[instance_id] += 1
            if reward > 0:
                resolved_per_instance[instance_id] += 1

        # Track how many instances have resolution rate 0% or 100%
        num_resolved_0 = 0
        num_resolved_1 = 0

        # Print ratio and update counts
        for instance in sorted(total_per_instance):
            total = total_per_instance[instance]
            resolved = resolved_per_instance[instance]

            if resolved == 0:
                num_resolved_0 += 1
            elif resolved == total:
                num_resolved_1 += 1

        rollout_metrics["rollout_metrics/num_all_resolved"] = num_resolved_1
        rollout_metrics["rollout_metrics/num_none_resolved"] = num_resolved_0
        rollout_metrics["rollout_metrics/finish_tool_ratio"] = sum(
            1 for reason in finish_reason_list if reason == "FINISH_TOOL"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/context_exceed_ratio"] = sum(
            1 for reason in finish_reason_list if reason == "CONTEXT_WINDOW_EXCEEDED"
        ) / len(finish_reason_list)
        # Ratio of trajectories stopped by iteration cap; avoid 'max' in key to prevent max-reduction
        rollout_metrics["rollout_metrics/iter_cap_ratio"] = sum(
            1 for reason in finish_reason_list if reason == "max_iterations_reached"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/stuck_in_a_loop_ratio"] = sum(
            1 for reason in finish_reason_list if reason == "stuck_in_a_loop"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/error_runtime"] = sum(
            1 for reason in finish_reason_list if reason == "error_runtime"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/error_evaluation"] = sum(
            1 for reason in finish_reason_list if reason == "error_evaluation"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/num_mask_out"] = sum(1 for mask in loss_mask if all(m == 0 for m in mask))
        rollout_metrics["rollout_metrics/num_mask_non_zero_reward"] = sum(
            1 for mask, reward in zip(loss_mask, resolved_list) if all(m == 0 for m in mask) and reward > 0
        )
        # Track overlong + bad format filtered rollouts
        rollout_metrics["rollout_metrics/num_overlong_filtered"] = num_overlong_filtered if overlong_filter_enabled else 0
        rollout_metrics["rollout_metrics/raw_reward"] = raw_reward
        rollout_metrics["rollout_metrics/gt_reward"] = sum(gt_reward_list) / len(gt_reward_list) if gt_reward_list else 0.0
        rollout_metrics["rollout_metrics/ft_reward"] = sum(ft_reward_list) / len(ft_reward_list) if ft_reward_list else 0.0
        rollout_metrics["rollout_metrics/rubric_reward"] = sum(rubric_reward_list) / len(rubric_reward_list) if rubric_reward_list else 0.0
        # Itemized rubric scores (for rubric-based evaluation)
        rollout_metrics["rollout_metrics/rubric_output_grading"] = sum(rubric_output_grading_list) / len(rubric_output_grading_list) if rubric_output_grading_list else 0.0
        rollout_metrics["rollout_metrics/rubric_methodology"] = sum(rubric_methodology_list) / len(rubric_methodology_list) if rubric_methodology_list else 0.0
        rollout_metrics["rollout_metrics/rubric_code_handling"] = sum(rubric_code_handling_list) / len(rubric_code_handling_list) if rubric_code_handling_list else 0.0
        rollout_metrics["rollout_metrics/rubric_reasoning"] = sum(rubric_reasoning_list) / len(rubric_reasoning_list) if rubric_reasoning_list else 0.0
        
        # Calculate pass@n (percentage of instances with at least one success)
        # Note: num_none_resolved counts instances where resolution rate is 0 (meaning 0 successes)
        # So pass@n = (total_instances - instances_with_0_success) / total_instances
        num_instances = len(total_per_instance_gt)
        
        # Calculate how many instances had 0 GT success
        num_resolved_0_gt = 0
        for instance in sorted(total_per_instance_gt):
            total = total_per_instance_gt[instance]
            resolved = resolved_per_instance_gt[instance]
            if resolved == 0:
                num_resolved_0_gt += 1

        if num_instances > 0:
            pass_at_n = (num_instances - num_resolved_0_gt) / num_instances
            rollout_metrics["rollout_metrics/pass_at_n_percentage"] = pass_at_n
        else:
            rollout_metrics["rollout_metrics/pass_at_n_percentage"] = 0.0
            
        rollout_metrics["rollout_metrics/num_empty_messages"] = num_empty_messages

        # Count unique instances per task name (read from instance/task data only)
        task_counts = defaultdict(int)
        seen_instance_ids = set()
        for inst in instance_list:
            inst_id = inst["instance_id"]
            if inst_id in seen_instance_ids:
                continue
            seen_instance_ids.add(inst_id)

            task_name = inst.get("task_name")
            if task_name is None:
                continue
            task_counts[task_name] += 1

        for task_name, count in task_counts.items():
            rollout_metrics[f"rollout_metrics/instances_per_task/{task_name}"] = int(count)

        # Optional aggregation of tool-call profiling if available
        try:
            cfg_profile = None
            for path in ("generator.profile_tools", "skyrl_agent.profile_tools"):
                try:
                    cfg_profile = OmegaConf.select(self.cfg, path)
                except Exception:
                    cfg_profile = None
                if cfg_profile is not None:
                    break
            if cfg_profile is None:
                cfg_profile = os.getenv("SKYAGENT_PROFILE_TOOLS", "0") == "1"
            profile_enabled = bool(cfg_profile)
            tool_calls_per_traj = []
            tool_calls_per_traj_nf = []  # exclude finish
            tool_name_totals = defaultdict(int)
            if profile_enabled:
                for res in matched_results:
                    state = res.get("state") or {}
                    prof = state.get("tool_profile") if isinstance(state, dict) else None
                    if prof and isinstance(prof, dict):
                        total = prof.get("tool_calls_total")
                        by_name = prof.get("tool_calls_by_name") or {}
                        if isinstance(total, int):
                            tool_calls_per_traj.append(total)
                        if isinstance(by_name, dict):
                            # accumulate per-tool totals
                            for k, v in by_name.items():
                                try:
                                    tool_name_totals[k] += int(v)
                                except Exception:
                                    pass
                            # compute no-finish per-traj sum
                            try:
                                nf_sum = sum(int(v) for k, v in by_name.items() if k != "finish")
                                tool_calls_per_traj_nf.append(nf_sum)
                            except Exception:
                                pass

            def emit_distribution(prefix: str, vals: list[int]):
                if not vals:
                    return
                n = len(vals)
                s = sorted(vals)
                mean_val = sum(vals) / n
                mnv = s[0]
                mxv = s[-1]
                rollout_metrics[f"{prefix}_total"] = int(sum(vals))
                rollout_metrics[f"{prefix}_per_traj_mean"] = float(mean_val)
                rollout_metrics[f"{prefix}_per_traj_min"] = float(mnv)
                rollout_metrics[f"{prefix}_per_traj_max"] = float(mxv)

            # Emit distributions for overall and no-finish variants
            emit_distribution("rollout_metrics/tool_calls", tool_calls_per_traj)
            emit_distribution("rollout_metrics/tool_calls_no_finish", tool_calls_per_traj_nf)

            for name, cnt in tool_name_totals.items():
                rollout_metrics[f"rollout_metrics/tool_name/{name}"] = int(cnt)
        except Exception:
            pass

        print("rollout metrics:", rollout_metrics)

        print(f"Finish reason: {finish_reason_list}")

        use_log_heavy = self.cfg.get("use_log_heavy", True)
        log_heavy_freq = self.cfg.get("log_heavy_freq", 8)
        
        
        logger.info(f"use_log_heavy: {use_log_heavy}, log_heavy_freq: {log_heavy_freq}")
        logger.info(f"global_step: {global_step}")

        # global_step is 1-indexed
        if use_log_heavy and (global_step - 1) % log_heavy_freq == 0:
            print(f"\n[Log Heavy] Logging samples for global_step {global_step}")
            # Sample up to 8 instances
            sample_indices = random.sample(range(len(matched_results)), min(8, len(matched_results)))
            
            for idx in sample_indices:
                res = matched_results[idx]
                instance_id = res.get("instance_id")
                trajectory_id = res.get("trajectory_id")
                
                # Find the corresponding data source/task name
                instance_data = instance_list[idx]
                
                # Construct log payload
                # Get rubric details if available
                rubric_details = res.get("rubric_details", {})
                log_payload = {
                    "global_step": global_step,
                    "instance_id": str(instance_id),
                    "trajectory_id": str(trajectory_id),
                    "task_name": instance_data.get("data_source") or instance_data.get("task_name", "unknown"),
                    "messages": res.get("messages", []),
                    "reward": res.get("reward", 0.0),
                    "gt_reward": res.get("gt_reward", 0.0),
                    "ft_reward": res.get("ft_reward", 0.0),
                    "rubric_reward": res.get("rubric_reward", 0.0),
                    # Itemized rubric scores
                    "rubric_output_grading": rubric_details.get("output_grading", 0.0),
                    "rubric_methodology": rubric_details.get("methodology_knowhow", 0.0),
                    "rubric_code_handling": rubric_details.get("code_data_handling", 0.0),
                    "rubric_reasoning": rubric_details.get("reasoning_coherence", 0.0),
                    "rubric_rationale": rubric_details.get("rationale", ""),
                    "rubric_weaknesses": rubric_details.get("weaknesses", []),
                    # "rollout_metrics": rollout_metrics, # Can be verbose to print every time
                    "finish_reason": res.get("finish_reason"),
                }
                
                print(f"--- Sample {idx} ---")
                try:
                    print(json.dumps(log_payload, indent=2, default=str))
                except Exception as e:
                    print(f"Failed to json dump payload: {e}")
                    print(log_payload)
                print("------------------\n")

        # Build rollout_logprobs aligned with response_ids (for TIS)
        # Each trajectory may have logprobs from multiple assistant generation steps
        # Note: response_ids and response_assistant_mask are already truncated at this point
        rollout_logprobs_list = []
        has_any_logprobs = False
        
        for idx, (res, mask, resp_ids) in enumerate(zip(matched_results, response_assistant_mask, response_ids)):
            # Sanity check: mask and response_ids should have the same length
            # (they come from the same apply_chat_template call and are truncated together)
            if len(mask) != len(resp_ids):
                raise ValueError(
                    f"[BUG] Trajectory {idx}: response_assistant_mask length ({len(mask)}) != "
                    f"response_ids length ({len(resp_ids)}). This indicates a bug in tokenization/truncation."
                )
            
            traj_logprobs = res.get("logprobs")  # List of lists (one per assistant generation)
            
            if traj_logprobs is None or len(traj_logprobs) == 0:
                # No logprobs available for this trajectory
                rollout_logprobs_list.append(None)
                continue
            
            has_any_logprobs = True
            
            # Flatten all logprobs from all assistant messages
            flat_logprobs = []
            for step_logprobs in traj_logprobs:
                if step_logprobs:
                    flat_logprobs.extend(step_logprobs)
            
            # Align logprobs with response_ids using the assistant mask
            # Fill non-assistant tokens with 0.0, assistant tokens with actual logprobs
            aligned_logprobs = []
            logprob_idx = 0
            for m in mask:
                if m == 1:  # Assistant token
                    if logprob_idx < len(flat_logprobs):
                        aligned_logprobs.append(float(flat_logprobs[logprob_idx]))
                        logprob_idx += 1
                    else:
                        aligned_logprobs.append(0.0)  # Padding if we run out of logprobs
                else:  # Non-assistant token (user messages, generation prompts, etc.)
                    aligned_logprobs.append(0.0)
            
            rollout_logprobs_list.append(aligned_logprobs)
        
        # Finalize rollout_logprobs
        if has_any_logprobs:
            # Fill None entries with zeros matching response_ids length
            num_none_logprobs = 0
            for idx, lp in enumerate(rollout_logprobs_list):
                if lp is None:
                    rollout_logprobs_list[idx] = [0.0] * len(response_ids[idx])
                    num_none_logprobs += 1
            if num_none_logprobs > 0:
                logger.warning(
                    f"[TIS] {num_none_logprobs}/{len(rollout_logprobs_list)} trajectories had no logprobs. "
                    "These will be filled with zeros."
                )
            rollout_logprobs = rollout_logprobs_list
        else:
            logger.warning(
                "[TIS] No logprobs available for any trajectory in this batch. "
                "If TIS is enabled, ensure sampling_params.logprobs is set in the agent config "
                "and the inference backend supports returning logprobs."
            )
            rollout_logprobs = None

        # Create tensor dictionary
        output = {
            "prompt_token_ids": prompt_input_ids,
            "response_ids": response_ids,
            "rewards": resolved_list,
            "loss_masks": loss_mask,
            "stop_reasons": None,
            "rollout_logprobs": rollout_logprobs,
            "rollout_metrics": rollout_metrics,
        }

        return output

    async def run(self, input_batch: Any, val_mode: bool = False) -> Any:
        """
        Generate trajectories for the given prompts using the configured agents.

        Args:
            prompts: A dictionary containing training instances.
            val_mode: Whether we're running validation.

        Returns:
            Results converted to the appropriate output format based on infer backend.
        """
        self.batch = build_generator_input(self.cfg.generator.infer_backend, input_batch).input_batch

        if val_mode:
            num_trajectories = self.cfg.generator.val_config.num_trajectories
            sampling_params = self.cfg.generator.val_config.sampling_params
        else:
            sampling_params = self.cfg.generator.sampling_params
            num_trajectories = self.cfg.generator.num_trajectories

        # Initialize agents and other components
        self._initialize_trajectories(val_mode=val_mode)

        generator_dispatcher: DispatcherType | None = DISPATCHER_REGISTRY.get(self.cfg.dispatcher.type)
        if not generator_dispatcher:
            raise ValueError(f"Unknown generator type: {self.cfg.dispatcher.type}")
        else:
            logger.info(f"Using generator dispatcher: {self.cfg.dispatcher.type}")
            init_fn = "initialize_trajectory"
            run_fn = "generate_trajectory"
            eval_fn = "evaluate_trajectory"
            dispatcher_cfg = {
                "sampling_params": sampling_params,
                "max_parallel_agents": self.cfg.dispatcher.max_parallel_agents,
                "max_eval_parallel_agents": self.cfg.dispatcher.max_eval_parallel_agents,
                "num_instances": len(self.batch),
                "num_trajectories": num_trajectories,
            }
            await generator_dispatcher(
                dispatcher_cfg, self.trajectories, init_fn=init_fn, run_fn=run_fn, eval_fn=eval_fn
            )

        output = self._post_process_results(val_mode=val_mode, global_step=input_batch.get("batch_metadata").global_step if hasattr(input_batch.get("batch_metadata"), "global_step") else 0)

        # reset after run
        self.trajectories = {}

        return build_generator_output(self.cfg.generator.infer_backend, output).result
