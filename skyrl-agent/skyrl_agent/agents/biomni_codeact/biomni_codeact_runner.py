import os
import copy
import uuid
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

from skyrl_agent.agents.base import BaseTrajectory

# Biomni CodeAct runtime + agent (ported in)
# from verl.workers.agentic.biomni.biomni_codeact import (
#     BiomniRuntimeClient,
#     BiomniCodeActAgent,
# )

from skyrl_agent.agents.biomni_codeact.biomni_codeact_agent import (
    BiomniCodeActAgent,
    BiomniRuntimeClient,
    gen_chat_template,
)


class BiomniCodeActTrajectory(BaseTrajectory):
    # Use custom chat template from biomni_qwen3.jinja for post-processing tokenization.
    # This ensures tokenization consistency between rollout (generation) and training.
    # The same template is used for both normal and thinking modes since the jinja
    # template already handles <think> blocks appropriately.
    CHAT_TEMPLATE = gen_chat_template
    CHAT_TEMPLATE_THINKING = gen_chat_template

    async def initialize_trajectory(self):
        pass

    def _messages_to_prompt(self, messages):
        """
        Flatten a list of OpenAI-style messages (or a plain string) into a single prompt string.

        Prompt datasets used by biomni currently store the instruction as a single string rather
        than structured chat turns, so we normalize any str/list[str] inputs here.
        """
        if isinstance(messages, str):
            norm_msgs = [{"role": "user", "content": messages}]
        elif isinstance(messages, list) and messages and isinstance(messages[0], str):
            norm_msgs = [{"role": "user", "content": msg} for msg in messages]
        else:
            norm_msgs = messages

        parts = []
        for m in norm_msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)

    async def generate_trajectory(self) -> None:
        data = self.data
        instance_id = data["instance_id"] if data["instance_id"] else self.cfg.instance_id
        instance = pd.Series(data["instance"])

        # Build prompt string from task instruction messages
        instruction_msgs = self.task.get_instruction(instance)
        prompt_str = self._messages_to_prompt(instruction_msgs)

        # Bridge AsyncInferBackend to the Biomni agent engine API
        class _BackendAsEngine:
            def __init__(self, backend):
                self.backend = backend

            async def async_generate(self, input_ids, sampling_params):
                response_str, meta_info = await self.backend.async_generate_ids(
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                    request_id=f"{instance_id}-{uuid.uuid4().hex}",
                )
                # Extract logprobs and output_tokens from meta_info if available
                logprobs = meta_info.get("logprobs") if isinstance(meta_info, dict) else None
                stop_reason = meta_info.get("finish_reason") if isinstance(meta_info, dict) else meta_info
                output_tokens = meta_info.get("output_tokens") if isinstance(meta_info, dict) else None
                return {
                    "text": response_str,
                    "stop_reason": stop_reason,
                    "logprobs": logprobs,
                    "output_tokens": output_tokens,
                }

        engine_adapter = _BackendAsEngine(self.infer_engine)

        runtime_url = os.getenv("BIOMNI_RUNTIME_URL") or "http://localhost:8000"
        # Only stop on <|im_end|> (turn delimiter) - NOT on </execute> or </solution>.
        # This lets the model learn to generate complete turns via format reward.
        # include_stop_str_in_output=True: vLLM returns the stop token and its logprob
        # so the agent can strip it for TITO but use it for tokenization correctness.
        sampling_params = copy.deepcopy(self.cfg.sampling_params)
        sampling_params["stop"] = ["<|im_end|>"]
        sampling_params["include_stop_str_in_output"] = True

        def _safe_int(val: Optional[int]) -> Optional[int]:
            try:
                if val is None:
                    return None
                val = int(val)
                return val if val > 0 else None
            except (TypeError, ValueError):
                return None

        configured_prompt_budget = _safe_int(self.cfg.max_prompt_length)
        tokenizer_ctx_limit = _safe_int(getattr(self.tokenizer, "model_max_length", None))
        backend_ctx_limit = _safe_int(getattr(self.infer_engine, "max_model_len", None))
        env_ctx_limit = _safe_int(os.getenv("BIOMNI_MAX_MODEL_LEN"))
        # Hugging Face sometimes sets an astronomically large sentinel when no limit is known.
        if tokenizer_ctx_limit and tokenizer_ctx_limit > 10**8:
            tokenizer_ctx_limit = None

        cfg_ctx_limit = _safe_int(getattr(self.cfg, "max_model_len", None))
        fallback_ctx_limit = _safe_int(os.getenv("BIOMNI_FALLBACK_MAX_MODEL_LEN")) or 32768
        llm_ctx_limit = (
            backend_ctx_limit
            or env_ctx_limit
            or cfg_ctx_limit
            or tokenizer_ctx_limit
            or fallback_ctx_limit
        )

        resp_budget = (
            _safe_int(sampling_params.get("max_tokens"))
            or _safe_int(sampling_params.get("max_generate_length"))
            or _safe_int(sampling_params.get("max_new_tokens"))
            or 4096
        )

        max_prompt_len = configured_prompt_budget or llm_ctx_limit
        if llm_ctx_limit:
            if resp_budget and resp_budget < llm_ctx_limit:
                ctx_budget = llm_ctx_limit - resp_budget
            else:
                ctx_budget = max(1, llm_ctx_limit - 1)  # leave at least 1 token for decode
            if ctx_budget <= 0:
                ctx_budget = max(512, llm_ctx_limit // 2)
            max_prompt_len = min(max_prompt_len, ctx_budget)

        if max_prompt_len <= 0:
            max_prompt_len = 1024

        if configured_prompt_budget and max_prompt_len < configured_prompt_budget:
            logger.warning(
                "Clamping Biomni max_prompt_length from {} to {} tokens (tokenizer limit={}, reserved_for_output={})",
                configured_prompt_budget,
                max_prompt_len,
                llm_ctx_limit,
                resp_budget,
            )
        
        # task_name is inside instance, not at top-level data
        task_name = instance.get("task_name") if hasattr(instance, "get") else None
        
        # print("instantiating rollout agent")
        # print(f"Task name: {task_name}")
        # print(f"Instance ID: {instance_id}")
        # print(f"Prompt: {prompt_str}")

        # Run single Biomni agent within dispatcher-managed trajectory
        async with BiomniRuntimeClient(runtime_url) as rt:
            self.agent = BiomniCodeActAgent(
                prompt=prompt_str,
                instance_id=instance_id,
                task_name=task_name,
                runtime=rt,
                infer_engine=engine_adapter,
                tokenizer=self.tokenizer,
                sampling_params=sampling_params,
                max_prompt_len=max_prompt_len,
                max_iterations=self.cfg.max_iterations,
                qwen3_enable_thinking=self.cfg.qwen3_enable_thinking,
            )
            prompt_token_limit = llm_ctx_limit
            skip_due_to_length = False
            prompt_token_count: Optional[int] = None
            if prompt_token_limit:
                try:
                    prompt_token_count = self.agent.estimate_initial_prompt_tokens()
                except Exception as exc:
                    logger.warning(
                        "Failed to estimate prompt length for instance %s trajectory %s (%s)",
                        instance_id,
                        self.cfg.trajectory_id,
                        exc,
                    )
                else:
                    if prompt_token_count > prompt_token_limit:
                        skip_due_to_length = True

            if skip_due_to_length:
                logger.warning(
                    "Skipping instance %s trajectory %s because prompt uses %s tokens (limit %s)",
                    instance_id,
                    self.cfg.trajectory_id,
                    prompt_token_count,
                    prompt_token_limit,
                )
                self.result = {
                    "instance_id": instance_id,
                    "trajectory_id": self.cfg.trajectory_id,
                    "messages": [],
                    "results": None,
                    "finish_reason": "CONTEXT_WINDOW_EXCEEDED",
                    "state": {
                        "skip_reason": "prompt_length_exceeded",
                        "prompt_tokens": prompt_token_count,
                        "prompt_token_limit": prompt_token_limit,
                    },
                    "reward": 0,
                }
                return

            result = await self.agent.run()

        msgs_out = result.get("messages", [])
        solution = result.get("solution")
        iterations = result.get("iterations")
        logprobs = result.get("logprobs")  # List of lists of logprobs for each generation step

        finish_reason = "FINISH_TOOL" if solution else ("max_iterations_reached" if iterations and iterations >= self.cfg.max_iterations else None)

        transitions = result.get("transitions")  # List[Transition] for token-in-token-out TIS

        self.result = {
            "instance_id": instance_id,
            "trajectory_id": self.cfg.trajectory_id,
            "messages": msgs_out,
            "results": solution,
            "finish_reason": finish_reason,
            "state": {},
            "transitions": transitions,  # Token-in-token-out transitions for TIS
            "logprobs": logprobs,  # Legacy logprobs fallback for TIS
        }

    async def evaluate_trajectory(self) -> None:
        # instance_id = self.cfg.instance_id
        # trajectory_id = self.cfg.trajectory_id
        data = self.data
        instance = data["instance"]
        
        instance_id = instance["instance_id"]
        
        # assert instance_id == instance["instance_id"]
        result = self.result.get("results")

        skip_reason = (self.result.get("state") or {}).get("skip_reason")
        if skip_reason == "prompt_length_exceeded":
            # Reward already set to 0 during generation, nothing else to evaluate.
            self.result.setdefault("reward", 0)
            return

        try:
            # Use the new BiomniRewardAdapter.compute_rewards
            from skyrl_agent.tasks.biomni_reward_adapter import BiomniRewardAdapter
            
            # task_name is inside instance, not at top-level data
            task_name = instance.get("task_name") if isinstance(instance, dict) else None
            
            metrics = BiomniRewardAdapter.compute_rewards(
                instance=instance,
                solution=result,
                messages=self.result.get("messages", []),
                instance_id=instance_id,
                task_name=task_name
            )
            
            self.result["reward"] = metrics["score"]
            self.result["gt_reward"] = metrics["gt_reward"]
            self.result["ft_reward"] = metrics["ft_reward"]
            
        except Exception as e:
            print(f"Error evaluating result: {e}")
            self.result["reward"] = 0
            self.result["eval_error"] = str(e)
