import os
import copy
import uuid
import pandas as pd

from skyrl_agent.agents.base import BaseTrajectory

# Biomni CodeAct runtime + agent (ported in)
# from verl.workers.agentic.biomni.biomni_codeact import (
#     BiomniRuntimeClient,
#     BiomniCodeActAgent,
# )

from skyrl_agent.agents.biomni_codeact.biomni_codeact_agent import BiomniCodeActAgent, BiomniRuntimeClient


class BiomniCodeActTrajectory(BaseTrajectory):
    async def initialize_trajectory(self):
        pass

    def _messages_to_prompt(self, messages):
        # Flatten a list of OpenAI-style messages into a single prompt string
        parts = []
        for m in messages:
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
                response_str, stop_reason = await self.backend.async_generate_ids(
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                    request_id=f"{instance_id}-{uuid.uuid4().hex}",
                )
                return {"text": response_str, "stop_reason": stop_reason}

        engine_adapter = _BackendAsEngine(self.infer_engine)

        runtime_url = os.getenv("BIOMNI_RUNTIME_URL") or "http://localhost:8000"
        # Ensure stop tokens are set similarly to Biomni group
        sampling_params = copy.deepcopy(self.cfg.sampling_params)
        sampling_params.update({
            "stop": ["</execute>", "</solution>"],
            "no_stop_trim": True,
        })

        # Run single Biomni agent within dispatcher-managed trajectory
        async with BiomniRuntimeClient(runtime_url) as rt:
            self.agent = BiomniCodeActAgent(
                prompt=prompt_str,
                instance_id=instance_id,
                task_name=self.task.__class__.__name__,
                runtime=rt,
                infer_engine=engine_adapter,
                tokenizer=self.tokenizer,
                sampling_params=sampling_params,
                max_prompt_len=self.cfg.max_prompt_length,
                max_iterations=self.cfg.max_iterations,
                qwen3_enable_thinking=self.cfg.qwen3_enable_thinking,
            )
            result = await self.agent.run()

        msgs_out = result.get("messages", [])
        solution = result.get("solution")
        iterations = result.get("iterations")

        finish_reason = "FINISH_TOOL" if solution else ("max_iterations_reached" if iterations and iterations >= self.cfg.max_iterations else None)

        self.result = {
            "instance_id": instance_id,
            "trajectory_id": self.cfg.trajectory_id,
            "messages": msgs_out,
            "results": solution,
            "finish_reason": finish_reason,
            "state": {},
        }

    async def evaluate_trajectory(self) -> None:
        instance_id = self.cfg.instance_id
        trajectory_id = self.cfg.trajectory_id
        data = self.data
        instance_id = data["instance_id"] if data["instance_id"] else self.cfg.instance_id
        instance = data["instance"]
        result = self.result.get("results")

        try:
            eval_result = await self.task.evaluate_result(
                result,
                instance,
                data["data_source"],
                instance_id,
                trajectory_id,
            )
            self.result["reward"] = eval_result
        except Exception as e:
            print(f"Error evaluating result: {e}")
            self.result["reward"] = 0
            self.result["eval_error"] = str(e)
