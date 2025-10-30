import os
import copy
import uuid
import pandas as pd
import torch

from skyrl_agent.agents.base import BaseTrajectory

# Biomni CodeAct runtime + group (ported in)
from verl import DataProto
from verl.workers.agentic.biomni.biomni_codeact import BiomniCodeActAgentGroup


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

        # Prepare a single-instance DataProto batch
        batch_dp = DataProto.from_dict(
            tensors={"input_ids": torch.zeros(1, 1, dtype=torch.long)},
            non_tensors={
                "raw_prompt": [prompt_str],
                "instance_id": [instance_id],
                "task_name": [self.task.__class__.__name__],
            },
        )

        runtime_url = os.getenv("BIOMNI_RUNTIME_URL") or self.cfg.tools.get("biomni_runtime_url", "http://localhost:8000")

        grp = BiomniCodeActAgentGroup(
            batch=batch_dp,
            num_trajectories=1,
            infer_engine=engine_adapter,
            tokenizer=self.tokenizer,
            sampling_params=copy.deepcopy(self.cfg.sampling_params),
            runtime_url=runtime_url,
        )

        dp = grp.run()

        # Extract first (and only) trajectory output
        msgs_out = dp.non_tensor_batch["messages"][0]
        solution = dp.non_tensor_batch.get("solution", [None])[0]
        iterations = dp.non_tensor_batch.get("iterations", [None])[0]

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
