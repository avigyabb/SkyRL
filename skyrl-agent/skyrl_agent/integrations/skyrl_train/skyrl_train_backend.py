from typing import Any, List
from ..base import AsyncInferBackend, GeneratorOutput, GeneratorInput


class SkyRLBackend(AsyncInferBackend):
    def __init__(self, infer_engine, cfg: Any = None):
        self.client = infer_engine

    async def async_generate_prompts(self, prompts: Any, sampling_params: Any, **kwargs) -> List[str]:
        input_obj = {
            "prompts": [prompts],
            "session_ids": [kwargs.get("request_id", None)],
            "sampling_params": sampling_params,
        }
        output = await self.client.generate(input_obj)
        return output["responses"][0], output["stop_reasons"][0]

    async def async_generate_ids(self, input_ids: List[int], sampling_params: Any, **kwargs) -> List[str]:
        input_obj = {
            "prompt_token_ids": [input_ids],
            "session_ids": [kwargs.get("request_id", None)],
            "sampling_params": sampling_params,
        }
        output = await self.client.generate(input_obj)
        # todo(@csy) probably need to be finish_reason
        # https://github.com/vllm-project/vllm/blob/a0f8a7964694a6077689b242b5eca95de392d4bb/vllm/v1/engine/__init__.py#L22
        return output["responses"][0], output["stop_reasons"][0]


class SkyRLGeneratorOutput(GeneratorOutput):
    def __init__(self, result: Any):
        from skyrl_train.generators.utils import get_rollout_metrics  # type: ignore[import]

        # Add more skyrl-specific rollout metrics.
        assert "rollout_metrics" in result, "rollout_metrics should be in the result"
        skyrl_rollout_metrics = get_rollout_metrics(result["response_ids"], result["rewards"])
        result["rollout_metrics"].update(skyrl_rollout_metrics)
        self.result = result


class SkyRLGeneratorInput(GeneratorInput):
    def __init__(self, input_batch: Any):
        env_extras = input_batch.get("env_extras") or []
        prompts = input_batch.get("prompts") or []

        if env_extras and len(env_extras) != len(prompts):
            raise ValueError(
                f"env_extras length ({len(env_extras)}) must match prompts length ({len(prompts)}) for SkyRLGeneratorInput"
            )

        merged_batch = []
        for idx, prompt in enumerate(prompts):
            extras = env_extras[idx] if idx < len(env_extras) else None
            payload = dict(extras) if isinstance(extras, dict) else {}

            payload.setdefault("prompt", prompt)
            merged_batch.append(payload)

        self.input_batch = merged_batch if merged_batch else env_extras
