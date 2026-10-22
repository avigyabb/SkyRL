"""Plain vLLM on the full Muse Spark checkpoint across nodes (Ray executor), no SkyRL.

Usage: python vllm_full_smoke.py <model> <tp> <pp>
"""

import sys

from vllm import LLM, SamplingParams


def main():
    model, tp, pp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        distributed_executor_backend="ray",
        language_model_only=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    prompts = ["Question: What is 17 * 3? Answer:", "The capital of France is"]
    for out in llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=24, logprobs=1)):
        lps = [next(iter(lp.values())).logprob for lp in out.outputs[0].logprobs]
        print(f"VLLM_FULL {out.outputs[0].text!r} mean_lp={sum(lps) / len(lps):.3f}", flush=True)


if __name__ == "__main__":
    main()
