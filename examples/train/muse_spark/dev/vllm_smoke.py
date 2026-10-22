"""Plain vLLM load + greedy generate on a Muse Spark checkpoint. Usage: python vllm_smoke.py <model> <tp>"""

import sys

from vllm import LLM, SamplingParams


def main():
    model, tp = sys.argv[1], int(sys.argv[2])
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        language_model_only=True,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    prompts = ["Question: What is 17 * 3? Answer:", "The capital of France is"]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=16, logprobs=1))
    for out in outs:
        lps = [next(iter(lp.values())).logprob for lp in out.outputs[0].logprobs]
        print(f"VLLM_OUT {out.outputs[0].text!r} mean_lp={sum(lps) / len(lps):.3f}", flush=True)


if __name__ == "__main__":
    main()
