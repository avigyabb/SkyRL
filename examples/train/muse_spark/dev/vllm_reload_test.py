"""Does reloading the *same* checkpoint into vLLM change its outputs? Isolates vLLM reload bugs.

Usage: python vllm_reload_test.py <model> <tp> [names_substring_filter]
Prompt logprobs on fixed text before and after `reload_weights(weights_path=model)`.
"""

import sys

from vllm import LLM, SamplingParams

TEXT = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."


def score(llm):
    out = llm.generate([TEXT], SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))[0]
    lps = [next(iter(d.values())).logprob for d in out.prompt_logprobs[1:]]
    return lps


def main():
    model, tp = sys.argv[1], int(sys.argv[2])
    ext = sys.argv[3] if len(sys.argv) > 3 else None
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        language_model_only=True,
        max_model_len=4096,
        **({"worker_extension_cls": ext} if ext else {}),
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    before = score(llm)
    llm.collective_rpc("reload_weights", kwargs={"weights_path": model})
    after = score(llm)
    diff = [abs(a - b) for a, b in zip(before, after)]
    print(
        f"RELOAD before={sum(before)/len(before):.4f} after={sum(after)/len(after):.4f} "
        f"|diff| mean={sum(diff)/len(diff):.4f} max={max(diff):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
