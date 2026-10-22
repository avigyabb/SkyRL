"""HF transformers reference logprobs for a Muse Spark checkpoint (run in hf-venv).

Usage: python hf_reference.py <model_dir> <out.pt>
Saves {"input_ids": [B, S], "logprobs": [B, S-1] (fp32, of the next token), "texts": [...]}.
"""

import sys

import torch
from transformers import AutoTokenizer, MuseSparkForConditionalGeneration

TEXTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?",
    "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Once upon a time, in a small village nestled between two mountains, there lived an old clockmaker who",
]


def main():
    model_dir, out = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = MuseSparkForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    ).eval()
    results = {"texts": TEXTS, "input_ids": [], "logprobs": []}
    for text in TEXTS:
        ids = tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(input_ids=ids.to(model.device)).logits.float()
        logprobs = torch.log_softmax(logits[0, :-1], dim=-1).gather(-1, ids[0, 1:, None].to(logits.device))[:, 0]
        results["input_ids"].append(ids[0].cpu())
        results["logprobs"].append(logprobs.cpu())
        print(f"{len(ids[0])} tokens, mean logprob {logprobs.mean():.4f}", flush=True)
    torch.save(results, out)


if __name__ == "__main__":
    main()
