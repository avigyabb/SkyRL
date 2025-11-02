import os
import asyncio
from pathlib import Path
from transformers import AutoTokenizer

from skyrl_agent import AutoAgentRunner


def build_demo_dataset():
    """
    Minimal demo batch for Biomni CodeAct.
    Each item provides a ChatML-style prompt list and an optional data_source.
    """
    return [
        {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Use <execute>...</execute> to run a short Python snippet that prints 1+1, "
                        "then return the sum inside <solution>...</solution>."
                    ),
                }
            ],
            "data_source": "codegen.demo",
        }
    ]


if __name__ == "__main__":
    # Biomni runtime URL (the code execution server)
    os.environ.setdefault("BIOMNI_RUNTIME_URL", "http://localhost:8001")
    os.environ["OPENAI_API_KEY"] = "sc"  # dummy key, assumes an unath'ed vLLM service running locally

    # Model and tokenizer should match your YAML config's generator.backend_config.model_name
    model_name = os.getenv("BIOMNI_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Point to the Biomni CodeAct YAML
    yaml_path = str(Path(__file__).parent / "biomni_codeact.yaml")

    # Build a tiny demo dataset
    dataset = build_demo_dataset()

    # Create the runner from YAML; backend is constructed from YAML (openai_server)
    agent_generator = AutoAgentRunner.from_task(
        yaml_path,
        infer_engine=None,  # backend built from YAML
        tokenizer=tokenizer,
    )

    output = asyncio.run(agent_generator.run(dataset))
    # Print aggregate rewards if available
    rewards = output.get("rewards") if isinstance(output, dict) else None
    if rewards is not None:
        print("rewards:", rewards)
    else:
        print(output)