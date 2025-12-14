import os
import asyncio
from pathlib import Path
from transformers import AutoTokenizer

from skyrl_agent import AutoAgentRunner


USER_PROMPT = """Your task is to identify likely causal genes within a locus for a given GWAS phenotype. From the list, provide only the likely causal gene (matching one of the given genes). 
Identify the causal gene.
GWAS phenotype: Nausea
Genes in locus: {ANKK1},{CLDN25},{DRD2},{HTR3A},{HTR3B},{NNMT},{TMPRSS5},{TTC12},{USP28},{ZBTB16},{ZW10}
You must output only the name of the gene in your final solution, e.g., <solution>BRCA1</solution>, with no other text between the <solution> and </solution> tags."""

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
                    "content": USER_PROMPT,
                }
            ],
            "data_source": "demo.biomni",
            # minimal reward payload expected by GeneralReactTask.evaluate_result
            "reward_model": {"ground_truth": "DRD2"},
            "extra_info": {},
        }
    ]


if __name__ == "__main__":
    # Biomni runtime URL (the code execution server)
    os.environ.setdefault("BIOMNI_RUNTIME_URL", "http://localhost:8000")
    os.environ["OPENAI_API_KEY"] = "sc"  # dummy key, assumes an unath'ed vLLM service running locally

    # Model and tokenizer should match your YAML config's generator.backend_config.model_name
    model_name = os.getenv("BIOMNI_MODEL", "biomni/Biomni-R0-32B-Preview")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Point to the Biomni CodeAct YAML
    yaml_path = str(Path(__file__).parent / "biomni_codeact.yaml")

    # Build a tiny demo dataset
    dataset = build_demo_dataset()
    
    print(dataset[0]["prompt"])

    # Create the runner from YAML; backend is constructed from YAML (openai_server)
    agent_generator = AutoAgentRunner.from_task(
        yaml_path,
        infer_engine=None,  # backend built from YAML
        tokenizer=tokenizer,
    )

    output = asyncio.run(agent_generator.run(dataset))
    # print(output)
    # Print aggregate rewards if available
    rewards = output.get("rewards") if isinstance(output, dict) else None
    if rewards is not None:
        print("rewards:", rewards)
    else:
        print(output)