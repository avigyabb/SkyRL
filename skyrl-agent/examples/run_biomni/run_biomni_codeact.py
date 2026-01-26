import os
import asyncio
from pathlib import Path
from transformers import AutoTokenizer

from skyrl_agent import AutoAgentRunner


USER_PROMPT = """Your task is to identify likely causal genes within a locus for a given GWAS phenotype. From the list, provide the most likely causal gene. 
Identify the causal gene.
GWAS phenotype: Iron status biomarkers (iron levels)
Genes in locus: {ACHE},{ACTL6B},{AGFG2},{ENSG00000289690},{ENSG00000289760},{EPHB4},{EPO},{FBXO24},{GAL3ST4},{GIGYF1},{GNB2},{GPC2},{LAMTOR4},{LRCH4},{MEPCE},{MOSPD3},{MUC12},{MUC17},{MUC3A},{NYAP1},{PCOLCE},{PILRA},{PILRB},{POP7},{PPP1R35},{PVRIG},{SAP25},{SERPINE1},{SLC12A9},{SPACDR},{SPDYE3},{SRRT},{STAG3},{TFR2},{TRAPPC14},{TRIM56},{TRIP6},{TSC22D4},{UFSP1},{ZAN},{ZCWPW1}.
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
            "instance_id": 309,
            # minimal reward payload expected by GeneralReactTask.evaluate_result
            "reward_model": {"ground_truth": "TFR2"},
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