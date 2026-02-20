"""
Sanity check script for Biomni CodeAct with LLM-based rubric rewards.

This script runs a full agent trajectory and evaluates it using:
- gt_reward: ground truth from task.reward() (inferred from task_name + instance_id)
- rubric_reward: LLM critic evaluation (max 5, normalized from 50)
- ft_reward: format validation (max 1)
Total reward = gt_reward + rubric_reward + ft_reward (max 7)

Prerequisites:
- ANTHROPIC_API_KEY must be set for the LLM critic
- Biomni runtime server running (default: http://localhost:8000)
- vLLM/OpenAI-compatible inference server running (default: http://localhost:30000)

Usage:
    python run_biomni_codeact_rubric.py
"""

import os
import sys
import asyncio
from pathlib import Path
from transformers import AutoTokenizer

from skyrl_agent import AutoAgentRunner


class StandaloneBatch(list):
    """
    Wrapper for standalone batch that supports both list iteration and .get() for batch_metadata.
    
    The AgentRunner.run() method expects input_batch to:
    1. Be iterable (for batch processing)
    2. Have .get("batch_metadata") method (for global_step extraction)
    
    During training, input_batch is a dict with batch_metadata.
    For standalone scripts, we use this wrapper to provide both interfaces.
    """
    def get(self, key, default=None):
        if key == "batch_metadata":
            return None  # No metadata for standalone runs
        return default


# Example prompt for gwas_causal_gene task (instance_id=309)
USER_PROMPT = """Your task is to identify likely causal genes within a locus for a given GWAS phenotype. From the list, provide the most likely causal gene. 
Identify the causal gene.
GWAS phenotype: Iron status biomarkers (iron levels)
Genes in locus: {ACHE},{ACTL6B},{AGFG2},{ENSG00000289690},{ENSG00000289760},{EPHB4},{EPO},{FBXO24},{GAL3ST4},{GIGYF1},{GNB2},{GPC2},{LAMTOR4},{LRCH4},{MEPCE},{MOSPD3},{MUC12},{MUC17},{MUC3A},{NYAP1},{PCOLCE},{PILRA},{PILRB},{POP7},{PPP1R35},{PVRIG},{SAP25},{SERPINE1},{SLC12A9},{SPACDR},{SPDYE3},{SRRT},{STAG3},{TFR2},{TRAPPC14},{TRIM56},{TRIP6},{TSC22D4},{UFSP1},{ZAN},{ZCWPW1}."""


def build_demo_dataset():
    """
    Demo batch for Biomni CodeAct with rubric rewards.
    Uses gwas_causal_gene_opentargets task for testing.
    
    The rubric reward adapter will infer ground truth from task_name + instance_id,
    so no reward_model field is needed.
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
            "task_name": "gwas_causal_gene_opentargets",  # Task name for rubric lookup
            "instance_id": 309,  # Instance ID for task.get_example() - ground truth is TFR2
            "reward_model": {"ground_truth": "TFR2"},
            "extra_info": {},
        }
    ]


def run_trajectory():
    """
    Run a full agent trajectory with rubric-based evaluation.
    """
    print("\n" + "="*60)
    print("Biomni CodeAct with Rubric Rewards - Sanity Check")
    print("="*60 + "\n")
    
    # Check for ANTHROPIC_API_KEY
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set!")
        print("Please set ANTHROPIC_API_KEY before running rubric evaluation.")
        return False
    
    # Biomni runtime URL (the code execution server)
    os.environ.setdefault("BIOMNI_RUNTIME_URL", "http://localhost:8000")
    os.environ.setdefault("OPENAI_API_KEY", "sc")  # dummy key for vLLM

    # Model and tokenizer should match your YAML config's generator.backend_config.model_name
    model_name = os.getenv("BIOMNI_MODEL", "biomni/Biomni-R0-32B-Preview")
    print(f"Using model: {model_name}")
    print(f"Biomni runtime: {os.environ.get('BIOMNI_RUNTIME_URL')}")
    print(f"Critic model: {os.getenv('BIOMNI_CRITIC_MODEL', 'claude-sonnet-4-5')}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Point to the Biomni CodeAct Rubric YAML
    yaml_path = str(Path(__file__).parent / "biomni_codeact_rubric.yaml")
    print(f"Using config: {yaml_path}")

    # Build the demo dataset (wrapped in StandaloneBatch for compatibility with AgentRunner.run())
    dataset = StandaloneBatch(build_demo_dataset())
    
    print(f"\nTask: {dataset[0]['task_name']}")
    print(f"Instance ID: {dataset[0]['instance_id']}")
    print(f"Expected ground truth: TFR2")
    print(f"\nPrompt:\n{dataset[0]['prompt'][0]['content'][:300]}...")

    # Create the runner from YAML
    agent_generator = AutoAgentRunner.from_task(
        yaml_path,
        infer_engine=None,  # backend built from YAML
        tokenizer=tokenizer,
    )

    print("\n" + "-"*60)
    print("Running agent trajectory...")
    print("-"*60 + "\n")
    
    output = asyncio.run(agent_generator.run(dataset))
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if isinstance(output, dict):
        rewards = output.get("rewards")
        if rewards is not None:
            print(f"\nRewards: {rewards}")
        
        rollout_metrics = output.get("rollout_metrics", {})
        if rollout_metrics:
            print("\nRollout Metrics:")
            print(f"  Raw Reward (total): {rollout_metrics.get('rollout_metrics/raw_reward', 'N/A')}")
            print(f"  GT Reward: {rollout_metrics.get('rollout_metrics/gt_reward', 'N/A')}")
            print(f"  Rubric Reward: {rollout_metrics.get('rollout_metrics/rubric_reward', 'N/A')}")
            print(f"  Format Reward: {rollout_metrics.get('rollout_metrics/ft_reward', 'N/A')}")
            
            # Print other useful metrics if available
            if 'rollout_metrics/pass_at_n_percentage' in rollout_metrics:
                print(f"  Pass@N: {rollout_metrics['rollout_metrics/pass_at_n_percentage']}")
            if 'rollout_metrics/finish_tool_ratio' in rollout_metrics:
                print(f"  Finish Tool Ratio: {rollout_metrics['rollout_metrics/finish_tool_ratio']}")
    else:
        print(output)
    
    print("\n" + "="*60)
    print("✅ Sanity check completed!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = run_trajectory()
    if not success:
        exit(1)
