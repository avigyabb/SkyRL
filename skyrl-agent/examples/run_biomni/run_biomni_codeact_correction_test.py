"""
Dry-run test for the LLM correction generation system.

Tests that the reward adapter can:
1. Identify format errors in agent trajectories
2. Generate corrections that fix common 8B model format issues:
   - Multiple </think> tokens
   - Missing </think> token
   - Hallucinated observations in the same message
   - Incorrect tag usage
3. Parse corrections back to valid turn indices

Usage (inside Docker container):
    export ANTHROPIC_API_KEY=...
    export GENERATE_CORRECTIONS=true
    python run_biomni_codeact_correction_test.py
"""

import os
import sys
import json

os.environ.setdefault("OPENAI_API_KEY", "sc")
os.environ.setdefault("GENERATE_CORRECTIONS", "true")

from skyrl_agent.tasks.biomni_rubric_reward_adapter import (
    BiomniRubricRewardAdapter,
    format_messages_to_text,
)


SYNTHETIC_TRAJECTORIES = {
    "multiple_think_close": {
        "description": "Agent produces multiple </think> tokens",
        "messages": [
            {"role": "system", "content": "You are a biomedical agent."},
            {"role": "user", "content": "Identify the causal gene for iron deficiency from: TFR2, ACHE, EPO."},
            {"role": "assistant", "content": "<think>I need to look up genes related to iron.</think>\n</think>\n<execute>\nimport pandas as pd\nprint('hello')\n</execute>"},
            {"role": "user", "content": "<observation>\nhello\n</observation>"},
            {"role": "assistant", "content": "<think>The answer is TFR2.</think>\n<solution>TFR2</solution>"},
        ],
    },
    "missing_think_close": {
        "description": "Agent missing </think> token",
        "messages": [
            {"role": "system", "content": "You are a biomedical agent."},
            {"role": "user", "content": "Identify the causal gene for iron deficiency from: TFR2, ACHE, EPO."},
            {"role": "assistant", "content": "<think>I should search for this gene\n<execute>\nprint('searching...')\n</execute>"},
            {"role": "user", "content": "<observation>\nsearching...\n</observation>"},
            {"role": "assistant", "content": "<think>Based on my analysis, the answer is TFR2.</think>\n<solution>TFR2</solution>"},
        ],
    },
    "hallucinated_observation": {
        "description": "Agent hallucinating observations within its own message",
        "messages": [
            {"role": "system", "content": "You are a biomedical agent."},
            {"role": "user", "content": "Identify the causal gene for iron deficiency from: TFR2, ACHE, EPO."},
            {"role": "assistant", "content": "<think>Let me check the gene database.</think>\n<execute>\nimport requests\nresult = requests.get('http://api.example.com/genes')\nprint(result.json())\n</execute>\nOutput:\n{'genes': ['TFR2', 'EPO']}\n<think>Now I see TFR2 is involved in iron metabolism.</think>\n<execute>\nprint('TFR2 confirmed')\n</execute>"},
            {"role": "user", "content": "<observation>\nTFR2 confirmed\n</observation>"},
            {"role": "assistant", "content": "<think>The causal gene is TFR2.</think>\n<solution>TFR2</solution>"},
        ],
    },
    "wrong_tag_for_final": {
        "description": "Agent uses <execute> for final answer instead of <solution>",
        "messages": [
            {"role": "system", "content": "You are a biomedical agent."},
            {"role": "user", "content": "Identify the causal gene for iron deficiency from: TFR2, ACHE, EPO."},
            {"role": "assistant", "content": "<think>Let me search.</think>\n<execute>\nprint('TFR2')\n</execute>"},
            {"role": "user", "content": "<observation>\nTFR2\n</observation>"},
            {"role": "assistant", "content": "<think>The answer is TFR2.</think>\n<execute>\nprint('Final answer: TFR2')\n</execute>"},
        ],
    },
    "valid_trajectory": {
        "description": "A valid trajectory (should generate 0 corrections or quality-focused ones)",
        "messages": [
            {"role": "system", "content": "You are a biomedical agent."},
            {"role": "user", "content": "Identify the causal gene for iron deficiency from: TFR2, ACHE, EPO."},
            {"role": "assistant", "content": "<think>I need to investigate which gene is most related to iron metabolism. TFR2 (Transferrin Receptor 2) is known to be involved in iron homeostasis.</think>\n<execute>\nimport gget\nresult = gget.info(['TFR2'])\nprint(result)\n</execute>"},
            {"role": "user", "content": "<observation>\nTFR2 is involved in iron homeostasis...\n</observation>"},
            {"role": "assistant", "content": "<think>Based on my research, TFR2 is the transferrin receptor 2 gene, which plays a crucial role in iron homeostasis. Mutations in TFR2 cause hereditary hemochromatosis type 3, directly linking it to iron status biomarkers.</think>\n<solution>TFR2</solution>"},
        ],
    },
}


def test_per_turn_format_check():
    """Test the per-turn format check identifies errors correctly."""
    print("\n" + "=" * 70)
    print("TEST 1: Per-Turn Format Check")
    print("=" * 70)

    for name, case in SYNTHETIC_TRAJECTORIES.items():
        results = BiomniRubricRewardAdapter._per_turn_format_check(case["messages"])
        ft = BiomniRubricRewardAdapter._validate_format(case["messages"])
        print(f"\n  [{name}] ({case['description']})")
        print(f"    ft_reward: {ft}")
        for r in results:
            status = "VALID" if r["is_valid"] else f"INVALID: {r['error']}"
            print(f"    msg_idx={r['msg_index']} is_last={r['is_last']}: {status}")

    print("\n  [PASS] Per-turn format check completed.")


def test_turn_mapping():
    """Test turn mapping from labels to message indices."""
    print("\n" + "=" * 70)
    print("TEST 2: Turn Mapping")
    print("=" * 70)

    for name, case in SYNTHETIC_TRAJECTORIES.items():
        mapping = BiomniRubricRewardAdapter._build_turn_mapping(case["messages"])
        print(f"\n  [{name}]: {mapping}")

    print("\n  [PASS] Turn mapping completed.")


def test_correction_generation():
    """Test correction generation on synthetic trajectories with known errors."""
    print("\n" + "=" * 70)
    print("TEST 3: Correction Generation (LLM call)")
    print("=" * 70)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [SKIP] ANTHROPIC_API_KEY not set. Skipping LLM correction test.")
        return False

    BiomniRubricRewardAdapter._ensure_initialized()

    test_cases = ["multiple_think_close", "missing_think_close", "hallucinated_observation", "wrong_tag_for_final"]
    all_passed = True

    for name in test_cases:
        case = SYNTHETIC_TRAJECTORIES[name]
        messages = case["messages"]
        trajectory_text = format_messages_to_text(messages)
        ft_reward = BiomniRubricRewardAdapter._validate_format(messages)

        print(f"\n  [{name}] ({case['description']})")
        print(f"    ft_reward: {ft_reward}")

        rubric_results = {
            "rubric_weaknesses": [f"Format error: {case['description']}"],
            "rubric_rationale": f"The agent has a format issue: {case['description']}",
        }

        corrections = BiomniRubricRewardAdapter._generate_corrections(
            messages=messages,
            trajectory_text=trajectory_text,
            rubric_results=rubric_results,
            ft_reward=ft_reward,
            max_corrections=3,
        )

        print(f"    Generated {len(corrections)} correction(s):")
        for i, corr in enumerate(corrections):
            print(f"      [{i}] target: {corr['target_label']} (msg_idx={corr['assistant_msg_index']})")
            text = corr["correction_text"]
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"          text: {preview}")

            per_turn_check = BiomniRubricRewardAdapter._per_turn_format_check(messages)
            is_last = any(
                r["msg_index"] == corr["assistant_msg_index"] and r["is_last"]
                for r in per_turn_check
            )
            single_msg = [{"role": "assistant", "content": corr["correction_text"]}]
            from skyrl_agent.tasks.biomni_rubric_reward_adapter import BiomniRubricRewardAdapter as adapter
            check_msgs = [{"role": "assistant", "content": corr["correction_text"]}]
            low = corr["correction_text"].lower().rstrip()
            has_think = low.lstrip().startswith("<think>")
            has_one_think = low.count("<think>") == 1 and low.count("</think>") == 1
            ends_ok = low.endswith("</execute>") or low.endswith("</solution>")
            format_ok = has_think and has_one_think and ends_ok
            print(f"          correction_format_valid: {format_ok}")
            if not format_ok:
                all_passed = False
                print(f"          [WARN] Correction itself has format issues!")

        if len(corrections) == 0 and ft_reward == 0.0:
            print(f"    [WARN] No corrections generated for a format-failing trajectory!")
            all_passed = False

    if all_passed:
        print(f"\n  [PASS] All correction tests passed.")
    else:
        print(f"\n  [WARN] Some corrections had issues. Review above.")
    return all_passed


def test_full_compute_rewards_with_corrections():
    """Test compute_rewards with generate_corrections=True on a synthetic trajectory."""
    print("\n" + "=" * 70)
    print("TEST 4: Full compute_rewards with corrections")
    print("=" * 70)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [SKIP] ANTHROPIC_API_KEY not set.")
        return False

    case = SYNTHETIC_TRAJECTORIES["hallucinated_observation"]
    messages = case["messages"]

    print(f"  Testing: {case['description']}")

    result = BiomniRubricRewardAdapter.compute_rewards(
        instance={"task_name": "gwas_causal_gene_opentargets", "instance_id": 309},
        solution=None,
        messages=messages,
        instance_id=309,
        task_name="gwas_causal_gene_opentargets",
        generate_corrections=True,
        max_corrections=5,
    )

    print(f"  score: {result['score']}")
    print(f"  gt_reward: {result['gt_reward']}")
    print(f"  rubric_reward: {result['rubric_reward']}")
    print(f"  ft_reward: {result['ft_reward']}")
    print(f"  corrections: {len(result.get('corrections', []))}")

    for i, corr in enumerate(result.get("corrections", [])):
        print(f"    [{i}] {corr['target_label']} -> msg_idx={corr['assistant_msg_index']}")
        print(f"        preview: {corr['correction_text'][:150]}...")

    print(f"\n  [PASS] Full compute_rewards with corrections completed.")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("LLM Correction Generation - Dry Run Test")
    print("=" * 70)

    test_per_turn_format_check()
    test_turn_mapping()

    if os.environ.get("ANTHROPIC_API_KEY"):
        test_correction_generation()
        test_full_compute_rewards_with_corrections()
    else:
        print("\n[SKIP] ANTHROPIC_API_KEY not set. Skipping LLM-based tests.")
        print("Set ANTHROPIC_API_KEY to test correction generation.")

    print("\n" + "=" * 70)
    print("Dry run complete!")
    print("=" * 70)
