#!/usr/bin/env python3
"""
Standalone test script for verifying all 8 target tasks work correctly.

Tests for each task:
  1. get_example(0) -- loads data, returns valid prompt + answer
  2. output_class() -- Pydantic schema has valid JSON schema (catches list vs list[str] bugs)
  3. reward(0, mock_answer) -- reward computation works
  4. get_rubric(0, mock_output, "mock") -- rubric renders without errors
  5. get_demonstration(0) -- demo renders without errors

Run inside Docker:
  docker exec skyrl-train python3 /workspace/SkyRL/skyrl-agent/tests/test_8task_pipeline.py
"""

import sys
import json
import traceback

sys.path.insert(0, "/workspace/SkyRL/skyrl-agent")
sys.path.insert(0, "/workspace/SkyRL/skyrl-train")

BENCHMARK_ROOT = "/mnt/biomni_filestore/biomni/biomni_resources/benchmark/"

TASKS = {}
RESULTS = {}


def init_tasks():
    from skyrl_agent.agents.biomni_codeact.task.rare_disease_diagnosis import rare_disease_diagnosis
    from skyrl_agent.agents.biomni_codeact.task.gwas_variant_prioritization import gwas_variant_prioritization
    from skyrl_agent.agents.biomni_codeact.task.patient_gene_detection import patient_gene_detection
    from skyrl_agent.agents.biomni_codeact.task.screen_gene_retrieval import screen_gene_retrieval
    from skyrl_agent.agents.biomni_codeact.task.crispr_delivery import crispr_delivery
    from skyrl_agent.agents.biomni_codeact.task.gwas_causal_gene import gwas_causal_gene

    TASKS["rare_disease_diagnosis"] = rare_disease_diagnosis(BENCHMARK_ROOT)
    TASKS["gwas_variant_prioritization"] = gwas_variant_prioritization(BENCHMARK_ROOT, num_samples=10000)
    TASKS["patient_gene_detection"] = patient_gene_detection(BENCHMARK_ROOT, num_samples=10000)
    TASKS["screen_gene_retrieval"] = screen_gene_retrieval()
    TASKS["crispr_delivery"] = crispr_delivery(num_samples=10000)
    for ds in ["opentargets", "pharmaprojects", "gwas_catalog"]:
        TASKS[f"gwas_causal_gene_{ds}"] = gwas_causal_gene(
            path=BENCHMARK_ROOT, dataset=ds, num_samples=100000
        )


def check(task_name, check_name, fn):
    try:
        result = fn()
        print(f"  [{check_name}] OK")
        return True, result
    except Exception as e:
        print(f"  [{check_name}] FAILED: {e}")
        traceback.print_exc()
        return False, None


def test_task(task_name, task):
    print(f"\n{'='*60}")
    print(f"Testing: {task_name} ({len(task)} examples)")
    print(f"{'='*60}")

    all_ok = True

    # 1. get_example
    ok, ex = check(task_name, "get_example(0)", lambda: task.get_example(0))
    all_ok &= ok
    if ok:
        print(f"    prompt length: {len(str(ex.get('prompt', '')))} chars")
        answer_key = "answer" if "answer" in ex else "target_gene" if "target_gene" in ex else "?"
        print(f"    answer key: {answer_key}, value: {str(ex.get(answer_key, 'N/A'))[:80]}")

    # 2. output_class -- validate JSON schema
    def check_schema():
        oc = task.output_class()
        schema = oc.model_json_schema()
        props = schema.get("properties", {})
        for prop_name, prop_def in props.items():
            has_type = "type" in prop_def or "anyOf" in prop_def or "oneOf" in prop_def or "allOf" in prop_def
            if not has_type:
                raise ValueError(
                    f"Property '{prop_name}' has no 'type' field in JSON schema. "
                    f"Schema: {json.dumps(prop_def, indent=2)}"
                )
        return schema

    ok, schema = check(task_name, "output_class schema", check_schema)
    all_ok &= ok

    # 3. reward -- with a mock answer
    def check_reward():
        if task_name == "crispr_delivery":
            return task.reward(0, "c")
        elif task_name == "rare_disease_diagnosis":
            return task.reward(0, {"disease_name": "test", "OMIM_ID": "000000"})
        elif task_name == "patient_gene_detection":
            return task.reward(0, {"causal_genes": ["FAKE_GENE"]})
        elif task_name.startswith("gwas_causal_gene"):
            return task.reward(0, "FAKE_GENE")
        elif task_name == "gwas_variant_prioritization":
            return task.reward(0, "rs000000")
        elif task_name == "screen_gene_retrieval":
            return task.reward(0, "FAKE_GENE")
        else:
            return task.reward(0, "mock")

    ok, reward = check(task_name, "reward(0, mock)", check_reward)
    all_ok &= ok
    if ok:
        print(f"    reward value: {reward}")

    # 4. get_rubric
    def check_rubric():
        mock_output = "A" if task_name == "crispr_delivery" else "mock_output"
        rubric = task.get_rubric(0, mock_output, "mock trajectory text")
        assert len(rubric) > 100, f"Rubric too short: {len(rubric)} chars"
        assert "CRITERION 1" in rubric or "Criterion 1" in rubric or "criterion 1" in rubric.lower(), "Missing Criterion 1"
        return len(rubric)

    ok, rubric_len = check(task_name, "get_rubric(0, mock, mock)", check_rubric)
    all_ok &= ok
    if ok:
        print(f"    rubric length: {rubric_len} chars")

    # 5. get_demonstration
    def check_demo():
        assert hasattr(task, 'get_demonstration'), "Missing get_demonstration method"
        demo = task.get_demonstration(0)
        assert len(demo) > 50, f"Demo too short: {len(demo)} chars"
        assert "<execute>" in demo, "Demo missing <execute> blocks"
        return len(demo)

    ok, demo_len = check(task_name, "get_demonstration(0)", check_demo)
    all_ok &= ok
    if ok:
        print(f"    demo length: {demo_len} chars")

    return all_ok


def main():
    print("Initializing tasks...")
    try:
        init_tasks()
    except Exception as e:
        print(f"FATAL: Failed to initialize tasks: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\nLoaded {len(TASKS)} tasks: {list(TASKS.keys())}")

    passed = 0
    failed = 0
    for name, task in TASKS.items():
        if test_task(name, task):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TASKS)} tasks")
    print(f"{'='*60}")

    if failed > 0:
        print("\nFAILED TASKS:")
        sys.exit(1)
    else:
        print("\nALL TASKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
