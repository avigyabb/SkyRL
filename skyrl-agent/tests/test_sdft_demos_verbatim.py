"""
Verbatim demo tester v2: improved answer matching.
"""
import requests, re, time, sys, importlib, importlib.util

HOST = "http://10.138.0.3:8000"

spec_bt = importlib.util.spec_from_file_location('base_task', '/workspace/SkyRL/skyrl-agent/skyrl_agent/agents/biomni_codeact/task/base_task.py')
bt = importlib.util.module_from_spec(spec_bt); spec_bt.loader.exec_module(bt)
sys.modules['skyrl_agent.agents.biomni_codeact.task.base_task'] = bt

def check_answer_in_output(answer, all_output):
    """Smarter answer matching that handles different formats."""
    if answer is None:
        return True
    
    out_lower = all_output.lower()
    
    if isinstance(answer, str):
        if answer.lower() in out_lower:
            return True
        # Try key words (split on spaces, check if most words appear)
        words = [w for w in answer.lower().split() if len(w) > 3]
        if words and sum(1 for w in words if w in out_lower) >= len(words) * 0.6:
            return True
    
    elif isinstance(answer, list):
        # For list answers like ['ENSG00000210194'], check each element
        for item in answer:
            if str(item).lower() in out_lower:
                return True
    
    elif isinstance(answer, dict):
        for v in answer.values():
            if str(v).lower() in out_lower:
                return True
    
    return False

TASKS = [
    ("gwas_causal_gene", "/workspace/SkyRL/skyrl-agent/skyrl_agent/agents/biomni_codeact/task/gwas_causal_gene.py",
     "gwas_causal_gene", {"path": "/mnt/biomni_filestore/biomni/biomni_resources/benchmark", "dataset": "opentargets", "num_samples": 1000},
     lambda ex: ex["answer"], [0, 5]),
    ("gwas_causal_gene_gwas_catalog", "/workspace/SkyRL/skyrl-agent/skyrl_agent/agents/biomni_codeact/task/gwas_causal_gene.py",
     "gwas_causal_gene", {"path": "/mnt/biomni_filestore/biomni/biomni_resources/benchmark", "dataset": "gwas_catalog", "num_samples": 1000},
     lambda ex: ex["answer"], [0]),
    ("rare_disease_diagnosis", "/workspace/SkyRL/skyrl-agent/skyrl_agent/agents/biomni_codeact/task/rare_disease_diagnosis.py",
     "rare_disease_diagnosis", {"path": "/mnt/biomni_filestore/biomni/biomni_resources/benchmark"},
     lambda ex: ex["answer"], [0, 3]),
    ("patient_gene_detection", "/workspace/SkyRL/skyrl-agent/skyrl_agent/agents/biomni_codeact/task/patient_gene_detection.py",
     "patient_gene_detection", {"path": "/mnt/biomni_filestore/biomni/biomni_resources/benchmark", "num_samples": 1000},
     lambda ex: ex["answer"], [0, 2]),
    ("screen_gene_retrieval", "/workspace/SkyRL/skyrl-agent/skyrl_agent/agents/biomni_codeact/task/screen_gene_retrieval.py",
     "screen_gene_retrieval", {},
     lambda ex: ex["target_gene"], [0, 1]),
    ("crispr_delivery", "/workspace/SkyRL/skyrl-agent/skyrl_agent/agents/biomni_codeact/task/crispr_delivery.py",
     "crispr_delivery", {"num_samples": 1000},
     lambda ex: None, [0, 1]),
]

def extract_execute_blocks(demo_text):
    return re.findall(r'<execute>\n?(.*?)\n?</execute>', demo_text, re.DOTALL)

def run_block(sid, ck, code, timeout=600):
    r = requests.post(f"{HOST}/execute", json={"code": code, "session_id": sid}, cookies=ck, timeout=timeout)
    out = r.json().get("output", "")
    ok = r.status_code == 200 and "Traceback" not in out
    return out, ok

results = []

for task_name, path, cls_name, kwargs, answer_fn, indices in TASKS:
    spec = importlib.util.spec_from_file_location(task_name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cls = getattr(mod, cls_name)
    task = cls(**kwargs)
    
    for idx in indices:
        ex = task.get_example(idx)
        demo = task.get_demonstration(idx)
        answer = answer_fn(ex)
        blocks = extract_execute_blocks(demo)
        
        print(f"\n{'='*80}")
        print(f"TASK: {task_name} | Instance: {idx} | Answer: {answer}")
        print(f"Demo: {len(demo)} chars, {len(blocks)} execute blocks")
        print(f"{'='*80}")
        
        r = requests.post(f"{HOST}/start_runtime", timeout=30)
        sid = r.json()["session_id"]
        ck = r.cookies
        
        all_ok = True
        all_output = ""
        
        for bi, block in enumerate(blocks):
            t0 = time.time()
            out, ok = run_block(sid, ck, block, timeout=600)
            elapsed = time.time() - t0
            
            print(f"  [{'PASS' if ok else 'FAIL'}] Block {bi+1}/{len(blocks)} ({elapsed:.1f}s)")
            if not ok:
                all_ok = False
                print(f"    ERROR: {out[:400]}")
            else:
                print(f"    Output: {out[:200]}...")
            all_output += out + "\n"
        
        answer_found = check_answer_in_output(answer, all_output)
        
        requests.post(f"{HOST}/delete_runtime", json={"session_id": sid}, cookies=ck)
        
        verdict = "PASS" if all_ok and answer_found else "BLOCKS_OK_NO_ANSWER" if all_ok else "FAIL"
        print(f"\n  >>> VERDICT: {verdict}")
        if not answer_found and answer:
            print(f"  >>> Answer '{answer}' NOT found in output")
        
        results.append((task_name, idx, verdict, len(blocks)))

print(f"\n\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
for task_name, idx, verdict, nblocks in results:
    print(f"  [{verdict}] {task_name} idx={idx} ({nblocks} blocks)")
pass_count = sum(1 for _, _, v, _ in results if v == "PASS")
print(f"\n  {pass_count}/{len(results)} passed")
