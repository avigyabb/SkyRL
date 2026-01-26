import sys, json, numpy as np, pandas as pd, requests
from pydantic import BaseModel, Field
import ast

from verl.workers.agentic.biomni.task.base_task import base_task

class rare_disease_diagnosis(base_task):
    def __init__(self, path = './data', num_samples = None):
        # data_path = os.path.join(path, 'rare_disease_diagnosis', 'mygene.json')
        data_path = '/dfs/user/kexinh/BioAgentOS/data/mygene.json'
        data = []
        with open(data_path, "r") as file:
            for line in file:
                data.append(json.loads(line))

        if num_samples is None:
            self.data = pd.DataFrame(data)
        else:
            self.data = pd.DataFrame(data)[:num_samples]

        # Ensure the data contains all necessary columns
        required_columns = ['id', 'positive_phenotypes', 'all_candidate_genes', 'omim', 'disease_name', 'orpha_id']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Dataset is missing required column: {col}")

        self.query = []
        self.answer = []
        for _, row in self.data.iterrows():
            phenotypes = row['positive_phenotypes']
            candidate_genes = row['all_candidate_genes']
            disease_name = row['disease_name']
            omim_id = row['omim']

            self.query.append({
                "phenotypes": phenotypes,
                "candidate_genes": candidate_genes
            })
            self.answer.append({
                "disease_name": disease_name,
                "OMIM_ID": omim_id
            })

        self.task_description = """
Task: given a patient's phenotypes and a list of candidate genes, diagnose the rare disease that the patient has.
Phenotypes: {phenotype_list}
Candidate genes: {candidate_genes}

Select the most likely disease and justify your answer."""

        self.completion_checker = """
Given an answer and a solution, check if the answer is correct.

Answer: {answer}
Solution: {solution}

Return 'task completed' if the answer is correct, and 'task not completed' otherwise."""

    def __len__(self):
        return len(self.query)
        
    def get_example(self, index=None):
        if index is None:
            index = np.random.randint(len(self.query))
        
        q = self.query[index]
        a = self.answer[index]

        prompt = self.task_description.format(
            phenotype_list=', '.join(q['phenotypes']),
            candidate_genes=q['candidate_genes']
        )
            
        return {"prompt": prompt, "answer": a, "instance_id": index}

    def split(self, ratio=0.8, seed=42):
        np.random.seed(seed)
        indices = np.arange(len(self.query))
        np.random.shuffle(indices)
        split_idx = int(ratio * len(self.query))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices

    def reward(self, input, output):
        if not isinstance(output, dict):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                try:
                    output = ast.literal_eval(output)
                except (ValueError, SyntaxError):
                    # If both fail, return 0
                    return 0
        
        answer = self.get_example(input)['answer']
        
        print("Instance_id: ", input)
        print("Prompt: ", self.get_example(input)['prompt'])
        
        print("True answer:")
        print(answer)
        print(answer['OMIM_ID'])
        print("Predicted answer:")
        print(output)
        print(output.get('OMIM_ID'))
        
        return 1 if output.get('OMIM_ID') == answer['OMIM_ID'] else 0

    def get_iterator(self):
        for i in range(len(self.query)):
            yield self.get_example(i)

    def output_class(self):
        from pydantic import BaseModel, Field
        from typing import Optional
        
        class DiagnosisOutput(BaseModel):
            """A diagnosis for a rare disease."""
            
            disease_name: Optional[str] = Field(
                description="The name of the diagnosed rare disease, e.g., 'Marfan Syndrome'"
            )
            OMIM_ID: Optional[str] = Field(
                description="The OMIM ID of the diagnosed disease, e.g., '154700'"
            )
        
        return DiagnosisOutput

    def evaluate(self, response, ground_truth = None):
        from sklearn.metrics import accuracy_score
        if ground_truth is None:
            ground_truth = self.answer
        predicted = response
        correct = []
        results = []
        for pred, gt in zip(predicted, ground_truth):
            # Use the LLM-based completion checker to verify each prediction
            check_prompt = self.completion_checker.format(
                answer=json.dumps(pred),
                solution=json.dumps(gt)
            )
            # Assuming an LLM API call here; replace with the actual implementation
            result = self.call_llm_to_check(check_prompt)
            correct.append(result == 'task completed')
            results.append(result)
        
        accuracy = accuracy_score([1] * len(correct), correct)
        return {
            'completion_rate': accuracy,
            'num_of_tasks_completed': sum(correct),
            'num_of_total_tasks': len(correct),
            'results': results
        }

    def get_rubric(self, input, parsed_output, raw_output):
        """
        Rubric for rare_disease_diagnosis.
        - input: same as in reward(self, input, output)  (instance index)
        - parsed_output: same as output in reward(...); expected to be a dict-like object containing OMIM_ID and disease_name
        - raw_output: the model's raw text output (may include code blocks + observations if present)
        """

        import json, ast, re

        # =========================
        # Instance context (prompt + reference)
        # =========================
        ex = self.get_example(input)
        prompt = ex["prompt"]
        answer = ex["answer"] or {}
        gt_disease = str(answer.get("disease_name", "")).strip()
        gt_omim = answer.get("OMIM_ID", "")

        # Pull additional instance metadata when available (optional, for grader context only)
        gt_orpha = None
        try:
            if hasattr(self, "data") and input is not None and int(input) < len(self.data):
                gt_orpha = self.data.iloc[int(input)].get("orpha_id", None)
        except Exception:
            gt_orpha = None

        # =========================
        # Normalize parsed output (robust to strings / list-like)
        # =========================
        def _to_dict(x):
            if isinstance(x, dict):
                return x
            if isinstance(x, (list, tuple)) and len(x) > 0:
                return _to_dict(x[0])
            if isinstance(x, str):
                s = x.strip()
                if not s:
                    return {}
                try:
                    return json.loads(s)
                except Exception:
                    try:
                        y = ast.literal_eval(s)
                        return y if isinstance(y, dict) else {}
                    except Exception:
                        return {}
            return {}

        pred_obj = _to_dict(parsed_output)

        # Common fallbacks; do NOT overcomplicate
        pred_disease = str(
            pred_obj.get("disease_name", pred_obj.get("Disease", pred_obj.get("disease", "")))
        ).strip()
        pred_omim_raw = pred_obj.get("OMIM_ID", pred_obj.get("omim", pred_obj.get("OMIM", "")))

        def _norm_omim(x):
            if x is None:
                return ""
            s = str(x).strip()
            # capture digits from "OMIM:123456" / "123456" / "{'OMIM_ID': 123456}"-style
            m = re.search(r"(\d{3,9})", s)
            return m.group(1) if m else s

        def _norm_name(x):
            s = "" if x is None else str(x).strip().lower()
            s = re.sub(r"[^a-z0-9]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        gt_omim_norm = _norm_omim(gt_omim)
        pred_omim_norm = _norm_omim(pred_omim_raw)

        exact_omim_match = (str(pred_omim_raw).strip() == str(gt_omim).strip())
        normalized_omim_match = bool(gt_omim_norm) and (pred_omim_norm == gt_omim_norm)
        disease_name_match = bool(gt_disease) and (_norm_name(pred_disease) == _norm_name(gt_disease))

        parsed_display = parsed_output if parsed_output is not None else "No parsed output available"

        # =========================
        # Helpful context for grader (phenotypes + candidate genes)
        # =========================
        phenos, cand_genes = None, None
        try:
            q = self.query[int(input)]
            phenos = q.get("phenotypes", None)
            cand_genes = q.get("candidate_genes", None)
        except Exception:
            phenos, cand_genes = None, None

        fence = "```"

        rubric = f"""
    You are grading a biomedical agent’s answer for a rare disease diagnosis task.

Task context: The agent is given a patient’s phenotypes and a list of candidate genes. The agent must diagnose the MOST likely rare disease and justify the diagnosis.

The agent can interact with a set of tools, software, and databases in a remote environment through executing code blocks. Outputs or errors from the code blocks will be returned as Observations.

========================
USER QUERY (verbatim)
========================
{fence}text
{prompt}
{fence}

========================
AGENT FULL TRAJECTORY (verbatim)
========================
{fence}text
{raw_output}
{fence}

========================
PARSED OUTPUT (system-extracted)
========================
{parsed_display}

========================
INSTANCE CONTEXT (for your reference)
========================
- Phenotypes (as provided): {phenos if phenos is not None else "Unavailable"}
- Candidate genes (as provided): {cand_genes if cand_genes is not None else "Unavailable"}

========================
REFERENCE KEY FOR THIS INSTANCE (for correctness scoring)
========================
- Ground-truth diagnosis (disease name): {gt_disease if gt_disease else "Unavailable"}
- Ground-truth OMIM_ID: {gt_omim if gt_omim else "Unavailable"}
- (Optional) Ground-truth Orphanet ID (context only): {gt_orpha if gt_orpha else "Unavailable"}

========================
PREDICTION NORMALIZATION (pre-computed)
========================
- Predicted disease_name (from parsed output): {pred_disease if pred_disease else "EMPTY/NOT PROVIDED"}
- Predicted OMIM_ID (raw): {pred_omim_raw if str(pred_omim_raw).strip() else "EMPTY/NOT PROVIDED"}
- Predicted OMIM_ID (normalized digits): {pred_omim_norm if pred_omim_norm else "EMPTY/NOT PARSABLE"}
- Ground-truth OMIM_ID (normalized digits): {gt_omim_norm if gt_omim_norm else "EMPTY/NOT PARSABLE"}
- Exact OMIM string match: {str(exact_omim_match)}
- Normalized OMIM match (digits-equivalent): {str(normalized_omim_match)}
- Disease-name normalized match: {str(disease_name_match)}

============================================================
CRITERION 1 (0-20): Output grading (correctness + format)
============================================================

1A. Diagnosis correctness (0-15 points)
Award points using the reference key above:

- 15.0 points: Exact OMIM match (Exact OMIM string match = True).
- 12.0 points: Normalized OMIM match is True BUT exact OMIM match is False
  (i.e., correct OMIM numerically but formatting/type differences such as "OMIM:xxxxx" vs "xxxxx", int vs str).
- 7.5 points: OMIM does NOT match (normalized match False) OR OMIM missing, BUT disease-name normalized match is True.
- 0.0 points: Otherwise.

Notes:
- If the agent outputs multiple diseases/OMIM IDs, score based on the SINGLE final stated diagnosis.
- If the agent “hedges” between multiple diagnoses without a clear final pick, cap correctness at 7.5 even if one option matches.

1B. Format & reporting quality (0-5 points)
Unless the prompt demands a strict schema (it does not here), the agent should produce a readable Markdown report.
Award 1 point for each satisfied item:

+1: A clear “Final diagnosis” statement near the top that includes BOTH disease name and OMIM_ID (or explicitly says OMIM is unknown).
+1: A concise, scannable structure (headings/bullets) that separates: phenotype summary, gene/disease evidence, and conclusion.
+1: Traceability: factual claims about gene–disease links are backed by sources observable in the trajectory (e.g., OMIM/Orphanet/Monarch/GeneReviews/ClinGen/primary literature; URLs or tool outputs are referenced).
+1: Faithful summarization of the trajectory’s *actual* findings (no invented “results,” no claiming queries were run if they were not).
+1: Minimal extraneous content; avoids copying long irrelevant tool output; avoids unrelated background encyclopedia text.

Max total for Criterion 1 = 15 + 5 = 20.

============================================================
CRITERION 2 (0-10): Methodology / clinical diagnosis know-how
============================================================
This criterion rewards scientifically grounded, systematic diagnostic reasoning (phenotype handling -> mechanistic matching -> differential exclusion). Award points only if the item is satisfied based on evidence in the agent trajectory. If unclear, do not award.

Item 2.1: Phenotypic hierarchy and translation (2.0 points)
+2.0 if the agent explicitly distinguishes between 'cardinal' (key defining) and non-specific symptoms AND/OR translates raw patient descriptions into standardized terms (e.g., HPO, biochemical markers).
+1.0 if the agent lists all phenotypes as equally important keyword matches without weighting or standardization.
+0.0 if the agent ignores phenotype nuances, hallucinates symptoms not in the input, or misinterprets key clinical signs.

Item 2.2: Genotype-phenotype mapping and mechanistic coherence (3.0 points)
+3.0 if the agent performs a comprehensive overlap analysis (e.g., "Matches 5/6 key features") AND provides a mechanistic bridge connecting the gene function to the phenotype (e.g., "COL1A1 defect disrupts collagen structure -> bone fragility").
+1.5 if the agent states a match exists based on broad categories (e.g., "Both involve the heart") or statistical association without explaining the biological mechanism.
+0.0 if the agent misses obvious contradictions (e.g., diagnosing a skeletal dysplasia in a patient with normal bone growth) or relies on hallucinated associations.

Item 2.3: Handling of negative evidence (1.0 point)
+1.0 if the agent explicitly acknowledges and addresses expected symptoms that are *missing* in the patient (e.g., "Marfan is likely, but the lack of aortic dilation is atypical").
+0.0 if the agent ignores missing features to force a "perfect" fit or fails to note significant discrepancies.

Item 2.4: Differential exclusion strategy (2.0 points)
+2.0 if the agent provides specific, evidence-based reasons for rejecting *other* candidate genes (e.g., "Excluded Gene B due to recessive inheritance pattern vs family history" or "Gene C causes renal failure, absent here").
+1.0 if the agent simply states other genes are "less likely" without specific evidence or comparative analysis.
+0.0 if the agent discusses only the chosen diagnosis and ignores the rejected candidates entirely.

Item 2.5: Confirmatory validation and certainty calibration (2.0 points)
+2.0 if the agent proposes a concrete, high-value clinical or biochemical test to confirm the diagnosis (e.g., "Assay for enzyme X activity") AND expresses the conclusion with appropriate certainty (e.g., "Strong candidate," "Variant of uncertain significance").
+1.0 if the agent suggests generic steps (e.g., "consult a doctor," "genetic counseling") without specific validation measures OR claims 100% certainty in ambiguous cases.
+0.0 if no validation step is proposed or if the confidence level is grossly mismatched to the evidence.

Max total score = 2.0 + 3.0 + 1.0 + 2.0 + 2.0 = 10.

============================================================
CRITERION 3 (0-10): Code quality / data handling integrity
============================================================
This criterion grades the agent's capability to write functional code and handle data loading with integrity.
The agent should produce code with correct imports, syntax, and no obvious runtime errors. The agent should load all required datasets correctly, and perform appropriate schema sanity checks (e.g., inspecting dataset dimensions and column names, verifying key columns/index are usable) before accessing any data. The agent should conduct appropriate data cleaning (e.g., deduplication, missingness handling) when needed.

Item 3.1 (2 pts): Code correctness (imports/syntax/runtime)
+2 if executed code blocks (as shown) avoid agent-caused:
- ImportError/ModuleNotFoundError from hallucinated packages,
- Syntax/Indentation errors,
- Tool failures caused by incorrect input construction, or inappropriate handling of tool outputs,
- obvious NameError/KeyError/AttributeError from careless variable/field usage.
+1 if there are minor errors but they are promptly corrected and do not drive wrong conclusions.
+0 if repeated or uncorrected agent-caused errors occur.
Do NOT penalize clear env errors or platform/tool instability that is not the agent’s fault.

Item 3.2 (4 pts): Data/API response sanity checks before use
+4 if, when handling JSON/tables/results, the agent:
- Inspects keys/columns/shape before indexing,
- Verifies required fields exist (e.g., disease name, OMIM, gene symbol),
- Handles missing/empty results without pretending success.
+0 if it directly indexes assumed keys/columns (often hallucinated) without inspection and this affects reasoning.

Item 3.3 (2 pts): Identifier hygiene (genes + OMIM)
+2 if the agent keeps identifiers consistent and unambiguous:
- Uses standard gene symbols (HGNC), avoids silent alias confusion,
- Preserves OMIM ID accurately (no digit loss, no swapping with Orpha IDs),
- Clearly distinguishes gene vs disease vs phenotype identifiers.

Item 3.4 (2 pts): No fabricated executions or data accesses
+2 if the agent does NOT claim to have run a query, accessed OMIM/Orphanet, or retrieved evidence that is not present in the trajectory observations.
+0 if any such fabricated tool execution/data retrieval appears.

Max total for Criterion 3 = 2 + 4 + 2 + 2 = 10.

============================================================
CRITERION 4 (0-10): Reasoning quality / coherence
============================================================
This criterion assesses the agent’s reasoning coherence and scientific rigor, including correct interpretation of evidence.
Award points ONLY if the behavior is clearly demonstrated.

Item 4.1 (2 pts): Plan and plan adherence
+2 if the agent states a plan (e.g., “map phenotypes → prioritize genes → validate disease”) and follows it,
or explicitly updates the plan based on intermediate findings.

Item 4.2 (2 pts): Correct interpretation of the task inputs
+2 if the agent correctly treats:
- phenotypes as the observed clinical evidence,
- candidate genes as constraints for diagnosis,
- and does not ignore either.

Item 4.3 (2 pts): Evidence-to-claim linkage and final answer consistency
+2 if the final diagnosis is a direct consequence of cited evidence in the trace, with no major logical leaps,
and the final pick is consistent throughout (no contradictory “final answers”).

Item 4.4 (2 pts): Alternatives and calibration (avoids premature guessing)
+2 if the agent considers at least one alternative candidate gene/disease and explains why it is less likely,
and communicates uncertainty appropriately when evidence is thin (no unjustified certainty).

Item 4.5 (2 pts): No hallucinated clinical facts/citations; clear assumptions vs facts
+2 if the agent does not invent patient features, lab results, inheritance patterns, or citations;
and clearly labels assumptions as assumptions.
+0 if any hallucination/fabrication appears.

Max total for Criterion 4 = 2 + 2 + 2 + 2 + 2 = 10.

========================
Return your final grading as JSON with the following keys:
{{
  "output_grading": <0-20>,
  "methodology_knowhow": <0-10>,
  "code_data_handling": <0-10>,
  "reasoning_coherence": <0-10>,
  "total": <0-50>,
  "rationale": "<detailed rationale justifying the scores you give per criterion>",
  "weaknesses": "<a list of weaknesses in the agent's trajectory, can be an empty list if the agent's output, methodology, code and data handling, and reasoning coherence are perfect in all aspects>"
}}
"""

        return rubric.strip()
