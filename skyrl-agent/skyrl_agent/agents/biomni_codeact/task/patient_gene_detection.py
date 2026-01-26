import json
from pydantic import BaseModel, Field
import sys
import pandas as pd
import numpy as np
import ast
from verl.workers.agentic.biomni.task.base_task import base_task

class patient_gene_detection(base_task):
    def __init__(self, path = './data', num_samples=100):
        self.data = pd.read_pickle(path + '/patient_gene_detection/patient_gene_detection_benchmark.pkl')
        
        self.query = []
        self.answer = []
        self.data = self.data[:num_samples]
        for idx in range(len(self.data)):
            patient = self.data.iloc[idx]

            phenotypes = patient['phenotypes']
            candidate_genes = patient['candidate_genes']
            true_genes = patient['true_genes']

            self.query.append({
                "phenotypes": phenotypes,
                "candidate_genes": candidate_genes
            })
            self.answer.append({
                "true_genes": true_genes
            })

        self.task_description = """
Task: Given a patient's phenotypes and a list of candidate genes, identify the causal gene.
Phenotypes: {phenotype_list}
Candidate genes: {candidate_genes}

Justify your answer."""

    def __len__(self):
        return len(self.query)

    def get_example(self, index=None):
        if index is None:
            index = np.random.randint(len(self.query))
        
        q = self.query[index]
        a = self.answer[index]
        
        prompt = self.task_description.format(
            phenotype_list=', '.join(q['phenotypes']),
            candidate_genes=', '.join(q['candidate_genes'])
        )
        answer = a["true_genes"]

        return {"prompt": prompt, "answer": answer, "instance_id": index}

    def get_iterator(self):
        for i in range(len(self.query)):
            yield self.get_example(i)

    def split(self, ratio=0.8, seed=42):
        np.random.seed(seed)
        indices = np.arange(len(self.query))
        np.random.shuffle(indices)
        split_idx = int(ratio * len(self.query))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices

    def reward(self, input, output):
        # parse output to dict/json
        # Try json.loads first for valid JSON, then fall back to ast.literal_eval for Python dict syntax
        if not isinstance(output, dict):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                try:
                    output = ast.literal_eval(output)
                except (ValueError, SyntaxError):
                    # If both fail, return 0
                    return 0.0
        
        print("Instance_id: ", input)
        print("Prompt: ", self.get_example(input)['prompt'])
        
        true_genes = self.get_example(input)["answer"]
        # Use .get() for safer dictionary access
        predicted_genes = output.get('causal_genes', [])
        print("True answer:")
        print(true_genes)
        print("Predicted answer:")
        print(predicted_genes)
        if predicted_genes and np.intersect1d(true_genes, predicted_genes):
            return 1.0
        else:
            return 0.0

    def output_class(self):
        class GeneDetectionOutput(BaseModel):
            """List of predicted causal genes for a patient."""

            causal_genes: list = Field(
                description="The list of causal gene(s) identified, e.g., ['ENSG00000138449']. Please Use ENSG ID."
            )
        
        return GeneDetectionOutput

    def get_rubric(self, input, parsed_output, raw_output):
        """
        Rubric for patient_gene_detection.
        - input: same as in reward(self, input, output)  (index into the task's sampled patients)
        - parsed_output: same as output in reward(...); typically a dict with key 'causal_genes' -> list[str]
        - raw_output: the model's raw text output (may include code blocks + observations if present)
        """

        import json
        import ast

        # Always include the user-facing prompt and the raw model output.
        ex = self.get_example(input)
        prompt = ex["prompt"]

        # Pull structured fields for this instance (more reliable than reparsing the prompt).
        phenotypes = []
        candidate_genes = []
        true_genes = []
        try:
            q = self.query[input]
            phenotypes = list(q.get("phenotypes", [])) if isinstance(q, dict) else []
            candidate_genes = list(q.get("candidate_genes", [])) if isinstance(q, dict) else []
            true_genes = list(self.answer[input].get("true_genes", [])) if isinstance(self.answer[input], dict) else []
        except Exception:
            # Fallback to whatever is in get_example (answer only), and leave the rest empty.
            true_genes = ex.get("answer", []) if isinstance(ex.get("answer", []), list) else [ex.get("answer", [])]

        # -----------------------
        # Normalize predictions
        # -----------------------
        if not isinstance(parsed_output, dict):
            try:
                parsed_output = json.loads(parsed_output)
            except json.JSONDecodeError:
                try:
                    parsed_output = ast.literal_eval(parsed_output)
                except (ValueError, SyntaxError):
                    # If both fail, return 0
                    print("Error: Failed to parse parsed_output")
                    print(parsed_output)
        def _norm_gene(x):
            s = "" if x is None else str(x)
            s = s.strip().strip('"').strip("'").strip()
            return s.upper()

        def _coerce_to_pred_list(po):
            # parsed_output is expected to already be parsed by the system,
            # but handle a few common variants robustly.
            if po is None:
                return []
            if isinstance(po, dict):
                v = po.get("causal_genes", po.get("causal_gene", po.get("gene", po.get("genes", []))))
                if isinstance(v, str):
                    return [v]
                if isinstance(v, (list, tuple)):
                    return list(v)
                return []
            if isinstance(po, (list, tuple)):
                return list(po)
            if isinstance(po, str):
                s = po.strip()
                # Try to parse JSON / python dict / python list
                try:
                    obj = json.loads(s)
                    return _coerce_to_pred_list(obj)
                except Exception:
                    try:
                        obj = ast.literal_eval(s)
                        return _coerce_to_pred_list(obj)
                    except Exception:
                        # Treat as a single gene symbol string
                        return [s] if s else []
            # Unknown type
            return []

        pred_raw_list = _coerce_to_pred_list(parsed_output)
        pred_list = [_norm_gene(g) for g in pred_raw_list if _norm_gene(g)]

        cand_set = set(_norm_gene(g) for g in (candidate_genes or []) if _norm_gene(g))
        gt_set = set(_norm_gene(g) for g in (true_genes or []) if _norm_gene(g))

        overlap = sorted(list(set(pred_list) & gt_set))
        pred_in_candidates = [g for g in pred_list if g in cand_set] if cand_set else []
        pred_outside_candidates = [g for g in pred_list if (cand_set and g not in cand_set)] if cand_set else []

        # Shotgun heuristics: penalize dumping many genes (esp. most/all candidates).
        num_candidates = len(cand_set) if cand_set else 0
        num_pred = len(pred_list)
        shotgun = False
        if num_candidates > 0:
            if num_pred >= max(8, int(0.5 * num_candidates)):
                shotgun = True
            if num_pred >= num_candidates:
                shotgun = True

        # Display helpers
        parsed_display = parsed_output if parsed_output is not None else "No parsed output available"
        fence = "```"

        rubric = f"""
You are grading a biomedical agent’s answer for a rare-disease / patient gene prioritization task.

Task: Given a patient's phenotypes and a provided list of candidate genes, identify the causal gene(s) and justify the choice.
Unless the user prompt explicitly requests a strict output format, the expected response is a readable Markdown report with a clearly indicated final answer.

The agent may write and execute code and use biomedical databases/tools; any outputs or errors from those executions would appear in the trajectory as Observations.

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
INSTANCE CONTEXT (from prompt/task)
========================
Phenotypes: {phenotypes if phenotypes else "Unavailable"}
Candidate genes: {sorted(list(cand_set)) if cand_set else (candidate_genes if candidate_genes else "Unavailable")}

========================
REFERENCE KEY FOR THIS INSTANCE (for output correctness scoring)
========================
Ground-truth causal gene(s): {sorted(list(gt_set)) if gt_set else (true_genes if true_genes else "Unavailable")}

Derived checks (precomputed for you):
- Normalized predicted gene list: {pred_list if pred_list else "EMPTY"}
- Overlap(prediction, ground-truth): {overlap if overlap else "NONE"}
- Predicted genes within candidate list: {pred_in_candidates if pred_in_candidates else "NONE / N/A"}
- Predicted genes outside candidate list: {pred_outside_candidates if pred_outside_candidates else "NONE / N/A"}
- Shotgun prediction heuristic triggered: {str(shotgun)}  (means the agent listed many genes, e.g., >= half of candidates)

Total possible score = 50 points across 4 criteria:
- Criterion 1: Output grading (0-20)
- Criterion 2: Methodology / biomedical know-how (0-10)
- Criterion 3: Code quality / data handling integrity (0-10)
- Criterion 4: Reasoning quality / coherence (0-10)

============================================================
CRITERION 1 (0-20): Output grading (correctness + format)
============================================================

1A. Output correctness (0-15 points)
Award based on whether the agent identifies the true causal gene(s) WITHOUT “shotgunning”.
Use ONLY the precomputed overlap and the candidate-list checks above.

- 15.0 points:
  - overlap is NON-EMPTY, AND
  - prediction is specific (NOT shotgun), AND
  - (if candidate list is available) at least one predicted gene is in the candidate list and the final answer is not dominated by out-of-candidate genes.

- 12.0 points:
  - overlap is NON-EMPTY, AND
  - prediction includes some unnecessary extra genes (e.g., 3-5 genes) OR includes minor out-of-candidate noise,
  - but it is NOT shotgun.

- 7.5 points:
  - overlap is NON-EMPTY, BUT shotgun == True (e.g., the agent dumps many/most candidate genes to guarantee overlap),
  - OR the agent lists many genes with no clear single final selection.

- 0.0 points:
  - overlap is EMPTY, OR
  - no usable prediction is provided.

Important: If the agent’s final answer contains the true gene but is buried among a long list of candidates, treat it as shotgun and cap at 7.5.

1B. Format & Markdown report quality (0-5 points)
Unless the prompt explicitly demands JSON-only, prefer a concise Markdown report.
Award 1 point for each item satisfied:

[1 pt] Clear “Final answer” near the top, listing the selected gene(s) unambiguously (and ideally as a short list).
[1 pt] States the key phenotypes driving the diagnosis and connects them to gene/disease mechanisms (not generic filler).
[1 pt] Summarizes what the agent actually did (queries run, databases checked, papers consulted), aligned to the trajectory.
[1 pt] Includes traceable references (e.g., named databases like OMIM/ClinVar/Orphanet/Monarch, PMID/DOI, or direct links) that are consistent with the observed tool outputs.
[1 pt] Clean structure: headings/bullets, no contradictory “final” picks, no long unstructured walls of text.

Max for Criterion 1 = 15 + 5 = 20.

============================================================
CRITERION 2 (0-10): Methodology / Clinical Genetics Know-How
============================================================
This criterion rewards scientifically grounded, systematic evidence gathering and clinical reasoning.
Award points only if the item is satisfied based on evidence in the agent trajectory. If unclear, do not award.

Item 2.1: Phenotype Extraction & Standardization (1.5 points)
+1.5 if the agent explicitly extracts key phenotypes from the raw description AND standardizes them
     (e.g., maps to Human Phenotype Ontology [HPO] terms or uses standard medical terminology)
     prior to analysis.
+0.5 if the agent uses raw text or vague descriptions without attempting to medicalize terms
     (e.g., searching "floppy baby" instead of "hypotonia").
Do NOT award if the agent ignores key phenotypes or hallucinates symptoms not in the prompt.

Item 2.2: External Knowledge Retrieval (Source Quality) (1.5 points)
+1.5 if the agent queries specific, authoritative biomedical databases (e.g., OMIM, Orphanet, ClinVar,
     NCBI Gene) OR performs targeted literature searches to retrieve gene-disease associations.
+0.5 if the agent relies solely on internal parametric knowledge (pre-training) to describe genes
     without performing tool calls to verify current data.
Do NOT award if the agent retrieves irrelevant data (e.g., gene expression in unrelated tissues).

Item 2.3: Systematic Phenotype-Genotype Scoring (2.5 points)
+2.5 if the agent demonstrates a systematic comparison strategy by:
     - Calculating overlap counts/similarity scores (e.g., Jaccard index) between patient and gene phenotypes, OR
     - Explicitly listing present/absent symptoms for each candidate in the thought trace.
+1.0 if the agent makes vague assertions (e.g., "This looks like a match") without explicit feature comparison.
Do NOT award if the match is based on non-clinical features (e.g., "I chose this because it is well-known").

Item 2.4: Differential Diagnosis & Exclusion Logic (2.0 points)
+2.0 if the agent explicitly provides negative evidence for the top runner-up candidates
     (e.g., "Gene B was excluded because it causes renal failure, which this patient lacks").
+1.0 if the agent identifies the correct gene but ignores other candidates entirely in the reasoning.
Do NOT award if the agent rejects a correct candidate based on false information.

Item 2.5: Handling Semantic Variability (1.0 point)
+1.0 if the agent accounts for synonyms or physiological relationships (e.g., matching "Retinal detachment"
     to "Stickler Syndrome" or "Eye abnormalities").
Do NOT award if the agent fails to match a gene due to strict string matching (e.g., rejecting
"Growth delay" because patient had "Short stature").

Item 2.6: Final Answer Accuracy & Justification (1.5 points)
+1.5 if the agent identifies the correct causal gene as the primary answer AND provides a clean
     Markdown report with a logical justification summarizing the evidence.
+0.5 if the agent lists the correct gene as a top candidate but fails to commit, or lists it as a tie
     with an incorrect gene.
Do NOT award if the agent selects the wrong gene or provides no clear final answer.

Max total score = 1.5 + 1.5 + 2.5 + 2.0 + 1.0 + 1.5 = 10.

============================================================
CRITERION 3 (0-10): Code quality / data handling integrity
============================================================
This criterion grades code correctness, robustness, and integrity in handling tool outputs (web/database queries) and any intermediate tables the agent uses.

Item 3.1: Clean code execution (imports + syntax + obvious runtime correctness) (2 points)
+2 if executed code blocks are free of ImportError/ModuleNotFoundError due to hallucinated imports, SyntaxError/IndentationError, tool failures caused by incorrect input construction, and obvious code-caused runtime failures (e.g., NameError, KeyError from unverified columns, wrong object types).
+1 if occasional such errors occur but are promptly diagnosed and corrected.
+0 if repeated code errors occur or the code is largely non-functional.
Important: Do NOT penalize environment/tool instability clearly external to the code (e.g., timeouts/resource limits/env package failures), unless the agent's code is the direct cause.

Item 3.2: Failure recovery and alternative pathing (2 points)
+2 if the agent recovers from failed queries/tool errors by retrying appropriately, switching sources, or adjusting the approach, and does not treat failures as successes.
+0 if it fails to recover, ignores errors, or proceeds with conclusions as if failed steps succeeded.

Item 3.3: Output parsing, schema sanity checks, and data-handling hygiene (4 points)
+4 if the agent handles external results and intermediate data rigorously, for example:
   - checks that query results are non-empty before indexing,
   - validates expected fields exist in retrieved data (JSON/tables),
   - prints/describes shapes/keys/columns before accessing critical values,
   - deduplicates or reconciles conflicting hits when necessary (e.g., multiple diseases per gene),
+2 if it does some checks but still relies on brittle assumptions (partial schema checks; limited validation).
+1 if it frequently indexes into results without checks but does not clearly derail the final conclusion.
+0 if it hallucinates columns/fields, repeatedly mis-parses outputs, or mis-handles data in a way that drives the conclusion.

Item 3.4: No fabricated tool executions, outputs, or data accesses (2 points)
+2 if all claimed tool/database lookups and extracted facts are traceable to the shown trajectory/observations.
+0 if the agent claims to have queried a resource or retrieved specific results that are not supported by the trace.

Max total score for Criterion 3 = 2 + 2 + 4 + 2 = 10.

============================================================
CRITERION 4 (0-10): Reasoning quality / coherence
============================================================
This criterion assesses whether the reasoning is logically coherent, faithful to evidence, and scientifically disciplined.
Assign points ONLY if satisfied based on explicit content in the trajectory.

Item 4.1: Planning and plan adherence (2 points)
+2 if the agent provides a plan and follows it; any plan change is justified by observed results.
+1 if a plan exists but is loosely followed.
+0 if no plan or the actions are incoherent relative to the plan.

Item 4.2: Correct interpretation of task constraints (2 points)
+2 if the agent treats the candidate list as the constraint set and does not “invent” non-candidate answers as final.
+1 if it mentions non-candidates but still clearly selects from candidates in the final answer.
+0 if it selects an out-of-candidate gene as final (when candidates are available) or misreads the prompt.

Item 4.3: Evidence-to-claim linkage (2 points)
+2 if each major claim (gene→phenotype match) is supported by an explicit observation/citation or clearly stated inference.
+1 if linkage is partially supported but includes some leaps.
+0 if reasoning is largely speculative.

Item 4.4: Alternatives and anti-shotgun behavior (2 points)
+2 if the agent meaningfully rules out at least one plausible alternative candidate gene (why it fits worse),
and avoids listing many genes “just in case”.
+1 if it briefly mentions alternatives but without substantive comparison.
+0 if it shotguns many genes or avoids comparative reasoning.

Item 4.5: No hallucinations; calibrated uncertainty (2 points)
+2 if the agent avoids fabricated gene-disease claims/citations and clearly separates facts from hypotheses;
uses calibrated language if evidence is ambiguous.
+0 if any hallucinated citations/results/data are present.

Max for Criterion 4 = 2 + 2 + 2 + 2 + 2 = 10.

========================
Return your final grading as JSON with the following keys:
{{
  "output_grading": <0-20>,
  "methodology_knowhow": <0-10>,
  "code_data_handling": <0-10>,
  "reasoning_coherence": <0-10>,
  "total": <0-50>,
  "rationale": "<detailed rationale justification grounded in the trajectory; cite specific behaviors/failures>",
  "weaknesses": "<a list of weaknesses in the agent's trajectory, can be an empty list if the agent's output, methodology, code and data handling, and reasoning coherence are perfect in all aspects>"
}}
"""
        return rubric.strip()
