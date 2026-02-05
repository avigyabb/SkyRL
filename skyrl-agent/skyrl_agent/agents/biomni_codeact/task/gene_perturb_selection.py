from skyrl_agent.agents.biomni_codeact.task.base_task import base_task
import json, pandas as pd, numpy as np

class gene_perturb_selection(base_task):
    def __init__(self, path = './data', dataset='IL2',
                num_query_genes=10):
        json_file_path = path + '/gene_perturb_selection/' + dataset + '.json'
        with open(json_file_path, 'r') as f:
            prompt_data = json.load(f)

        self.task_description = prompt_data['Task']
        
        ground_truth_path = path + '/gene_perturb_selection/ground_truth_' + dataset + '.csv'
        hit_genes_path = path + '/gene_perturb_selection/topmovers_' + dataset + '.npy'

        self.ground_truth = pd.read_csv(ground_truth_path, index_col=0)
        self.all_hit_genes = np.load(hit_genes_path)

        self.query = []
        self.answer = []
        np.random.seed(42)
        non_hit_genes = np.setdiff1d(self.ground_truth.index.values, self.all_hit_genes)
        for hit in self.all_hit_genes:
            sampled_non_hit_genes = np.random.choice(non_hit_genes, num_query_genes-1, replace=False).tolist()
            sampled_non_hit_genes += [hit]
            np.random.shuffle(sampled_non_hit_genes)
            self.query.append(','.join(sampled_non_hit_genes))
            self.answer.append(hit)

        self.prompt = "Your task is to {task_description}. \n From the list of potential genes, provide one most confident gene (matching one of the given genes). \n Gene list: {gene_list}"

    def get_example(self, index = None):
        if index is None:
            index = np.random.randint(len(self.query))
        
        q = self.query[index]
        a = self.answer[index]
        return {"prompt": self.prompt.format(task_description = self.task_description, gene_list = q), 
                "answer": a}

    def reward(self, input, output):
        if isinstance(output, dict):
            output = output['gene_name']
        print(f"Gene perturb selection output: {output}")
        answer = self.get_example(input)['answer']
        print(f"Gene perturb selection answer: {answer}")
        return 1 if output == answer else 0

    def split(self, ratio = 0.8, seed = 42):
        np.random.seed(seed)
        indices = np.arange(len(self.query))
        np.random.shuffle(indices)
        split_idx = int(ratio * len(self.query))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices
    
    def get_iterator(self):
        for i in range(len(self.query)):
            yield self.get_example(i)

    def output_class(self):
        from pydantic import BaseModel, Field
        from typing import Optional
        class GeneOutput(BaseModel):
            """A selected gene to conduct perturbation."""

            gene_name: Optional[str] = Field(
                description="A selected gene to conduct perturbation., e.g. {BRCA1}"
            )
        return GeneOutput

    def evaluate(self, response):
        ## expected a list/array of symbols
        from sklearn.metrics import accuracy_score
        predicted = [i.strip('{}') for i in response]
        ground_truth = self.answer
        return {
            'accuracy': accuracy_score(ground_truth, predicted),
            'miss_num': len(np.setdiff1d(predicted, self.ground_truth.index.values)),
            'average_absolute_perturbation_effect': self.ground_truth.loc[np.intersect1d(predicted, self.ground_truth.index.values)].Score.abs().mean(),
            'hit_ratio': np.mean([1 if gene in self.all_hit_genes else 0 for gene in predicted])
        }

    def get_rubric(self, input, parsed_output, raw_output):
        """
        Rubric for gene_perturb_selection.
        - input: same as in reward(self, input, output)  (index into the task's sampled queries)
        - parsed_output: same as output in reward(...); expected to be a single gene symbol string, but may arrive
        as list-like via the system parser
        - raw_output: the model's raw text output (may include code blocks + observations if present)
        """

        import re
        import numpy as np
        
        if isinstance(parsed_output, dict):
            parsed_output = parsed_output['gene_name']

        # Always include the user-facing prompt and the raw model output in the rubric.
        ex = self.get_example(input)
        prompt = ex["prompt"]
        ground_truth = ex.get("answer", "")

        # ----------------------------
        # Normalization helpers
        # ----------------------------
        def _norm_gene(x: str) -> str:
            """
            Conservative normalization to align with reward() semantics:
            - strip whitespace
            - strip common wrappers/punctuation
            - do NOT change case (reward uses exact match)
            """
            if x is None:
                return ""
            s = str(x).strip()
            s = s.strip("{}[]()<>\"'` \t\r\n")
            # Remove trailing punctuation often produced in prose
            s = re.sub(r"[.,;:]+$", "", s).strip()
            return s

        # ----------------------------
        # Pick a single "primary" prediction
        # ----------------------------
        if isinstance(parsed_output, (list, tuple, np.ndarray)):
            pred_list = [_norm_gene(x) for x in parsed_output if _norm_gene(x)]
            pred_primary = pred_list[0] if pred_list else ""
        else:
            pred_list = [_norm_gene(parsed_output)] if _norm_gene(parsed_output) else []
            pred_primary = pred_list[0] if pred_list else ""

        gt = _norm_gene(ground_truth)
        pred = _norm_gene(pred_primary)

        exact_match = (pred == gt) and bool(pred)

        # ----------------------------
        # Parse candidate genes from the prompt (tool-less)
        # ----------------------------
        candidates = []
        for line in prompt.splitlines():
            if "gene list" in line.lower() and ":" in line:
                cand_str = line.split(":", 1)[1]
                candidates = [_norm_gene(c) for c in cand_str.split(",") if _norm_gene(c)]
                break

        candidate_set = set(candidates)
        in_candidates = (pred in candidate_set) if pred else False

        # If, for any reason, multiple hit genes appear among candidates, surface it for the grader.
        # (This should normally be 1, but this makes the rubric robust to dataset/prompt changes.)
        hit_candidates = []
        try:
            if candidates and hasattr(self, "all_hit_genes"):
                hit_set = set([_norm_gene(x) for x in list(self.all_hit_genes)])
                hit_candidates = [c for c in candidates if c in hit_set]
        except Exception:
            hit_candidates = []

        fence = "```"

        rubric = f"""
You are grading a biomedical agent’s answer for a gene perturbation target selection task.

Task type:
- The user provides (i) a task objective (task description) and (ii) a list of candidate genes.
- The agent must select ONE most confident gene from the provided list.

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
{parsed_output if parsed_output is not None else "No parsed output available"}

========================
CANDIDATE GENES (parsed from prompt)
========================
- Candidates ({len(candidates)}): {candidates if candidates else "Could not parse candidate genes from the prompt."}

========================
REFERENCE KEY FOR THIS INSTANCE (for output correctness scoring)
========================
- Ground-truth gene (full-credit): {gt if gt else "MISSING"}
- Parsed prediction (normalized, primary): {pred if pred else "EMPTY"}
- Exact match to ground truth: {str(exact_match)}
- Prediction is among candidate list: {str(in_candidates)}
- Candidate genes that are known “hits” (if available): {hit_candidates if hit_candidates else "Not provided / none detected."}

============================================================
CRITERION 1 (0-20): Output grading (correctness + format)
============================================================

1A. Output correctness (0-15 points)
Award points STRICTLY based on the system-extracted parsed output and the reference key above.
- 15.0 points: The parsed prediction exactly matches the ground-truth gene (after minor whitespace/punctuation stripping only).
- 0.0 points: Otherwise.

Notes:
- No “close enough” partial credit for a different gene symbol; this is a single-label selection task.
- If the parsed output is not one of the provided candidate genes, correctness must be 0.

1B. Markdown format & communication quality (0-5 points)
Unless the prompt explicitly demands a strict machine output, evaluate whether the agent produced a clean, readable Markdown report:

Award 1 point for each satisfied item (0-5):
- (1) Clear final answer near the top (e.g., “Final answer: <GENE>”) and it is a SINGLE gene (no competing “final” picks).
- (1) Includes a brief, faithful summary of the method used (how the agent evaluated candidates) without irrelevant filler.
- (1) Includes traceable references for factual claims (e.g., paper title/PMID/DOI; authoritative database pages; or clearly attributable tool outputs from the trajectory).
- (1) Structured and scannable (headings and/or bullets; avoids a wall of text; separates evidence from conclusion).
- (1) No major presentation defects that hinder readability (broken formatting, extremely verbose digressions, or contradictory sections).

Max for Criterion 1 = 15 + 5 = 20.

============================================================
CRITERION 2 (0-10): Methodology / biomedical know-how
============================================================
This criterion rewards scientifically grounded, task-appropriate target-selection practice. Award points ONLY if the behavior is explicitly evidenced in the trajectory. If unclear, do not award.

Item 2.1: Correctly operationalizes the task objective from the prompt (2 points)
+2 if the agent correctly identifies the biological objective and directionality implied by the task description
(e.g., increase vs decrease a phenotype; activate vs inhibit a pathway; enhance vs suppress a cellular function),
and uses that objective as the decision criterion.
Do NOT award if the agent misreads the objective, ignores directionality, or substitutes a different goal.

Item 2.2: Systematic comparison across candidates (3 points)
+3 if the agent evaluates MOST or ALL listed candidate genes (not just the selected one),
and provides a comparative rationale (why the chosen gene is better than at least one alternative).
+1 if it considers at least one alternative but coverage is thin.
+0 if it jumps to a single gene with no evidence of comparing candidates.

Item 2.3: Mechanistic plausibility tied to biological context (2 points)
+2 if the justification links the chosen gene to the objective via credible biology (pathway placement, regulatory role,
cell-type relevance, known ligand/receptor/signaling relationships, transcriptional control, etc.) that is consistent with the task description.
Do NOT award if the reasoning is generic (“important gene”, “immune-related”) without connecting to the described objective.

Item 2.4: Evidence-seeking behavior using credible biomedical sources (2 points)
+2 if the agent consults at least TWO credible sources/knowledge bases (examples: peer-reviewed literature;
authoritative pathway/annotation resources; perturbation/CRISPR screen resources; expression/perturbation repositories)
and uses them to support the choice.
+1 if it uses ONE credible source meaningfully.
+0 if it provides no evidence beyond unsupported claims, or cites sources that appear fabricated/untraceable in the trajectory.

Item 2.5: Confounders and intervention logic (1 point)
+1 if the agent considers at least one common confounder relevant to perturbation target selection, such as:
- essentiality/viability effects that could masquerade as phenotype changes,
- broad pleiotropy/off-target interpretability concerns,
- pathway redundancy/compensation,
- feasibility of perturbation (e.g., druggability vs genetic perturbation) IF relevant to the stated task.
Do NOT award if it invents dataset-specific results.

Max total score for Criterion 2 = 2 + 3 + 2 + 2 + 1 = 10.

============================================================
CRITERION 3 (0-10): Code quality / data handling integrity
============================================================
This criterion grades the agent's capability to write functional code and handle data loading with integrity.
The agent should produce code with correct imports, syntax, and no obvious runtime errors. The agent should load all required datasets correctly, and perform appropriate schema sanity checks (e.g., inspecting dataset dimensions and column names, verifying key columns/index are usable) before accessing any data. The agent should conduct appropriate data cleaning (e.g., deduplication, missingness handling) when needed.

Item 3.1: Code correctness hygiene (imports/syntax/obvious runtime logic) (3 points)
+3 if any code shown is internally consistent and free of clear errors (hallucinated imports, syntax errors, undefined variables), tool failures caused by incorrect input construction, or inappropriate handling of tool outputs.
+1 if minor errors appear but are promptly corrected and do not affect the final conclusion.
+0 if repeated or serious code issues occur (e.g., persistent ImportError from non-existent modules; repeated KeyErrors from guessing columns).
Important: Do NOT penalize environment/tool instability clearly external to the code (e.g., timeouts/resource limits/env package failures), unless the agent's code is the direct cause.

Item 3.2: Data access discipline and schema sanity checks (4 points)
+4 if the agent loads any data and first inspects it (shape/columns/identifiers) before using it,
and explains how fields map to biological claims (e.g., confirms effect direction, p-values, gene identifiers).
+1 if it loads data but performs weak/partial validation.
+0 if it fabricates dataset structure, repeatedly guesses columns without inspection, or misinterprets key fields (directionality, identifiers).

Item 3.3: Robustness / recovery from failures (3 points)
+3 if the agent handles failed tool calls or code execution responsibly: acknowledges failures, retries or uses alternatives,
and does not proceed as though a failure succeeded.
+0 if it ignores errors or “pretends” to have results despite failures.

Max total score for Criterion 3 = 3 + 4 + 3 = 10.

============================================================
CRITERION 4 (0-10): Reasoning quality / coherence
============================================================
Assign points ONLY if evidenced in the trajectory and final report.

Item 4.1: Plan and plan adherence (2 points)
+2 if the agent states a plan (even brief) and follows it; any plan updates are justified by intermediate findings.

Item 4.2: Correct interpretation of intermediate evidence (2 points)
+2 if the agent correctly interprets key concepts returned by tools/sources (gene function, pathway role, directionality, cell context),
and does not conflate correlation vs causation without caveats.

Item 4.3: Evidence-to-claim linkage and consistency (2 points)
+2 if the conclusion follows from the evidence presented (no major leaps), and the final answer is consistent with the preceding analysis.

Item 4.4: Alternatives and non-cherry-picking (2 points)
+2 if the agent explicitly rules out plausible alternative candidates (or compares tradeoffs) rather than cherry-picking only supporting facts.

Item 4.5: No hallucinations; calibrated confidence (2 points)
+2 if the agent does not invent facts/citations/results and uses calibrated language (states assumptions as assumptions).
+1 if the agent does not fabricate citations/results, but can be overconfident when making claims.
+0 if ANY hallucinated facts, citations, or results are present.

Max for Criterion 4 = 2 + 2 + 2 + 2 + 2 = 10.

========================
Return your final grading as JSON with the following keys:
{{
  "output_grading": <0-20>,
  "methodology_knowhow": <0-10>,
  "code_data_handling": <0-10>,
  "reasoning_coherence": <0-10>,
  "total": <0-50>,
  "rationale": "<detailed rationale justification for each score bucket>"
  "weaknesses": "<a list of weaknesses in the agent's trajectory, can be an empty list if the agent's output, methodology, code and data handling, and reasoning coherence are perfect in all aspects>"
}}
"""
        return rubric.strip()
