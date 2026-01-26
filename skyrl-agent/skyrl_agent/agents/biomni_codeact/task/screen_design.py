from verl.workers.agentic.biomni.task.base_task import base_task
import pandas as pd
import numpy as np
import os
import glob
import re


class screen_design(base_task):
    def __init__(self, top_k=20):
        """
        Initialize the screen_design task class for gene screening

        Parameters:
        -----------
        top_k : int, default=100
            Number of top genes to recommend
        """
        print(f"Initializing screen_design task with top_k={top_k}")
        
        # Data paths
        self.data_dir = '/dfs/project/bioagentos/data/screen_data'
        self.screens_dir = os.path.join(self.data_dir, 'BIOGRID-ORCS-ALL-homo_sapiens-1.1.16.screens')
        
        # Load ground truth data
        self.ground_truth = pd.read_csv(os.path.join(self.data_dir, 'top_100_genes_per_screen.csv'))
        
        # Load user requests (contexts)
        self.user_requests = pd.read_csv(os.path.join(self.data_dir, 'user_requests.csv'))
        
        # Convert ID to int for matching
        self.user_requests['ID'] = self.user_requests['ID'].astype(int)
        
        # Get unique screen IDs that have both context and ground truth
        all_screen_ids = set(self.ground_truth['SCREEN_ID'].unique())
        context_screen_ids = set(self.user_requests['ID'].unique())
        self.available_screen_ids = sorted(list(all_screen_ids.intersection(context_screen_ids)))
        self.screen_ids = self.available_screen_ids
            
        self.num_examples = len(self.screen_ids)
        self.top_k = top_k
        
        # Define the prompt template
        self.prompt = (
            "Your task is to identify the top {top_k} genes that are most relevant for the following research context:\n\n"
            "{context}\n\n"
            "Based on the research context, identify the top {top_k} genes that would be most important to investigate."
        )

    def __len__(self):
        return self.num_examples

    def get_screen_data(self, screen_id):
        """Load the screen data file for a given screen ID"""
        screen_file_pattern = f"BIOGRID-ORCS-SCREEN_{screen_id}-*.screen.tab.txt"
        screen_file_path = glob.glob(os.path.join(self.screens_dir, screen_file_pattern))
        
        if not screen_file_path:
            return None
        
        # Load the screen data file
        screen_data = pd.read_csv(screen_file_path[0], sep='\t', comment='#')
        return screen_data

    def _get_example_by_index(self, index=None):
        """Get a single example from the dataset by its screen_id index"""
        if index is None:
            index = np.random.randint(self.num_examples)
        
        screen_id = self.screen_ids[index]
        
        # Get the context for this screen
        context = self.user_requests[self.user_requests['ID'] == screen_id]['Request'].values[0]
        
        return {
            "screen_id": screen_id,
            "index": screen_id,
            "prompt": self.prompt.format(
                context=context,
                top_k=self.top_k
            )
        }
    
    def _get_example_from_screen_id(self, screen_id):
        if not screen_id in self.screen_ids:
            return self._get_example_by_index()
        for i, sid in enumerate(self.screen_ids):
            if sid == screen_id:
                return self._get_example_by_index(i)
        return None
    
    def get_example(self, index=None):
        """Get a single example from the dataset by its screen_id"""
        return self._get_example_from_screen_id(index)

    def get_iterator(self):
        """Iterate through all examples in the dataset"""
        for i in range(self.num_examples):
            yield self._get_example_by_index(i)

    def evaluate(self, screen_id, predicted_genes):
        """
        Evaluate the predictions against ground truth
        
        Parameters:
        -----------
        screen_id : int
            Screen ID
        predicted_genes : list
            List of predicted gene symbols
            
        Returns:
        --------
        dict
            Dictionary containing precision, recall and F1 scores
        """
        # Get ground truth genes for this screen
        true_genes = self.ground_truth[self.ground_truth['SCREEN_ID'] == screen_id]['OFFICIAL_SYMBOL'].values
        
        # Convert to sets for comparison
        true_set = set(true_genes)  # use full ground truth genes for comparison
        pred_set = set(predicted_genes[:self.top_k])  # Top K predicted genes
        
        print(f"Screen design true genes: {true_genes}")
        print(f"Screen design predicted genes: {predicted_genes}")
        # Calculate metrics
        if len(pred_set) == 0:
            precision = 0.0
        else:
            precision = len(true_set.intersection(pred_set)) / len(pred_set)
            
        if len(true_set) == 0:
            recall = 0.0
        else:
            recall = len(true_set.intersection(pred_set)) / len(true_set)
            
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": len(true_set.intersection(pred_set)),
            "false_positives": len(pred_set - true_set),
            "false_negatives": len(true_set - pred_set)
        }


    def reward(self, input, output):
        """Calculate reward as precision score for the predictions"""
        try:
            # screen_id = input['screen_id']
            screen_id = input
            
            if not output:
                return 0.0
            
            if isinstance(output, dict):
                output = output['genes']
            
            # Parse output to extract gene names
            # Assume output is a comma-separated list of gene symbols
            genes = [gene.strip("\"").strip("\'").strip() for gene in output.split(',')]
            
            if len(genes) != self.top_k:
                return 0.0
            
            # Evaluate and return F1 score as reward
            metrics = self.evaluate(screen_id, genes)
            return metrics['precision']
        except Exception as e:
            print(f"Error in reward function: {e}")
            return 0.0

    def split(self, ratio=0.8, seed=42):
        """Split the dataset into train and validation sets"""
        np.random.seed(seed)
        indices = np.arange(self.num_examples)
        np.random.shuffle(indices)
        split_idx = int(ratio * self.num_examples)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices

    def output_class(self):
        """Define the output class for the task"""
        from pydantic import BaseModel, Field
        from typing import Optional

        class gene_list(BaseModel):
            """List of genes for the screen design task"""

            genes: str = Field(
                description="""A comma-separated list of gene symbols that are most relevant for the given research context.
                The output should be in the format: GENE1, GENE2, GENE3, ..., GENEK"""
            )
        return gene_list

    def get_rubric(self, input, parsed_output, raw_output):
        import re
        import math
        import numpy as np
        
        if isinstance(parsed_output, dict):
            parsed_output = parsed_output['genes']

        ex = self.get_example(input)
        prompt = ex["prompt"]
        screen_id = ex.get("screen_id", input)
        top_k = int(getattr(self, "top_k", 20))

        fence = "```"

        # -----------------------------
        # Ground-truth reference set
        # -----------------------------
        true_genes = []
        try:
            df_gt = self.ground_truth[self.ground_truth["SCREEN_ID"] == int(screen_id)]
            if not df_gt.empty:
                true_genes = df_gt["OFFICIAL_SYMBOL"].dropna().astype(str).tolist()
        except Exception:
            true_genes = []

        true_set = set(true_genes)

        # -----------------------------
        # Normalize / parse prediction
        # -----------------------------
        def _clean_token(tok: str) -> str:
            t = "" if tok is None else str(tok)
            t = t.strip().strip('"').strip("'").strip()
            t = re.sub(r"^\s*[-*•]+\s*", "", t)
            t = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", t)
            return t.strip()

        def _to_gene_list(x):
            if x is None:
                return []
            if isinstance(x, dict):
                x = x.get("genes", x.get("gene", x.get("answer", x)))
            if isinstance(x, (list, tuple, np.ndarray)):
                out = [_clean_token(i) for i in x]
                out = [i for i in out if i]
                if len(out) == 1 and ("," in out[0] or "\n" in out[0] or ";" in out[0]):
                    return _to_gene_list(out[0])
                return out
            if isinstance(x, str):
                s = x.strip()
                s = re.sub(r"^\s*[\[\(\{]\s*", "", s)
                s = re.sub(r"\s*[\]\)\}]\s*$", "", s)
                parts = re.split(r"[,;\n]+", s)
                out = [_clean_token(p) for p in parts]
                return [i for i in out if i]
            return [_clean_token(x)]

        pred_all = _to_gene_list(parsed_output)
        pred_topk = pred_all[:top_k]

        # Diagnostics
        duplicates = []
        seen = set()
        for g in pred_topk:
            if g in seen:
                duplicates.append(g)
            seen.add(g)

        # Overlap counts (use set for TP definition; duplicates handled separately in 1A)
        pred_set = set(pred_topk)
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        # Two precision views:
        # - precision_set aligns with the task's reward (since reward uses set(predicted[:k]) as well)
        # - precision_at_k is stricter and scales cleanly with top_k (prevents duplicate-inflation)
        precision_set = (tp / len(pred_set)) if len(pred_set) else 0.0
        precision_at_k = (tp / top_k) if top_k else 0.0

        recall = (tp / len(true_set)) if len(true_set) else 0.0
        f1 = (2 * precision_set * recall / (precision_set + recall)) if (precision_set + recall) else 0.0

        # Scaled thresholds (lenient; adjust fractions as desired)
        thr_10 = max(1, math.ceil(0.30 * top_k))  # ~6 when k=20
        thr_8  = max(1, math.ceil(0.20 * top_k))  # ~4 when k=20
        thr_5  = max(1, math.ceil(0.10 * top_k))  # ~2 when k=20
        thr_2  = 1

        parsed_display = parsed_output if parsed_output is not None else "No parsed output available"

        rubric = f"""
You are grading a biomedical agent’s answer for a gene screening design task.
The agent is given a research context and must propose the top {top_k} human gene symbols to investigate.

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
OVERLAP METRICS (computed)
========================
- TP (unique predicted genes intersect reference): {tp}
- FP (unique predicted not in reference): {fp}
- FN (reference not captured): {fn}
- precision_set (TP / |unique predicted|): {precision_set:.3f}   [reward-aligned]
- precision@k (TP / {top_k}): {precision_at_k:.3f}              [scale-stable]
- recall: {recall:.3f}
- F1 (using precision_set): {f1:.3f}

============================================================
CRITERION 1 (0-20): Output grading (correctness + format)
============================================================

1A. Gene-list validity & compliance (0-5 points)
- +2: Exactly {top_k} gene symbols with no missing/empty entries.
- +1: No duplicates within top-{top_k}.
- +1: Tokens are clean gene symbols (no free-text explanations inside the list).
- +1: Single unambiguous final list.

1B. Reference overlap (0-10 points)
Use TP with scaled cutoffs (equivalently, precision@k tiers). Award EXACTLY ONE:
- 10: TP ≥ {thr_10}  (≈ precision@k ≥ 0.30)
- 8:  TP ≥ {thr_8} and < {thr_10}  (≈ 0.20-0.29)
- 5:  TP ≥ {thr_5} and < {thr_8}   (≈ 0.10-0.19)
- 2:  TP ≥ 1 and < {thr_5}         (≈ 0.01-0.09)
- 0:  TP = 0

Important: If the agent fails 1A’s “exactly {top_k} genes” requirement, cap 1B at 5 points.


============================================================
CRITERION 2 (0-10): Methodology / biomedical know-how
============================================================
This criterion rewards scientifically grounded, context-aware gene prioritization methodology.
Award points only if the item is satisfied based on explicit evidence in the agent trajectory.
If unclear, do not award.

Item 2.1: Contextual Anchoring & Primary MOA Identification (3.0 points)
+3.0 if the agent explicitly identifies BOTH:
  1. The primary molecular target/mechanism of the drug (e.g., identifying CRBN/E3 ligase complex for Lenalidomide), AND
  2. The specific biological dependencies of the cell model (e.g., recognizing BC-3 is a Primary Effusion Lymphoma/PEL line dependent on viral latency or NF-kB/IRF4).
+1.5 if the agent identifies the drug target correctly but treats the cell line as generic "cancer" without accounting for its specific viral/tissue lineage constraints.
+0.0 if the agent hallucinates a target (e.g., p53) or focuses on generic stress responses without identifying the drug's primary binding partner.

Item 2.2: Information Retrieval & Analogical Reasoning (2.0 points)
+2.0 if the agent executes high-quality searches (PubMed, DepMap, COSMIC) AND, where direct data is sparse, applies valid analogical reasoning (e.g., inferring PEL resistance mechanisms from Multiple Myeloma studies due to shared IMiD sensitivity).
+1.0 if the agent relies solely on general LLM knowledge or performs searches using only broad terms (e.g., "drug resistance genes") without leveraging specific disease/drug keywords.
+0.0 if the agent invents data or searches for irrelevant domains.

Item 2.3: Coverage of Distinct Resistance Mechanisms (3.0 points)
+3.0 if the selected genes cover at least THREE distinct biological categories of resistance relevant to the context, such as:
  - Direct Target Modulation (e.g., CRBN, DDB1 loss),
  - Downstream Effector Stability (e.g., IKZF1, IKZF3, CK1a),
  - Pathway Bypass/Compensatory Signaling (e.g., IL6, STAT3, MYC),
  - Drug Efflux/Metabolism (e.g., ABCB1),
  - Apoptosis Blockade (e.g., MCL1, BCL2).
+1.5 if the list is heavily skewed toward only one mechanism (e.g., listing 20 variants of ubiquitin ligases) or misses obvious direct targets.
+0.0 if the selection is random or monotonic.

Item 2.4: Candidate Specificity & Justification (2.0 points)
+2.0 if the agent provides a specific, mechanistically sound rationale for each gene AND filters out biologically implausible targets (e.g., excluding Estrogen Receptor for a B-cell lymphoma; excluding solid tumor drivers like EGFR).
+1.0 if the gene list is generally plausible but justifications are generic (e.g., "important for cell growth") rather than specific to the drug/disease interaction.
+0.0 if the list contains significant hallucinations or targets irrelevant to the tissue type.

Max total score = 3.0 + 2.0 + 3.0 + 2.0 = 10.

============================================================
CRITERION 3 (0-10): Code quality / data handling integrity
============================================================
This criterion grades the agent's capability to write functional code and handle data loading with integrity.
The agent should produce code with correct imports, syntax, and no obvious runtime errors. The agent should load all required datasets correctly, and perform appropriate schema sanity checks (e.g., inspecting dataset dimensions and column names, verifying key columns/index are usable) before accessing any data. The agent should conduct appropriate data cleaning (e.g., deduplication, missingness handling) when needed.

Item 3.1: Code correctness (imports/syntax/runtime/tool use) (2 points)
+2 if executed code blocks (as shown) avoid agent-caused:
- ImportError/ModuleNotFoundError from hallucinated packages,
- Syntax/Indentation errors,
- Tool failures caused by incorrect input construction, or inappropriate handling of tool outputs,
- obvious NameError/KeyError/AttributeError from careless variable/field usage.
+1 if there are minor errors but they are promptly corrected and do not drive wrong conclusions.
+0 if repeated or uncorrected agent-caused errors occur.
Do NOT penalize clear env errors or platform/tool instability that is not the agent’s fault.

3.2 Data-access realism and schema hygiene (4 points)
+4 if the agent handles external results and intermediate data rigorously, for example:
   - checks that query results are non-empty before indexing,
   - validates expected fields exist in retrieved data (JSON/tables),
   - prints/describes shapes/keys/columns before accessing critical values,
   - deduplicates or reconciles conflicting hits when necessary (e.g., multiple diseases per gene),
+2 if it does some checks but still relies on brittle assumptions (partial schema checks; limited validation).
+1 if it frequently indexes into results without checks but does not clearly derail the final conclusion.
+0 if it hallucinates columns/fields, repeatedly mis-parses outputs, or mis-handles data in a way that drives the conclusion.

3.3 Robustness and recovery (2 points)
+2 if the agent adapts when a tool/query/file access fails (retries reasonably, look for alternative approaches) and does not proceed as if failures succeeded.
+0 if it fails to recover from failures or fabricates success.

3.4 No fabricated executions, results, or citations (2 points)
+2 if it does not claim to have run analyses or obtained database results that are not supported by the trace/observations.

Max total score for Criterion 3 = 10.

============================================================
CRITERION 4 (0-10): Reasoning quality / coherence
============================================================
This criterion assesses the agent’s reasoning coherence and scientific rigor, including correct interpretation of evidence.
Award points only if supported by the trajectory/output.

4.1 Planfulness and plan adherence (2 points)
+2 if the agent states a plan and follows it; if it changes plan, it explains why using intermediate observations.

4.2 Correct interpretation of evidence and terms (2 points)
+2 if it correctly interprets key concepts in the context and any evidence it cites (directionality, mechanism, dataset fields if used),
and does not confuse gene/protein names or mix species without justification.

4.3 Coherent linkage from context → evidence → final gene list (2 points)
+2 if the final list clearly follows from stated criteria/evidence with no major leaps or contradictions.

4.4 Non-cherry-picking behavior (2 points)
+2 if it considers plausible alternative pathways/hypotheses or explicitly explains exclusions, and avoids “storytime” reasoning disconnected from evidence.

4.5 Calibration and non-hallucination (2 points)
+2 if it distinguishes assumptions vs facts, avoids overconfident claims when evidence is thin, and does not fabricate results.
+1 if the agent does not fabricate citations/results, but can be overconfident when making claims.
+0 if it fabricates citations/results.

Max total score for Criterion 4 = 10.

========================
Return your final grading as JSON with the following keys:
{{
  "output_grading": <0-20>,
  "methodology_knowhow": <0-10>,
  "code_data_handling": <0-10>,
  "reasoning_coherence": <0-10>,
  "total": <0-50>,
  "rationale": "<detailed justification tied to the rubric items and the specific trajectory>",
  "weaknesses": "<a list of weaknesses in the agent's trajectory, can be an empty list if the agent's output, methodology, code and data handling, and reasoning coherence are perfect in all aspects>"
}}
""".strip()

        return rubric
