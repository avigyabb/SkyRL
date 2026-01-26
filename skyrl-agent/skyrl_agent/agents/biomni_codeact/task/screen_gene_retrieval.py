from verl.workers.agentic.biomni.task.base_task import base_task
import pandas as pd
import numpy as np
import os
import glob
import re
import random


class screen_gene_retrieval(base_task):
    def __init__(self, num_negative_controls=10, seed=42):
        """
        Initialize the screen_gene_retrieval task class for gene retrieval from screens

        Parameters:
        -----------
        num_negative_controls : int, default=10
            Number of negative control genes to include in the candidate list
        seed : int, default=42
            Random seed for reproducibility
        """
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
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
        self.num_negative_controls = num_negative_controls
        
        # Get all available genes for negative control selection
        self.all_genes = set(self.ground_truth['OFFICIAL_SYMBOL'].unique())
        
        # Define the prompt template
        self.prompt = (
            "Your task is to identify the gene with the strongest perturbation effect for the following research context:\n\n"
            "{context}\n\n"
            "From the following list of candidate genes, select the ONE gene that would have the strongest perturbation effect "
            "in this experimental context:\n\n"
            "Candidate genes: {gene_list}\n\n"
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

    def _get_negative_controls(self, screen_id, target_gene):
        """Get negative control genes for a given screen - genes NOT in top 100"""
        # Get the top 100 genes for this screen (positive hits)
        screen_top_genes = set(self.ground_truth[self.ground_truth['SCREEN_ID'] == screen_id]['OFFICIAL_SYMBOL'].values)
        
        # Get all genes that are NOT in the top 100 for this screen (true negative controls)
        negative_control_candidates = list(self.all_genes - screen_top_genes)
        
        # If we don't have enough negative controls, this shouldn't happen given the large gene pool
        # but we'll handle it gracefully
        if len(negative_control_candidates) < self.num_negative_controls:
            print(f"Warning: Only {len(negative_control_candidates)} negative controls available for screen {screen_id}")
            return negative_control_candidates
        
        # Sort candidates to ensure consistent ordering across runs
        negative_control_candidates.sort()
        
        # Use screen_id as seed for deterministic selection
        screen_random = np.random.RandomState(seed=screen_id)
        
        # Sample the required number of negative controls from genes NOT in top 100
        negative_controls = screen_random.choice(
            negative_control_candidates, 
            size=self.num_negative_controls, 
            replace=False
        ).tolist()
        
        return negative_controls

    def get_example(self, index=None):
        """Get a single example from the dataset"""
        if index is None:
            index = np.random.randint(self.num_examples)
        
        screen_id = self.screen_ids[index]
        
        # Get the context for this screen
        context = self.user_requests[self.user_requests['ID'] == screen_id]['Request'].values[0]
        
        # Get the top gene (target gene with strongest perturbation effect)
        screen_genes = self.ground_truth[self.ground_truth['SCREEN_ID'] == screen_id]
        # Assuming the first gene in the sorted list is the top perturbation gene
        target_gene = screen_genes.iloc[0]['OFFICIAL_SYMBOL']
        
        # Get negative control genes
        negative_controls = self._get_negative_controls(screen_id, target_gene)
        
        # Create randomized candidate list
        candidate_genes = [target_gene] + negative_controls
        
        # Use screen_id as seed for deterministic shuffling
        screen_random = np.random.RandomState(seed=screen_id + 1000)  # Add offset to differentiate from negative control seed
        screen_random.shuffle(candidate_genes)
        
        gene_list_str = ", ".join(candidate_genes)
        
        return {
            "screen_id": screen_id,
            "index": index,
            "target_gene": target_gene,
            "candidate_genes": candidate_genes,
            "context": context,
            "prompt": self.prompt.format(
                context=context,
                gene_list=gene_list_str
            )
        }
    
    def get_example_from_screen_id(self, screen_id):
        """Get example by screen ID"""
        for i, sid in enumerate(self.screen_ids):
            if sid == screen_id:
                return self.get_example(i)
        return None

    def get_iterator(self):
        """Iterate through all examples in the dataset"""
        for i in range(self.num_examples):
            yield self.get_example(i)

    def evaluate(self, predictions, targets):
        """
        Evaluate the predictions against ground truth
        
        Parameters:
        -----------
        predictions : list
            List of predicted gene symbols
        targets : list
            List of target gene symbols
            
        Returns:
        --------
        dict
            Dictionary containing accuracy and other metrics
        """
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have the same length")
        
        correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
        accuracy = correct / len(predictions)
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(predictions)
        }

    def reward(self, input_data, output):
        """Calculate reward based on whether the correct gene was selected"""
        try:
            if isinstance(output, dict):
                output = output['selected_gene']
            if isinstance(input_data, dict):
                target_gene = input_data.get('target_gene')
            else:
                # If input_data is just an index, get the example
                example = self.get_example(input_data)
                target_gene = example['target_gene']
            
            print(f"Screen gene retrieval output: {output}")
            print(f"Screen gene retrieval answer: {target_gene}")
            
            if not output:
                return 0.0
            
            # Clean the output (remove any extra whitespace, punctuation)
            predicted_gene = output.strip().upper()
            target_gene = target_gene.strip().upper()
            
            # Return 1 if correct, 0 if incorrect
            return 1.0 if predicted_gene == target_gene else 0.0
            
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

        class gene_selection(BaseModel):
            """Gene selection for screen gene retrieval task"""

            selected_gene: str = Field(
                description="""The gene symbol of the selected gene that has the strongest perturbation effect.
                Output should be a single gene symbol, e.g., BRCA1"""
            )
        return gene_selection

    def output_parser(self, output, llm):
        """Parse the output to extract the selected gene"""
        from pydantic import BaseModel, Field
        from typing import Optional
        
        class gene_selection(BaseModel):
            """Gene selection for screen gene retrieval task"""

            selected_gene: str = Field(
                description="""The gene symbol of the selected gene that has the strongest perturbation effect.
                Output should be a single gene symbol, e.g., BRCA1"""
            )

        output_parser = llm.with_structured_output(gene_selection)
        result = output_parser.invoke(output)
        return result.selected_gene 

    def get_rubric(self, input, parsed_output, raw_output):
        """
        Rubric for screen_gene_retrieval.

        - input: same as in reward(self, input_data, output)
            * may be an integer index into this task’s examples OR a dict returned by get_example(...)
        - parsed_output: same as output in reward(...); expected to be a single gene symbol string,
            but may arrive list-like via the system parser
        - raw_output: the model's raw text output (may include code blocks + observations if present)

        Returns:
            A formatted rubric string for a tool-less grader.
        """

        # --------------------------
        # Pull the user-facing prompt and per-instance reference key
        # --------------------------
        if isinstance(parsed_output, dict):
            parsed_output = parsed_output['selected_gene']
        try:
            if isinstance(input, dict):
                ex = input
            else:
                ex = self.get_example(input)
        except Exception:
            ex = {}

        prompt = ex.get("prompt", "")
        context = ex.get("context", "")
        screen_id = ex.get("screen_id", None)
        target_gene = ex.get("target_gene", "")
        candidate_genes = ex.get("candidate_genes", [])

        # --------------------------
        # Normalize prediction for correctness checks
        # --------------------------
        import numpy as np

        def _norm_gene(x):
            s = "" if x is None else str(x)
            # keep it simple and robust: strip whitespace and common punctuation
            s = s.strip().strip("`").strip().strip(",").strip(";").strip(".").strip()
            # many graders will treat symbols case-insensitively; the reward uses upper()
            return s.upper()

        # If list-like, use first non-empty item as the prediction for scoring
        pred_raw = ""
        if isinstance(parsed_output, (list, tuple, np.ndarray)):
            for item in parsed_output:
                if _norm_gene(item):
                    pred_raw = item
                    break
        else:
            pred_raw = parsed_output

        pred = _norm_gene(pred_raw)
        gt = _norm_gene(target_gene)

        cand_norm = [_norm_gene(g) for g in candidate_genes] if candidate_genes else []
        in_candidates = (pred in set(cand_norm)) if pred and cand_norm else False
        exact_match = bool(pred) and bool(gt) and (pred == gt)

        parsed_display = parsed_output if parsed_output is not None else "No parsed output available"

        fence = "```"

        rubric = f"""
You are grading a biomedical agent’s answer for a **screen hit gene retrieval** task.

Task intent: The agent is given a **research context** (what the screen is about) and a **short candidate gene list**.
The agent must select **exactly ONE gene** from the candidate list that has the **strongest perturbation effect** in that experimental context.

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
INSTANCE CONTEXT (from the task prompt)
========================
- Screen ID (if available in prompt metadata): {screen_id if screen_id is not None else "Not provided"}
- Research context (verbatim, if available): {context if context else "Not provided separately (see prompt above)."}
- Candidate genes (as provided to agent): {candidate_genes if candidate_genes else "Not available (see prompt above)."}

========================
REFERENCE KEY FOR THIS INSTANCE (for correctness scoring)
========================
- Ground-truth top gene (full-credit): {gt if gt else "Not provided"}
- Parsed prediction (normalized; first item if list-like): {pred if pred else "EMPTY"}
- Prediction is among candidate genes: {str(in_candidates)}
- Exact match to ground truth: {str(exact_match)}

============================================================
CRITERION 1 (0-20): Output grading (correctness + format)
============================================================

1A. **Correctness of the selected gene** (0-15 points)
Award points based on the parsed prediction vs. the reference key above:
- +15 if the parsed prediction **exactly matches** the ground-truth top gene (case-insensitive symbol match).
- +3 if the parsed prediction is **one of the provided candidate genes** but is not the ground-truth top gene.
  (Rationale: the agent respected the candidate constraint, but chose the wrong hit.)
- +0 if the parsed prediction is empty, not a gene symbol, or not in the candidate list.

1B. **Markdown format, clarity, and traceability** (0-5 points; 1 point each)
Unless the prompt required a different format, award 1 point for each satisfied item:
- (1) The response clearly states **a single final answer** near the top (e.g., “Final answer: TP53”).
- (1) The response is a **clean Markdown report** (headings/bullets; not a wall of text).
- (1) The response explicitly ties the gene choice to the **given research context** (not generic gene facts).
- (1) The response provides **traceable references** aligned with the trajectory (e.g., BioGRID ORCS screen page, PMID/DOI, database query outputs shown in Observations).
      *Do not award if citations/sources appear fabricated or do not correspond to anything in the trajectory.*
- (1) The response avoids contradictions (no multiple “final answers”; no switching genes at the end).

Max Criterion 1 score = 20.

============================================================
CRITERION 2 (0-10): Methodology / biomedical know-how
============================================================
This criterion rewards scientifically grounded, context-specific gene screening reasoning
(drug mechanism -> cell line context -> synthetic lethality logic). Award points only if the item
is satisfied based on evidence in the agent trajectory. If unclear, do not award.

Item 2.1: Contextualizes Drug Mechanism and Cell Line Background (3 points)
+3 if the agent explicitly defines BOTH:
  1. The drug's molecular mechanism of action (e.g., PARP inhibition/trapping) and its DNA repair implications.
  2. The specific genetic background of the cell line (e.g., TP53 status, HR proficiency) relevant to drug sensitivity.
+1.5 if the agent defines only one of the above (drug MOA or cell line background) with sufficient detail.
Do NOT award if the agent treats the drug as a generic "chemotherapy" or ignores the cell line's genetic properties entirely.

Item 2.2: Executes Targeted Data Retrieval (2 points)
+2 if the agent queries (or explicitly simulates querying) domain-specific pharmacogenomic databases (e.g., DepMap, BioGRID ORCS, COSMIC)
  OR searches for specific high-confidence literature (e.g., "Olaparib synthetic lethality screens").
+1 if the agent relies on general web searches (e.g., "Olaparib gene interactions") but successfully retrieves relevant interaction data.
Do NOT award if the agent hallucinates interactions without search or uses irrelevant keywords.

Item 2.3: Applies Pathway/Mechanistic Intersection Logic (3 points)
+3 if the agent justifies the selection by mechanistically linking the Drug MOA to the Candidate Gene's specific pathway function.
  (e.g., "Olaparib induces DSBs requiring HR for repair; RAD51D is an essential HR component, creating synthetic lethality upon loss.")
+1.5 if the agent identifies the correct pathway but lacks the specific mechanistic linkage (e.g., "Both are involved in DNA repair").
Do NOT award if the link is generic ("gene is associated with cancer") or biologically incorrect.

Item 2.4: Performs Comparative Reasoning Against Distractors (2 points)
+2 if the agent explicitly compares the top candidate against at least one plausible distractor gene, explaining why the top candidate
  has a *stronger* perturbation effect (e.g., "While SESN1 is stress-related, it is not a direct DNA repair effector like RAD51D").
+1 if the agent selects the correct gene but does not provide comparative reasoning against other candidates.
Do NOT award if the agent selects the wrong gene or selects based on popularity rather than perturbation strength.

Max total score = 3 + 2 + 3 + 2 = 10.

============================================================
CRITERION 3 (0-10): Code quality / data handling integrity
============================================================
This criterion grades the agent's capability to write functional code and handle data loading with integrity.
The agent should produce code with correct imports, syntax, and no obvious runtime errors. The agent should load all required datasets correctly, and perform appropriate schema sanity checks (e.g., inspecting dataset dimensions and column names, verifying key columns/index are usable) before accessing any data. The agent should conduct appropriate data cleaning (e.g., deduplication, missingness handling) when needed.

Item 3.1: **No obvious code defects (imports/syntax/runtime)** (2 points)
+2 if code blocks (if any) avoid:
- hallucinated/unavailable imports (ModuleNotFoundError attributable to the agent),
- syntax/indentation errors,
- tool failures caused by incorrect input construction, or inappropriate handling of tool outputs,
- obvious runtime errors caused by the agent’s logic (NameError, KeyError from invented columns, etc.).
+1 if minor errors occur but the agent quickly fixes them and proceeds correctly.
+0 if repeated code defects occur.
Important: Do NOT penalize clear env errors or platform/tool instability that is not the agent’s fault.

Item 3.2: **Data/schema sanity checks before use** (4 points)
- +4 if the agent loads screen tables/results and explicitly checks schema before accessing:
  shape/dimensions, column names, identifier columns, and confirms candidate gene presence/matching.
- +1 if the agent uses tabular data but does so with weak/partial schema checks (e.g., accesses columns without inspection).
- +0 if the agent relies on hallucinated columns/fields or misreads key columns (e.g., treats p-values backwards) and that affects conclusions.

Item 3.3: **Failure recovery and alternative routes** (2 points)
+2 if the agent responds to failures by adapting: retries, alternative endpoints, different queries/resources,
or a clear fallback method, without treating failures as successes.
+0 if the agent stalls, ignores errors, or proceeds as if failed steps worked.

Item 3.4: **No fabricated tool executions or fabricated data access** (2 points)
+2 if the agent does not claim to have executed analyses, retrieved screen scores, or queried databases unless those outputs appear in the trajectory.
+0 if any fabricated results, screen tables, p-values, rankings, or “I queried X and found Y” statements appear without supporting trace evidence.

Max Criterion 3 score = 10.

============================================================
CRITERION 4 (0-10): Reasoning quality / coherence
============================================================
This criterion assesses the agent’s reasoning coherence and scientific rigor, including correct interpretation of evidence.
Award points only for behaviors clearly demonstrated in the trajectory.

Item 4.1: **Plan and plan adherence** (2 points)
+2 if the agent provides a plan and follows it; if it changes course, it explains why based on observations/results.

Item 4.2: **Correct interpretation of intermediate evidence** (2 points)
+2 if the agent correctly interprets what it finds (e.g., directionality of effects, meaning of screen scores,
p-values/FDR if present, “higher vs lower” semantics) without misstatements.

Item 4.3: **Evidence → conclusion linkage** (2 points)
+2 if the final selection clearly follows from cited evidence (screen rank/score or well-argued comparisons) with no major logical leaps.

Item 4.4: **Avoids premature guessing; handles uncertainty appropriately** (2 points)
+2 if the agent does not jump to a final answer before attempting retrieval/comparison,
and if evidence is insufficient, it clearly states uncertainty and explains the fallback choice conservatively.

Item 4.5: **No hallucinations; calibrated claims** (2 points)
+2 if the agent avoids inventing data, avoids overconfident claims unsupported by the trace,
and distinguishes assumptions from findings.

Max Criterion 4 score = 10.

========================
Return your final grading as JSON with the following keys:
{{
  "output_grading": <0-20>,
  "methodology_knowhow": <0-10>,
  "code_data_handling": <0-10>,
  "reasoning_coherence": <0-10>,
  "total": <0-50>,
  "rationale": "<detailed rationale justification tied to the rubric items above>",
  "weaknesses": "<a list of weaknesses in the agent's trajectory, can be an empty list if the agent's output, methodology, code and data handling, and reasoning coherence are perfect in all aspects>"
}}
"""
        return rubric.strip()
