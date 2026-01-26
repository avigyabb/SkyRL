from verl.workers.agentic.biomni.task.base_task import base_task
import pandas as pd
import numpy as np

class gwas_causal_gene(base_task):

    def __init__(self, path = './data', dataset = 'opentargets', num_samples = 100):
        if dataset not in ['opentargets', 'gwas_catalog', 'pharmaprojects']:
            raise ValueError('dataset must be one of opentargets, gwas_catalog, pharmaprojects')
        
        query_path = path + '/gwas_causal_gene/' + dataset + '_step2.for_llm.tsv'
        answer_path = path + '/gwas_causal_gene/' + dataset + '_step2.labels'

        self.prompt = "Your task is to identify likely causal genes within a locus for a given GWAS phenotype. From the list, provide the most likely causal gene. \nIdentify the causal gene.\nGWAS phenotype: {trait}\nGenes in locus: {gene_str}."
        self.query = pd.read_csv(query_path, sep = '\t').sample(frac = 1, random_state = 42).reset_index(drop=True)[:num_samples]
        self.answer = pd.read_csv(answer_path, sep = '\t').sample(frac = 1, random_state = 42).reset_index(drop=True)[:num_samples]

    def __len__(self):
        return len(self.query)

    def get_example(self, index = None):
        if index is None:
            index = np.random.randint(len(self.query))
            q = self.query.iloc[index]
        else:
            q = self.query.iloc[index]
        
        return {"prompt": self.prompt.format(trait = q.description, gene_str = q.symbol_gene_string), 
                "answer": self.answer.iloc[index].symbol,
                "instance_id": index}

    def split(self, ratio = 0.8, seed = 42):
        np.random.seed(seed)
        indices = np.arange(len(self.query))
        np.random.shuffle(indices)
        split_idx = int(ratio * len(self.query))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices

    def reward(self, input, output):
        if isinstance(output, dict):
            output = output['causal_gene']
        print(f"Gwas causal gene output: {output}")
        answer = self.get_example(input)['answer']
        print(f"Gwas causal gene answer: {answer}")
        return 1 if output == answer else 0

    def get_iterator(self):
        for i in range(len(self.query)):
            yield self.get_example(i)

    def evaluate(self, response):
        ## expected a list/array of symbols
        from sklearn.metrics import accuracy_score
        predicted = [i.strip('{}') for i in response]
        ground_truth = self.answer['symbol'].values
        accuracy = accuracy_score(ground_truth, predicted)

        return {
            'accuracy': accuracy
        }

    def output_class(self):
        from pydantic import BaseModel, Field
        from typing import Optional
        class GeneOutput(BaseModel):
            """Causal gene output."""

            causal_gene: Optional[str] = Field(
                description="causal gene in the format of {gene_name}, e.g. {BRCA1}"
            )
        return GeneOutput

    def output_parser(self, output, llm):
        from pydantic import BaseModel, Field
        from typing import Optional
        class GeneOutput(BaseModel):
            """Causal gene output."""

            causal_gene: Optional[str] = Field(
                description="causal gene in the format of {gene_name}, e.g. {BRCA1}"
            )

        output_parser = llm.with_structured_output(GeneOutput)
        output = output_parser.invoke(output)
        return output.causal_gene

    def get_rubric(self, input, parsed_output, raw_output):
        """
        Rubric for gwas_causal_gene.
        - input: same as in reward(self, input, output)  (index into the task’s sampled queries)
        - parsed_output: same as output in reward(...); expected to be a single gene symbol string,
        but may arrive list-like via the system parser
        - raw_output: the model’s raw text output (may include code blocks + observations if present)
        """

        import re
        import numpy as np
        
        if isinstance(parsed_output, dict):
            parsed_output = parsed_output['causal_gene']

        # Always include the user-facing prompt and the raw model output in the rubric.
        ex = self.get_example(input)
        prompt = ex["prompt"]
        ground_truth = ex["answer"]

        # ----------------------------
        # Helpers: normalize + parsing
        # ----------------------------
        def _norm_gene(x):
            """Normalize a gene symbol for robust equality checks."""
            if x is None:
                return ""
            s = str(x).strip()
            # remove common wrappers/braces/quotes produced by parsers
            s = s.strip("{}[]()\"' \t\r\n")
            # collapse internal whitespace
            s = re.sub(r"\s+", "", s)
            return s.upper()

        # Normalize parsed_output display and pick a single "final" prediction for scoring.
        if isinstance(parsed_output, (list, tuple, np.ndarray)):
            pred_list = [_norm_gene(x) for x in parsed_output if _norm_gene(x)]
            pred = pred_list[0] if pred_list else ""
        else:
            pred = _norm_gene(parsed_output)

        gt = _norm_gene(ground_truth)
        exact_match = (pred == gt) if pred and gt else False

        # Extract candidate genes from the prompt: "Genes in locus: {gene_str}."
        candidates = []
        m = re.search(r"Genes in locus:\s*(.+?)\.\s*$", prompt.strip(), flags=re.IGNORECASE | re.DOTALL)
        if m:
            gene_str = m.group(1).strip()
            # split on common delimiters while preserving symbols like "HLA-DQA1"
            parts = re.split(r"[,\|\;/\n\t]+", gene_str)
            candidates = [p.strip() for p in parts if p and p.strip()]

        cand_norm = [_norm_gene(c) for c in candidates if _norm_gene(c)]
        cand_set = set(cand_norm)
        in_candidates = (pred in cand_set) if pred else False

        fence = "```"

        rubric = f"""
You are grading a biomedical agent’s answer for a GWAS causal-gene-in-locus prioritization task.
The agent must select the SINGLE most likely causal gene for the given GWAS phenotype from the provided “Genes in locus” list.

The agent may write code and use tools in a remote biomedical environment; any tool outputs or errors appear in the trajectory as Observations.

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
- Candidate list ({len(candidates)}): {candidates if candidates else "Could not parse candidate genes from the prompt."}
- Normalized candidates ({len(cand_set)} unique): {sorted(list(cand_set))[:50]}{" ... (truncated)" if len(cand_set) > 50 else ""}

========================
REFERENCE KEY FOR THIS INSTANCE (for output correctness scoring)
========================
- Ground-truth causal gene (full-credit): {gt}
- Parsed prediction (normalized, first item if list-like): {pred if pred else "EMPTY"}
- Exact match to ground truth: {str(exact_match)}
- Prediction is among candidate list: {str(in_candidates)}

============================================================
CRITERION 1 (0-20): Output grading (correctness + format)
============================================================
This criterion measures (A) whether the predicted gene matches the reference key and (B) whether the final response is presented clearly in the expected default format.

Unless the prompt explicitly requests a non-markdown format, assume the agent should return a concise, readable MARKDOWN REPORT.

1A. Output correctness (0-15 points)
- 15.0: Parsed prediction exactly matches the ground-truth gene symbol.
- 0.0: Otherwise.
Important: Do not award partial credit for “close” gene symbols (e.g., paralogs, pathway neighbors) because the task is to choose the single most likely causal gene from a provided list and the benchmark is exact-match.

1B. Markdown report quality & formatting (0-5 points)
Award points ONLY if each item is present and correct in the agent’s FINAL response (not buried in the middle of the trace).
- +1.0: States ONE unambiguous final answer near the top (e.g., “Final answer: GENE”), and it matches the parsed prediction (no multiple competing finals).
- +1.0: Uses scannable markdown structure (headers and/or bullets) with minimal fluff.
- +1.0: Faithfully summarizes key supporting evidence used in the trace (e.g., fine-mapping/credible set, colocalization/eQTL, functional annotation, phenotype biology) rather than generic claims.
- +1.0: References are traceable to the agent’s own tool outputs / web results in the trajectory (e.g., names the resource and cites URLs/PMIDs that actually appear in the trace); do NOT award if citations are fabricated or not traceable.
- +1.0: Calibrated language and limitations: distinguishes evidence from hypotheses; does not overclaim causality if only prioritization evidence is shown.

Max total score for Criterion 1 = 15 + 5 = 20.

============================================================
CRITERION 2 (0-10): Methodology / biomedical know-how
============================================================
This criterion rewards a scientifically rigorous "Post-GWAS Prioritization" strategy. 
It evaluates how the agent moves from a statistical association (locus) to a biological 
conclusion (gene) by integrating phenotype-specific biological priors, functional genomics, 
and comparative reasoning. Award points based on explicit evidence in the trajectory.

Item 2.1: Phenotype-Specific Biological Prioritization (3.0 points)
+3.0 if the agent explicitly prioritizes candidates by cross-referencing the gene list against 
     Mendelian disease databases (e.g., OMIM, Orphanet) OR Mouse Knockout phenotypes (e.g., MGI) 
     relevant to the specific trait (e.g., "Does a mutation here cause Hemochromatosis/Anemia?").
+1.5 if the agent relies solely on Gene Ontology (GO) terms or general pathway databases (KEGG/Reactome) 
     to find broad functional matches without checking for specific disease/phenotype causality.
+0.0 if the agent prioritizes genes based on name similarity, generic descriptions without phenotype 
     relevance, or purely on distance to the lead SNP without biological justification.

Item 2.2: Functional Genomic & Tissue Contextualization (2.5 points)
+2.5 if the agent investigates candidates using BOTH:
     1. Tissue specificity relevant to the trait (e.g., checking Liver expression for Iron, 
        Whole Blood for immune traits via GTEx/HPA).
     2. Functional variant evidence (e.g., coding variants, eQTL/pQTL colocalization, or 
        chromatin interactions).
+1.0 if the agent checks tissue expression levels alone ("Is it expressed in the liver?") but 
     ignores genetic regulatory evidence (eQTLs) or protein-altering potential.
+0.0 if the agent ignores tissue context (e.g., citing brain expression for a metabolic trait) 
     or functional data entirely.

Item 2.3: Comparative Exclusion of Distractors (2.0 points)
+2.0 if the agent explicitly reasons against other plausible candidates in the locus 
     (e.g., "Gene B is nearby but is a housekeeping gene," or "Gene C is expressed in the 
     wrong tissue," or "Gene D has no linked coding variants").
+1.0 if the agent mentions other genes in the list but dismisses them without stating a specific 
     biological or statistical reason.
+0.0 if the agent focuses exclusively on the top candidate with no comparative analysis of the 
     surrounding locus.

Item 2.4: Synthesis, Accuracy, and Mechanism (2.5 points)
+2.5 if the agent identifies the correct causal gene AND provides a specific biological mechanism 
     explaining the association (e.g., "TFR2 regulates Hepcidin signaling in response to iron loading").
+1.5 if the agent identifies the correct gene but provides a vague or generic mechanism 
     (e.g., "It is involved in iron transport").
+0.5 if the agent identifies a plausible but incorrect gene (e.g., a paralog or neighbor with similar function) 
     using sound reasoning.
+0.0 if the agent identifies the wrong gene based on hallucinated evidence or poor logic.

Max total score = 3.0 + 2.5 + 2.0 + 2.5 = 10.

============================================================
CRITERION 3 (0-10): Code quality / data handling integrity
============================================================
This criterion grades the agent's capability to write functional code and handle data loading with integrity.
The agent should produce code with correct imports, syntax, and no obvious runtime errors. The agent should load all required datasets correctly, and perform appropriate schema sanity checks (e.g., inspecting dataset dimensions and column names, verifying key columns/index are usable) before accessing any data. The agent should conduct appropriate data cleaning (e.g., deduplication, missingness handling) when needed.

Item 3.1: No code-caused failures (imports/syntax/logic) (2 points)
- +2: Code blocks are syntactically correct; no obvious code-caused failures (ImportError from hallucinated modules, SyntaxError, NameError, KeyError from fabricated columns); no tool failures caused by incorrect input construction, or inappropriate handling of tool outputs.
- +1: Minor code-caused errors occur but are promptly corrected and do not derail the analysis.
- +0: Repeated hallucinated imports, broken code, or persistent code-caused failures.

Item 3.2: Failure recovery (2 points)
+2 if the agent can successfully recover from code execution errors or tool call failures (if any) by fixing any code issues, handling flaky tools with retries, and actively searching for alternative options when a tool is broken or consistently fails.
+0 if the agent fails to recover from errors, does not attempt to find alternative solutions when one tool fails, or proceeds as if the execution succeeded (when it's actually not successful).

Item 3.3: Data handling sanity checks (4 points)
- +4: If the agent successfully loads all required datasets, and it inspects schema/fields before using them (prints columns/keys, confirms identifiers, interprets p-values/directions correctly, handles missingness/duplicates where relevant).
- +2: Some checks are present but incomplete (e.g., reads a table but does not verify key columns before filtering), yet interpretation remains largely correct.
- +1: Minimal checks; attempts to reference fields/columns without inspection but quickly corrects after an error.
- +0: Uses hallucinated columns/fields, misreads p-values/directions, or bases conclusions on misinterpreted tables.
If no data tables are used (pure literature/database narrative without tables), award +4 by default.

Item 3.4: No fabricated executions/data (2 points)
- +2: No claims of “I queried X and found Y” unless the trace contains the corresponding tool output/observation.
- +0: Any fabricated tool execution, fabricated database hits, fabricated statistics, or fabricated citations not grounded in the trace.

Max total score for Criterion 3 = 2 + 2 + 4 + 2 = 10.

============================================================
CRITERION 4 (0-10): Reasoning quality / coherence
============================================================
This criterion assesses whether the reasoning is logically coherent, planful, and scientifically disciplined.

Item 4.1: Plan present and followed (2 points)
- +2: Provides a plan (even brief) and follows it; if it changes, explains why based on observations.
- +1: Plan is implicit but the steps are coherent.
- +0: No plan and the approach is scattershot.

Item 4.2: Correctly interprets the task and key concepts (2 points)
- +2: Correctly interprets “causal gene within a locus” as gene prioritization using genetic + functional evidence; respects that only one gene must be selected.
- +0: Misinterprets task (e.g., picks variants instead of gene; ignores locus list constraint; confuses phenotype).

Item 4.3: Evidence-to-claim linkage (2 points)
- +2: Major claims are supported by specific evidence presented in the trace; avoids unexplained leaps.
- +1: Some leaps, but overall linkage is mostly understandable.
- +0: Conclusions are largely unsupported or contradict cited evidence.

Item 4.4: Rigor: compares candidates; avoids cherry-picking / p-hacking analogs (2 points)
- +2: Weighs multiple evidence streams consistently; does not cherry-pick one convenient fact while ignoring contradictory evidence; if quantitative evidence appears, does not selectively report only favorable numbers.
- +0: Cherry-picks or “motivated reasoning” dominates.

Item 4.5: No hallucinations; calibrated confidence (2 points)
- +2: No fabricated facts/citations/results; explicitly notes uncertainty where evidence is weak.
- +1: No clear fabrications, but unjustifiably overconfident language.
- +0: Any hallucinated data/citations/results.

Max total score for Criterion 4 = 2 + 2 + 2 + 2 + 2 = 10.

========================
Return your final grading as JSON with the following keys:
{{
  "output_grading": <0-20>,
  "methodology_knowhow": <0-10>,
  "code_data_handling": <0-10>,
  "reasoning_coherence": <0-10>,
  "total": <0-50>,
  "rationale": "Detailed rationale justifying the scores, referencing specific parts of the trajectory.",
  "weaknesses": "<a list of weaknesses in the agent's trajectory, can be an empty list if the agent's output, methodology, code and data handling, and reasoning coherence are perfect in all aspects>"
}}
"""
        return rubric.strip()
