from skyrl_agent.agents.biomni_codeact.task.base_task import base_task
import pandas as pd
import numpy as np

class gwas_variant_prioritization(base_task):

    def __init__(self, path = './data', num_samples = 100):

        self.df = pd.read_pickle(path + '/gwas_variant_prioritization/gwas_gold_standards_benchmark.pkl')
        self.df = self.df.sample(frac = 1, random_state = 42).reset_index(drop=True)[:num_samples]

        self.prompt = "Your task is to identify the most promising variant associated wtih a given GWAS phenotype for futher examination. \nFrom the list, prioritize the top associated variant (matching one of the given variant). \nGWAS phenotype: {trait}\nVariants: {variant_list}"
        self.num_examples = len(self.df)
        self.answer = self.df.rsid.values

    def __len__(self):
        return self.num_examples

    def get_example(self, index = None):
        if index is None:
            index = np.random.randint(self.num_examples)
        q = self.df.iloc[index]
        trait = q.trait_name

        total_list = [q.rsid] + q.random_rsids
        np.random.seed(index)
        np.random.shuffle(total_list)

        return {"prompt": self.prompt.format(trait = trait, variant_list = ', '.join(total_list)), 
                "answer": q.rsid,
                "instance_id": index}

    def split(self, ratio = 0.8, seed = 42):
        np.random.seed(seed)
        indices = np.arange(self.num_examples)
        np.random.shuffle(indices)
        split_idx = int(ratio * self.num_examples)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices

    def reward(self, input, output):
        if isinstance(output, dict):
            output = output['variant']
        print(f"Gwas variant prioritization output: {output}")
        answer = self.get_example(input)['answer']
        print(f"Gwas variant prioritization answer: {answer}")
        return 1 if output == answer else 0

    def get_iterator(self):
        for i in range(len(self.query)):
            yield self.get_example(i)

    def evaluate(self, response):
        ## expected a list/array of symbols
        from sklearn.metrics import accuracy_score
        predicted = [i.strip('{}') for i in response]
        ground_truth = self.answer
        accuracy = accuracy_score(ground_truth, predicted)

        total_chisq = []
        miss_num = 0
        for idx, i in enumerate(predicted):
            if i in self.df.ID.values:
                tmp_df = self.df[self.df['index'] == idx]
                total_chisq.append(tmp_df[tmp_df.ID == predicted].CHISQ.values[0])
            else:
                miss_num += 1

        return {
            'accuracy': accuracy,
            'miss_num': miss_num,
            'mean_chisq': np.mean(total_chisq)
        }

    def output_class(self):
        from pydantic import BaseModel, Field
        from typing import Optional
        class VariantOutput(BaseModel):
            """Prioritized variant output."""

            variant: Optional[str] = Field(
                description="variant ID in the format of {variant_ID}, e.g. {2_57743808_G_A}"
            )
        return VariantOutput


    def get_rubric(self, input, parsed_output, raw_output):
            """
            Rubric for GWAS variant prioritization.
            - input: same as in reward(self, input, output)  (index into the task's sampled cases)
            - parsed_output: same as output in reward(...); expected to be a single variant ID (e.g., 'rs123'),but may arrive as a list/array of strings via the system parser
            - raw_output: the model's raw text output (may include code blocks + observations if present)
            """

            import numpy as np
            
            if isinstance(parsed_output, dict):
                parsed_output = parsed_output['variant']

            ex = self.get_example(input)
            prompt = ex["prompt"]

            ground_truth = ex["answer"]

            def _norm(x):
                s = "" if x is None else str(x)
                return s.strip().strip("{}").strip()

            gt = _norm(ground_truth)

            # Normalize parsed_output display and pick a single "final" prediction for correctness scoring.
            if isinstance(parsed_output, (list, tuple, np.ndarray)):
                pred_list = [_norm(x) for x in parsed_output if _norm(x)]
                pred = pred_list[0] if pred_list else ""
            else:
                pred_list = [_norm(parsed_output)] if _norm(parsed_output) else []
                pred = pred_list[0] if pred_list else ""

            parsed_display = parsed_output if parsed_output is not None else "No parsed output available"
            pred_n = _norm(pred)
            exact_match = (pred_n == gt)

            # Extract candidate variants from the prompt (lightweight, tool-less parsing)
            candidates = []
            for line in prompt.splitlines():
                if "variants:" in line.lower():
                    cand_str = line.split(":", 1)[1]
                    candidates = [c.strip() for c in cand_str.split(",") if c.strip()]
                    break

            in_candidates = (pred_n in set(candidates)) if pred_n and candidates else False

            fence = "```"

            rubric = f"""
You are grading a biomedical agent’s answer for a GWAS variant prioritization task.
The agent must identify the MOST promising / top-associated variant for the given GWAS phenotype,
and it MUST be one of the provided variants.

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
CANDIDATE VARIANTS (parsed from prompt)
========================
{candidates if candidates else "Could not parse candidate variants from the prompt."}

========================
REFERENCE KEY FOR THIS INSTANCE (for output correctness scoring)
========================
- Ground-truth top variant (full-credit): {gt}
- Parsed prediction (normalized, first item if list-like): {pred_n if pred_n else "EMPTY"}
- Exact match to ground truth: {str(exact_match)}
- Prediction is among candidate list: {str(in_candidates)}

============================================================
CRITERION 1 (0-20): Output grading (correctness + format)
============================================================
This criterion measures (A) whether the chosen variant matches the reference key and (B) whether the answer is clearly and correctly presented.

1A. Choice correctness (0-15 points)
- 15.0: Parsed prediction exactly matches the ground-truth top variant (string identity after minor whitespace/braces normalization).
- 0.0: Otherwise.

1B. Format & clarity (0-5 points)
Evaluate whether the response is a clean, human-readable Markdown report that:
- Clearly states a SINGLE final selection (variant ID) near the top (e.g., “Final answer: rsXXXX”). (1 pt)
- Efficiently and faithfully summarizes the main procedures and key findings used to choose the variant
  (e.g., association evidence + functional/biological context), without irrelevant filler. (1 pt)
- Uses traceable citations/references when making factual claims (e.g., names the resource/study; links/citations if present in the trajectory). (1 pt)
- Is concise, structured, and scannable (headings and/or bullets; avoids rambling; no multiple competing “final” picks). (1 pt)
- Uses correct markdown formatting and avoids obvious presentation defects (broken headings, unreadable walls of text). (1 pt)

Max total score = (1A + 1B) = 20.

============================================================
CRITERION 2 (0-10): Methodology / GWAS prioritization know-how
============================================================
This criterion rewards scientifically grounded, task-appropriate GWAS variant prioritization methodology.
Award points only if the item is satisfied based on explicit evidence in the agent trajectory. If unclear, do not award.

Item 2.1: Direct GWAS association evidence for the target phenotype (3.0 points)
+3.0 if the agent uses authoritative GWAS resources (e.g., GWAS Catalog, Open Targets Genetics, large biobank portals)
     AND provides concrete association outputs (at least p-value plus effect size/OR/beta),
     ideally noting study context (cohort/ancestry/sample size) or replication.
+1.5 if the agent provides association evidence from less authoritative sources (e.g., general web search, 
     secondary reports, or non-direct publications) without primary GWAS database verification.
+0.0 if there is no direct association evidence or the choice is driven by unsupported claims.

Item 2.2: Related phenotype / PheWAS expansion (2.5 points)
+2.5 if the agent expands to related traits or a PheWAS-style scan with clear biological justification 
     and uses results to support/temper the ranking (e.g., pleiotropy flags, consistency across related endpoints).
+1.0 if it queries related phenotypes with reasonable justification but presents incomplete results or weak integration.
+0.0 if no attempt is made when direct evidence is thin/ambiguous, or if proxies are irrelevant.

Item 2.3: Variant→gene mapping and phenotype-specific biological context (1.5 points)
+1.5 if the agent maps variants to likely effector gene(s) using accepted evidence (eQTL/colocalization, 
     chromatin interaction, fine-mapped assignment, or coherent nearest-gene rationale) 
     AND ties gene/pathway function specifically to the phenotype.
+0.5 if mapping is reasonable but with limited functional support or generic plausibility language.
+0.0 if mapping is absent or clearly speculative/mismatched.

Item 2.4: Functional annotation independent of association p-values (1.0 point)
+1.0 if the agent uses >=2 distinct functional evidence types (e.g., coding consequence + CADD/SpliceAI; 
     regulatory annotations; conservation; tissue-relevant eQTL) and explains how they influence ranking.
+0.5 if it uses one meaningful functional evidence type or provides only vague functional language.
+0.0 if no functional annotation beyond "strong association."

Item 2.5: Integrated ranking with systematic coverage (2.0 points)
+2.0 if the agent evaluates ALL provided variants (or provides complete accounting) AND applies a consistent, 
     justified weighting/integration scheme across evidence types, handling missing data appropriately.
+1.0 if all variants are evaluated but weighting rationale is implicit, or some variants omitted without justification.
+0.0 if the agent discusses only the chosen variant or provides no evidence of comparative evaluation.

Max total score = 3.0 + 2.5 + 1.5 + 1.0 + 2.0 = 10.

============================================================
CRITERION 3 (0-10): Code quality / data handling integrity
============================================================
This criterion grades the agent's capability to write functional code and handle data loading with integrity.
The agent should produce code with correct imports, syntax, and no obvious runtime errors. The agent should load all required datasets correctly, and perform appropriate schema sanity checks (e.g., inspecting dataset dimensions and column names, verifying key columns/index are usable) before accessing any data. The agent should conduct appropriate data cleaning (e.g., deduplication, missingness handling) when needed.

Item 3.1: Clean code execution: imports + syntax + obvious runtime correctness (2 points)
+2 if all executed code blocks are free of:
  - ImportError / ModuleNotFoundError caused by incorrect/hallucinated imports,
  - SyntaxError / IndentationError,
  - Tool failures caused by incorrect input construction or inappropriate handling of tool outputs,
  - other obvious code-caused failures (e.g., NameError from undefined variables, AttributeError from wrong object type,
    KeyError due to referencing non-existent columns) indicating faulty code logic.
+1 if the agent makes occasional such errors but promptly fixes them and proceeds correctly.
+0 if the agent repeatedly makes these errors.
Important: Do NOT penalize environment/tool instability clearly external to the code (timeouts/resource limits), unless the agent’s code is the direct cause.

Item 3.2: Failure recovery (2 points)
+2 if no code/tool errors occur, or if the agent successfully recovers from errors by fixing code issues,
   handling flaky tools with retries/alternatives, and not treating failures as successes.
+0 if the agent fails to recover, does not try alternatives, or proceeds as if failed executions succeeded.

Item 3.3: Data loading and integrity (4 points)
+4 if the agent successfully loads all required tabular data (e.g., GWAS hits, variant annotation tables, PheWAS outputs), and follows appropriate data handling practices:
  - confirms the variant identifier field (e.g., rsID/chr:pos) is present and consistently formatted,
  - checks dimensions and column names (e.g., prints shape/columns or describes them explicitly),
  - verifies decision-critical columns are present and correctly interpreted (e.g., p-value, effect size/beta/OR, allele, direction),
  - avoids hard-referencing unverified columns; handles missingness/duplicates if relevant.
+2 if the agent successfully loads all required data, but schema checks are partially handled or slightly wrong
    (e.g., checks columns but not identifiers; does not confirm p-value/effect-size semantics; minor confusion that does not clearly drive the final choice).
+1 if the agent attempts to load relevant datasets, but fails schema integrity:
  - attempts to access columns/rows without inspecting the table,
  - uses hallucinated column names (but corrects them after encountering key errors),
  - minor confusions that do not clearly drive the final choice
+0 if the agent fails to load the data, fails to recover from dataset access errors, misidentifies the variant identifier column, or misreads p-values/effect sizes/directionality from the table, leading to downstream misalignments or incorrect conclusions.

Item 3.4: No fabricated tool executions or data accesses (2 points)
+2 if the agent does not claim it executed tools/queries or accessed data when it did not (e.g., “I queried GWAS Catalog and found p=…”
without any traceable evidence in the trajectory).
+0 if any fabricated executions/data access claims appear.

Max total score = 2 + 2 + 4 + 2 = 10.

============================================================
CRITERION 4 (0-10): Reasoning quality / coherence
============================================================
This criterion assesses the agent’s reasoning coherence and scientific rigor, including correct interpretation of evidence.
Assign points ONLY if satisfied based on explicit content in the trajectory.

Item 4.1: Plan and plan adherence (2 points)
+2 if the response lays out a plan and follows it (or explicitly updates it based on findings).

Item 4.2: Correct interpretation of task and key evidence (user query + any intermediate results) (2 points)
+2 if the agent correctly interprets the decision-critical information available in the task AND any intermediate evidence it cites, including dataset columns/values, directionality of comparisons, and statistical quantities (e.g., p-values/effect sizes) when they appear.
Do NOT award if it misreads directionality, confuses labels/columns, or invents results.

Item 4.3: Evidence-to-claim linkage and final answer consistency (2 points)
+2 if claims are clearly supported by cited evidence (no major leaps/contradictions), the conclusion follows from comparison across candidates, and the ranking logic is explicit; the final answer matches the earlier ranking/argument and the stated confidence is appropriate.

Item 4.4: Rigor: considers alternatives; avoids cherry-picking / p-hacking analogs (2 points)
+2 if the agent rules out at least one plausible alternative (or compares tradeoffs across candidates) and does not cherry-pick convenient details
while ignoring major constraints. If quantitative results are used, avoids post hoc significance claims or selective reporting.

Item 4.5: No hallucinated data/experiments/citations; calibrated claims (2 points)
+2 if the agent does not fabricate data entries, study outcomes, or citations; distinguishes assumptions from facts; avoids unjustified certainty.
+0 if any hallucinations/fabrications are present.

Max total score = 2 + 2 + 2 + 2 + 2 = 10.

========================
Return your final grading as JSON with the following keys:
{{
  "output_grading": <0-20>,
  "methodology_knowhow": <0-10>,
  "code_data_handling": <0-10>,
  "reasoning_coherence": <0-10>,
  "total": <0-50>,
  "rationale": "<detailed rationale justifying the scores you give per grading item>",
  "weaknesses": "<a list of weaknesses in the agent's trajectory, can be an empty list if the agent's output, methodology, code and data handling, and reasoning coherence are perfect in all aspects>"
}}
"""
            return rubric.strip()
