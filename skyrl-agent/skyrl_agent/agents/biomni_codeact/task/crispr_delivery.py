from verl.workers.agentic.biomni.task.base_task import base_task
import pandas as pd
import numpy as np


class crispr_delivery(base_task):
    def __init__(self, num_samples=100):
        # Load the benchmark dataset
        self.df = pd.read_csv('/dfs/user/kexinh/BioAgentOS/data/crispr_delivery.csv')
        
        # Define the delivery methods and weights for scoring
        self.delivery_methods = {
            'a': 'Plasmid Transfection',
            'b': 'Lentivirus/Retrovirus',
            'c': 'RNP/mRNA electroporation',
            'd': 'RNP/mRNA microinjection',
            'e': 'mRNA LNP',
            'f': 'AAV'
        }
        
        self.weight_first_choice = 2
        self.weight_second_choice = 1
        
        # Clean and format the prompt template for better readability
        raw_prompt = self.df['Prompt'].iloc[0]
        # Strip extra quotes to make it cleaner
        if raw_prompt.startswith('"""'):
            raw_prompt = raw_prompt.strip('"')
        
        # Create a cleaner template that will hold the user's case description
        self.prompt_template = """Given the case description, identify the MOST relevant CRISPR delivery method from the options below:

a. Plasmid Transfection
b. Lentivirus/Retrovirus
c. RNP/mRNA electroporation
d. RNP/mRNA microinjection
e. mRNA LNP
f. AAV

Category: {category}
Case Description: {case_description}

Please select the most relevant method and justify your answer.
"""
        
        # Get unique categories of inputs
        self.categories = self.df['Category'].dropna().unique()
        self.inputs_by_category = {category: self.df[self.df['Category'] == category]['Input'].values 
                                  for category in self.categories}
        self.num_examples = min(num_samples, len(self.df))
        
        # Sample case descriptions from different categories
        np.random.seed(42)
        self.selected_cases = []
        categories_list = list(self.categories)
        for i in range(self.num_examples):
            category_idx = i % len(categories_list)
            category = categories_list[category_idx]
            cases = self.inputs_by_category[category]
            if len(cases) > 0:
                self.selected_cases.append(np.random.choice(cases))
            if len(self.selected_cases) >= num_samples:
                break
                
        self.num_examples = len(self.selected_cases)

    def __len__(self):
        return self.num_examples

    def get_example(self, index=None):
        if index is None:
            index = np.random.randint(self.num_examples)
        case_description = self.selected_cases[index]
        
        # Find the category for this case description
        category = None
        for cat, inputs in self.inputs_by_category.items():
            if case_description in inputs:
                category = cat
                break
        
        return {
            "prompt": self.prompt_template.format(
                case_description=case_description,
                category=category if category else "Unknown"
            ),
            "case_description": case_description,
            "category": category
        }

    def get_iterator(self):
        for i in range(self.num_examples):
            yield self.get_example(i)

    def evaluate(self, case_description, predictions):
        """Evaluate the model's predictions against the ground truth."""
        # Find the row in the dataframe corresponding to the case description
        row = self.df[self.df['Input'] == case_description]
        if row.empty:
            return {"score": 0, "explanation": "Case description not found in dataset"}
        
        # Get the ground truth scores for each delivery method
        scores = {method: row[method].values[0] for method in 'abcdef'}
        
        # Calculate the score based on the predictions
        first_choice = predictions
        
        first_choice_score = scores.get(first_choice, 0)
        
        # 2->1, 1->0.5, 0->0
        normalized_score = first_choice_score/2

        # Find the best possible score
        #best_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        #best_first_choice = best_methods[0][0] if len(best_methods) > 0 else ''
        #best_score = scores.get(best_first_choice, 0)
        
        #normalized_score = first_choice_score / best_score if best_score > 0 else 0
        
        return {
            "score": normalized_score
        }

    def reward(self, input, output):
        """Calculate a reward score from 0 to 1 for the given predictions."""
        if isinstance(output, dict):
            output = output['Answer']
        print(f"Crispr delivery output: {output}")
        case_description = self.selected_cases[input]
        result = self.evaluate(case_description, output)
        return result["score"]

    def split(self, ratio=0.8, seed=42):
        np.random.seed(seed)
        indices = np.arange(self.num_examples)
        np.random.shuffle(indices)
        split_idx = int(ratio * self.num_examples)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices

    def output_class(self):
        from pydantic import BaseModel, Field
        from typing import Optional

        class crispr_delivery_prediction(BaseModel):
            """Prediction of the most relevant CRISPR delivery method"""

            Answer: str = Field(
                description="""The most relevant CRISPR delivery method (a, b, c, d, e, or f)."""
            )
        return crispr_delivery_prediction 
    
    def get_rubric(self, input, parsed_output, raw_output):
        """
        Rubric for crispr_delivery.
        - input: same as in reward(self, input, output)  (index into the task's sampled cases)
        - parsed_output: same as output in reward(...); expected to be a single option letter: 'a'..'f'
        - raw_output: the model's raw text output
        """
        
        if isinstance(parsed_output, dict):
            parsed_output = parsed_output['Answer']

        # Always include the user-facing prompt and the raw model output in the rubric.
        ex = self.get_example(input)
        prompt = ex["prompt"]
        case_description = ex.get("case_description", None)

        # Compute the per-instance reference key (full-credit and partial-credit options).
        # IMPORTANT: The grader is tool-less; we therefore embed the reference key directly in the rubric text.
        full_credit = []
        partial_credit = []
        ref_scores = {}

        try:
            if case_description is not None:
                row = self.df[self.df["Input"] == case_description]
                if not row.empty:
                    ref_scores = {m: int(row[m].values[0]) for m in "abcdef"}
                    full_credit = [m for m, s in ref_scores.items() if s == 2]
                    partial_credit = [m for m, s in ref_scores.items() if s == 1]
        except Exception as e:
            # If anything unexpected happens, keep reference lists empty and let the rubric fall back to reasoning-only grading.
            print(f"Error in getting reference key for crispr delivery task: {e}")
            full_credit, partial_credit, ref_scores = [], [], {}

        # Delivery method legend (to help the grader judge justification quality)
        legend_lines = "\n".join([f"- {k}. {v}" for k, v in self.delivery_methods.items()])

        # Normalize parsed_output display
        parsed_display = parsed_output if parsed_output is not None else "No parsed output available"
        
        fence = "```"

        rubric = f"""
You are grading a biomedical agent’s answer for a CRISPR delivery-method selection task.
The agent must choose the MOST relevant delivery method (one of a-f) for the given case, and justify the choice.

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
OPTIONS LEGEND
========================
{legend_lines}

========================
REFERENCE KEY FOR THIS INSTANCE (for output correctness scoring)
========================
- Full-credit option letter(s): {full_credit if full_credit else "Not provided (fallback to reasoning-only correctness judgment)."}
- Partial-credit option letter(s) (close-but-not-best): {partial_credit if partial_credit else "None (or not provided)."}
- (Optional) Reference scores by letter (higher is better): {ref_scores if ref_scores else "Not provided."}

============================================================
CRITERION 1 (0-20): Output grading (correctness + format)
============================================================
This criterion measures (A) whether the chosen option matches the reference key and (B) whether the answer is clearly and correctly presented.

1A. Choice correctness (0-15 points)
- 15.0: Parsed output is in the full-credit option letter(s).
- 7.5: Parsed output is in the partial-credit option letter(s) AND is not in full-credit.
- 0.0: Parsed output is not in full-credit or partial-credit.
Fallback rule (only if full/partial reference key is not provided):
- Judge correctness from scientific plausibility given the prompt, but be conservative: if the case is ambiguous and the agent does not justify key constraints, cap at 7.5.

1B. Format & clarity (0-5 points)
Evaluate whether the response is a clean, human-readable Markdown report that:
- Clearly states a SINGLE final selection (letter + method name) near the top. (1 pt)
- Efficiently and faithfully summarizes the main procedures and key findings from the agent trajectory. (1 pt)
- Uses traceable citations/references from web search results when making factual claims. (1 pt)
- Is concise, structured, and scannable (headings and/or bullets; avoids rambling). (1 pt)
- Uses correct markdown formatting with no syntax or grammar errors. (1 pt)

Max total score = (1A + 1B) = 20.

============================================================
CRITERION 2 (0-10): Methodology / biomedical know-how
============================================================
This criterion rewards scientifically grounded, case-specific delivery selection reasoning 
(constraints -> mapping -> tradeoffs). Award points only if the item is satisfied based on 
evidence in the agent trajectory. If unclear, do not award.

Item 2.1: Extracts dominant delivery context constraints (3 points)
+3 if the agent correctly identifies the most decision-critical context constraints from the 
case (or states reasonable assumptions if missing), such as:
  - in vivo vs ex vivo,
  - embryo/zygote/oocyte vs somatic context,
  - target tissue/cell type (primary vs immortalized; hard-to-transfect vs easy),
  - screening vs therapeutic/scalable setting.
Do NOT award if the agent misreads the context or invents constraints as facts.

Item 2.2: Considers cargo modality and expression-duration needs (2 points)
+2 if the agent explicitly reasons about at least ONE of:
  - cargo form (plasmid DNA vs mRNA vs RNP),
  - transient vs stable expression requirement,
  - integration vs non-integration implications (where relevant).
Do NOT award if it asserts incorrect modality properties (e.g., claims lentivirus is 
non-integrating without qualification).

Item 2.3: Justifies selection with method-specific tradeoffs (3 points)
+3 if the agent supports its chosen delivery method with at least TWO correct, method-specific 
tradeoffs tied to the case context.

Examples of valid method-specific tradeoffs:
  - Electroporation (RNP/mRNA): efficient ex vivo delivery to primary cells; transient exposure; 
    possible cytotoxicity.
  - Microinjection (RNP/mRNA): embryo/zygote/oocyte suitability; precise early-stage delivery.
  - AAV: tissue tropism; in vivo efficiency; packaging limits; pre-existing immunity.
  - mRNA LNP: in vivo transient delivery; formulation-dependent tropism; non-integrating.
  - Lentivirus/retrovirus: stable integration/long-term expression; good for pooled screens; 
    integration risk.
  - Plasmid transfection: convenient in vitro for easy-to-transfect lines; often poor for 
    primary cells/in vivo.

Do NOT award if justification is generic ("viral vectors are efficient") without tying to 
method-specific properties and case needs.

Item 2.4: Grounds recommendation in published evidence via web search (2 points)
+2 if the agent searches for and cites real-world evidence from at least TWO relevant, 
high-impact published papers (or authoritative sources) that support the delivery method 
choice for the specific context (e.g., cell type, cargo, application).
+1 if only ONE relevant source is cited with appropriate context.
Do NOT award if:
  - No web search is performed,
  - Citations are fabricated or hallucinated,
  - Sources cited are tangential or do not actually support the recommendation.

Max total score = 3 + 2 + 3 + 2 = 10.

============================================================
CRITERION 3 (0-10): Code quality / data handling integrity
============================================================
This criterion grades the agent's capability to write functional code and handle data loading with integrity.
The agent should produce code with correct imports, syntax, and no obvious runtime errors. The agent should load all required datasets correctly, and perform appropriate schema sanity checks (e.g., inspecting dataset dimensions and column names, verifying key columns/index are usable) before accessing any data. The agent should conduct appropriate data cleaning (e.g., deduplication, missingness handling) when needed.

Item 3.1: Clean code execution: imports + syntax + obvious runtime correctness (2 points)
+2 if all executed code blocks run without:
  - ImportError / ModuleNotFoundError caused by incorrect/hallucinated imports,
  - SyntaxError / IndentationError,
  - Tool failures caused by incorrect input construction or inappropriate handling of tool outputs,
  - other obvious code-caused failures (e.g., NameError from undefined variables, AttributeError from wrong object type,
    KeyError due to referencing non-existent columns) indicating faulty code logic.
+1 if the agent occasionally makes the above errors, but most of the code blocks are clean, and the agent is able to promptly fix any errors that occur.
+0 if the agent repeatedly makes the above errors.
Important: Do NOT penalize environment/tool instability clearly external to the code (e.g., timeouts/resource limits/env package failures), unless the agent's code is the direct cause.

Item 3.2: Failure recovery (2 points)
+2 if the agent can successfully recover from code execution errors or tool call failures (if any) by fixing any code issues, handling flaky tools with retries, and actively searching for alternative options when a tool is broken or consistently fails.
+0 if the agent fails to recover from errors, does not attempt to find alternative solutions when one tool fails, or proceeds as if the execution succeeded (when it's actually not successful).

Item 3.3: Data loading and integrity (4 points)
+4 if the agent does not attempt any data loading, as the task does not require any data analysis; or if the agent successfully retrieves relevant evidence with appropriate data handling practices.
+1 if the agent attempts to load some data, but does not follow appropriate data handling practices (e.g. inspecting the schema before accessing data)
+0 if the agent attempts to load some data, but the data are loaded incorrectly or misinterprets the data which leads to downstream misalignments.

Item 3.4: No fabricated tool executions or data accesses (2 points)
+2 if the agent does not claim it has executed a tool or code block or accessed certain data when it in fact did not.

Max total score = 2 + 2 + 4 + 2 = 10.

============================================================
CRITERION 4 (0-10): Reasoning quality / coherence
============================================================
This criterion assesses the agent's ability to reason coherently and conduct rigorous statistical procedures (when applicable) without cherry-picking, p-hacking, overconfidence, or jumping to conclusions. Assign points ONLY if the item is satisfied based on evidence in the agent's trajectory. If unclear, do not award.

Item 4.1: Plan and plan adherence (1 point)
+1 if the agent provides a clear plan and follows it; if the plan changes, it justifies the change based on intermediate results.

Item 4.2: User query interpretation (1 point)
+1 if the agent correctly interprets the decision-critical information available in the user prompt.

Item 4.3: Final answer consistency (2 points)
+2 if the final answer is consistent with the plan, the evidence, and the reasoning trace.

Item 4.4: Coherent linkage from constraints/evidence -> method choice (2 points)
+2 if the conclusion follows from stated constraints via a coherent argument (no major leaps/contradictions) and is case-grounded.

Item 4.5: Rigor: considers alternatives; avoids cherry-picking / p-hacking analogs (2 points)
+2 if the agent rules out at least one plausible alternative (or compares tradeoffs) and does not cherry-pick convenient details
while ignoring major constraints. If quantitative results are used, avoids post hoc significance claims or selective reporting.

Item 4.6: No hallucinated data/experiments/citations; calibrated claims (2 points)
+2 if the agent does not fabricate data entries, experiment outcomes, or citations; distinguishes assumptions from facts; avoids overconfidence.
+1 if the agent avoids any hallucinations or fabrications, but can make overconfident claims or decisions.
+0 if ANY hallucinations or fabrications are present in the agent trajectory.

Max total score = 1 + 1 + 2 + 2 + 2 + 2 = 10.

========================
Return your final grading as JSON with the following keys:
{{
  "output_grading": <0-20>,
  "methodology_knowhow": <0-10>,
  "code_data_handling": <0-10>,
  "reasoning_coherence": <0-10>,
  "total": <0-50>,
  "rationale": "<a detailed rationale justifying the scores you give per grading item>",
  "weaknesses": "<a list of weaknesses in the agent's trajectory, can be an empty list if the agent's output, methodology, code and data handling, and reasoning coherence are perfect in all aspects>"
}}
"""

        return rubric