from verl.workers.agentic.biomni.task.base_task import base_task
import pandas as pd
import numpy as np
np.random.seed(42)

def shuffle(x):
    np.random.shuffle(x)
    return x

class lab_bench(base_task):

    def __init__(self, path = './data', dataset = 'DbQA'):
        if dataset not in ['DbQA', 'SeqQA']:
            raise ValueError("dataset must be one of 'DbQA', 'SeqQA'")
        
        self.dataset = dataset  # Store dataset type
        df = pd.read_parquet(path + '/' + dataset + '/train-00000-of-00001.parquet')
    
        self.prompt = """The following is a multiple choice question about biology.

Question: {question}
Options:
{options}

Justify your answer."""
        
        np.random.seed(42)
        df['options'] = df.apply(lambda x: shuffle(x.distractors.tolist() + [x.ideal] + ['Insufficient information to answer the question.']), axis=1)
        df['options_letters'] = df.options.apply(lambda x: '\n'.join([chr(ord('A') + i) + '.' + item for i, item in enumerate(x)]))
        df['letter_answer'] = df.apply(lambda x: chr(ord('A') + np.where(np.array(x.options) == x.ideal)[0][0]), axis = 1)
        df['letter_refrain'] = df.apply(lambda x: chr(ord('A') + np.where(np.array(x.options) == 'Insufficient information to answer the question.')[0][0]), axis = 1)

        self.query = df.question.values
        self.options = df.options_letters.values
        self.answer = df.letter_answer.values
        self.refrain_label = df.letter_refrain.values
        
        # Store protocol information if available
        self.protocol = df.protocol.values if 'protocol' in df.columns else None

    def __len__(self):
        return len(self.query)

    def get_example(self, index = None):
        if index is None:
            index = np.random.randint(len(self.query))

        if self.dataset == 'ProtocolQA' and self.protocol is not None:
            return {"prompt": self.prompt.format(
                protocol = self.protocol[index],
                question = self.query[index], 
                options = self.options[index]
            ), "answer": self.answer[index],
                "instance_id": index}
        else:
            return {"prompt": self.prompt.format(
                question = self.query[index], 
                options = self.options[index]
            ), "answer": self.answer[index],
                "instance_id": index}

    def split(self, ratio = 0.8, seed = 42):
        np.random.seed(seed)
        indices = np.random.permutation(len(self.query))
        split_index = int(len(self.query) * ratio)
        return indices[:split_index], indices[split_index:]

    def reward(self, index, answer):
        if isinstance(answer, dict):
            answer = answer['choice']
        print(f"Lab bench output: {answer}")
        print(f"Lab bench answer: {self.answer[index]}")
        return 1 if self.answer[index] == answer else 0

    def get_iterator(self): 
        for i in range(len(self.query)):
            yield self.get_example(i)

    def evaluate(self, response):
        ## expected a list/array of symbols
        from sklearn.metrics import accuracy_score
        ground_truth = self.answer
        response = np.array(response)

        return {
            'accuracy': accuracy_score(ground_truth, response),
            'coverage': np.mean(response != self.refrain_label),
            'refrain_ratio': np.mean(response == self.refrain_label),
            'precision': accuracy_score(ground_truth[np.where(response != self.refrain_label)], response[np.where(response != self.refrain_label)]),
        }

    def output_class(self):
        from pydantic import BaseModel, Field
        from typing import Optional
        class MultipleChoiceOutput(BaseModel):
            """Multiple choice output."""

            choice: Optional[str] = Field(
                description="Multiple choice answer. For example, if there is <answer>A</answer> in the prompt, the output should be 'A'."
            )
        return MultipleChoiceOutput

    def get_rubric(self, input, parsed_output, raw_output):
        """
        Rubric for lab_bench (DbQA / SeqQA) multiple-choice biology questions.

        - input: same as in reward(self, index, answer) (index into the dataset)
        - parsed_output: same as answer in reward(...); expected to be a single option letter (e.g., "A"),
        but may arrive list-like via a system parser.
        - raw_output: the model's raw text output (may include code blocks + observations if present)

        Returns: formatted rubric string (grader is tool-less).
        """

        import re
        import numpy as np
        
        if isinstance(parsed_output, dict):
            parsed_output = parsed_output['choice']

        # -------------------------
        # Pull instance prompt + key
        # -------------------------
        ex = self.get_example(input)
        prompt = ex["prompt"]

        # Ground-truth answer letter for this instance
        gt_letter = str(self.answer[input]).strip() if hasattr(self, "answer") else str(ex.get("answer", "")).strip()

        # Refrain ("Insufficient information...") option letter for this instance (if available)
        refrain_letter = ""
        if hasattr(self, "refrain_label") and self.refrain_label is not None:
            try:
                refrain_letter = str(self.refrain_label[input]).strip()
            except Exception:
                refrain_letter = ""

        # -------------------------
        # Normalize parsed output
        # -------------------------
        def _norm_letter(x):
            if x is None:
                return ""
            s = str(x).strip()
            # Common patterns: "A", "A.", "choice: A", "{A}", etc.
            m = re.search(r"\b([A-Z])\b", s.upper())
            return m.group(1) if m else ""

        if isinstance(parsed_output, (list, tuple, np.ndarray)):
            pred_list = [_norm_letter(x) for x in parsed_output]
            pred_list = [p for p in pred_list if p]
            pred = pred_list[0] if pred_list else ""
        else:
            pred_list = [_norm_letter(parsed_output)] if _norm_letter(parsed_output) else []
            pred = pred_list[0] if pred_list else ""

        # -------------------------
        # Lightweight option parsing from prompt (for grader convenience)
        # -------------------------
        # Expected option lines: "A.<text>", "B.<text>", ...
        options_map = {}
        for line in prompt.splitlines():
            line_s = line.strip()
            if len(line_s) >= 2 and line_s[0].isalpha() and line_s[1] == ".":
                letter = line_s[0].upper()
                text = line_s[2:].strip()
                if letter and text and letter not in options_map:
                    options_map[letter] = text

        gt_text = options_map.get(gt_letter, "")
        refrain_text = options_map.get(refrain_letter, "") if refrain_letter else ""

        # # -------------------------
        # # Detect a "final answer letter" in raw_output (helps handle parser mismatches)
        # # -------------------------
        # detected_final = ""
        # # Prefer patterns that look like explicit final selections
        # patterns = [
        #     r"(final\s*answer|final\s*choice|answer|choice)\s*[:\-]\s*([A-Z])\b",
        #     r"^\s*([A-Z])\s*[\.\)]\s*$",
        #     r"^\s*final\s*[:\-]\s*([A-Z])\b",
        # ]
        # for pat in patterns:
        #     m = re.search(pat, raw_output, flags=re.IGNORECASE | re.MULTILINE)
        #     if m:
        #         detected_final = m.group(m.lastindex).upper()
        #         break

        # -------------------------
        # Convenience booleans
        # -------------------------
        exact_match = (pred == gt_letter) if (pred and gt_letter) else False
        chose_refrain = (pred == refrain_letter) if (pred and refrain_letter) else False
        valid_letter = pred in options_map if (pred and options_map) else bool(pred)

        # -------------------------
        # Render rubric
        # -------------------------
        fence = "```"
        dataset_name = getattr(self, "dataset", "Unknown")

        # Pretty-print options
        if options_map:
            options_block = "\n".join([f"- {k}. {v}" for k, v in sorted(options_map.items())])
        else:
            options_block = "Could not parse options from the prompt."

        rubric = f"""
You are grading a biomedical agent’s answer for a multiple-choice biology question (lab_bench; dataset={dataset_name}).
The agent must choose exactly ONE option letter (A, B, C, …) and justify it.
Unless the prompt explicitly demands a strict machine format, the expected response is a concise, well-structured Markdown report.

The agent may write code and use tools in a remote environment. Any tool outputs or errors would appear in the raw trajectory as Observations.

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
- Parsed output (raw): {parsed_output if parsed_output is not None else "None"}
- Parsed prediction (normalized letter): {pred if pred else "EMPTY / UNPARSEABLE"}
- Prediction is a valid option letter (prompt-derived): {str(valid_letter)}
- Exact match to ground truth: {str(exact_match)}
- Chose "Insufficient information" option: {str(chose_refrain)}

========================
OPTIONS (parsed from prompt)
========================
{options_block}

========================
REFERENCE KEY FOR THIS INSTANCE (for output correctness scoring)
========================
- Ground-truth correct letter (full credit): {gt_letter if gt_letter else "UNKNOWN"}
- Ground-truth option text: {gt_text if gt_text else "UNKNOWN / NOT PARSED"}
- "Insufficient information..." letter in this prompt: {refrain_letter if refrain_letter else "UNKNOWN / NOT AVAILABLE"}
- "Insufficient information..." option text: {refrain_text if refrain_text else "UNKNOWN / NOT PARSED"}

============================================================
CRITERION 1 (0-20): Output grading (correctness + format)
============================================================
This criterion measures (A) whether the final selected option matches the ground truth and (B) whether the final response is readable, well-formed, and faithful to the work shown in the trajectory.

1A. Answer correctness (0-15 points)
Award points using ONLY the reference key and what is actually stated as the final selection.
- +15 if the normalized parsed prediction equals the ground-truth letter.
- +10 ONLY if BOTH are true:
    (i) the parsed prediction does NOT equal the ground truth, AND
    (ii) the raw_output clearly and explicitly marks the ground-truth letter as the final answer (e.g., “Final answer: {gt_letter}”),
        indicating a formatting/parser mismatch rather than a substantive error.
- +0 otherwise.
Important notes:
- If the agent’s “final answer” is missing, contradictory, or lists multiple letters, award +0.
- If the agent chose the “Insufficient information” letter ({refrain_letter}) but the ground-truth letter is different, award +0.

1B. Markdown report quality & compliance (0-5 points; +1 each item if satisfied)
Unless the prompt demands a strict non-markdown format, evaluate whether the response is a clean Markdown report:
- +1 Final selection is unambiguous and near the top (e.g., “Final answer: X”), with exactly one letter.
- +1 The justification is logically structured (headings/bullets) and focused on decision-critical evidence (no rambling).
- +1 Claims are traceable to the agent’s own observations / cited sources in the trajectory (not bare assertions).
- +1 Proper handling of uncertainty: avoids overconfident language when evidence is thin; uses “insufficient information” only with clear rationale.
- +1 Presentation quality: readable, correct markdown, minimal noise (e.g., does not paste huge irrelevant logs).

Max Criterion 1 score = 15 + 5 = 20.

============================================================
CRITERION 2 (0-10): Methodology / biomedical know-how
============================================================
This criterion rewards task-appropriate, scientifically grounded methodology for answering multiple-choice biology questions.
Award points ONLY if the behavior is clearly demonstrated in the trajectory. If unclear, do not award.

Item 2.1: Correctly identifies the question type and what evidence is needed (2 points)
+2 if the agent correctly recognizes what is being asked (e.g., gene set membership vs gene function vs pathway vs sequence property),
and translates it into an evidence plan (what to look up / compute to distinguish options).
Do NOT award if it misreads the question or answers a different question.

Item 2.2: Uses authoritative biomedical sources or standard analyses appropriate to the question type (4 points)
Award points based on demonstrated use of strong methodology:
- +4 if the agent uses at least ONE authoritative resource/analysis that directly answers the question type, such as:
    • Curated biological databases (examples: MSigDB/GSEA for named gene sets; NCBI Gene/UniProt for gene facts; Ensembl;
      GEO for expression studies; ClinVar for variant clinical interpretation; Reactome/KEGG for pathways), OR
    • Standard bioinformatics analyses for sequence questions (examples: sanity-check sequence alphabet/length, alignment/BLAST-style search,
      domain/motif inference, or similar accepted methods),
  AND clearly connects returned evidence to the choice.
- +2 if the agent uses weaker/indirect evidence (generic web summaries) but still ties it to the decision.
- +0 if the agent guesses without evidence or relies on non-credible sources.

Item 2.3: Triangulation / cross-check (2 points)
+2 if the agent cross-checks the key claim using a second independent source or an internal consistency check
(e.g., validates a gene set member list after retrieval; corroborates a gene’s function across two trusted databases; sanity-checks sequence inference).
+0 otherwise.

Item 2.4: Proper use of the “Insufficient information” option (2 points)
+2 if the agent selects “Insufficient information” ONLY when it has:
  (i) attempted reasonable retrieval/analysis steps, AND
  (ii) clearly states what missing information prevents resolution, AND
  (iii) avoids inventing facts.
+1 if it declines with a reasonable rationale but with limited attempt to retrieve evidence.
+0 if it uses “Insufficient information” prematurely (when decisive evidence is available/obtained in-trace) or as a default escape hatch.

Max Criterion 2 score = 2 + 4 + 2 + 2 = 10.

============================================================
CRITERION 3 (0-10): Code quality / data handling integrity
============================================================
This criterion grades the agent's capability to write functional code and handle retrieved data/outputs with integrity.
The agent is expected to use code; award points for correctness and disciplined handling of tool outputs.

Item 3.1: Clean code execution (imports + syntax + obvious runtime correctness) (2 points)
+2 if code blocks (if any) are free of clear agent-caused issues:
  - hallucinated/non-existent imports,
  - syntax/indentation errors,
  - obvious NameError/AttributeError/KeyError from careless variable/column references,
  - tool failures caused by incorrect input construction or inappropriate handling of tool outputs.
+1 if minor mistakes occur but are quickly fixed.
+0 if repeated or severe code mistakes occur.
Do NOT penalize clearly external failures (tool downtime/timeouts) unless the code is the direct cause.

Item 3.2: Data handling integrity (inspect before indexing) (4 points)
+4 if, when the agent retrieves structured outputs (tables, gene lists, query results, JSON),
it performs basic integrity checks BEFORE relying on them, such as:
  - confirming the target entity is found (non-empty results),
  - inspecting schema/fields (keys/columns) prior to access,
  - validating counts/dimensions when relevant (e.g., “N genes found”),
  - ensuring it is using the correct comparison direction / label mapping,
  - avoids hard-coding unverified column names.
+1 if partial checks are present but incomplete.
+0 if it indexes into data blindly (hallucinated columns/keys) or misreads returned fields, leading to downstream misalignments.

Item 3.3: Failure recovery and robust fallback behavior (2 points)
+2 if the agent recovers from tool or code failures by:
  - correcting code issues,
  - switching to alternative retrieval paths when a page/tool fails,
  - and never treating a failed run as if it succeeded.
+0 if it gives up prematurely or proceeds as if a failed step succeeded.

Item 3.4: No fabricated tool executions, citations, or data access (2 points)
+2 if the agent never claims it queried a database, executed code, or retrieved a specific result unless that evidence is present in the trajectory.
+0 if any fabricated execution/result/citation occurs (e.g., cites “I found gene X in dataset” without showing retrieval).

Max Criterion 3 score = 2 + 4 + 2 + 2 = 10.

============================================================
CRITERION 4 (0-10): Reasoning quality / coherence
============================================================
This criterion assesses whether the reasoning is logically coherent, faithful to evidence, and scientifically disciplined.
Assign points ONLY if satisfied based on explicit content in the trajectory.

Item 4.1: Plan and plan adherence (2 points)
+2 if the agent states a plan and follows it; if the plan changes, it explains why based on observations.
+0 otherwise.

Item 4.2: Correct interpretation of intermediate evidence (2 points)
+2 if the agent correctly interprets returned evidence (e.g., directionality, membership checks, sequence properties) and does not distort results.
+0 if it misinterprets or misstates key outputs.

Item 4.3: Option discrimination (eliminates alternatives using evidence) (2 points)
+2 if the agent explicitly rules out competing options using evidence or logic tied to the prompt.
+0 if it merely asserts the chosen option is right without engaging alternatives.

Item 4.4: Scientific rigor and calibrated claims (2 points)
+2 if the agent avoids premature guessing, clearly distinguishes facts vs assumptions, and does not overclaim certainty.
+0 if it leaps to conclusions or expresses unwarranted confidence given thin evidence.

Item 4.5: No hallucinated facts/experiments/results (2 points)
+2 if the reasoning does not introduce fabricated data, fabricated experiments, or invented “findings.”
+0 if any hallucinations are present.

Max Criterion 4 score = 2 + 2 + 2 + 2 + 2 = 10.

========================
Return your final grading as JSON with the following keys:
{{
  "output_grading": <0-20>,
  "methodology_knowhow": <0-10>,
  "code_data_handling": <0-10>,
  "reasoning_coherence": <0-10>,
  "total": <0-50>,
  "rationale": "<detailed, concrete justification tied to the rubric items>",
  "weaknesses": "<a list of weaknesses in the agent's trajectory, can be an empty list if the agent's output, methodology, code and data handling, and reasoning coherence are perfect in all aspects>"
}}
""".strip()

        return rubric
