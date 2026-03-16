import logging
import random
import re
import os
import json
import ast
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

# Import task classes
from skyrl_agent.agents.biomni_codeact.task.screen_design import screen_design
from skyrl_agent.agents.biomni_codeact.task.gwas_causal_gene import gwas_causal_gene
from skyrl_agent.agents.biomni_codeact.task.crispr_delivery import crispr_delivery
from skyrl_agent.agents.biomni_codeact.task.rare_disease_diagnosis import rare_disease_diagnosis
from skyrl_agent.agents.biomni_codeact.task.gwas_variant_prioritization import gwas_variant_prioritization
from skyrl_agent.agents.biomni_codeact.task.patient_gene_detection import patient_gene_detection
from skyrl_agent.agents.biomni_codeact.task.lab_bench import lab_bench
from skyrl_agent.agents.biomni_codeact.task.screen_gene_retrieval import screen_gene_retrieval


class CriticMetrics(BaseModel):
    """Output metrics from the LLM critic."""
    
    output_grading: float = Field(description="A float number between 0 and 20 representing the score of the first rubric criterion")
    methodology_knowhow: float = Field(description="A float number between 0 and 10 representing the score of the second rubric criterion")
    code_data_handling: float = Field(description="A float number between 0 and 10 representing the score of the third rubric criterion")
    reasoning_coherence: float = Field(description="A float number between 0 and 10 representing the score of the fourth rubric criterion")
    total: float = Field(description="A float number between 0 and 50 representing the total score")
    rationale: str = Field(description="Detailed, concrete justification tied to the rubric items")
    weaknesses: list[str] = Field(description="A list of weaknesses in the agent's trajectory, can be an empty list if the agent's output, methodology, code and data handling, and reasoning coherence are perfect in all aspects")


class SingleCorrection(BaseModel):
    target: str = Field(description="'Turn N' or 'FINAL ANSWER' -- which action turn this correction targets")
    correction: str = Field(description="Corrected assistant message in <think>...</think><execute/solution>...</execute/solution> format")


class CorrectionOutput(BaseModel):
    """Up to 5 corrections per rollout trajectory. Each targets a different action turn.
    Return fewer corrections (or all null) if the trajectory is already good."""
    correction_1: Optional[SingleCorrection] = None
    correction_2: Optional[SingleCorrection] = None
    correction_3: Optional[SingleCorrection] = None
    correction_4: Optional[SingleCorrection] = None
    correction_5: Optional[SingleCorrection] = None


# System prompt for the LLM judge (exact wording from print_rubrics.py)
SYSTEM_PROMPT = """You are evaluateGPT. Your job is to evaluate the quality of the reasoning, coding, tool execution, and final output of a biomni agent given a user defined biomedical task.

When evaluating the agent's trajectory, you must stricktly adhere to the provided rubric, and justify your score for each criterion.

You should be stringent in your grading of the agent's trajectory. You should look closely at the agent's reasoning, coding, tool execution, and final output. You should identify any weaknesses in the agent's trajectory, and only award points when the agent satisfies all requirements of a rubric item. High-scoring trajectories should be accurate, precise, and rigorous, with expert-level qualities.
"""

# Rubric modifier (exact wording from print_rubrics.py)
RUBRIC_MODIFIER = """
**IMPORTANT:** Be extra strict in your grading of output, methodology, code and data handling, and reasoning coherence.

For instance, if the agent tries to access an column (e.g., `geneSymbol`, `ensembl_gene_id`, etc.) without checking the dataset schema (especially when it leads to a key error), you should penalize it in the grading even if the agent later recovers. Moreover, if the agent hallucinates an import (e.g., tries to import a package that doesn't exist which leads to an import error), you **SHOULD NOT** give full credits in the coding and data handling criteria.

Similarly, if the agent makes an overconfident claim without ruling out the alternatives, or if the agent does not perfectly handle/interpret **ANY** intermediate observations or tool outputs, you **SHOULD NOT** give full credits in methodology and/or reasoning criteria.

You should aim to identify a list of weaknesses in the agent's trajectory before proceeding with item-wise grading. Only give a perfect score if the agent's demonstrates expert-level reasoning accuracy and rigor, and its output, methodology, code and data handling, and reasoning coherence are perfect in all aspects.
"""


def format_messages_to_text(messages: List[Dict[str, str]]) -> str:
    """
    Convert messages list to a prettified trajectory with numbered turns.
    
    Pattern:
    - system message (skipped in output)
    - user message (initial query, shown separately)
    - agent message, user message (observation) pairs -> Turn 1, Turn 2, ...
    - final agent message -> Final Answer
    
    Note: Using "Turn" instead of "Step" to distinguish from the agent's internal
    step numbering within each action output.
    """
    # Convert numpy array or pandas objects to list if needed
    if isinstance(messages, (np.ndarray, pd.Series)) or hasattr(messages, '__array__'):
        messages = list(messages)
    
    if len(messages) == 0:
        return ""
    
    text_parts = []
    turn_num = 0
    i = 0
    
    # Skip system message if present
    if i < len(messages) and messages[i].get('role') == 'system':
        i += 1
    
    # Get initial user query
    if i < len(messages) and messages[i].get('role') == 'user':
        i += 1
    
    # Process action-observation pairs
    pending_action = None
    while i < len(messages):
        msg = messages[i]
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'assistant':
            # This is an action
            pending_action = content
            
            # Check if there's a following observation
            if i + 1 < len(messages) and messages[i + 1].get('role') == 'user':
                # This is a turn with action + observation
                turn_num += 1
                observation = messages[i + 1].get('content', '')
                text_parts.append(f"╔══════════════════════════════════════════════════════════════════════════════╗")
                text_parts.append(f"║                                   TURN {turn_num}                                   ║")
                text_parts.append(f"╚══════════════════════════════════════════════════════════════════════════════╝")
                text_parts.append(f"\n>>> AGENT ACTION:\n{pending_action}")
                text_parts.append(f"{'-'*20}\n")
                text_parts.append(f"\n>>> OBSERVATION:\n{observation}")
                text_parts.append("")  # blank line
                pending_action = None
                i += 2  # Skip the observation
                continue
            else:
                # This is the final answer (no observation follows)
                text_parts.append(f"╔══════════════════════════════════════════════════════════════════════════════╗")
                text_parts.append(f"║                               FINAL ANSWER                                   ║")
                text_parts.append(f"╚══════════════════════════════════════════════════════════════════════════════╝")
                text_parts.append(f"\n{pending_action}")
                pending_action = None
                i += 1
        else:
            # Unexpected pattern - just add it
            i += 1
    
    # Handle any remaining pending action
    if pending_action:
        text_parts.append(f"╔══════════════════════════════════════════════════════════════════════════════╗")
        text_parts.append(f"║                               FINAL ANSWER                                   ║")
        text_parts.append(f"╚══════════════════════════════════════════════════════════════════════════════╝")
        text_parts.append(f"\n{pending_action}")
    
    return "\n".join(text_parts)


CORRECTION_SYSTEM_PROMPT_FORMAT = """You are an expert AI trainer specializing in correcting biomedical agent trajectories. \
Your task is to generate corrected versions of assistant turns that have FORMAT ERRORS.

The agent's total reward is gated by format validation — when format fails, the total reward is 0 \
regardless of how good the reasoning or methodology is. Fixing format is the highest priority.

STRICT FORMAT RULES for every corrected assistant message:
1. Non-final turns MUST follow: <think>REASONING</think><execute>CODE</execute>
2. Final turn MUST follow: <think>REASONING</think><solution>ANSWER</solution>
3. Exactly ONE <think>...</think> block
4. Exactly ONE outer action block (<execute> or <solution>) after </think>
5. Must END with </execute> or </solution>
6. NO extra <think> or </think> tags after the think block
7. The assistant MUST NOT include tool/environment outputs in its own message — observations come from the environment, not the assistant
8. The assistant MUST NOT hallucinate execution results

COMMON ERRORS TO FIX:
- Multiple <think> or </think> tokens → merge into one <think>...</think>
- Missing </think> → add it before the action block
- Wrong tag nesting → restructure to proper format
- Agent writing fake observations/outputs within its message → remove them, end with </execute>
- Using <solution> for non-final turns → change to <execute>
- Using <execute> for final turns → change to <solution>
"""

CORRECTION_SYSTEM_PROMPT_RUBRIC = """You are an expert AI trainer specializing in improving biomedical agent trajectories. \
Your task is to generate improved versions of assistant turns based on rubric evaluation feedback.

The agent's format is correct, so focus entirely on QUALITY improvements according to the rubric:
- Better methodology (using authoritative databases, systematic approaches)
- Better code and data handling (error handling, efficient queries, proper parsing)
- Better reasoning coherence (clear chain-of-thought, justified decisions)
- Better output quality (accurate answers, proper citations, clear presentation)

Each corrected turn must still follow the format: <think>REASONING</think><execute>CODE</execute> \
(or <think>REASONING</think><solution>ANSWER</solution> for the final turn).
"""


class BiomniRubricRewardAdapter:
    """
    LLM-based rubric reward adapter for Biomni tasks.
    Uses an LLM critic to evaluate agent trajectories against task-specific rubrics.
    """
    _initialized: bool = False
    _task_mapping: Dict[str, Any] = {}
    _llm_judge = None
    _aux_llm = None
    _correction_llm = None

    @classmethod
    def _ensure_initialized(cls, model: str = "claude-sonnet-4-5"):
        if cls._initialized:
            return
        
        benchmark_root = '/mnt/biomni_filestore/biomni/biomni_resources/benchmark/'
        
        cls._task_mapping = {
            "rare_disease_diagnosis": rare_disease_diagnosis(benchmark_root),
            "gwas_variant_prioritization": gwas_variant_prioritization(benchmark_root, num_samples=10000),
            "patient_gene_detection": patient_gene_detection(benchmark_root, num_samples=10000),
            "lab_bench_dbqa": lab_bench(benchmark_root, dataset="DbQA"),
            "lab_bench_seqqa": lab_bench(benchmark_root, dataset="SeqQA"),
            "screen_gene_retrieval": screen_gene_retrieval(),
            "screen_design": screen_design(top_k=20),
            "crispr_delivery": crispr_delivery(num_samples=10000),
        }
        
        for data_name in ['opentargets', 'pharmaprojects', 'gwas_catalog']:
            cls._task_mapping[f"gwas_causal_gene_{data_name}"] = gwas_causal_gene(
                path=benchmark_root, dataset=data_name, num_samples=100000
            )
        
        # Initialize the LLM judge
        llm = ChatAnthropic(
            model=model, 
            temperature=1.0,
            max_tokens=32768,
            thinking={"type": "enabled", "budget_tokens": 2000},
        )
        cls._llm_judge = llm.with_structured_output(CriticMetrics, method="json_schema")
        
        # Initialize auxiliary LLM for result formatting (no structured output)
        cls._aux_llm = ChatAnthropic(
            model=model,
            temperature=0.7,
            max_tokens=32768
        )

        # Initialize correction LLM for generating trajectory corrections
        # Claude extended thinking requires temperature=1.0
        correction_llm = ChatAnthropic(
            model=model,
            temperature=1.0,
            max_tokens=32768,
            thinking={"type": "enabled", "budget_tokens": 4000},
        )
        cls._correction_llm = correction_llm.with_structured_output(CorrectionOutput, method="json_schema")
            
        cls._initialized = True

    @staticmethod
    def _validate_format(messages: List[Dict[str, str]]) -> float:
        """
        Check formatting rules for every assistant message:
        - Non-last: <think>...</think>...<execute>...</execute>
        - Last:     <think>...</think>...<solution>...</solution>
        
        Tags inside the outer action block are OK (e.g., <solution> nested inside <execute>).
        The FIRST action tag after </think> determines the outer block type.
        Must end with the matching close tag (soft stop sequence via reward).
        """

        def _valid_block(content: str, *, is_last: bool) -> bool:
            low = content.lower()
            stripped = low.rstrip()  # Define early for logging

            # Rule 1: Must start with <think>
            if not low.lstrip().startswith("<think>"):
                logger.warning("not start with <think>: %s", stripped)
                return False

            # Rule 2: Exactly one <think> and one </think>
            if low.count("<think>") != 1 or low.count("</think>") != 1:
                logger.warning("not exactly one <think> and one </think>: %s", content)
                return False

            # Rule 3: Must end with </execute> or </solution>
            if not (stripped.endswith("</execute>") or stripped.endswith("</solution>")):
                logger.warning("not end with </execute> or </solution>: %s", stripped)
                return False

            # Split on </think> to separate think block from action block
            parts = low.split("</think>", 1)
            after_think = parts[1]

            # Rule 4: Find which action tag appears FIRST after </think>
            # This determines the OUTER action type (nested tags inside are OK)
            exec_pos = after_think.find("<execute>")
            sol_pos = after_think.find("<solution>")

            # Must have at least one action block
            if exec_pos == -1 and sol_pos == -1:
                logger.warning("no action tag after </think>: %s", stripped)
                return False

            # Determine outer action type by which appears first
            if exec_pos == -1:
                outer_is_execute = False
            elif sol_pos == -1:
                outer_is_execute = True
            else:
                outer_is_execute = exec_pos < sol_pos

            # Rule 5: Verify ending matches the outer action type
            # (ensures the first action block wraps all content until the end)
            if outer_is_execute and not stripped.endswith("</execute>"):
                logger.warning("outer is <execute> but doesn't end with </execute>: %s", stripped)
                return False
            if not outer_is_execute and not stripped.endswith("</solution>"):
                logger.warning("outer is <solution> but doesn't end with </solution>: %s", stripped)
                return False

            # Rule 6: Verify there's only ONE outer action block (not multiple sequential blocks)
            # Use find() for O(n) instead of character-by-character O(n²) scanning
            open_tag = "<execute>" if outer_is_execute else "<solution>"
            close_tag = "</execute>" if outer_is_execute else "</solution>"
            depth = 0
            pos = 0
            outer_block_closed = False
            while pos < len(after_think):
                next_open = after_think.find(open_tag, pos)
                next_close = after_think.find(close_tag, pos)
                
                # No more tags found
                if next_open == -1 and next_close == -1:
                    break
                
                # Determine which comes first
                if next_close == -1 or (next_open != -1 and next_open < next_close):
                    # Opening tag comes first
                    if outer_block_closed:
                        logger.warning("multiple outer %s blocks detected: %s", open_tag, stripped)
                        return False
                    depth += 1
                    pos = next_open + len(open_tag)
                else:
                    # Closing tag comes first
                    depth -= 1
                    pos = next_close + len(close_tag)
                    if depth == 0:
                        outer_block_closed = True

            # Rule 7: Last message must have <solution> as outer, non-last must have <execute>
            if is_last and outer_is_execute:
                logger.warning("is_last but outer is <execute>, expected <solution>: %s", stripped)
                return False
            if not is_last and not outer_is_execute:
                logger.warning("not is_last but outer is <solution>, expected <execute>: %s", stripped)
                return False

            # Rule 8: No extra <think> or </think> after the think block
            if "<think>" in after_think or "</think>" in after_think:
                logger.warning("<think> or </think> in after_think: %s", stripped)
                return False

            return True

        assistant_indices = [idx for idx, m in enumerate(messages) if m.get("role") == "assistant"]
        if not assistant_indices:
            return 0.0
            
        last_assistant_idx = assistant_indices[-1]
        
        for idx in assistant_indices:
            content = messages[idx].get("content", "")
            is_last_msg = (idx == last_assistant_idx)
            if not _valid_block(content, is_last=is_last_msg):
                return 0.0
                
        return 1.0

    @classmethod
    def _result_formatting(cls, output_class, task_intention: str, messages: List[Dict[str, str]]) -> Optional[Dict]:
        """
        Parse the agent's free-form solution into structured output expected by each task.
        Ported from biomni_agent.result_formatting().
        """
        from langchain_core.prompts import ChatPromptTemplate
        
        format_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are evaluateGPT, tasked with extract and parse the task output based on the history of an agent. "
                        "Review the entire history of messages provided. "
                        "Here is the task output requirement: \n"
                        f"'{task_intention.replace('{', '{{').replace('}', '}}')}'.\n"
                    ),
                ),
                ("placeholder", "{messages}"),
            ]
        )
        
        # Get the last message
        last_message = messages[-1].get("content", "") if messages else ""
        if isinstance(last_message, list):
            last_message = last_message[-1].get('text', '') if last_message else ""
        
        try:
            checker_llm = format_check_prompt | cls._aux_llm.with_structured_output(output_class, method="json_schema")
            result = checker_llm.invoke({"messages": [("user", last_message)]})
            if not isinstance(result, dict):
                result = result.dict() if hasattr(result, 'dict') else result.model_dump()
            return result
        except Exception as e:
            logger.warning(f"Error in result_formatting: {e}")
            return None

    @classmethod
    def _result_formatting_llm_free(cls, messages: List[Dict[str, str]]) -> str:
        """
        Extract solution content without using LLM - just extract from <solution> tags.
        """
        if not messages:
            return ""
        
        last_message = messages[-1].get("content", "")
        if isinstance(last_message, list):
            last_message = last_message[-1].get('text', '') if last_message else ""
        
        # Extract content between solution tags
        solution_match = re.search(r'<solution>(.*?)</solution>', last_message, re.DOTALL)
        if solution_match:
            return solution_match.group(1).strip()
        return ""

    @classmethod
    def _get_prompt_from_instance(cls, task, instance_id) -> str:
        """Get the prompt for an instance from the task."""
        try:
            example = task.get_example(instance_id)
            return example.get("prompt", "")
        except Exception:
            return ""

    @classmethod
    def _evaluate_with_rubric(
        cls,
        task,
        instance_id: Any,
        parsed_output: Optional[Dict],
        raw_output: str,
        task_name: str
    ) -> Dict[str, float]:
        """
        Evaluate the trajectory using the task-specific rubric and LLM judge.
        Returns the rubric scores normalized to max 5.
        Includes "rubric_eval_failed" bool to signal upstream masking.
        """
        import time as _time

        _FAILED_RESULT = {
            "rubric_reward": 0.0,
            "output_grading": 0.0,
            "methodology_knowhow": 0.0,
            "code_data_handling": 0.0,
            "reasoning_coherence": 0.0,
            "rubric_total": 0.0,
            "rubric_rationale": "",
            "rubric_weaknesses": [],
            "rubric_eval_failed": True,
        }

        try:
            if not hasattr(task, 'get_rubric'):
                logger.warning(f"Task {task_name} does not have get_rubric method")
                result = _FAILED_RESULT.copy()
                result["rubric_rationale"] = "Task does not support rubric evaluation"
                return result
            
            rubric = task.get_rubric(instance_id, parsed_output, raw_output)
            rubric = rubric + "\n" + RUBRIC_MODIFIER
            
            judge_messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=rubric)
            ]
            
            max_retries = 3
            last_error = None
            for attempt in range(max_retries):
                try:
                    eval_output: CriticMetrics = cls._llm_judge.invoke(judge_messages)
                    break
                    # if attempt < max_retries - 1:
                    #     eval_output: CriticMetrics = cls._llm_judge.invoke(judge_messages)
                    # else:
                    #     # Last retry: fall back to judge without thinking to avoid
                    #     # the "structured output + thinking" incompatibility
                    #     logger.warning(
                    #         f"Rubric eval attempt {attempt + 1}/{max_retries} for {task_name}: "
                    #         f"falling back to non-thinking judge"
                    #     )
                    #     fallback_llm = ChatAnthropic(
                    #         model=cls._llm_judge.bound.first.model if hasattr(cls._llm_judge, 'bound') else "claude-sonnet-4-5",
                    #         temperature=1.0,
                    #         max_tokens=32768,
                    #     )
                    #     fallback_judge = fallback_llm.with_structured_output(CriticMetrics)
                    #     eval_output = fallback_judge.invoke(judge_messages)
                    # break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        backoff = 2 ** (attempt + 1)
                        logger.warning(
                            f"Rubric eval attempt {attempt + 1}/{max_retries} failed for {task_name}: {e}. "
                            f"Retrying in {backoff}s..."
                        )
                        _time.sleep(backoff)
                    else:
                        logger.warning(
                            f"Rubric eval failed after {max_retries} attempts for {task_name}: {e}"
                        )
                        import traceback
                        traceback.print_exc()
                        result = _FAILED_RESULT.copy()
                        result["rubric_rationale"] = f"Error during evaluation after {max_retries} retries: {str(e)}"
                        return result
            
            # Validate scores are within bounds
            assert 0 <= eval_output.output_grading <= 20, f"output_grading out of bounds: {eval_output.output_grading}"
            assert 0 <= eval_output.methodology_knowhow <= 10, f"methodology_knowhow out of bounds: {eval_output.methodology_knowhow}"
            assert 0 <= eval_output.code_data_handling <= 10, f"code_data_handling out of bounds: {eval_output.code_data_handling}"
            assert 0 <= eval_output.reasoning_coherence <= 10, f"reasoning_coherence out of bounds: {eval_output.reasoning_coherence}"
            assert 0 <= eval_output.total <= 50, f"total out of bounds: {eval_output.total}"
            
            computed_total = (
                eval_output.output_grading + 
                eval_output.methodology_knowhow + 
                eval_output.code_data_handling + 
                eval_output.reasoning_coherence
            )
            
            if abs(computed_total - eval_output.total) > 0.01:
                logger.warning(
                    f"Score mismatch: computed={computed_total}, reported={eval_output.total}. "
                    f"Using computed total."
                )
                eval_output.total = computed_total
            
            rubric_reward = eval_output.total / 10.0
            
            return {
                "rubric_reward": rubric_reward,
                "output_grading": eval_output.output_grading,
                "methodology_knowhow": eval_output.methodology_knowhow,
                "code_data_handling": eval_output.code_data_handling,
                "reasoning_coherence": eval_output.reasoning_coherence,
                "rubric_total": eval_output.total,
                "rubric_rationale": eval_output.rationale,
                "rubric_weaknesses": eval_output.weaknesses,
                "rubric_eval_failed": False,
            }
            
        except Exception as e:
            logger.warning(f"Error in rubric evaluation for {task_name}: {e}")
            import traceback
            traceback.print_exc()
            result = _FAILED_RESULT.copy()
            result["rubric_rationale"] = f"Error during evaluation: {str(e)}"
            return result

    @staticmethod
    def _per_turn_format_check(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Check format validity per assistant turn. Returns list of per-turn results."""

        def _check_block(content: str, *, is_last: bool) -> tuple:
            """Returns (is_valid, error_description)."""
            low = content.lower()
            stripped = low.rstrip()

            if not low.lstrip().startswith("<think>"):
                return False, "does not start with <think>"
            if low.count("<think>") != 1 or low.count("</think>") != 1:
                return False, f"has {low.count('<think>')} <think> and {low.count('</think>')} </think> (expected exactly 1 each)"
            if not (stripped.endswith("</execute>") or stripped.endswith("</solution>")):
                return False, "does not end with </execute> or </solution>"

            parts = low.split("</think>", 1)
            after_think = parts[1]
            exec_pos = after_think.find("<execute>")
            sol_pos = after_think.find("<solution>")

            if exec_pos == -1 and sol_pos == -1:
                return False, "no action tag after </think>"

            if exec_pos == -1:
                outer_is_execute = False
            elif sol_pos == -1:
                outer_is_execute = True
            else:
                outer_is_execute = exec_pos < sol_pos

            if outer_is_execute and not stripped.endswith("</execute>"):
                return False, "outer is <execute> but doesn't end with </execute>"
            if not outer_is_execute and not stripped.endswith("</solution>"):
                return False, "outer is <solution> but doesn't end with </solution>"

            if is_last and outer_is_execute:
                return False, "final turn uses <execute> instead of <solution>"
            if not is_last and not outer_is_execute:
                return False, "non-final turn uses <solution> instead of <execute>"

            if "<think>" in after_think or "</think>" in after_think:
                return False, "extra <think>/<\\/think> tags after the think block"

            return True, "valid"

        assistant_indices = [idx for idx, m in enumerate(messages) if m.get("role") == "assistant"]
        if not assistant_indices:
            return []

        last_assistant_idx = assistant_indices[-1]
        results = []
        for idx in assistant_indices:
            content = messages[idx].get("content", "")
            is_last_msg = (idx == last_assistant_idx)
            is_valid, error_msg = _check_block(content, is_last=is_last_msg)
            results.append({
                "msg_index": idx,
                "is_last": is_last_msg,
                "is_valid": is_valid,
                "error": error_msg,
            })
        return results

    @staticmethod
    def _build_turn_mapping(messages: List[Dict[str, str]]) -> Dict[str, int]:
        """Map 'Turn N' / 'FINAL ANSWER' labels to assistant message indices in the messages list."""
        mapping = {}
        i = 0
        turn_num = 0

        if i < len(messages) and messages[i].get('role') == 'system':
            i += 1
        if i < len(messages) and messages[i].get('role') == 'user':
            i += 1

        while i < len(messages):
            msg = messages[i]
            if msg.get('role') == 'assistant':
                if i + 1 < len(messages) and messages[i + 1].get('role') == 'user':
                    turn_num += 1
                    mapping[f"Turn {turn_num}"] = i
                    i += 2
                else:
                    mapping["FINAL ANSWER"] = i
                    i += 1
            else:
                i += 1

        return mapping

    @classmethod
    def _generate_corrections(
        cls,
        messages: List[Dict[str, str]],
        trajectory_text: str,
        rubric_results: Dict[str, Any],
        ft_reward: float,
        max_corrections: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate up to max_corrections corrections for problematic turns using the LLM."""
        import time as _time

        turn_mapping = cls._build_turn_mapping(messages)
        format_failed = ft_reward == 0.0
        weaknesses = rubric_results.get("rubric_weaknesses", [])
        rationale = rubric_results.get("rubric_rationale", "")

        if format_failed:
            per_turn = cls._per_turn_format_check(messages)
            reverse_mapping = {v: k for k, v in turn_mapping.items()}
            format_lines = []
            for info in per_turn:
                label = reverse_mapping.get(info["msg_index"], f"msg_idx={info['msg_index']}")
                status = "VALID" if info["is_valid"] else f"INVALID — {info['error']}"
                format_lines.append(f"  {label}: {status}")
            format_text = "\n".join(format_lines) if format_lines else "  (no assistant turns)"

            rubric_note = ""
            if weaknesses:
                rubric_note = (
                    "\n\nRUBRIC WEAKNESSES (secondary — fix format first):\n"
                    + "\n".join(f"  - {w}" for w in weaknesses)
                )

            human_content = (
                f"Here is an agent's trajectory for a biomedical task:\n\n"
                f"{trajectory_text}\n\n"
                f"FORMAT VALIDATION RESULTS (per turn):\n{format_text}\n\n"
                f"The format check FAILED (ft_reward=0), so the total reward is 0 regardless of rubric quality. "
                f"Fix format errors to unblock the reward signal.{rubric_note}\n\n"
                f"Generate up to {max_corrections} corrections targeting turns with format errors. "
                f"Each correction MUST:\n"
                f"1. Target a specific turn (e.g., 'Turn 1', 'Turn 2', 'FINAL ANSWER')\n"
                f"2. Provide a corrected assistant message that follows ALL format rules\n"
                f"3. Preserve the intent of the original message while fixing the format\n"
                f"Return fewer corrections if there are fewer format errors. Return null for unused slots."
            )
            system_prompt = CORRECTION_SYSTEM_PROMPT_FORMAT
            correction_mode = "format"
        else:
            rubric_section = "RUBRIC EVALUATION WEAKNESSES:\n" + "\n".join(f"  - {w}" for w in weaknesses)
            if rationale:
                rubric_section += f"\n\nDETAILED RATIONALE:\n{rationale}"
            scores = {
                "output_grading": rubric_results.get("output_grading", "?"),
                "methodology": rubric_results.get("methodology_knowhow", "?"),
                "code_handling": rubric_results.get("code_data_handling", "?"),
                "reasoning": rubric_results.get("reasoning_coherence", "?"),
            }
            scores_text = ", ".join(f"{k}={v}" for k, v in scores.items())

            human_content = (
                f"Here is an agent's trajectory for a biomedical task:\n\n"
                f"{trajectory_text}\n\n"
                f"RUBRIC SCORES: {scores_text}\n\n"
                f"{rubric_section}\n\n"
                f"According to the rubric scorings and identified weaknesses, generate up to {max_corrections} "
                f"corrections targeting turns that would most improve the trajectory's quality. "
                f"Each correction MUST:\n"
                f"1. Target a specific turn (e.g., 'Turn 1', 'Turn 2', 'FINAL ANSWER')\n"
                f"2. Provide an improved assistant message addressing the rubric weaknesses\n"
                f"3. Preserve the overall approach while improving methodology, code quality, or reasoning\n"
                f"Return fewer corrections if there are fewer issues. Return null for unused slots."
            )
            system_prompt = CORRECTION_SYSTEM_PROMPT_RUBRIC
            correction_mode = "rubric"

        correction_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]

        prompt_without_trajectory = human_content.replace(trajectory_text, "<TRAJECTORY_OMITTED>")
        logger.info(
            f"Correction prompt [{correction_mode}] (ft_reward={ft_reward}, weaknesses={len(weaknesses)}):\n"
            f"--- SYSTEM ---\n{system_prompt}\n"
            f"--- HUMAN (trajectory omitted) ---\n{prompt_without_trajectory}"
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result: CorrectionOutput = cls._correction_llm.invoke(correction_messages)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff = 2 ** (attempt + 1)
                    logger.warning(f"Correction generation attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {backoff}s...")
                    _time.sleep(backoff)
                else:
                    logger.warning(f"Correction generation failed after {max_retries} attempts: {e}")
                    return []

        corrections = []
        for slot in [result.correction_1, result.correction_2, result.correction_3, result.correction_4, result.correction_5]:
            if slot is None:
                continue
            target = slot.target.strip()
            msg_idx = turn_mapping.get(target)
            if msg_idx is None:
                for key, idx in turn_mapping.items():
                    if key.lower().replace(" ", "") == target.lower().replace(" ", ""):
                        msg_idx = idx
                        break
            if msg_idx is None:
                logger.warning(f"Correction targets unknown turn '{target}', skipping. Available: {list(turn_mapping.keys())}")
                continue
            corrections.append({
                "assistant_msg_index": msg_idx,
                "correction_text": slot.correction,
                "target_label": target,
            })

        logger.info(f"Generated {len(corrections)} corrections targeting: {[c['target_label'] for c in corrections]}")
        if corrections:
            sample = random.choice(corrections)
            sample_idx = sample["assistant_msg_index"]
            original_text = messages[sample_idx].get("content", "") if 0 <= sample_idx < len(messages) else "<out of range>"
            logger.info(
                f"Sample correction [{sample['target_label']}] (msg_idx={sample_idx}):\n"
                f"  ORIGINAL: {original_text}\n"
                f"  CORRECTED: {sample['correction_text']}"
            )
        return corrections

    @classmethod
    def compute_rewards(
        cls,
        instance: Any,
        solution: Optional[str],
        messages: List[Dict[str, str]],
        *,
        instance_id: Optional[Any] = None,
        task_name: Optional[str] = None,
        model: str = "claude-sonnet-4-5",
        generate_corrections: bool = False,
        max_corrections: int = 5,
    ) -> Dict[str, Any]:
        """
        Compute rewards using:
        1. GT reward from task.reward() with parsed output
        2. Rubric reward from LLM critic (normalized to 5)
        3. Format reward from format validation (max 1)
        
        Total reward = (gt_reward + rubric_reward) * ft_reward (max score 6)
        
        Returns:
            Dictionary containing:
            - score: total reward (max 7)
            - gt_reward: ground truth reward from task.reward()
            - rubric_reward: LLM-based rubric reward (max 5)
            - ft_reward: format validation reward (max 1)
            - rubric_details: detailed rubric scores and rationale
        """
        cls._ensure_initialized(model=model)

        if not task_name:
            if isinstance(instance, dict):
                task_name = instance.get("task_name")
            elif hasattr(instance, "get"):
                task_name = instance.get("task_name")
        
        gt_reward = 0.0
        rubric_results = {
            "rubric_reward": 0.0,
            "output_grading": 0.0,
            "methodology_knowhow": 0.0,
            "code_data_handling": 0.0,
            "reasoning_coherence": 0.0,
            "rubric_total": 0.0,
            "rubric_rationale": "",
            "rubric_weaknesses": [],
            "rubric_eval_failed": False,
        }
        
        logger.info("-----------Computing rubric reward for task: %s, instance_id: %s-------------", task_name, instance_id)
        logger.info("task_name: %s", task_name)
        logger.info("instance_id: %s", instance_id)
        logger.info("solution (raw): %s", solution)
        
        # Compute format reward
        ft_reward = cls._validate_format(messages)
        logger.info("ft_reward: %s", ft_reward)

        if task_name and task_name in cls._task_mapping:
            task = cls._task_mapping[task_name]
            
            # Get the task's expected output class for parsing
            parsed_output = None
            try:
                if hasattr(task, 'output_class'):
                    output_class = task.output_class()
                    # Get the task intention/prompt for result formatting
                    task_intention = cls._get_prompt_from_instance(task, instance_id)
                    parsed_output = cls._result_formatting(output_class, task_intention, messages)
                    logger.info("parsed_output: %s", parsed_output)
            except Exception as e:
                logger.warning(f"Error parsing output for {task_name}: {e}")
                # Fallback: try to use raw solution
                parsed_output = solution
            
            # Compute GT reward using parsed output
            if parsed_output is not None:
                try:
                    inp = instance_id
                    gt_reward = float(task.reward(inp, parsed_output))
                    logger.info("gt_reward: %s", gt_reward)
                except Exception as e:
                    logger.warning(f"Error computing GT reward for {task_name}: {e}")
                    gt_reward = 0.0
            else:
                logger.warning(f"No parsed output available for {task_name}")
                gt_reward = 0.0
            
            # Compute rubric reward using LLM judge
            raw_output = format_messages_to_text(messages)
            rubric_results = cls._evaluate_with_rubric(
                task=task,
                instance_id=instance_id,
                parsed_output=parsed_output,
                raw_output=raw_output,
                task_name=task_name
            )
            logger.info("rubric_reward: %s", rubric_results["rubric_reward"])
            logger.info("rubric_details: output_grading=%s, methodology=%s, code=%s, reasoning=%s",
                       rubric_results["output_grading"],
                       rubric_results["methodology_knowhow"],
                       rubric_results["code_data_handling"],
                       rubric_results["reasoning_coherence"])
        else:
            logger.warning(f"Unexpected task name: {task_name}, or solution is None")
        
        
        # Total reward = (gt_reward + rubric_reward) * ft_reward (max score 6)
        rubric_reward = rubric_results["rubric_reward"]
        total_reward = (gt_reward + rubric_reward) * ft_reward
        
        logger.info("total_reward: %s (gt=%s + rubric=%s + ft=%s)", 
                   total_reward, gt_reward, rubric_reward, ft_reward)
        
        rubric_eval_failed = rubric_results.get("rubric_eval_failed", False)
        if rubric_eval_failed:
            logger.warning("Rubric evaluation failed for task=%s instance=%s; trajectory will be masked from training", task_name, instance_id)

        # Generate corrections if enabled
        corrections = []
        if generate_corrections and not rubric_eval_failed:
            has_assistant = any(m.get("role") == "assistant" for m in messages)
            has_issues = ft_reward == 0.0 or rubric_results.get("rubric_weaknesses", [])
            if has_assistant and has_issues:
                raw_output = format_messages_to_text(messages)
                corrections = cls._generate_corrections(
                    messages=messages,
                    trajectory_text=raw_output,
                    rubric_results=rubric_results,
                    ft_reward=ft_reward,
                    max_corrections=max_corrections,
                )
                logger.info("Corrections for task=%s instance=%s: %d generated", task_name, instance_id, len(corrections))

        return {
            "score": total_reward,
            "gt_reward": gt_reward,
            "rubric_reward": rubric_reward,
            "ft_reward": ft_reward,
            "rubric_eval_failed": rubric_eval_failed,
            "corrections": corrections,
            "rubric_details": {
                "output_grading": rubric_results["output_grading"],
                "methodology_knowhow": rubric_results["methodology_knowhow"],
                "code_data_handling": rubric_results["code_data_handling"],
                "reasoning_coherence": rubric_results["reasoning_coherence"],
                "rubric_total": rubric_results["rubric_total"],
                "rationale": rubric_results["rubric_rationale"],
                "weaknesses": rubric_results["rubric_weaknesses"]
            }
        }
