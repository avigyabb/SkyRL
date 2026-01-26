import logging
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


class BiomniRubricRewardAdapter:
    """
    LLM-based rubric reward adapter for Biomni tasks.
    Uses an LLM critic to evaluate agent trajectories against task-specific rubrics.
    """
    _initialized: bool = False
    _task_mapping: Dict[str, Any] = {}
    _llm_judge = None
    _aux_llm = None

    @classmethod
    def _ensure_initialized(cls, model: str = "claude-sonnet-4-5"):
        if cls._initialized:
            return
        
        benchmark_root = '/dfs/project/bioagentos/biomni_data/benchmark/'
        
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
        cls._llm_judge = llm.with_structured_output(CriticMetrics)
        
        # Initialize auxiliary LLM for result formatting (no structured output)
        cls._aux_llm = ChatAnthropic(
            model=model,
            temperature=0.7,
            max_tokens=32768
        )
            
        cls._initialized = True

    @staticmethod
    def _validate_format(messages: List[Dict[str, str]]) -> float:
        """
        Check formatting rules:
        1. <think>...</think> ... <execute>...</execute>
        2. <think>...</think> ... <solution>...</solution>
        """
        tag_pattern = re.compile(r"</?(think|execute|solution)>", re.IGNORECASE)

        def _valid_block(content: str, *, is_last: bool) -> bool:
            if "<execute>" in content and "<solution>" in content:
                return False
            
            tags = [m.group(0) for m in tag_pattern.finditer(content)]
            if len(tags) != 4:
                return False
                
            if tags[0].lower() != "<think>" or tags[1].lower() != "</think>":
                return False
                
            second_open, second_close = tags[2], tags[3]
            if second_open.lower() not in ("<execute>", "<solution>"):
                return False
                
            expected_close = second_open.replace("<", "</")
            if second_close.lower() != expected_close.lower():
                return False
                
            if second_open.lower() == "<solution>" and not is_last:
                return False
            elif second_open.lower() != "<solution>" and is_last:
                return False
                
            think_block = content.split(tags[0], 1)[1].split(tags[1], 1)[0]
            if "<execute>" in think_block or "<solution>" in think_block:
                return False
                
            after_think = content.split(tags[1], 1)[1]
            if "<think>" in after_think or "</think>" in after_think:
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
            checker_llm = format_check_prompt | cls._aux_llm.with_structured_output(output_class)
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
        """
        try:
            # Get the rubric from the task
            if not hasattr(task, 'get_rubric'):
                logger.warning(f"Task {task_name} does not have get_rubric method")
                return {
                    "rubric_reward": 0.0,
                    "output_grading": 0.0,
                    "methodology_knowhow": 0.0,
                    "code_data_handling": 0.0,
                    "reasoning_coherence": 0.0,
                    "rubric_total": 0.0,
                    "rubric_rationale": "Task does not support rubric evaluation",
                    "rubric_weaknesses": []
                }
            
            rubric = task.get_rubric(instance_id, parsed_output, raw_output)
            rubric = rubric + "\n" + RUBRIC_MODIFIER
            
            # Invoke the LLM judge
            judge_messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=rubric)
            ]
            
            eval_output: CriticMetrics = cls._llm_judge.invoke(judge_messages)
            
            # Validate scores are within bounds
            assert 0 <= eval_output.output_grading <= 20, f"output_grading out of bounds: {eval_output.output_grading}"
            assert 0 <= eval_output.methodology_knowhow <= 10, f"methodology_knowhow out of bounds: {eval_output.methodology_knowhow}"
            assert 0 <= eval_output.code_data_handling <= 10, f"code_data_handling out of bounds: {eval_output.code_data_handling}"
            assert 0 <= eval_output.reasoning_coherence <= 10, f"reasoning_coherence out of bounds: {eval_output.reasoning_coherence}"
            assert 0 <= eval_output.total <= 50, f"total out of bounds: {eval_output.total}"
            
            # Verify itemized scores add up to total
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
            
            # Normalize total score to 5 (divide by 10)
            rubric_reward = eval_output.total / 10.0
            
            return {
                "rubric_reward": rubric_reward,
                "output_grading": eval_output.output_grading,
                "methodology_knowhow": eval_output.methodology_knowhow,
                "code_data_handling": eval_output.code_data_handling,
                "reasoning_coherence": eval_output.reasoning_coherence,
                "rubric_total": eval_output.total,
                "rubric_rationale": eval_output.rationale,
                "rubric_weaknesses": eval_output.weaknesses
            }
            
        except Exception as e:
            logger.warning(f"Error in rubric evaluation for {task_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "rubric_reward": 0.0,
                "output_grading": 0.0,
                "methodology_knowhow": 0.0,
                "code_data_handling": 0.0,
                "reasoning_coherence": 0.0,
                "rubric_total": 0.0,
                "rubric_rationale": f"Error during evaluation: {str(e)}",
                "rubric_weaknesses": []
            }

    @classmethod
    def compute_rewards(
        cls,
        instance: Any,
        solution: Optional[str],
        messages: List[Dict[str, str]],
        *,
        instance_id: Optional[Any] = None,
        task_name: Optional[str] = None,
        model: str = "claude-sonnet-4-5"
    ) -> Dict[str, Any]:
        """
        Compute rewards using:
        1. GT reward from task.reward() with parsed output
        2. Rubric reward from LLM critic (normalized to 5)
        3. Format reward from format validation (max 1)
        
        Total reward = gt_reward + rubric_reward + ft_reward (max score 7)
        
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
            "rubric_weaknesses": []
        }
        
        logger.info("-----------Computing rubric reward for task: %s, instance_id: %s-------------", task_name, instance_id)
        logger.info("task_name: %s", task_name)
        logger.info("instance_id: %s", instance_id)
        logger.info("solution (raw): %s", solution[:500] if solution else None)

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
        
        # Compute format reward
        ft_reward = cls._validate_format(messages)
        logger.info("ft_reward: %s", ft_reward)
        
        # Total reward = gt_reward + rubric_reward + ft_reward (max score 7)
        rubric_reward = rubric_results["rubric_reward"]
        total_reward = gt_reward + rubric_reward + ft_reward
        
        logger.info("total_reward: %s (gt=%s + rubric=%s + ft=%s)", 
                   total_reward, gt_reward, rubric_reward, ft_reward)
        
        return {
            "score": total_reward,
            "gt_reward": gt_reward,
            "rubric_reward": rubric_reward,
            "ft_reward": ft_reward,
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
