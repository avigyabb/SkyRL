import logging
import re
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

from skyrl_agent.agents.biomni_codeact.task.screen_design import screen_design
from skyrl_agent.agents.biomni_codeact.task.gwas_causal_gene import gwas_causal_gene
from skyrl_agent.agents.biomni_codeact.task.crispr_delivery import crispr_delivery
from skyrl_agent.agents.biomni_codeact.task.rare_disease_diagnosis import rare_disease_diagnosis
from skyrl_agent.agents.biomni_codeact.task.gwas_variant_prioritization import gwas_variant_prioritization
from skyrl_agent.agents.biomni_codeact.task.patient_gene_detection import patient_gene_detection
from skyrl_agent.agents.biomni_codeact.task.lab_bench import lab_bench
from skyrl_agent.agents.biomni_codeact.task.screen_gene_retrieval import screen_gene_retrieval


class BiomniRewardAdapter:
    _initialized: bool = False
    _task_mapping: Dict[str, Any] = {}

    @classmethod
    def _ensure_initialized(cls):
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
                logger.warning("Format violation (no <think> at start): %s", content[:200])
                return False

            # Rule 2: Exactly one <think> and one </think>
            if low.count("<think>") != 1 or low.count("</think>") != 1:
                logger.warning("Format violation (not exactly one <think>/<//think>): %s", content[:200])
                return False

            # Rule 3: Must end with </execute> or </solution>
            if not (stripped.endswith("</execute>") or stripped.endswith("</solution>")):
                logger.warning("Format violation (not ending with </execute> or </solution>): %s", content[-200:])
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
                logger.warning("Format violation (no action tag after </think>): %s", content[:200])
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
                logger.warning("Format violation (outer is <execute> but doesn't end with </execute>): %s", content[-200:])
                return False
            if not outer_is_execute and not stripped.endswith("</solution>"):
                logger.warning("Format violation (outer is <solution> but doesn't end with </solution>): %s", content[-200:])
                return False

            # Rule 6: Verify there's only ONE outer action block (not multiple sequential blocks)
            # Use find() for O(n) instead of character-by-character O(n^2) scanning
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
                        logger.warning("Format violation (multiple outer %s blocks): %s", open_tag, content[:200])
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
                logger.warning("Format violation (last message uses <execute>, expected <solution>): %s", content[:200])
                return False
            if not is_last and not outer_is_execute:
                logger.warning("Format violation (non-last message uses <solution>, expected <execute>): %s", content[:200])
                return False

            # Rule 8: No extra <think> or </think> after the think block
            if "<think>" in after_think or "</think>" in after_think:
                logger.warning("Format violation (<think>/<//think> found after think block): %s", content[:200])
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
    def compute_rewards(
        cls,
        instance: Any,
        solution: Optional[str],
        messages: List[Dict[str, str]],
        *,
        instance_id: Optional[Any] = None,
        task_name: Optional[str] = None,
    ) -> Dict[str, float]:
        cls._ensure_initialized()

        if not task_name:
            if isinstance(instance, dict):
                task_name = instance.get("task_name")
            elif hasattr(instance, "get"):
                task_name = instance.get("task_name")
        
        gt_reward = 0.0
        logger.info("-----------Computing reward for task: %s, instance_id: %s-------------", task_name, instance_id)
        logger.info("task_name: %s", task_name)
        logger.info("instance_id: %s", instance_id)
        logger.info("solution: %s", solution)


        if task_name and task_name in cls._task_mapping and solution:
            try:
                inp = instance_id
                # The original code passes instance_id (as int/index) to reward function
                gt_reward = float(cls._task_mapping[task_name].reward(inp, solution))
                logger.info("gt_reward: %s", gt_reward)
            except Exception as e:
                logger.warning(f"Error computing GT reward for {task_name}: {e}")
                gt_reward = 0.0
        else:
            logger.warning(f"Unexpected task name: {task_name}, or solution is None")
            gt_reward = 0.0
        
        if gt_reward == 0.0:
            logger.info("last message: %s", messages[-1])
        
        ft_reward = cls._validate_format(messages)
        logger.info("ft_reward: %s", ft_reward)
        total_reward = gt_reward + ft_reward
        
        return {
            "score": total_reward,
            "gt_reward": gt_reward,
            "ft_reward": ft_reward
        }
