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
        logger.info(f"last 2 message: {messages[-2]}\n\n{messages[-1]}")

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
        
        ft_reward = cls._validate_format(messages)
        logger.info("ft_reward: %s", ft_reward)
        total_reward = gt_reward + ft_reward
        
        return {
            "score": total_reward,
            "gt_reward": gt_reward,
            "ft_reward": ft_reward
        }
