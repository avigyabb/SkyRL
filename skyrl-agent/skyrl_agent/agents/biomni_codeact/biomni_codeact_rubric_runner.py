"""
BiomniCodeActRubricTrajectory: A trajectory runner that uses LLM-based rubric rewards.

This extends the base BiomniCodeActTrajectory but uses BiomniRubricRewardAdapter
instead of BiomniRewardAdapter for evaluation.
"""

import os
import logging

logger = logging.getLogger(__name__)

from skyrl_agent.agents.biomni_codeact.biomni_codeact_runner import BiomniCodeActTrajectory


class BiomniCodeActRubricTrajectory(BiomniCodeActTrajectory):
    """
    A trajectory runner that uses LLM-based rubric rewards for evaluation.
    
    Inherits from BiomniCodeActTrajectory and only overrides evaluate_trajectory()
    to use BiomniRubricRewardAdapter instead of BiomniRewardAdapter.
    
    The rubric-based reward returns:
    - gt_reward: ground truth reward from task.reward()
    - rubric_reward: LLM-based rubric reward (max 5, normalized from 50)
    - ft_reward: format validation reward (max 1)
    - total score = gt_reward + rubric_reward + ft_reward (max 7)
    """

    async def evaluate_trajectory(self) -> None:
        data = self.data
        instance = data["instance"]
        
        instance_id = instance["instance_id"]
        result = self.result.get("results")

        skip_reason = (self.result.get("state") or {}).get("skip_reason")
        if skip_reason == "prompt_length_exceeded":
            # Reward already set to 0 during generation, nothing else to evaluate.
            self.result.setdefault("reward", 0)
            self.result.setdefault("rubric_reward", 0)
            return

        try:
            # Use the new BiomniRubricRewardAdapter with LLM-based rubric evaluation
            from skyrl_agent.tasks.biomni_rubric_reward_adapter import BiomniRubricRewardAdapter
            
            # Get LLM model for critic from environment or use default
            critic_model = os.getenv("BIOMNI_CRITIC_MODEL", "claude-sonnet-4-5")
            
            # task_name is inside instance, not at top-level data
            task_name = instance.get("task_name") if isinstance(instance, dict) else None
            
            metrics = BiomniRubricRewardAdapter.compute_rewards(
                instance=instance,
                solution=result,
                messages=self.result.get("messages", []),
                instance_id=instance_id,
                task_name=task_name,
                model=critic_model
            )
            
            self.result["reward"] = metrics["score"]
            self.result["gt_reward"] = metrics["gt_reward"]
            self.result["rubric_reward"] = metrics["rubric_reward"]
            self.result["ft_reward"] = metrics["ft_reward"]
            
            # Store detailed rubric information for logging
            self.result["rubric_details"] = metrics.get("rubric_details", {})
            
            logger.info(
                "Rubric evaluation complete for instance %s: "
                "total=%s, gt=%s, rubric=%s, ft=%s",
                instance_id,
                metrics["score"],
                metrics["gt_reward"],
                metrics["rubric_reward"],
                metrics["ft_reward"]
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error evaluating result with rubric: {e}")
            self.result["reward"] = 0
            self.result["gt_reward"] = 0
            self.result["rubric_reward"] = 0
            self.result["ft_reward"] = 0
            self.result["eval_error"] = str(e)
