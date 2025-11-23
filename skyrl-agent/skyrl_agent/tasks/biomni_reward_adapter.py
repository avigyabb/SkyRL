import importlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class BiomniRewardAdapter:
    """
    Lightweight bridge that reuses the Biomni task implementations shipped with
    BioAgentOS to compute rule-based rewards (e.g., screen_design precision).

    The adapter lazy-loads task classes on first use and caches the instances.
    Set the following environment variables to point it at the appropriate data:

        BIOMNI_ENV_SCREEN_ROOT  - path to BioAgentOS/biomni_env_screen checkout
        BIOMNI_SCREEN_DATA_ROOT - path containing the screen_design CSV assets
        BIOMNI_SCREEN_TOP_K     - override for screen_design top-k (default 100)
    """

    _initialized: bool = False
    _builders: Dict[str, Callable[[], Any]] = {}
    _task_cache: Dict[str, Any] = {}

    @classmethod
    def score(
        cls,
        instance: Any,
        solution: Optional[str],
        *,
        instance_id: Optional[Any] = None,
    ) -> Optional[float]:
        print(f"cls: {cls}, instance: {instance}, solution: {solution}, instance_id: {instance_id}")
        """Return a float reward if a Biomni task can score this instance."""
        if not solution:
            return 0.0

        cls._ensure_initialized()
        if not cls._builders:
            return None

        payload = cls._coerce_instance(instance)
        task_name = (payload.get("task_name") or payload.get("data_source"))
        if not task_name:
            return None
        task_name = str(task_name)

        builder = cls._builders.get(task_name)
        if not builder:
            return None

        key = instance_id
        if key is None:
            key = payload.get("instance_id") or payload.get("screen_id")
        if key is None:
            logger.debug("BiomniRewardAdapter: missing instance_id for %s", task_name)
            return None

        try:
            key = int(key)
        except (TypeError, ValueError):
            logger.debug(
                "BiomniRewardAdapter: could not coerce instance_id %r to int for %s",
                key,
                task_name,
            )
            return None

        task = cls._task_cache.get(task_name)
        if task is None:
            try:
                task = builder()
            except Exception as exc:  # pragma: no cover - best-effort integration
                logger.warning(
                    "BiomniRewardAdapter: failed to build task %s (%s)",
                    task_name,
                    exc,
                )
                return None
            cls._task_cache[task_name] = task

        solution_text = cls._normalize_solution_for_task(
            task_name, cls._extract_solution(solution), task
        )
        print(f"solution_text: {solution_text}")
        if not solution_text:
            return 0.0

        try:
            reward = task.reward(key, solution_text)
            return float(reward)
        except Exception as exc:  # pragma: no cover - depends on external assets
            logger.warning(
                "BiomniRewardAdapter: task %s failed to score instance %s (%s)",
                task_name,
                key,
                exc,
            )
            return None

    # ------------------------------------------------------------------ helpers
    @classmethod
    def _ensure_initialized(cls) -> None:
        if cls._initialized:
            return
        cls._initialized = True

        # Prefer the local BioAgentOS checkout; this avoids relying on env vars when
        # skyrl-agent is co-located with the biomni runtime checkout.
        hardcoded_root = Path("/home/ray/default/BioAgentOS/biomni_env_screen")
        root = os.getenv("BIOMNI_ENV_SCREEN_ROOT")
        if not root and hardcoded_root.exists():
            root = str(hardcoded_root)
        if not root:
            candidate = (
                Path(__file__).resolve().parents[3] / "BioAgentOS" / "biomni_env_screen"
            )
            if candidate.exists():
                root = str(candidate)
        if root and root not in sys.path:
            sys.path.append(root)

        cls._register_tasks()

    @classmethod
    def _register_tasks(cls) -> None:
        """Register Biomni tasks we know how to evaluate."""
        cls._register_screen_design()
        cls._register_crispr_delivery()
        # Additional Biomni tasks can be registered here in the future.

    @classmethod
    def _register_screen_design(cls) -> None:
        try:
            module = importlib.import_module("biomni.task.screen_design")
            ScreenDesign = getattr(module, "screen_design")
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug("BiomniRewardAdapter: screen_design import failed (%s)", exc)
            return

        def _factory():
            top_k = int(os.getenv("BIOMNI_SCREEN_TOP_K", "100"))
            return ScreenDesign(top_k=top_k)

        cls._builders["screen_design"] = _factory

    @classmethod
    def _register_crispr_delivery(cls) -> None:
        try:
            module = importlib.import_module("biomni.task.crispr_delivery")
            CrisprDelivery = getattr(module, "crispr_delivery")
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug("BiomniRewardAdapter: crispr_delivery import failed (%s)", exc)
            return

        def _factory():
            return CrisprDelivery()

        cls._builders["crispr_delivery"] = _factory

    @staticmethod
    def _coerce_instance(instance: Any) -> Dict[str, Any]:
        if instance is None:
            return {}
        if isinstance(instance, dict):
            return instance
        try:
            # pandas Series supports to_dict()
            if hasattr(instance, "to_dict"):
                return instance.to_dict()
        except Exception:
            pass
        return dict(instance) if isinstance(instance, (list, tuple)) else {}

    @staticmethod
    def _extract_solution(solution: Any) -> Optional[str]:
        if solution is None:
            return None
        if isinstance(solution, str):
            return solution.strip()
        return str(solution)

    @classmethod
    def _normalize_solution_for_task(
        cls, task_name: str, solution_text: Optional[str], task: Any
    ) -> Optional[str]:
        if not solution_text:
            return None
        if not task_name:
            return solution_text
        task_key = str(task_name).lower()
        if task_key == "crispr_delivery":
            return cls._normalize_crispr_delivery_solution(solution_text, task)
        return solution_text

    @staticmethod
    def _normalize_crispr_delivery_solution(
        solution_text: str, task: Any
    ) -> Optional[str]:
        """Extract a single-letter answer (a-f) from free-form CRISPR outputs."""
        if not solution_text:
            return None

        text = solution_text.strip()
        match = re.search(r"\b([a-f])\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()

        delivery_map = getattr(task, "delivery_methods", {}) or {}
        text_lower = text.lower()
        for letter, description in delivery_map.items():
            if description.lower() in text_lower:
                return str(letter).lower()

        text_lower = text_lower.lstrip("-: ").strip()
        if text_lower and text_lower[0] in "abcdef":
            return text_lower[0]

        return None


