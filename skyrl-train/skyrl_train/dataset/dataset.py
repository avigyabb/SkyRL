import datasets
from loguru import logger
import os
from collections import Counter
from typing import List, Optional, Set
from transformers import PreTrainedTokenizerBase

# --- Task filtering (hardcoded) ---
# Column name used for task-based filtering
TASK_NAME_KEY = "task_name"

# Include ONLY these tasks (None = include all). Overrides EXCLUDE_TASKS.
TARGET_TASKS: Optional[Set[str]] = None
# Example: TARGET_TASKS = {"rare_disease_diagnosis", "crispr_delivery"}

# Exclude these tasks (None = exclude none). Ignored if TARGET_TASKS is set.
EXCLUDE_TASKS: Optional[Set[str]] = {"screen_design"}
# Example: EXCLUDE_TASKS = {"hle"}

# Max examples per task (None = no limit). Applied after include/exclude filtering.
MAX_EXAMPLES_PER_TASK: Optional[int] = None
# Example: MAX_EXAMPLES_PER_TASK = 100


class PromptDataset:
    def __init__(
        self,
        datasets: str | List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        num_workers: int = 8,
        prompt_key: str = "prompt",
        env_class_key: str = "env_class",
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompt_key = prompt_key
        self.env_class_key = env_class_key
        self.num_workers = num_workers

        self.datasets = datasets
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]

        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        loaded_datasets = []
        for source in self.datasets:
            ext = os.path.splitext(source)[-1].lower()
            if ext == ".parquet":
                ds = datasets.load_dataset("parquet", data_files=source, keep_in_memory=True)["train"]
            elif ext in [".json", ".jsonl"]:
                ds = datasets.load_dataset("json", data_files=source, keep_in_memory=True)["train"]
            else:
                # Treat as HF dataset spec: "name" or "name:split"
                dataset_name, has_split, split = source.partition(":")
                try:
                    ds_dict = datasets.load_dataset(path=dataset_name, keep_in_memory=True)
                except ValueError:
                    raise ValueError(f"Dataset `{dataset_name}` not found on Hugging Face.")
                split = split if has_split else "train"
                if split not in ds_dict:
                    raise ValueError(
                        f"Split `{split}` not found in dataset `{dataset_name}`. Configured split was `{split}` and default is `train`"
                    )
                ds = ds_dict[split]
            loaded_datasets.append(ds)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(loaded_datasets)

        logger.info(f"Total dataset size: {len(self.dataframe)}")

        # --- Task-based filtering ---
        has_task_col = TASK_NAME_KEY in self.dataframe.column_names
        if has_task_col:
            task_counts = Counter(self.dataframe[TASK_NAME_KEY])
            logger.info(f"Task distribution before filtering: {dict(task_counts)}")

        if TARGET_TASKS is not None and has_task_col:
            self.dataframe = self.dataframe.filter(
                lambda row: row[TASK_NAME_KEY] in TARGET_TASKS,
                desc=f"Keeping only target tasks: {TARGET_TASKS}",
            )
            logger.info(f"After TARGET_TASKS filter: {len(self.dataframe)} rows")
        elif EXCLUDE_TASKS is not None and has_task_col:
            self.dataframe = self.dataframe.filter(
                lambda row: row[TASK_NAME_KEY] not in EXCLUDE_TASKS,
                desc=f"Excluding tasks: {EXCLUDE_TASKS}",
            )
            logger.info(f"After EXCLUDE_TASKS filter: {len(self.dataframe)} rows")

        if MAX_EXAMPLES_PER_TASK is not None and has_task_col:
            task_counts = Counter(self.dataframe[TASK_NAME_KEY])
            keep_indices = []
            per_task_count: dict[str, int] = {}
            for idx in range(len(self.dataframe)):
                task = self.dataframe[idx][TASK_NAME_KEY]
                per_task_count.setdefault(task, 0)
                if per_task_count[task] < MAX_EXAMPLES_PER_TASK:
                    keep_indices.append(idx)
                    per_task_count[task] += 1
            self.dataframe = self.dataframe.select(keep_indices)
            capped_counts = {t: min(c, MAX_EXAMPLES_PER_TASK) for t, c in task_counts.items()}
            logger.info(f"After MAX_EXAMPLES_PER_TASK={MAX_EXAMPLES_PER_TASK}: {len(self.dataframe)} rows, per-task: {capped_counts}")

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = self.dataframe.filter(
            lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))
            <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        logger.info(f"Filtered dataset size: {len(self.dataframe)}")

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        messages = row_dict.pop(self.prompt_key)
        env_class = row_dict.pop(self.env_class_key, None)

        extra = {key: value for key, value in row_dict.items() if key != self.prompt_key and key != self.env_class_key}
        uid = str(item)

        return messages, env_class, extra, uid

    def collate_fn(self, item_list):
        all_inputs = []
        for prompt, env_class, env_extras, item_uids in item_list:
            all_inputs.append({"prompt": prompt, "env_class": env_class, "env_extras": env_extras, "uid": item_uids})
        return all_inputs

    def __len__(self):
        return len(self.dataframe)
