import asyncio, re, json, uuid, logging, aiohttp, time, os, tempfile, copy
import sys
sys.path.append("/afs/cs.stanford.edu/u/lansong/SkyRL/")
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple, Union

import torch
# from tensordict import TensorDict
# from verl import DataProto
# from verl.utils.model import compute_position_id_with_mask
# import verl.utils.torch_functional as verl_F
# from verl.workers.agentic.biomni.prompt_manager import PromptManager
from skyrl_agent.agents.biomni_codeact.prompt_manager import PromptManager
# from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Sentinel value returned by _llm_generate when context exceeds max_prompt_len.
# This is NOT model output - it signals the run() loop to terminate cleanly.
_CONTEXT_OVERFLOW_SENTINEL = "__CONTEXT_OVERFLOW__"



# -- Qwen-3 chat templates ------------------------------------------------

# Load chat template from external jinja file
_CHAT_TEMPLATE_PATH = Path(__file__).parent / "biomni_qwen3.jinja"
with open(_CHAT_TEMPLATE_PATH, "r") as f:
    gen_chat_template = f.read()



def convert_right_padding_to_left(tokenizer, input_ids, attention_mask, device, max_len=None):
    """
    Converts right-padded tensors to left-padded tensors with optional custom length.
    
    Args:
        tokenizer: The tokenizer object with pad_token_id attribute
        input_ids (torch.Tensor): Right-padded input IDs tensor of shape [batch_size, seq_length]
        attention_mask (torch.Tensor): Right-padded attention mask tensor of shape [batch_size, seq_length]
        device: The device to place the new tensors on
        max_len (int, optional): The desired maximum length of the returned tensors.
                                If None, uses the original sequence length.
    
    Returns:
        tuple: (left_padded_input_ids, left_padded_attention_mask)
    """
    batch_size, orig_seq_length = input_ids.size()
    
    # Use original length if max_len is not specified
    seq_length = max_len if max_len is not None else orig_seq_length
    
    # Create new tensors with the desired size
    left_padded_input_ids = torch.full((batch_size, seq_length), 
                                     tokenizer.pad_token_id, 
                                     dtype=input_ids.dtype, 
                                     device=device)
    left_padded_attention_mask = torch.zeros((batch_size, seq_length), 
                                           dtype=attention_mask.dtype, 
                                           device=device)
    
    for i in range(batch_size):
        # Get the non-padded length of this sequence
        seq_len = attention_mask[i].sum().item()
        
        # Trim sequence if it's longer than max_len
        if seq_len > seq_length:
            logger.warning(f"Trimming sequence length from {seq_len} to {seq_length}")
            seq_len = seq_length
        
        # Calculate the offset for left padding
        offset = seq_length - seq_len
        
        # Copy the non-padded tokens to the end
        left_padded_input_ids[i, offset:] = input_ids[i, :seq_len]
        left_padded_attention_mask[i, offset:] = 1  # Set attention mask for non-padding tokens
    
    return left_padded_input_ids, left_padded_attention_mask

def pad_to_max_length_right(tokenizer, encodings, max_length, device):
    """
    Pads tokenizer outputs to a specific maximum length with configurable padding side.
    
    Args:
        tokenizer: The tokenizer object with pad_token_id attribute
        encodings (dict): Dictionary containing 'input_ids', 'attention_mask', and optionally 'assistant_masks'
        max_length (int): The desired maximum length to pad to
        device: The device to place the tensors on
        
    Returns:
        dict: Dictionary with padded tensors for 'input_ids', 'attention_mask', and 'assistant_masks' if present
    """
    batch_size = len(encodings['input_ids'])
    
    # Initialize output tensors
    padded_input_ids = torch.full((batch_size, max_length), 
                                tokenizer.pad_token_id, 
                                dtype=torch.long, 
                                device=device)
    padded_attention_mask = torch.zeros((batch_size, max_length), 
                                      dtype=torch.long, 
                                      device=device)
    padded_assistant_mask = torch.zeros((batch_size, max_length), 
                                          dtype=torch.long, 
                                          device=device)
    
    # Fill tensors with actual values
    num_trimmed = 0
    for i in range(batch_size):
        seq_len = encodings["attention_mask"][i].sum().item() if isinstance(encodings["attention_mask"][i], torch.Tensor) else sum(encodings["attention_mask"][i])
        # Trim if longer than max_length
        actual_len = min(seq_len, max_length)
        if seq_len > max_length:
            logger.warning(
                f"Trimming sequence length from {seq_len} to {actual_len} for batch item {i}"
            )
            num_trimmed += 1
        
        # Right padding - copy sequence data to the beginning
        padded_input_ids[i, :actual_len] = torch.tensor(encodings['input_ids'][i][:actual_len], device=device)
        padded_attention_mask[i, :actual_len] = torch.tensor(encodings['attention_mask'][i][:actual_len], device=device)
        padded_assistant_mask[i, :actual_len] = torch.tensor(encodings['assistant_masks'][i][:actual_len], device=device)
    
    logger.info(f"Trimmed {num_trimmed*100 / max(batch_size, 1)}% of samples in the batch of size {batch_size}")
    return padded_input_ids, padded_attention_mask, padded_assistant_mask



class BiomniRuntimeClient:
    """Thin async wrapper around server.py endpoints."""
    def __init__(self, base_url: str = "http://localhost:8000", request_timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.session_id: Optional[str] = None
        self._client: Optional[aiohttp.ClientSession] = None
        self._request_timeout = float(request_timeout)

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self._request_timeout)
        self._client = aiohttp.ClientSession(timeout=timeout)      # one connection pool per runtime
        self.session_id = await self._start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self.session_id and self._client:
                try:
                    await self._client.post(
                        f"{self.base}/delete_runtime",
                        json={"session_id": self.session_id},
                        timeout=self._request_timeout,
                    )
                except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError, aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Failed to delete runtime session {self.session_id}: {e}")
                    # Don't raise here since we're in cleanup
        finally:
            if self._client and not self._client.closed:
                await self._client.close()
            self._client = None
            self.session_id = None

    # low-level helpers -------------------------------------------------
    async def _start(self) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self._client.post(
                    f"{self.base}/start_runtime", timeout=self._request_timeout
                ) as r:
                    r.raise_for_status()
                    return (await r.json())["session_id"]
            except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError, aiohttp.ClientError) as e:
                logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1.0 * (attempt + 1))  # exponential backoff
            except asyncio.TimeoutError as e:
                logger.warning(f"Timeout error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1.0 * (attempt + 1))

    async def execute(self, code: str, timeout: int = 600) -> str:
        """Run *code* inside the persistent namespace of this session."""
        payload = {"session_id": self.session_id,
                   "code": code,
                   "timeout_seconds": timeout}
        max_retries = 1
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                async with self._client.post(f"{self.base}/execute", json=payload,
                                             timeout=timeout+5) as r:
                    # Check for 404 with "Unknown session_id" error
                    if r.status == 404:
                        logger.warning(f"Session {self.session_id} not found on server, creating new session")
                        self.session_id = await self._start()
                        payload["session_id"] = self.session_id
                        continue
                    
                    r.raise_for_status()
                    output = (await r.json())["output"]
                    
                    duration = time.time() - start_time
                    # logger.info(f"Execution finished in {duration:.2f}s")
                    if duration > 180:
                        logger.warning(f"Code execution took {duration:.2f} seconds. Code:\n{code}\nOutput:\n{output}")
                    
                    return output
            except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError, aiohttp.ClientError) as e:
                logger.warning(f"Connection error during execute on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to execute code after {max_retries} attempts: {e}")
                await asyncio.sleep(1.0 * (attempt + 1))  # exponential backoff
            except asyncio.TimeoutError as e:
                logger.warning(f"Timeout error during execute on attempt {attempt + 1}/{max_retries}: {e}. Code being executed:\n{code}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Code execution timed out after {max_retries} attempts: {e}")
                await asyncio.sleep(1.0 * (attempt + 1))


# ----------------------------------------------------------------------
# Helper utils
# ----------------------------------------------------------------------
# Use negative lookahead to prevent matching content that contains nested tags
# This ensures <solution>...<solution>...</solution> doesn't match the outer pair incorrectly
_TAG_RGX = {
    "solution": re.compile(r"<solution>((?:(?!<solution>|</solution>).)*)</solution>", re.DOTALL | re.IGNORECASE),
    "execute":  re.compile(r"<execute>((?:(?!<execute>|</execute>).)*)</execute>",  re.DOTALL | re.IGNORECASE),
    "think":    re.compile(r"<think>(.*?)</think>",      re.DOTALL | re.IGNORECASE),
}

def _parse_first(match_type: str, text: str) -> Optional[str]:
    m = _TAG_RGX[match_type].search(text)
    return m.group(1).strip() if m else None

def _parse_last(match_type: str, text: str) -> Optional[str]:
    """Parse the LAST occurrence of the given tag type.
    
    This handles cases where the LLM mentions tags in its thinking before actually using them,
    e.g., '<think>I should now provide the final answer in the required format using the <solution> tags.</think>'
    """
    matches = _TAG_RGX[match_type].findall(text)
    return matches[-1].strip() if matches else None

def _parse_after_think(match_type: str, text: str) -> Optional[str]:
    """Parse the first occurrence of match_type AFTER </think>.
    
    This correctly handles cases where the model mentions tags in its thinking:
    e.g., '<think>I should use <execute></execute> tags</think><execute>real_code()</execute>'
    """
    low = text.lower()
    think_end = low.find("</think>")
    if think_end == -1:
        after = text  # no </think> found, search full text as fallback
    else:
        after = text[think_end + len("</think>"):]
    m = _TAG_RGX[match_type].search(after)
    return m.group(1).strip() if m else None


# ----------------------------------------------------------------------
# One Agent
# ----------------------------------------------------------------------
class BiomniCodeActAgent:
    """
    Rollout loop for a single problem instance.

    Parameters
    ----------
    prompt: str
        The initial user prompt / task description.
    runtime: BiomniRuntimeClient
        A *connected* runtime. The agent does NOT own it - caller decides lifespan.
    infer_engine, tokenizer, sampling_params
        Passed straight to sglang.
    max_iterations : int
        Hard limit to avoid infinite loops.
    """
    def __init__(
        self,
        prompt: str,
        instance_id: int,
        task_name: str,
        runtime: BiomniRuntimeClient,
        infer_engine,
        tokenizer: PreTrainedTokenizerBase,
        sampling_params: Dict[str, Any],
        max_prompt_len: int = 31744,
        max_iterations: int = 32,
        qwen3_enable_thinking: bool = True,
    ):
        self.runtime = runtime
        self.engine = infer_engine
        self.tok = tokenizer
        self.sampling_params = sampling_params
        self.instance_id = instance_id
        self.task_name = task_name
        self.max_prompt_len = max_prompt_len
        self.max_iterations = max_iterations
        self.qwen3_enable_thinking = qwen3_enable_thinking
        # Use relative path from this module for portability
        self.prompt_manager = PromptManager(tool_path=str(Path(__file__).parent / "tool"))

        # -- conversation memory ------------------------------------------------
        self.messages = self.prompt_manager.get_initial_messages(prompt, task_name)
        self.log: List[Dict[str, str]] = []   # optional external logging
        
        # Track logprobs for each assistant generation (for TIS)
        # Each entry is a list of logprobs for one generation step
        self.all_logprobs: List[List[float]] = []

    def _build_prompt_input_ids(self) -> List[int]:
        return self.tok.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            chat_template=gen_chat_template,
            enable_thinking=self.qwen3_enable_thinking,
        )

    def estimate_initial_prompt_tokens(self) -> int:
        """Return the token count for the current conversation state (primarily the initial prompt)."""
        return len(self._build_prompt_input_ids())

    # ------------------------------------------------------------------
    # generation & routing
    # ------------------------------------------------------------------
    async def _llm_generate(self) -> Tuple[str, Optional[List[float]]]:
        """Call sglang engine asynchronously and return *raw* assistant string and logprobs.
        
        Returns:
            Tuple of (text, logprobs) where logprobs may be None if not available.
        """
        input_ids = self._build_prompt_input_ids()
        if len(input_ids) > self.max_prompt_len:
            logger.warning(
                f"Instance {self.instance_id}: Context ({len(input_ids)} tokens) exceeded "
                f"max_prompt_len ({self.max_prompt_len}). Stopping agent."
            )
            return _CONTEXT_OVERFLOW_SENTINEL, None
        
        max_retries = 1
        for _ in range(max_retries):
            res = await self.engine.async_generate(
                input_ids=input_ids,
                sampling_params=self.sampling_params,
            )
            think_open = res["text"].count("<think>")
            think_close = res["text"].count("</think>")
            execute_open = res["text"].count("<execute>")
            execute_close = res["text"].count("</execute>")
            solution_open = res["text"].count("<solution>")
            solution_close = res["text"].count("</solution>")

            if (think_open != think_close or 
                execute_open != execute_close or 
                solution_open != solution_close):
                continue
            else:
                break
        
        # Extract logprobs if available
        logprobs = res.get("logprobs")
        if logprobs is None:
            logger.warning(
                f"[TIS] No logprobs returned for instance {self.instance_id}. "
                "Ensure sampling_params.logprobs is set (e.g., logprobs=0) in the agent config if you want to use TIS."
            )
        return res["text"], logprobs

    async def run(self) -> Dict[str, Any]:
        """
        Execute the interaction loop.

        Returns
        -------
        dict
            {
              "messages": <full conversation>,
              "solution": str | None,
              "iterations": int,
              "logprobs": List[List[float]] | None - logprobs for each assistant generation
            }
        """
        solution: Optional[str] = None

        for step in range(1, self.max_iterations + 1):
            assistant_reply, step_logprobs = await self._llm_generate()
            
            # This is a sentinel, not actual model output - stop the loop.
            # Concatenate to the last user message to avoid consecutive user messages
            if assistant_reply == _CONTEXT_OVERFLOW_SENTINEL:
                context_limit_msg = "\n\n[CONTEXT_LIMIT] Context window exceeded. Terminating."
                if self.messages and self.messages[-1]["role"] == "user":
                    self.messages[-1]["content"] += context_limit_msg
                    if self.log and self.log[-1]["role"] == "user":
                        self.log[-1]["content"] += context_limit_msg
                else:
                    # Shouldn't happen in normal flow, but handle gracefully
                    logger.warning(f"Last user message not found while handling context overflow sentinel.")
                    logger.warning(f"Last message: {self.messages[-1]}")
                    self.messages.append({"role": "user", "content": context_limit_msg.strip()})
                    self.log.append({"role": "user", "content": context_limit_msg.strip()})
                break
            
            # Track logprobs for this generation step (for TIS)
            if step_logprobs is not None:
                self.all_logprobs.append(step_logprobs)

            # -- parse ----------------------------------------------------------
            # No stop sequences -- model generates freely. The format reward teaches
            # clean stopping. We parse action tags only AFTER </think> to allow the
            # model to reason about its own format in think blocks.
            
            # Prepend <think>\n since vLLM returns text AFTER the generation prompt
            # which already includes <think>\n. This ensures:
            # 1. Format validation sees the complete message starting with <think>
            # 2. Stored message matches what training will see after template encoding
            if not assistant_reply.lstrip().lower().startswith("<think>"):
                assistant_reply = "<think>\n" + assistant_reply
            
            self.messages.append({"role": "assistant", "content": assistant_reply})
            self.log.append({"role": "assistant", "content": assistant_reply})
            
            sol = _parse_after_think("solution", assistant_reply)
            code = _parse_after_think("execute", assistant_reply)
            
            
            if sol and code:
                self.messages.append({"role": "user", "content": "Multiple tags (<execute> and <solution>) detected.\nPlease include only one of them in your response."})
                logger.warning(f"Multiple tags (<execute> and <solution>) detected from assistant reply: {assistant_reply}")
                self.log.append({"role": "user", "content": "Multiple tags (<execute> and <solution>) detected.\nPlease include only one of them in your response."})
                error_count = sum(
                    1
                    for m in self.messages
                    if m["role"] == "user" and "Multiple tags (<execute> and <solution>) detected." in m["content"]
                )
                if error_count >= 2:
                    # self.messages.append(
                    #     {"role": "user",
                    #     "content": "Execution terminated due to repeated parsing errors."}
                    # )
                    logger.warning(f"Execution terminated due to repeated parsing errors.")
                    logger.warning(f"messages: {self.messages}")
                    break



            if sol:
                solution = sol
                break
            
            if code is not None:
                try:
                    out = await self.runtime.execute(code)
                except Exception as e:
                    out = f"[runtime-error] {e}"
                
                # Check if context is getting close to max_prompt_len and add warning to observation
                context_warning = ""
                current_tokens = len(self._build_prompt_input_ids())
                if current_tokens > self.max_prompt_len - 2048:
                    context_warning = (
                        "\n\n[CONTEXT WARNING] You are running low on context space. "
                        "Please provide your final answer wrapped in <solution></solution> now."
                    )
                    logger.warning(
                        f"Instance {self.instance_id}: Context at {current_tokens}/{self.max_prompt_len} tokens. "
                        "Prompted agent for final answer."
                    )
                
                # feed runtime output back as user message (with optional warning appended)
                observation_content = f"<observation>{out}</observation>{context_warning}"
                self.messages.append({"role": "user", "content": observation_content})
                self.log.append({"role": "user", "content": observation_content})
                continue

            # optional <think> branch – do nothing but continue
            if _parse_first("think", assistant_reply) is not None:
                continue

            # Malformed – corrective feedback
            
            self.messages.append(
                {"role": "user",
                 "content": "There are no tags (e.g. <execute><solution>). "
                            "Please follow the instruction, fix and update."}
            )
        
        if not solution:
            logger.warning(f"No solution found for instance {self.instance_id} after {step} iterations, showing the last two message...")
            import json
            print(json.dumps(self.messages[-2:], indent=2))

        return {
            "messages": self.messages,
            "solution": solution,
            "iterations": step,
            "logprobs": self.all_logprobs if self.all_logprobs else None,
        }

# new
if __name__ == "__main__":
    """Minimal runnable example for BiomniCodeActAgent.

    Requirements at runtime:
      - Biomni execution server reachable at RUNTIME_URL (default http://localhost:8000)
      - An LLM backend reachable at LLM_API_URL for OpenAI-compatible inference
    """
    import os
    import json
    import asyncio
    from transformers import AutoTokenizer
    from skyrl_agent.integrations.openai import OpenAIBackend, OpenAIBackendConfig

    # Configuration via environment variables with sensible defaults
    RUNTIME_URL = os.getenv("RUNTIME_URL", "http://localhost:8000")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")
    LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:30000")

    # Tokenizer and inference engine
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    backend_config = OpenAIBackendConfig(model_name=MODEL_NAME, api_url=LLM_API_URL)
    infer_engine = OpenAIBackend(infer_engine=None, cfg=backend_config)

    # Simple demo prompt and sampling params
    demo_prompt = (
        "Use <execute>...</execute> to run short Python code that prints 1+1, then return the sum inside "
        "<solution>...</solution>."
    )
    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 512,
    }

    async def main():
        async with BiomniRuntimeClient(RUNTIME_URL) as rt:
            agent = BiomniCodeActAgent(
                prompt=demo_prompt,
                instance_id=0,
                task_name="biomni_demo",
                runtime=rt,
                infer_engine=infer_engine,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                max_prompt_len=12048,
                max_iterations=8,
                qwen3_enable_thinking=True,
            )
            result = await agent.run()
            print(json.dumps(result, indent=2))

    asyncio.run(main())