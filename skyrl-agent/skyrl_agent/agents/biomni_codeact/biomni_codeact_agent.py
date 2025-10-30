import asyncio, re, json, uuid, logging, aiohttp, time, os, tempfile, copy
import sys
sys.path.append("/afs/cs.stanford.edu/u/lansong/SkyRL/")
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple

import torch
from tensordict import TensorDict
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.workers.agentic.biomni.prompt_manager import PromptManager
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)



# -- Qwen-3 chat templates ------------------------------------------------

gen_chat_template = r"""
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
"""

# includes everything
resp_chat_template = (
    "{% for message in messages %}"
    "{% if (message['role'] != 'assistant') %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% elif (message['role'] == 'assistant')%}"
    "{{'<|im_start|>' + message['role'] + '\\n'}}"
    "{% generation %}"
    "{{message['content'] + '<|im_end|>'}}"
    "{% endgeneration %}"
    "{{'\\n'}}"
    "{% endif %}"
    "{% endfor %}"
)

# drops previous <think> blocks
resp_chat_template_qwen3_thinking = (
    "{% for message in messages %}"
    "{% if (message['role'] != 'assistant') %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% elif (message['role'] == 'assistant')%}"
    "{{'<|im_start|>' + message['role'] + '\\n'}}"
    "{% generation %}"
    "{% set full_content = message['content'] %}"
    "{% set mycontent = message['content'] %}"
    "{% set is_last_message = loop.last and messages[-1]['role'] == 'assistant' %}"
    "{% if '</think>' in full_content and not is_last_message %}"
    "{% set mycontent = full_content.split('</think>')[-1].lstrip('\\n') %}"
    "{% endif %}"
    "{{mycontent + '<|im_end|>'}}"
    "{% endgeneration %}"
    "{{'\\n'}}"
    "{% endif %}"
    "{% endfor %}"
)



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
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base = base_url.rstrip("/")
        self.session_id: Optional[str] = None
        self._client: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._client = aiohttp.ClientSession()      # one connection pool per runtime
        self.session_id = await self._start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session_id:
            try:
                await self._client.post(f"{self.base}/delete_runtime",
                                        json={"session_id": self.session_id})
            except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError, aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Failed to delete runtime session {self.session_id}: {e}")
                # Don't raise here since we're in cleanup
            finally:
                await self._client.close()

    # low-level helpers -------------------------------------------------
    async def _start(self) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self._client.post(f"{self.base}/start_runtime") as r:
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

    async def execute(self, code: str, timeout: int = 1800) -> str:
        """Run *code* inside the persistent namespace of this session."""
        payload = {"session_id": self.session_id,
                   "code": code,
                   "timeout_seconds": timeout}
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                async with self._client.post(f"{self.base}/execute", json=payload,
                                             timeout=timeout+5) as r:
                    # Check for 404 with "Unknown session_id" error
                    if r.status == 404:
                        logger.warning(f"Session {self.session_id} not found on server, creating new session")
                        self.session_id = await self._start()
                        payload["session_id"] = self.session_id
                        continue
                    
                    r.raise_for_status()
                    return (await r.json())["output"]
            except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError, aiohttp.ClientError) as e:
                logger.warning(f"Connection error during execute on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to execute code after {max_retries} attempts: {e}")
                await asyncio.sleep(1.0 * (attempt + 1))  # exponential backoff
            except asyncio.TimeoutError as e:
                logger.warning(f"Timeout error during execute on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Code execution timed out after {max_retries} attempts: {e}")
                await asyncio.sleep(1.0 * (attempt + 1))


# ----------------------------------------------------------------------
# Helper utils
# ----------------------------------------------------------------------
_TAG_RGX = {
    "solution": re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE),
    "execute":  re.compile(r"<execute>(.*?)</execute>",  re.DOTALL | re.IGNORECASE),
    "think":    re.compile(r"<think>(.*?)</think>",      re.DOTALL | re.IGNORECASE),
}

def _parse_first(match_type: str, text: str) -> Optional[str]:
    m = _TAG_RGX[match_type].search(text)
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
        self.prompt_manager = PromptManager(tool_path="verl/workers/agentic/biomni/tool")

        # -- conversation memory ------------------------------------------------
        self.messages = self.prompt_manager.get_initial_messages(prompt, task_name)
        self.log: List[Dict[str, str]] = []   # optional external logging

    # ------------------------------------------------------------------
    # generation & routing
    # ------------------------------------------------------------------
    async def _llm_generate(self) -> str:
        """Call sglang engine asynchronously and return *raw* assistant string."""
        input_ids = self.tok.apply_chat_template(
            self.messages, add_generation_prompt=True,
            tokenize=True, chat_template=gen_chat_template, enable_thinking=self.qwen3_enable_thinking
        )
        # input_ids = self.tok.apply_chat_template(
        #     self.messages, add_generation_prompt=True,
        #     tokenize=True, enable_thinking=self.qwen3_enable_thinking
        # )
        if len(input_ids) >= self.max_prompt_len:
            return "The context is too long. Exit now."
        
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
        
        return res["text"]

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
            }
        """
        solution: Optional[str] = None

        for step in range(1, self.max_iterations + 1):
            assistant_reply = await self._llm_generate()
            if assistant_reply == "The context is too long. Exit now.":
                self.messages.append({"role": "user", "content": "The context is too long. Exit now."})
                self.log.append({"role": "user", "content": "The context is too long. Exit now."})
                break

            # -- parse ----------------------------------------------------------
            if '<execute>' in assistant_reply and '</execute>' not in assistant_reply:
                assistant_reply += '</execute>'
            if '<solution>' in assistant_reply and '</solution>' not in assistant_reply:
                assistant_reply += '</solution>'
            if '<think>' in assistant_reply and '</think>' not in assistant_reply:
                assistant_reply += '</think>'
            if '</think>' in assistant_reply and '<think>' not in assistant_reply:
                assistant_reply = "<think>" + assistant_reply
            
            if '<execute>' not in assistant_reply and '<solution>' not in assistant_reply and '<think>' not in assistant_reply:
                # treat the entire message as think
                assistant_reply = "<think>" + assistant_reply + "</think>"
            
            self.messages.append({"role": "assistant", "content": assistant_reply})
            self.log.append({"role": "assistant", "content": assistant_reply})
            
            if '<execute>' in assistant_reply and '<solution>' in assistant_reply:
                self.messages.append({"role": "user", "content": "Multiple tags (<execute> and <solution>) detected.\nPlease include only one of them in your response."})
                self.log.append({"role": "user", "content": "Multiple tags (<execute> and <solution>) detected.\nPlease include only one of them in your response."})
                error_count = sum(
                    1
                    for m in self.messages
                    if m["role"] == "user" and "Multiple tags (<execute> and <solution>) detected." in m["content"]
                )
                if error_count >= 2:
                    self.messages.append(
                        {"role": "user",
                        "content": "Execution terminated due to repeated parsing errors."}
                    )
                    break
                
            
            sol = _parse_first("solution", assistant_reply)
            if sol:
                solution = sol
                break
            
            code = _parse_first("execute", assistant_reply)
            if code is not None:
                try:
                    out = await self.runtime.execute(code)
                except Exception as e:
                    out = f"[runtime-error] {e}"
                # feed runtime output back as user message
                self.messages.append({"role": "user", "content": f"<observation>{out}</observation>"})
                self.log.append({"role": "user", "content": f"<observation>{out}</observation>"})
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
            print(f"[WARNING] No solution found for instance {self.instance_id} after {step} iterations, showing the last two message...")
            import json
            print(json.dumps(self.messages[-2:], indent=2))

        return {
            "messages": self.messages,
            "solution": solution,
            "iterations": step,
        }


# def _left_pad(input_ids: torch.Tensor,
#               mask: torch.Tensor,
#               pad_token_id: int,
#               tgt_len: int,
#               device) -> Tuple[torch.Tensor, torch.Tensor]:
#     bs, cur = input_ids.shape
#     out_ids = torch.full((bs, tgt_len), pad_token_id,
#                          dtype=torch.long, device=device)
#     out_mask = torch.zeros((bs, tgt_len),
#                            dtype=torch.long, device=device)
#     for i in range(bs):
#         seq_len = int(mask[i].sum())
#         seq_len = min(seq_len, tgt_len)
#         offset = tgt_len - seq_len
#         out_ids[i, offset:] = input_ids[i, :seq_len]
#         out_mask[i, offset:] = 1
#     return out_ids, out_mask


# def _pad_right(t: torch.Tensor, length: int, val: int) -> torch.Tensor:
#     if t.shape[1] >= length:
#         return t[:, :length]
#     pad = (0, length - t.shape[1])
#     return torch.nn.functional.pad(t, pad, value=val)


class BiomniCodeActAgentGroup:
    """
    Generate num_trajectories rollouts for each DataProto item in batch.

    Each item in batch must carry `non_tensor_batch['raw_prompt']`
    (string with the initial task prompt).  Nothing else is required.
    """
    def __init__(
        self,
        batch: DataProto,
        num_trajectories: int,
        infer_engine,
        tokenizer: PreTrainedTokenizerBase,
        sampling_params: Dict[str, Any],
        *,
        runtime_url: str = "http://localhost:8000",
        device: str | torch.device = "cpu",
        max_prompt_length: int = 31744,
        max_response_length: int = 3072,
        max_starting_message_length: int = 12000,
        max_iterations: int = 32,
        max_parallel_agents: int = 256,
        qwen3_enable_thinking: bool = True,
        remove_think_tokens: bool = False,
    ):
        self.orig_batch = batch
        self.nt = num_trajectories
        self.engine = infer_engine
        self.tok = tokenizer
        self.sampling_params = sampling_params
        
        self.sampling_params.update({
            "stop": ["</execute>", "</solution>"],
            "no_stop_trim": True,
        })

        self.runtime_url = runtime_url
        self.device = device
        self.max_prompt_len = max_prompt_length
        self.max_resp_len = max_response_length
        self.total_len = self.max_prompt_len + self.max_resp_len
        self.max_iters = max_iterations
        self.max_parallel = max_parallel_agents
        self.qwen_think = qwen3_enable_thinking
        self.remove_think_tokens = remove_think_tokens
        self.max_starting_message_length = max_starting_message_length
    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(self) -> DataProto:
        """
        Run the async pipeline on the current thread. If there is already a
        running event loop, we create a fresh one; otherwise we reuse the
        default loop.
        """
        try:
            loop = asyncio.get_event_loop()
            # If that loop is already running, we cannot call run_until_complete on it.
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # No event loop exists in this thread, so make a brand‐new one.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(self._async_run_group())
        finally:
            # Any cleanup (if you have open handles, etc.)
            self.close()
    
    def close(self):
        pass

    # ------------------------------------------------------------------
    # asyncio helpers
    # ------------------------------------------------------------------
    async def _async_run_group(self) -> DataProto:
        """
        Launch at most `self.max_parallel` roll-outs concurrently and
        **return results in canonical (instance, trajectory) order**.

        We attach (batch_idx, traj_idx) to every coroutine so that after
        `asyncio.gather` we can sort deterministically, no matter which
        tasks finished first.
        """
        sem = asyncio.Semaphore(self.max_parallel)
        tasks: list[asyncio.Task] = []

        async def _run_single(batch_idx: int, traj_idx: int, instance_id: int, prompt: str, task_name: str):
            async with sem, BiomniRuntimeClient(self.runtime_url) as rt:
                agent = BiomniCodeActAgent(
                    prompt=prompt,
                    instance_id=instance_id,
                    task_name=task_name,
                    runtime=rt,
                    infer_engine=self.engine,
                    tokenizer=self.tok,
                    sampling_params=copy.deepcopy(self.sampling_params),
                    max_prompt_len=self.max_prompt_len,
                    max_iterations=self.max_iters,
                    qwen3_enable_thinking=self.qwen_think,
                )
                result = await agent.run()
                # **Return the key together with the payload**
                return (batch_idx, traj_idx, result)

        # schedule (deterministic order)
        for batch_idx, data_item in enumerate(self.orig_batch):
            prompt = data_item.non_tensor_batch["raw_prompt"]
            instance_id = data_item.non_tensor_batch["instance_id"]
            task_name = data_item.non_tensor_batch["task_name"]
            for traj_idx in range(self.nt):
                tasks.append(asyncio.create_task(_run_single(batch_idx, traj_idx, instance_id, prompt, task_name)))

        # run all tasks
        gathered: list[tuple[int, int, dict]] = await asyncio.gather(*tasks)

        # --------------------------------------------------------------
        # Re-order into canonical layout expected by _pack_results
        # --------------------------------------------------------------
        # shape: [|batch| * nt] where fast-changing index is trajectory
        ordered: list[dict] = [None] * len(gathered)      # type: ignore
        for batch_idx, traj_idx, payload in gathered:
            ordinal = batch_idx * self.nt + traj_idx      # flat index
            ordered[ordinal] = payload

        # sanity
        assert all(x is not None for x in ordered)

        return self._pack_results(ordered)
    
    
    # async def _async_run_group(self) -> DataProto:
    #     sem = asyncio.Semaphore(self.max_parallel)
    #     tasks = []  # type: List[asyncio.Task]

    #     async def _run_single(prompt: str, instance_id):
    #         """Run one trajectory and return its result.

    #         Returning the result instead of mutating a shared list preserves
    #         the original ordering when we later await `asyncio.gather`, which
    #         guarantees that the gathered outputs correspond index-wise to the
    #         order in which the tasks were created.
    #         """
    #         async with sem, BiomniRuntimeClient(self.runtime_url) as rt:
    #             agent = BiomniCodeActAgent(
    #                 prompt=prompt,
    #                 instance_id=instance_id,
    #                 runtime=rt,
    #                 infer_engine=self.engine,
    #                 tokenizer=self.tok,
    #                 sampling_params=self.sampling_params,
    #                 max_prompt_len=self.max_prompt_len,
    #                 max_iterations=self.max_iters,
    #                 qwen3_enable_thinking=self.qwen_think,
    #             )
    #             return await agent.run()

    #     # schedule
    #     for data_item in self.orig_batch:
    #         instr, instance_id = data_item.non_tensor_batch["raw_prompt"], data_item.non_tensor_batch["instance_id"]
    #         for _ in range(self.nt):
    #             tasks.append(asyncio.create_task(_run_single(instr, instance_id)))

    #     # Gather returns results in the same order as *tasks* were added,
    #     # which matches the order of `orig_batch` expansion (deterministic).
    #     outputs: List[Dict[str, Any]] = await asyncio.gather(*tasks)
    #     return self._pack_results(outputs)

    # ------------------------------------------------------------------
    # packing into DataProto (same tensor layout as CodeAct) ------------
    # ------------------------------------------------------------------
    def _pack_results(self, outputs: List[Dict[str, Any]]) -> DataProto:
        # 1) split conversation into PROMPT (up to just before first assistant)
        #    and RESPONSE (from first assistant on)
        # prompts, responses = [], []
        # for convo in outputs:
        #     msgs = convo["messages"]
        #     split_at = next(
        #         (i for i, m in enumerate(msgs) if m["role"] == "assistant"), len(msgs)
        #     )
        #     prompts.append(msgs[:split_at])
        #     responses.append(msgs[split_at:])
        prompts, responses = [], []

        for convo in outputs:
            msgs = convo["messages"]
            split_at = next(
                (i for i, m in enumerate(msgs) if m["role"] == "assistant"),
                len(msgs),
            )
            prompts.append(msgs[:split_at])
            responses.append(msgs[split_at:])

        # 2) tokenizer encode with chat template
        prompt_enc = self.tok.apply_chat_template(
            prompts, add_generation_prompt=False, return_dict=True, padding=True
        )
        resp_enc = self.tok.apply_chat_template(
            responses, add_generation_prompt=False, return_dict=True,
            chat_template=resp_chat_template_qwen3_thinking if self.remove_think_tokens else resp_chat_template,
            return_assistant_tokens_mask=True, padding=True
        )

        pad_id = self.tok.pad_token_id or self.tok.eos_token_id
        # left-pad prompt to fixed len
        pr_ids = torch.tensor(prompt_enc["input_ids"], device=self.device)
        pr_mask = torch.tensor(prompt_enc["attention_mask"], device=self.device)
        pr_ids, pr_mask = convert_right_padding_to_left(
            self.tok, pr_ids, pr_mask, self.device, max_len=self.max_starting_message_length
        )

        # right-pad response to fixed len using shared helper
        rsp_ids, rsp_mask, rsp_ass_mask = pad_to_max_length_right(
            self.tok, resp_enc, self.total_len, self.device
        )

        # concat
        input_ids = torch.cat([pr_ids, rsp_ids], dim=1)              # (B, total_len)
        attn_mask = torch.cat([pr_mask, rsp_mask], dim=1)
        pos_ids = compute_position_id_with_mask(attn_mask)
        
        # Repeat instance_id and task_name for each trajectory to match the number of outputs
        orig_instance_ids = self.orig_batch.non_tensor_batch["instance_id"].tolist()
        orig_task_names = self.orig_batch.non_tensor_batch["task_name"].tolist()
        
        # Each original instance generates self.nt trajectories
        repeated_instance_ids = []
        repeated_task_names = []
        for instance_id, task_name in zip(orig_instance_ids, orig_task_names):
            repeated_instance_ids.extend([instance_id] * self.nt)
            repeated_task_names.extend([task_name] * self.nt)

        batch = TensorDict(
            {
                "input_ids": input_ids,
                "responses": rsp_ids,
                "attention_mask": attn_mask,
                "position_ids": pos_ids,
                "loss_mask": rsp_ass_mask,        # train only on assistant tokens
            },
            batch_size=input_ids.size(0),
        )

        # non-tensor info (solution string, iterations, raw messages)
        non_tensor = {
            "solution": [o["solution"] for o in outputs],
            "iterations": [o["iterations"] for o in outputs],
            "messages": [o["messages"] for o in outputs],
            "instance_id": repeated_instance_ids,
            "task_name": repeated_task_names,
        }

        return DataProto.from_dict(tensors=batch, non_tensors=non_tensor)



# if __name__ == "__main__":
#     import argparse, json
#     from sglang import Engine
#     from transformers import AutoModelForCausalLM, AutoTokenizer

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", required=True)
#     parser.add_argument("--raw_prompt", required=False)
#     parser.add_argument("--instance_id", type=int, required=False)
#     args = parser.parse_args()

#     eng = Engine(
#                 model_path=args.model_path,
#                 port=40000,
#                 dtype="float16",
#                 max_total_tokens=60*32768,
#                 max_prefill_tokens=2*32768,
#                 enable_memory_saver=True,
#                 mem_fraction_static=0.4,
#                 tp_size=4,
#                 log_level="INFO",
#                 # enable_metrics=True,
#             )
#     tok = AutoTokenizer.from_pretrained(args.model_path)
#     sampling = {"max_new_tokens": 4096, "temperature": 0.6}
    
#     if args.raw_prompt is not None and args.instance_id is not None:
#             dummy = DataProto.from_dict(
#                 tensors={"input_ids": torch.zeros(1, 32768, dtype=torch.long)},
#                 non_tensors={"raw_prompt": [args.raw_prompt], "instance_id": [args.instance_id], "task_name": ["screen_design"]},
#             )
#     else:
#         from verl.workers.agentic.biomni.task.screen_design import screen_design
#         task = screen_design(top_k=100)
#         example = task.get_example()
#         print(example)
#         dummy = DataProto.from_dict(
#             tensors={"input_ids": torch.zeros(1, 32768, dtype=torch.long)},
#             non_tensors={"raw_prompt": [example["prompt"]], "instance_id": [example["screen_id"]], "task_name": ["screen_design"]},
#         )

#     grp = BiomniCodeActAgentGroup(
#         batch=dummy,
#         num_trajectories=1,
#         infer_engine=eng,
#         tokenizer=tok,
#         sampling_params=sampling,
#         runtime_url="http://172.24.75.232:8000",
#     )
#     dp = grp.run()
#     msg0 = dp.non_tensor_batch["messages"][0].tolist()
#     print(json.dumps(msg0, indent=2))
    
#     print("="*20)
    
#     from verl.workers.reward_manager.biomni import BiomniRewardManager
#     reward_manager = BiomniRewardManager(tokenizer=tok, num_examine=1, config={"top_k": 100})
#     result_tensor, result_metrics = reward_manager(dp)
#     response_ids = dp.batch['responses']
#     response_length = response_ids.shape[-1]
#     valid_response_length = dp.batch['attention_mask'][:, -response_length:].sum(-1)
#     print(result_metrics)
#     print(result_tensor['all'][0][valid_response_length[0] - 1])


if __name__ == "__main__":
    import argparse, json, copy
    from sglang import Engine
    from transformers import AutoTokenizer
    import torch

    from verl.workers.agentic.biomni.task.screen_design import screen_design
    from verl.workers.agentic.biomni.task.rare_disease_diagnosis import rare_disease_diagnosis
    from verl.workers.agentic.biomni.task.gwas_causal_gene import gwas_causal_gene
    from verl.workers.agentic.biomni.task.patient_gene_detection import patient_gene_detection

    # ------------------------- CLI ----------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--n_instances", type=int, default=2,
                        help="How many task examples to sample")
    parser.add_argument("--n_traj", type=int, default=2,
                        help="Trajectories per instance")
    parser.add_argument("--use_rope_scaling", action="store_true",
                        help="Use rope scaling")
    args = parser.parse_args()

    # -------------------------- engine ------------------------------
    total_len = 32768 if not args.use_rope_scaling else 131072
    max_response_length = 3072 if not args.use_rope_scaling else 8192
    max_prompt_length = total_len - max_response_length
    
    engine_kwargs = dict(
        model_path=args.model_path,
        port=40000,
        dtype="float16",
        max_total_tokens=60 * total_len,
        max_prefill_tokens=2 * total_len,
        enable_memory_saver=True,
        mem_fraction_static=0.4,
        tp_size=4,
        log_level="INFO",
    )
    if args.use_rope_scaling:
        new_ctx = 131072                      # 4 × original_limit
        override = {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768
            },
            "max_position_embeddings": new_ctx,
            "model_max_length": new_ctx
        }
        engine_kwargs["json_model_override_args"] = json.dumps(override)
        # os.environ["SGLANG_MAX_CONTEXT_LEN"] = str(new_ctx)  # belt-and-braces
    eng = Engine(**engine_kwargs)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    base_sampling = {"max_new_tokens": 8192, "temperature": 0.6}

    # --------------------- build real batch -------------------------
    task_map = {
        "rare_disease_diagnosis": rare_disease_diagnosis('/dfs/project/bioagentos/biomni_data/benchmark/'),
        "screen_design": screen_design(top_k = 20),
        "patient_gene_detection": patient_gene_detection('/dfs/project/bioagentos/biomni_data/benchmark/', num_samples = 10000),
        "gwas_causal_gene_pharmaprojects": gwas_causal_gene(path = '/dfs/project/bioagentos/biomni_data/benchmark/', dataset = "pharmaprojects", num_samples = 100000),
    }
    task = task_map["screen_design"]
    examples = [task.get_example() for _ in range(args.n_instances)]

    prompts      = [ex["prompt"]    for ex in examples]
    instance_ids = [ex["screen_id"] for ex in examples]
    task_names   = ["screen_design"] * args.n_instances
    
    task = task_map["gwas_causal_gene_pharmaprojects"]
    new_examples = [task.get_example() for _ in range(args.n_instances)]
    examples.extend(new_examples)
    prompts.extend([ex["prompt"] for ex in new_examples])
    instance_ids.extend([ex["instance_id"] for ex in new_examples])
    task_names.extend(["gwas_causal_gene_pharmaprojects"] * args.n_instances)
    
    task = task_map["patient_gene_detection"]
    new_examples = [task.get_example() for _ in range(args.n_instances)]
    examples.extend(new_examples)
    prompts.extend([ex["prompt"] for ex in new_examples])
    instance_ids.extend([ex["instance_id"] for ex in new_examples])
    task_names.extend(["patient_gene_detection"] * args.n_instances)
    
    task = task_map["rare_disease_diagnosis"]
    new_examples = [task.get_example() for _ in range(args.n_instances)]
    examples.extend(new_examples)
    prompts.extend([ex["prompt"] for ex in new_examples])
    instance_ids.extend([ex["instance_id"] for ex in new_examples])
    task_names.extend(["rare_disease_diagnosis"] * args.n_instances)

    batch_dp = DataProto.from_dict(
        tensors={"input_ids": torch.zeros(len(prompts), 1, dtype=torch.long)},
        non_tensors={
            "raw_prompt":  prompts,
            "instance_id": instance_ids,
            "task_name":   task_names,
        },
    )

    # -------------------- run roll-outs -----------------------------
    grp = BiomniCodeActAgentGroup(
        batch=batch_dp,
        num_trajectories=args.n_traj,
        infer_engine=eng,
        tokenizer=tok,
        sampling_params=copy.deepcopy(base_sampling),
        runtime_url="http://172.24.75.90:8000",
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
    )
    dp = grp.run()

    # -------------- pairing + prompt-presence checks ----------------
    ids_out   = list(dp.non_tensor_batch["instance_id"])
    msgs_out  = dp.non_tensor_batch["messages"]

    # expected flat order: [id0,id0,...,id1,id1,...]
    expected_ids = []
    expected_prompts = []
    for i, _id in enumerate(instance_ids):
        expected_ids.extend([_id] * args.n_traj)
        expected_prompts.extend([prompts[i]] * args.n_traj)

    assert ids_out == expected_ids, (
        f"✗ instance_id ordering mismatch:\nexpected {expected_ids}\n got      {ids_out}"
    )
    assert len(ids_out) == len(msgs_out), "✗ row count mismatch ids vs. messages"

    # prompt must appear in its own row’s conversation **before** first assistant
    for row_idx, (msg_list, prompt_txt) in enumerate(zip(msgs_out, expected_prompts)):
        # flatten to quickly search
        joined = " ".join(m["content"] for m in msg_list)
        assert prompt_txt in joined, f"✗ prompt missing in row {row_idx}"

        # stronger check: make sure it’s in the *prompt* slice
        first_assist = next((i for i, m in enumerate(msg_list) if m["role"] == "assistant"), len(msg_list))
        prompt_slice = " ".join(m["content"] for m in msg_list[:first_assist])
        assert prompt_txt in prompt_slice, f"✗ prompt appears after assistant in row {row_idx}"

    print("✓ Pairing + prompt-placement tests passed.")

    # -------------- length statistics --------------
    # Calculate total length for each sequence (sum of attention mask)
    total_lengths = dp.batch["attention_mask"].sum(dim=1).cpu().numpy()
    avg_length = total_lengths.mean()
    max_length = total_lengths.max()
    
    print(f"\n--- Length Statistics ---")
    print(f"Average total length: {avg_length:.2f}")
    print(f"Max total length: {max_length}")
    print(f"Number of sequences: {len(total_lengths)}")
    # Find the index of the longest trajectory
    longest_idx = total_lengths.argmax()
    print("\n--- Longest Trajectory (Full Dump) ---")
    print(json.dumps(msgs_out[longest_idx], indent=2))

    # # -------------- pretty-print a sample conversation --------------
    # sample_json = json.dumps(msgs_out[0], indent=2)
    # print("\n--- first conversation ---")
    # print(sample_json[:2000] + ("…" if len(sample_json) > 2000 else ""))

    # ---------------- optional reward eval --------------------------
    try:
        from verl.workers.reward_manager.biomni import BiomniRewardManager

        reward_manager = BiomniRewardManager(tokenizer=tok, num_examine=1,
                                             config={"top_k": 100})
        r_tensor, r_metrics = reward_manager(dp)
        resp_ids  = dp.batch["responses"]
        resp_len  = resp_ids.shape[-1]
        valid_len = dp.batch["attention_mask"][:, -resp_len:].sum(-1)
        print("\nReward metrics:", r_metrics)
        first_tokens = [r_tensor["all"][i][valid_len[i] - 1] for i in range(len(valid_len))]
        print("Reward first tokens:", first_tokens)
    except ImportError:
        pass


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
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8001")

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