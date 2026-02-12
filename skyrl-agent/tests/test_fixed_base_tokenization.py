"""Unit tests for fixed-base tokenization pattern.

Tests verify that the fixed-base approach produces consistent tokenization:
- base_conversation_token_ids remain the same across calls
- apply_chat_template([*base, *obs]) produces consistent prefix
- observation deltas are correctly computed
- Multi-turn conversations produce correct delta sequences
"""

import importlib.util
import os
import sys

# Load utils.py directly to bypass skyrl_agent.__init__
_utils_path = os.path.join(os.path.dirname(__file__), "..", "skyrl_agent", "functional", "utils.py")
_spec = importlib.util.spec_from_file_location("skyrl_agent.functional.utils", os.path.abspath(_utils_path))
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)

Transition = _utils.Transition
Observation = _utils.Observation
TokensWithLogprobs = _utils.TokensWithLogprobs
transitions_to_training_data = _utils.transitions_to_training_data
_is_prefix = _utils._is_prefix


# ---------------------------------------------------------------------------
# Mock tokenizer for testing fixed-base tokenization logic
# ---------------------------------------------------------------------------
class MockTokenizer:
    """A simple mock tokenizer that assigns token IDs based on string content.
    
    This simulates the fixed-base tokenization pattern without requiring
    a real tokenizer model. Each unique character/word gets a deterministic ID.
    """
    
    def __init__(self):
        self.eos_token_id = 99
        self.pad_token_id = 0
        self._next_id = 100
        self._vocab = {}
    
    def _get_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = self._next_id
            self._next_id += 1
        return self._vocab[token]
    
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kwargs):
        """Mock apply_chat_template that produces deterministic token sequences.
        
        Each message produces a fixed sequence of tokens:
        - System: [1, 2, <content_tokens>, 3]
        - User: [4, 5, <content_tokens>, 6]
        - Assistant: [7, 8, <content_tokens>, 9]
        - Generation prompt: [7, 8, 10]  (start of assistant + <think>\n)
        """
        tokens = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            content_tokens = [self._get_id(c) for c in content]
            
            if role == "system":
                tokens.extend([1, 2] + content_tokens + [3])
            elif role == "user":
                tokens.extend([4, 5] + content_tokens + [6])
            elif role == "assistant":
                tokens.extend([7, 8] + content_tokens + [9])
        
        if add_generation_prompt:
            tokens.extend([7, 8, 10])  # <|im_start|>assistant\n<think>\n
        
        return tokens


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFixedBaseSetup:
    """Test that base_conversation and base_conversation_token_ids are set up correctly."""
    
    def test_base_tokens_consistent(self):
        """base_conversation_token_ids should be the same across multiple calls."""
        tok = MockTokenizer()
        base_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Solve this problem."},
        ]
        
        ids1 = tok.apply_chat_template(base_messages, add_generation_prompt=False)
        ids2 = tok.apply_chat_template(base_messages, add_generation_prompt=False)
        assert ids1 == ids2, "base_conversation_token_ids should be deterministic"
    
    def test_gen_prompt_extends_base(self):
        """With add_generation_prompt=True, output should extend base tokens."""
        tok = MockTokenizer()
        base_messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Question"},
        ]
        
        base_ids = tok.apply_chat_template(base_messages, add_generation_prompt=False)
        full_ids = tok.apply_chat_template(base_messages, add_generation_prompt=True)
        
        assert _is_prefix(base_ids, full_ids), "base_ids should be prefix of full_ids"
        # The extra tokens are the generation prompt
        gen_prompt = full_ids[len(base_ids):]
        assert len(gen_prompt) > 0, "generation prompt should add tokens"


class TestObservationDelta:
    """Test observation delta computation using fixed-base approach."""
    
    def test_delta_after_assistant_and_observation(self):
        """Delta for obs_messages should include assistant + user + gen_prompt tokens."""
        tok = MockTokenizer()
        base_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
        ]
        base_ids = tok.apply_chat_template(base_messages, add_generation_prompt=False)
        
        # After turn 1: assistant replied, user sent observation
        obs_messages = [
            {"role": "assistant", "content": "thinking and code"},
            {"role": "user", "content": "observation output"},
        ]
        
        full_ids = tok.apply_chat_template(
            [*base_messages, *obs_messages],
            add_generation_prompt=True,
        )
        
        # Delta should be everything after base_ids
        delta = full_ids[len(base_ids):]
        
        assert len(delta) > 0, "delta should contain assistant + user + gen_prompt tokens"
        assert _is_prefix(base_ids, full_ids), "base_ids should be prefix of full_ids"
    
    def test_delta_grows_monotonically(self):
        """As conversation grows, full_ids should always extend."""
        tok = MockTokenizer()
        base_messages = [{"role": "user", "content": "start"}]
        base_ids = tok.apply_chat_template(base_messages, add_generation_prompt=False)
        
        obs_messages = []
        prev_len = len(base_ids)
        
        for i in range(3):
            obs_messages.append({"role": "assistant", "content": f"response{i}"})
            obs_messages.append({"role": "user", "content": f"obs{i}"})
            
            full_ids = tok.apply_chat_template(
                [*base_messages, *obs_messages],
                add_generation_prompt=True,
            )
            
            assert len(full_ids) > prev_len, f"Turn {i}: full_ids should grow"
            assert _is_prefix(base_ids, full_ids), f"Turn {i}: base should be prefix"
            prev_len = len(full_ids)


class TestFixedBasePrefixMatching:
    """Test that fixed-base tokenization produces correct prefix matching for transitions."""
    
    def test_two_turn_prefix_match(self):
        """Verify that Turn 1's (ob + ac) is a prefix of Turn 2's ob."""
        tok = MockTokenizer()
        base_messages = [{"role": "user", "content": "q"}]
        
        # Turn 1: initial prompt with gen prompt
        ob1 = tok.apply_chat_template(base_messages, add_generation_prompt=True)
        ac1 = [50, 51, 52]  # model output tokens
        
        # Simulate: messages now include the assistant reply and user observation
        obs_messages = [
            {"role": "assistant", "content": "reply1"},
            {"role": "user", "content": "obs1"},
        ]
        ob2 = tok.apply_chat_template(
            [*base_messages, *obs_messages],
            add_generation_prompt=True,
        )
        ac2 = [60, 61]
        
        # Create transitions
        t1 = Transition(
            ob=Observation(input_ids=ob1),
            ac=TokensWithLogprobs(token_ids=ac1, logprobs=[-0.1, -0.2, -0.3]),
            reward=0.0, episode_done=False,
        )
        t2 = Transition(
            ob=Observation(input_ids=ob2),
            ac=TokensWithLogprobs(token_ids=ac2, logprobs=[-0.4, -0.5]),
            reward=0.0, episode_done=False,
        )
        
        # If ob1 + ac1 is a prefix of ob2, they should merge into one datum
        full_seq_after_t1 = ob1 + ac1
        is_prefix = _is_prefix(full_seq_after_t1, ob2)
        
        data = transitions_to_training_data([t1, t2])
        
        if is_prefix:
            # Should merge: 1 datum
            assert len(data) == 1, f"Expected merge but got {len(data)} data (prefix={is_prefix})"
            d = data[0]
            # Action tokens should have mask=1
            action_logprobs = [lp for lp, m in zip(d.response_logprobs, d.response_mask) if m == 1.0]
            assert action_logprobs == [-0.1, -0.2, -0.3, -0.4, -0.5]
        else:
            # Won't merge due to re-tokenization mismatch: 2 data
            assert len(data) == 2, f"Expected split but got {len(data)} data (prefix={is_prefix})"
    
    def test_eos_between_turns_has_mask_zero(self):
        """EOS token between turns should get mask=0 (observation, not action)."""
        EOS = 99
        
        # Turn 1: ob=[1,2,3], ac=[10,11] (EOS stripped)
        # Turn 2: ob=[1,2,3,10,11, EOS, 20,21, 7,8,10], ac=[30]
        # The EOS was appended to input_ids by the agent after Turn 1 stopped
        ob1 = [1, 2, 3]
        ac1 = [10, 11]
        ob2 = [1, 2, 3, 10, 11, EOS, 20, 21, 7, 8, 10]
        ac2 = [30]
        
        t1 = Transition(
            ob=Observation(input_ids=ob1),
            ac=TokensWithLogprobs(token_ids=ac1, logprobs=[-0.1, -0.2]),
            reward=0.0, episode_done=False,
        )
        t2 = Transition(
            ob=Observation(input_ids=ob2),
            ac=TokensWithLogprobs(token_ids=ac2, logprobs=[-0.3]),
            reward=0.0, episode_done=False,
        )
        
        data = transitions_to_training_data([t1, t2])
        assert len(data) == 1
        d = data[0]
        
        # Full response: ac1 + delta_ob + ac2 = [10,11, EOS,20,21,7,8,10, 30]
        # Mask:          [1, 1,  0,  0, 0, 0,0,0,  1]
        delta_ob = ob2[len(ob1) + len(ac1):]  # = [EOS, 20, 21, 7, 8, 10]
        expected_response = ac1 + delta_ob + ac2
        assert d.response_tokens == expected_response
        
        # Verify EOS has mask=0
        eos_pos_in_response = len(ac1)  # first position after ac1
        assert d.response_mask[eos_pos_in_response] == 0.0, "EOS should have mask=0"
        
        # Verify all delta_ob have mask=0
        for i in range(len(ac1), len(ac1) + len(delta_ob)):
            assert d.response_mask[i] == 0.0, f"delta_ob position {i} should have mask=0"
        
        # Verify action tokens have mask=1
        for i in range(len(ac1)):
            assert d.response_mask[i] == 1.0, f"ac1 position {i} should have mask=1"
        assert d.response_mask[-1] == 1.0, "ac2 should have mask=1"


if __name__ == "__main__":
    test_classes = [
        TestFixedBaseSetup,
        TestObservationDelta,
        TestFixedBasePrefixMatching,
    ]
    failed = 0
    passed = 0
    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    print(f"  PASS: {cls.__name__}.{method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL: {cls.__name__}.{method_name}: {e}")
                    failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
