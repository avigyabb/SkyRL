"""Integration tests for the full TITO (Token-in-Token-Out) flow.

Tests the end-to-end flow of:
1. Agent generates tokens (simulated vLLM output)
2. EOS is stripped from action tokens
3. Transitions are recorded with correct observations and actions
4. transitions_to_training_data produces correct masks and logprobs
5. The training pipeline would receive correctly aligned data

These tests simulate the complete flow without requiring actual vLLM or tokenizer.
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
TrainingDatum = _utils.TrainingDatum
transitions_to_training_data = _utils.transitions_to_training_data
_is_prefix = _utils._is_prefix


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EOS_TOKEN_ID = 99


def strip_eos(token_ids, logprobs, eos_id=EOS_TOKEN_ID):
    """Simulate EOS stripping as done in _llm_generate."""
    ac_tokens = list(token_ids)
    ac_logprobs = list(logprobs) if logprobs else None
    if ac_tokens and ac_tokens[-1] == eos_id:
        ac_tokens = ac_tokens[:-1]
        if ac_logprobs and len(ac_logprobs) > 0:
            ac_logprobs = ac_logprobs[:-1]
    return ac_tokens, ac_logprobs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestEndToEndSingleTurn:
    """Test single-turn TITO flow."""

    def test_single_turn_with_eos(self):
        """Single turn where vLLM returns EOS: EOS is stripped, mask is correct."""
        prompt_ids = [1, 2, 3, 4, 5]
        vllm_output_tokens = [10, 11, 12, EOS_TOKEN_ID]
        vllm_logprobs = [-0.1, -0.2, -0.3, -0.9]

        ac_tokens, ac_logprobs = strip_eos(vllm_output_tokens, vllm_logprobs)
        assert ac_tokens == [10, 11, 12]
        assert ac_logprobs == [-0.1, -0.2, -0.3]

        t = Transition(
            ob=Observation(input_ids=prompt_ids),
            ac=TokensWithLogprobs(token_ids=ac_tokens, logprobs=ac_logprobs),
            reward=1.0,
            episode_done=True,
        )

        data = transitions_to_training_data([t])
        assert len(data) == 1
        d = data[0]

        assert d.input_tokens == prompt_ids
        assert d.response_tokens == [10, 11, 12]
        assert d.response_logprobs == [-0.1, -0.2, -0.3]
        assert d.response_mask == [1.0, 1.0, 1.0]

    def test_single_turn_without_eos(self):
        """Single turn where vLLM does NOT return EOS (max_tokens reached)."""
        prompt_ids = [1, 2, 3]
        vllm_output_tokens = [10, 11]
        vllm_logprobs = [-0.5, -0.6]

        ac_tokens, ac_logprobs = strip_eos(vllm_output_tokens, vllm_logprobs)
        assert ac_tokens == [10, 11]

        t = Transition(
            ob=Observation(input_ids=prompt_ids),
            ac=TokensWithLogprobs(token_ids=ac_tokens, logprobs=ac_logprobs),
            reward=0.0,
            episode_done=False,
        )

        data = transitions_to_training_data([t])
        assert len(data) == 1
        assert data[0].response_mask == [1.0, 1.0]


class TestEndToEndMultiTurn:
    """Test multi-turn TITO flow simulating the full agent loop."""

    def test_two_turn_conversation(self):
        """Two turns: execute + observation + solution."""
        prompt_ids = [1, 2, 3, 4, 5]
        vllm_out_1 = [10, 11, 12, EOS_TOKEN_ID]
        vllm_lp_1 = [-0.1, -0.2, -0.3, -0.9]

        ac1, lp1 = strip_eos(vllm_out_1, vllm_lp_1)

        # Running input_ids after Turn 1 with EOS appended
        running_ids = list(prompt_ids) + list(ac1) + [EOS_TOKEN_ID]

        # Observation delta
        obs_delta = [20, 21, 22, 23, 24]
        running_ids.extend(obs_delta)

        ob2 = list(running_ids)
        vllm_out_2 = [30, 31, EOS_TOKEN_ID]
        vllm_lp_2 = [-0.4, -0.5, -0.8]
        ac2, lp2 = strip_eos(vllm_out_2, vllm_lp_2)

        t1 = Transition(
            ob=Observation(input_ids=prompt_ids),
            ac=TokensWithLogprobs(token_ids=ac1, logprobs=lp1),
            reward=0.0, episode_done=False,
        )
        t2 = Transition(
            ob=Observation(input_ids=ob2),
            ac=TokensWithLogprobs(token_ids=ac2, logprobs=lp2),
            reward=1.0, episode_done=True,
        )

        # Verify prefix matching
        full_after_t1 = prompt_ids + ac1
        assert _is_prefix(full_after_t1, ob2)

        data = transitions_to_training_data([t1, t2])
        assert len(data) == 1

        d = data[0]
        assert d.input_tokens == prompt_ids

        delta_ob = ob2[len(prompt_ids) + len(ac1):]
        expected_response = ac1 + delta_ob + ac2
        assert d.response_tokens == expected_response

        expected_mask = [1.0] * len(ac1) + [0.0] * len(delta_ob) + [1.0] * len(ac2)
        assert d.response_mask == expected_mask

        expected_logprobs = lp1 + [0.0] * len(delta_ob) + lp2
        assert d.response_logprobs == expected_logprobs

        # Verify EOS in delta_ob has mask=0
        eos_pos = len(ac1)
        assert d.response_tokens[eos_pos] == EOS_TOKEN_ID
        assert d.response_mask[eos_pos] == 0.0

    def test_three_turn_all_logprobs_correct(self):
        """Three turns: verify all logprobs are correctly aligned."""
        ob1 = [1, 2, 3]
        ac1 = [10, 11]
        lp1 = [-1.0, -2.0]

        ob2 = [1, 2, 3, 10, 11, EOS_TOKEN_ID, 20, 21]
        ac2 = [30]
        lp2 = [-3.0]

        ob3 = [1, 2, 3, 10, 11, EOS_TOKEN_ID, 20, 21, 30, EOS_TOKEN_ID, 40]
        ac3 = [50, 51]
        lp3 = [-5.0, -6.0]

        transitions = [
            Transition(
                ob=Observation(input_ids=ob1),
                ac=TokensWithLogprobs(token_ids=ac1, logprobs=lp1),
                reward=0.0, episode_done=False,
            ),
            Transition(
                ob=Observation(input_ids=ob2),
                ac=TokensWithLogprobs(token_ids=ac2, logprobs=lp2),
                reward=0.0, episode_done=False,
            ),
            Transition(
                ob=Observation(input_ids=ob3),
                ac=TokensWithLogprobs(token_ids=ac3, logprobs=lp3),
                reward=1.0, episode_done=True,
            ),
        ]

        data = transitions_to_training_data(transitions)
        assert len(data) == 1

        d = data[0]
        action_logprobs = [lp for lp, m in zip(d.response_logprobs, d.response_mask) if m == 1.0]
        assert action_logprobs == [-1.0, -2.0, -3.0, -5.0, -6.0]

        for lp, m in zip(d.response_logprobs, d.response_mask):
            if m == 1.0:
                assert lp != 0.0


class TestTISWeightCorrectness:
    """Verify that TIS importance weights would be mathematically correct."""

    def test_no_dummy_logprobs_at_trained_positions(self):
        """All mask=1 positions must have real logprobs, not 0.0 placeholders."""
        ob1 = [1, 2]
        ac1 = [10, 11, 12]
        lp1 = [-0.5, -0.3, -0.7]

        ob2 = [1, 2, 10, 11, 12, EOS_TOKEN_ID, 20, 21]
        ac2 = [30, 31]
        lp2 = [-0.4, -0.6]

        t1 = Transition(
            ob=Observation(input_ids=ob1),
            ac=TokensWithLogprobs(token_ids=ac1, logprobs=lp1),
            reward=0.0, episode_done=False,
        )
        t2 = Transition(
            ob=Observation(input_ids=ob2),
            ac=TokensWithLogprobs(token_ids=ac2, logprobs=lp2),
            reward=1.0, episode_done=True,
        )

        data = transitions_to_training_data([t1, t2])
        d = data[0]

        for i, (lp, m) in enumerate(zip(d.response_logprobs, d.response_mask)):
            if m == 1.0:
                assert lp != 0.0, (
                    f"Position {i}: mask=1 but logprob=0.0. "
                    "This would cause incorrect TIS importance weights."
                )

    def test_eos_excluded_from_tis(self):
        """EOS tokens must have mask=0."""
        ob1 = [1, 2, 3]
        ac1 = [10]
        lp1 = [-0.5]

        ob2 = [1, 2, 3, 10, EOS_TOKEN_ID, 20]
        ac2 = [30]
        lp2 = [-0.6]

        t1 = Transition(
            ob=Observation(input_ids=ob1),
            ac=TokensWithLogprobs(token_ids=ac1, logprobs=lp1),
            reward=0.0, episode_done=False,
        )
        t2 = Transition(
            ob=Observation(input_ids=ob2),
            ac=TokensWithLogprobs(token_ids=ac2, logprobs=lp2),
            reward=1.0, episode_done=True,
        )

        data = transitions_to_training_data([t1, t2])
        d = data[0]

        eos_found = False
        for i, (tok, m) in enumerate(zip(d.response_tokens, d.response_mask)):
            if tok == EOS_TOKEN_ID:
                assert m == 0.0, f"EOS at response position {i} must have mask=0"
                eos_found = True

        assert eos_found


class TestEdgeCasesTITO:
    """Edge cases in the TITO flow."""

    def test_model_generates_eos_mid_sequence(self):
        """EOS in middle of output should NOT be stripped."""
        vllm_tokens = [10, EOS_TOKEN_ID, 11, 12]
        vllm_logprobs = [-0.1, -0.9, -0.2, -0.3]

        ac_tokens, ac_logprobs = strip_eos(vllm_tokens, vllm_logprobs)
        assert ac_tokens == [10, EOS_TOKEN_ID, 11, 12]
        assert ac_logprobs == [-0.1, -0.9, -0.2, -0.3]

    def test_empty_output_after_eos_strip(self):
        """If model only generates EOS, stripping produces empty tokens."""
        vllm_tokens = [EOS_TOKEN_ID]
        vllm_logprobs = [-0.9]

        ac_tokens, ac_logprobs = strip_eos(vllm_tokens, vllm_logprobs)
        assert ac_tokens == []
        assert ac_logprobs == []

    def test_none_logprobs_handling(self):
        """If logprobs is None, strip_eos should handle gracefully."""
        vllm_tokens = [10, 11, EOS_TOKEN_ID]

        ac_tokens, ac_logprobs = strip_eos(vllm_tokens, None)
        assert ac_tokens == [10, 11]
        assert ac_logprobs is None


if __name__ == "__main__":
    test_classes = [
        TestEndToEndSingleTurn,
        TestEndToEndMultiTurn,
        TestTISWeightCorrectness,
        TestEdgeCasesTITO,
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
