"""Unit tests for transitions_to_training_data and related utilities.

These tests verify the token-in-token-out pattern for TIS alignment:
- Single transitions produce correct training data
- Multi-turn incremental observations merge correctly
- Logprobs align exactly with action tokens
- Non-incremental observations split into separate data
- Edge cases are handled gracefully
"""

import importlib.util
import os
import sys

# Load utils.py directly to bypass skyrl_agent.__init__ (which requires omegaconf etc.)
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
# Helper
# ---------------------------------------------------------------------------
def _make_transition(ob_ids, ac_ids, logprobs=None, reward=0.0, done=False):
    """Convenience factory for Transition objects."""
    if logprobs is None:
        logprobs = [-0.1] * len(ac_ids)
    return Transition(
        ob=Observation(input_ids=ob_ids),
        ac=TokensWithLogprobs(token_ids=ac_ids, logprobs=logprobs),
        reward=reward,
        episode_done=done,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestSingleTransition:
    def test_basic(self):
        """Single transition → one TrainingDatum with correct split."""
        t = _make_transition([1, 2, 3, 4, 5], [10, 11, 12], logprobs=[-0.1, -0.2, -0.3])
        data = transitions_to_training_data([t])

        assert len(data) == 1
        assert data[0].input_tokens == [1, 2, 3, 4, 5]
        assert data[0].response_tokens == [10, 11, 12]
        assert data[0].response_logprobs == [-0.1, -0.2, -0.3]
        assert data[0].response_mask == [1.0, 1.0, 1.0]

    def test_no_logprobs_defaults_to_zero(self):
        """When logprobs is None, defaults to 0.0 per token."""
        t = Transition(
            ob=Observation(input_ids=[1, 2]),
            ac=TokensWithLogprobs(token_ids=[10, 11], logprobs=None),
            reward=0.0,
            episode_done=False,
        )
        data = transitions_to_training_data([t])
        assert data[0].response_logprobs == [0.0, 0.0]


class TestMultiTurnIncremental:
    def test_two_turns_merge(self):
        """Two incremental turns merge into one datum."""
        # Turn 1: prompt [1,2,3], response [10,11]
        t1 = _make_transition([1, 2, 3], [10, 11], logprobs=[-0.1, -0.2])
        # Turn 2: ob = prev_full + user_turn with template tokens
        # prev_full = [1,2,3,10,11], new user turn = [20,21]
        t2 = _make_transition([1, 2, 3, 10, 11, 20, 21], [30, 31], logprobs=[-0.3, -0.4])

        data = transitions_to_training_data([t1, t2])

        assert len(data) == 1
        d = data[0]
        # Input tokens = initial prompt
        assert d.input_tokens == [1, 2, 3]
        # Response tokens = ac1 + delta_ob + ac2
        assert d.response_tokens == [10, 11, 20, 21, 30, 31]
        # Mask: ac1=[1,1], delta_ob=[0,0], ac2=[1,1]
        assert d.response_mask == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        # Logprobs: ac1 logprobs, delta_ob zeros, ac2 logprobs
        assert d.response_logprobs == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]

    def test_three_turns_merge(self):
        """Three incremental turns merge into one datum."""
        t1 = _make_transition([1, 2], [10], logprobs=[-0.1])
        t2 = _make_transition([1, 2, 10, 20], [30], logprobs=[-0.3])
        t3 = _make_transition([1, 2, 10, 20, 30, 40, 41], [50, 51], logprobs=[-0.5, -0.6])

        data = transitions_to_training_data([t1, t2, t3])

        assert len(data) == 1
        d = data[0]
        assert d.input_tokens == [1, 2]
        # response: ac1(10) + delta_ob2(20) + ac2(30) + delta_ob3(40,41) + ac3(50,51)
        assert d.response_tokens == [10, 20, 30, 40, 41, 50, 51]
        assert d.response_mask == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]


class TestNonIncrementalSplit:
    def test_non_incremental_creates_separate_data(self):
        """Non-incremental observations create separate training data."""
        t1 = _make_transition([1, 2, 3], [10, 11], logprobs=[-0.1, -0.2])
        # This observation does NOT start with [1,2,3,10,11,...] → separate datum
        t2 = _make_transition([100, 200], [30, 31], logprobs=[-0.3, -0.4])

        data = transitions_to_training_data([t1, t2])

        assert len(data) == 2
        assert data[0].input_tokens == [1, 2, 3]
        assert data[0].response_tokens == [10, 11]
        assert data[1].input_tokens == [100, 200]
        assert data[1].response_tokens == [30, 31]


class TestLogprobAlignment:
    def test_logprobs_count_equals_action_tokens(self):
        """Logprob count == number of mask==1 positions in response."""
        t = _make_transition(
            [1, 2, 3],
            [10, 11, 12, 13],
            logprobs=[-0.1, -0.2, -0.3, -0.4],
        )
        data = transitions_to_training_data([t])

        d = data[0]
        action_positions = sum(1 for m in d.response_mask if m == 1.0)
        action_logprobs = [lp for lp, m in zip(d.response_logprobs, d.response_mask) if m == 1.0]

        assert action_positions == 4
        assert len(action_logprobs) == 4
        assert action_logprobs == [-0.1, -0.2, -0.3, -0.4]

    def test_multi_turn_logprob_alignment(self):
        """In multi-turn, logprobs still align exactly with action tokens."""
        t1 = _make_transition([1], [10, 11], logprobs=[-1.0, -2.0])
        t2 = _make_transition([1, 10, 11, 20], [30], logprobs=[-3.0])

        data = transitions_to_training_data([t1, t2])

        d = data[0]
        action_logprobs = [lp for lp, m in zip(d.response_logprobs, d.response_mask) if m == 1.0]
        assert action_logprobs == [-1.0, -2.0, -3.0]


class TestEdgeCases:
    def test_empty_transitions(self):
        """Empty list returns empty data."""
        assert transitions_to_training_data([]) == []

    def test_empty_observation(self):
        """Empty observation still works."""
        t = _make_transition([], [10, 11], logprobs=[-0.1, -0.2])
        data = transitions_to_training_data([t])
        assert len(data) == 1
        assert data[0].input_tokens == []
        assert data[0].response_tokens == [10, 11]

    def test_is_prefix_helper(self):
        """Verify _is_prefix utility."""
        assert _is_prefix([1, 2], [1, 2, 3]) is True
        assert _is_prefix([1, 2, 3], [1, 2, 3]) is True
        assert _is_prefix([1, 2, 3, 4], [1, 2, 3]) is False
        assert _is_prefix([1, 3], [1, 2, 3]) is False
        assert _is_prefix([], [1, 2]) is True


class TestEOSHandling:
    """Test EOS handling pattern used by the biomni agent."""

    def test_eos_in_observation_delta_has_mask_zero(self):
        """EOS in observation delta gets mask=0."""
        # Turn 1: ob=[1,2,3], ac=[10,11] (EOS already stripped by agent)
        t1 = _make_transition([1, 2, 3], [10, 11], logprobs=[-0.1, -0.2])
        # Turn 2: ob includes EOS (99) appended by agent to running input_ids, then delta [20,21]
        t2 = _make_transition([1, 2, 3, 10, 11, 99, 20, 21], [30], logprobs=[-0.3])

        data = transitions_to_training_data([t1, t2])

        assert len(data) == 1
        d = data[0]
        # Observation delta [99, 20, 21] all get mask=0; EOS (99) must have mask=0
        assert d.response_tokens == [10, 11, 99, 20, 21, 30]
        assert d.response_mask == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # Verify EOS token 99 has mask=0
        token_99_idx = d.response_tokens.index(99)
        assert d.response_mask[token_99_idx] == 0.0

    def test_eos_stripped_logprobs_align(self):
        """After EOS stripping, logprobs still align with action tokens."""
        # ac_tokens = [10,11] with EOS already stripped; logprobs align
        t = _make_transition([1, 2, 3], [10, 11], logprobs=[-0.1, -0.2])
        data = transitions_to_training_data([t])

        assert len(data) == 1
        d = data[0]
        assert d.response_mask == [1.0, 1.0]
        assert d.response_logprobs == [-0.1, -0.2]

    def test_multi_turn_eos_between_turns(self):
        """Three turns with EOS tokens between them; all EOS get mask=0."""
        # Turn 1: ob=[1,2], ac=[10]
        t1 = _make_transition([1, 2], [10], logprobs=[-0.1])
        # Turn 2: ob has EOS (99) between turns
        t2 = _make_transition([1, 2, 10, 99, 20], [30], logprobs=[-0.3])
        # Turn 3: ob has EOS (99) between turns
        t3 = _make_transition([1, 2, 10, 99, 20, 30, 99, 40], [50], logprobs=[-0.5])

        data = transitions_to_training_data([t1, t2, t3])

        assert len(data) == 1
        d = data[0]
        assert d.input_tokens == [1, 2]
        assert d.response_tokens == [10, 99, 20, 30, 99, 40, 50]
        assert d.response_mask == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        # All EOS (99) have mask=0, all action tokens have mask=1
        for i, tok in enumerate(d.response_tokens):
            if tok == 99:
                assert d.response_mask[i] == 0.0, f"EOS at index {i} should have mask=0"
            if tok in (10, 30, 50):
                assert d.response_mask[i] == 1.0, f"Action {tok} at index {i} should have mask=1"


if __name__ == "__main__":
    # Run tests without pytest
    test_classes = [
        TestSingleTransition,
        TestMultiTurnIncremental,
        TestNonIncrementalSplit,
        TestLogprobAlignment,
        TestEdgeCases,
        TestEOSHandling,
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
