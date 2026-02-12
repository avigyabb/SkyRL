"""Unit tests for _validate_format and _parse_after_think.

Tests the BiomniRewardAdapter._validate_format static method and the
_parse_after_think function used for format validation of assistant messages.
"""

import importlib.util
import os
import re
import sys
import types

# ---------------------------------------------------------------------------
# Mock the biomni task imports that biomni_reward_adapter needs at load time
# _validate_format itself doesn't use them, but the module imports them
# ---------------------------------------------------------------------------
def _make_mock_task_module(name):
    m = types.ModuleType(name)
    # Each task module exports a callable with the same name (e.g., screen_design)
    attr_name = name.split(".")[-1]
    setattr(m, attr_name, lambda *a, **kw: None)
    return m


def _setup_mocks():
    base = "skyrl_agent.agents.biomni_codeact.task"
    task_names = [
        "screen_design", "gwas_causal_gene", "crispr_delivery",
        "rare_disease_diagnosis", "gwas_variant_prioritization",
        "patient_gene_detection", "lab_bench", "screen_gene_retrieval",
    ]
    # Ensure parent modules exist
    for part in ["skyrl_agent", "skyrl_agent.agents", "skyrl_agent.agents.biomni_codeact",
                 "skyrl_agent.agents.biomni_codeact.task"]:
        if part not in sys.modules:
            sys.modules[part] = types.ModuleType(part)
    for name in task_names:
        full = f"{base}.{name}"
        if full not in sys.modules:
            sys.modules[full] = _make_mock_task_module(full)


_setup_mocks()

# Load biomni_reward_adapter and extract _validate_format
_path = os.path.join(os.path.dirname(__file__), "..", "skyrl_agent", "tasks", "biomni_reward_adapter.py")
_spec = importlib.util.spec_from_file_location("biomni_reward_adapter", os.path.abspath(_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_validate_format = _mod.BiomniRewardAdapter._validate_format

# ---------------------------------------------------------------------------
# Inline _parse_after_think for testing (avoids loading agent module)
# ---------------------------------------------------------------------------
_TAG_RGX = {
    "solution": re.compile(r"<solution>((?:(?!<solution>|</solution>).)*)</solution>", re.DOTALL | re.IGNORECASE),
    "execute": re.compile(r"<execute>((?:(?!<execute>|</execute>).)*)</execute>", re.DOTALL | re.IGNORECASE),
}


def _parse_after_think(match_type: str, text: str):
    """Parse the first occurrence of match_type AFTER </think>."""
    low = text.lower()
    think_end = low.find("</think>")
    if think_end == -1:
        after = text
    else:
        after = text[think_end + len("</think>"):]
    m = _TAG_RGX[match_type].search(after)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# TestValidFormatBasic
# ---------------------------------------------------------------------------
class TestValidFormatBasic:
    def test_valid_single_execute(self):
        """Full valid conversation: non-last execute, last solution."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>reasoning</think><execute>code</execute>"},
            {"role": "user", "content": "obs"},
            {"role": "assistant", "content": "<think>done</think><solution>answer</solution>"},
        ]
        assert _validate_format(messages) == 1.0

    def test_valid_single_solution(self):
        """Last-only assistant with <think>...</think><solution>...</solution> → 1.0."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>reasoning</think><solution>final answer</solution>"},
        ]
        assert _validate_format(messages) == 1.0

    def test_no_think_tag(self):
        """Assistant without <think> → 0.0."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<execute>code</execute>"},
        ]
        assert _validate_format(messages) == 0.0

    def test_no_closing_tag(self):
        """<think>...</think><execute>code (no closing) → 0.0."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>reasoning</think><execute>code"},
        ]
        assert _validate_format(messages) == 0.0

    def test_empty_messages(self):
        """No assistant messages → 0.0."""
        messages = [{"role": "user", "content": "hi"}]
        assert _validate_format(messages) == 0.0


# ---------------------------------------------------------------------------
# TestValidFormatNested
# ---------------------------------------------------------------------------
class TestValidFormatNested:
    def test_nested_tags_ok(self):
        """Nested tags inside outer block are OK."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>thought</think><execute>if x: print(\"<solution>hi</solution>\")\n</execute>"},
            {"role": "user", "content": "obs"},
            {"role": "assistant", "content": "<think>done</think><solution>answer</solution>"},
        ]
        assert _validate_format(messages) == 1.0

    def test_multiple_outer_blocks_fail(self):
        """<think>t</think><execute>a</execute><execute>b</execute> → 0.0."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>t</think><execute>a</execute><execute>b</execute>"},
            {"role": "user", "content": "obs"},
            {"role": "assistant", "content": "<think>done</think><solution>answer</solution>"},
        ]
        assert _validate_format(messages) == 0.0

    def test_text_between_think_and_action(self):
        """<think>t</think> some text <execute>code</execute> → 1.0."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>t</think> some text <execute>code</execute>"},
            {"role": "user", "content": "obs"},
            {"role": "assistant", "content": "<think>done</think><solution>answer</solution>"},
        ]
        assert _validate_format(messages) == 1.0


# ---------------------------------------------------------------------------
# TestValidFormatLastMessage
# ---------------------------------------------------------------------------
class TestValidFormatLastMessage:
    def test_last_must_be_solution(self):
        """Last assistant with <execute> outer → 0.0."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>reasoning</think><execute>code</execute>"},
        ]
        assert _validate_format(messages) == 0.0

    def test_non_last_must_be_execute(self):
        """Non-last assistant with <solution> outer → 0.0."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>reasoning</think><solution>answer</solution>"},
            {"role": "user", "content": "obs"},
            {"role": "assistant", "content": "<think>done</think><solution>final</solution>"},
        ]
        assert _validate_format(messages) == 0.0


# ---------------------------------------------------------------------------
# TestParseAfterThink
# ---------------------------------------------------------------------------
class TestParseAfterThink:
    def test_tags_in_think_ignored(self):
        """<think>I'll use <execute>x</execute></think><execute>real_code</execute> → returns 'real_code'."""
        text = "<think>I'll use <execute>x</execute></think><execute>real_code</execute>"
        assert _parse_after_think("execute", text) == "real_code"

    def test_solution_after_think(self):
        """<think>thinking</think><solution>final answer</solution> → returns 'final answer'."""
        text = "<think>thinking</think><solution>final answer</solution>"
        assert _parse_after_think("solution", text) == "final answer"

    def test_no_think_close(self):
        """some text <execute>code</execute> → returns 'code' (fallback)."""
        text = "some text <execute>code</execute>"
        assert _parse_after_think("execute", text) == "code"

    def test_no_match(self):
        """<think>thinking</think>no tags here → returns None."""
        text = "<think>thinking</think>no tags here"
        assert _parse_after_think("execute", text) is None
        assert _parse_after_think("solution", text) is None


if __name__ == "__main__":
    test_classes = [
        TestValidFormatBasic,
        TestValidFormatNested,
        TestValidFormatLastMessage,
        TestParseAfterThink,
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
