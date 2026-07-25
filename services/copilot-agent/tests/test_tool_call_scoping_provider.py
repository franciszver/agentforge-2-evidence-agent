"""Issue #158: pins the default posture of the per-tool-call scoping gate
and its ``/chat`` dependency seam (``app.chat.get_require_tool_call_scoping``).

Mirrors ``tests/test_answer_grounding_provider.py``'s shape for the issue
#153 gate -- default OFF, and the dependency just returns the flag's current
value (no lazily-built client needed, since the gate is deterministic)."""

from __future__ import annotations

from app.chat import get_require_tool_call_scoping
from app.config import Settings


def test_tool_call_scoping_gate_defaults_to_disabled():
    assert Settings().copilot_extraction_tool_call_scoping_enabled is False


def test_dependency_returns_true_when_the_flag_is_on():
    settings = Settings(copilot_extraction_tool_call_scoping_enabled=True)

    assert get_require_tool_call_scoping(settings=settings) is True


def test_dependency_returns_false_when_the_flag_is_off():
    settings = Settings(copilot_extraction_tool_call_scoping_enabled=False)

    assert get_require_tool_call_scoping(settings=settings) is False
