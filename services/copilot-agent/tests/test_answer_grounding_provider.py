"""Issue #153: pins the default posture of the claim-in-answer grounding gate
and its ``/chat`` dependency seam (``app.chat.get_require_answer_grounding``).

Mirrors ``tests/test_support_judge_provider.py``'s shape for the issue #47
gate -- default OFF here (the owner has not yet reviewed eval numbers), and
the dependency just returns the flag's current value (no lazily-built client
needed, since the gate is deterministic)."""

from __future__ import annotations

from app.chat import get_require_answer_grounding
from app.config import Settings


def test_claim_answer_grounding_gate_defaults_to_disabled():
    assert Settings().copilot_claim_answer_grounding_enabled is False


def test_dependency_returns_true_when_the_flag_is_on():
    settings = Settings(copilot_claim_answer_grounding_enabled=True)

    assert get_require_answer_grounding(settings=settings) is True


def test_dependency_returns_false_when_the_flag_is_off():
    settings = Settings(copilot_claim_answer_grounding_enabled=False)

    assert get_require_answer_grounding(settings=settings) is False
