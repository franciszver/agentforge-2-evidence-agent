"""Regression guard for issue #192 gate-1 finding 1.

``evals/runner/issue_192_injection_battery.py`` used to reconstruct
``app.semantic_support.judge_support``'s message shape itself (importing its
private ``_SYSTEM_PROMPT``/``_INSTRUCTIONS_TEMPLATE``/``_CONTEXT_BLOCK_TEMPLATE``
and re-assembling ``messages``) rather than calling the production code path
-- so if that module's message assembly ever changed shape, this battery
would silently keep attacking the OLD shape, reporting resistance for a
prompt that no longer ships. The fix: the runner's ``_judge_full`` now calls
``app.semantic_support.judge_support_full`` directly, the same public seam
``judge_support`` (the actual production entry point) delegates to.

This test proves the runner's call path and the production call path are
byte-identical for the same inputs -- the guard against a future refactor
reintroducing a divergent second copy of the message assembly (e.g. if
``judge_support`` were changed to build its ``messages`` independently of
``judge_support_full`` again, this test would catch the drift; see the PR
report for the mutation proof)."""

from __future__ import annotations

from typing import Any

from app.semantic_support import SemanticSupportJudgement, SupportVerdict, judge_support
from runner.issue_192_injection_battery import JudgeName, _judge_full


class _MessageCapturingJudge:
    """Records the exact ``messages`` list handed to ``.extract`` (never a
    live call -- always returns the same scripted judgement)."""

    def __init__(self, response: SemanticSupportJudgement) -> None:
        self._response = response
        self.messages_seen: list[list[dict[str, str]]] = []

    def extract(self, prompt_or_messages: Any, schema: type, *, options: Any = None) -> Any:
        assert schema is SemanticSupportJudgement
        assert isinstance(prompt_or_messages, list)
        self.messages_seen.append(prompt_or_messages)
        return self._response


def test_runner_semantic_support_path_matches_production_judge_support_messages() -> None:
    claim_text = "The patient's LDL cholesterol was 165 mg/dL, above the target range."
    quote = "Lipid panel results: LDL cholesterol 165 mg/dL. Target LDL below 100 mg/dL."
    response = SemanticSupportJudgement(verdict=SupportVerdict.SUPPORTED, reason="matches")

    production_judge = _MessageCapturingJudge(response)
    judge_support(claim_text, quote, production_judge)

    runner_judge = _MessageCapturingJudge(response)
    _judge_full(runner_judge, JudgeName.SEMANTIC_SUPPORT, claim_text, quote)

    assert production_judge.messages_seen == runner_judge.messages_seen
    assert len(production_judge.messages_seen[0]) == 2  # system + user
