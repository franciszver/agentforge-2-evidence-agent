"""Shared test doubles for ``evals/runner/tests``.

``ReasoningCaptureOllamaClient`` was previously duplicated near-verbatim
between ``test_patient_fact_wiring.py`` and ``test_guideline_context_wiring.py``
(both needed the same thing: a scripted ``OllamaLike`` that answers
immediately, with no tool dispatch, while recording every ``messages`` list
its free-text reasoning (``chat``) call receives). Promoted here,
parameterized by the ``FinalAnswer`` text and the ``VerifiedAnswer`` claims
each test needs, so both wiring tests share one implementation.
"""

from __future__ import annotations

from app.schemas.planner import FinalAnswer, PlannerAction, PlannerDecision
from app.schemas.verification import Claim, VerifiedAnswer
from app.semantic_support import SemanticSupportJudgement, SupportVerdict


class ReasoningCaptureOllamaClient:
    """Scripted double: answers immediately (no tool dispatch needed) and
    records every ``messages`` list the free-text reasoning (``chat``) call
    receives. Its ``extract`` double, for the claim-extraction call, emits
    ``claims`` verbatim -- so a test can also prove ``run_verification``
    built its ``DocumentFactIndex``/citation-attachment pass from whatever
    fixture it was given, not a stale/empty one."""

    def __init__(self, *, final_answer: str = "placeholder", claims: list[Claim] | None = None) -> None:
        self.chat_messages: list[list[dict[str, str]]] = []
        self._final_answer = final_answer
        self._claims = claims if claims is not None else []

    def extract(self, prompt_or_messages, schema, *, options=None):
        if schema is PlannerDecision:
            return PlannerDecision(action=PlannerAction.ANSWER, final_answer="placeholder", reason="answering directly")
        if schema is FinalAnswer:
            return FinalAnswer(answer=self._final_answer)
        if schema is VerifiedAnswer:
            return VerifiedAnswer(claims=self._claims)
        if schema is SemanticSupportJudgement:
            return SemanticSupportJudgement(verdict=SupportVerdict.SUPPORTED, reason="not under test here")
        raise AssertionError(f"unexpected extract schema in this test: {schema}")

    def chat(self, messages, *, options=None) -> str:
        self.chat_messages.append(list(messages))
        return "placeholder reasoning"
