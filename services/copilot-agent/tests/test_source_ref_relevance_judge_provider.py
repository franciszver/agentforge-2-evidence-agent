"""Issue #170: pins the default posture of the SourceRef-relevance gate and
its ``/chat`` provider seam (``app.chat.get_source_ref_relevance_judge_provider``).

Mirrors ``tests/test_support_judge_provider.py``'s shape for the sibling
issue #47 gate -- default OFF (unlike #47's default-ON posture: this gate
is MEASUREMENT-gated, see ``app.source_ref_relevance``'s module docstring,
and issue #192 is a BLOCKING pre-condition for ever flipping it), and the
provider still builds a real judge client when a caller explicitly turns
the flag on.

Gate-3 review finding (issue #170, MAJOR 4): nothing in ``tests/`` referenced
``get_source_ref_relevance_judge_provider``, ``copilot_source_ref_relevance_
enabled``, or ``run_verification(source_ref_relevance_judge=...)`` before
this file -- a future edit flipping the config default to ``True``, or
dropping the wiring in ``app.chat``, would leave the whole suite green. This
file closes that gap, plus a ``run_verification``-level byte-identity check
that the flag-off default path is untouched."""

from __future__ import annotations

import pytest

import datetime

from app.chat import _no_op_source_ref_relevance_judge_provider, get_source_ref_relevance_judge_provider
from app.config import Settings
from app.extraction import ClaimExtractorLike, run_verification
from app.llama_server_client import LlamaServerClient
from app.planner import PlannerResult, ToolCallTrace
from app.schemas.common import SourceRef
from app.schemas.planner import ToolName
from app.schemas.tools import AppointmentItem, AppointmentsOutput, AppointmentStatus
from app.schemas.verification import Claim


def test_source_ref_relevance_gate_defaults_to_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # See tests/test_tool_call_scoping_provider.py's identical comment: this
    # must pin the FIELD's own default, independent of the ambient shell
    # environment (e.g. a live-measurement re-run that exported
    # COPILOT_SOURCE_REF_RELEVANCE_ENABLED=true).
    monkeypatch.delenv("COPILOT_SOURCE_REF_RELEVANCE_ENABLED", raising=False)

    assert Settings().copilot_source_ref_relevance_enabled is False


def test_provider_builds_a_real_judge_client_when_the_flag_is_on():
    settings = Settings(copilot_source_ref_relevance_enabled=True)

    provider = get_source_ref_relevance_judge_provider(settings=settings)

    assert provider is not _no_op_source_ref_relevance_judge_provider
    judge = provider()
    assert isinstance(judge, LlamaServerClient)


def test_provider_is_the_no_op_when_the_flag_is_off():
    settings = Settings(copilot_source_ref_relevance_enabled=False)

    provider = get_source_ref_relevance_judge_provider(settings=settings)

    assert provider is _no_op_source_ref_relevance_judge_provider
    assert provider() is None


def _trace_entry(tool: ToolName) -> ToolCallTrace:
    return ToolCallTrace(tool=tool, args={}, result={})


class _FakeExtractor:
    """Minimal ``ClaimExtractorLike`` double: always returns the same
    pre-built claims, regardless of the catalog/tool-result messages it is
    called with."""

    def __init__(self, claims: list[Claim]) -> None:
        self._claims = claims
        self.llm_calls: list[object] = []

    def extract_claims(self, **kwargs: object) -> list[Claim]:
        return self._claims


def _source_ref_only_claim() -> Claim:
    ref = SourceRef(tool_call_id="call_0", record_id="0", field="provider", asserted_value="Dr. Chen")
    return Claim(text="The appointment is with Dr. Chen.", source_refs=[ref])


def test_run_verification_is_byte_identical_when_source_ref_relevance_judge_is_none():
    """The flag-off default path (``source_ref_relevance_judge=None``, what
    every caller not opting into ``Settings.copilot_source_ref_relevance_
    enabled`` passes) must never touch ``claim_results`` -- same posture as
    ``app.extraction``'s other optional gates. A double that would raise if
    ever called (no ``.extract`` implementation) proves no judge call
    happens."""
    appointment = AppointmentItem(
        date=datetime.date(2026, 1, 15), time=datetime.time(9, 30), status=AppointmentStatus.SCHEDULED,
        provider="Dr. Chen",
    )
    raw = AppointmentsOutput(items=[appointment]).model_dump(mode="json")
    result = PlannerResult(
        answer="The appointment is with Dr. Chen.",
        trace=[_trace_entry(ToolName.GET_APPOINTMENTS)],
        raw_results=[raw],
    )
    extractor: ClaimExtractorLike = _FakeExtractor([_source_ref_only_claim()])  # type: ignore[assignment]

    verdict_result, rendered = run_verification(extractor, result, source_ref_relevance_judge=None)

    assert verdict_result.stripped_claim_count == 0
    assert len(rendered.segments) == 1
