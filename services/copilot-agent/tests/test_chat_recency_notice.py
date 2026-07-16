"""Production-path tests for the #153 recency notice in ``POST /chat``.

These guard the LIVE endpoint's behavior directly, not only via the offline
eval: the ``answer`` SSE frame a real user receives must carry the
deterministic recency notice for a stale cited record, and must NOT carry it
for fresh data. Everything is hermetic -- the planner is a scripted double
and the wall clock is injected via the ``get_clock`` dependency override, so
assertions are deterministic.

The clock is injected tz-AWARE (production reads ``datetime.now(timezone
.utc)``); the stale-lab case pairs it with a NAIVE record date (the shape
OpenEMR commonly stores), and a separate case uses a tz-AWARE record date --
together regression-guarding the naive/aware comparison fix in
``app.verification`` (a mismatch there would raise ``TypeError`` and 500 the
live stream on the first stale record).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.chat import (
    get_claim_extractor,
    get_clock,
    get_planner_factory,
    get_token_validator,
)
from app.main import app
from app.planner import PlannerResult, ToolCallTrace
from app.schemas.planner import ToolName

_FIXED_NOW = datetime(2026, 7, 15, tzinfo=timezone.utc)
_NOTICE_MARKER = "may not reflect the patient's current status"


class _FakePlanner:
    def __init__(self, trace: list[ToolCallTrace], answer: str, raw_results: list[dict | None]) -> None:
        self._result = PlannerResult(answer=answer, trace=trace, raw_results=raw_results)

    def run(self, question: str) -> PlannerResult:
        return self._result


class _FakeExtractor:
    def extract_claims(self, *, answer, tools, raw_results):  # type: ignore[no-untyped-def]
        return []


@pytest.fixture(autouse=True)
def _reset_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def _wire(planner: _FakePlanner) -> None:
    app.dependency_overrides[get_token_validator] = lambda: (lambda token: None)
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: planner)
    app.dependency_overrides[get_claim_extractor] = lambda: _FakeExtractor()
    app.dependency_overrides[get_clock] = lambda: (lambda: _FIXED_NOW)


def _answer_frame(response_text: str) -> str:
    for block in response_text.strip().split("\n\n"):
        lines = block.splitlines()
        if any(line.strip() == "event: answer" for line in lines):
            data = "".join(line[len("data:") :].strip() for line in lines if line.startswith("data:"))
            return json.loads(data)["answer"]
    raise AssertionError(f"no answer frame in response: {response_text!r}")


_client = TestClient(app)


def _post(message: str) -> str:
    response = _client.post(
        "/chat",
        json={"message": message, "patient_id": 3},
        headers={"Authorization": "Bearer good-token"},
    )
    assert response.status_code == 200
    return response.text


def _stale_lab_planner(date: str) -> _FakePlanner:
    trace = [ToolCallTrace(tool=ToolName.GET_RECENT_LABS, args={}, result={"summary": "q"}, error=None)]
    raw_results = [{"items": [{"test_name": "A1c", "value": "7.2", "unit": "%", "date": date}]}]
    return _FakePlanner(trace=trace, answer="Her current A1c is 7.2%, which is high.", raw_results=raw_results)


def test_answer_frame_carries_recency_notice_for_a_stale_cited_record():
    # Aware injected clock vs NAIVE record date -- the exact naive/aware
    # mismatch that would TypeError without the fix.
    _wire(_stale_lab_planner("2014-02-01T09:00:00"))

    answer = _answer_frame(_post("What is her current A1c?"))

    assert _NOTICE_MARKER in answer
    assert "2014-02-01" in answer
    # Original answer is preserved; the notice is appended, not a replacement.
    assert answer.startswith("Her current A1c is 7.2%, which is high.")


def test_answer_frame_has_no_recency_notice_for_fresh_data():
    trace = [ToolCallTrace(tool=ToolName.GET_VITALS, args={}, result={"summary": "q"}, error=None)]
    raw_results = [{"items": [{"vital_type": "weight", "value": 150, "date": "2026-06-01T09:00:00"}]}]
    _wire(_FakePlanner(trace=trace, answer="Her weight is 150 lb.", raw_results=raw_results))

    answer = _answer_frame(_post("What is her weight?"))

    assert _NOTICE_MARKER not in answer
    assert answer == "Her weight is 150 lb."


def test_answer_frame_carries_recency_notice_for_a_tz_aware_stale_record_date():
    # tz-AWARE record date + aware clock -- must not 500 and must surface the
    # notice (regression guard for app.verification._as_aware_utc).
    _wire(_stale_lab_planner("2014-02-01T09:00:00+00:00"))

    answer = _answer_frame(_post("What is her current A1c?"))

    assert _NOTICE_MARKER in answer
    assert "2014-02-01" in answer
