"""Red-first tests for ``evals/runner/census_source_ref_claims.py`` (issue
#130): the offline, deterministic (no LLM, no I/O beyond reading the
committed recordings) structural census over ``evals/recordings/*.json``.
See that module's docstring for the exposure-vs.-uncited invariant this
suite's cases lock down.
"""

from __future__ import annotations

import json
from pathlib import Path

from runner.census_source_ref_claims import (
    CaseCensusRow,
    census_all,
    count_claims_in_calls,
    render_table,
)
from runner.ollama_replay import RecordedCall

_ONLY_SOURCE_REF_CLAIM = {
    "document_citations": [],
    "source_refs": [{"asserted_value": "148.0", "field": "blood_pressure_systolic", "record_id": "0", "tool_call_id": "call_0"}],
    "text": "The patient's systolic blood pressure was 148 mmHg.",
}

_DOCUMENT_CITED_CLAIM = {
    "document_citations": [
        {
            "field_or_chunk_id": "x#y",
            "page_or_section": "Y",
            "quote_or_value": "some quote",
            "source_id": "x",
            "source_type": "guideline_chunk",
        }
    ],
    "source_refs": [],
    "text": "Guideline-backed claim.",
}

_MIXED_CLAIM = {
    "document_citations": [
        {
            "field_or_chunk_id": "x#y",
            "page_or_section": "Y",
            "quote_or_value": "some quote",
            "source_id": "x",
            "source_type": "guideline_chunk",
        }
    ],
    "source_refs": [{"asserted_value": "1", "field": "f", "record_id": "0", "tool_call_id": "call_0"}],
    "text": "Both kinds of citation.",
}

_UNCITED_CLAIM = {
    "document_citations": [],
    "source_refs": [],
    "text": "No citation at all.",
}


def _verified_answer_call(claims: list[dict]) -> dict:
    return {"kind": "extract", "schema": "VerifiedAnswer", "response": {"claims": claims}}


def _planner_decision_call() -> dict:
    # A non-VerifiedAnswer call (e.g. PlannerDecision) must be ignored, not
    # mistaken for a claims-bearing call.
    return {"kind": "extract", "schema": "PlannerDecision", "response": {"action": "answer"}}


def _recorded(*calls: dict) -> list[RecordedCall]:
    """Adapt the JSON-shaped call fixtures above into the typed
    ``RecordedCall`` list ``count_claims_in_calls`` now consumes."""
    return [RecordedCall.from_json(call) for call in calls]


# --- count_claims_in_calls: per-recording counting -------------------------


def test_count_claims_in_calls_counts_source_ref_only_claim() -> None:
    calls = _recorded(_planner_decision_call(), _verified_answer_call([_ONLY_SOURCE_REF_CLAIM]))
    counts = count_claims_in_calls(calls)
    assert counts.total_claims == 1
    assert counts.source_ref_only_claims == 1
    assert counts.uncited_claims == 0


def test_count_claims_in_calls_document_cited_claim_not_counted_as_exposure() -> None:
    calls = _recorded(_verified_answer_call([_DOCUMENT_CITED_CLAIM]))
    counts = count_claims_in_calls(calls)
    assert counts.total_claims == 1
    assert counts.source_ref_only_claims == 0


def test_count_claims_in_calls_mixed_claim_not_counted_as_exposure() -> None:
    # Has a DocumentCitation alongside a SourceRef -- NOT "all SourceRefs".
    calls = _recorded(_verified_answer_call([_MIXED_CLAIM]))
    counts = count_claims_in_calls(calls)
    assert counts.total_claims == 1
    assert counts.source_ref_only_claims == 0


def test_count_claims_in_calls_uncited_claim_counted_separately() -> None:
    calls = _recorded(_verified_answer_call([_UNCITED_CLAIM]))
    counts = count_claims_in_calls(calls)
    assert counts.total_claims == 1
    assert counts.source_ref_only_claims == 0
    assert counts.uncited_claims == 1


def test_count_claims_in_calls_no_verified_answer_call_reports_zero() -> None:
    calls = _recorded(_planner_decision_call())
    counts = count_claims_in_calls(calls)
    assert counts.total_claims == 0
    assert counts.source_ref_only_claims == 0
    assert counts.uncited_claims == 0


def test_count_claims_in_calls_mixed_population_across_claims() -> None:
    calls = _recorded(_verified_answer_call([_ONLY_SOURCE_REF_CLAIM, _DOCUMENT_CITED_CLAIM, _UNCITED_CLAIM]))
    counts = count_claims_in_calls(calls)
    assert counts.total_claims == 3
    assert counts.source_ref_only_claims == 1
    assert counts.uncited_claims == 1


# --- census_all: walking a recordings directory -----------------------------


def test_census_all_walks_every_json_file_and_sums_totals(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    (recordings_dir / "case-a.json").write_text(
        json.dumps({"calls": [_verified_answer_call([_ONLY_SOURCE_REF_CLAIM, _DOCUMENT_CITED_CLAIM])]}),
        encoding="utf-8",
    )
    (recordings_dir / "case-b.json").write_text(
        json.dumps({"calls": [_verified_answer_call([_ONLY_SOURCE_REF_CLAIM])]}),
        encoding="utf-8",
    )
    # A tool-selection-only recording with no claims-bearing call at all.
    (recordings_dir / "case-c.json").write_text(
        json.dumps({"calls": [_planner_decision_call()]}),
        encoding="utf-8",
    )

    rows = census_all(recordings_dir)

    by_case = {row.case_id: row for row in rows}
    assert by_case["case-a"].total_claims == 2
    assert by_case["case-a"].source_ref_only_claims == 1
    assert by_case["case-b"].total_claims == 1
    assert by_case["case-b"].source_ref_only_claims == 1
    assert by_case["case-c"].total_claims == 0

    total_claims = sum(row.total_claims for row in rows)
    total_exposure = sum(row.source_ref_only_claims for row in rows)
    assert total_claims == 3
    assert total_exposure == 2


def test_census_all_rows_sorted_by_case_id(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    (recordings_dir / "zzz-case.json").write_text(json.dumps({"calls": []}), encoding="utf-8")
    (recordings_dir / "aaa-case.json").write_text(json.dumps({"calls": []}), encoding="utf-8")

    rows = census_all(recordings_dir)

    assert [row.case_id for row in rows] == ["aaa-case", "zzz-case"]


# --- render_table: rendering rows to a markdown table -----------------------


def test_render_table_includes_every_case_and_a_total_row() -> None:
    rows = [
        CaseCensusRow(case_id="case-a", total_claims=2, source_ref_only_claims=1, uncited_claims=0),
        CaseCensusRow(case_id="case-b", total_claims=1, source_ref_only_claims=1, uncited_claims=0),
    ]
    table = render_table(rows)
    assert "case-a" in table
    assert "case-b" in table
    assert "**Total**" in table or "Total" in table
    # Total exposure (2) and total claims (3) both surface somewhere in the table.
    assert "3" in table
    assert "2" in table
