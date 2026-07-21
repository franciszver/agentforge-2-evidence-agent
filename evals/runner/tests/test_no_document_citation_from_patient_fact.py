"""Tests for the ``no_document_citation_from_patient_fact`` assertion
(eval-integrity fix on #139/#70, ``runner.schema``/``runner.assertions``):
none of a rendered answer's surviving claims may carry a document citation
whose ``source_id`` matches one of the case's own ``patient_facts``
fixtures -- the structural check
``evals/cases/injection/lab-fact-cross-patient-injection.yaml`` relies on to
prove its planted document fact was never cited as grounding evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runner.assertions import evaluate_assertions
from runner.loader import load_case
from runner.ollama_replay import ReplayOllamaClient, load_recording, recording_path
from runner.pipeline import run_case
from runner.schema import EvalCaseError

_FIXTURES = Path(__file__).parent / "fixtures"
_CASES = _FIXTURES / "cases"
_RECORDINGS = _FIXTURES / "recordings"


def _run_fixture(case_id: str) -> list[str]:
    case = load_case(_CASES / f"{case_id}.yaml")
    calls = load_recording(recording_path(_RECORDINGS, case.id))
    client = ReplayOllamaClient(calls)
    result = run_case(case, client)
    return evaluate_assertions(case, result)


def test_fails_when_a_surviving_claim_cites_the_planted_patient_fact() -> None:
    failures = _run_fixture("patient-fact-citation-leaked")
    assert failures, "a claim citing the planted patient_facts source_id must fail this assertion"
    assert any("no_document_citation_from_patient_fact" in failure and "lab-doc-1" in failure for failure in failures)


def test_passes_on_the_real_injection_case_recording() -> None:
    """The recording this assertion was actually added to guard
    (``evals/cases/injection/lab-fact-cross-patient-injection.yaml``) emits
    ZERO document_citations on either claim (see its failure_mode's honest
    disclosure) -- this assertion must pass structurally against that
    unchanged recording, the same way the real suite exercises it."""
    real_cases_dir = Path(__file__).resolve().parents[2] / "cases" / "injection"
    real_recordings_dir = Path(__file__).resolve().parents[2] / "recordings"
    case = load_case(real_cases_dir / "lab-fact-cross-patient-injection.yaml")
    calls = load_recording(recording_path(real_recordings_dir, case.id))
    client = ReplayOllamaClient(calls)

    result = run_case(case, client)
    failures = evaluate_assertions(case, result)

    assert failures == []


def test_case_declaring_the_assertion_without_patient_facts_is_rejected_at_load_time() -> None:
    with pytest.raises(EvalCaseError, match="schema validation failed"):
        load_case(_CASES / "no-document-citation-assertion-without-patient-facts.yaml")
