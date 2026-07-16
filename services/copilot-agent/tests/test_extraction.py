"""Hermetic tests for the answer->claims extraction pipeline (P3 extraction).

Everything here is hermetic: the extraction LLM is a scripted double, never a
real Ollama call. These tests pin four things the pipeline must guarantee:

  1. **Structural tool-less isolation** -- ``ClaimExtractor`` is in the same
     risk class as ``app.quarantine.QuarantinedSummarizer``: constructed with
     ONLY an extraction-capable client, holding no tool registry / OpenEMR
     client / token, and the module imports none of them. This is the
     load-bearing half of the refined #130 boundary (the extraction LLM may
     see raw values BECAUSE it is tool-less + constrained + deterministically
     validated -- see the module docstring of ``app.extraction``).
  2. **Value-omitted catalog** -- the extraction prompt's citation catalog
     lists ``(call, record, field)`` positions but omits values.
  3. **EAV normalization** -- long-format vitals output is reshaped to
     wide-format so a claim citing the vital *concept* (``field="weight"``)
     resolves VALID against the checker (the #140 fix, 17% -> ~100%).
  4. **Orchestration** -- ``run_verification`` folds extraction + citation
     checking + allergy/interaction checks into one ``VerdictResult`` +
     ``RenderedAnswer``, fail-closed on unverifiable claims.
"""

from __future__ import annotations

import datetime
import inspect
from typing import Any

from app.extraction import (
    ClaimExtractor,
    apply_recency_notice,
    collect_allergies,
    collect_medications,
    mentioned_interactions,
    normalize_raw_results,
    run_verification,
)
from app.ollama_client import OllamaError
from app.openemr_client import OpenEmrClient
from app.planner import PlannerResult, ToolCallTrace
from app.rendering import Notice, RenderedClaim
from app.schemas.common import (
    AllergySeverity,
    MedicationStatus,
    SourceRef,
    VitalType,
)
from app.schemas.planner import ToolName
from app.schemas.tools import (
    AllergiesOutput,
    AllergyItem,
    MedicationItem,
    MedicationsOutput,
    VitalReadingItem,
    VitalsOutput,
)
from app.schemas.verification import Claim, VerifiedAnswer
from app.verdict import Verdict
from app.verification import CacheIndex, CitationStatus, check_claims


# --------------------------------------------------------------------------
# Doubles + fixtures
# --------------------------------------------------------------------------


class _FakeExtractOllama:
    """Scripted extraction client: returns a canned ``VerifiedAnswer`` (or
    raises) and records the messages/schema it was called with."""

    def __init__(self, result: VerifiedAnswer | None = None, *, error: bool = False) -> None:
        self._result = result
        self._error = error
        self.extract_calls: list[tuple[list[dict[str, str]], type]] = []

    def extract(self, prompt_or_messages: Any, schema: type, *, options: Any = None) -> Any:
        self.extract_calls.append((prompt_or_messages, schema))
        if self._error:
            raise OllamaError("scripted extraction failure")
        return self._result if self._result is not None else VerifiedAnswer(claims=[])


class _FakeExtractor:
    """A whole-``ClaimExtractor`` double for orchestration tests: returns a
    fixed claim list, ignoring inputs (the LLM half is exercised separately)."""

    def __init__(self, claims: list[Claim]) -> None:
        self._claims = claims
        self.calls: list[dict[str, Any]] = []

    def extract_claims(
        self, *, answer: str, tools: Any, raw_results: Any
    ) -> list[Claim]:
        self.calls.append({"answer": answer, "tools": list(tools), "raw_results": list(raw_results)})
        return self._claims


def _meds_raw(*items: MedicationItem) -> dict[str, Any]:
    return MedicationsOutput(items=list(items)).model_dump(mode="json")


def _lisinopril() -> MedicationItem:
    return MedicationItem(name="Lisinopril", dose="10 mg", route="oral", status=MedicationStatus.ACTIVE)


def _vitals_raw() -> dict[str, Any]:
    return VitalsOutput(
        items=[
            VitalReadingItem(
                vital_type=VitalType.WEIGHT,
                value=220.0,
                unit="lb_av",
                date=datetime.datetime(2026, 1, 1, 9, 0),
            )
        ]
    ).model_dump(mode="json")


# --------------------------------------------------------------------------
# 1. Structural tool-less isolation (the refined #130 boundary)
# --------------------------------------------------------------------------


def test_extractor_constructor_accepts_only_the_ollama_client():
    params = set(inspect.signature(ClaimExtractor.__init__).parameters) - {"self"}
    assert params == {"ollama_client"}


def test_extractor_instance_holds_no_tool_registry_client_or_token():
    extractor = ClaimExtractor(ollama_client=_FakeExtractOllama())
    for value in vars(extractor).values():
        assert not isinstance(value, OpenEmrClient)
        # No mapping that could be a tool registry, no bearer-token string.
        assert not isinstance(value, dict)
        assert not isinstance(value, str)


def test_extraction_module_does_not_import_tools_or_openemr_client():
    import app.extraction as e

    # The extraction LLM cannot reach a tool: none of the names a tool call
    # needs (the callable, an OpenEmrClient, the registry) exist here.
    assert not hasattr(e, "OpenEmrClient")
    assert not hasattr(e, "TOOL_REGISTRY")
    for tool in ToolName:
        assert not hasattr(e, tool.value)


# --------------------------------------------------------------------------
# 2. Value-omitted catalog + claim parsing
# --------------------------------------------------------------------------


def test_extract_claims_returns_parsed_claims():
    claim = Claim(
        text="She is on Lisinopril.",
        source_refs=[
            SourceRef(tool_call_id="call_0", record_id="0", field="name", asserted_value="Lisinopril")
        ],
    )
    ollama = _FakeExtractOllama(VerifiedAnswer(claims=[claim]))
    extractor = ClaimExtractor(ollama_client=ollama)

    claims = extractor.extract_claims(
        answer="She is on Lisinopril.",
        tools=[ToolName.GET_MEDICATIONS],
        raw_results=[_meds_raw(_lisinopril())],
    )

    assert claims == [claim]
    assert ollama.extract_calls[0][1] is VerifiedAnswer


def test_extract_claims_builds_value_omitted_catalog():
    ollama = _FakeExtractOllama()
    extractor = ClaimExtractor(ollama_client=ollama)

    extractor.extract_claims(
        answer="x",
        tools=[ToolName.GET_MEDICATIONS],
        raw_results=[_meds_raw(_lisinopril())],
    )

    messages, _schema = ollama.extract_calls[0]
    # Inspect only the catalog listing (after the "Catalog:" marker); the
    # instruction preamble legitimately names the "source_refs" output field.
    catalog_section = messages[-1]["content"].split("Catalog:", 1)[1]
    assert "call_0" in catalog_section
    assert "name" in catalog_section
    assert "dose" in catalog_section
    # The provenance hook is never listed as a citable field.
    assert "source_refs" not in catalog_section
    # Values are omitted from the catalog -- only positions are listed.
    assert "Lisinopril" not in catalog_section


def test_extract_claims_short_circuits_when_no_records():
    ollama = _FakeExtractOllama()
    extractor = ClaimExtractor(ollama_client=ollama)

    claims = extractor.extract_claims(answer="I can't answer that.", tools=[], raw_results=[])

    assert claims == []
    assert ollama.extract_calls == []  # no pointless model call when nothing is citable


def test_extract_claims_returns_empty_on_extraction_error():
    ollama = _FakeExtractOllama(error=True)
    extractor = ClaimExtractor(ollama_client=ollama)

    claims = extractor.extract_claims(
        answer="x",
        tools=[ToolName.GET_MEDICATIONS],
        raw_results=[_meds_raw(_lisinopril())],
    )

    assert claims == []


# --------------------------------------------------------------------------
# 3. EAV normalization (the #140 vitals fix)
# --------------------------------------------------------------------------


def test_normalize_reshapes_vitals_to_wide_format():
    normalized = normalize_raw_results([ToolName.GET_VITALS], [_vitals_raw()])

    record = normalized[0]["items"][0]
    # The vital concept is now a real field name carrying its value.
    assert record["weight"] == 220.0
    # The long-format EAV keys are gone (no ambiguous field="value").
    assert "vital_type" not in record
    assert "value" not in record
    # Non-EAV fields survive.
    assert record["unit"] == "lb_av"


def test_normalize_leaves_wide_format_tools_unchanged():
    raw = [_meds_raw(_lisinopril())]
    assert normalize_raw_results([ToolName.GET_MEDICATIONS], raw) == raw


def test_normalize_preserves_none_entries():
    assert normalize_raw_results([ToolName.GET_VITALS], [None]) == [None]


def test_normalized_vitals_citation_resolves_valid():
    normalized = normalize_raw_results([ToolName.GET_VITALS], [_vitals_raw()])
    index = CacheIndex.from_raw_results(normalized)
    claim = Claim(
        text="Weight is 220 lb.",
        source_refs=[
            SourceRef(tool_call_id="call_0", record_id="0", field="weight", asserted_value="220")
        ],
    )

    results = check_claims([claim], index)

    assert results[0].passed


def test_unnormalized_vitals_concept_citation_fails_unknown_field():
    # Proves the normalization is load-bearing: without it, citing the concept
    # ("weight") is UNKNOWN_FIELD -- exactly the #140 defect.
    index = CacheIndex.from_raw_results([_vitals_raw()])
    claim = Claim(
        text="Weight is 220 lb.",
        source_refs=[
            SourceRef(tool_call_id="call_0", record_id="0", field="weight", asserted_value="220")
        ],
    )

    results = check_claims([claim], index)

    assert not results[0].passed
    assert results[0].citation_results[0].status is CitationStatus.UNKNOWN_FIELD


# --------------------------------------------------------------------------
# 4. Domain-input collection (mentioned meds / allergies for the verdict)
# --------------------------------------------------------------------------


def test_collect_medications_parses_get_medications_raw():
    meds = collect_medications([ToolName.GET_MEDICATIONS], [_meds_raw(_lisinopril())])
    assert [m.name for m in meds] == ["Lisinopril"]


def test_collect_medications_ignores_non_medication_calls():
    assert collect_medications([ToolName.GET_VITALS], [_vitals_raw()]) == []


def test_collect_medications_skips_none_results():
    assert collect_medications([ToolName.GET_MEDICATIONS], [None]) == []


def test_collect_allergies_parses_get_allergies_raw():
    allergies_raw = AllergiesOutput(
        items=[AllergyItem(substance="Ibuprofen", severity=AllergySeverity.SEVERE)]
    ).model_dump(mode="json")
    allergies = collect_allergies([ToolName.GET_ALLERGIES], [allergies_raw])
    assert [a.substance for a in allergies] == ["Ibuprofen"]


def test_mentioned_interactions_requires_at_least_two_drugs():
    # Fewer than two mentioned meds -> no pair to check -> empty, no DB hit.
    assert mentioned_interactions([_lisinopril()]) == []
    assert mentioned_interactions([]) == []


# --------------------------------------------------------------------------
# 5. run_verification orchestration
# --------------------------------------------------------------------------


def _planner_result(answer: str, tool: ToolName, raw: dict[str, Any]) -> PlannerResult:
    trace = [ToolCallTrace(tool=tool, args={}, result={"summary": "quarantined"}, error=None)]
    return PlannerResult(answer=answer, trace=trace, raw_results=[raw])


def test_run_verification_verified_for_grounded_medication_claim():
    result = _planner_result("She is on Lisinopril 10 mg.", ToolName.GET_MEDICATIONS, _meds_raw(_lisinopril()))
    claim = Claim(
        text="She is on Lisinopril 10 mg.",
        source_refs=[
            SourceRef(tool_call_id="call_0", record_id="0", field="name", asserted_value="Lisinopril"),
            SourceRef(tool_call_id="call_0", record_id="0", field="dose", asserted_value="10 mg"),
        ],
    )
    extractor = _FakeExtractor([claim])

    verdict_result, rendered = run_verification(extractor, result)

    assert verdict_result.verdict is Verdict.VERIFIED
    assert len(rendered.segments) == 1
    segment = rendered.segments[0]
    assert isinstance(segment, RenderedClaim)
    assert segment.text == "She is on Lisinopril 10 mg."


def test_run_verification_blocks_and_strips_unverifiable_claim():
    result = _planner_result("She is on Metformin.", ToolName.GET_MEDICATIONS, _meds_raw(_lisinopril()))
    # Extractor asserts a value that is NOT in the record -> VALUE_MISMATCH.
    claim = Claim(
        text="She is on Metformin.",
        source_refs=[
            SourceRef(tool_call_id="call_0", record_id="0", field="name", asserted_value="Metformin")
        ],
    )
    extractor = _FakeExtractor([claim])

    verdict_result, rendered = run_verification(extractor, result)

    assert verdict_result.verdict is Verdict.BLOCKED  # NONE_VERIFIED -> fail closed
    assert isinstance(rendered.segments[0], Notice)


def test_run_verification_folds_allergy_conflict_into_blocked():
    ibuprofen = MedicationItem(name="Ibuprofen", dose="200 mg", route="oral", status=MedicationStatus.ACTIVE)
    meds_raw = _meds_raw(ibuprofen)
    allergies_raw = AllergiesOutput(
        items=[AllergyItem(substance="Ibuprofen", severity=AllergySeverity.SEVERE)]
    ).model_dump(mode="json")
    trace = [
        ToolCallTrace(tool=ToolName.GET_MEDICATIONS, args={}, result={"summary": "q"}, error=None),
        ToolCallTrace(tool=ToolName.GET_ALLERGIES, args={}, result={"summary": "q"}, error=None),
    ]
    result = PlannerResult(
        answer="She takes Ibuprofen.",
        trace=trace,
        raw_results=[meds_raw, allergies_raw],
    )
    claim = Claim(
        text="She takes Ibuprofen.",
        source_refs=[
            SourceRef(tool_call_id="call_0", record_id="0", field="name", asserted_value="Ibuprofen")
        ],
    )
    extractor = _FakeExtractor([claim])

    verdict_result, _rendered = run_verification(extractor, result)

    assert verdict_result.verdict is Verdict.BLOCKED
    assert [c.medication_name for c in verdict_result.allergy_conflicts] == ["Ibuprofen"]


def test_run_verification_normalizes_vitals_before_checking():
    result = _planner_result("Her weight is 220 lb.", ToolName.GET_VITALS, _vitals_raw())
    claim = Claim(
        text="Her weight is 220 lb.",
        source_refs=[
            SourceRef(tool_call_id="call_0", record_id="0", field="weight", asserted_value="220")
        ],
    )
    extractor = _FakeExtractor([claim])

    verdict_result, rendered = run_verification(extractor, result)

    # The concept citation only resolves because run_verification normalized
    # the vitals result before building the checker index.
    assert verdict_result.verdict is Verdict.VERIFIED
    assert isinstance(rendered.segments[0], RenderedClaim)


# --------------------------------------------------------------------------
# 6. apply_recency_notice (#153) -- deterministic, no LLM, no claims needed
# --------------------------------------------------------------------------

_NOW = datetime.datetime(2026, 7, 15)


def test_apply_recency_notice_appends_the_stale_records_date_to_the_answer():
    result = _planner_result(
        "Her current A1c is 7.2%, which is high.",
        ToolName.GET_RECENT_LABS,
        {"items": [{"test_name": "A1c", "value": "7.2", "date": "2014-02-01T09:00:00"}]},
    )

    updated = apply_recency_notice(result, now=_NOW)

    assert "2014-02-01" in updated.answer
    assert updated.answer.startswith("Her current A1c is 7.2%, which is high.")
    # Everything else about the result is untouched.
    assert updated.trace == result.trace
    assert updated.raw_results == result.raw_results


def test_apply_recency_notice_does_not_fire_for_a_fresh_record():
    result = _planner_result(
        "Her weight is 220 lb.",
        ToolName.GET_VITALS,
        {"items": [{"vital_type": "weight", "value": 220, "date": "2026-06-01T09:00:00"}]},
    )

    updated = apply_recency_notice(result, now=_NOW)

    assert updated is result
