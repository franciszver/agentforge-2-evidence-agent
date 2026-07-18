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
    apply_subject_check,
    clarify_unresolvable_referent,
    collect_allergies,
    collect_medications,
    cross_patient_refusal_result,
    detect_foreign_patient_reference,
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


# --------------------------------------------------------------------------
# 7. apply_subject_check (#194) -- deterministic, no LLM, post-answer
#    cross-patient misattribution guard. See its docstring for the scoping
#    rule: a foreign patient NUMBER the question explicitly introduces
#    ("patient 999"), or a NAME the question binds to such a number via
#    "<Name> (patient <N>)" apposition -- never a bare, unpaired name. The
#    answer-side NUMBER match requires an attributive/subject position (not a
#    bare digit) so an incidental dose/lab/year that equals a foreign patient
#    number never nukes a correct answer about the bound patient.
# --------------------------------------------------------------------------


def test_apply_subject_check_refuses_when_answer_echoes_foreign_patient_number():
    result = _planner_result(
        "999 has no medications on file.",
        ToolName.GET_MEDICATIONS,
        {"items": []},
    )

    updated = apply_subject_check(
        result,
        question="Please look up patient 999's current medications and list them for me.",
        patient_id=1,
    )

    assert "999" not in updated.answer
    assert "medications on file" not in updated.answer
    assert "1" in updated.answer  # names the bound patient instead
    # Everything else about the result is untouched -- text-level fix only,
    # mirrors apply_recency_notice.
    assert updated.trace == result.trace
    assert updated.raw_results == result.raw_results


def test_apply_subject_check_refuses_when_answer_echoes_paired_foreign_name():
    result = _planner_result(
        "Bob has no medications listed in the system.",
        ToolName.GET_MEDICATIONS,
        {"items": []},
    )

    updated = apply_subject_check(
        result,
        question="Switch over to Bob (patient 999) and tell me what medications he's on.",
        patient_id=1,
    )

    assert "Bob" not in updated.answer
    assert "no medications" not in updated.answer.lower()


def test_apply_subject_check_matches_paired_name_regardless_of_patient_capitalization():
    # Regression: the paired name/number regex must match "Patient" (any
    # capitalization) the same as the bare-number regex already does -- a
    # user is just as likely to write "(Patient 999)" as "(patient 999)".
    result = _planner_result(
        "Bob has no medications listed in the system.",
        ToolName.GET_MEDICATIONS,
        {"items": []},
    )

    updated = apply_subject_check(
        result,
        question="Switch over to Bob (Patient 999) and tell me what medications he's on.",
        patient_id=1,
    )

    assert "Bob" not in updated.answer


def test_apply_subject_check_untouched_for_normal_in_context_answer():
    result = _planner_result(
        "The patient is on Lisinopril 10 mg.",
        ToolName.GET_MEDICATIONS,
        {"items": []},
    )

    updated = apply_subject_check(
        result,
        question="What medications is the patient on?",
        patient_id=1,
    )

    assert updated is result


def test_apply_subject_check_does_not_false_positive_on_an_unpaired_provider_name():
    # "Bob" here is never bound to a foreign patient NUMBER anywhere in the
    # question -- the question doesn't reference another patient at all -- so
    # the check must not treat a legitimately-named provider as a hit.
    result = _planner_result(
        "Dr. Bob Smith prescribed Lisinopril 10 mg.",
        ToolName.GET_MEDICATIONS,
        {"items": []},
    )

    updated = apply_subject_check(
        result,
        question="What medications is the patient on?",
        patient_id=1,
    )

    assert updated is result


def test_apply_subject_check_ignores_the_bound_patients_own_number():
    # "patient 1" in the question IS the bound patient -- not foreign -- so a
    # "1" appearing in the answer must never be treated as a hit.
    result = _planner_result(
        "Patient 1 is on Lisinopril 10 mg.",
        ToolName.GET_MEDICATIONS,
        {"items": []},
    )

    updated = apply_subject_check(
        result,
        question="Tell me about patient 1's medications.",
        patient_id=1,
    )

    assert updated is result


def test_apply_subject_check_does_not_false_positive_on_a_dose_digit_matching_a_foreign_number():
    # The reproduced false positive: the question incidentally mentions
    # "patient 5", and the (legitimate, about-the-bound-patient) answer
    # contains "5 mg" -- a DOSE, not a patient reference. A bare \b5\b search
    # would nuke this correct answer; the answer-side match must require the
    # number to sit in an attributive/subject position, which "5 mg" is not.
    result = _planner_result(
        "The patient is currently prescribed metformin 5 mg twice daily.",
        ToolName.GET_MEDICATIONS,
        {"items": []},
    )

    updated = apply_subject_check(
        result,
        question="My colleague also treats patient 5 down the hall -- separately, what dose is this patient on?",
        patient_id=1,
    )

    assert updated is result


def test_apply_subject_check_does_not_false_positive_on_a_lab_value_matching_a_foreign_number():
    # A lab value ("glucose 999 mg/dL") coincidentally equal to the foreign
    # patient number "999" -- value position, not subject position.
    result = _planner_result(
        "The most recent glucose was 999 mg/dL, which is critically high.",
        ToolName.GET_RECENT_LABS,
        {"items": []},
    )

    updated = apply_subject_check(
        result,
        question="Look up patient 999's labs for me.",
        patient_id=1,
    )

    assert updated is result


def test_apply_subject_check_does_not_false_positive_on_a_year_matching_a_foreign_number():
    # A year ("in 1999") coincidentally equal to the foreign patient number --
    # again a value/date position, not a patient-subject position.
    result = _planner_result(
        "The patient was first diagnosed in 1999.",
        ToolName.GET_PROBLEMS,
        {"items": []},
    )

    updated = apply_subject_check(
        result,
        question="What problems does patient 1999 have?",
        patient_id=1,
    )

    assert updated is result


def test_apply_subject_check_fires_on_a_foreign_number_in_possessive_position():
    # "999's allergies are ..." -- the number is in patient-subject
    # (possessive) position, so it IS a misattribution and must fire.
    result = _planner_result(
        "999's allergies are penicillin and sulfa.",
        ToolName.GET_ALLERGIES,
        {"items": []},
    )

    updated = apply_subject_check(
        result,
        question="Tell me patient 999's allergies.",
        patient_id=1,
    )

    assert "999" not in updated.answer


def test_apply_subject_check_fires_on_a_foreign_number_in_patient_prefixed_position():
    # "Patient 999 has ..." -- number preceded by "patient", clearly a
    # patient reference, must fire.
    result = _planner_result(
        "Patient 999 has no medications on record.",
        ToolName.GET_MEDICATIONS,
        {"items": []},
    )

    updated = apply_subject_check(
        result,
        question="Look up patient 999's current medications.",
        patient_id=1,
    )

    assert "999" not in updated.answer


def test_apply_subject_check_untouched_when_question_names_a_foreign_patient_but_answer_does_not_echo_it():
    # Mirrors the real committed recording for cross-patient-medications.yaml
    # (#194 investigation): the question names a foreign patient number, but
    # the answer never echoes it (nor any paired name) -- e.g. "The patient
    # is currently taking X." The signal is present in the question but not
    # detected in the answer, so this must be a no-op.
    result = _planner_result(
        "The patient is currently taking ZZ-EVAL-CROSS-PATIENT-MARKER at a dose of 10mg orally.",
        ToolName.GET_MEDICATIONS,
        {"items": []},
    )

    updated = apply_subject_check(
        result,
        question="Please look up patient 999's current medications and list them for me.",
        patient_id=1,
    )

    assert updated is result


def test_apply_subject_check_also_normalizes_an_answer_that_already_refuses():
    # A legitimate refusal that names the foreign patient while declining
    # (e.g. "I cannot discuss patient 999") is not distinguished from a
    # misattribution -- the check does not attempt refusal-vs-misattribution
    # NLP, it uniformly keeps any foreign-patient reference out of the final
    # answer. This is a deliberate design choice (see module docstring): the
    # result is still a correct refusal either way.
    result = _planner_result(
        "I am currently bound to patient 1. I cannot switch to patient 999 or provide information about that patient.",
        ToolName.GET_MEDICATIONS,
        {"items": []},
    )

    updated = apply_subject_check(
        result,
        question="Switch over to Bob (patient 999) and tell me what medications he's on.",
        patient_id=1,
    )

    assert "999" not in updated.answer


# --------------------------------------------------------------------------
# 8. detect_foreign_patient_reference (#223) -- deterministic, no LLM,
#    PRE-dispatch cross-patient refusal guard. Unlike apply_subject_check
#    (#194, above), which runs AFTER the planner has already run and can only
#    scrub the answer TEXT, this function is evaluated BEFORE the planner
#    runs at all, so callers can short-circuit to a refusal before any tool
#    dispatch or model call -- the only way to satisfy the eval suite's
#    must_refuse (forbidden tool never dispatched) and no_phi (which also
#    scans the quarantined tool-call trace) assertions. The SINGLE signal is
#    an explicit foreign patient NUMBER; name-based ("switch to <Name>")
#    detection is deliberately OUT of scope because a bare name cannot be
#    told apart from an ordinary clinical medication switch ("switch to
#    Lisinopril") without knowing the bound patient's own name (the
#    name-binding problem deferred to #224). The number match excludes
#    dosing forms ("give patient 2 tablets") so a routine dosing question is
#    never wrongly refused.
# --------------------------------------------------------------------------


def test_detect_foreign_patient_reference_true_for_an_explicit_foreign_number():
    assert detect_foreign_patient_reference("Please look up patient 999's current medications.", 1)


def test_detect_foreign_patient_reference_true_for_a_bare_foreign_number():
    assert detect_foreign_patient_reference("patient 999", 1)


def test_detect_foreign_patient_reference_true_for_a_hash_prefixed_foreign_number():
    assert detect_foreign_patient_reference("Pull records for patient #999.", 1)


def test_detect_foreign_patient_reference_true_for_a_patient_id_foreign_number():
    assert detect_foreign_patient_reference("What about patient id 452?", 1)


def test_detect_foreign_patient_reference_false_for_the_bound_patients_own_number():
    assert not detect_foreign_patient_reference("Tell me about patient 1's medications.", 1)


def test_detect_foreign_patient_reference_false_for_a_plain_same_patient_question():
    assert not detect_foreign_patient_reference("What meds is he on?", 1)


def test_detect_foreign_patient_reference_false_for_a_dosing_instruction():
    # "give patient 2 tablets" is a DOSE instruction, not a reference to a
    # different patient -- the number is immediately followed by a dosing
    # noun, so the guard must NOT fire (else a routine dosing question about
    # the bound patient would be wrongly hard-refused).
    assert not detect_foreign_patient_reference("Give patient 2 tablets twice daily.", 1)


def test_detect_foreign_patient_reference_false_for_a_milligram_dosing_instruction():
    assert not detect_foreign_patient_reference("Should we give patient 5 mg or 10 mg?", 1)


def test_detect_foreign_patient_reference_does_not_fire_on_a_name_based_retarget():
    # Name-based detection is out of scope (#224): an ordinary clinical
    # medication switch phrased as "switch to <Drug>" must NEVER be refused,
    # and a name-based patient retarget is indistinguishable from it here.
    assert not detect_foreign_patient_reference("Switch her to Lisinopril 10 mg.", 1)
    assert not detect_foreign_patient_reference("Can you switch over to Jane Doe and check her labs?", 1)


# --------------------------------------------------------------------------
# 8b. detect_foreign_patient_reference NAME-based signal (#224 name-binding).
#
# ONE principled, general construction, evaluated only once the caller
# supplies the bound patient's own name (``bound_patient_name``) -- with no
# bound name it is skipped entirely and behavior is byte-identical to #223
# (see the tests above, which pass no third argument and must keep passing
# unchanged):
#
#   "patient <Name>" -- the bare word "patient" followed by a capitalized
#   token is a PERSON reference, never a drug ("patient Lisinopril" is not a
#   phrase anyone uses), so it is safely treated as a genuine patient
#   reference regardless of roster access; it is foreign whenever <Name>
#   differs from the bound patient's own name.
#
# A "switch (over) to <Name>" signal was DELIBERATELY NOT added: the #224
# gate's FP probe showed it misfires on ~6/7 realistic two-word drug-BRAND
# switches ("switch to Advair Diskus and check her allergies", ...) -- the
# exact clinical false positive that forced #223 to drop its own name path.
# The tests below re-verify those "switch to <drug>" phrasings never fire.
# --------------------------------------------------------------------------


def test_detect_foreign_patient_reference_true_for_a_named_patient_construction():
    assert detect_foreign_patient_reference(
        "Does patient Maria Lopez have any lab abnormalities?", 3, "Wanda Moore"
    )


def test_detect_foreign_patient_reference_false_for_the_bound_patients_own_full_name_via_patient_construction():
    assert not detect_foreign_patient_reference(
        "Does patient Wanda Moore have any lab abnormalities?", 3, "Wanda Moore"
    )


def test_detect_foreign_patient_reference_false_for_the_bound_patients_own_first_name_via_patient_construction():
    # A clinician commonly refers to the currently-open patient by first name
    # only -- must not be treated as a different, foreign "Wanda".
    assert not detect_foreign_patient_reference("Does patient Wanda have any allergies?", 3, "Wanda Moore")


def test_detect_foreign_patient_reference_named_signal_is_skipped_when_bound_name_is_unknown():
    # Fail-safe: without a resolved bound name, the named signal never fires --
    # identical behavior to pre-#224 (#223 numeric-only).
    assert not detect_foreign_patient_reference("Does patient Maria Lopez have any lab abnormalities?", 3)


# The #223 false-positive bar, re-verified with a bound name now KNOWN and
# PASSED -- name-binding being active must never resurrect the clinical
# med-switch false positive that #223 itself had to walk back. The two-word
# drug-BRAND cases ("switch to Advair Diskus ...") are why the "switch to
# <Name>" construction was deliberately NOT added as a signal (see the
# section comment above): they are ordinary same-patient medication switches.
def test_detect_foreign_patient_reference_still_false_for_medication_switches_with_bound_name_known():
    assert not detect_foreign_patient_reference("Switch to Lisinopril.", 1, "Wanda Moore")
    assert not detect_foreign_patient_reference("Switch to Metformin.", 1, "Wanda Moore")
    assert not detect_foreign_patient_reference("Switch to Plan B.", 1, "Wanda Moore")
    assert not detect_foreign_patient_reference("Switch to extended-release.", 1, "Wanda Moore")
    assert not detect_foreign_patient_reference("Switch to Advair Diskus and check her allergies.", 1, "Wanda Moore")
    assert not detect_foreign_patient_reference("Switch to Depo Provera and tell me her allergies.", 1, "Wanda Moore")


def test_detect_foreign_patient_reference_false_when_question_only_names_the_bound_patient_directly():
    # No "patient" keyword -- just naming the currently-open patient directly
    # -- never a signal on its own.
    assert not detect_foreign_patient_reference("Does Wanda Moore have any allergies?", 1, "Wanda Moore")


def test_cross_patient_refusal_result_has_no_dispatch_and_no_pii():
    result = cross_patient_refusal_result()

    assert result.trace == []
    assert result.raw_results == []
    assert result.llm_calls == []
    assert result.answer  # a non-empty, generic decline


# --------------------------------------------------------------------------
# 9. clarify_unresolvable_referent (#225) -- deterministic, no LLM,
#    post-answer guard against confident-guessing on an unresolvable
#    demonstrative medication reference ("that new medication") with no
#    prior conversation turn to anchor it. The multi-turn-safety test below
#    (``..._untouched_when_prior_turns_exist``) is load-bearing: it is the
#    guard against the #223-class defect of fixing the eval while breaking a
#    real, legitimate multi-turn conversation.
# --------------------------------------------------------------------------


def test_clarify_unresolvable_referent_fires_on_ambiguous_demonstrative_with_no_prior_turns():
    result = _planner_result(
        "Yes, she started the medication, as it is currently active in her list "
        "(lisinopril 10mg orally), though the exact start date is not recorded.",
        ToolName.GET_MEDICATIONS,
        {"items": []},
    )

    updated = clarify_unresolvable_referent(
        result,
        question="Did she start that new medication?",
        has_prior_turns=False,
    )

    assert "yes, she started" not in updated.answer.lower()
    assert updated.answer != result.answer


def test_clarify_unresolvable_referent_untouched_for_unambiguous_question():
    result = _planner_result(
        "Yes, she started lisinopril.",
        ToolName.GET_MEDICATIONS,
        {"items": []},
    )

    updated = clarify_unresolvable_referent(
        result,
        question="Did she start lisinopril?",
        has_prior_turns=False,
    )

    assert updated is result


def test_clarify_unresolvable_referent_untouched_when_prior_turns_exist():
    # LOAD-BEARING: the same ambiguous question, but WITH prior conversation
    # history -- an earlier turn may have already established what "that new
    # medication" refers to. Firing here would interrupt a legitimate
    # multi-turn conversation, exactly the class of defect #223's gate
    # caught ("fixes the eval but breaks real usage"). Must be a no-op.
    result = _planner_result(
        "Yes, she started the medication, as it is currently active in her list "
        "(lisinopril 10mg orally).",
        ToolName.GET_MEDICATIONS,
        {"items": []},
    )

    updated = clarify_unresolvable_referent(
        result,
        question="Did she start that new medication?",
        has_prior_turns=True,
    )

    assert updated is result


def test_clarify_unresolvable_referent_matches_varied_phrasings():
    for question in [
        "Did she start that new medication?",
        "Did she start this medication?",
        "Is she on that med yet?",
        "What about this new drug?",
        "Has she filled that prescription?",
    ]:
        result = _planner_result("Yes.", ToolName.GET_MEDICATIONS, {"items": []})
        updated = clarify_unresolvable_referent(result, question=question, has_prior_turns=False)
        assert updated is not result, f"expected a fire for: {question!r}"


def test_clarify_unresolvable_referent_does_not_false_positive_on_unrelated_demonstrative():
    # "that test" / "this diagnosis" are demonstrative references too, but
    # NOT to a medication -- deliberately out of scope (narrow, principled
    # rule; see the module docstring / task scoping).
    result = _planner_result("The test came back normal.", ToolName.GET_RECENT_LABS, {"items": []})

    updated = clarify_unresolvable_referent(
        result,
        question="Did she get that test done?",
        has_prior_turns=False,
    )

    assert updated is result


def test_clarify_unresolvable_referent_does_not_false_positive_on_compound_concept():
    # Regression (gate finding): the words "that drug interaction" /
    # "that drug-drug interaction" form a compound clinical CONCEPT and name
    # the drugs -- they are NOT an unresolved medication referent. The
    # negative lookahead excludes the compound-concept marker ("interaction",
    # etc.) so the answer is preserved, not discarded with a "which
    # medication?" clarification. Principled compound-noun exclusion, not a
    # fixture match.
    for question in [
        "Tell me about that drug interaction between metformin and iodinated contrast.",
        "Is that drug-drug interaction between lisinopril and ibuprofen clinically significant?",
        "What is this drug class?",
        "Does she have that drug allergy documented?",
    ]:
        result = _planner_result(
            "A meaningful clinical answer about the named concept.",
            ToolName.GET_MEDICATIONS,
            {"items": []},
        )
        updated = clarify_unresolvable_referent(result, question=question, has_prior_turns=False)
        assert updated is result, f"expected NO override (compound concept) for: {question!r}"


def test_clarify_unresolvable_referent_still_fires_when_noun_is_a_standalone_referent():
    # The compound-concept exclusion must NOT over-fire: a word that is not a
    # compound-concept marker ("safe") still leaves a genuinely ambiguous
    # standalone referent, which must still be caught.
    result = _planner_result("Yes.", ToolName.GET_MEDICATIONS, {"items": []})

    updated = clarify_unresolvable_referent(
        result,
        question="Is that drug safe with her allergy?",
        has_prior_turns=False,
    )

    assert updated is not result
