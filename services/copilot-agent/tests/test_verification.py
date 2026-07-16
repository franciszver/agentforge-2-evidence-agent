"""Exhaustive matrix for the deterministic citation checker (P3.2).

``app.verification`` re-validates every claim's ``source_refs`` against the
RAW (pre-quarantine) tool results from the conversation
(``PlannerResult.raw_results``, the verifier-only channel) -- no LLM, no I/O,
no clock. See the module docstring in ``app.verification`` for the design
decisions this matrix exercises: verifying RAW values (so free-text like a
drug name IS checkable -- the trust story), the ``asserted_value`` extension
to ``SourceRef``, the positional ``tool_call_id``/``record_id`` scheme, the
defensive redaction-sentinel branch, and the conservative type-coercion
rules.

Hermetic and fully deterministic: no fixtures touch a real Ollama/OpenEMR.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.quarantine import REDACTED_SENTINEL
from app.schemas.common import SourceRef
from app.schemas.planner import ToolName
from app.schemas.verification import Claim
from app.verification import (
    CacheIndex,
    CitationStatus,
    check_claim,
    check_claims,
    check_source_ref,
    recency_notices,
    stale_record_date,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _index(*raw_results: dict | None) -> CacheIndex:
    """Build a CacheIndex from raw (pre-quarantine) tool-result dumps,
    positionally aligned 1:1 with the trace (call_0, call_1, ...)."""
    return CacheIndex.from_raw_results(list(raw_results))


def _ref(
    *,
    tool_call_id: str = "call_0",
    record_id: str = "0",
    field: str = "status",
    asserted_value: str | None = None,
) -> SourceRef:
    return SourceRef(tool_call_id=tool_call_id, record_id=record_id, field=field, asserted_value=asserted_value)


_MEDS_RESULT = {
    "items": [
        {"name": "Lisinopril", "dose": "10mg", "status": "active", "start_date": "2024-01-01", "end_date": None},
        {"name": "Atorvastatin", "dose": "20mg", "status": "discontinued", "start_date": "2020-05-01", "end_date": "2023-01-01"},
    ]
}

_PATIENT_SUMMARY_RESULT = {"patient_id": 42, "first_name": "Jane", "medication_count": 3}


# ---------------------------------------------------------------------------
# CacheIndex.from_raw_results -- positional tool_call_id / record_id scheme
# ---------------------------------------------------------------------------


def test_empty_index_resolves_nothing():
    index = _index()

    result = check_source_ref(_ref(asserted_value="active"), index)

    assert result.status is CitationStatus.UNKNOWN_TOOL_CALL


def test_tool_call_ids_are_positional_in_trace_order():
    index = _index(_MEDS_RESULT, _PATIENT_SUMMARY_RESULT)

    first_call = check_source_ref(_ref(tool_call_id="call_0", record_id="0", field="status", asserted_value="active"), index)
    second_call = check_source_ref(
        _ref(tool_call_id="call_1", record_id="0", field="first_name", asserted_value="Jane"), index
    )

    assert first_call.status is CitationStatus.VALID
    assert second_call.status is CitationStatus.VALID


def test_list_shaped_result_indexes_items_positionally():
    index = _index(_MEDS_RESULT)

    first_item = check_source_ref(_ref(record_id="0", field="status", asserted_value="active"), index)
    second_item = check_source_ref(_ref(record_id="1", field="status", asserted_value="discontinued"), index)

    assert first_item.status is CitationStatus.VALID
    assert second_item.status is CitationStatus.VALID


def test_single_object_result_is_one_record_at_id_zero():
    index = _index(_PATIENT_SUMMARY_RESULT)

    result = check_source_ref(_ref(record_id="0", field="medication_count", asserted_value="3"), index)

    assert result.status is CitationStatus.VALID


def test_none_entry_registers_the_tool_call_id_with_zero_records():
    # A trace entry that produced no output (binding violation / API error)
    # is carried as ``None`` in raw_results, positionally.
    index = _index(None)

    result = check_source_ref(_ref(record_id="0", field="status", asserted_value="active"), index)

    # The call happened (id known) but produced no records -- distinct from
    # a ref naming a call that was never made (UNKNOWN_TOOL_CALL).
    assert result.status is CitationStatus.UNKNOWN_RECORD


# ---------------------------------------------------------------------------
# check_source_ref -- structural resolution failures
# ---------------------------------------------------------------------------


def test_unknown_tool_call_id_fails():
    index = _index(_MEDS_RESULT)

    result = check_source_ref(_ref(tool_call_id="call_99", asserted_value="active"), index)

    assert result.status is CitationStatus.UNKNOWN_TOOL_CALL


def test_non_numeric_record_id_fails():
    index = _index(_MEDS_RESULT)

    result = check_source_ref(_ref(record_id="med-1", field="status", asserted_value="active"), index)

    assert result.status is CitationStatus.UNKNOWN_RECORD


def test_negative_record_id_fails():
    index = _index(_MEDS_RESULT)

    result = check_source_ref(_ref(record_id="-1", field="status", asserted_value="active"), index)

    assert result.status is CitationStatus.UNKNOWN_RECORD


def test_out_of_range_record_id_fails():
    index = _index(_MEDS_RESULT)

    result = check_source_ref(_ref(record_id="2", field="status", asserted_value="active"), index)

    assert result.status is CitationStatus.UNKNOWN_RECORD


def test_unknown_field_fails():
    index = _index(_MEDS_RESULT)

    result = check_source_ref(_ref(record_id="0", field="nonexistent_field", asserted_value="x"), index)

    assert result.status is CitationStatus.UNKNOWN_FIELD


def test_redacted_field_is_defensive_fail_closed():
    # A raw result should NEVER contain the quarantine sentinel; this is the
    # belt-and-suspenders branch. If one somehow does, fail closed rather
    # than compare an asserted value against placeholder text.
    index = _index({"items": [{"name": REDACTED_SENTINEL, "status": "active"}]})

    result = check_source_ref(_ref(record_id="0", field="name", asserted_value="Lisinopril"), index)

    assert result.status is CitationStatus.REDACTED_FIELD


def test_missing_asserted_value_fails_closed():
    index = _index(_MEDS_RESULT)

    result = check_source_ref(_ref(record_id="0", field="status", asserted_value=None), index)

    assert result.status is CitationStatus.NO_ASSERTED_VALUE


def test_null_cached_field_value_is_a_mismatch():
    index = _index(_MEDS_RESULT)

    result = check_source_ref(_ref(record_id="0", field="end_date", asserted_value="2025-01-01"), index)

    assert result.status is CitationStatus.VALUE_MISMATCH


def test_non_scalar_cached_field_value_is_a_mismatch():
    index = _index({"items": [{"tags": ["a", "b"]}]})

    result = check_source_ref(_ref(record_id="0", field="tags", asserted_value="a"), index)

    assert result.status is CitationStatus.VALUE_MISMATCH


# ---------------------------------------------------------------------------
# check_source_ref -- valid citation (incl. the free-text trust story)
# ---------------------------------------------------------------------------


def test_valid_citation_passes():
    index = _index(_MEDS_RESULT)

    result = check_source_ref(_ref(record_id="0", field="status", asserted_value="active"), index)

    assert result.status is CitationStatus.VALID
    assert result.passed is True


def test_free_text_drug_name_in_record_verifies_the_trust_story():
    # THE demo case: the planner LLM asserts "Lisinopril" (derived from the
    # quarantine summary); the checker deterministically confirms the RAW
    # medication.name really is "Lisinopril" -> VALID. This is exactly what
    # verifying against the quarantine-redacted skeleton could NOT do.
    index = _index(_MEDS_RESULT)

    result = check_source_ref(_ref(record_id="0", field="name", asserted_value="Lisinopril"), index)

    assert result.status is CitationStatus.VALID


def test_hallucinated_drug_name_not_in_record_is_a_mismatch():
    # The other half of the trust story: a drug the patient is NOT on
    # mismatches the raw record value -> VALUE_MISMATCH -> P3.3 strips it.
    index = _index(_MEDS_RESULT)

    result = check_source_ref(_ref(record_id="0", field="name", asserted_value="Metformin"), index)

    assert result.status is CitationStatus.VALUE_MISMATCH


# ---------------------------------------------------------------------------
# Type-coercion edges (Q3)
# ---------------------------------------------------------------------------


def test_string_case_insensitive_match():
    index = _index({"items": [{"name": "Lisinopril"}]})

    result = check_source_ref(_ref(record_id="0", field="name", asserted_value="lisinopril"), index)

    assert result.status is CitationStatus.VALID


def test_string_whitespace_insensitive_match():
    index = _index({"items": [{"name": "Lisinopril"}]})

    result = check_source_ref(_ref(record_id="0", field="name", asserted_value="  Lisinopril  "), index)

    assert result.status is CitationStatus.VALID


def test_string_mismatch():
    index = _index({"items": [{"name": "Lisinopril"}]})

    result = check_source_ref(_ref(record_id="0", field="name", asserted_value="Atorvastatin"), index)

    assert result.status is CitationStatus.VALUE_MISMATCH


def test_enum_value_case_insensitive_match():
    index = _index(_MEDS_RESULT)

    result = check_source_ref(_ref(record_id="0", field="status", asserted_value="Active"), index)

    assert result.status is CitationStatus.VALID


def test_enum_value_mismatch():
    index = _index(_MEDS_RESULT)

    result = check_source_ref(_ref(record_id="0", field="status", asserted_value="discontinued"), index)

    assert result.status is CitationStatus.VALUE_MISMATCH


def test_int_string_matches_int_value():
    index = _index(_PATIENT_SUMMARY_RESULT)

    result = check_source_ref(_ref(record_id="0", field="medication_count", asserted_value="3"), index)

    assert result.status is CitationStatus.VALID


def test_float_string_matches_int_value_via_numeric_equality():
    index = _index(_PATIENT_SUMMARY_RESULT)

    result = check_source_ref(_ref(record_id="0", field="medication_count", asserted_value="3.0"), index)

    assert result.status is CitationStatus.VALID


def test_numeric_string_mismatch():
    index = _index(_PATIENT_SUMMARY_RESULT)

    result = check_source_ref(_ref(record_id="0", field="medication_count", asserted_value="4"), index)

    assert result.status is CitationStatus.VALUE_MISMATCH


def test_non_numeric_string_against_numeric_value_is_a_mismatch():
    index = _index(_PATIENT_SUMMARY_RESULT)

    result = check_source_ref(_ref(record_id="0", field="medication_count", asserted_value="a lot"), index)

    assert result.status is CitationStatus.VALUE_MISMATCH


def test_date_string_exact_match():
    index = _index(_MEDS_RESULT)

    result = check_source_ref(_ref(record_id="0", field="start_date", asserted_value="2024-01-01"), index)

    assert result.status is CitationStatus.VALID


def test_date_string_mismatch():
    index = _index(_MEDS_RESULT)

    result = check_source_ref(_ref(record_id="0", field="start_date", asserted_value="2024-02-02"), index)

    assert result.status is CitationStatus.VALUE_MISMATCH


def test_date_truncated_to_date_only_does_not_match_full_timestamp():
    """Conservative-by-design: no date-specific parsing/truncation."""
    index = _index({"items": [{"date": "2026-06-01T09:00:00"}]})

    result = check_source_ref(_ref(record_id="0", field="date", asserted_value="2026-06-01"), index)

    assert result.status is CitationStatus.VALUE_MISMATCH


def test_bool_true_string_matches_true():
    index = _index({"items": [{"flag": True}]})

    result = check_source_ref(_ref(record_id="0", field="flag", asserted_value="true"), index)

    assert result.status is CitationStatus.VALID


def test_bool_false_string_matches_false():
    index = _index({"items": [{"flag": False}]})

    result = check_source_ref(_ref(record_id="0", field="flag", asserted_value="FALSE"), index)

    assert result.status is CitationStatus.VALID


def test_bool_string_mismatch_wrong_value():
    index = _index({"items": [{"flag": True}]})

    result = check_source_ref(_ref(record_id="0", field="flag", asserted_value="false"), index)

    assert result.status is CitationStatus.VALUE_MISMATCH


def test_bool_string_invalid_token_is_a_mismatch():
    index = _index({"items": [{"flag": True}]})

    result = check_source_ref(_ref(record_id="0", field="flag", asserted_value="yes"), index)

    assert result.status is CitationStatus.VALUE_MISMATCH


# ---------------------------------------------------------------------------
# check_claim -- AND semantics across multiple source_refs
# ---------------------------------------------------------------------------


def test_claim_with_single_valid_citation_passes():
    index = _index(_MEDS_RESULT)
    claim = Claim(
        text="The medication is active.",
        source_refs=[_ref(record_id="0", field="status", asserted_value="active")],
    )

    result = check_claim(claim, index)

    assert result.passed is True
    assert len(result.citation_results) == 1


def test_claim_with_single_invalid_citation_fails():
    index = _index(_MEDS_RESULT)
    claim = Claim(
        text="The medication is discontinued.",
        source_refs=[_ref(record_id="0", field="status", asserted_value="discontinued")],
    )

    result = check_claim(claim, index)

    assert result.passed is False


def test_claim_with_multiple_refs_all_valid_passes():
    index = _index(_MEDS_RESULT)
    claim = Claim(
        text="Lisinopril, started 2024-01-01, currently active.",
        source_refs=[
            _ref(record_id="0", field="name", asserted_value="Lisinopril"),
            _ref(record_id="0", field="start_date", asserted_value="2024-01-01"),
            _ref(record_id="0", field="status", asserted_value="active"),
        ],
    )

    result = check_claim(claim, index)

    assert result.passed is True
    assert all(r.passed for r in result.citation_results)


def test_claim_with_multiple_refs_one_invalid_fails_the_whole_claim():
    """AND semantics: one bad citation sinks an otherwise-valid claim."""
    index = _index(_MEDS_RESULT)
    claim = Claim(
        text="Started 2024-01-01, currently discontinued.",
        source_refs=[
            _ref(record_id="0", field="start_date", asserted_value="2024-01-01"),
            _ref(record_id="0", field="status", asserted_value="discontinued"),
        ],
    )

    result = check_claim(claim, index)

    assert result.passed is False
    # Both citations are still reported -- not short-circuited -- for P3.3.
    assert len(result.citation_results) == 2
    assert result.citation_results[0].passed is True
    assert result.citation_results[1].passed is False


def test_claim_with_zero_citations_reaching_the_checker_fails_closed():
    """Not reachable via normal ``Claim`` construction (P3.1's min_length=1),
    but nothing prevents a caller from bypassing validation -- the vacuous
    ``all([])`` trap must not silently verify a claim with no citations."""
    claim = Claim.model_construct(text="Unsupported claim.", source_refs=[])
    index = _index()

    result = check_claim(claim, index)

    assert result.citation_results == []
    assert result.passed is False


# ---------------------------------------------------------------------------
# check_claims -- batch
# ---------------------------------------------------------------------------


def test_check_claims_on_empty_list_returns_empty_list():
    index = _index(_MEDS_RESULT)

    assert check_claims([], index) == []


def test_check_claims_reports_mixed_pass_fail_for_multiple_claims():
    index = _index(_MEDS_RESULT)
    passing = Claim(
        text="Active.",
        source_refs=[_ref(record_id="0", field="status", asserted_value="active")],
    )
    failing = Claim(
        text="Discontinued.",
        source_refs=[_ref(record_id="0", field="status", asserted_value="discontinued")],
    )

    results = check_claims([passing, failing], index)

    assert [r.passed for r in results] == [True, False]


# ---------------------------------------------------------------------------
# stale_record_date / recency_notices (#153) -- deterministic recency check
#
# See the module docstring's "Recency notices" section for why this scans
# every record actually returned this turn (``PlannerResult.raw_results``)
# rather than only claim-cited ones: claim extraction is an LLM call the eval
# harness only makes lazily, so a citation-gated check could never fire for a
# turn whose recorded run has no extraction call -- exactly the stale_data
# cases this feature targets.
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 7, 15)  # fixed clock for every hermetic test below


def test_stale_record_date_fires_for_a_lab_result_older_than_threshold():
    record = {"date": "2014-02-01T09:00:00"}

    result = stale_record_date(ToolName.GET_RECENT_LABS, record, _NOW)

    assert result == datetime(2014, 2, 1, 9, 0, 0)


def test_stale_record_date_fires_for_a_vitals_reading_older_than_threshold():
    record = {"date": "2014-02-01T09:00:00"}

    result = stale_record_date(ToolName.GET_VITALS, record, _NOW)

    assert result == datetime(2014, 2, 1, 9, 0, 0)


def test_stale_record_date_fires_for_an_encounter_older_than_threshold():
    record = {"date": "2014-02-01T10:00:00"}

    result = stale_record_date(ToolName.GET_ENCOUNTERS, record, _NOW)

    assert result == datetime(2014, 2, 1, 10, 0, 0)


def test_stale_record_date_does_not_fire_for_a_fresh_record():
    record = {"date": "2026-06-01T09:00:00"}

    result = stale_record_date(ToolName.GET_RECENT_LABS, record, _NOW)

    assert result is None


def test_stale_record_date_does_not_fire_for_a_tool_with_no_threshold():
    # get_medications has no recency threshold -- a stale-looking date on an
    # unmonitored tool is not a claim this checker makes.
    record = {"date": "2014-02-01T09:00:00"}

    result = stale_record_date(ToolName.GET_MEDICATIONS, record, _NOW)

    assert result is None


def test_stale_record_date_does_not_fire_for_a_missing_date_field():
    record = {"test_name": "A1c"}

    result = stale_record_date(ToolName.GET_RECENT_LABS, record, _NOW)

    assert result is None


def test_stale_record_date_does_not_fire_for_an_unparseable_date():
    record = {"date": "not-a-date"}

    result = stale_record_date(ToolName.GET_RECENT_LABS, record, _NOW)

    assert result is None


def test_stale_record_date_does_not_fire_for_a_non_string_date():
    record = {"date": 12345}

    result = stale_record_date(ToolName.GET_RECENT_LABS, record, _NOW)

    assert result is None


def test_recency_notices_includes_the_stale_records_date():
    tools = [ToolName.GET_RECENT_LABS]
    raw_results = [{"items": [{"test_name": "A1c", "value": "7.2", "date": "2014-02-01T09:00:00"}]}]

    notices = recency_notices(tools, raw_results, _NOW)

    assert len(notices) == 1
    assert "2014-02-01" in notices[0]
    # Clinician-facing label, not the raw snake_case enum value.
    assert "lab results" in notices[0]
    assert "get_recent_labs" not in notices[0]
    assert "may not reflect the patient's current status" in notices[0]


def test_recency_notices_empty_for_all_fresh_records():
    tools = [ToolName.GET_VITALS]
    raw_results = [{"items": [{"vital_type": "weight", "value": 150, "date": "2026-06-01T09:00:00"}]}]

    assert recency_notices(tools, raw_results, _NOW) == []


def test_recency_notices_empty_for_no_tool_calls():
    assert recency_notices([], [], _NOW) == []


def test_recency_notices_skips_a_call_with_no_output():
    # A trace entry that produced no output (binding violation / API error)
    # is ``None`` in raw_results -- zero records, zero notices.
    assert recency_notices([ToolName.GET_RECENT_LABS], [None], _NOW) == []


def test_recency_notices_dedupes_multiple_records_with_the_same_stale_date():
    # stale-only-vitals.yaml's real shape: two vitals readings (systolic,
    # diastolic) sharing one stale date -- one notice, not two.
    tools = [ToolName.GET_VITALS]
    raw_results = [
        {
            "items": [
                {"vital_type": "blood_pressure_systolic", "value": 118, "date": "2014-02-01T09:00:00"},
                {"vital_type": "blood_pressure_diastolic", "value": 76, "date": "2014-02-01T09:00:00"},
            ]
        }
    ]

    notices = recency_notices(tools, raw_results, _NOW)

    assert len(notices) == 1


# ---------------------------------------------------------------------------
# Timezone safety (#153): real OpenEMR/FHIR record dates can be tz-AWARE
# (offset-qualified), while an injected ``now`` may be naive (the eval's fixed
# clock) or aware (production ``datetime.now(timezone.utc)``). Comparing a
# naive against an aware datetime raises ``TypeError`` -- which would crash a
# live ``/chat`` on the first stale record -- so the comparison must normalize
# both sides. Naive datetimes are treated as UTC. The returned date is the raw
# parsed value (aware stays aware, naive stays naive) -- only the staleness
# COMPARISON is normalized.
# ---------------------------------------------------------------------------


def test_stale_record_date_tz_aware_record_against_naive_now():
    record = {"date": "2014-02-01T09:00:00+00:00"}  # aware

    result = stale_record_date(ToolName.GET_RECENT_LABS, record, _NOW)  # _NOW naive

    assert result == datetime(2014, 2, 1, 9, 0, 0, tzinfo=timezone.utc)


def test_stale_record_date_naive_record_against_tz_aware_now():
    record = {"date": "2014-02-01T09:00:00"}  # naive
    now_aware = datetime(2026, 7, 15, tzinfo=timezone.utc)

    result = stale_record_date(ToolName.GET_RECENT_LABS, record, now_aware)

    assert result == datetime(2014, 2, 1, 9, 0, 0)


def test_stale_record_date_fresh_tz_aware_record_against_naive_now_does_not_fire():
    record = {"date": "2026-06-01T09:00:00+00:00"}  # aware, recent

    result = stale_record_date(ToolName.GET_RECENT_LABS, record, _NOW)

    assert result is None


def test_recency_notices_tz_aware_records_do_not_raise():
    tools = [ToolName.GET_VITALS]
    raw_results = [
        {"items": [{"vital_type": "weight", "value": 150, "date": "2014-02-01T09:00:00+00:00"}]}
    ]

    notices = recency_notices(tools, raw_results, _NOW)  # naive now, aware record

    assert len(notices) == 1
    assert "2014-02-01" in notices[0]


def test_recency_notice_uses_clinician_friendly_labels_per_tool():
    # The label mapping surfaces clinician-facing wording, never the raw
    # snake_case enum, for every reading tool that carries a threshold.
    stale = "2014-02-01T09:00:00"
    vitals = recency_notices([ToolName.GET_VITALS], [{"items": [{"date": stale}]}], _NOW)
    encounters = recency_notices([ToolName.GET_ENCOUNTERS], [{"items": [{"date": stale}]}], _NOW)

    assert "vital signs" in vitals[0] and "get_vitals" not in vitals[0]
    assert "encounter records" in encounters[0] and "get_encounters" not in encounters[0]
