"""Hermetic, unit-level tests for issue #158's per-tool-call scoping gate
(``app.tool_call_scoping``), isolated from ``run_verification``'s
orchestration (covered separately in ``tests/test_extraction.py``, section
5c). Mirrors ``tests/test_answer_grounding_provider.py``'s / the grounding
tests' shape for #153's per-claim gate, at the coarser per-CALL granularity
this module implements instead -- see its module docstring for the rule."""

from __future__ import annotations

from typing import Any

from app.schemas.common import SourceRef
from app.schemas.verification import Claim
from app.tool_call_scoping import apply_tool_call_scoping, engaged_call_ids
from app.verification import CacheIndex, CitationStatus, check_claims


def _vitals_raw() -> dict[str, Any]:
    return {"items": [{"weight": 220.0, "unit": "lb_av", "date": "2026-01-01T09:00:00"}]}


def _allergies_raw() -> dict[str, Any]:
    return {"items": [{"substance": "Penicillin", "reaction": None, "severity": "severe"}]}


# --------------------------------------------------------------------------
# engaged_call_ids -- the engagement rule itself
# --------------------------------------------------------------------------


def test_engaged_call_ids_includes_a_call_whose_value_the_answer_quotes():
    engaged = engaged_call_ids([_vitals_raw()], "Her weight is 220 lb.")

    assert engaged == frozenset({"call_0"})


def test_engaged_call_ids_excludes_a_call_the_answer_never_mentions():
    engaged = engaged_call_ids(
        [_vitals_raw(), _allergies_raw()], "Her weight is 220 lb."
    )

    # call_0 (vitals, "220") is engaged; call_1 (allergies, "Penicillin") is
    # not -- the answer shares no tokens with call_1's values.
    assert engaged == frozenset({"call_0"})


def test_engaged_call_ids_a_bare_quoted_number_engages_the_call_containing_it():
    # Numbers count as significant tokens -- an answer that quotes "220"
    # with NO other shared vocabulary still engages the call containing it.
    engaged = engaged_call_ids([_vitals_raw()], "220")

    assert engaged == frozenset({"call_0"})


def test_engaged_call_ids_field_names_alone_do_not_engage_a_call():
    # Only VALUES are tokenized, never field names -- an answer using the
    # word "weight" would not by itself prove engagement if "weight" never
    # appears as a VALUE anywhere (it's the field name here, not a value).
    engaged = engaged_call_ids([_vitals_raw()], "Her weight is unavailable.")

    assert engaged == frozenset()


def test_engaged_call_ids_a_call_with_no_records_is_never_engaged():
    engaged = engaged_call_ids([None, _vitals_raw()], "Her weight is 220 lb.")

    assert engaged == frozenset({"call_1"})


def test_engaged_call_ids_empty_raw_results_engages_nothing():
    assert engaged_call_ids([], "Her weight is 220 lb.") == frozenset()


def test_engaged_call_ids_zero_significant_token_answer_engages_no_calls():
    # The documented fail-closed edge case: an answer with nothing left
    # after stopword/single-character filtering ("It is." -> zero
    # significant tokens) engages NO calls, never vacuously all of them.
    engaged = engaged_call_ids([_vitals_raw(), _allergies_raw()], "It is.")

    assert engaged == frozenset()


def test_engaged_call_ids_a_null_field_never_spuriously_engages_a_call():
    # Correctness regression (gate finding #6): a record field whose value
    # is Python ``None`` must NOT contribute the literal token "none" to the
    # call's value-token set -- "none" is not in _STOPWORDS (only "no"/
    # "not" are), so naive ``str(None)`` stringification would make ANY
    # answer containing the word "none" ("No known allergies -- none
    # noted.") spuriously engage a call purely because one of its fields
    # happens to be null, never because of real record data. This call's
    # ONLY field is None -- if the bug were present, this would be the ONE
    # token distinguishing it, and the call would wrongly show as engaged.
    call_with_only_a_null_field = [{"items": [{"reaction": None}]}]

    engaged = engaged_call_ids(call_with_only_a_null_field, "None reported.")

    assert engaged == frozenset()


def test_engaged_call_ids_keeps_boolean_values_as_real_tokens():
    # Adjacent hazard, checked and decided: unlike ``None`` above, a bool
    # value IS a real, citable value (e.g. an "active"/"resolved" status
    # flag) -- it stringifies to "true"/"false" and DOES tokenize, so an
    # answer that echoes it back still proves engagement. Not a bug; pinned
    # here so a future change can't silently start skipping bools too.
    call_with_a_bool_field = [{"active": True}]

    engaged = engaged_call_ids(call_with_a_bool_field, "The record is true today.")

    assert engaged == frozenset({"call_0"})


# --------------------------------------------------------------------------
# apply_tool_call_scoping -- the enforcement half
# --------------------------------------------------------------------------


def test_apply_tool_call_scoping_downgrades_a_citation_of_an_unengaged_call():
    engaged_claim = Claim(
        text="Her weight is 220 lb.",
        source_refs=[SourceRef(tool_call_id="call_0", record_id="0", field="weight", asserted_value="220")],
    )
    unengaged_claim = Claim(
        text="She is allergic to Penicillin.",
        source_refs=[
            SourceRef(tool_call_id="call_1", record_id="0", field="substance", asserted_value="Penicillin")
        ],
    )
    index = CacheIndex.from_raw_results([_vitals_raw(), _allergies_raw()])
    claim_results = check_claims([engaged_claim, unengaged_claim], index)
    assert all(result.passed for result in claim_results)  # both pass provenance before the gate

    scoped = apply_tool_call_scoping(claim_results, frozenset({"call_0"}))

    assert scoped[0].passed  # the engaged claim is untouched
    assert not scoped[1].passed  # the unengaged claim is downgraded
    assert scoped[1].citation_results[0].status is CitationStatus.TOOL_CALL_NOT_ENGAGED


def test_apply_tool_call_scoping_only_downgrades_the_unengaged_citation_not_the_whole_claim():
    # A claim with TWO citations, one to an engaged call and one to an
    # unengaged call: only the unengaged citation is downgraded -- the
    # engaged one keeps VALID (unlike #153's apply_answer_grounding, which
    # downgrades EVERY citation on an ungrounded claim, since that check is
    # a whole-claim verdict; this one is per-citation).
    mixed_claim = Claim(
        text="Her weight is 220 lb and she is allergic to Penicillin.",
        source_refs=[
            SourceRef(tool_call_id="call_0", record_id="0", field="weight", asserted_value="220"),
            SourceRef(tool_call_id="call_1", record_id="0", field="substance", asserted_value="Penicillin"),
        ],
    )
    index = CacheIndex.from_raw_results([_vitals_raw(), _allergies_raw()])
    claim_results = check_claims([mixed_claim], index)
    assert claim_results[0].passed

    scoped = apply_tool_call_scoping(claim_results, frozenset({"call_0"}))

    assert not scoped[0].passed  # AND-aggregation still fails the whole claim
    assert scoped[0].citation_results[0].status is CitationStatus.VALID  # call_0 untouched
    assert scoped[0].citation_results[1].status is CitationStatus.TOOL_CALL_NOT_ENGAGED  # call_1 downgraded


def test_apply_tool_call_scoping_leaves_an_already_failed_claim_unchanged():
    index = CacheIndex.from_raw_results([_vitals_raw()])
    claim = Claim(
        text="Her weight is 300 lb.",
        source_refs=[SourceRef(tool_call_id="call_0", record_id="0", field="weight", asserted_value="300")],
    )
    claim_results = check_claims([claim], index)
    assert not claim_results[0].passed  # VALUE_MISMATCH

    scoped = apply_tool_call_scoping(claim_results, frozenset())

    assert scoped == claim_results
