"""Hermetic tests for the patient-context binding guard (P2.16).

``app.authz.enforce_patient_binding`` is the single, named enforcement point
for the conversation's patient-context binding (plan §4.2): a divergent
patient id smuggled into a tool call is REFUSED loudly (a typed
``PatientBindingViolation``), not silently dropped. This is defense-in-depth
narrowing on top of OpenEMR's RBAC, NOT a second authorization system.
"""

from __future__ import annotations

import pytest

from app.authz import PatientBindingViolation, enforce_patient_binding

BOUND_PATIENT_ID = 42


def test_divergent_smuggled_patient_id_raises_typed_violation() -> None:
    with pytest.raises(PatientBindingViolation):
        enforce_patient_binding(bound_patient_id=BOUND_PATIENT_ID, tool_args={"patient_id": "999999"})


def test_non_numeric_patient_id_is_treated_as_divergent_and_refused() -> None:
    # Can't prove it names the bound patient, so the safe default is refusal.
    with pytest.raises(PatientBindingViolation):
        enforce_patient_binding(bound_patient_id=BOUND_PATIENT_ID, tool_args={"patient_id": "not-a-number"})


def test_absent_patient_id_passes() -> None:
    enforce_patient_binding(bound_patient_id=BOUND_PATIENT_ID, tool_args={"limit": "3"})
    enforce_patient_binding(bound_patient_id=BOUND_PATIENT_ID, tool_args=None)
    enforce_patient_binding(bound_patient_id=BOUND_PATIENT_ID, tool_args={})


def test_patient_id_equal_to_bound_is_redundant_but_allowed() -> None:
    # An id equal to the bound one is not a cross-patient attempt.
    enforce_patient_binding(bound_patient_id=BOUND_PATIENT_ID, tool_args={"patient_id": str(BOUND_PATIENT_ID)})


def test_violation_carries_bound_and_requested_ids_for_audit() -> None:
    with pytest.raises(PatientBindingViolation) as exc_info:
        enforce_patient_binding(bound_patient_id=BOUND_PATIENT_ID, tool_args={"patient_id": "999999"})
    violation = exc_info.value
    assert violation.bound_patient_id == BOUND_PATIENT_ID
    assert str(violation.requested_patient_id) == "999999"


def test_violation_message_leaks_no_record_content() -> None:
    with pytest.raises(PatientBindingViolation) as exc_info:
        enforce_patient_binding(bound_patient_id=BOUND_PATIENT_ID, tool_args={"patient_id": "999999"})
    # The user-facing message names the invariant, not any record data.
    assert "binding" in str(exc_info.value).lower()
