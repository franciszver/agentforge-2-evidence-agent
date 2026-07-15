"""Patient-context binding enforcement (P2.16, plan §4.2).

Defense-in-depth NARROWING: every conversation is anchored to the one patient
id the panel was opened on, and the tool layer refuses to act on any other. A
physician's OpenEMR token may well be authorized to fetch any chart -- this
guard does not change that (role enforcement stays in OpenEMR, and this is NOT
a second RBAC). Its single job is to stop the *agent* from being steered to a
different patient than the bound one -- whether by a hallucinated id or a
patient id smuggled into ``tool_args`` via an injected note.

The planner already never reads ``patient_id`` out of ``tool_args`` (see
``app.planner``), so a smuggled id was previously dropped silently. That is
safe but invisible. ``enforce_patient_binding`` turns the drop into a LOUD,
auditable refusal (a typed ``PatientBindingViolation``), so a cross-patient
attempt is recorded rather than silently ignored -- the named invariant the
plan sequenced here, rather than an emergent property of a filter.
"""

from __future__ import annotations

from collections.abc import Mapping


class PatientBindingViolation(Exception):
    """A tool call tried to target a patient other than the bound one.

    Carries both ids purely for auditing; the message names the invariant and
    deliberately contains no record content (zero PHI on refusal).
    """

    def __init__(self, *, bound_patient_id: int, requested_patient_id: str) -> None:
        self.bound_patient_id = bound_patient_id
        self.requested_patient_id = requested_patient_id
        super().__init__(
            "patient-context binding violation: this conversation is bound to a "
            "single patient and refuses to act on any other"
        )


def enforce_patient_binding(*, bound_patient_id: int, tool_args: Mapping[str, str] | None) -> None:
    """Refuse loudly if ``tool_args`` tries to target a divergent patient.

    Raises ``PatientBindingViolation`` when ``tool_args`` carries a
    ``patient_id`` whose value is not exactly the bound id. An absent
    ``patient_id`` -- the normal case -- passes, as does one equal to the bound
    id (redundant, but not a cross-patient attempt). A non-numeric or otherwise
    non-matching value cannot be proven to name the bound patient, so the safe
    default is refusal.

    This is the single enforcement point for the binding invariant; the planner
    calls it before every tool dispatch.
    """
    if not tool_args:
        return
    requested = tool_args.get("patient_id")
    if requested is None:
        return
    if requested.strip() != str(bound_patient_id):
        raise PatientBindingViolation(bound_patient_id=bound_patient_id, requested_patient_id=requested)
