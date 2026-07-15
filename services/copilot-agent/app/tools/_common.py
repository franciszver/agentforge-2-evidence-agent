"""Shared helpers for per-section patient-data tools (P2.4+).

Currently just the pid -> uuid lookup needed by patient sub-resources that
are keyed by UUID rather than the internal ``pid`` (an OpenEMR REST API
inconsistency documented in ``app.tools.patient_summary``'s module
docstring). ``get_allergies`` needs this to build its UUID-keyed request
path; ``get_medications`` reuses it purely as a patient-existence check (its
own sub-resource is pid-keyed, so it needs no uuid, but the check keeps
"unknown patient" and "known patient, empty section" distinguishable -- see
each tool's module docstring).

Factored out here rather than duplicated, since two tools need it; kept
minimal and does not touch ``app.tools.patient_summary``'s own private
implementation of the same lookup (P2.3, not otherwise broken).
"""

from __future__ import annotations

from app.openemr_client import ErrorCategory, OpenEmrApiError, OpenEmrClient


def resolve_patient_uuid(client: OpenEmrClient, token: str, patient_id: int) -> str:
    """Fetch the patient roster and return the matching ``pid``'s uuid.

    Raises ``OpenEmrApiError(NOT_FOUND)`` if no record matches -- a missing
    patient is always an error here, never an empty result.
    """
    payload = client.get_rest("patient", token=token)
    records = payload.get("data") if isinstance(payload, dict) else None
    for record in records or []:
        if isinstance(record, dict) and record.get("pid") == patient_id:
            uuid = record.get("uuid")
            if isinstance(uuid, str) and uuid:
                return uuid
            raise OpenEmrApiError(ErrorCategory.UNEXPECTED, "OpenEMR patient record missing uuid")
    raise OpenEmrApiError(ErrorCategory.NOT_FOUND, "OpenEMR patient not found")
