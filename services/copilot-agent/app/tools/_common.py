"""Shared helpers for per-section patient-data tools (P2.4+).

The pid -> uuid lookup needed by patient sub-resources that are keyed by
UUID rather than the internal ``pid`` (an OpenEMR REST API inconsistency
documented in ``app.tools.patient_summary``'s module docstring).
``get_allergies`` needs this to build its UUID-keyed request path;
``get_medications`` reuses it purely as a patient-existence check (its own
sub-resource is pid-keyed, so it needs no uuid, but the check keeps
"unknown patient" and "known patient, empty section" distinguishable -- see
each tool's module docstring). ``get_problems``, ``get_recent_labs``, and
``get_vitals`` (P2.5) reuse it the same way.

Also factors out the FHIR ``Observation`` bundle-fetch used by both
``get_recent_labs`` and ``get_vitals`` (P2.5), and the ISO-8601 datetime
parser both need for ``effectiveDateTime``.

Factored out here rather than duplicated, since multiple tools need each
piece; kept minimal and does not touch ``app.tools.patient_summary``'s own
private implementation of the pid -> uuid lookup (P2.3, not otherwise
broken).
"""

from __future__ import annotations

import datetime
from typing import Any

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


def fetch_fhir_observations(
    client: OpenEmrClient, token: str, patient_uuid: str, category: str
) -> list[dict[str, Any]]:
    """Fetch a category-filtered ``Observation`` Bundle and return its resources.

    Live probing (dev stack, all three demo patients) confirmed the
    zero-results shape omits the ``"entry"`` key entirely (rather than
    returning ``200`` + an empty ``entry`` list) -- handled here as an empty
    result, not an error. A real 403/401/timeout propagates naturally via
    ``OpenEmrClient``.
    """
    bundle = client.get_fhir("Observation", token=token, params={"patient": patient_uuid, "category": category})
    entries = bundle.get("entry") if isinstance(bundle, dict) else None
    resources = []
    for entry in entries or []:
        if isinstance(entry, dict):
            resource = entry.get("resource")
            if isinstance(resource, dict):
                resources.append(resource)
    return resources


def parse_fhir_datetime(value: Any) -> datetime.datetime | None:
    """Parse a FHIR ``effectiveDateTime`` string (e.g. ``"...T21:47:33+00:00"``)."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
