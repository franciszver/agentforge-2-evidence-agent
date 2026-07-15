"""``get_medications`` tool (UC2 backbone).

Endpoint, established by probing the live dev API (demo patient Phil
Belford, pubpid 1): REST ``GET /apis/default/api/patient/{pid}/medication``
-- pid-keyed (same ``ListRestController``-backed resource family P2.3
documented for medication/appointment; see
``app/tools/patient_summary.py``'s module docstring). A patient with zero
medications returns HTTP 404 with an empty body, not 200 + ``[]`` -- treated
as an empty list, not an error.

Before that call, the patient's existence is checked via
``app.tools._common.resolve_patient_uuid`` (its uuid return value is unused
here -- the medication endpoint takes the pid directly). This extra round
trip exists solely so "unknown patient" and "known patient, zero
medications" stay distinguishable: both would otherwise produce the exact
same 404-empty response from the pid-keyed endpoint.

OpenEMR quirk (dose/route gap): this endpoint's rows (the ``lists`` table,
``type='medication'``) carry no dosage or route columns at all -- that data
lives in a separate ``lists_medication.drug_dosage_instructions`` free-text
field which this REST resource does not join or expose. ``dose``/``route``
therefore always fall back to ``""`` (the schema requires non-optional
``str``); there is no source field to populate them from via this endpoint.
Medication status is derived from the boolean-ish ``activity`` column (1 =
active, 0 = discontinued) -- the ``lists`` table has no richer status
vocabulary, so anything else maps to ``MedicationStatus.UNKNOWN``.
"""

from __future__ import annotations

import datetime
from typing import Any

from app.openemr_client import ErrorCategory, OpenEmrApiError, OpenEmrClient
from app.schemas.common import MedicationStatus
from app.schemas.tools import MedicationItem, MedicationsOutput
from app.tools._common import resolve_patient_uuid


def get_medications(client: OpenEmrClient, token: str, patient_id: int) -> MedicationsOutput:
    # Existence check only -- raises NOT_FOUND for an unknown patient rather
    # than letting it look identical to a known patient with zero medications.
    resolve_patient_uuid(client, token, patient_id)

    try:
        payload = client.get_rest(f"patient/{patient_id}/medication", token=token)
    except OpenEmrApiError as exc:
        if exc.category is ErrorCategory.NOT_FOUND:
            return MedicationsOutput(items=[])
        raise

    records = payload if isinstance(payload, list) else (payload.get("data") if isinstance(payload, dict) else None)
    items = [_map_medication(record) for record in records or [] if isinstance(record, dict)]
    return MedicationsOutput(items=items)


def _map_medication(record: dict[str, Any]) -> MedicationItem:
    return MedicationItem(
        name=str(record.get("title") or ""),
        dose=str(record.get("dosage") or ""),
        route=str(record.get("route") or ""),
        status=_map_status(record.get("activity")),
        start_date=_parse_date(record.get("begdate")),
        end_date=_parse_date(record.get("enddate")),
    )


def _map_status(activity: Any) -> MedicationStatus:
    if activity == 1:
        return MedicationStatus.ACTIVE
    if activity == 0:
        return MedicationStatus.DISCONTINUED
    return MedicationStatus.UNKNOWN


def _parse_date(value: Any) -> datetime.date | None:
    """Parse an OpenEMR ``"YYYY-MM-DD HH:MM:SS"`` (or bare date) string."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.date.fromisoformat(value[:10])
    except ValueError:
        return None
