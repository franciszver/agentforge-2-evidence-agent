"""``get_problems`` tool (UC1/UC3 backbone).

Endpoint, established by probing the live dev API (demo patients Phil
Belford pubpid 1 and Susan Underwood pubpid 2): REST
``GET /apis/default/api/patient/{uuid}/medical_problem`` -- UUID-keyed (the
same OpenEMR REST inconsistency P2.3 documented for allergy/medical_problem/
encounter; see ``app/tools/patient_summary.py``'s module docstring), so the
pid must first be resolved to a uuid via
``app.tools._common.resolve_patient_uuid``. That lookup also doubles as the
patient-existence check: a nonexistent pid raises ``NOT_FOUND`` there and
never reaches the medical_problem call, so "unknown patient" and "known
patient, zero problems" stay distinguishable. Empty state for this
sub-resource is 200 + ``{"data": []}`` (not a 404) -- also per P2.3's finding
for the uuid-keyed resource family.

OpenEMR quirk (diagnosis coding): the ``diagnosis`` field is a dict keyed by
ICD code when a diagnosis is coded (observed live for Phil's "HTN" problem:
``{"401.0": {"code": "401.0", ...}}``), or the empty string ``""`` when not
coded (observed live for Susan's "diabetes" problem). ``icd_code`` reads the
first key of that dict when present, else ``None``.

OpenEMR quirk (status vocabulary gap): the ``lists`` table backing this
sub-resource has no dedicated problem-status column richer than the
boolean-ish ``activity`` (1 = active, 0 = not active) and ``enddate`` (set
when a problem is explicitly resolved) columns -- the same shape P2.4's
``get_medications`` mapped ``activity`` from. As with ``MedicationStatus``,
the ``ProblemStatus`` enum (app/schemas/common.py) carries an ``UNKNOWN``
member for the ambiguous case. Mapping: ``activity == 1`` -> ``ACTIVE``;
``activity == 0`` with a nonempty ``enddate`` -> ``RESOLVED``;
``activity == 0`` without an ``enddate`` -> ``INACTIVE`` (OpenEMR's
data-supported "not active"); any other ``activity`` value (missing, null,
unexpected) -> ``UNKNOWN`` -- fail loud rather than claim ``INACTIVE`` and
risk hiding a genuinely active problem.
"""

from __future__ import annotations

import datetime
from typing import Any

from app.openemr_client import OpenEmrClient
from app.schemas.common import ProblemStatus
from app.schemas.tools import ProblemItem, ProblemsOutput
from app.tools._common import resolve_patient_uuid


def get_problems(client: OpenEmrClient, token: str, patient_id: int) -> ProblemsOutput:
    patient_uuid = resolve_patient_uuid(client, token, patient_id)

    payload = client.get_rest(f"patient/{patient_uuid}/medical_problem", token=token)
    records = payload.get("data") if isinstance(payload, dict) else (payload if isinstance(payload, list) else None)
    items = [_map_problem(record) for record in records or [] if isinstance(record, dict)]
    return ProblemsOutput(items=items)


def _map_problem(record: dict[str, Any]) -> ProblemItem:
    return ProblemItem(
        title=str(record.get("title") or ""),
        icd_code=_icd_code(record.get("diagnosis")),
        status=_map_status(record.get("activity"), record.get("enddate")),
        onset_date=_parse_date(record.get("begdate")),
    )


def _icd_code(diagnosis: Any) -> str | None:
    if not isinstance(diagnosis, dict) or not diagnosis:
        return None
    code = next(iter(diagnosis))
    return code if isinstance(code, str) and code else None


def _map_status(activity: Any, enddate: Any) -> ProblemStatus:
    if activity == 1:
        return ProblemStatus.ACTIVE
    if activity == 0:
        # 0 is OpenEMR's data-supported "not active"; an ``enddate`` marks a
        # formally resolved problem, otherwise it is inactive.
        if isinstance(enddate, str) and enddate:
            return ProblemStatus.RESOLVED
        return ProblemStatus.INACTIVE
    # ``activity`` outside {0, 1} (missing/null/unexpected) -> we cannot tell
    # whether the problem is active. Fail loud with UNKNOWN rather than claim
    # INACTIVE and risk hiding a genuinely active problem.
    return ProblemStatus.UNKNOWN


def _parse_date(value: Any) -> datetime.date | None:
    """Parse an OpenEMR ``"YYYY-MM-DD HH:MM:SS"`` (or bare date) string."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.date.fromisoformat(value[:10])
    except ValueError:
        return None
