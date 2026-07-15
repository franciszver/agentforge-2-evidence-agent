"""``get_allergies`` tool (UC2 backbone).

Endpoint, established by probing the live dev API (demo patient Phil
Belford, pubpid 1): REST ``GET /apis/default/api/patient/{uuid}/allergy`` --
UUID-keyed (the same OpenEMR REST inconsistency P2.3 documented for
allergy/medical_problem/encounter; see
``app/tools/patient_summary.py``'s module docstring), so the pid must first
be resolved to a uuid via ``app.tools._common.resolve_patient_uuid``. That
lookup also doubles as the patient-existence check: a nonexistent pid raises
``NOT_FOUND`` there and never reaches the allergy call, so "unknown patient"
and "known patient, zero allergies" stay distinguishable. Empty state for
this sub-resource is 200 + ``{"data": []}`` (not a 404) -- also per P2.3's
finding for the uuid-keyed resource family.

OpenEMR quirk (severity vocabulary): the ``severity_al`` column stores an
8-value CCDA severity vocabulary (``unassigned``, ``mild``,
``mild_to_moderate``, ``moderate``, ``moderate_to_severe``, ``severe``,
``life_threatening_severity``, ``fatal`` -- seeded in ``sql/database.sql``
under ``list_id='severity_ccda'``) that does not map 1:1 onto the tool's
3-value ``AllergySeverity`` enum. Values are bucketed, rounding UP toward
the more severe bucket on ambiguity (``mild_to_moderate`` -> ``MODERATE``;
``moderate_to_severe``, ``life_threatening_severity``, ``fatal`` ->
``SEVERE``) -- safer to overestimate a clinical severity than underestimate
it. ``unassigned``, ``null``, and any unrecognized value fall back to
``UNKNOWN``. An empty ``reaction`` string is treated as "not recorded" and
mapped to ``None`` (the field is optional).
"""

from __future__ import annotations

from typing import Any

from app.openemr_client import OpenEmrClient
from app.schemas.common import AllergySeverity
from app.schemas.tools import AllergiesOutput, AllergyItem
from app.tools._common import resolve_patient_uuid

_SEVERITY_MAP = {
    "mild": AllergySeverity.MILD,
    "mild_to_moderate": AllergySeverity.MODERATE,
    "moderate": AllergySeverity.MODERATE,
    "moderate_to_severe": AllergySeverity.SEVERE,
    "severe": AllergySeverity.SEVERE,
    "life_threatening_severity": AllergySeverity.SEVERE,
    "fatal": AllergySeverity.SEVERE,
}


def get_allergies(client: OpenEmrClient, token: str, patient_id: int) -> AllergiesOutput:
    patient_uuid = resolve_patient_uuid(client, token, patient_id)

    payload = client.get_rest(f"patient/{patient_uuid}/allergy", token=token)
    records = payload.get("data") if isinstance(payload, dict) else (payload if isinstance(payload, list) else None)
    items = [_map_allergy(record) for record in records or [] if isinstance(record, dict)]
    return AllergiesOutput(items=items)


def _map_allergy(record: dict[str, Any]) -> AllergyItem:
    reaction = record.get("reaction")
    severity = record.get("severity_al")
    return AllergyItem(
        substance=str(record.get("title") or ""),
        reaction=reaction if isinstance(reaction, str) and reaction else None,
        severity=_SEVERITY_MAP.get(severity, AllergySeverity.UNKNOWN) if isinstance(severity, str) else AllergySeverity.UNKNOWN,
    )
