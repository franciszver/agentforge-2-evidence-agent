"""``get_encounters`` tool (UC1 "what changed" / UC4 drill-down backbone).

Endpoint, established by probing the live dev API (demo patients Phil
Belford pubpid 1 and Susan Underwood pubpid 2, the seeded multi-encounter
fixture -- see ``docs/TEST_PLAN.md`` §7): REST
``GET /apis/default/api/patient/{uuid}/encounter`` -- UUID-keyed (the same
OpenEMR REST inconsistency P2.3 documented for allergy/medical_problem/
encounter; see ``app/tools/patient_summary.py``'s module docstring), so the
pid must first be resolved to a uuid via
``app.tools._common.resolve_patient_uuid``. That lookup also doubles as the
patient-existence check: a nonexistent pid raises ``NOT_FOUND`` there and
never reaches the encounter call, so "unknown patient" and "known patient,
zero encounters" stay distinguishable. Empty state for this sub-resource is
200 + ``{"data": []}`` (not a 404) -- also per P2.3's finding for the
uuid-keyed resource family.

OpenEMR quirk (provider name gap): this endpoint's rows carry
``provider_username`` (joined from the ``users`` table) but no first/last
name -- and it was ``null`` for every live demo encounter (the seeded
provider join did not resolve). ``provider`` therefore reads
``provider_username`` verbatim when present, else ``None``; there is no
richer name field this resource exposes.

OpenEMR quirk (encounter type vocabulary): ``class_code`` is the standard
HL7 v3 ``ActEncounterCode`` (verified live via ``list_options`` where
``list_id='_ActEncounterCode'``: ``AMB``/``EMER``/``HH``/``IMP``/``OBSENC``/
``PRENC``/``SS``/``VR``), not the human-facing ``pc_catname`` visit category
(seeded values in this demo dataset -- "Office Visit", "Established
Patient", "New Patient" -- carry no telehealth/hospital/procedure signal).
``_ENCOUNTER_TYPE_MAP`` maps the subset of codes with an unambiguous
``EncounterType`` correspondent (``AMB`` -> ``OFFICE_VISIT``, ``VR`` ->
``TELEHEALTH``, ``IMP``/``EMER`` -> ``HOSPITAL``, ``OBSENC`` -> ``PROCEDURE``,
``HH``/``PRENC``/``SS`` -> ``OTHER``); a missing or unrecognized code maps to
the ``EncounterType.UNKNOWN`` member added here (schema had no UNKNOWN
member previously -- P2.5 precedent: fail loud over mislabeling an
encounter's type).

``start_date``/``end_date``/``limit`` filtering and the most-recent-first
sort are applied client-side after fetching the full encounter list, the
same "fine at demo scale" tradeoff P2.5's ``get_recent_labs``/``get_vitals``
made for their FHIR Bundle fetches (the REST endpoint likely supports
narrower server-side date filters, but client-side keeps this testable
without depending on that support). The date range is inclusive of both
boundary dates.
"""

from __future__ import annotations

import datetime
from typing import Any

from app.openemr_client import OpenEmrClient
from app.schemas.common import EncounterType
from app.schemas.tools import EncounterItem, EncountersOutput
from app.tools._common import resolve_patient_uuid

_ENCOUNTER_TYPE_MAP = {
    "AMB": EncounterType.OFFICE_VISIT,
    "VR": EncounterType.TELEHEALTH,
    "IMP": EncounterType.HOSPITAL,
    "EMER": EncounterType.HOSPITAL,
    "OBSENC": EncounterType.PROCEDURE,
    "HH": EncounterType.OTHER,
    "PRENC": EncounterType.OTHER,
    "SS": EncounterType.OTHER,
}


def get_encounters(
    client: OpenEmrClient,
    token: str,
    patient_id: int,
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
    limit: int | None = None,
) -> EncountersOutput:
    patient_uuid = resolve_patient_uuid(client, token, patient_id)

    payload = client.get_rest(f"patient/{patient_uuid}/encounter", token=token)
    records = payload.get("data") if isinstance(payload, dict) else (payload if isinstance(payload, list) else None)
    items = [item for item in (_map_encounter(record) for record in records or [] if isinstance(record, dict)) if item is not None]

    if start_date is not None:
        items = [item for item in items if item.date.date() >= start_date]
    if end_date is not None:
        items = [item for item in items if item.date.date() <= end_date]
    items.sort(key=lambda item: item.date, reverse=True)
    if limit is not None:
        items = items[:limit]

    return EncountersOutput(items=items)


def _map_encounter(record: dict[str, Any]) -> EncounterItem | None:
    encounter_id = record.get("eid")
    date = _parse_datetime(record.get("date"))
    if not isinstance(encounter_id, int) or encounter_id <= 0 or date is None:
        return None

    reason = record.get("reason")
    provider = record.get("provider_username")
    return EncounterItem(
        encounter_id=encounter_id,
        date=date,
        reason=reason if isinstance(reason, str) and reason else None,
        provider=provider if isinstance(provider, str) and provider else None,
        encounter_type=_map_encounter_type(record.get("class_code")),
    )


def _map_encounter_type(class_code: Any) -> EncounterType:
    if isinstance(class_code, str) and class_code in _ENCOUNTER_TYPE_MAP:
        return _ENCOUNTER_TYPE_MAP[class_code]
    # Missing or unrecognized class_code -> we cannot tell what kind of
    # encounter this was. Fail loud with UNKNOWN rather than guess a
    # concrete type.
    return EncounterType.UNKNOWN


def _parse_datetime(value: Any) -> datetime.datetime | None:
    """Parse an OpenEMR ``"YYYY-MM-DD HH:MM:SS"`` datetime string."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
