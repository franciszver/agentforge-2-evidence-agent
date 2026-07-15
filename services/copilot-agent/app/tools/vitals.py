"""``get_vitals`` tool (UC1 backbone).

Endpoint, per P2.3's finding: REST has no patient-level vitals list (only
nested under a specific encounter), so this uses FHIR
``GET /apis/default/fhir/Observation?patient={uuid}&category=vital-signs``
(see ``app/tools/patient_summary.py``'s module docstring). The pid is
resolved to a uuid first via ``app.tools._common.resolve_patient_uuid`` --
both to build the FHIR ``patient`` search param and as the patient-existence
check, so "unknown patient" (raises ``NOT_FOUND``) and "known patient, zero
vitals" (returns ``items=[]``) stay distinguishable. Zero-results shape is
the same omitted-``"entry"``-key Bundle documented in
``app.tools._common.fetch_fhir_observations``.

OpenEMR quirk (panel vs. leaf Observations, live-verified against demo
patient Wanda Moore, pubpid 3): a single vitals-taking produces *multiple*
Observation resources in the Bundle, not one row per vital type mapped
1:1 -- there is a parent "panel" Observation (LOINC 85353-1) with no value
of its own, only a ``hasMember`` list of the individual readings as
sibling Bundle entries; and a "Blood pressure systolic and diastolic"
Observation (LOINC 85354-9) that likewise carries no top-level value --
its systolic (8480-6) and diastolic (8462-4) readings live in its
``component`` array instead. Both shapes are handled by mapping *both* an
Observation's own ``code``/``valueQuantity`` (if it maps to a known
``VitalType`` and has a value) *and* each of its ``component`` entries the
same way -- panels and vital types this tool doesn't track (e.g.
Temperature Location, head circumference) simply produce no match and are
skipped.

``limit``/``since`` filtering and the most-recent-first sort are applied
client-side, for the same reasons as ``app.tools.labs`` (see its module
docstring).
"""

from __future__ import annotations

import datetime
from typing import Any

from app.openemr_client import OpenEmrClient
from app.schemas.common import VitalType
from app.schemas.tools import VitalReadingItem, VitalsOutput
from app.tools._common import fetch_fhir_observations, parse_fhir_datetime, resolve_patient_uuid

# LOINC code -> VitalType, live-verified against demo patient Wanda Moore's
# vital-signs Bundle (see module docstring). "59408-5" is an alternate LOINC
# for oxygen saturation (by pulse oximetry) seen live alongside "2708-6" in
# the same Observation's code.coding array.
_VITAL_TYPE_BY_LOINC = {
    "8480-6": VitalType.BLOOD_PRESSURE_SYSTOLIC,
    "8462-4": VitalType.BLOOD_PRESSURE_DIASTOLIC,
    "8867-4": VitalType.HEART_RATE,
    "8310-5": VitalType.TEMPERATURE,
    "9279-1": VitalType.RESPIRATORY_RATE,
    "2708-6": VitalType.OXYGEN_SATURATION,
    "59408-5": VitalType.OXYGEN_SATURATION,
    "8302-2": VitalType.HEIGHT,
    "29463-7": VitalType.WEIGHT,
    "39156-5": VitalType.BMI,
}


def get_vitals(
    client: OpenEmrClient,
    token: str,
    patient_id: int,
    limit: int | None = None,
    since: datetime.date | None = None,
) -> VitalsOutput:
    patient_uuid = resolve_patient_uuid(client, token, patient_id)

    resources = fetch_fhir_observations(client, token, patient_uuid, "vital-signs")
    items = [item for resource in resources for item in _map_vitals(resource)]

    if since is not None:
        items = [item for item in items if item.date.date() >= since]
    items.sort(key=lambda item: item.date, reverse=True)
    if limit is not None:
        items = items[:limit]

    return VitalsOutput(items=items)


def _map_vitals(resource: dict[str, Any]) -> list[VitalReadingItem]:
    date = parse_fhir_datetime(resource.get("effectiveDateTime"))
    if date is None:
        return []

    items = []
    top_level = _map_reading(resource, date)
    if top_level is not None:
        items.append(top_level)
    for component in resource.get("component") or []:
        if isinstance(component, dict):
            reading = _map_reading(component, date)
            if reading is not None:
                items.append(reading)
    return items


def _map_reading(node: dict[str, Any], date: datetime.datetime) -> VitalReadingItem | None:
    vital_type = _loinc_vital_type(node.get("code"))
    if vital_type is None:
        return None

    quantity = node.get("valueQuantity")
    if not isinstance(quantity, dict):
        return None
    value = quantity.get("value")
    if not isinstance(value, (int, float)):
        return None

    unit = quantity.get("unit")
    return VitalReadingItem(
        vital_type=vital_type,
        value=float(value),
        unit=unit if isinstance(unit, str) else "",
        date=date,
    )


def _loinc_vital_type(code: Any) -> VitalType | None:
    if not isinstance(code, dict):
        return None
    codings = code.get("coding")
    if not isinstance(codings, list):
        return None
    for coding in codings:
        if isinstance(coding, dict):
            loinc_code = coding.get("code")
            if isinstance(loinc_code, str) and loinc_code in _VITAL_TYPE_BY_LOINC:
                return _VITAL_TYPE_BY_LOINC[loinc_code]
    return None
