"""``get_recent_labs`` tool (UC1/UC3 backbone, e.g. "last three A1c").

Endpoint, per P2.3's finding for vitals/labs: REST has no patient-level labs
list, so this uses FHIR ``GET /apis/default/fhir/Observation?patient={uuid}
&category=laboratory`` (see ``app/tools/patient_summary.py``'s module
docstring). The pid is resolved to a uuid first via
``app.tools._common.resolve_patient_uuid`` -- both to build the FHIR
``patient`` search param and as the patient-existence check, so "unknown
patient" (raises ``NOT_FOUND``) and "known patient, zero labs" (returns
``items=[]``) stay distinguishable.

OpenEMR quirk (empty-bundle shape, live-verified): the pinned demo dataset
ships zero lab Observations for *all three* demo patients (not just Wanda,
pubpid 3 -- docs/TEST_PLAN.md §7's ``no-labs`` note), so this was probed
directly. The zero-results Bundle omits the ``"entry"`` key entirely rather
than returning ``200`` + an empty ``entry`` list; handled in
``app.tools._common.fetch_fhir_observations``. Because no demo patient has
lab data, no live example of a *populated* laboratory Bundle exists -- the
mapping logic below (value/unit/reference-range/interpretation extraction)
is modeled on the standard FHIR R4 ``Observation`` shape, the same resource
family confirmed live for vital-signs (category swapped per the FHIR spec)
and hermetically tested against synthetic fixtures in
``tests/test_tool_labs.py``.

Value handling: unlike vitals, a lab result's value is not always numeric
(e.g. a culture result of "No growth"), so ``LabResultItem.value`` is a
``str``. ``valueQuantity`` is stringified; ``valueString`` and
``valueCodeableConcept`` (text, then first coding's display) are read
verbatim; if none are present the value is ``""``.

Abnormal flag: read from the standard HL7 v3 ``ObservationInterpretation``
codes (N/H/L/HH/LL). Two distinct cases are handled differently for
clinical safety: an *absent* ``interpretation`` field maps to ``NORMAL``
(the standard EHR convention where no flag means "nothing was flagged"),
but an interpretation that *is* present yet carries no code this tool
recognizes (e.g. the HL7 "A"/"AA" abnormal codes, which have no direction
mappable onto HIGH/LOW) maps to ``AbnormalFlag.UNKNOWN`` -- never ``NORMAL``,
so a possibly-abnormal result the system couldn't classify is surfaced as
"interpretation not available" rather than falsely reassuring.

``limit``/``since`` filtering and the most-recent-first sort are applied
client-side after fetching the full category Bundle. The FHIR endpoint
likely supports ``date=ge{since}`` and ``_sort=-date`` server-side, but at
this demo-data scale client-side filtering keeps the logic simple, testable
without depending on the server's search-param support, and consistent with
``get_patient_summary``'s "fine at demo scale" tradeoff for its own patient
lookup.
"""

from __future__ import annotations

import datetime
from typing import Any

from app.openemr_client import OpenEmrClient
from app.schemas.common import AbnormalFlag
from app.schemas.tools import LabResultItem, RecentLabsOutput
from app.tools._common import fetch_fhir_observations, parse_fhir_datetime, resolve_patient_uuid

_ABNORMAL_FLAG_BY_CODE = {
    "N": AbnormalFlag.NORMAL,
    "H": AbnormalFlag.HIGH,
    "L": AbnormalFlag.LOW,
    "HH": AbnormalFlag.CRITICAL_HIGH,
    "LL": AbnormalFlag.CRITICAL_LOW,
}


def get_recent_labs(
    client: OpenEmrClient,
    token: str,
    patient_id: int,
    limit: int | None = None,
    since: datetime.date | None = None,
) -> RecentLabsOutput:
    patient_uuid = resolve_patient_uuid(client, token, patient_id)

    resources = fetch_fhir_observations(client, token, patient_uuid, "laboratory")
    items = [item for item in (_map_lab(resource) for resource in resources) if item is not None]

    if since is not None:
        items = [item for item in items if item.date.date() >= since]
    items.sort(key=lambda item: item.date, reverse=True)
    if limit is not None:
        items = items[:limit]

    return RecentLabsOutput(items=items)


def _map_lab(resource: dict[str, Any]) -> LabResultItem | None:
    date = parse_fhir_datetime(resource.get("effectiveDateTime"))
    if date is None:
        return None
    value, unit = _extract_value(resource)
    return LabResultItem(
        test_name=_test_name(resource.get("code")),
        value=value,
        unit=unit,
        reference_range=_reference_range(resource.get("referenceRange")),
        date=date,
        abnormal_flag=_abnormal_flag(resource.get("interpretation")),
    )


def _test_name(code: Any) -> str:
    if isinstance(code, dict):
        codings = code.get("coding")
        if isinstance(codings, list) and codings and isinstance(codings[0], dict):
            display = codings[0].get("display")
            if isinstance(display, str) and display:
                return display
        text = code.get("text")
        if isinstance(text, str) and text:
            return text
    return ""


def _extract_value(resource: dict[str, Any]) -> tuple[str, str | None]:
    quantity = resource.get("valueQuantity")
    if isinstance(quantity, dict) and "value" in quantity:
        unit = quantity.get("unit")
        return str(quantity["value"]), (unit if isinstance(unit, str) else None)

    value_string = resource.get("valueString")
    if isinstance(value_string, str):
        return value_string, None

    concept = resource.get("valueCodeableConcept")
    if isinstance(concept, dict):
        text = concept.get("text")
        if isinstance(text, str) and text:
            return text, None
        codings = concept.get("coding")
        if isinstance(codings, list) and codings and isinstance(codings[0], dict):
            display = codings[0].get("display")
            if isinstance(display, str):
                return display, None

    return "", None


def _reference_range(ranges: Any) -> str | None:
    if not isinstance(ranges, list) or not ranges or not isinstance(ranges[0], dict):
        return None
    first = ranges[0]

    text = first.get("text")
    if isinstance(text, str) and text:
        return text

    low = first.get("low") if isinstance(first.get("low"), dict) else None
    high = first.get("high") if isinstance(first.get("high"), dict) else None
    low_value = low.get("value") if low else None
    high_value = high.get("value") if high else None
    if low_value is None or high_value is None:
        return None

    unit = (low and low.get("unit")) or (high and high.get("unit"))
    range_text = f"{low_value}-{high_value}"
    return f"{range_text} {unit}" if isinstance(unit, str) and unit else range_text


def _abnormal_flag(interpretation: Any) -> AbnormalFlag:
    # No interpretation recorded at all -> nothing was flagged. An absent
    # FHIR ``interpretation`` field is the standard EHR convention for "not
    # flagged", distinct from an interpretation the system can't classify.
    if not isinstance(interpretation, list) or not interpretation:
        return AbnormalFlag.NORMAL

    for item in interpretation:
        if not isinstance(item, dict):
            continue
        codings = item.get("coding")
        if not isinstance(codings, list):
            continue
        for coding in codings:
            if isinstance(coding, dict):
                code = coding.get("code")
                if isinstance(code, str) and code in _ABNORMAL_FLAG_BY_CODE:
                    return _ABNORMAL_FLAG_BY_CODE[code]

    # An interpretation *is* present but no coding maps to a flag this tool
    # recognizes (e.g. HL7 "A"/"AA" abnormal codes, which carry no direction
    # mappable onto HIGH/LOW). Fail loud with UNKNOWN rather than claim NORMAL
    # -- never label an unrecognized-but-present interpretation as normal.
    return AbnormalFlag.UNKNOWN
