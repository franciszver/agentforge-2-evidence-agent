"""``get_patient_summary`` tool (UC1 -- pre-visit brief backbone).

Fetches a patient's demographics plus per-section record counts
(medications, allergies, problems, recent labs, vitals, encounters,
appointments) and maps them into ``PatientSummaryOutput``. This is a
synthesis tool: it returns demographics + counts only, not the underlying
records (those are the P2.4-P2.8 per-section tools). ``source_refs`` is
left ``None`` -- populating it is the P3.1 verification layer's job.

Endpoint choices, established by probing the live dev API (demo patient
Phil Belford, pubpid 1):

  * Demographics: REST ``GET /apis/default/api/patient`` (the full list,
    filtered client-side by ``pid``). REST's flat JSON needs no unwrapping
    of FHIR US-Core extensions (birthsex/race/ethnicity codings) to reach
    name/DOB/sex, and it conveniently returns the patient's ``uuid`` in the
    same payload -- needed anyway for the UUID-keyed sub-resource calls
    below, saving a round trip. NOTE: OpenEMR's REST patient list only
    supports filtering by demographic fields (fname, lname, DOB, ...); the
    internal ``pid`` this tool takes as input is deliberately not a public
    filter, so this fetches the full roster and selects the matching
    record. Fine at demo/dev scale; a large patient panel would want a more
    targeted lookup, but that is out of scope for this tool.
  * medication / appointment counts: REST
    ``GET /apis/default/api/patient/{pid}/medication`` and .../appointment
    -- these sub-resources are keyed by the numeric ``pid`` directly (no
    uuid lookup needed). OpenEMR quirk: a patient with zero records here
    returns HTTP 404 with an empty body (not 200 + ``[]``) -- treated as
    count 0, not an error.
  * allergy / problem / encounter counts: REST
    ``.../patient/{uuid}/allergy``, ``.../medical_problem``,
    ``.../encounter`` -- these sub-resources are keyed by the patient
    *uuid*, not pid (an OpenEMR REST API inconsistency). Empty state here
    is 200 + ``{"data": []}``, not a 404.
  * vital / recent_lab counts: FHIR
    ``GET /apis/default/fhir/Observation?patient={uuid}&category=vital-signs``
    (and ``category=laboratory``) -- REST has no patient-level vitals list
    (only nested under a specific encounter), so FHIR is used here. The
    Bundle's ``total`` is read directly; no need to page through ``entry``.
"""

from __future__ import annotations

import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.openemr_client import ErrorCategory, OpenEmrApiError, OpenEmrClient
from app.schemas.common import Sex
from app.schemas.tools import PatientSummaryOutput

_SEX_MAP = {"male": Sex.MALE, "female": Sex.FEMALE, "other": Sex.OTHER}


def get_patient_summary(client: OpenEmrClient, token: str, patient_id: int) -> PatientSummaryOutput:
    demographics = _fetch_demographics(client, token, patient_id)
    patient_uuid = demographics["uuid"]

    # The 7 section counts are independent reads; fan them out concurrently
    # (httpx.Client is thread-safe) instead of paying for 7 sequential round
    # trips -- this tool sits on the pre-visit-brief request's latency path.
    with ThreadPoolExecutor(max_workers=7) as pool:
        medication = pool.submit(_count_rest_list, client, token, f"patient/{patient_id}/medication")
        allergy = pool.submit(_count_rest_list, client, token, f"patient/{patient_uuid}/allergy")
        problem = pool.submit(_count_rest_list, client, token, f"patient/{patient_uuid}/medical_problem")
        recent_lab = pool.submit(_count_fhir_bundle, client, token, patient_uuid, "laboratory")
        vital = pool.submit(_count_fhir_bundle, client, token, patient_uuid, "vital-signs")
        encounter = pool.submit(_count_rest_list, client, token, f"patient/{patient_uuid}/encounter")
        appointment = pool.submit(_count_rest_list, client, token, f"patient/{patient_id}/appointment")

        return PatientSummaryOutput(
            patient_id=patient_id,
            first_name=demographics["fname"],
            last_name=demographics["lname"],
            date_of_birth=datetime.date.fromisoformat(demographics["DOB"]),
            sex=_SEX_MAP.get(str(demographics.get("sex", "")).lower(), Sex.UNKNOWN),
            medication_count=medication.result(),
            allergy_count=allergy.result(),
            problem_count=problem.result(),
            recent_lab_count=recent_lab.result(),
            vital_count=vital.result(),
            encounter_count=encounter.result(),
            appointment_count=appointment.result(),
        )


def get_patient_name(client: OpenEmrClient, token: str, patient_id: int) -> str | None:
    """The patient's own "First Last" display name, or ``None`` if it cannot
    be resolved (patient not found, any OpenEMR API error).

    A single demographics-only round trip via ``_fetch_demographics`` --
    NOT the full ``get_patient_summary``, which additionally fans out 7
    concurrent section-count calls this caller has no use for. Used to
    resolve the bound patient's own display name for the #224 name-binding
    cross-patient guard (``app.extraction.detect_foreign_patient_reference``);
    callers there treat ``None`` as "name-binding unavailable" and fall back
    to numeric-only detection rather than treating this as a hard failure.
    """
    try:
        demographics = _fetch_demographics(client, token, patient_id)
    except OpenEmrApiError:
        return None
    fname, lname = demographics.get("fname"), demographics.get("lname")
    parts = [part for part in (fname, lname) if isinstance(part, str) and part]
    return " ".join(parts) if parts else None


def _fetch_demographics(client: OpenEmrClient, token: str, patient_id: int) -> dict[str, Any]:
    """Fetch the patient roster and select the matching ``pid``.

    A 403/401/timeout/etc. here propagates naturally via ``OpenEmrClient`` --
    the patient itself is not an optional section. No matching record is
    also an error (``NOT_FOUND``): unlike an empty *section*, a missing
    patient is not a valid state for a summary request.
    """
    payload = client.get_rest("patient", token=token)
    records = payload.get("data") if isinstance(payload, dict) else None
    for record in records or []:
        if record.get("pid") == patient_id:
            return record
    raise OpenEmrApiError(ErrorCategory.NOT_FOUND, "OpenEMR patient not found")


def _count_rest_list(client: OpenEmrClient, token: str, path: str) -> int:
    """Count records from a REST sub-resource list endpoint.

    Handles both shapes seen live: a bare JSON list (medication,
    appointment) and ``{"data": [...]}`` (allergy, medical_problem,
    encounter). A 404 (OpenEMR's "no records" signal for the bare-list
    endpoints) is a valid empty state, not an error -- count 0.
    """
    try:
        payload = client.get_rest(path, token=token)
    except OpenEmrApiError as exc:
        if exc.category is ErrorCategory.NOT_FOUND:
            return 0
        raise
    if isinstance(payload, list):
        return len(payload)
    data = payload.get("data") if isinstance(payload, dict) else None
    return len(data) if isinstance(data, list) else 0


def _count_fhir_bundle(client: OpenEmrClient, token: str, patient_uuid: str, category: str) -> int:
    """Count records via a FHIR ``Observation`` search Bundle's ``total``."""
    bundle = client.get_fhir("Observation", token=token, params={"patient": patient_uuid, "category": category})
    total = bundle.get("total") if isinstance(bundle, dict) else None
    return total if isinstance(total, int) else 0
