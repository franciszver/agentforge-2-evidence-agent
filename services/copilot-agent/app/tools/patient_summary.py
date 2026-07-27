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
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple

from app.openemr_client import ErrorCategory, OpenEmrApiError, OpenEmrClient
from app.schemas.common import Sex
from app.schemas.tools import PatientSummaryOutput

_logger = logging.getLogger(__name__)

_SEX_MAP = {"male": Sex.MALE, "female": Sex.FEMALE, "other": Sex.OTHER}


class RosterEntry(NamedTuple):
    """One patient's (pid, "First Last" display name) pair, as returned by
    ``get_patient_roster`` (Phase 1 #237, made patient-agnostic by #174).

    A plain ``NamedTuple``, not a ``ToolSchemaModel`` -- this never crosses
    the tool-output JSON boundary (``app.schemas.tools``); it is purely an
    internal shape shared between ``get_patient_roster``,
    ``app.planner.Planner.resolve_patient_roster``, ``app.chat.RosterCache``,
    and ``app.extraction._matches_roster``. Carrying ``pid`` alongside
    ``name`` is what lets exclusion of the CALLER's bound patient happen at
    comparison time (see ``_matches_roster``) instead of at fetch time --
    the fetch itself no longer knows, or needs to know, which patient is
    asking.
    """

    pid: int
    name: str


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
    resolve the bound patient's own display name for the Phase 1 #224 name-binding
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


def _fetch_all_patients(client: OpenEmrClient, token: str) -> list[dict[str, Any]]:
    """Fetch the full REST patient roster (``GET /apis/default/api/patient``),
    unfiltered. Shared by ``_fetch_demographics`` (select one pid) and
    ``get_patient_roster`` (Phase 1 #237 -- collect every OTHER patient's name)."""
    payload = client.get_rest("patient", token=token)
    records = payload.get("data") if isinstance(payload, dict) else None
    return [record for record in records or [] if isinstance(record, dict)]


def _fetch_demographics(client: OpenEmrClient, token: str, patient_id: int) -> dict[str, Any]:
    """Fetch the patient roster and select the matching ``pid``.

    A 403/401/timeout/etc. here propagates naturally via ``OpenEmrClient`` --
    the patient itself is not an optional section. No matching record is
    also an error (``NOT_FOUND``): unlike an empty *section*, a missing
    patient is not a valid state for a summary request.
    """
    for record in _fetch_all_patients(client, token):
        if record.get("pid") == patient_id:
            return record
    raise OpenEmrApiError(ErrorCategory.NOT_FOUND, "OpenEMR patient not found")


def get_patient_roster(client: OpenEmrClient, token: str) -> list[RosterEntry]:
    """Every patient's (pid, "First Last" display name) pair (Phase 1 #237
    roster-based cross-patient detection).

    Issue #174: deliberately does NOT take a ``patient_id`` to exclude, and
    no longer excludes anyone. Excluding the CALLER's bound patient is the
    caller's job now, at COMPARISON time, keyed by ``pid`` (see
    ``app.extraction._matches_roster``) rather than by name -- a name
    comparison could wrongly exclude a different patient who happens to
    share the bound patient's name, and pid is already known to every
    caller with no extra fetch.

    This fetch runs AS ``token``'s own bearer, so it has exactly the
    per-caller state that implies -- NOT none. Whether the result is
    actually caller-invariant depends entirely on the auth mode; see
    ``app.chat.RosterCache`` for the full analysis. With
    ``copilot_per_user_token_enabled`` OFF, every caller authenticates as
    the same dev-bridge demo-clinician identity, so the roster genuinely
    IS byte-identical across every conversation, which is what makes it
    safe to serve from ONE process-wide, TTL'd, unkeyed cache entry
    (``app.chat.RosterCache``) shared by every conversation instead of
    being resolved -- and retained -- separately per conversation. With
    the flag ON, each caller authenticates as their OWN OpenEMR user
    account, and OpenEMR's role- and resource-scoped REST authorization
    can return a genuinely DIFFERENT roster to a different principal from
    the identical endpoint -- which is exactly why #182 keys
    ``RosterCache`` by principal (the introspected ``sub``) rather than
    serving one shared entry in that mode. Callers of this function do not
    need to pick which mode applies -- ``RosterCache.get_or_fetch`` makes
    that decision -- but this fetch itself must never be described as
    having no per-caller state; it has exactly as much as OpenEMR's own
    authorization model gives ``token``.

    Fail-safe: ``[]`` on any OpenEMR API error (timeout, insufficient scope,
    ...), never a raised exception -- callers
    (``app.extraction.detect_foreign_patient_reference``) treat an empty/
    unavailable roster as "roster signal unavailable" and skip it entirely,
    the same posture ``get_patient_name`` already takes for the name-binding
    signal. Reuses the same full-roster fetch as ``_fetch_demographics``
    (``GET /apis/default/api/patient``) -- this tool is only ever invoked
    lazily, when a candidate "switch to <Name>" construction has already
    matched, so it is not an extra round trip on the common (no such
    construction) path.
    """
    try:
        records = _fetch_all_patients(client, token)
    except OpenEmrApiError:
        return []
    entries: list[RosterEntry] = []
    for record in records:
        fname, lname = record.get("fname"), record.get("lname")
        parts = [part for part in (fname, lname) if isinstance(part, str) and part]
        pid = _coerce_pid(record.get("pid"))
        if pid is None:
            # Gate 2 (Opus) re-review MINOR: unlike ``_fetch_demographics``/
            # ``resolve_patient_uuid`` (which compare ``pid`` with ``==`` and
            # never gate inclusion on its TYPE), this function decides
            # whether to keep a record at all based on whether ``pid``
            # resolves to an int. Pre-#174 a non-int-but-numeric ``pid``
            # still produced a name (comparison-based exclusion just
            # silently no-opped); dropping the record here instead would
            # make signal 3 go silently empty on such a payload -- a WORSE
            # failure direction. Logged (no PHI -- no name, no raw pid
            # value) so a real occurrence is visible rather than a silent
            # roster gap.
            if parts:
                _logger.warning(
                    "dropped a patient roster record with a non-integer pid",
                    extra={"pid_type": type(record.get("pid")).__name__},
                )
            continue
        if parts:
            entries.append(RosterEntry(pid=pid, name=" ".join(parts)))
    return entries


def _coerce_pid(raw: Any) -> int | None:
    """Best-effort int coercion for a REST patient record's ``pid`` field.

    ``pid`` is normally already a JSON int, but a MySQL-backed PHP REST
    layer could plausibly serialize it as a numeric string --
    ``str.isdecimal()`` (no sign, no decimal point: a real OpenEMR ``pid``
    is a positive auto-increment integer, never negative or fractional).

    Gate 3 (Opus) re-review, CONFIRMED subtlety: this is ``isdecimal()``,
    deliberately NOT ``isdigit()``. ``isdigit()`` also returns ``True`` for
    Unicode category No characters -- superscripts ("²") and circled
    digits ("①") -- that ``int()`` then REJECTS with ``ValueError``,
    which would escape this function uncaught (``get_patient_roster``'s own
    ``try`` wraps only ``_fetch_all_patients``, not this coercion),
    propagating all the way up through ``resolve_patient_roster`` /
    ``_roster_provider`` into the ``_stream_chat`` pre-dispatch guard --
    directly contradicting this function's own "never raises" contract.
    ``isdecimal()`` is ``True`` only for Unicode category Nd (what ``int()``
    actually accepts), so it can never trigger that path. Not a realistic
    OpenEMR payload (robustness, not exploitability) -- but the predicate
    must be correct regardless of how likely the input is.

    ``None`` for anything else (missing, non-numeric, float, ...) -- the
    caller logs and drops that record rather than guessing further.
    """
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.isdecimal():
        return int(raw)
    return None


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
