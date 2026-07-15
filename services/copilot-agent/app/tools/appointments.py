"""``get_appointments`` tool (UC1/UC4 backbone).

Endpoint, established by probing the live dev API (demo patients Phil
Belford pubpid 1, Susan Underwood pubpid 2, Wanda Moore pubpid 3): REST
``GET /apis/default/api/patient/{pid}/appointment`` -- pid-keyed (same
``ListRestController``-backed resource family P2.3 documented for
medication/appointment; see ``app/tools/patient_summary.py``'s module
docstring), returning a bare JSON list. A patient with zero appointments
returns HTTP 404 with an empty body, not 200 + ``[]`` -- treated as an empty
list, not an error (same quirk P2.4's ``get_medications`` handled for its
own pid-keyed endpoint).

Before that call, the patient's existence is checked via
``app.tools._common.resolve_patient_uuid`` (its uuid return value is unused
here -- the appointment endpoint takes the pid directly). This extra round
trip exists solely so "unknown patient" and "known patient, zero
appointments" stay distinguishable: both would otherwise produce the exact
same 404-empty response from the pid-keyed endpoint.

OpenEMR quirk (provider name): unlike the encounter endpoint, this resource
*does* expose the provider's name directly as ``pce_aid_fname``/
``pce_aid_lname`` (joined from the appointment's assigned provider, not the
encounter's) -- no separate lookup needed. ``provider`` joins the two,
``None`` if both are blank.

OpenEMR quirk (status vocabulary): ``pc_apptstatus`` is a single-character
(or short-code) status drawn from the ``apptstat`` list (verified live via
``list_options``: ``-``/``!``/``?``/``@``/``*``/``#``/``%``/``^``/``+``/``<``/
``>``/``~``/``$``/``AVM``/``CALL``/``EMAIL``/``SMS``/``x``). Only the subset
with an unambiguous ``AppointmentStatus`` correspondent is mapped in
``_APPOINTMENT_STATUS_MAP`` (``-`` -> ``SCHEDULED``, arrived/in-room codes ->
``CHECKED_IN``, ``>`` checked-out -> ``COMPLETED``, ``?`` -> ``NO_SHOW``,
cancellation codes -> ``CANCELLED``, confirmation channels -> ``CONFIRMED``);
administrative/ambiguous codes (e.g. ``!`` left without visit, ``^`` pending,
``#`` insurance issue, ``+`` chart pulled, ``$`` coding done, ``CALL``) and
any missing/unrecognized code map to the ``AppointmentStatus.UNKNOWN`` member
added here (schema had no UNKNOWN member previously -- P2.5 precedent: fail
loud over guessing a concrete status for a code with no clean mapping).

``start_date``/``end_date`` filtering and the most-recent-first sort are
applied client-side after fetching the full appointment list, the same
"fine at demo scale" tradeoff P2.5's FHIR-backed tools made. The date range
is inclusive of both boundary dates. Unlike ``get_encounters``, this tool's
schema (``GetAppointmentsInput``) has no ``limit`` field, so none is applied.
"""

from __future__ import annotations

import datetime
from typing import Any

from app.openemr_client import ErrorCategory, OpenEmrApiError, OpenEmrClient
from app.schemas.common import AppointmentStatus
from app.schemas.tools import AppointmentItem, AppointmentsOutput
from app.tools._common import resolve_patient_uuid

_APPOINTMENT_STATUS_MAP = {
    "-": AppointmentStatus.SCHEDULED,
    "@": AppointmentStatus.CHECKED_IN,
    "~": AppointmentStatus.CHECKED_IN,
    "<": AppointmentStatus.CHECKED_IN,
    ">": AppointmentStatus.COMPLETED,
    "?": AppointmentStatus.NO_SHOW,
    "%": AppointmentStatus.CANCELLED,
    "x": AppointmentStatus.CANCELLED,
    "AVM": AppointmentStatus.CONFIRMED,
    "EMAIL": AppointmentStatus.CONFIRMED,
    "SMS": AppointmentStatus.CONFIRMED,
}


def get_appointments(
    client: OpenEmrClient,
    token: str,
    patient_id: int,
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
) -> AppointmentsOutput:
    # Existence check only -- raises NOT_FOUND for an unknown patient rather
    # than letting it look identical to a known patient with zero appointments.
    resolve_patient_uuid(client, token, patient_id)

    try:
        payload = client.get_rest(f"patient/{patient_id}/appointment", token=token)
    except OpenEmrApiError as exc:
        if exc.category is ErrorCategory.NOT_FOUND:
            return AppointmentsOutput(items=[])
        raise

    records = payload if isinstance(payload, list) else (payload.get("data") if isinstance(payload, dict) else None)
    items = [item for item in (_map_appointment(record) for record in records or [] if isinstance(record, dict)) if item is not None]

    if start_date is not None:
        items = [item for item in items if item.date >= start_date]
    if end_date is not None:
        items = [item for item in items if item.date <= end_date]
    items.sort(key=lambda item: (item.date, item.time), reverse=True)

    return AppointmentsOutput(items=items)


def _map_appointment(record: dict[str, Any]) -> AppointmentItem | None:
    date = _parse_date(record.get("pc_eventDate"))
    time = _parse_time(record.get("pc_startTime"))
    if date is None or time is None:
        return None

    return AppointmentItem(
        date=date,
        time=time,
        status=_map_status(record.get("pc_apptstatus")),
        provider=_provider_name(record.get("pce_aid_fname"), record.get("pce_aid_lname")),
    )


def _provider_name(fname: Any, lname: Any) -> str | None:
    parts = [part for part in (fname, lname) if isinstance(part, str) and part]
    name = " ".join(parts)
    return name or None


def _map_status(apptstatus: Any) -> AppointmentStatus:
    if isinstance(apptstatus, str) and apptstatus in _APPOINTMENT_STATUS_MAP:
        return _APPOINTMENT_STATUS_MAP[apptstatus]
    # Missing, unrecognized, or administrative/ambiguous code (e.g. "!" left
    # without visit, "^" pending) -> fail loud with UNKNOWN rather than guess
    # a concrete status.
    return AppointmentStatus.UNKNOWN


def _parse_date(value: Any) -> datetime.date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.date.fromisoformat(value[:10])
    except ValueError:
        return None


def _parse_time(value: Any) -> datetime.time | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.time.fromisoformat(value)
    except ValueError:
        return None
