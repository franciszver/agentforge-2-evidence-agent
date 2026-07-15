"""Hermetic + one live-integration test for the ``get_appointments`` tool.

All hermetic HTTP is served by ``httpx.MockTransport`` so the suite never
touches the network; fixture shapes are modeled on real responses observed
against the dev stack for demo patients Phil Belford (pubpid 1) and Susan
Underwood (pubpid 2) -- see ``app/tools/appointments.py``'s module docstring
for the endpoint quirks those shapes encode. The single
``@pytest.mark.integration`` test hits the real running dev stack and is
skipped by default (minimal CI runs hermetic tests only).
"""

from __future__ import annotations

import datetime
import os

import httpx
import pytest

from app.config import Settings
from app.openemr_auth import fetch_token_password_grant, register_client
from app.openemr_client import ErrorCategory, OpenEmrApiError, OpenEmrClient
from app.schemas.common import AppointmentStatus
from app.tools.appointments import get_appointments

PHIL_UUID = "a243a1bb-178f-4092-8c67-52dfaf67fca6"

PHIL_PATIENT_LIST_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [
        {"pid": 1, "fname": "Phil", "lname": "Belford", "uuid": PHIL_UUID},
    ],
}

# Bare-list shape observed live for GET /apis/default/api/patient/{pid}/appointment
# (pid-keyed, same resource family as medication -- see module docstring).
# Trimmed to the fields the tool reads; a second, later appointment was added
# to the live-observed single-appointment fixture to exercise date-range
# filtering and sorting.
PHIL_APPOINTMENT_BODY = [
    {
        "pc_eid": 10,
        "pc_uuid": "a243a1bb-15e7-4684-8c5d-b1a04af11fe6",
        "pid": 1,
        "pce_aid_fname": "Billy",
        "pce_aid_lname": "Smith",
        "pc_apptstatus": "-",
        "pc_eventDate": "2014-01-31",
        "pc_startTime": "14:30:00",
        "pc_title": "Office Visit",
    },
    {
        "pc_eid": 21,
        "pc_uuid": "a243a1bb-15e7-4684-8c5d-b1a04af11fe7",
        "pid": 1,
        "pce_aid_fname": "Donna",
        "pce_aid_lname": "Lee",
        "pc_apptstatus": "@",
        "pc_eventDate": "2026-08-01",
        "pc_startTime": "09:00:00",
        "pc_title": "Follow-up",
    },
]


def test_happy_path_maps_and_sorts_most_recent_first(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PHIL_PATIENT_LIST_BODY)
        if path == "/apis/default/api/patient/1/appointment":
            return httpx.Response(200, json=PHIL_APPOINTMENT_BODY)
        raise AssertionError(f"unexpected request: {path}")

    result = get_appointments(make_openemr_client(handler), token="tok", patient_id=1)

    assert len(result.items) == 2
    newest, oldest = result.items
    assert newest.date == datetime.date(2026, 8, 1)
    assert newest.time == datetime.time(9, 0, 0)
    assert newest.provider == "Donna Lee"
    assert newest.status == AppointmentStatus.CHECKED_IN

    assert oldest.date == datetime.date(2014, 1, 31)
    assert oldest.time == datetime.time(14, 30, 0)
    assert oldest.provider == "Billy Smith"
    assert oldest.status == AppointmentStatus.SCHEDULED


def test_since_filter_excludes_older_appointment(make_openemr_client):
    """The headline behavior: a start_date after 2014 excludes Phil's 2014
    appointment but keeps the later 2026 follow-up."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PHIL_PATIENT_LIST_BODY)
        if path == "/apis/default/api/patient/1/appointment":
            return httpx.Response(200, json=PHIL_APPOINTMENT_BODY)
        raise AssertionError(f"unexpected request: {path}")

    result = get_appointments(
        make_openemr_client(handler), token="tok", patient_id=1, start_date=datetime.date(2020, 1, 1)
    )

    assert len(result.items) == 1
    assert result.items[0].date == datetime.date(2026, 8, 1)


def test_until_filter_excludes_newer_appointment(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PHIL_PATIENT_LIST_BODY)
        if path == "/apis/default/api/patient/1/appointment":
            return httpx.Response(200, json=PHIL_APPOINTMENT_BODY)
        raise AssertionError(f"unexpected request: {path}")

    result = get_appointments(
        make_openemr_client(handler), token="tok", patient_id=1, end_date=datetime.date(2020, 1, 1)
    )

    assert len(result.items) == 1
    assert result.items[0].date == datetime.date(2014, 1, 31)


def test_date_range_is_inclusive_of_boundary_dates(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PHIL_PATIENT_LIST_BODY)
        if path == "/apis/default/api/patient/1/appointment":
            return httpx.Response(200, json=PHIL_APPOINTMENT_BODY)
        raise AssertionError(f"unexpected request: {path}")

    result = get_appointments(
        make_openemr_client(handler),
        token="tok",
        patient_id=1,
        start_date=datetime.date(2014, 1, 31),
        end_date=datetime.date(2014, 1, 31),
    )

    assert len(result.items) == 1
    assert result.items[0].date == datetime.date(2014, 1, 31)


def test_empty_appointment_list_yields_empty_items_not_an_error(make_openemr_client):
    """OpenEMR quirk (same as medication, P2.4): this pid-keyed sub-resource
    returns HTTP 404 with an empty body for zero records, not 200 + []."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PHIL_PATIENT_LIST_BODY)
        if path == "/apis/default/api/patient/1/appointment":
            return httpx.Response(404, json={"error": "not found"})
        raise AssertionError(f"unexpected request: {path}")

    result = get_appointments(make_openemr_client(handler), token="tok", patient_id=1)

    assert result.items == []


def test_unknown_patient_raises_not_found_not_empty(make_openemr_client):
    """A patient that does not exist is not the same as a known patient with
    no appointments -- the existence check must raise, not return empty."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apis/default/api/patient":
            return httpx.Response(200, json=PHIL_PATIENT_LIST_BODY)
        raise AssertionError(f"unexpected request beyond the existence check: {request.url.path}")

    with pytest.raises(OpenEmrApiError) as excinfo:
        get_appointments(make_openemr_client(handler), token="tok", patient_id=999999)

    assert excinfo.value.category == ErrorCategory.NOT_FOUND


def test_forbidden_on_appointment_list_propagates(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PHIL_PATIENT_LIST_BODY)
        if path == "/apis/default/api/patient/1/appointment":
            return httpx.Response(403, json={"error": "insufficient_scope"})
        raise AssertionError(f"unexpected request: {path}")

    with pytest.raises(OpenEmrApiError) as excinfo:
        get_appointments(make_openemr_client(handler), token="tok", patient_id=1)

    assert excinfo.value.category == ErrorCategory.FORBIDDEN


def test_unrecognized_apptstatus_maps_to_unknown_not_scheduled(make_openemr_client):
    """Fail-loud: an apptstatus code this tool doesn't map (e.g. the
    ambiguous "! Left w/o visit" or "^ Pending" administrative flags) must
    surface as UNKNOWN, never a guessed concrete status."""
    body = [
        {
            "pc_eid": 30,
            "pid": 1,
            "pce_aid_fname": "Billy",
            "pce_aid_lname": "Smith",
            "pc_apptstatus": "^",
            "pc_eventDate": "2024-01-01",
            "pc_startTime": "10:00:00",
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PHIL_PATIENT_LIST_BODY)
        if path == "/apis/default/api/patient/1/appointment":
            return httpx.Response(200, json=body)
        raise AssertionError(f"unexpected request: {path}")

    result = get_appointments(make_openemr_client(handler), token="tok", patient_id=1)

    assert len(result.items) == 1
    assert result.items[0].status == AppointmentStatus.UNKNOWN


@pytest.mark.integration
def test_live_get_appointments_against_dev_stack_demo_patient_phil():
    """Live end-to-end check against the running dev stack (demo patient Phil
    Belford, pubpid 1), who is seeded with a single 2014 appointment.

    Registers a fresh confidential client scoped for the resources this tool
    reads, enables it via the same dev-only path
    ``scripts/verify-oauth-dev.sh`` uses, fetches a user token via the
    password grant, and calls the real tool. Requires the dev stack up;
    skipped by default in minimal CI runs (``pytest -m "not integration"``).
    """
    import subprocess

    base_url = os.environ.get("OPENEMR_BASE_URL", "https://localhost:9300")
    settings = Settings()
    scope = "openid offline_access api:oemr user/patient.read user/appointment.read"

    with httpx.Client(verify=False, timeout=15.0) as setup_client:
        creds = register_client(
            setup_client,
            base_url=base_url,
            registration_path=settings.openemr_oauth_registration_path,
            client_name="copilot-agent-test-appointments",
            redirect_uris=["https://localhost:9300/oauth2/default/callback"],
            scope=scope,
        )

        mysql_container = os.environ.get("MYSQL_CONTAINER", "development-easy-mysql-1")
        subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                mysql_container,
                "sh",
                "-c",
                "$(command -v mariadb || command -v mysql) -uopenemr -popenemr openemr -e "
                f"\"UPDATE oauth_clients SET is_enabled=1 WHERE client_id='{creds.client_id}';\"",
            ],
            check=True,
            capture_output=True,
        )

        token = fetch_token_password_grant(
            setup_client,
            base_url=base_url,
            token_path=settings.openemr_oauth_token_path,
            client_id=creds.client_id,
            client_secret=creds.client_secret,
            username=os.environ.get("OPENEMR_DEV_USER", "admin"),
            password=os.environ.get("OPENEMR_DEV_PASS", "pass"),
            scope=scope,
        )

    client = OpenEmrClient.from_settings(Settings(openemr_base_url=base_url))
    result = get_appointments(client, token=token.access_token, patient_id=1)

    assert isinstance(result.items, list)
    assert len(result.items) >= 1
