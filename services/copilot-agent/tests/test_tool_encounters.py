"""Hermetic + one live-integration test for the ``get_encounters`` tool.

All hermetic HTTP is served by ``httpx.MockTransport`` so the suite never
touches the network; fixture shapes are modeled on real responses observed
against the dev stack for demo patients Phil Belford (pubpid 1) and Susan
Underwood (pubpid 2, the seeded multi-encounter fixture -- see
``docs/TEST_PLAN.md`` §7) -- see ``app/tools/encounters.py``'s module
docstring for the endpoint quirks those shapes encode. The single
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
from app.schemas.common import EncounterType
from app.tools.encounters import get_encounters

PHIL_UUID = "a243a1bb-178f-4092-8c67-52dfaf67fca6"
SUSAN_UUID = "a243a1bb-1793-4fb0-9c00-4e42dfcb57fe"

PHIL_PATIENT_LIST_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [
        {"pid": 1, "fname": "Phil", "lname": "Belford", "uuid": PHIL_UUID},
    ],
}

SUSAN_PATIENT_LIST_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [
        {"pid": 2, "fname": "Susan", "lname": "Underwood", "uuid": SUSAN_UUID},
    ],
}

# {"data": [...]} shape observed live for GET
# /apis/default/api/patient/{uuid}/encounter (UUID-keyed sub-resource, same
# family as allergy/medical_problem -- see module docstring). Trimmed to the
# fields the tool reads. Susan (pubpid 2) carries the seeded second, more
# recent encounter (docs/TEST_PLAN.md §7's multi-encounter fixture).
SUSAN_ENCOUNTER_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [
        {
            "eid": 13,
            "euuid": "a243a1bb-0e95-41bf-b01e-ad60064f983e",
            "date": "2026-05-31 03:29:18",
            "reason": "Follow-up: toe re-check (seed.py fixture)",
            "class_code": "AMB",
            "class_title": "ambulatory",
            "pc_catname": "Office Visit",
            "pid": 2,
            "provider_id": 0,
            "provider_uuid": None,
            "provider_username": None,
        },
        {
            "eid": 8,
            "euuid": "a243a1bb-0e8d-4bed-9c93-06478c348485",
            "date": "2014-02-01 00:00:00",
            "reason": "toe pain",
            "class_code": "AMB",
            "class_title": "ambulatory",
            "pc_catname": "New Patient",
            "pid": 2,
            "provider_id": 1,
            "provider_uuid": None,
            "provider_username": "admin",
        },
    ],
}

PHIL_ENCOUNTER_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [
        {
            "eid": 5,
            "euuid": "a243a1bb-0e87-4498-b86d-dc3a17831e9f",
            "date": "2014-02-01 00:00:00",
            "reason": "Sad",
            "class_code": "AMB",
            "class_title": "ambulatory",
            "pc_catname": "Established Patient",
            "pid": 1,
            "provider_id": 1,
            "provider_uuid": None,
            "provider_username": None,
        },
    ],
}


def test_happy_path_maps_and_sorts_most_recent_first(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=SUSAN_PATIENT_LIST_BODY)
        if path == f"/apis/default/api/patient/{SUSAN_UUID}/encounter":
            return httpx.Response(200, json=SUSAN_ENCOUNTER_BODY)
        raise AssertionError(f"unexpected request: {path}")

    result = get_encounters(make_openemr_client(handler), token="tok", patient_id=2)

    assert len(result.items) == 2
    newest, oldest = result.items
    assert newest.encounter_id == 13
    assert newest.date == datetime.datetime(2026, 5, 31, 3, 29, 18)
    assert newest.reason == "Follow-up: toe re-check (seed.py fixture)"
    assert newest.provider is None
    assert newest.encounter_type == EncounterType.OFFICE_VISIT

    assert oldest.encounter_id == 8
    assert oldest.date == datetime.datetime(2014, 2, 1, 0, 0, 0)
    assert oldest.reason == "toe pain"
    assert oldest.provider == "admin"
    assert oldest.encounter_type == EncounterType.OFFICE_VISIT


def test_since_filter_excludes_older_encounter(make_openemr_client):
    """The headline behavior: a start_date after 2014 excludes Susan's stale
    2014 encounter but keeps the seeded 2026 follow-up."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=SUSAN_PATIENT_LIST_BODY)
        if path == f"/apis/default/api/patient/{SUSAN_UUID}/encounter":
            return httpx.Response(200, json=SUSAN_ENCOUNTER_BODY)
        raise AssertionError(f"unexpected request: {path}")

    result = get_encounters(
        make_openemr_client(handler), token="tok", patient_id=2, start_date=datetime.date(2020, 1, 1)
    )

    assert len(result.items) == 1
    assert result.items[0].encounter_id == 13


def test_until_filter_excludes_newer_encounter(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=SUSAN_PATIENT_LIST_BODY)
        if path == f"/apis/default/api/patient/{SUSAN_UUID}/encounter":
            return httpx.Response(200, json=SUSAN_ENCOUNTER_BODY)
        raise AssertionError(f"unexpected request: {path}")

    result = get_encounters(
        make_openemr_client(handler), token="tok", patient_id=2, end_date=datetime.date(2020, 1, 1)
    )

    assert len(result.items) == 1
    assert result.items[0].encounter_id == 8


def test_date_range_is_inclusive_of_boundary_dates(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=SUSAN_PATIENT_LIST_BODY)
        if path == f"/apis/default/api/patient/{SUSAN_UUID}/encounter":
            return httpx.Response(200, json=SUSAN_ENCOUNTER_BODY)
        raise AssertionError(f"unexpected request: {path}")

    result = get_encounters(
        make_openemr_client(handler),
        token="tok",
        patient_id=2,
        start_date=datetime.date(2014, 2, 1),
        end_date=datetime.date(2014, 2, 1),
    )

    assert len(result.items) == 1
    assert result.items[0].encounter_id == 8


def test_limit_keeps_most_recent_n(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=SUSAN_PATIENT_LIST_BODY)
        if path == f"/apis/default/api/patient/{SUSAN_UUID}/encounter":
            return httpx.Response(200, json=SUSAN_ENCOUNTER_BODY)
        raise AssertionError(f"unexpected request: {path}")

    result = get_encounters(make_openemr_client(handler), token="tok", patient_id=2, limit=1)

    assert len(result.items) == 1
    assert result.items[0].encounter_id == 13


def test_empty_encounter_list_yields_empty_items_not_an_error(make_openemr_client):
    """OpenEMR quirk (same as allergy/problem, P2.4/P2.5): this uuid-keyed
    sub-resource returns 200 + {"data": []} for zero records, not a 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PHIL_PATIENT_LIST_BODY)
        if path == f"/apis/default/api/patient/{PHIL_UUID}/encounter":
            return httpx.Response(200, json={"validationErrors": [], "internalErrors": [], "data": []})
        raise AssertionError(f"unexpected request: {path}")

    result = get_encounters(make_openemr_client(handler), token="tok", patient_id=1)

    assert result.items == []


def test_unknown_patient_raises_not_found_not_empty(make_openemr_client):
    """A patient that does not exist is not the same as a known patient with
    no encounters -- the uuid lookup must raise, not return empty."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apis/default/api/patient":
            return httpx.Response(200, json=PHIL_PATIENT_LIST_BODY)
        raise AssertionError(f"unexpected request beyond the uuid lookup: {request.url.path}")

    with pytest.raises(OpenEmrApiError) as excinfo:
        get_encounters(make_openemr_client(handler), token="tok", patient_id=999999)

    assert excinfo.value.category == ErrorCategory.NOT_FOUND


def test_forbidden_on_encounter_list_propagates(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PHIL_PATIENT_LIST_BODY)
        if path == f"/apis/default/api/patient/{PHIL_UUID}/encounter":
            return httpx.Response(403, json={"error": "insufficient_scope"})
        raise AssertionError(f"unexpected request: {path}")

    with pytest.raises(OpenEmrApiError) as excinfo:
        get_encounters(make_openemr_client(handler), token="tok", patient_id=1)

    assert excinfo.value.category == ErrorCategory.FORBIDDEN


def test_unrecognized_class_code_maps_to_unknown_encounter_type(make_openemr_client):
    """Clinical-safety-adjacent fail-loud: a ``class_code`` this tool doesn't
    recognize must surface as UNKNOWN, never a guessed concrete type."""
    body = {
        "validationErrors": [],
        "internalErrors": [],
        "data": [
            {
                "eid": 99,
                "euuid": "a243a1bb-0000-0000-0000-000000000099",
                "date": "2024-01-01 00:00:00",
                "reason": "Mystery visit",
                "class_code": "SOME_UNMAPPED_CODE",
                "pid": 1,
                "provider_username": None,
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PHIL_PATIENT_LIST_BODY)
        if path == f"/apis/default/api/patient/{PHIL_UUID}/encounter":
            return httpx.Response(200, json=body)
        raise AssertionError(f"unexpected request: {path}")

    result = get_encounters(make_openemr_client(handler), token="tok", patient_id=1)

    assert len(result.items) == 1
    assert result.items[0].encounter_type == EncounterType.UNKNOWN


@pytest.mark.integration
def test_live_get_encounters_against_dev_stack_multi_encounter_patient_susan():
    """Live end-to-end check against the running dev stack (demo patient
    Susan Underwood, pubpid 2 -- the seeded multi-encounter fixture, see
    ``docs/TEST_PLAN.md`` §7). Also proves the ``start_date`` filter narrows
    her two encounters down to the one seeded in 2026.

    Registers a fresh confidential client scoped for the resources this tool
    reads, enables it via the same dev-only path
    ``scripts/verify-oauth-dev.sh`` uses, fetches a user token via the
    password grant, and calls the real tool. Requires the dev stack up;
    skipped by default in minimal CI runs (``pytest -m "not integration"``).
    """
    import subprocess

    base_url = os.environ.get("OPENEMR_BASE_URL", "https://localhost:9300")
    settings = Settings()
    scope = "openid offline_access api:oemr user/patient.read user/encounter.read"

    with httpx.Client(verify=False, timeout=15.0) as setup_client:
        creds = register_client(
            setup_client,
            base_url=base_url,
            registration_path=settings.openemr_oauth_registration_path,
            client_name="copilot-agent-test-encounters",
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

    full_result = get_encounters(client, token=token.access_token, patient_id=2)
    assert len(full_result.items) >= 2

    filtered_result = get_encounters(
        client, token=token.access_token, patient_id=2, start_date=datetime.date(2020, 1, 1)
    )
    assert len(filtered_result.items) < len(full_result.items)
