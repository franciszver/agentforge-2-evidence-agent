"""Hermetic + one live-integration test for the ``get_problems`` tool.

All hermetic HTTP is served by ``httpx.MockTransport`` so the suite never
touches the network; fixture shapes are modeled on real responses observed
against the dev stack for demo patients Phil Belford (pubpid 1) and Susan
Underwood (pubpid 2) -- see ``app/tools/problems.py``'s module docstring for
the endpoint quirks those shapes encode. The single
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
from app.schemas.common import ProblemStatus
from app.tools.problems import get_problems

PHIL_UUID = "a243a1bb-178f-4092-8c67-52dfaf67fca6"

# Trimmed to the fields the tool reads; modeled on the real REST payload
# shape for GET /apis/default/api/patient.
PATIENT_LIST_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [
        {"pid": 1, "fname": "Phil", "lname": "Belford", "uuid": PHIL_UUID},
    ],
}

# {"data": [...]} shape observed live for GET
# /apis/default/api/patient/{uuid}/medical_problem (UUID-keyed sub-resource,
# same family as allergy -- see module docstring). "diagnosis" is a dict
# keyed by ICD code when coded, or "" when not coded (observed live for
# Susan Underwood's "diabetes" problem).
PROBLEM_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [
        {
            "id": 1,
            "title": "HTN",
            "pid": 1,
            "activity": 1,
            "begdate": None,
            "enddate": None,
            "diagnosis": {"401.0": {"code": "401.0", "description": "", "code_type": "ICD9", "system": None}},
        },
        {
            "id": 2,
            "title": "Chronic Renal Insuficiency",
            "pid": 1,
            "activity": 0,
            "begdate": "2018-03-01 00:00:00",
            "enddate": "2020-06-01 00:00:00",
            "diagnosis": "",
        },
        {
            "id": 3,
            "title": "Seasonal allergy watch",
            "pid": 1,
            "activity": 0,
            "begdate": None,
            "enddate": None,
            "diagnosis": "",
        },
    ],
}


def test_happy_path_maps_problem_rows(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == f"/apis/default/api/patient/{PHIL_UUID}/medical_problem":
            return httpx.Response(200, json=PROBLEM_BODY)
        raise AssertionError(f"unexpected request: {path}")

    result = get_problems(make_openemr_client(handler), token="tok", patient_id=1)

    assert len(result.items) == 3
    active, resolved, inactive = result.items
    assert active.title == "HTN"
    assert active.icd_code == "401.0"
    assert active.status == ProblemStatus.ACTIVE
    assert active.onset_date is None

    assert resolved.title == "Chronic Renal Insuficiency"
    assert resolved.icd_code is None
    assert resolved.status == ProblemStatus.RESOLVED
    assert resolved.onset_date == datetime.date(2018, 3, 1)

    assert inactive.title == "Seasonal allergy watch"
    assert inactive.icd_code is None
    assert inactive.status == ProblemStatus.INACTIVE


def test_unrecognized_activity_maps_to_unknown_not_inactive(make_openemr_client):
    """Clinical safety: an ``activity`` value outside {0, 1} (here missing)
    must surface as UNKNOWN, never INACTIVE -- claiming a problem is inactive
    when its state is genuinely unknown could hide a genuinely active one."""
    body = {
        "validationErrors": [],
        "internalErrors": [],
        "data": [
            {"id": 9, "title": "Undetermined problem", "pid": 1, "begdate": None, "enddate": None, "diagnosis": ""},
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == f"/apis/default/api/patient/{PHIL_UUID}/medical_problem":
            return httpx.Response(200, json=body)
        raise AssertionError(f"unexpected request: {path}")

    result = get_problems(make_openemr_client(handler), token="tok", patient_id=1)

    assert len(result.items) == 1
    assert result.items[0].status == ProblemStatus.UNKNOWN


def test_empty_problem_list_yields_empty_items_not_an_error(make_openemr_client):
    """OpenEMR quirk (same as allergy, P2.4): this uuid-keyed sub-resource
    returns 200 + {"data": []} for zero records, not a 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == f"/apis/default/api/patient/{PHIL_UUID}/medical_problem":
            return httpx.Response(200, json={"validationErrors": [], "internalErrors": [], "data": []})
        raise AssertionError(f"unexpected request: {path}")

    result = get_problems(make_openemr_client(handler), token="tok", patient_id=1)

    assert result.items == []


def test_unknown_patient_raises_not_found_not_empty(make_openemr_client):
    """A patient that does not exist is not the same as a known patient with
    no problems -- the uuid lookup must raise, not return empty."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        raise AssertionError(f"unexpected request beyond the uuid lookup: {request.url.path}")

    with pytest.raises(OpenEmrApiError) as excinfo:
        get_problems(make_openemr_client(handler), token="tok", patient_id=999999)

    assert excinfo.value.category == ErrorCategory.NOT_FOUND


def test_forbidden_on_problem_list_propagates(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == f"/apis/default/api/patient/{PHIL_UUID}/medical_problem":
            return httpx.Response(403, json={"error": "insufficient_scope"})
        raise AssertionError(f"unexpected request: {path}")

    with pytest.raises(OpenEmrApiError) as excinfo:
        get_problems(make_openemr_client(handler), token="tok", patient_id=1)

    assert excinfo.value.category == ErrorCategory.FORBIDDEN


@pytest.mark.integration
def test_live_get_problems_against_dev_stack_demo_patient_phil():
    """Live end-to-end check against the running dev stack (demo patient Phil).

    Registers a fresh confidential client scoped for the resources this tool
    reads, enables it via the same dev-only path
    ``scripts/verify-oauth-dev.sh`` uses, fetches a user token via the
    password grant, and calls the real tool against demo patient Phil
    (pubpid 1), who is seeded with >=1 medical problem (HTN). Requires the
    dev stack up; skipped by default in minimal CI runs
    (``pytest -m "not integration"``).
    """
    import subprocess

    base_url = os.environ.get("OPENEMR_BASE_URL", "https://localhost:9300")
    settings = Settings()
    scope = "openid offline_access api:oemr user/patient.read user/medical_problem.read"

    with httpx.Client(verify=False, timeout=15.0) as setup_client:
        creds = register_client(
            setup_client,
            base_url=base_url,
            registration_path=settings.openemr_oauth_registration_path,
            client_name="copilot-agent-test-problems",
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
    result = get_problems(client, token=token.access_token, patient_id=1)

    assert len(result.items) >= 1
    assert any("htn" in item.title.lower() for item in result.items)
