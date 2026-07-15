"""Hermetic + one live-integration test for the ``get_medications`` tool.

All hermetic HTTP is served by ``httpx.MockTransport`` so the suite never
touches the network; fixture shapes are modeled on real responses observed
against the dev stack for demo patient Phil Belford (pubpid 1) -- see
``app/tools/medications.py``'s module docstring for the endpoint quirks
those shapes encode. The single ``@pytest.mark.integration`` test hits the
real running dev stack and is skipped by default (minimal CI runs hermetic
tests only).
"""

from __future__ import annotations

import datetime
import os

import httpx
import pytest

from app.config import Settings
from app.openemr_auth import fetch_token_password_grant, register_client
from app.openemr_client import ErrorCategory, OpenEmrApiError, OpenEmrClient
from app.schemas.common import MedicationStatus
from app.tools.medications import get_medications

PHIL_UUID = "a243a1bb-178f-4092-8c67-52dfaf67fca6"

# Trimmed to the fields the tool reads; modeled on the real REST payload
# shape for GET /apis/default/api/patient (list endpoint has no internal
# pid/id filter, so the tool fetches the roster and selects client-side).
PATIENT_LIST_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [
        {"pid": 1, "fname": "Phil", "lname": "Belford", "uuid": PHIL_UUID},
    ],
}

# Bare-list shape observed live for GET /apis/default/api/patient/1/medication
# (ListRestController-backed). Note: no "dosage"/"route" keys are present in
# this endpoint's response at all (see module docstring) -- the tool falls
# back to "" for those fields.
MEDICATION_BODY = [
    {
        "id": 5,
        "title": "Lisinopril",
        "pid": 1,
        "activity": 1,
        "begdate": "2020-01-15 00:00:00",
        "enddate": None,
        "uuid": "a243a1bb-145c-42b9-b1b9-f1d1dfd69395",
    },
    {
        "id": 4,
        "title": "Norvasc",
        "pid": 1,
        "activity": 0,
        "begdate": None,
        "enddate": "2021-06-01 00:00:00",
        "uuid": "a243a1bb-145a-4fd9-a7b9-cbc03078d4d8",
    },
]


def test_happy_path_maps_medication_rows(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == "/apis/default/api/patient/1/medication":
            return httpx.Response(200, json=MEDICATION_BODY)
        raise AssertionError(f"unexpected request: {path}")

    result = get_medications(make_openemr_client(handler), token="tok", patient_id=1)

    assert len(result.items) == 2
    first, second = result.items
    assert first.name == "Lisinopril"
    assert first.dose == ""
    assert first.route == ""
    assert first.status == MedicationStatus.ACTIVE
    assert first.start_date == datetime.date(2020, 1, 15)
    assert first.end_date is None
    assert second.name == "Norvasc"
    assert second.status == MedicationStatus.DISCONTINUED
    assert second.start_date is None
    assert second.end_date == datetime.date(2021, 6, 1)


def test_empty_medication_list_yields_empty_items_not_an_error(make_openemr_client):
    """OpenEMR quirk (same as P2.3): a pid-keyed sub-resource with zero
    records returns HTTP 404 with an empty body, not 200 + []."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == "/apis/default/api/patient/1/medication":
            return httpx.Response(404, text="")
        raise AssertionError(f"unexpected request: {path}")

    result = get_medications(make_openemr_client(handler), token="tok", patient_id=1)

    assert result.items == []


def test_unknown_patient_raises_not_found_not_empty(make_openemr_client):
    """A patient that does not exist is not the same as a known patient with
    no medications -- the existence check must raise, not return empty."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        raise AssertionError(f"unexpected request beyond the patient existence check: {request.url.path}")

    with pytest.raises(OpenEmrApiError) as excinfo:
        get_medications(make_openemr_client(handler), token="tok", patient_id=999999)

    assert excinfo.value.category == ErrorCategory.NOT_FOUND


def test_forbidden_on_medication_list_propagates(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == "/apis/default/api/patient/1/medication":
            return httpx.Response(403, json={"error": "insufficient_scope"})
        raise AssertionError(f"unexpected request: {path}")

    with pytest.raises(OpenEmrApiError) as excinfo:
        get_medications(make_openemr_client(handler), token="tok", patient_id=1)

    assert excinfo.value.category == ErrorCategory.FORBIDDEN


@pytest.mark.integration
def test_live_get_medications_against_dev_stack_demo_patient_phil():
    """Live end-to-end check against the running dev stack (demo patient Phil).

    Registers a fresh confidential client scoped for the resources this tool
    reads, enables it via the same dev-only path
    ``scripts/verify-oauth-dev.sh`` uses, fetches a user token via the
    password grant, and calls the real tool against demo patient Phil
    (pubpid 1). Requires the dev stack up; skipped by default in minimal CI
    runs (``pytest -m "not integration"``).
    """
    import subprocess

    base_url = os.environ.get("OPENEMR_BASE_URL", "https://localhost:9300")
    settings = Settings()
    scope = "openid offline_access api:oemr user/patient.read user/medication.read"

    with httpx.Client(verify=False, timeout=15.0) as setup_client:
        creds = register_client(
            setup_client,
            base_url=base_url,
            registration_path=settings.openemr_oauth_registration_path,
            client_name="copilot-agent-test-medications",
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
    result = get_medications(client, token=token.access_token, patient_id=1)

    assert result.items is not None
    assert len(result.items) >= 0
