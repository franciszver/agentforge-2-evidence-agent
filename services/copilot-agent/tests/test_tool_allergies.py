"""Hermetic + one live-integration test for the ``get_allergies`` tool.

All hermetic HTTP is served by ``httpx.MockTransport`` so the suite never
touches the network; fixture shapes are modeled on real responses observed
against the dev stack for demo patient Phil Belford (pubpid 1) -- see
``app/tools/allergies.py``'s module docstring for the endpoint quirks those
shapes encode. The single ``@pytest.mark.integration`` test hits the real
running dev stack and is skipped by default (minimal CI runs hermetic tests
only).
"""

from __future__ import annotations

import os

import httpx
import pytest

from app.config import Settings
from app.openemr_auth import fetch_token_password_grant, register_client
from app.openemr_client import ErrorCategory, OpenEmrApiError, OpenEmrClient
from app.schemas.common import AllergySeverity
from app.tools.allergies import get_allergies

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
# /apis/default/api/patient/{uuid}/allergy (UUID-keyed sub-resource).
# ``severity_al`` values are drawn from the CCDA severity vocabulary seeded
# in sql/database.sql (list_id "severity_ccda") -- a superset of the tool's
# 3-value AllergySeverity enum, so the tool buckets it (see module docstring).
ALLERGY_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [
        {"id": 3, "title": "penicillin", "pid": 1, "reaction": "hives", "severity_al": "moderate_to_severe"},
        {"id": 10, "title": "Ibuprofen", "pid": 1, "reaction": "", "severity_al": None},
    ],
}


def test_happy_path_maps_allergy_rows(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == f"/apis/default/api/patient/{PHIL_UUID}/allergy":
            return httpx.Response(200, json=ALLERGY_BODY)
        raise AssertionError(f"unexpected request: {path}")

    result = get_allergies(make_openemr_client(handler), token="tok", patient_id=1)

    assert len(result.items) == 2
    first, second = result.items
    assert first.substance == "penicillin"
    assert first.reaction == "hives"
    assert first.severity == AllergySeverity.SEVERE
    assert second.substance == "Ibuprofen"
    assert second.reaction is None
    assert second.severity == AllergySeverity.UNKNOWN


def test_empty_allergy_list_yields_empty_items_not_an_error(make_openemr_client):
    """OpenEMR quirk (same as P2.3): this uuid-keyed sub-resource returns
    200 + {"data": []} for zero records, not a 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == f"/apis/default/api/patient/{PHIL_UUID}/allergy":
            return httpx.Response(200, json={"validationErrors": [], "internalErrors": [], "data": []})
        raise AssertionError(f"unexpected request: {path}")

    result = get_allergies(make_openemr_client(handler), token="tok", patient_id=1)

    assert result.items == []


def test_unknown_patient_raises_not_found_not_empty(make_openemr_client):
    """A patient that does not exist is not the same as a known patient with
    no allergies -- the uuid lookup must raise, not return empty."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        raise AssertionError(f"unexpected request beyond the uuid lookup: {request.url.path}")

    with pytest.raises(OpenEmrApiError) as excinfo:
        get_allergies(make_openemr_client(handler), token="tok", patient_id=999999)

    assert excinfo.value.category == ErrorCategory.NOT_FOUND


def test_forbidden_on_allergy_list_propagates(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == f"/apis/default/api/patient/{PHIL_UUID}/allergy":
            return httpx.Response(403, json={"error": "insufficient_scope"})
        raise AssertionError(f"unexpected request: {path}")

    with pytest.raises(OpenEmrApiError) as excinfo:
        get_allergies(make_openemr_client(handler), token="tok", patient_id=1)

    assert excinfo.value.category == ErrorCategory.FORBIDDEN


@pytest.mark.integration
def test_live_get_allergies_against_dev_stack_demo_patient_phil():
    """Live end-to-end check against the running dev stack (demo patient Phil).

    Registers a fresh confidential client scoped for the resources this tool
    reads, enables it via the same dev-only path
    ``scripts/verify-oauth-dev.sh`` uses, fetches a user token via the
    password grant, and calls the real tool against demo patient Phil
    (pubpid 1), who is seeded with >=2 allergies including Ibuprofen (see
    docs/TEST_PLAN.md §7). Requires the dev stack up; skipped by default in
    minimal CI runs (``pytest -m "not integration"``).
    """
    import subprocess

    base_url = os.environ.get("OPENEMR_BASE_URL", "https://localhost:9300")
    settings = Settings()
    scope = "openid offline_access api:oemr user/patient.read user/allergy.read"

    with httpx.Client(verify=False, timeout=15.0) as setup_client:
        creds = register_client(
            setup_client,
            base_url=base_url,
            registration_path=settings.openemr_oauth_registration_path,
            client_name="copilot-agent-test-allergies",
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
    result = get_allergies(client, token=token.access_token, patient_id=1)

    assert len(result.items) >= 2
    assert any("ibuprofen" in item.substance.lower() for item in result.items)
