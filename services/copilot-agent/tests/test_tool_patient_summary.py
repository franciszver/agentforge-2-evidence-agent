"""Hermetic + one live-integration test for the ``get_patient_summary`` tool.

All hermetic HTTP is served by ``httpx.MockTransport`` so the suite never
touches the network; fixture shapes are modeled on real responses observed
against the dev stack for demo patient Phil Belford (pubpid 1). The single
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
from app.schemas.common import Sex
from app.tools.patient_summary import get_patient_summary

PHIL_UUID = "a243a1bb-178f-4092-8c67-52dfaf67fca6"

# Trimmed to the fields the tool reads; modeled on the real REST payload
# shape for GET /apis/default/api/patient (list endpoint has no internal
# pid/id filter, so the tool fetches the roster and selects client-side).
PATIENT_LIST_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [
        {
            "pid": 1,
            "fname": "Phil",
            "lname": "Belford",
            "DOB": "1972-02-09",
            "sex": "Male",
            "uuid": PHIL_UUID,
        },
        {
            "pid": 2,
            "fname": "Susan",
            "lname": "Underwood",
            "DOB": "1980-05-01",
            "sex": "Female",
            "uuid": "b1a2c3d4-0000-4000-8000-000000000002",
        },
    ],
}

# Bare-list shape observed for GET /apis/default/api/patient/{pid}/medication
# and .../appointment (ListRestController / AppointmentRestController).
MEDICATION_BODY = [
    {"id": 4, "title": "Norvasc", "pid": 1, "uuid": "a243a1bb-145a-4fd9-a7b9-cbc03078d4d8"},
    {"id": 5, "title": "Lisinopril", "pid": 1, "uuid": "a243a1bb-145c-42b9-b1b9-f1d1dfd69395"},
]
APPOINTMENT_BODY = [
    {"pc_eid": 10, "pid": 1, "pc_eventDate": "2014-01-31", "pc_startTime": "14:30:00"},
]

# Wrapped {"data": [...]} shape observed for the UUID-keyed sub-resources
# (allergy, medical_problem, encounter).
ALLERGY_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [
        {"id": 3, "title": "penicillin", "pid": 1},
        {"id": 10, "title": "Ibuprofen", "pid": 1},
    ],
}
MEDICAL_PROBLEM_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [
        {"id": 1, "title": "HTN", "pid": 1},
        {"id": 2, "title": "Chronic Renal Insuficiency", "pid": 1},
    ],
}
ENCOUNTER_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [{"eid": 5, "date": "2014-02-01 00:00:00", "pid": 1}],
}

# FHIR Bundle shape observed for GET /apis/default/fhir/Observation.
VITALS_BUNDLE = {"resourceType": "Bundle", "type": "collection", "total": 15}
LABS_BUNDLE_EMPTY = {"resourceType": "Bundle", "type": "collection", "total": 0}


def test_happy_path_maps_demographics_and_section_counts(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        query = dict(request.url.params)
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == "/apis/default/api/patient/1/medication":
            return httpx.Response(200, json=MEDICATION_BODY)
        if path == "/apis/default/api/patient/1/appointment":
            return httpx.Response(200, json=APPOINTMENT_BODY)
        if path == f"/apis/default/api/patient/{PHIL_UUID}/allergy":
            return httpx.Response(200, json=ALLERGY_BODY)
        if path == f"/apis/default/api/patient/{PHIL_UUID}/medical_problem":
            return httpx.Response(200, json=MEDICAL_PROBLEM_BODY)
        if path == f"/apis/default/api/patient/{PHIL_UUID}/encounter":
            return httpx.Response(200, json=ENCOUNTER_BODY)
        if path == "/apis/default/fhir/Observation" and query.get("category") == "vital-signs":
            return httpx.Response(200, json=VITALS_BUNDLE)
        if path == "/apis/default/fhir/Observation" and query.get("category") == "laboratory":
            return httpx.Response(200, json=LABS_BUNDLE_EMPTY)
        raise AssertionError(f"unexpected request: {path} {query}")

    result = get_patient_summary(make_openemr_client(handler), token="tok", patient_id=1)

    assert result.patient_id == 1
    assert result.first_name == "Phil"
    assert result.last_name == "Belford"
    assert result.date_of_birth == datetime.date(1972, 2, 9)
    assert result.sex == Sex.MALE
    assert result.medication_count == 2
    assert result.allergy_count == 2
    assert result.problem_count == 2
    assert result.encounter_count == 1
    assert result.appointment_count == 1
    assert result.vital_count == 15
    assert result.recent_lab_count == 0
    assert result.source_refs is None


def test_empty_sections_yield_zero_counts_not_an_error(make_openemr_client):
    """Empty/404 sections are a valid state -- zero counts, no exception.

    Covers both OpenEMR empty-section shapes observed live: the
    ListRestController-backed sub-resources (medication, appointment)
    return HTTP 404 with an empty body for zero records, while the
    UUID-keyed sub-resources (allergy, medical_problem, encounter) return
    200 with ``{"data": []}``.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        query = dict(request.url.params)
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path in ("/apis/default/api/patient/1/medication", "/apis/default/api/patient/1/appointment"):
            return httpx.Response(404, text="")
        if path in (
            f"/apis/default/api/patient/{PHIL_UUID}/allergy",
            f"/apis/default/api/patient/{PHIL_UUID}/medical_problem",
            f"/apis/default/api/patient/{PHIL_UUID}/encounter",
        ):
            return httpx.Response(200, json={"validationErrors": [], "internalErrors": [], "data": []})
        if path == "/apis/default/fhir/Observation" and query.get("category") in ("vital-signs", "laboratory"):
            return httpx.Response(200, json=LABS_BUNDLE_EMPTY)
        raise AssertionError(f"unexpected request: {path} {query}")

    result = get_patient_summary(make_openemr_client(handler), token="tok", patient_id=1)

    assert result.medication_count == 0
    assert result.appointment_count == 0
    assert result.allergy_count == 0
    assert result.problem_count == 0
    assert result.encounter_count == 0
    assert result.vital_count == 0
    assert result.recent_lab_count == 0


def test_forbidden_on_patient_propagates_and_does_not_swallow(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apis/default/api/patient":
            return httpx.Response(403, json={"error": "insufficient_scope"})
        raise AssertionError(f"unexpected request beyond the patient fetch: {request.url.path}")

    with pytest.raises(OpenEmrApiError) as excinfo:
        get_patient_summary(make_openemr_client(handler), token="tok", patient_id=1)

    assert excinfo.value.category == ErrorCategory.FORBIDDEN


def test_patient_not_found_raises_not_found(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        raise AssertionError(f"unexpected request beyond the patient fetch: {request.url.path}")

    with pytest.raises(OpenEmrApiError) as excinfo:
        get_patient_summary(make_openemr_client(handler), token="tok", patient_id=999999)

    assert excinfo.value.category == ErrorCategory.NOT_FOUND


@pytest.mark.integration
def test_live_get_patient_summary_against_dev_stack_demo_patient_phil():
    """Live end-to-end check against the running dev stack (demo patient Phil).

    Registers a fresh confidential client scoped for every resource this
    tool reads, enables it via the same dev-only path
    ``scripts/verify-oauth-dev.sh`` uses, fetches a user token via the
    password grant, and calls the real tool against demo patient Phil
    (pubpid 1). Requires the dev stack up; skipped by default in minimal
    CI runs (``pytest -m "not integration"``).
    """
    import subprocess

    base_url = os.environ.get("OPENEMR_BASE_URL", "https://localhost:9300")
    settings = Settings()
    scope = (
        "openid offline_access api:oemr api:fhir "
        "user/patient.read user/Patient.read "
        "user/medication.read user/allergy.read "
        "user/medical_problem.read user/encounter.read "
        "user/vital.read user/Observation.read "
        "user/appointment.read"
    )

    with httpx.Client(verify=False, timeout=15.0) as setup_client:
        creds = register_client(
            setup_client,
            base_url=base_url,
            registration_path=settings.openemr_oauth_registration_path,
            client_name="copilot-agent-test-patient-summary",
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
    result = get_patient_summary(client, token=token.access_token, patient_id=1)

    assert result.first_name == "Phil"
    assert result.last_name == "Belford"
    assert result.date_of_birth == datetime.date(1972, 2, 9)
    assert result.allergy_count >= 1
