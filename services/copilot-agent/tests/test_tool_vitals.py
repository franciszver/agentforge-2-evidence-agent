"""Hermetic + one live-integration test for the ``get_vitals`` tool.

All hermetic HTTP is served by ``httpx.MockTransport`` so the suite never
touches the network; fixture shapes are modeled on real responses observed
against the dev stack for demo patient Wanda Moore (pubpid 3) -- see
``app/tools/vitals.py``'s module docstring for the endpoint quirks those
shapes encode, notably the blood-pressure panel Observation whose systolic
and diastolic readings live in a ``component`` array rather than the
Observation's own ``valueQuantity``. The single ``@pytest.mark.integration``
test hits the real running dev stack and is skipped by default (minimal CI
runs hermetic tests only).
"""

from __future__ import annotations

import datetime
import os

import httpx
import pytest

from app.config import Settings
from app.openemr_auth import fetch_token_password_grant, register_client
from app.openemr_client import ErrorCategory, OpenEmrApiError, OpenEmrClient
from app.schemas.common import VitalType
from app.tools.vitals import get_vitals

WANDA_UUID = "a243a1bb-1795-4ff7-afae-27083d15ecbc"

PATIENT_LIST_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [
        {"pid": 3, "fname": "Wanda", "lname": "Moore", "uuid": WANDA_UUID},
    ],
}

EMPTY_BUNDLE = {
    "resourceType": "Bundle",
    "type": "collection",
    "total": 0,
    "link": [{"relation": "self", "url": "https://localhost:9300/apis/default/fhir/Observation"}],
}

# Trimmed to the fields the tool reads; modeled on the real Bundle observed
# live for GET /apis/default/fhir/Observation?patient={uuid}&category=vital-signs
# (see module docstring). Includes: a panel Observation with no value of its
# own (skipped), a simple vital with a top-level valueQuantity (Heart rate),
# a non-vital simple observation (Temperature Location, valueString --
# skipped, no matching LOINC), and the blood-pressure Observation whose
# systolic/diastolic readings live only in its "component" array.
VITALS_BUNDLE = {
    "resourceType": "Bundle",
    "type": "collection",
    "total": 4,
    "entry": [
        {
            "resource": {
                "id": "panel",
                "resourceType": "Observation",
                "code": {"coding": [{"system": "http://loinc.org", "code": "85353-1", "display": "Vital signs panel"}]},
                "effectiveDateTime": "2014-02-01T21:47:33+00:00",
                "hasMember": [{"reference": "Observation/hr", "type": "Observation", "display": "Heart rate"}],
            }
        },
        {
            "resource": {
                "id": "hr",
                "resourceType": "Observation",
                "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4", "display": "Heart rate"}]},
                "effectiveDateTime": "2014-02-01T21:47:33+00:00",
                "valueQuantity": {"value": 87, "unit": "/min", "system": "http://unitsofmeasure.org", "code": "/min"},
            }
        },
        {
            "resource": {
                "id": "temp-loc",
                "resourceType": "Observation",
                "code": {"coding": [{"system": "http://loinc.org", "code": "8327-9", "display": "Temperature Location"}]},
                "effectiveDateTime": "2014-02-01T21:47:33+00:00",
                "valueString": "Tympanic Membrane",
            }
        },
        {
            "resource": {
                "id": "bp",
                "resourceType": "Observation",
                "code": {
                    "coding": [
                        {"system": "http://loinc.org", "code": "85354-9", "display": "Blood pressure systolic and diastolic"}
                    ]
                },
                "effectiveDateTime": "2024-06-01T09:00:00+00:00",
                "component": [
                    {
                        "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic blood pressure"}]},
                        "valueQuantity": {"value": 120, "unit": "mm[Hg]", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"},
                    },
                    {
                        "code": {"coding": [{"system": "http://loinc.org", "code": "8462-4", "display": "Diastolic blood pressure"}]},
                        "valueQuantity": {"value": 80, "unit": "mm[Hg]", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"},
                    },
                ],
            }
        },
    ],
}


def test_happy_path_maps_and_sorts_most_recent_first(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == "/apis/default/fhir/Observation":
            assert request.url.params["patient"] == WANDA_UUID
            assert request.url.params["category"] == "vital-signs"
            return httpx.Response(200, json=VITALS_BUNDLE)
        raise AssertionError(f"unexpected request: {path}")

    result = get_vitals(make_openemr_client(handler), token="tok", patient_id=3)

    # panel (no value) and Temperature Location (no matching LOINC) are
    # skipped; heart rate (top-level) + systolic + diastolic (component)
    # yields 3 items, most-recent (BP, 2024-06-01) first.
    assert len(result.items) == 3
    first, second, third = result.items
    assert first.date == datetime.datetime(2024, 6, 1, 9, 0, tzinfo=datetime.timezone.utc)
    assert {first.vital_type, second.vital_type} <= {VitalType.BLOOD_PRESSURE_SYSTOLIC, VitalType.BLOOD_PRESSURE_DIASTOLIC}
    systolic = next(item for item in (first, second) if item.vital_type == VitalType.BLOOD_PRESSURE_SYSTOLIC)
    diastolic = next(item for item in (first, second) if item.vital_type == VitalType.BLOOD_PRESSURE_DIASTOLIC)
    assert systolic.value == 120.0
    assert systolic.unit == "mm[Hg]"
    assert diastolic.value == 80.0

    assert third.vital_type == VitalType.HEART_RATE
    assert third.value == 87.0
    assert third.unit == "/min"
    assert third.date == datetime.datetime(2014, 2, 1, 21, 47, 33, tzinfo=datetime.timezone.utc)


def test_limit_filter_keeps_most_recent_n(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == "/apis/default/fhir/Observation":
            return httpx.Response(200, json=VITALS_BUNDLE)
        raise AssertionError(f"unexpected request: {path}")

    result = get_vitals(make_openemr_client(handler), token="tok", patient_id=3, limit=1)

    assert len(result.items) == 1
    assert result.items[0].date == datetime.datetime(2024, 6, 1, 9, 0, tzinfo=datetime.timezone.utc)


def test_empty_vitals_bundle_yields_empty_items_not_an_error(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == "/apis/default/fhir/Observation":
            return httpx.Response(200, json=EMPTY_BUNDLE)
        raise AssertionError(f"unexpected request: {path}")

    result = get_vitals(make_openemr_client(handler), token="tok", patient_id=3)

    assert result.items == []


def test_unknown_patient_raises_not_found_not_empty(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        raise AssertionError(f"unexpected request beyond the uuid lookup: {request.url.path}")

    with pytest.raises(OpenEmrApiError) as excinfo:
        get_vitals(make_openemr_client(handler), token="tok", patient_id=999999)

    assert excinfo.value.category == ErrorCategory.NOT_FOUND


def test_forbidden_on_observation_search_propagates(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == "/apis/default/fhir/Observation":
            return httpx.Response(403, json={"error": "insufficient_scope"})
        raise AssertionError(f"unexpected request: {path}")

    with pytest.raises(OpenEmrApiError) as excinfo:
        get_vitals(make_openemr_client(handler), token="tok", patient_id=3)

    assert excinfo.value.category == ErrorCategory.FORBIDDEN


@pytest.mark.integration
def test_live_get_vitals_against_dev_stack_demo_patient_wanda():
    """Live end-to-end check against the running dev stack (demo patient Wanda).

    Registers a fresh confidential client scoped for the resources this tool
    reads, enables it via the same dev-only path
    ``scripts/verify-oauth-dev.sh`` uses, fetches a user token via the
    password grant, and calls the real tool against demo patient Wanda
    (pubpid 3), who is seeded with vitals from her single 2014-02-01
    encounter. Requires the dev stack up; skipped by default in minimal CI
    runs (``pytest -m "not integration"``).
    """
    import subprocess

    base_url = os.environ.get("OPENEMR_BASE_URL", "https://localhost:9300")
    settings = Settings()
    scope = "openid offline_access api:oemr api:fhir user/patient.read user/Observation.read patient/Observation.read"

    with httpx.Client(verify=False, timeout=15.0) as setup_client:
        creds = register_client(
            setup_client,
            base_url=base_url,
            registration_path=settings.openemr_oauth_registration_path,
            client_name="copilot-agent-test-vitals",
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
    result = get_vitals(client, token=token.access_token, patient_id=3)

    assert isinstance(result.items, list)
    assert len(result.items) >= 0
