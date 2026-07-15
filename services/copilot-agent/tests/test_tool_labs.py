"""Hermetic + one live-integration test for the ``get_recent_labs`` tool.

All hermetic HTTP is served by ``httpx.MockTransport`` so the suite never
touches the network; the patient-roster and empty-bundle fixture shapes are
modeled on real responses observed against the dev stack (see
``app/tools/labs.py``'s module docstring). The demo dataset ships zero lab
Observations for all three demo patients (docs/TEST_PLAN.md §7's
``no-labs`` note), so no live example of a *populated* laboratory Bundle was
available; the happy-path fixture below is modeled on the standard FHIR R4
``Observation`` shape instead (the same resource family confirmed live for
vital-signs, category ``laboratory`` swapped in per the FHIR spec). The
single ``@pytest.mark.integration`` test hits the real running dev stack and
is skipped by default (minimal CI runs hermetic tests only).
"""

from __future__ import annotations

import datetime
import os

import httpx
import pytest

from app.config import Settings
from app.openemr_auth import fetch_token_password_grant, register_client
from app.openemr_client import ErrorCategory, OpenEmrApiError, OpenEmrClient
from app.schemas.common import AbnormalFlag
from app.tools.labs import get_recent_labs

WANDA_UUID = "a243a1bb-1795-4ff7-afae-27083d15ecbc"

PATIENT_LIST_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [
        {"pid": 3, "fname": "Wanda", "lname": "Moore", "uuid": WANDA_UUID},
    ],
}

# Zero-results shape observed live for GET
# /apis/default/fhir/Observation?patient={uuid}&category=... -- the "entry"
# key is omitted entirely, not an empty list (see module docstring).
EMPTY_BUNDLE = {
    "resourceType": "Bundle",
    "type": "collection",
    "total": 0,
    "link": [{"relation": "self", "url": "https://localhost:9300/apis/default/fhir/Observation"}],
}


def _lab_entry(
    *,
    obs_id: str,
    display: str,
    date: str,
    value: object = None,
    unit: str | None = None,
    value_string: str | None = None,
    ref_low: float | None = None,
    ref_high: float | None = None,
    interpretation_code: str | None = None,
) -> dict:
    resource: dict = {
        "id": obs_id,
        "resourceType": "Observation",
        "status": "final",
        "category": [
            {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]}
        ],
        "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": display}]},
        "effectiveDateTime": date,
    }
    if value is not None:
        resource["valueQuantity"] = {"value": value, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit}
    if value_string is not None:
        resource["valueString"] = value_string
    if ref_low is not None or ref_high is not None:
        low = {"value": ref_low, "unit": unit} if ref_low is not None else None
        high = {"value": ref_high, "unit": unit} if ref_high is not None else None
        range_entry = {}
        if low is not None:
            range_entry["low"] = low
        if high is not None:
            range_entry["high"] = high
        resource["referenceRange"] = [range_entry]
    if interpretation_code is not None:
        resource["interpretation"] = [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": interpretation_code,
                    }
                ]
            }
        ]
    return {"fullUrl": f"https://localhost:9300/apis/default/fhir/Observation/{obs_id}", "resource": resource}


LABS_BUNDLE = {
    "resourceType": "Bundle",
    "type": "collection",
    "total": 3,
    "entry": [
        _lab_entry(
            obs_id="lab-1",
            display="Hemoglobin A1c",
            date="2024-01-05T10:00:00+00:00",
            value=7.2,
            unit="%",
            ref_low=4.0,
            ref_high=5.6,
            interpretation_code="H",
        ),
        _lab_entry(
            obs_id="lab-2",
            display="Hemoglobin A1c",
            date="2024-03-10T10:00:00+00:00",
            value=6.8,
            unit="%",
            ref_low=4.0,
            ref_high=5.6,
            interpretation_code="N",
        ),
        _lab_entry(
            obs_id="lab-3",
            display="Urine Culture",
            date="2024-02-01T10:00:00+00:00",
            value_string="No growth",
        ),
    ],
}


def test_happy_path_maps_and_sorts_most_recent_first(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == "/apis/default/fhir/Observation":
            assert request.url.params["patient"] == WANDA_UUID
            assert request.url.params["category"] == "laboratory"
            return httpx.Response(200, json=LABS_BUNDLE)
        raise AssertionError(f"unexpected request: {path}")

    result = get_recent_labs(make_openemr_client(handler), token="tok", patient_id=3)

    assert len(result.items) == 3
    first, second, third = result.items
    # Most-recent-first: 2024-03-10, 2024-02-01, 2024-01-05.
    assert first.date == datetime.datetime(2024, 3, 10, 10, 0, tzinfo=datetime.timezone.utc)
    assert first.test_name == "Hemoglobin A1c"
    assert first.value == "6.8"
    assert first.unit == "%"
    assert first.reference_range == "4.0-5.6 %"
    assert first.abnormal_flag == AbnormalFlag.NORMAL

    assert second.test_name == "Urine Culture"
    assert second.value == "No growth"
    assert second.unit is None

    assert third.value == "7.2"
    assert third.abnormal_flag == AbnormalFlag.HIGH


def test_unrecognized_interpretation_maps_to_unknown_not_normal(make_openemr_client):
    """Clinical safety: an interpretation that *is* present but carries a code
    this tool doesn't map (e.g. HL7 "A" = abnormal) must surface as UNKNOWN,
    never as a falsely-reassuring NORMAL."""
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "total": 1,
        "entry": [
            _lab_entry(
                obs_id="lab-abn",
                display="Some Panel",
                date="2024-05-01T10:00:00+00:00",
                value=1.0,
                unit="x",
                interpretation_code="A",
            ),
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == "/apis/default/fhir/Observation":
            return httpx.Response(200, json=bundle)
        raise AssertionError(f"unexpected request: {path}")

    result = get_recent_labs(make_openemr_client(handler), token="tok", patient_id=3)

    assert len(result.items) == 1
    assert result.items[0].abnormal_flag == AbnormalFlag.UNKNOWN


def test_absent_interpretation_maps_to_normal(make_openemr_client):
    """An Observation with no ``interpretation`` field at all is the standard
    EHR "nothing flagged" case -> NORMAL (distinct from an unrecognized code)."""
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "total": 1,
        "entry": [
            _lab_entry(obs_id="lab-plain", display="Some Test", date="2024-05-01T10:00:00+00:00", value=1.0, unit="x"),
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == "/apis/default/fhir/Observation":
            return httpx.Response(200, json=bundle)
        raise AssertionError(f"unexpected request: {path}")

    result = get_recent_labs(make_openemr_client(handler), token="tok", patient_id=3)

    assert len(result.items) == 1
    assert result.items[0].abnormal_flag == AbnormalFlag.NORMAL


def test_limit_filter_keeps_most_recent_n(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == "/apis/default/fhir/Observation":
            return httpx.Response(200, json=LABS_BUNDLE)
        raise AssertionError(f"unexpected request: {path}")

    result = get_recent_labs(make_openemr_client(handler), token="tok", patient_id=3, limit=2)

    assert len(result.items) == 2
    assert result.items[0].date == datetime.datetime(2024, 3, 10, 10, 0, tzinfo=datetime.timezone.utc)
    assert result.items[1].date == datetime.datetime(2024, 2, 1, 10, 0, tzinfo=datetime.timezone.utc)


def test_empty_lab_bundle_yields_empty_items_not_an_error(make_openemr_client):
    """The Wanda/no-labs case: a known patient with zero lab Observations
    gets items=[], not an error -- this is the "missing data" eval state."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        if path == "/apis/default/fhir/Observation":
            return httpx.Response(200, json=EMPTY_BUNDLE)
        raise AssertionError(f"unexpected request: {path}")

    result = get_recent_labs(make_openemr_client(handler), token="tok", patient_id=3)

    assert result.items == []


def test_unknown_patient_raises_not_found_not_empty(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apis/default/api/patient":
            return httpx.Response(200, json=PATIENT_LIST_BODY)
        raise AssertionError(f"unexpected request beyond the uuid lookup: {request.url.path}")

    with pytest.raises(OpenEmrApiError) as excinfo:
        get_recent_labs(make_openemr_client(handler), token="tok", patient_id=999999)

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
        get_recent_labs(make_openemr_client(handler), token="tok", patient_id=3)

    assert excinfo.value.category == ErrorCategory.FORBIDDEN


@pytest.mark.integration
def test_live_get_recent_labs_against_dev_stack_demo_patient_wanda_has_no_labs():
    """Live end-to-end check against the running dev stack (demo patient Wanda).

    Registers a fresh confidential client scoped for the resources this tool
    reads, enables it via the same dev-only path
    ``scripts/verify-oauth-dev.sh`` uses, fetches a user token via the
    password grant, and calls the real tool against demo patient Wanda
    (pubpid 3), who has zero lab Observations in the unmodified demo dataset
    (docs/TEST_PLAN.md §7's ``no-labs`` note) -- the tool must return
    items=[], not raise. Requires the dev stack up; skipped by default in
    minimal CI runs (``pytest -m "not integration"``).
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
            client_name="copilot-agent-test-labs",
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
    result = get_recent_labs(client, token=token.access_token, patient_id=3)

    assert result.items == []
