"""Hermetic tests for the OpenEMR REST/FHIR API client.

All HTTP is served by ``httpx.MockTransport`` so the suite never touches the
network. Live end-to-end verification lives outside the test suite.
"""

import httpx
import pytest

from app.config import Settings
from app.openemr_client import ErrorCategory, OpenEmrApiError, OpenEmrClient


def _client(handler) -> OpenEmrClient:
    return OpenEmrClient(
        base_url="https://openemr",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_get_rest_success_returns_json_and_hits_correct_url_with_bearer_header():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("Authorization")
        captured["query"] = str(request.url.query, "utf-8")
        return httpx.Response(200, json={"pid": "1", "fname": "Ada"})

    client = _client(handler)
    result = client.get_rest("patient/1", token="my-token", params={"limit": "5"})

    assert result == {"pid": "1", "fname": "Ada"}
    assert captured["path"] == "/apis/default/api/patient/1"
    assert captured["authorization"] == "Bearer my-token"
    assert captured["query"] == "limit=5"


def test_get_fhir_success_hits_correct_fhir_base():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/apis/default/fhir/Patient"
        return httpx.Response(200, json={"resourceType": "Bundle", "total": 0})

    client = _client(handler)
    result = client.get_fhir("Patient", token="my-token")

    assert result == {"resourceType": "Bundle", "total": 0}


def test_get_rest_401_raises_unauthorized_without_leaking_token_or_body():
    token = "super-secret-bearer-token"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid_token", "patient_ssn": "111-22-3333"})

    client = _client(handler)
    with pytest.raises(OpenEmrApiError) as excinfo:
        client.get_rest("patient/1", token=token)

    assert excinfo.value.category == ErrorCategory.UNAUTHORIZED
    message = str(excinfo.value)
    assert token not in message
    assert "111-22-3333" not in message


def test_get_rest_403_raises_forbidden():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "insufficient_scope"})

    client = _client(handler)
    with pytest.raises(OpenEmrApiError) as excinfo:
        client.get_rest("patient/1", token="tok")

    assert excinfo.value.category == ErrorCategory.FORBIDDEN


def test_get_rest_404_raises_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    client = _client(handler)
    with pytest.raises(OpenEmrApiError) as excinfo:
        client.get_rest("patient/999", token="tok")

    assert excinfo.value.category == ErrorCategory.NOT_FOUND


def test_get_rest_timeout_raises_timeout_category():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    client = _client(handler)
    with pytest.raises(OpenEmrApiError) as excinfo:
        client.get_rest("patient/1", token="tok")

    assert excinfo.value.category == ErrorCategory.TIMEOUT


def test_get_rest_unexpected_status_raises_unexpected_without_leaking_body():
    secret_body = "internal-stack-trace-detail"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=secret_body)

    client = _client(handler)
    with pytest.raises(OpenEmrApiError) as excinfo:
        client.get_rest("patient/1", token="tok")

    assert excinfo.value.category == ErrorCategory.UNEXPECTED
    assert secret_body not in str(excinfo.value)


@pytest.mark.parametrize("method_name", ["get_rest", "get_fhir"])
@pytest.mark.parametrize(
    "malicious_path",
    [
        "https://evil.example/x",
        "//evil.example/x",
        "@evil.example",
        "../../../etc/passwd",
    ],
)
def test_rejects_paths_that_could_escape_the_configured_host(method_name, malicious_path):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    client = _client(handler)
    method = getattr(client, method_name)
    with pytest.raises(OpenEmrApiError) as excinfo:
        method(malicious_path, token="tok")

    assert excinfo.value.category == ErrorCategory.INVALID_PATH
    assert calls == []


def test_from_settings_builds_client_targeting_configured_base_url():
    settings = Settings(openemr_base_url="https://openemr.example")

    client = OpenEmrClient.from_settings(settings)

    assert client._host == "openemr.example"
    assert client._scheme == "https"
