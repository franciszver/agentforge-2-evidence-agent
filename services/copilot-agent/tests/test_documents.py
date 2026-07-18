"""Hermetic tests for ``GET /documents/{source_id}`` (P3.7 citation overlay).

Serves the stored source PDF a document citation points at, so the module UI
can open it scrolled to the cited page when a clinician clicks a
document-sourced claim. Access-controlled: bearer-token gated (same seam as
every other agent endpoint) and ``source_id`` is validated against
``LocalIngestionStore``'s own uuid4().hex naming -- anything else (path
traversal attempts included) is rejected before any filesystem lookup.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.chat import TokenValidationError, get_token_validator
from app.documents import get_document_store
from app.ingestion import LocalIngestionStore
from app.main import app

VALID_TOKEN = "test-token"


def _accept_valid_token(token: str) -> None:
    if token != VALID_TOKEN:
        raise TokenValidationError("invalid token")


@pytest.fixture
def store(tmp_path: Path) -> LocalIngestionStore:
    return LocalIngestionStore(tmp_path)


@pytest.fixture
def client(store: LocalIngestionStore) -> TestClient:
    app.dependency_overrides[get_token_validator] = lambda: _accept_valid_token
    app.dependency_overrides[get_document_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_token_validator, None)
        app.dependency_overrides.pop(get_document_store, None)


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {VALID_TOKEN}"}


def test_serves_a_stored_document_by_its_source_id(client: TestClient, store: LocalIngestionStore, tmp_path: Path):
    pdf_path = tmp_path / "upload.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf bytes")
    source_id = store.save_source_document(1, "lab_pdf", pdf_path)

    resp = client.get(f"/documents/{source_id}", headers=_auth_headers())

    assert resp.status_code == 200
    assert resp.content == b"%PDF-1.4 fake pdf bytes"
    assert resp.headers["content-type"] == "application/pdf"


def test_unknown_but_well_formed_source_id_is_404(client: TestClient):
    unknown_id = uuid.uuid4().hex

    resp = client.get(f"/documents/{unknown_id}", headers=_auth_headers())

    assert resp.status_code == 404


@pytest.mark.parametrize(
    "malicious_id",
    [
        "../../../../etc/passwd",
        "..%2f..%2fetc%2fpasswd",
        "not-hex-chars-!!!!",
        "short",
        "a" * 32 + "-extra",
    ],
)
def test_malformed_or_path_traversal_source_id_is_rejected_without_filesystem_access(
    client: TestClient, malicious_id: str
):
    resp = client.get(f"/documents/{malicious_id}", headers=_auth_headers())

    # Either a clean 400 (rejected by the source_id format check) or FastAPI's
    # own 404 route-not-found for a path containing "/" -- NEVER a 200/500
    # that would indicate the traversal attempt reached the filesystem.
    assert resp.status_code in (400, 404)
    assert resp.status_code != 200


def test_missing_or_invalid_bearer_token_is_rejected(client: TestClient, store: LocalIngestionStore, tmp_path: Path):
    pdf_path = tmp_path / "upload.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf bytes")
    source_id = store.save_source_document(1, "lab_pdf", pdf_path)

    resp_no_auth = client.get(f"/documents/{source_id}")
    resp_bad_auth = client.get(f"/documents/{source_id}", headers={"Authorization": "Bearer wrong-token"})

    assert resp_no_auth.status_code == 401
    assert resp_bad_auth.status_code == 401
