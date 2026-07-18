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

from app.chat import (
    LaunchPatientMismatchError,
    TokenValidationError,
    get_launch_binding_checker,
    get_token_validator,
)
from app.documents import get_document_store
from app.ingestion import LocalIngestionStore
from app.main import app

VALID_TOKEN = "test-token"
PATIENT_A_TOKEN = "test-token-patient-a"
PATIENT_B_TOKEN = "test-token-patient-b"


def _accept_valid_token(token: str) -> None:
    if token != VALID_TOKEN:
        raise TokenValidationError("invalid token")


def _accept_patient_tokens(token: str) -> None:
    if token not in (PATIENT_A_TOKEN, PATIENT_B_TOKEN):
        raise TokenValidationError("invalid token")


def _bound_launch_checker(token: str, patient_id: int) -> None:
    """Test double for ``get_launch_binding_checker``'s flag-ON behavior:
    ``PATIENT_A_TOKEN`` is bound to patient 1, ``PATIENT_B_TOKEN`` to patient
    2 -- mirrors ``LaunchPatientBinder.verify``'s contract without a real
    OpenEMR introspection round trip."""
    bound_patient_id = {PATIENT_A_TOKEN: 1, PATIENT_B_TOKEN: 2}[token]
    if patient_id != bound_patient_id:
        raise LaunchPatientMismatchError(
            "request patient_id does not match the token launch context"
        )


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


@pytest.fixture
def bound_client(store: LocalIngestionStore) -> TestClient:
    """A client with the launch-patient binding checker wired ON -- the same
    flag-gated seam ``/chat`` enforces via ``get_launch_binding_checker``
    (finding: cross-patient IDOR on ``/documents`` when
    ``copilot_per_user_token_enabled`` is ON)."""
    app.dependency_overrides[get_token_validator] = lambda: _accept_patient_tokens
    app.dependency_overrides[get_launch_binding_checker] = lambda: _bound_launch_checker
    app.dependency_overrides[get_document_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_token_validator, None)
        app.dependency_overrides.pop(get_launch_binding_checker, None)
        app.dependency_overrides.pop(get_document_store, None)


def test_flag_on_cross_patient_token_is_rejected_with_403(
    bound_client: TestClient, store: LocalIngestionStore, tmp_path: Path
):
    """The core IDOR regression: with the launch-patient binding checker ON
    (mirrors ``copilot_per_user_token_enabled``), a token bound to patient 1
    must not be able to fetch patient 2's source document by source_id."""
    pdf_path = tmp_path / "upload.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 patient two's document")
    patient_two_source_id = store.save_source_document(2, "lab_pdf", pdf_path)

    resp = bound_client.get(f"/documents/{patient_two_source_id}", headers={"Authorization": f"Bearer {PATIENT_A_TOKEN}"})

    assert resp.status_code == 403


def test_flag_on_same_patient_token_is_allowed(
    bound_client: TestClient, store: LocalIngestionStore, tmp_path: Path
):
    """Same binding checker ON, but the token IS bound to the document's own
    patient -- must still succeed."""
    pdf_path = tmp_path / "upload.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 patient one's document")
    patient_one_source_id = store.save_source_document(1, "lab_pdf", pdf_path)

    resp = bound_client.get(f"/documents/{patient_one_source_id}", headers={"Authorization": f"Bearer {PATIENT_A_TOKEN}"})

    assert resp.status_code == 200
    assert resp.content == b"%PDF-1.4 patient one's document"


def test_flag_off_cross_patient_token_still_succeeds_unchanged(
    client: TestClient, store: LocalIngestionStore, tmp_path: Path
):
    """Flag OFF (the ``client`` fixture's default -- no binding checker
    override, so ``get_launch_binding_checker`` resolves to its flag-off
    no-op): behavior must be byte-identical to before this fix -- any valid
    token can fetch any document, since binding is deliberately inactive
    until the flag flips (Path-to-Production posture)."""
    pdf_path = tmp_path / "upload.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 patient two's document")
    patient_two_source_id = store.save_source_document(2, "lab_pdf", pdf_path)

    resp = client.get(f"/documents/{patient_two_source_id}", headers=_auth_headers())

    assert resp.status_code == 200
    assert resp.content == b"%PDF-1.4 patient two's document"


def _corrupt_meta_sidecar(tmp_path: Path, source_id: str) -> None:
    """Simulate a corrupt/partially-written meta/<source_id>.json sidecar
    (e.g. a crash mid-write) -- read_source_patient_id must fail closed to
    None rather than let json.JSONDecodeError propagate."""
    meta_path = tmp_path / "meta" / f"{source_id}.json"
    meta_path.write_text("{not valid json")


def test_flag_off_corrupt_meta_sidecar_still_serves_the_document(
    client: TestClient, store: LocalIngestionStore, tmp_path: Path
):
    """Regression: read_source_patient_id's json.loads was unguarded, so a
    corrupt sidecar 500'd GET /documents/{source_id} even with the flag OFF
    -- breaking the documented flag-OFF byte-identical guarantee (before the
    IDOR fix, the sidecar was never read at all)."""
    pdf_path = tmp_path / "upload.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf bytes")
    source_id = store.save_source_document(1, "lab_pdf", pdf_path)
    _corrupt_meta_sidecar(tmp_path, source_id)

    resp = client.get(f"/documents/{source_id}", headers=_auth_headers())

    assert resp.status_code == 200
    assert resp.content == b"%PDF-1.4 fake pdf bytes"


def test_flag_on_corrupt_meta_sidecar_fails_closed_to_403(
    bound_client: TestClient, store: LocalIngestionStore, tmp_path: Path
):
    """Flag ON + a corrupt sidecar: patient_id is unresolvable, which must
    fail CLOSED (403 via the sentinel -1 pid mismatching in
    LaunchPatientMismatchError), never a 500 and never a silent pass."""
    pdf_path = tmp_path / "upload.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf bytes")
    source_id = store.save_source_document(1, "lab_pdf", pdf_path)
    _corrupt_meta_sidecar(tmp_path, source_id)

    resp = bound_client.get(f"/documents/{source_id}", headers={"Authorization": f"Bearer {PATIENT_A_TOKEN}"})

    assert resp.status_code == 403
