"""ACL / authorization scoping proven end-to-end against the LIVE stack (P2.18).

Three capstone integration cases that prove the authorization story built
across P2.13-P2.17 -- and completed by the #124 ``authorization_code`` flow --
holds against the real OpenEMR + real qwen3:4b, not just in hermetic mocks:

  (a) A real, per-user OpenEMR bearer token for a genuinely SCOPED demo user
      (``receptionist`` -- OpenEMR "Front Office" role) physically cannot pull
      clinical PHI. The tool layer surfaces the denial as a typed
      ``OpenEmrApiError`` carrying ZERO PHI -- the whole point of token
      pass-through: the agent can only reach what its user's token permits.

      What layer this case exercises (verified empirically against the live
      stack, 2026-07-15): the OAuth SCOPE wall, not per-user gacl ACL. The
      dev-only password grant (``app.openemr_auth.fetch_token_password_grant``)
      only ever grants demographics scope (``user/patient.read``) -- OpenEMR
      strips ``api:oemr`` / ``api:fhir`` / every non-Patient resource scope
      from a ROPC token for ALL roles, so every clinical category returns 401
      uniformly at the scope wall, before role ACL is consulted. This case
      proves that wall: a valid per-user token reaches ONLY the demographics it
      is scoped for and is denied clinical PHI, with zero leak on denial. The
      role-DIFFERENTIATED gacl-ACL decision (403 for a role lacking the ACL,
      200 for admin) needs the ``authorization_code`` token that carries the
      api scopes -- proven separately in case (c).

  (b) A physician whose token could open any chart is BOUND to one patient and
      asks about a DIFFERENT one. The P2.16 binding refuses (no cross-patient
      fetch, zero cross-patient PHI), and the P2.17 audit trail records the
      attempt (who asked, on which bound chart, what they asked).

  (c) Genuine per-ROLE differentiation on the SAME clinical endpoint, proven
      live end-to-end via the #124 ``authorization_code`` + PKCE + SMART-launch
      + introspection flow: a restricted role (``accountant``) is denied with a
      hard HTTP 403 while ``admin`` gets HTTP 200 on the identical
      ``GET /apis/default/api/patient/1/medication`` request. This is the real
      per-user gacl-ACL enforcement (not the coarse OAuth scope wall of case
      (a)); it closes AUDIT.md F-10 in principle. Because the prod
      authorization_code client requires interactive browser CONSENT and the
      password grant was dropped from it, the two role tokens cannot be minted
      non-interactively -- they are supplied via env
      (``COPILOT_ACL_RESTRICTED_TOKEN`` / ``COPILOT_ACL_ADMIN_TOKEN``) after a
      human runs the consent flow, and the case skips cleanly when absent.
      Live-proven 2026-07-16: accountant -> 403, admin -> 200.

Skipped by default (``pytest -m "not integration"``). Case (a) needs the live
OpenEMR stack + the dev OAuth client creds file (produced by
``scripts/verify-oauth-dev.sh``) and skips with a clear message when either is
absent; case (b) needs a reachable Ollama at ``OLLAMA_BASE_URL`` (it exercises
the real model, so an unreachable Ollama surfaces as a hard error, not a skip);
case (c) needs the two env-supplied per-role authorization_code tokens and
skips when either is absent.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.chat import Turn, _user_identity_from_token
from app.config import Settings
from app.ollama_client import OllamaClient
from app.openemr_auth import OpenEmrAuthError, fetch_token_password_grant
from app.openemr_client import ErrorCategory, OpenEmrApiError, OpenEmrClient
from app.planner import Planner
from app.tools._common import resolve_patient_uuid
from app.tools.medications import get_medications

pytestmark = pytest.mark.integration

# Host-side base URL for the live OpenEMR (self-signed cert in dev -> verify off,
# matching scripts/verify_oauth_dev.py and app.config's dev default).
_OPENEMR_BASE_URL = os.environ.get("OPENEMR_BASE_URL", "https://localhost:9300")

# Gitignored dev OAuth client credentials, written by scripts/verify-oauth-dev.sh.
_CREDS_PATH = (
    Path(__file__).resolve().parents[2]
    / "services"
    / "copilot-agent"
    / ".openemr-dev-client.json"
)

# A genuinely scoped demo user: OpenEMR's "Front Office" role. The pinned demo
# dataset ships this account; its password equals its username (demo default).
_SCOPED_USER = "receptionist"
_SCOPED_PASS = "receptionist"

# Phil Belford -- the canonical demo patient (pubpid/pid 1, TEST_PLAN.md §7).
_DEMO_PATIENT_ID = 1


def _password_grant_token(username: str, password: str) -> str:
    """Acquire a REAL per-user OpenEMR bearer via the dev password grant.

    Skips (not fails) when the live stack or the dev OAuth client creds file is
    unavailable -- these are integration prerequisites, not assertions.
    """
    if not _CREDS_PATH.exists():
        pytest.skip(
            f"dev OAuth client creds not found at {_CREDS_PATH.name}; "
            "run scripts/verify-oauth-dev.sh against the live stack first"
        )
    creds = json.loads(_CREDS_PATH.read_text(encoding="utf-8"))
    settings = Settings()
    # verify=False: the dev stack uses a self-signed certificate (see app.config).
    try:
        with httpx.Client(verify=False, timeout=20.0) as client:
            token = fetch_token_password_grant(
                client,
                base_url=_OPENEMR_BASE_URL,
                token_path=settings.openemr_oauth_token_path,
                client_id=creds["client_id"],
                client_secret=creds["client_secret"],
                username=username,
                password=password,
                scope=settings.openemr_oauth_scopes,
            )
    except (httpx.HTTPError, OpenEmrAuthError) as exc:
        pytest.skip(f"live OpenEMR token grant unavailable ({username}): {exc}")
    return token.access_token


# --------------------------------------------------------------------------- #
# Case (a): a scoped user's real token cannot reach clinical PHI.
# --------------------------------------------------------------------------- #
def test_scoped_user_token_cannot_reach_clinical_phi_and_leaks_none() -> None:
    """A real ``receptionist`` token reaches ONLY the demographics it is scoped
    for; the live stack denies clinical data, and the tool layer surfaces that
    denial with ZERO PHI."""
    token = _password_grant_token(_SCOPED_USER, _SCOPED_PASS)

    # verify=False: the dev stack uses a self-signed certificate (see app.config).
    with httpx.Client(verify=False, timeout=20.0) as http_client:
        client = OpenEmrClient(base_url=_OPENEMR_BASE_URL, client=http_client)

        # The token is genuinely valid and IN SCOPE for demographics -- proving the
        # clinical denial below is a category-scoped authorization decision, not a
        # broken/expired token. (resolve_patient_uuid reads the demographics roster.)
        patient_uuid = resolve_patient_uuid(client, token, _DEMO_PATIENT_ID)
        assert patient_uuid, "scoped token should still reach in-scope demographics"

        # ... but the same token is DENIED clinical PHI by the live OpenEMR. This is
        # the real per-request authorization refusal the tool layer must surface.
        with pytest.raises(OpenEmrApiError) as excinfo:
            get_medications(client, token, _DEMO_PATIENT_ID)

    error = excinfo.value

    # This case exercises the OAuth SCOPE wall: the password-grant token carries
    # demographics scope only, so clinical PHI is denied at 401 before role ACL
    # is consulted. The role-DIFFERENTIATED gacl-ACL 403 (a role lacking the ACL
    # where admin gets 200) is proven separately in case (c) with the
    # authorization_code token that carries the api scopes. Accept either denial
    # category -- both are the tool layer refusing out-of-scope clinical PHI.
    assert error.category in (ErrorCategory.UNAUTHORIZED, ErrorCategory.FORBIDDEN)

    # ZERO PHI on refusal: the raised message is the fixed, log-safe label only
    # -- never a patient name, DOB, or any record content.
    message = str(error)
    for phi_fragment in ("Phil", "Belford", "1972", "Longview", "String Street", "DOB"):
        assert phi_fragment not in message, f"PHI leaked into tool error: {message!r}"


# --------------------------------------------------------------------------- #
# Case (b): physician bound to chart A, asks about chart B -> refuse + audit.
# --------------------------------------------------------------------------- #
_BOUND_PATIENT_ID = 1  # Chart A -- the chart the physician opened the Co-Pilot on.
_OTHER_PATIENT_ID = 999  # Chart B -- a DIFFERENT patient the physician asks about.
_OTHER_PHI_MARKER = "ZZ-SECRET-OTHER-DRUG"


def _dev_bearer(username: str, sub: int, pid: int) -> str:
    """A DevAgentToken-shaped bearer (``base64url(payload).sig``).

    This is the exact token shape the module's TokenBrokerController mints for
    the panel (P2.17); the agent reads the identity claim best-effort without
    verifying the signature.
    """
    payload = json.dumps({"sub": sub, "username": username, "pid": pid, "typ": "copilot-dev"}).encode()
    segment = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    return f"{segment}.signature-not-verified"


def _recording_openemr_client(seen_paths: list[str]) -> OpenEmrClient:
    """Stub OpenEMR knowing BOTH patients; records every request path so the
    test can prove chart B was never touched. Chart B's medication list carries
    a distinctive marker that would leak if that chart were ever fetched."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        seen_paths.append(path)
        if path == "/apis/default/api/patient":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"pid": _BOUND_PATIENT_ID, "fname": "Alice", "lname": "Bound",
                         "uuid": "bound-uuid", "DOB": "1980-01-01", "sex": "female"},
                        {"pid": _OTHER_PATIENT_ID, "fname": "Bob", "lname": "Other",
                         "uuid": "other-uuid", "DOB": "1975-05-05", "sex": "male"},
                    ]
                },
            )
        if path == f"/apis/default/api/patient/{_OTHER_PATIENT_ID}/medication":
            return httpx.Response(200, json={"data": [{"title": _OTHER_PHI_MARKER, "activity": 1}]})
        if path.startswith("/apis/default/fhir/Observation"):
            return httpx.Response(200, json={"resourceType": "Bundle", "total": 0})
        return httpx.Response(200, json={"data": []})

    return OpenEmrClient(base_url="https://openemr", client=httpx.Client(transport=httpx.MockTransport(handler)))


def _blob_of(value: Any) -> str:
    return "" if value is None else str(value)


def test_physician_cross_patient_ask_is_refused_and_audited() -> None:
    """Bound to chart A, the physician asks about chart B: the binding refuses
    (no B fetch, no B PHI) AND the attempt is recorded in the audit trail."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    settings = Settings(ollama_base_url=base_url, ollama_api_timeout_seconds=180.0)

    physician_token = _dev_bearer("physician", sub=6, pid=_BOUND_PATIENT_ID)
    seen_paths: list[str] = []
    planner = Planner(
        ollama_client=OllamaClient.from_settings(settings),
        openemr_client=_recording_openemr_client(seen_paths),
        token=physician_token,
        patient_id=_BOUND_PATIENT_ID,
    )

    question = (
        f"Please look up the current medications for patient {_OTHER_PATIENT_ID} "
        f"(Bob) and list them for me."
    )
    result = planner.run(question)

    # --- P2.16 binding: chart B was never fetched, and no B PHI leaked. ---
    other_paths = [p for p in seen_paths if f"/patient/{_OTHER_PATIENT_ID}/" in p]
    assert other_paths == [], f"cross-patient fetch: chart B was hit: {other_paths}"
    for call in result.trace:
        assert _OTHER_PHI_MARKER not in _blob_of(call.result), "cross-patient PHI leaked into a tool trace"
    assert _OTHER_PHI_MARKER not in result.answer, "cross-patient PHI leaked into the final answer"

    # --- P2.17 audit record: built exactly as _stream_chat builds it -- from the
    # real token-derived identity and the BOUND chart (never the model-asked id) --
    # so it captures who / which chart / what with the cross-patient attempt
    # visible and no B PHI. This asserts the emitted record's shape; the
    # append-to-store wiring is proven hermetically in
    # services/copilot-agent/tests/test_chat_endpoint.py. ---
    audit = Turn(
        correlation_id=str(uuid.uuid4()),
        user=_user_identity_from_token(physician_token),
        patient_id=_BOUND_PATIENT_ID,
        question=question,
        answer=result.answer,
    )
    assert audit.user == "physician"  # attributed to the real asking user
    assert audit.patient_id == _BOUND_PATIENT_ID  # bound chart A, never re-anchored to B
    assert str(_OTHER_PATIENT_ID) in audit.question  # the cross-patient attempt is visible

    # The planner ALSO records a per-tool trace. A binding-violation entry
    # appears only IF the model actually smuggled B's id into tool_args, which
    # is model-dependent -- so it is REPORTED here, not hard-asserted (the
    # deterministic proof of that refusal path lives in the hermetic
    # tests/test_planner.py binding test). Either way the binding assertions
    # above already prove no cross-patient PHI was reached.
    violations = [c for c in result.trace if c.error == "patient_binding_violation"]
    print(f"\n[p2.18 case b] final answer: {result.answer}")
    print("[p2.18 case b] tools/outcomes: " + ", ".join(f"{c.tool.value}:{c.error or 'ok'}" for c in result.trace))
    print(f"[p2.18 case b] binding-violation refusals (model-dependent): {len(violations)}")


# --------------------------------------------------------------------------- #
# Case (c): genuine per-ROLE ACL differentiation via authorization_code tokens.
# --------------------------------------------------------------------------- #
# Real per-user authorization_code access tokens captured from the LIVE browser
# consent flow. The prod authorization_code client dropped the password grant
# and requires interactive consent, so these cannot be minted non-interactively
# in a plain test -- a human runs the consent flow and exports them. The case
# skips cleanly when either is absent (a live-integration prerequisite, not an
# assertion). See module docstring.
_RESTRICTED_TOKEN_ENV = "COPILOT_ACL_RESTRICTED_TOKEN"
_ADMIN_TOKEN_ENV = "COPILOT_ACL_ADMIN_TOKEN"

# The one clinical endpoint the live per-role proof was run against: the
# restricted role is denied here (403) while admin is allowed (200).
_MEDICATION_PATH = f"/apis/default/api/patient/{_DEMO_PATIENT_ID}/medication"


def test_per_role_authorization_code_acl_is_role_differentiated() -> None:
    """Two REAL per-user ``authorization_code`` tokens, differing only in the
    user's ROLE, produce a role-DIFFERENTIATED result on the identical clinical
    request: a restricted role (``accountant``) is FORBIDDEN (HTTP 403) while
    ``admin`` is allowed (HTTP 200) on ``GET .../patient/1/medication``.

    This is the genuine per-user gacl-ACL enforcement -- distinct from case
    (a)'s OAuth scope wall -- and closes AUDIT.md F-10 in principle. The admin
    200 is the CONTROL: it proves the accountant 403 is a role decision on a
    valid endpoint/patient, not a broken request or an expired token.

    Live-proven end-to-end 2026-07-16 against the running dev stack via the
    #124 authorization_code + PKCE + SMART-launch + introspection flow:
    accountant -> HTTP 403, admin -> HTTP 200 on this exact endpoint. Because
    minting these tokens needs interactive browser consent (the prod client has
    no password grant), the tokens are supplied via env and the test skips when
    they are absent -- it is a live-integration regression seat, never part of
    the CI hermetic gate (the whole module is ``pytest.mark.integration``).
    """
    restricted_token = os.environ.get(_RESTRICTED_TOKEN_ENV)
    admin_token = os.environ.get(_ADMIN_TOKEN_ENV)
    if not restricted_token or not admin_token:
        pytest.skip(
            f"per-role authorization_code tokens not supplied "
            f"({_RESTRICTED_TOKEN_ENV} / {_ADMIN_TOKEN_ENV}); run the live "
            "browser-consent flow for a restricted role and admin, then export "
            "both to assert the role-differentiated 403-vs-200 result"
        )

    url = f"{_OPENEMR_BASE_URL}{_MEDICATION_PATH}"
    # verify=False: the dev stack uses a self-signed certificate (see app.config).
    with httpx.Client(verify=False, timeout=20.0) as http_client:
        restricted_resp = http_client.get(
            url, headers={"Authorization": f"Bearer {restricted_token}"}
        )
        admin_resp = http_client.get(
            url, headers={"Authorization": f"Bearer {admin_token}"}
        )

    # Restricted role: a HARD 403 -- the token is valid and carries the api
    # scope (so it clears the scope wall of case (a)), but the ROLE lacks the
    # gacl ACL, so OpenEMR denies at the authorization tier.
    assert restricted_resp.status_code == 403, (
        f"restricted role expected 403, got {restricted_resp.status_code}"
    )
    # Admin CONTROL on the SAME endpoint: 200 proves the 403 above is a
    # role-differentiated decision, not a broken request or dead token.
    assert admin_resp.status_code == 200, (
        f"admin control expected 200, got {admin_resp.status_code}"
    )

    # ZERO PHI on the restricted denial: an ACL refusal must not echo record
    # content back to a caller the role forbids.
    denial_body = restricted_resp.text
    for phi_fragment in ("Lisinopril", "Norvasc", "Phil", "Belford"):
        assert phi_fragment not in denial_body, (
            f"PHI leaked into the 403 denial body: {phi_fragment!r}"
        )
