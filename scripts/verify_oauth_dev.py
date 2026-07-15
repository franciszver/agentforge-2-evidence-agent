"""DEV-ONLY live verification helper for the OpenEMR OAuth dev token flow.

Driven by ``scripts/verify-oauth-dev.sh``. Exercises the real
``app.openemr_auth`` functions against the running dev stack so the proof
uses the shipped code, not a parallel reimplementation.

Secrets hygiene:
  * The client secret and tokens are written ONLY to the gitignored creds
    file and are NEVER printed. Output shows presence + length, redacted.
  * The client_id is not a secret (it is not a credential on its own) and is
    printed so the caller can enable the client via SQL.

Subcommands:
  register        Register a confidential client; write creds file.
  client-id       Print the stored client_id (for the SQL enable step).
  token-and-call  Fetch a user token (password grant) and GET fhir/Patient.
  bad-password    Prove the bad-credential path fails cleanly (no leak).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

from app.config import Settings
from app.openemr_auth import (
    OpenEmrAuthError,
    authenticated_get,
    fetch_token_password_grant,
    register_client,
)

# Host-side base URL (self-signed cert in dev -> TLS verify off).
BASE_URL = os.environ.get("OPENEMR_BASE_URL", "https://localhost:9300")
CREDS_PATH = Path(__file__).resolve().parent.parent / "services" / "copilot-agent" / ".openemr-dev-client.json"
REDIRECT_URIS = ["https://localhost:9300/oauth2/default/callback"]
PROOF_PATH = "/apis/default/fhir/Patient"
DEV_USERNAME = os.environ.get("OPENEMR_DEV_USER", "admin")
DEV_PASSWORD = os.environ.get("OPENEMR_DEV_PASS", "pass")


def _client() -> httpx.Client:
    # verify=False: dev stack uses a self-signed certificate.
    return httpx.Client(verify=False, timeout=15.0)


def _redacted(value: str) -> str:
    return f"<REDACTED len={len(value)}>"


def cmd_register() -> int:
    settings = Settings()
    with _client() as client:
        creds = register_client(
            client,
            base_url=BASE_URL,
            registration_path=settings.openemr_oauth_registration_path,
            client_name="copilot-agent-dev",
            redirect_uris=REDIRECT_URIS,
            scope=settings.openemr_oauth_scopes,
        )
    CREDS_PATH.write_text(
        json.dumps(
            {
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "registration_access_token": creds.registration_access_token,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        os.chmod(CREDS_PATH, 0o600)
    except OSError:
        pass  # best-effort on Windows
    print(f"registration OK: client_id={creds.client_id} client_secret={_redacted(creds.client_secret)}")
    print(f"creds written to gitignored file: {CREDS_PATH.name}")
    return 0


def _load_creds() -> dict[str, str]:
    if not CREDS_PATH.exists():
        print("ERROR: creds file not found; run 'register' first", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(CREDS_PATH.read_text(encoding="utf-8"))


def cmd_client_id() -> int:
    print(_load_creds()["client_id"])
    return 0


def cmd_token_and_call() -> int:
    creds = _load_creds()
    settings = Settings()
    with _client() as client:
        token = fetch_token_password_grant(
            client,
            base_url=BASE_URL,
            token_path=settings.openemr_oauth_token_path,
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            username=DEV_USERNAME,
            password=DEV_PASSWORD,
            scope=settings.openemr_oauth_scopes,
        )
        print(f"token acquired: access_token={_redacted(token.access_token)}")
        print(f"  refresh_token present: {token.refresh_token is not None}")
        print(f"  id_token present: {token.id_token is not None}")
        print(f"  granted scope: {token.scope}")

        response = authenticated_get(
            client, base_url=BASE_URL, path=PROOF_PATH, token=token.access_token
        )
    print(f"GET {PROOF_PATH} -> HTTP {response.status_code}")
    if response.status_code != 200:
        print(f"FAIL: expected 200, got {response.status_code}", file=sys.stderr)
        return 1
    try:
        body = response.json()
        print(f"  body resourceType={body.get('resourceType')} total={body.get('total')}")
    except ValueError:
        pass
    print("AUTHENTICATED CALL OK")
    return 0


def cmd_bad_password() -> int:
    creds = _load_creds()
    settings = Settings()
    wrong_password = "definitely-not-the-password"
    with _client() as client:
        try:
            fetch_token_password_grant(
                client,
                base_url=BASE_URL,
                token_path=settings.openemr_oauth_token_path,
                client_id=creds["client_id"],
                client_secret=creds["client_secret"],
                username=DEV_USERNAME,
                password=wrong_password,
                scope=settings.openemr_oauth_scopes,
            )
        except OpenEmrAuthError as exc:
            message = str(exc)
            # The raised message must not leak the secret or the password.
            assert creds["client_secret"] not in message, "SECURITY: secret leaked in error"
            assert wrong_password not in message, "SECURITY: password leaked in error"
            print(f"bad-password path failed cleanly: OpenEmrAuthError: {message}")
            print("BAD-PASSWORD CLEAN-FAILURE OK (no secret/password in message)")
            return 0
    print("FAIL: bad password did not raise OpenEmrAuthError", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["register", "client-id", "token-and-call", "bad-password"],
    )
    args = parser.parse_args()
    return {
        "register": cmd_register,
        "client-id": cmd_client_id,
        "token-and-call": cmd_token_and_call,
        "bad-password": cmd_bad_password,
    }[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
