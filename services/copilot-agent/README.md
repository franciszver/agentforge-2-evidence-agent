# copilot-agent

FastAPI service for the Clinical Co-Pilot agent.

## Tests

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

## OpenEMR OAuth (dev token flow)

`app/openemr_auth.py` registers a confidential OAuth2 client and obtains a
user bearer token so the agent can call OpenEMR APIs. To verify end-to-end
against a running dev stack:

```bash
bash scripts/verify-oauth-dev.sh
```

This registers a confidential client, enables it, fetches a token via the
password grant, and calls `GET /apis/default/fhir/Patient` (expects HTTP 200;
an empty Bundle is fine — demo data is seeded in P2.0). It also proves the
bad-credential path fails cleanly.

**DEV-ONLY.** Two shortcuts here are for the local dev loop only and must not
ship to production:

- **Password grant** (username/password → token). Production uses the OAuth2
  `authorization_code` grant (per plan §4.2).
- **Enabling the client via direct SQL.** OpenEMR registers new clients
  *disabled*; production enables them through admin UI/approval.

TLS verification is off because the dev stack uses a self-signed certificate
(`openemr_verify_ssl` defaults to `False`; set it `True` where a real cert is
enforced). The dev client secret and tokens are written only to the gitignored
`services/copilot-agent/.openemr-dev-client.json` and are never printed or
committed.

## OpenEMR OAuth (production authorization_code client) — #124 Phase 1

Production does **not** use the password grant. It uses the OAuth2
`authorization_code` grant against a confidential client that a browser drives
through OpenEMR's authorize/callback. Phase 1 (this change) sets up only the
**client registration**; the authorize/callback endpoints and token brokering
land in later phases.

Register the production client (runs inside the agent container, which is the
only host that can reach OpenEMR on the internal network):

```bash
bash scripts/register-copilot-prod-client.sh
```

This posts a confidential (`application_type: private`) registration with the
`authorization_code` + `refresh_token` grants **only** (deliberately no
`password` grant — a confidential prod client must not accept the resource-owner
password grant, or a leaked `client_secret` plus any clinician credential could
mint tokens directly, bypassing the authorization_code + consent flow) and:

- **Canonical `redirect_uri`** — the single source of truth in
  `app/config.py` (`copilot_prod_client_redirect_uri`):
  `https://localhost:9300/interface/modules/custom_modules/oe-module-clinical-copilot/public/oauth-callback.php`.
  This is the **browser-facing** host and the module's one-file-per-route OAuth
  callback (Phase 2 serves it). It deliberately differs from the internal
  `openemr` docker alias used for the server-side registration call. Phase 2's
  authorize/callback must match this URL byte-for-byte — OpenEMR enforces exact
  `redirect_uri` matching.
- **SMART scopes** (`copilot_prod_client_scopes`): the SMART-launch scopes
  (`openid offline_access launch launch/patient api:oemr api:fhir fhirUser`)
  **plus** the per-resource read scopes (`user/patient.read`,
  `user/medication.read`, `user/allergy.read`, `user/medical_problem.read`,
  `user/encounter.read`, `user/appointment.read`, `user/vital.read`,
  `user/procedure.read`, `user/Observation.read` — mirroring the
  known-accepted `copilot_dev_token_scopes`). Every scope exists in OpenEMR's
  `ServerScopeListEntity::getAllSupportedScopesList()`; dynamic registration
  **rejects the whole request with `invalid_scope`** on the first unrecognized
  scope (`AuthorizationController::validateScopesAgainstServerApprovedScopes`),
  so an unknown scope breaks registration outright rather than being silently
  dropped. `user/*.read` is **not** used — OpenEMR has no wildcard scope entry.
  The read scopes are **registered here, not deferred to authorize time**:
  `ScopeRepository::finalizeScopes` only lets a token carry scopes the client
  registered with, so an unregistered read scope requested at grant time is
  silently dropped from the final token — the client would get
  `api:oemr`/`api:fhir` but no resource-read authorization, and tool calls
  would fail.

**PKCE (S256):** nothing is declared at registration. PKCE is a redirect-time
concern — the `code_challenge` travels on the Phase 2 authorize request, not in
client-registration metadata (RFC 7591) — and OpenEMR's `CustomAuthCodeGrant`
supports S256 at the grant level. Phase 2's authorize/callback will use PKCE
(S256) since the code transits a browser redirect.

### Admin approval (production) vs. the dev SQL shortcut

OpenEMR registers every new client **disabled**. The two paths to enable it:

- **Production (real step):** an OpenEMR admin approves/enables the client in
  the admin UI — **Administration → Config → Connectors → OAuth2 Clients** —
  and enables the `copilot-agent-prod` client. No database is touched by hand.
- **Dev-only shortcut:** `scripts/bootstrap-copilot-dev-client.sh` runs
  `UPDATE oauth_clients SET is_enabled=1 …` directly against the dev database.
  This is a **dev-loop convenience only** and must never be used in production.

The production client secret is written only to the container-local
`copilot_prod_client_creds_path` (default `/data/openemr-prod-client.json`) and
is never printed or committed.

### Wire the client into the module globals (dev) — #124 Phase 6

Registering + enabling the client is not enough: the OpenEMR module reads the
client id/secret and the consent-enable flag from **globals**
(`OAuthConsentConfig::fromEnvironment()`), and nothing copies the creds file into
those globals automatically. For the dev loop, run:

```bash
bash scripts/wire-copilot-prod-globals.sh
```

This reads the creds file from the agent container and upserts
`clinical_copilot_prod_client_id`, `clinical_copilot_prod_client_secret`,
`clinical_copilot_oauth_consent_enabled=1`, and `clinical_copilot_oauth_verify_ssl=0`
(dev self-signed cert). It is **dev-only** (direct SQL + TLS opt-out). In
production an admin provisions the secret out-of-band and enables the flag
deliberately.

**Server-side token endpoint (browser origin vs. internal alias).** The consent
callback runs *inside* the openemr container and performs the
`authorization_code`→token exchange server-to-server. It must NOT reuse the
browser-facing origin (`https://localhost:9300`, a host port map apache does not
listen on inside the container) — a POST there fails outright. So the exchanger
targets `OAuthConsentConfig::DEFAULT_INTERNAL_TOKEN_URL`
(`https://openemr/oauth2/default/token`, the internal docker alias the agent
already uses), overridable via the `clinical_copilot_oauth_internal_token_url`
global. The browser-facing `authorize` URL, `redirect_uri`, and SMART `aud`
(the FHIR resource base) stay on the public origin — OpenEMR validates
`redirect_uri`/client/PKCE against stored state, not the request host.

### Operational gotchas (dev stack)

- **Rebuild the agent image after merges.** The `agent` service bakes its code
  into the image (no source bind-mount). After pulling changes, recreate it or
  the container runs stale code (e.g. missing `app/prod_client_registration.py`):
  `docker compose -f docker-compose.yml -f docker-compose.copilot.yml up -d --build agent`.
- **`/data` is ephemeral.** The agent has no volume for `/data`, so every
  recreate wipes the dev/prod creds files. Re-run
  `scripts/bootstrap-copilot-dev-client.sh` (and, for the consent flow,
  `scripts/register-copilot-prod-client.sh` + `scripts/wire-copilot-prod-globals.sh`)
  after recreating the agent.

## Container

```bash
docker build -t copilot-agent:dev .
docker run -d --rm -p 8099:8000 --name copilot-agent-test copilot-agent:dev
curl http://localhost:8099/health
docker stop copilot-agent-test
```
