# Clinical Co-Pilot — Security Audit

- **Status:** Finalized (Phase 5, P5.5). This document reports the security posture of the OpenEMR base the Clinical Co-Pilot is built on, plus how the Co-Pilot design responds to the gaps it found. The findings were established in the Phase 1 baseline audit and cross-checked against the in-source acknowledgements (in-code `@TODO` markers and tracked upstream issue references) noted per finding. Phase 6 live end-to-end verification of the #124 per-user OAuth flow later appended two Co-Pilot *integration* findings (F4, F5) in a dedicated section below; the base-platform audit itself is unchanged.
- **Target:** OpenEMR 8.2-dev base (the version vendored into this repo), examined on a running local dev stack.
- **Method:** candidate findings from static code analysis, each verified against source **and** a running instance before it appears here. Findings that failed verification were dropped.

## Executive summary

The Clinical Co-Pilot embeds an AI agent inside OpenEMR that reads patient data through OpenEMR's own APIs. Its trust model therefore inherits OpenEMR's security posture, so Phase 1 audited that base before any agent code was written. The goal was not to produce a long vulnerability list; it was to establish, with evidence, exactly which platform controls the Co-Pilot can lean on and which gaps it must compensate for in its own layer.

The headline is a split verdict. The OpenEMR base has genuinely strong **integrity and encryption primitives**: a SHA3-512 tamper-checksummed audit trail, AES-256 field and document encryption through the `CryptoGen` service, and a breakglass emergency-access path. These are real, working controls and this audit does not undermine them. The gaps are specific and localized: a handful of **PHI-bearing paths bypass those primitives** — audit and API-log tables that store patient data outside the encryption applied everywhere else, and an authorization tier that the interactive UI enforces but the REST/FHIR read path skips. The base is not broadly insecure; it has sharp, nameable edges, and the Co-Pilot's design is shaped around them.

The audit began with ten candidate findings drawn from code analysis. Each was then checked against source and reproduced (or not) on a live instance. **Seven were confirmed, one was confirmed only in part, and two did not reproduce and were dropped.** The two debunked candidates are reported here on purpose: a candidate that survives verification is worth more than one merely asserted, and showing the ones that failed is what makes the surviving findings credible. One candidate ("no rate limiting on login") was contradicted outright — OpenEMR ships a per-username and per-IP failed-login lockout, enabled by default. Another ("stale ACL cache") does not reproduce because the relevant cache is off by default.

The confirmed findings cluster into three themes. **Confidentiality-at-rest gaps:** `api_log` stores full request/response bodies (PHI) unencrypted, and `log.comments` is base64-encoded rather than encrypted — encoding is not encryption. **Authorization-consistency gaps:** the sensitivity ACL that hides high-sensitivity encounters in the UI is not enforced on the API/FHIR read path, and API scopes are role-level, not patient-granular, at the central layer. **Perimeter and deployment hardening:** CORS reflects any origin, the core session cookie omits `HttpOnly` and `Secure`, and the dev stack ships default credentials. Several of these are already acknowledged upstream — the CORS reflect and the authorization pipeline carry in-code `@TODO` markers, and the `log.comments` encryption was a deliberate, tracked upstream removal — which is noted per finding so severity reflects intent, not surprise.

Two of the sharpest findings — the sensitivity-ACL read bypass and the plaintext PHI in `api_log` — are described here at the level of vulnerability class, impact, affected code path, and remediation. Step-by-step reproduction detail is deliberately omitted: this is a constructive security analysis, not an exploit guide, and the sharp findings are reported at class-and-remediation altitude rather than as a weaponized how-to. Every finding is cited to a `file:line` code path so the analysis is checkable; the tone throughout is constructive security engineering, not an exploit drop.

## How to read this document

Findings are grouped by severity. Each carries a code-path citation so any claim can be traced. Where a finding shaped the Co-Pilot's own architecture, that is called out under **Co-Pilot response**. Line numbers reference the OpenEMR 8.2-dev tree vendored in this repository.

---

## Confirmed findings

### F-2 — `api_log` stores full request/response bodies (PHI) unencrypted at rest

- **Severity:** High
- **Class:** Sensitive data stored in cleartext.

**What it is.** OpenEMR's API request logger persists the full serialized response body — and stores the same content as the request body — into the `api_log` table with no encryption. On `KernelEvents::TERMINATE`, `ApiResponseLoggerListener` captures `$response->getContent()` and hands it to the log sink (`src/RestControllers/Subscriber/ApiResponseLoggerListener.php:62-103`), which inserts it verbatim: `LogTablesSink.php:98` performs a plain `insert('api_log', ...)` with no call to `CryptoGen`, `encryptStandard`, or even base64. Notably, the sink's own class docblock (`LogTablesSink.php:30-32`) describes an encrypted-storage path — "if `$crypto` is set, the api body data will be stored encrypted" — but the constructor takes only a `Connection` and no `$crypto` argument, so that path was never wired up. The exposure is a regression against the module's own intended design, not merely an omission. Full-body logging is the shipped default: `api_log_option` defaults to `'2'` ("Full Logging") in `library/globals.inc.php:2893-2903`.

**Impact (PHI / HIPAA).** Any FHIR or REST response containing PHI is written in the clear to `api_log`. Anyone with database read access — a DBA, a backup file, a leaked DB credential, or SQL injection elsewhere in the stack — can read full patient records without touching the application or its encryption keys. This defeats the AES-256-at-rest posture `CryptoGen` provides for documents and fields. Each row co-locates `patient_id`, `ip_address`, and the PHI body, raising the sensitivity of any single leaked row.

**Remediation.** Wire the `$crypto` dependency the docblock already anticipates: inject `CryptoGen` into `LogTablesSink` and store the bodies via `encryptStandard()` with a decryptable marker, mirroring the pattern the audit-log encryption column was built for. Additionally or alternatively, default `api_log_option` to `1` (minimal, bodies blanked) so full bodies are captured only on explicit opt-in, and document the PHI exposure of option `2`. InnoDB tablespace encryption is worthwhile defense-in-depth but is not a substitute for application-layer encryption of this column.

> **Disclosure altitude.** This finding was reproduced against a running instance, confirming the stored bodies are directly readable. Specific reproduction steps are deliberately omitted; the finding is reported at the level of class, impact, affected code path, and remediation rather than as a step-by-step exploit.

---

### F-9 — Sensitivity ACL not enforced on API / FHIR encounter read paths

- **Severity:** High
- **Class:** Broken access control (authorization inconsistency between UI and API).

**What it is.** OpenEMR gates visibility of high-sensitivity encounters through the `sensitivities` ACL (e.g. `sensitivities/high`). That gate is enforced on interactive paths — the encounter-history UI checks `AclMain::aclCheckCore('sensitivities', ...)` before rendering each row (`interface/patient_file/history/encounters.php:506-507`), and the encounter **write** path enforces it in `EncounterService::updateEncounter()` (`src/Services/EncounterService.php:449-452`). The central **read** query does not. `EncounterService::search()` (`src/Services/EncounterService.php:152`) selects the `sensitivity` column but never gates on it, and every read accessor (`getEncounterById`, `getEncounter`, `getOneByPidEid`) funnels through that unguarded method. The REST controller (`src/RestControllers/EncounterRestController.php:379,424`) and the FHIR encounter service and controller add no sensitivity check of their own (the FHIR path contains no `aclCheck` reference at all). The result: the API and FHIR read paths return high-sensitivity encounters to any caller who clears the coarse resource/patient authorization, regardless of whether that principal holds the sensitivity ACL the UI requires.

**Impact (PHI / HIPAA).** A user or OAuth client with general encounter-read scope but *without* the `sensitivities/high` ACL can retrieve, via REST or FHIR, encounters the interactive UI would hide. This is a horizontal information-disclosure gap specific to the sensitivity tier. Patient-level scoping still applies, so it is not an unbounded read — it is a bypass of the *sensitivity classification*, not of patient-level access.

**Remediation.** Enforce the sensitivity ACL at the service read boundary so every consumer inherits it. After `EncounterService::search()` fetches rows, drop or deny any row whose `sensitivity` fails `AclMain::aclCheckCore('sensitivities', $row['sensitivity'])`, mirroring the UI logic and the existing write-path check. Applying it in `search()` covers REST, FHIR, and internal callers in one place. Because `AclMain` resolves the principal from the session, confirm the ACL principal is correctly resolved in the token context before relying on it on API/FHIR requests.

> **Disclosure altitude.** This finding was verified on a running instance. The exact request sequence that demonstrates the bypass is deliberately omitted; the finding is reported at the level of class, impact, and remediation. The affected code path is named above so the fix can proceed.

---

### F-1 — CORS listener reflects any `Origin` without an allowlist

- **Severity:** Medium
- **Class:** Permissive cross-origin resource sharing.

**What it is.** `src/RestControllers/Subscriber/CORSListener.php` echoes the request `Origin` header straight into `Access-Control-Allow-Origin` with no allowlist check. In `onKernelResponse()` the caller-supplied origin is written unconditionally (`CORSListener.php:57`), and the OPTIONS preflight path both reflects the origin and sets `Access-Control-Allow-Credentials: true` (`CORSListener.php:67,73`). The gap is acknowledged in-code by a `@TODO` at `CORSListener.php:55`. Live probing confirmed the reflection fires on 200, 401, and 404 responses alike — it is applied blanket in `onKernelResponse`, independent of route match or auth outcome.

**Impact.** Reflecting an arbitrary origin defeats the purpose of a CORS allowlist: any web origin is told it may read API/FHIR responses in a browser. The impact is meaningfully **bounded** by two existing controls, however. The API uses Bearer-token auth, and browsers do not auto-attach Bearer tokens cross-origin, so a malicious page cannot ride a victim's ambient credentials the way it could with cookie auth. And the API session cookie is `HttpOnly; Secure; SameSite=Strict`, while the actual data responses set `Access-Control-Allow-Origin` but not `Access-Control-Allow-Credentials`, so browsers refuse credentialed cross-origin reads of them. The dangerous credentials-plus-reflection combination exists in the preflight source but was not observed on the wire in these probes. Net: a real hardening gap and defense-in-depth concern, not a turn-key exfiltration hole given token auth plus `SameSite=Strict`.

**Remediation.** Replace the unconditional reflect with an allowlist check driven by a configured set of trusted origins; echo the request origin only if it is a member, otherwise omit the header. Never combine `Access-Control-Allow-Credentials: true` with a wildcard or reflected origin. (A separate pre-existing bug was noted in passing: `CORSListener.php:69` uses a comma where a `=>` was intended, so the `Access-Control-Allow-Methods` header is never emitted on preflight.)

**Co-Pilot response.** The Co-Pilot agent does not rely on browser CORS for its own security. It runs local-only with no outbound egress and authenticates to OpenEMR with the logged-in user's bearer token rather than ambient cookies — the same properties that already bound this finding's impact. The agent adds no new credentialed cross-origin surface.

---

### F-3 — `log.comments` is base64-encoded, not encrypted, while holding PHI

- **Severity:** Medium
- **Class:** Encoding mistaken for encryption; sensitive data effectively in cleartext.

**What it is.** The audit-log `comments` column is stored base64-encoded — a reversible encoding with no secret — not encrypted. `EventAuditLogger.php:660-664` calls `base64_encode($comments)`, and the surrounding comment states the encryption path was deliberately removed (referencing upstream changes `#12118`/`#12120`). The paired `log_comment_encrypt` row is inserted with `'encrypt' => 'No'` hardcoded (`LogTablesSink.php:87-94`); only the SHA3-512 `checksum` and a version are stored. That checksum is a genuine **integrity** control — it makes tampering detectable — but it provides no **confidentiality**. Meanwhile the comments demonstrably contain PHI: for DML audit events, `$comments` is set to the full SQL statement plus every bound value (`EventAuditLogger.php:446-451`), so an insert or update into a patient table logs the patient's name, DOB, SSN, or diagnosis as bound values.

**Impact (PHI / HIPAA).** `log.comments` is effectively plaintext PHI, because base64 is trivially reversible with no key. Database-read access yields the audit narrative including patient-data values embedded in logged SQL — the same exposure class as F-2, via a different table. The risk is that the SHA3-512 checksum's presence can be mistaken for a confidentiality control it does not provide.

**Remediation.** Restore the removed encryption path: encrypt `comments` (and `user_notes`) with `CryptoGen::encryptStandard()`, set `log_comment_encrypt.encrypt = 'Yes'`, bump the version, and decrypt on the read/logview path when the flag is `Yes`. The schema was clearly built to support exactly this and is currently unused. If full re-encryption is out of scope, at minimum stop embedding raw bound PHI values in `comments` for patient tables (log column names and row ids, not values). Keep base64 as an inner transport for binary safety *under* encryption, not as the outer representation. This is a known-tradeoff regression rather than an oversight — the encryption was intentionally dropped upstream and is tracked there.

---

### F-5 — Core session cookie omits `HttpOnly` (and `Secure`)

- **Severity:** Medium
- **Class:** Session cookie hardening.

**What it is.** The core interactive OpenEMR session cookie is configured with `HttpOnly` off. `SessionConfigurationBuilder::forCore()` explicitly calls `->setCookieHttpOnly(false)` (`src/Common/Session/SessionConfigurationBuilder.php:83-91`), overriding the builder's own secure default; every other preset (`forOAuth`, `forApi`, `forPortal`, `forSetup`) keeps `HttpOnly` on. The `forCore()` path is what feeds the interactive browser session (`src/Common/Http/HttpSessionFactory.php:72`). The rationale is documented (`src/Common/Session/SessionUtil.php:8-17`): JavaScript must read the cookie to support the "separate patient logins in separate windows" feature via a custom `restore_session()`. Live `Set-Cookie` inspection confirmed the core `OpenEMR` cookie carries `SameSite=Strict` but neither `HttpOnly` nor `Secure`, even when served over HTTPS.

**Impact.** Any stored or reflected XSS in the authenticated core UI can exfiltrate the session ID via `document.cookie` and hijack a clinical user's session. The missing `Secure` flag additionally allows the cookie to leak over any plaintext HTTP request to the host.

**Remediation.** This is an upstream design tradeoff, not a Co-Pilot regression, so remediation is a hardening recommendation. The multi-login `restore_session()` mechanism blocks simply flipping `HttpOnly` on; a real fix moves the session-swap out of JS-readable cookies (a server-side handoff keyed by a short-lived, `HttpOnly` token). Independently and cleanly, `cookie_secure` should be `true` whenever the site is served over HTTPS — add `->setCookieSecure(true)` to `forCore()` gated on TLS, which costs nothing behaviorally and closes the plaintext-leak vector. A strict CSP is worthwhile defense-in-depth to shrink the XSS surface this gap amplifies.

---

### F-10 — API scopes are role-level, not patient-granular, at the central ACL layer

- **Severity:** Informational (platform baseline, by design).
- **Class:** Coarse-grained authorization; no central patient-context binding.

**What it is.** The central REST/FHIR authorization check is role + resource + permission, not patient-granular. `AuthorizationListener::onRestApiSecurityCheck` builds a scope string such as `user/Patient.read` and checks only that the token carries it (`src/RestControllers/Subscriber/AuthorizationListener.php:182-193`, matched by `HttpRestRequest.php:373`) — there is no patient identifier in the comparison. Patient-role tokens *are* self-bound to their own pid (a real, if narrow, central control), and a SMART `launch/patient` scope binds a provider token to one patient. But a plain provider `user/*` token receives no patient constraint at the central layer, and even when a launch binding is applied, the provider-to-patient gate is a stub that always returns `true` (`BearerTokenAuthorizationStrategy.php:479-485`). The underlying ACL is category-based (`AclMain::aclCheckCore('admin', 'users')` style), never "is this specific patient in this provider's panel."

**Impact.** A valid provider token with `user/<Resource>.read` scope can read *any* patient's records; access is gated by role and resource type, not by a provider-to-patient relationship. This is standard OpenEMR behavior — its trust model treats an authenticated provider as authorized to the whole clinic — so it is a platform baseline, not a bug to patch in OpenEMR core. The consequence for this project is concrete: **the base platform provides no central patient-context binding for the Co-Pilot to rely on.**

**Co-Pilot response.** This finding directly substantiates the Co-Pilot's patient-context-binding control (implementation plan §4.2). Because role ACLs alone are not patient-granular, the agent enforces minimum-necessary scope itself: every conversation is anchored to the `pid` the panel was opened on, and the tool layer refuses any request for a different patient id. This is defense-in-depth narrowing layered on top of OpenEMR's role enforcement, not a second RBAC — role enforcement stays in OpenEMR. The documented production path (ARCHITECTURE.md) is SMART patient-context tokens so the token itself carries the patient boundary, replacing reliance on the always-`true` core stub with a real boundary at the Co-Pilot layer. As of #124 Phase 6 that per-user path — `authorization_code` + PKCE + SMART launch + introspection — is built and **proven live end-to-end** (restricted role 403 vs admin 200 on the same endpoint), behind `copilot_per_user_token_enabled` (default OFF); see Co-Pilot integration finding **F4** below.

---

### F-6 — Dev stack ships default credentials

- **Severity:** Informational (deployment hardening, not a code vulnerability).
- **Class:** Weak default credentials in dev configuration.

**What it is.** The development docker-compose bakes in default credentials: `MYSQL_ROOT_PASSWORD: root`, `OE_USER: admin`, `OE_PASS: pass`, `MYSQL_PASS: openemr` (`docker/development-easy/docker-compose.yml:14,65-67`). These match the documented dev login and are accepted by the live dev stack.

**Impact.** None in the intended local-dev use. The risk is entirely one of **provenance and deployment discipline**: if this compose file, its env values, or an image built with them is ever promoted toward an internet-reachable or shared environment, it is trivially compromised (default admin plus default DB root).

**Remediation.** Frame as a documented hardening boundary, not a bug. Keep the dev defaults, but add an explicit note (and ideally a startup guard) that this compose file is dev-only and must never back a deployable image. Any non-local deployment path must require secrets via an env/secret store, force a first-boot admin password change, and never set `MYSQL_ROOT_PASSWORD` to a literal in committed YAML. This is explicitly **not** a vulnerability in the application code.

---

### F-8 — No enforced / automated audit-log retention policy (Partial)

- **Severity:** Low
- **Class:** Missing governed retention control. **Verdict: Partial** — a manual control exists; the automated one does not.

**What it is.** The `log`, `api_log`, and `log_comment_encrypt` tables grow unbounded with no scheduled retention or purge. A repo-wide search finds no cron job, scheduled task, or background service that trims them. What *does* exist is a manual, admin-invoked archive-and-delete: the "Backup/Delete Log Data" admin screen (`interface/main/backup.php:1032-1067`) exports `log` to a zipped CSV and runs a join-delete across all three tables for entries on or before an operator-chosen end date, followed by `OPTIMIZE TABLE`. The suggested end date ("end of year, two years ago") is only a pre-filled form value, not an enforced or scheduled policy. So the original "no policy at all" framing overstates it — hence the Partial verdict — but the real gap stands: there is no *enforced, automated, scheduled* retention; purging depends on an admin remembering to run the tool.

**Impact (HIPAA).** Unbounded accumulation of PHI-bearing audit rows (compounded by F-2 and F-3) enlarges the breach blast radius over time and complicates data-minimization obligations. Honest counterpoint: HIPAA also *requires* audit logs be retained for roughly six years, so unbounded-by-default is partly a conservative safety posture. The genuine gap is the absence of a *governed, configurable* retention control, not that logs should be deleted sooner.

**Remediation.** Add a configurable, scheduled retention job that respects the HIPAA minimum retention window and archives-then-purges beyond a policy horizon — reusing the existing join-delete plus `OPTIMIZE` logic, driven by a global setting and a background service rather than a manual click. Surface last-run and next-run in the admin UI.

---

## Dismissed candidates

Two candidates from the initial list did not survive verification and are **not** findings. They are recorded here because showing the failures is what makes the confirmed findings credible.

- **Rate limiting / brute-force lockout on login — NOT a finding.** The premise (unlimited unthrottled guessing) is contradicted by the code. `src/Common/Auth/AuthUtils.php` implements a two-tier failed-login counter — per-username and per-IP — wired into both interactive login (`library/auth.inc.php:62`) and the OAuth password grant (`UserRepository.php:92`), and it ships enabled with non-zero defaults (`password_max_failed_logins` = 20, `ip_max_failed_logins` = 100 in `globals.inc.php`). Genuine residual nuances exist (counter-based rather than backoff-based; a distributed-IP attacker dilutes the per-IP tier; a targeted account-lockout DoS is possible), but the finding as originally worded is incorrect and was dropped.
- **ACL cache staleness on role change — NOT reproduced.** The phpGACL `Cache_Lite` result cache is disabled by default (`src/Gacl/Gacl.php:68`, `_caching = FALSE`; `AclMain` constructs `Gacl` with no options), so its read/write helpers are hard no-ops and every `acl_query()` hits a live DB read. The other caching layer holds only the DB connection object for one PHP request, not query results, and is torn down at end of request. Under shipped defaults, a permission change is reflected on the very next request. The stale-permission risk is latent only under a non-default `caching => true` and is not an exploitable default-config issue.

---

## Co-Pilot integration findings (Phase 6 live verification)

The findings above audit the OpenEMR **base**. The two below are the Co-Pilot project's **own** findings — the `F<n>` series tracked in `prd/DECISIONS.md`, distinct from the base `F-<n>` series above (note the hyphen: base `F-5` is the session cookie; project `F5` is the live-E2E defect here). Both were established by the #124 Phase 6 **live end-to-end** verification against the running stack, and both bear directly on this audit's central authorization claim.

### F4 — Per-user ACL enforcement: PROVEN LIVE, flag-gated, default OFF

- **Status:** Capability built and proven live end-to-end; **open in practice by owner decision** (the default flag is OFF).
- **Relation to the base audit:** the agent-layer counterpart to base finding **F-10** (role-level, not patient-granular, central ACL). F-10 is a property of the OpenEMR base; F4 is whether *this agent* exercises OpenEMR's per-user ACL end-to-end at all.

**What it was.** Through Phase 2b the agent reached OpenEMR via a dev token bridge: the agent itself held one shared demo-clinician password-grant token and used it for every user's tool calls, so OpenEMR's ACL always saw the same identity. Per-user ACL was therefore *simulated* by the agent's patient-context binding, not *exercised* by OpenEMR. A live 5-user × 9-endpoint matrix (P2.18) confirmed the dev password grant never even reaches the role-ACL tier: OpenEMR strips the `api:oemr` / `api:fhir` scopes from a ROPC token for every role, so all clinical endpoints return a uniform 401 at the OAuth **scope wall** before any role ACL is consulted — no reachable case where a scoped role gets 403 while admin gets 200.

**What changed (built + proven live).** The #124 `authorization_code` + PKCE + SMART-launch flow is now implemented and verified end-to-end against the running stack (2026-07-16): a real OpenEMR consent issues a *per-user* token, stored **encrypted** (CryptoGen AES-256-GCM, `007`-prefixed ciphertext, plaintext absent at rest, decrypts back); the agent **introspects** that token (`active:true`, RFC 7662, client_secret_post) and forwards it — not a shared credential — on every tool call. With the per-user token now carrying the api scopes, OpenEMR's own `gacl` ACL is finally the enforcement point, and the role-differentiated result the whole trust boundary rests on is real: on the identical `GET /apis/default/api/patient/1/medication`, a restricted role (`accountant`) is denied **HTTP 403** while `admin` gets **HTTP 200**. This supersedes **#127** — the agent validates a real OpenEMR token via introspection rather than the dev HMAC `DevAgentToken`.

**Why it is still open in practice.** The capability is gated behind `copilot_per_user_token_enabled`, which **defaults OFF** — a deliberate owner decision to keep the demo's out-of-box UX free of a per-user consent step (the dev bridge stays the default, gated-off local fallback). So the per-user ACL is *genuinely enforceable and proven live*, closeable by a documented **one-line flag flip** the owner controls — not closed today, and not because the capability is missing, but because the default is off by choice. Framed precisely: **proven live, flag-gated, default off** — not "F4 closed."

**To close.** Flip `copilot_per_user_token_enabled` on (and complete the documented prod-client-creds → module-globals wiring), accepting the per-user consent step in the demo UX. Nothing else in the flow needs to change; the path is built and exercised.

### F5 — Live-E2E-caught external-contract bugs in the server-side OAuth exchange (RESOLVED, #190)

- **Status:** Resolved (#190).
- **Class:** External-contract mismatch invisible to isolated / mocked tests.

**What it was.** The first time the browser-consent → server-side token-exchange chain ran together live (Phase 6), three external-contract bugs surfaced that every isolated and mocked test had passed straight through:

1. **Server-side token-exchange URL.** The module's `authorization_code` → token exchange POSTed to the ServerConfig-derived **browser** origin (`localhost:9300`), which is unreachable from inside the openemr container (apache listens on 443 internally; 9300 is only a host port map) — the exchange failed, no token was stored, the callback 400'd. The `redirect_uri` is legitimately pinned to the browser-facing `localhost:9300` (it must match the registered value), so this was not fixable by dev config alone. The **agent** side already solved this exact internal-vs-browser split (`https://openemr` for server-to-server vs a browser-facing `redirect_uri`); the module did not.
2. **Authorize-leg `aud` / audience validation** on the browser authorize request.
3. **Introspection auth method.** The agent must present its client credentials via **client_secret_post** (form body), not HTTP Basic — OpenEMR's introspection endpoint ignores Basic creds there and returns a spurious `active:false`.

All three were fixed in **#190** (internal token URL decoupled from the browser-facing authorize / redirect / `aud`; introspection switched to client_secret_post), and the full flow then re-ran green live.

**The lesson.** The isolated tests and the pre-merge gates mocked OpenEMR's OAuth and introspection endpoints, so they validated the agent and module against a *model* of OpenEMR's contract — and passed while the real external contract was violated three ways. Live end-to-end verification caught what no hermetic test could, precisely because the bugs lived at the boundary the mocks stood in for. This is the same pattern as project findings **F1** (container-boot `httpx` gap) and **F2** (demo-import schema downgrade): a live/integration step catching what unit tests structurally cannot. The standing mitigation is that the authorization story now carries a **live-integration regression seat** (`evals/authorization/test_acl_scoping_e2e.py`, kept out of the CI hermetic gate) that asserts the real per-role 403-vs-200 result against the running stack.

---

## Cross-cutting analysis

**Performance.** This audit did not benchmark; performance measurement is deliberately deferred to Phase 5. The relevant baseline note for agent latency is that OpenEMR's REST/FHIR read path is the Co-Pilot's data source, and every agent tool call becomes an authenticated API round-trip against it. The audit surfaced no request-path bottleneck that would block Phase 1, but the API-log write on `KernelEvents::TERMINATE` (F-2) adds a synchronous per-request insert of the full response body, which is worth including when Phase 5 profiles end-to-end agent latency.

**Architecture.** The audit doubled as an integration-point map. The security-relevant surfaces the Co-Pilot touches are now identified with `file:line` precision: the REST/FHIR authorization pipeline (`AuthorizationListener`, `BearerTokenAuthorizationStrategy`, `HttpRestRequest`), the encounter read path (`EncounterService::search()` and its REST/FHIR controllers), the audit and API-logging sinks (`EventAuditLogger`, `LogTablesSink`, `ApiResponseLoggerListener`), the session configuration (`SessionConfigurationBuilder`, `HttpSessionFactory`), and the CORS listener. This map is the concrete starting point for the agent's tool layer and for reasoning about where the Co-Pilot must add its own controls versus where it can rely on the base.

**Data quality.** The audit ran against the standard OpenEMR dev dataset, which the Co-Pilot must tolerate. The demo data is sparse and inconsistently populated — encounters carry a `sensitivity` column that is frequently unset, patient records have partial demographics, and log tables reflect only synthetic verification traffic rather than realistic clinical volume. The agent's verification layer must treat missing or empty structured fields as "absent," never invent them, and its evals must include the missing-data case explicitly rather than assuming a fully-populated chart.

**Compliance.** OpenEMR ships several HIPAA-relevant controls that are present and working: a tamper-evident audit trail (SHA3-512 checksums), AES-256 field/document encryption via `CryptoGen`, breakglass emergency access, and role-based access control. The gaps this audit confirmed are confidentiality-at-rest for two PHI-bearing log tables (F-2, F-3), an authorization-consistency gap between UI and API for the sensitivity tier (F-9), a missing governed retention control (F-8), and session/perimeter hardening items (F-1, F-5). None of these undermine the integrity or encryption primitives; they are specific paths that bypass them. The Co-Pilot's own compliance posture — local-only inference, no egress, user-token auth, patient-context binding, and full audit of every agent action — is designed so the agent does not widen any of these gaps and, for F-10, actively narrows the platform baseline.

---

## Closing note

This is the finalized audit. The findings were established in the Phase 1 baseline pass — each verified against source and reproduced (or not) on a live instance — and cross-checked against the acknowledgements already present in the OpenEMR source: the in-code `@TODO` markers on the CORS reflect and the authorization pipeline, and the tracked upstream removal referenced in the `log.comments` code (`#12118`/`#12120`). Those in-source references are noted per finding so severity reflects intent rather than surprise; no external advisory or CVE identifiers are claimed. For the two sharpest findings — the sensitivity-ACL read bypass (F-9) and plaintext PHI in `api_log` (F-2) — exploit-level reproduction specifics are deliberately omitted, and each is reported at the level of class, impact, and remediation. The findings are reported to be fixed, not weaponized: the intent is constructive security engineering that makes both the base platform and the Co-Pilot built on it more trustworthy.
