"""Application configuration loaded from environment variables.

Dev-environment default: the OpenEMR instance on the internal docker
network uses a self-signed certificate, so ``OPENEMR_VERIFY_SSL``
defaults to ``False`` here. Override it via the environment in any
deployment where certificate verification must be enforced.
"""

from __future__ import annotations

import secrets

from pydantic import Field
from pydantic_settings import BaseSettings

# Issue #167 (VULN-0004): default caps for app.chat.ConversationStore's two
# eviction dimensions (conversation count, turns retained per conversation).
# Single-sourced here -- not duplicated as a literal in app/chat.py -- so the
# Settings field default below and ConversationStore's own constructor
# default (used by every bare ``ConversationStore()`` call outside the
# FastAPI dependency, e.g. every hermetic test in this repo) can't drift
# apart. See the two ``copilot_max_*`` fields below for the value
# justifications.
DEFAULT_MAX_STORED_CONVERSATIONS = 2000
DEFAULT_MAX_TURNS_PER_CONVERSATION = 50

# Issue #174: default TTL for app.chat.RosterCache, the process-wide,
# shared cache that replaced the per-``Conversation`` ``patient_roster``
# field (see that field's former docstring, now on ``Conversation`` itself,
# for the full memory/privacy analysis). Single-sourced here for the same
# reason as the two constants above. See
# ``Settings.copilot_roster_cache_ttl_seconds``'s docstring for the value
# justification.
DEFAULT_ROSTER_CACHE_TTL_SECONDS = 300.0

# Issue #173: default cap for app.body_size_limit.BodySizeLimitMiddleware --
# the OUTERMOST-registered ASGI middleware (see app/main.py's
# create_app -- Starlette applies add_middleware in REVERSE order of
# registration, so this one is added LAST), rejecting a request body before
# it is ever buffered/JSON-parsed. This is a DIFFERENT bound than
# app.chat.MAX_CHAT_MESSAGE_LENGTH (#167, 4000 chars) and
# app.feedback.MAX_COMMENT_LENGTH (2000 chars): those are pydantic field
# bounds that only fire AFTER the whole body has already been read off the
# wire and JSON-parsed -- this one fires BEFORE any of that, both from the
# pre-parse ``Content-Length`` header (when present and truthful) and from a
# running count of actual bytes received (when the header lies or is
# absent). The two checks have different reach: the ``Content-Length``
# pre-check runs for EVERY request regardless of method or route (it never
# calls ``receive()`` itself), but the streaming counter only accumulates
# bytes a route actually reads via ``receive()`` -- a route that ignores its
# body (a plain GET handler, e.g. ``/health``) never triggers it even for an
# oversized streamed body with no ``Content-Length``, since nothing ever
# calls the wrapped ``receive`` to count. In practice this cap fully covers
# app.feedback's request body and any future body-reading POST route, not
# just /chat, since those routes DO read their body.
#
# Value, MEASURED against the real ``ChatRequest`` constants, not estimated:
# pydantic's ``max_length=4000`` on ``message`` counts Unicode CODE POINTS,
# so 4000 code points from the astral plane (e.g. U+1F600, each 1 char to
# pydantic) is the true worst case, not 4000 ASCII bytes. The sanctioned
# PHP/Guzzle caller JSON-encodes with ``ensure_ascii``-equivalent escaping
# by default, so that IS the real wire encoding: each astral code point
# becomes a ``\uXXXX\uXXXX`` surrogate-pair escape, 12 bytes. 4000 such
# code points plus a 64-char ``conversation_id`` and JSON structural
# overhead (keys, quotes, the int ``patient_id``) measures to 48,127 bytes
# -- the actual worst-case legitimate ``/chat`` body, not a ~16-32KB
# estimate. ``/feedback``'s 2000-char comment is smaller still. 64KB
# (65536 bytes) gives ~1.36x headroom over that measured 48,127-byte worst
# case (NOT "comfortably over double" -- a future editor tightening this
# cap on the assumption of 2x+ headroom would start rejecting legitimate
# 4000-char non-ASCII messages) while staying at the low end of the
# reasonable range for a service that never legitimately needs to accept
# more than a few KB -- the measured threat model (issue #173 design
# comment) is a process with a foothold on the internal-only
# ``copilot_internal`` network posting directly to this service's
# unpublished port, so there is no legitimate caller this could ever be too
# tight for.
DEFAULT_MAX_REQUEST_BODY_BYTES = 65536


class Settings(BaseSettings):
    """Runtime configuration for the copilot-agent service."""

    openemr_base_url: str = "https://openemr"
    ollama_base_url: str = "http://ollama:11434"
    trace_db_path: str = "/data/traces.db"
    # HMAC key for TraceStore.hash_args (P4.2) -- keeps tool-call args
    # non-reversible even though they are often low-entropy (patient ids,
    # closed-set filter keys, date ranges); an unkeyed hash would let an
    # attacker with read access to traces.db precompute the hash over the
    # candidate space and recover the original args. NO hardcoded default:
    # this repo is public, so any literal default here would be a published
    # key -- defeating the keying entirely. Unset => a strong random key is
    # generated per process (fail-safe). Set TRACE_ARGS_HASH_SECRET in the
    # environment to pin a stable key when args_hash must stay comparable
    # across restarts (e.g. the P4.5 review dashboard).
    trace_args_hash_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    openemr_verify_ssl: bool = False
    # Per-request timeout for calls made by ``OpenEmrClient`` (app/openemr_client.py).
    openemr_api_timeout_seconds: float = 10.0

    # Model served by the internal Ollama instance and per-request timeout /
    # retry policy for ``OllamaClient`` (app/ollama_client.py).
    ollama_model: str = "qwen3:4b"
    ollama_api_timeout_seconds: float = 60.0
    ollama_extract_max_retries: int = 2
    # Dense-embedding model for hybrid guideline-corpus retrieval (P3.3,
    # app/retrieval.py) -- distinct from ollama_model (chat/extraction). Only
    # consulted when copilot_embed_engine == "ollama" (P3.10b, see below);
    # the llama_server embed engine is the default.
    ollama_embedding_model: str = "nomic-embed-text"

    # OAuth2 endpoints on the OpenEMR "default" site. Paths are relative to
    # ``openemr_base_url``.
    openemr_oauth_registration_path: str = "/oauth2/default/registration"
    openemr_oauth_token_path: str = "/oauth2/default/token"
    # RFC 7662 token introspection endpoint (#124 Phase 4). The agent
    # introspects the forwarded per-user bearer here, authenticating as the
    # confidential prod client via HTTP Basic auth.
    openemr_oauth_introspection_path: str = "/oauth2/default/introspect"
    # Superset of scopes requested for the dev token flow: OIDC + refresh,
    # standard + FHIR API, and a FHIR Patient read scope for the proof call.
    openemr_oauth_scopes: str = (
        "openid offline_access api:oemr api:fhir user/patient.read user/Patient.read"
    )

    # DEV-ONLY dev-token bridge (issue #126, finding F4). The agent obtains a
    # REAL OpenEMR user token server-side (dev password grant) for its tool
    # calls, because the browser's DevAgentToken is only an identity assertion,
    # not a real OpenEMR token. The real token never reaches the browser.
    # Identity for ACL is this demo clinician until #124 (production
    # authorization_code) lands. See app/dev_token_bridge.py.
    #
    # Path (inside the agent container) to the confidential-client credentials
    # written by scripts/bootstrap-copilot-dev-client.sh. Lives under the
    # appuser-writable /data dir so the running agent can read it.
    copilot_dev_client_creds_path: str = "/data/openemr-dev-client.json"
    # Demo clinician credential used for the dev password grant (dev defaults;
    # override via env in any non-default dev setup).
    copilot_dev_clinician_username: str = "admin"
    copilot_dev_clinician_password: str = "pass"
    # Resource read scopes the tools need. Only OpenEMR-recognized standard/FHIR
    # scope identifiers (see ServerScopeListEntity::apiScopes) -- e.g. problems
    # are user/medical_problem.read, labs/vitals use the FHIR Observation scope.
    copilot_dev_token_scopes: str = (
        "openid offline_access api:oemr api:fhir user/patient.read "
        "user/medication.read user/allergy.read user/medical_problem.read "
        "user/encounter.read user/appointment.read user/vital.read "
        "user/procedure.read user/Observation.read"
    )

    # Production authorization_code client (#124 Phase 1). Distinct from the dev
    # token bridge above: this client is driven by the browser via the OAuth2
    # authorization_code grant, and an OpenEMR admin (not a dev SQL shortcut)
    # enables it. See app/prod_client_registration.py and the README.
    #
    # CANONICAL redirect_uri -- the single source of truth Phase 2's authorize/
    # callback must match byte-for-byte (OpenEMR requires exact redirect_uri
    # matching). It is the BROWSER-facing host (localhost:9300) and the module's
    # one-file-per-route OAuth callback endpoint, NOT the internal ``openemr``
    # docker alias used for the server-side registration/token calls.
    copilot_prod_client_redirect_uri: str = (
        "https://localhost:9300/interface/modules/custom_modules/"
        "oe-module-clinical-copilot/public/oauth-callback.php"
    )
    # SMART-on-FHIR scopes for the production client. Every scope MUST exist in
    # OpenEMR's ServerScopeListEntity::getAllSupportedScopesList() -- dynamic
    # registration REJECTS the whole request with ``invalid_scope`` on the first
    # unrecognized scope (AuthorizationController::validateScopesAgainstServer-
    # ApprovedScopes). ``user/*.read`` is NOT used -- OpenEMR has no wildcard
    # scope entry, so it would be rejected.
    #
    # The per-resource read scopes are registered here (not deferred to authorize
    # time): ScopeRepository::finalizeScopes only lets a token carry scopes the
    # client REGISTERED with, so an unregistered read scope requested at grant
    # time is silently dropped -> tool calls would have api:oemr/api:fhir but no
    # resource-read authorization. These mirror the known-accepted
    # copilot_dev_token_scopes plus the SMART-launch scopes.
    copilot_prod_client_scopes: str = (
        "openid offline_access launch launch/patient api:oemr api:fhir fhirUser "
        "user/patient.read user/medication.read user/allergy.read "
        "user/medical_problem.read user/encounter.read user/appointment.read "
        "user/vital.read user/procedure.read user/Observation.read"
    )
    # Path (inside the agent container) for the production client credentials
    # written by the prod registration CLI. Distinct file from the dev bridge's.
    copilot_prod_client_creds_path: str = "/data/openemr-prod-client.json"

    # #124 Phase 4 (pivotal): when true, the agent (a) validates the request's
    # forwarded per-user OpenEMR bearer via introspection and (b) uses that same
    # token for every tool call, so OpenEMR enforces per-user ACL end-to-end.
    # Default OFF -- flag off keeps today's dev stub validator + DevTokenBridge
    # demo-clinician token, byte-identical. Flipped on with the module side in
    # Phase 6. Introspection results are cached (hash-keyed) for this many
    # seconds, further capped by each token's own ``exp``.
    copilot_per_user_token_enabled: bool = False
    copilot_introspection_cache_ttl_seconds: float = 60.0

    # #168 (VULN-0001, critical, Phase 3 red-team): with
    # copilot_per_user_token_enabled OFF (the shipped default above),
    # app.chat.get_token_validator used to always hand back a stub that
    # accepts ANY non-empty bearer token -- a caller on the internal docker
    # network with a garbage token got a normal 200 with patient data. The
    # fixed default is fail-closed: flag-off now REJECTS every token unless
    # this flag is also explicitly set.
    #
    # Setting this to True restores the OLD permissive behaviour (any
    # non-empty token accepted, no introspection call) for the flag-off path
    # only. It exists SOLELY so a local dev stack -- where the agent's port
    # is never published and DevTokenBridge already drives every real tool
    # call with its own demo-clinician token (see copilot_dev_* above) --
    # can still exercise /chat with an arbitrary bearer header without
    # standing up full per-user introspection.
    #
    # MUST NEVER be enabled outside local development. It does not gate any
    # tool call's authorization (that is always DevTokenBridge's token when
    # copilot_per_user_token_enabled is off) -- it only controls whether
    # /chat's own bearer check is real or a no-op. Enabling it in any shared
    # or internet-reachable environment reopens VULN-0001 exactly as filed.
    copilot_dev_accept_any_bearer_token: bool = False

    # Base dir for ``app.ingestion.LocalIngestionStore`` (P3.1's disclosed
    # local-disk placeholder for OpenEMR document storage -- see that
    # module's docstring). ``app.documents``'s GET /documents/{source_id}
    # (P3.7) reads stored source PDFs from here for the citation overlay.
    copilot_ingestion_base_dir: str = "/data/ingestion"

    # P3.10a (epic #52 step 1): which engine serves the text-generation LLM
    # roles -- planner chat/extract, claim extraction, and the LLM-as-reranker
    # relevance score (see app/chat.py's get_text_llm_client). Literal
    # "ollama" or "llama_server". Embeddings (ollama_embedding_model) and
    # vision-based document-ingestion extraction (app/supervisor.py's
    # IntakeExtractorWorker) ALWAYS use Ollama regardless of this flag --
    # see app/chat.py's _build_evidence_workers. Default "llama_server"
    # (P3.10e, issue #73, owner decision 2026-07-19): Qwen3-8B-Q5 is the
    # intended default answer model so the verified-citation capability is
    # live out of the box. Set to "ollama" for instant rollback.
    copilot_llm_engine: str = "llama_server"
    # Connection info for the llama-server instance serving the engine above
    # when copilot_llm_engine == "llama_server" (app/llama_server_client.py).
    llama_server_base_url: str = "http://llama-server:8080"
    # Label sent in the request body only -- llama-server ignores it for
    # routing (a single --model file is loaded), but the OpenAI-compatible
    # endpoint still requires the field to be present.
    llama_server_model: str = "qwen3-8b"
    llama_server_api_timeout_seconds: float = 60.0
    # Issue #93 (fix 2/4): raised 2 -> 3. Each attempt is independently
    # bounded by llama_server_api_timeout_seconds (60s) and _EXTRACT_MAX_TOKENS
    # (app/llama_server_client.py, ~51s decode headroom) -- a third attempt
    # only adds wall-clock time on the (already-failing) retry path, it does
    # not change any single attempt's timeout/token budget. Paired with the
    # retry-prompt improvement in LlamaServerClient.extract (each retry now
    # carries specific feedback about what was wrong, not an identical
    # re-roll), so the extra attempt has a genuinely different chance of
    # succeeding rather than just re-trying the same failure a third time.
    llama_server_extract_max_retries: int = 3

    # P3.10b (epic #52 step 2): which engine serves dense-vector embeddings
    # (nomic-embed-text) for hybrid guideline-corpus retrieval -- see
    # app/chat.py's _build_evidence_workers. Literal "ollama" or
    # "llama_server". A DEDICATED flag rather than reusing
    # copilot_llm_engine: the two roles are independently rollback-able (an
    # answer-model rollback to Ollama should not also silently move
    # embeddings, and vice versa). Default "llama_server" -- nomic-embed is
    # the same GGUF weights either way (app.llama_server_embed_client.
    # LlamaServerEmbedClient), so retrieval quality is unaffected; see the
    # parity check recorded in the P3.10b PR description.
    copilot_embed_engine: str = "llama_server"
    # Connection info for the SECOND llama-server instance, running in
    # `--embedding` mode, serving nomic-embed-text -- a distinct service from
    # llama_server_base_url above (which serves chat/extract). See
    # app/llama_server_embed_client.py and docker-compose.copilot.yml.
    llama_server_embed_base_url: str = "http://llama-server-embed:8080"
    llama_server_embed_model: str = "nomic-embed-text-v1.5"
    llama_server_embed_api_timeout_seconds: float = 60.0
    llama_server_embed_max_retries: int = 2

    # P3.9: when true, POST /chat additionally routes each turn's question
    # through the P3.5 supervisor's evidence-retriever worker (hybrid
    # retrieve + rerank over the PUBLIC guideline corpus, app.retrieval /
    # app.reranking) and offers the retrieved chunks to the claim extractor
    # as citable evidence -- see app/chat.py's get_evidence_retriever.
    # Default OFF -- evidence retrieval itself is flag-gated (no retrieval
    # call, no extra embedding round trip when off). Per-turn encounter
    # logging (P3.8, non-PHI counts/timings only) is always-on regardless of
    # this flag -- see app/chat.py's _log_encounter_record.
    copilot_evidence_retrieval_enabled: bool = False

    # Issue #167 (VULN-0004): caps how many conversations
    # app.chat.ConversationStore retains at once -- an unbounded in-memory
    # dict growing for the process lifetime is unbounded attacker-influenced
    # memory growth (any caller can start a new conversation). The store
    # evicts least-recently-used conversations once this cap is exceeded.
    # 2000 is generous headroom for a single-appliance deployment (this
    # service is not internet-facing -- port 8000 is internal-network only,
    # see the issue's "Reachability" section). Operator-tunable so a larger
    # or smaller deployment can adjust it.
    copilot_max_stored_conversations: int = DEFAULT_MAX_STORED_CONVERSATIONS

    # Issue #167 (VULN-0004), Gate 2 finding: the conversation-count cap
    # above does NOT bound a single conversation's own growth -- an attacker
    # who reuses ONE conversation_id and calls /chat repeatedly could grow
    # that one conversation's turn history forever, staying permanently
    # most-recently-used and so never becoming an eviction candidate. This
    # caps turns retained PER conversation; app.chat.ConversationStore.
    # append_turn drops the oldest turn once the cap is exceeded. Verified
    # safe to drop silently (not just truncate-with-warning): ``.history`` is
    # read in exactly one place in the whole service
    # (``app.chat._stream_chat``'s ``bool(conversation.history)`` -- a
    # "does this conversation have any prior turns at all" signal for
    # ``app.extraction.clarify_unresolvable_referent``) and is NEVER fed to
    # the planner as conversation context; dropping old turns cannot change
    # any answer, only shrink the retained (audit-record-shaped, P2.17) turn
    # list. 50 is generous for any realistic single clinical Q&A session (a
    # session with 50+ distinct chat turns would be unusual). This DOES now
    # bound a Conversation's total memory to a small fixed multiple of one
    # turn's size: issue #174 found and removed the one other unbounded term
    # on the same object, ``Conversation.patient_roster`` (every OTHER
    # patient's name, scaling with total patient count, cached per
    # conversation for its whole lifetime) -- see ``Conversation``'s
    # docstring in app/chat.py for the removal's full memory/privacy
    # analysis, and ``copilot_roster_cache_ttl_seconds`` below for its
    # process-wide-shared-cache replacement.
    copilot_max_turns_per_conversation: int = DEFAULT_MAX_TURNS_PER_CONVERSATION

    # Issue #174 (dominant memory term + privacy): app.chat.RosterCache is a
    # single, process-wide, TTL'd cache of OpenEMR's full patient roster,
    # replacing a per-``Conversation`` copy that scaled with total patient
    # count and was retained for the conversation's whole lifetime (see
    # ``Conversation``'s docstring). The roster feeds a *soft* heuristic
    # (app.extraction.detect_foreign_patient_reference's roster-based
    # "switch to <Name>" signal, routing to a refusal) -- the unconditional
    # numeric-id cross-patient signal is unaffected by any staleness here.
    # 300 seconds (5 minutes) trades a bounded, brief staleness window (a
    # patient added to OpenEMR within the last 5 minutes briefly misses the
    # name-match signal) against collapsing what used to be one full-roster
    # fetch per matching turn -- a ~2000x amplification against OpenEMR's
    # patient API at this service's target patient-panel scale -- down to
    # roughly one fetch per 5 minutes process-wide, regardless of how many
    # conversations or matching turns occur in that window. Operator-tunable
    # so a deployment with a rapidly-changing patient panel can shorten it,
    # or one prioritizing fewer OpenEMR round trips can lengthen it.
    copilot_roster_cache_ttl_seconds: float = DEFAULT_ROSTER_CACHE_TTL_SECONDS

    # Issue #47: when true, POST /chat additionally runs the semantic-support
    # LLM-judge (app.semantic_support) over every DocumentCitation whose
    # verbatim-provenance check already passed -- a citation only counts as
    # "verified" when BOTH the quote is real (existing check) AND the judge
    # affirms it actually supports the claim's prose. Default ON (issue #81):
    # the P3.9b measurement showed the gate reliable (6/6 adversarial caught,
    # 6/6 supported-correct), so "verified" now means provenance AND semantic
    # support in production. See app/semantic_support.py.
    copilot_semantic_support_enabled: bool = True

    # Issue #153: when true, run_verification additionally requires that each
    # extracted claim's own TEXT be deterministically grounded in the
    # planner's answer (app.answer_grounding.claim_is_grounded_in_answer) --
    # a claim citing a real, correctly-valued record the answer never
    # actually asserted (e.g. a hallucinated respiratory_rate claim on a
    # question about blood pressure) no longer counts toward a VERIFIED
    # verdict just because its citation resolves against raw tool data.
    # Deterministic, no LLM call -- unlike copilot_semantic_support_enabled
    # above. Default OFF: byte-identical to today.
    #
    # An adversarial review (issue #153) found the heuristic NOT fit to
    # enable as shipped: negation is unhandled (the stopword list drops
    # "not"/"no", so "Patient is not allergic to penicillin." is judged
    # grounded by a claim asserting the patient IS allergic), short claims
    # bypass the ratio easily, wrong-record numeric values pass, and routine
    # clinical abbreviations ("HR 72" vs "heart rate") cause false rejections
    # of legitimate claims -- see app.answer_grounding's module docstring for
    # the full list with examples. This flag stays OFF until that is
    # addressed; do not flip it based on this comment alone.
    #
    # `evals/runner/pipeline.py`'s `run_case` now threads this setting
    # through to `run_verification` (reading it fresh from the environment),
    # so re-running the eval suite with
    # `COPILOT_CLAIM_ANSWER_GROUNDING_ENABLED=true` does exercise the gate
    # against the existing recordings and surfaces per-category pass/fail
    # deltas -- but the committed recordings/assertions were authored
    # assuming the gate is off, so doing so today will show failures, not a
    # clean per-category strip-rate report. The owner is deciding the design
    # fix; only after that should recordings/assertions be revisited and an
    # actual per-category measurement taken before flipping this default.
    copilot_claim_answer_grounding_enabled: bool = False

    # Issue #158: when true, ``run_verification`` additionally scopes the
    # claim extractor's citable inputs to the tool CALLS the answer actually
    # engaged with (app.tool_call_scoping) -- coarser than #153's above
    # (per-CLAIM-text grounding) and orthogonal to it: this is a per-TOOL-
    # CALL cut, decided ONCE per turn from the answer's lexical overlap with
    # each call's record values, not per-claim/per-field. Closes the same
    # #149 gap #153 targeted (a claim citing a real, correctly-valued field
    # the answer never discussed) via a coarser, owner-approved mechanism
    # after #153's per-claim heuristic was found unfit to enable (negation
    # blind, short claims bypass the ratio, wrong values pass -- see
    # ``copilot_claim_answer_grounding_enabled`` above and
    # app.answer_grounding's module docstring).
    #
    # Two enforcement points, both gated by this ONE flag (see
    # app.tool_call_scoping's module docstring for the full rule):
    #   1. PREVENTION -- ``app.extraction.ClaimExtractor.extract_claims``
    #      narrows the catalog/tool-result messages the extractor sees to
    #      only the engaged calls' records (unengaged calls' positional
    #      ``call_i`` ids are skipped, never renumbered -- the id scheme is
    #      load-bearing, see app.verification's module docstring decision 2).
    #   2. ENFORCEMENT -- ``app.tool_call_scoping.apply_tool_call_scoping``
    #      downgrades any surviving citation of an unengaged call to the new
    #      ``CitationStatus.TOOL_CALL_NOT_ENGAGED``, the same fail-closed
    #      shape #153's ``NOT_GROUNDED_IN_ANSWER`` downgrade already
    #      established (Notice rendered, claim not verified) -- this is what
    #      actually holds when the extractor is a test double that ignores
    #      the narrowed catalog (every hermetic test's fake extractor).
    #
    # Deterministic, no LLM call. Default OFF: byte-identical to today (no
    # catalog change, no new checks, existing suite untouched). Coarse-first
    # by owner decision -- whether this is precise enough to ship ON, or
    # needs a finer per-claim/per-field cut layered on top, is a LATER
    # measurement task, not decided here.
    #
    # `evals/runner/pipeline.py`'s `run_case` threads this setting through to
    # `run_verification` exactly parallel to
    # `copilot_claim_answer_grounding_enabled` above, so re-running the eval
    # suite with `COPILOT_EXTRACTION_TOOL_CALL_SCOPING_ENABLED=true` exercises
    # this gate against the existing recordings.
    copilot_extraction_tool_call_scoping_enabled: bool = False

    # Issue #170: when true, ``run_verification`` additionally runs the
    # SourceRef-relevance LLM-judge (app.source_ref_relevance) over every
    # claim whose surviving citations are ALL ``SourceRef``s (zero
    # ``DocumentCitation``s -- the #130-census population,
    # ``evals/runner/census_source_ref_claims.py``) and which already passed
    # provenance re-validation -- the ``SourceRef`` counterpart to
    # ``copilot_semantic_support_enabled`` above. A claim carrying even one
    # DocumentCitation is untouched by this gate (already
    # ``copilot_semantic_support_enabled``'s territory).
    #
    # THIS IS THE #130/#170 MECHANISM CLASS, MEASURED AND DECLINED TWICE
    # ALREADY (#130: context-free judge, false-rejected genuinely valid terse
    # claims; #164/#163: a structurally different but adjacent gate, declined
    # for prevention-driven false blocks on absence-shaped answers). Default
    # OFF, and stays OFF unless a live re-measurement under the SAME protocol
    # #130 used (``evals/runner/issue_170_source_ref_relevance_spike.py`` --
    # 12 ``citation_present`` cases, >=8 draws/case) shows the established-
    # facts-context fix (module docstring's "Established-facts context")
    # closes the false-reject gap #130 found, per #130's own pre-registered
    # upgrade criteria. See ``docs/MODEL_AND_HARDWARE_SELECTION.md``'s issue
    # #130/#170 findings sections for the measured numbers before ever
    # flipping this default. Flag OFF: byte-identical to today, no extra LLM
    # call.
    #
    # `evals/runner/pipeline.py`'s `run_case` threads this setting through to
    # `run_verification` exactly parallel to
    # `copilot_extraction_tool_call_scoping_enabled` above.
    copilot_source_ref_relevance_enabled: bool = False

    # Issue #173: request-body-size cap enforced by
    # app.body_size_limit.BodySizeLimitMiddleware, the outermost ASGI
    # middleware in app.main.create_app. See DEFAULT_MAX_REQUEST_BODY_BYTES
    # above for the value justification. Operator-tunable for a deployment
    # with a legitimately larger (or, for defense in depth, smaller) need.
    copilot_max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES


def get_settings() -> Settings:
    """FastAPI dependency returning the current application settings."""
    return Settings()
