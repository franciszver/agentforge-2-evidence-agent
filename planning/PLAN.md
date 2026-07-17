# Plan: Complete AgentForge Phase 2 — Multimodal Evidence Agent & Document RAG

*Companion plan: `complete-agentforge-3-redteam.md` (Phase 3, runs after this phase is frozen).*

## Decisions locked (2026-07-17)

| Decision | Choice |
|---|---|
| LLM/RAG stack | **Fully local** — no PHI or clinical data leaves the machine. Local vision model, local embeddings, local reranker. No Cohere/cloud vision. |
| Execution model | **New Claude Code session**, opened inside this phase's repo. Paste-ready kickoff prompt lives in `instructions/INITIAL_PROMPT.md` (Phase 2 section). |
| Repo | `agentforge-2-evidence-agent` — full-history duplicate of Phase 1 final state (not a fork). |
| Phase sequencing | Phase 3 (red-team) attacks the co-pilot built here, so this phase must be frozen before Phase 3 begins. |

## What Phase 1 established (the foundation this phase reuses)

Repo: `github.com/franciszver/agentforge-1-clinical-copilot` (branch `main`, already public). Full-history duplicate of `Gauntlet-HQ/openemr-base-clean` with a local AI co-pilot layered on. Load-bearing assets to build on:

- **Local agent harness** (`services/copilot-agent/app/`): FastAPI service, Ollama serving Qwen3-4B, a custom single-tool-call-per-turn planner loop (`planner.py`, temp 0), hand-rolled `ollama_client.py` (streaming chat + schema-constrained `extract`).
- **Structural safety seams**: `authz.py` (`enforce_patient_binding` → `PatientBindingViolation`), `quarantine.py` (untrusted tool text is summarized by a tool-less LLM call before it can reach the planner — prompt-injection defense).
- **Deterministic verification layer** (`verification.py`, `verdict.py`): every claim carries citations re-validated against raw tool results with no model call; rolls up to `verified`/`partially_verified`/`blocked`.
- **Typed tools** (`app/tools/*`, `app/schemas/tools.py`) dispatched against OpenEMR REST/FHIR via `openemr_client.py`; local SQLite drug-interaction DB.
- **Eval harness** (`evals/`): record/replay via `ollama_replay.py` so CI runs adversarial categories (injection, hallucination_bait, authorization_probe, stale/missing data) deterministically, offline.
- **Observability**: correlation IDs (`correlation.py`), structured JSON logs, no-PHI SQLite trace store + dashboard + feedback→eval-case promotion loop.
- **Auth**: full OAuth2 / SMART-on-FHIR broker (per-user token, PKCE, launch-context patient binding).
- **Deploy**: `docker/development-easy/docker-compose.copilot.yml` (Ollama + agent overlay), `scripts/tailscale-serve-copilot.sh`.
- **CI** (`.github/workflows/copilot-ci.yml`): mypy, pytest, a 99% branch-coverage gate on trust-critical modules, offline eval replay, PHP module tests.
- **Decision log**: `prd/DECISIONS.md` (80KB, ~50 dated entries) — CONTINUE it in Phase 2, don't restart.

## Phase 1 as-built vs the brief — what Phase 2 actually inherits

Phase 1's technical core is over-built vs its brief, but several "public/production" deliverables were deliberately dropped or shipped off. These change Phase 2's starting assumptions (from a full read of `prd/DECISIONS.md` + `docs/ARCHITECTURE.md`, 2026-07-17):

- **Per-user ACL is OFF by default (the big trap).** The full OAuth2/PKCE/SMART/introspection flow (`#124`) is built and *proven live end-to-end* (restricted role 403 vs admin 200), but the shipped default is a shared "dev token bridge" — owner decision "Verify live, keep default off" to avoid a one-way door + consent-step demo friction. **Phase 2's supervisor/workers therefore run under a shared clinician token unless `copilot_per_user_token_enabled` is flipped.** Decide up front whether Phase 2 develops against ACL-ON (more realistic, more work) or ACL-OFF (matches shipped default). Record it.
- **No API collection exists.** The brief's Postman/Bruno requirement was never met in Phase 1 — Phase 2 builds it from scratch (it's a Phase 2 requirement too), not an inheritance.
- **Cost tracking is estimated, not measured.** Per-call token emission was never wired; the "cost analysis" is a back-of-envelope TCO section. Phase 2 requires *measured* per-encounter token/cost — so Phase 2 must actually implement token accounting, not extend an existing measured pipeline.
- **Deployment is Tailscale-only, not a public URL** (deliberate, zero-egress thesis). Phase 2's "deployed app" expectation is satisfied the same way; keep the Path-to-Production section.
- **Load data only covers 1/5/10 concurrent** (single 8GB GPU serializes at ~0.15 req/s, p50 59s @10). Phase 2's p95 ingestion SLO must be set against this real hardware ceiling, not an aspirational number.
- **Eval "pass" is redefined, not just counted.** 31 cases, 25 pass + 6 **strict `xfail`** (kept-failing real model weaknesses) = 0.8065, un-gameable by construction; assertion vocab is `first_tool_in / answer_contains / answer_not_contains / verdict / must_refuse / no_phi`. Phase 2's 50-case boolean gate **extends this harness** (record/replay, `ollama_replay`) and adds the brief's boolean rubric categories on top of this vocabulary.
- **Citation validity depends on wide-format tool outputs** (~100% wide vs 17% EAV/long). Phase 2's document-extraction facts must normalize to wide format to stay citable.
- **Verification reads raw pre-quarantine results on a separate verifier-only channel; the extraction LLM is treated as "summarizer-class-safe" (no tools/client/token).** Phase 2's new vision-extraction stage must fit inside that exact containment boundary.
- **Model-execution policy (owner, 2026-07-16):** subagents run on cheaper models for token savings; the top model only orchestrates. Carry this into Phase 2 execution.

## Gate 0 — Phase 1 must be frozen before Phase 2 duplicates it

Phase 2 is a full-history duplicate of Phase 1 *at its final state*, so Phase 1's Stage-4 hardening must land first — Phase 2 inherits whatever state Phase 1 is frozen in. Once these close, tag `agentforge-1-clinical-copilot` `v1.0` (Phase 2's README opens with "continues agentforge-1-clinical-copilot at v1.0").

### Phase 1 freeze status (readiness check, 2026-07-17)

Engineering is strong — repo is already **public**, GPL LICENSE inherited, `.env` hygiene clean, real CI with a 99% coverage gate + badge, and unusually thorough `docs/AUDIT.md` (34KB), `docs/ARCHITECTURE.md` (42KB, has diagram/Path-to-Production/TCO/eval results), and `prd/DECISIONS.md` (80KB). But these gaps must close before freezing/duplicating:

- [ ] **50-concurrent load baseline** — only 1/5/10-user data exists (`docs/ARCHITECTURE.md` "Measured results (P5.1)", script `services/copilot-agent/scripts/capacity_run.py`). Brief requires 50. Extend the existing script.
- [ ] **DEMO_SCRIPT.md** — does not exist. (Demo video #203 noted as "Next" in DECISIONS.md, not done.)
- [ ] **INTERVIEW_PREP.md** — does not exist. Mock-interview gate not yet run.
- [ ] **API collection (Bruno/Postman)** — none committed.
- [ ] **README polish** — no screenshots/GIF; Path-to-Production, eval results, and cost analysis exist only as cross-links to ARCHITECTURE.md, not summarized inline. `docs/USERS.md` is thin (2.6KB).
- [ ] **Repo metadata** — description empty, no topics set, not confirmed pinned.
- [ ] **Per-user OAuth/ACL (`#124` epic)** — built and proven live but shipped **flag-gated OFF** ("F4 documented-open-until-owner-flips"). Decide: flip ON for the frozen release, or document as a deliberate, defensible default with the flip as a Path-to-Production step. The one item with security-narrative weight — resolve intentionally, not by default.
- Minor/deferred (carry forward, just document): unresolved CORS TODO (`CORSListener.php:56-57`), `#185` async introspection, `#172` input PHI deterrent, `#175` encounter-planner tool-selection, P5.6 fresh-clone validation, `#16` tailscale demo GIF.

---

## Phase 2 scope over Phase 1

Document ingestion (lab PDF + intake form) with vision extraction, hybrid RAG + rerank over a guideline corpus, supervisor + 2 workers, citation contracts with PDF bounding-box overlays, 50-case boolean-rubric eval set as a PR-blocking CI gate.

### Fully-local tooling choices (to defend in the decision log)
- **Vision extraction:** a local vision-language model via Ollama (e.g. `qwen2.5-vl` or `llama3.2-vision`), schema-constrained decoding into Pydantic — mirrors the existing `ollama_client.extract` pattern. No cloud OCR/vision.
- **Embeddings (dense):** local via Ollama (`nomic-embed-text` or `bge-m3`).
- **Sparse:** BM25 over the corpus (rank_bm25, or SQLite FTS5 to stay dependency-light and consistent with existing SQLite use).
- **Reranker:** local cross-encoder (`bge-reranker-v2-m3`) run in-process (sentence-transformers) or as a tiny sidecar — explicitly **not** Cohere. Document as the local substitute for "Cohere Rerank or equivalent."
- **Vector store:** local — `sqlite-vec` (fits existing SQLite footprint) or a Chroma/Qdrant container in the compose overlay.
- **Orchestration:** extend the existing hand-rolled planner into a **supervisor + 2 workers** (intake-extractor, evidence-retriever) with explicit logged handoffs, OR adopt LangGraph if inspectable graph tracing is wanted. Recommendation: extend the custom loop first (keeps the dependency-light, fully-inspectable Phase 1 philosophy and a cleaner interview story); reach for LangGraph only if handoff/trace requirements get unwieldy. Decide at Architecture Defense.
- **Corpus:** a small, **public, non-PHI** clinical-guideline set curated to the co-pilot's use cases (a handful of open guideline documents). Since it's public, its indexing has no PHI-egress concern even under local-only — but keep the whole pipeline local for a single coherent story.

### Staged plan (APPROACH.md pipeline)
1. **Stage 1 — Foundation:** duplicate repo 1, verify it runs (`docker compose up` + agent + Ollama), pull the vision + embedding models, confirm CI green on the fresh clone. Gate: fresh clone runs in one command.
2. **Stage 2 — Docs (`W2_ARCHITECTURE.md`):** document Week-1-baseline vs Week-2-new boundary, document/resolve any Phase 1 debt, schemas + migration notes for any Week-1 schema change, testing strategy (unit/integration/eval-gate split), SLOs (p95 ingestion latency target you set & justify, retrieval hit-rate target), data model/lineage/access-control for Week-2 artifacts, incident-response + backup/recovery (RPO/RTO). Gate: every new capability traces to a use case.
3. **Stage 3 — Build (staged internally):**
   - (3a) **Ingestion tool** `attach_and_extract(patient_id, file_path, doc_type)` for `lab_pdf` and `intake_form`; stores source doc in OpenEMR, persists derived facts as FHIR resources; strict Pydantic schemas (lab: test/value/unit/ref-range/collection-date/abnormal-flag/citation; intake: demographics/chief-concern/meds/allergies/family-history/citation). Vision extraction with **no fabrication** — unreadable fields return "not found," never guessed.
   - (3b) **Hybrid RAG** (BM25 + dense) + local rerank over the guideline corpus.
   - (3c) **Supervisor + 2 workers** with explicit, logged handoffs; worker spans as children of the supervisor span (distributed tracing extending Phase 1 correlation IDs).
   - (3d) **Citation contract** on every clinical claim `{source_type, source_id, page_or_section, field_or_chunk_id, quote_or_value}` + **visual PDF bounding-box overlay** (click-to-source). Extend the existing verification layer to cover document-sourced claims.
   - (3e) **Observability/cost per encounter:** tool sequence, per-step latency, token usage + cost estimate, retrieval hits, extraction confidence, eval outcome — no raw PHI in logs (CI-verified scrubbing).
4. **Stage 3 gate — 50-case golden eval set** with **boolean** rubrics across `schema_valid`, `citation_present`, `factually_consistent`, `safe_refusal`, `no_phi_in_logs`; wired as a **PR-blocking CI gate** that fails the build if any category regresses >5% or drops below threshold. Golden set reproducible from the repo alone (extend the `ollama_replay` recording pattern). **This is the graded hard gate — graders inject a regression that this must catch.** Also add: OpenAPI 3.0 spec + contract tests, integration tests with fixtures/stubs that pass without live API, dependency/security scans, `/health` vs `/ready` (degraded-aware), alerting (extraction-failure rate, retrieval latency, eval regression >5%), updated Bruno/Postman collection, perf baselines vs Phase 1.
5. **Stage 4 — Hardening & repo polish:** Stage-4 checklist again; README clearly separates Week-1 baseline vs Week-2 behavior; flip repo public when it passes.
6. **Stage 5 — Demo assets:** new `DEMO_SCRIPT.md`; seed a patient whose lab PDF + intake tell a story (a changed value the RAG surfaces, a citation overlay, one graceful failure — e.g. an unreadable scan handled honestly). Timed dry run ≤5 min.
7. **Stage 6 — Learning & interview prep:** new `INTERVIEW_PREP.md` (vision-without-hallucination, hybrid-RAG design, supervisor/worker trust split, the eval-gate-catches-regression story, p95 SLO justification, PHI-in-logs audit). Mock interview before recording. Then generate/refine the Phase 3 kickoff prompt to reference the real Phase 2 attack surface.

**Biggest risks:** (a) local VLM extraction accuracy on scanned PDFs — mitigate with schema-constrained decoding + the citation/verification backstop + explicit "not found" behavior; (b) the CI eval gate must genuinely catch an injected regression — build & test that gate deliberately, don't assume; (c) keeping Phase-1 evals green while adding Week-2 surface.

## Execution handoff

Open a fresh Claude Code session **inside** `C:\Users\franc\Projects\agentforge-2-evidence-agent`, paste the Phase 2 kickoff prompt from `instructions/INITIAL_PROMPT.md`, let it plan-before-build, then run Stages 1–6. Start only after Gate 0 (Phase 1 frozen at v1.0) passes.
