# Phase 2 Kickoff Prompt — Multimodal Evidence Agent & Document RAG

Open a Claude Code session **in this repo** and paste the prompt below verbatim.
Paths point at the copies in this `planning/` folder, so it is self-contained.

**Phase 3 kickoff:** this prompt's own closing paragraph called for refining
the Phase 3 kickoff prompt once Phase 2's real attack surface existed. That
document is now written, against what Phase 2 actually built (not the
planned shape below) — see `planning/PHASE3_KICKOFF_PROMPT.md`.

**Prerequisite:** Phase 1 (`agentforge-1-clinical-copilot`) frozen at `v1.0` —
close the Gate 0 gaps listed in `planning/PLAN.md` first. This repo is already a
full-history duplicate of Phase 1 (`upstream` → the Phase 1 GitHub repo); when
Phase 1's freeze lands, run `git fetch upstream && git merge upstream/main` to
pull it forward.

---

```
You are building AgentForge Phase 2: Multimodal Evidence Agent & Document RAG.
This repo is a full-history duplicate of agentforge-1-clinical-copilot at its
v1.0 frozen state (remote "upstream"). The README must open with "continues
agentforge-1-clinical-copilot at v1.0" and clearly separate Week-1 baseline
behavior from Week-2 new behavior. This is one of three standalone AgentForge
repos; Phase 3 (agentforge-3-redteam) will later attack the co-pilot you build
here.

Read these completely before doing anything else (all in this repo):
1. The brief (source of truth for requirements):
   ./planning/2_AgentForge_MultimodalAgent_DocumentRAG.html
2. The delivery playbook (process, gates, deliverables):
   ./planning/APPROACH.md
3. The cross-phase plan (decisions already made, tooling choices, staging,
   and how Phase 1 as-built changes what you inherit):
   ./planning/PLAN.md

Then study the Phase 1 codebase you inherited — the real architecture, not
assumptions: services/copilot-agent/app/ (planner.py single-tool-per-turn loop,
ollama_client.py chat+extract, authz.py patient binding, quarantine.py injection
seam, verification.py + verdict.py deterministic citation checker, tools/*,
schemas/*), evals/ (ollama_replay.py record/replay), correlation.py + the trace
store/dashboard, the OAuth2/SMART broker, and prd/DECISIONS.md (the existing
~50-entry decision log — CONTINUE it, do not restart it).

HARD CONSTRAINT — FULLY LOCAL, NO PHI EGRESS. Phase 1's entire security story is
"no PHI leaves the machine" (local Ollama, no cloud LLM). Preserve it. Do NOT use
Cohere Rerank or cloud vision. Use: a local vision-language model via Ollama
(qwen2.5-vl or llama3.2-vision) with schema-constrained extraction; local
embeddings (nomic-embed-text or bge-m3); BM25 sparse (SQLite FTS5 or rank_bm25);
a LOCAL cross-encoder reranker (bge-reranker-v2-m3), documented as the local
substitute for "Cohere or equivalent"; a local vector store (sqlite-vec or a
Chroma/Qdrant container in the compose overlay). Justify each in the decision log.

Deliver per the brief and the plan's Phase 2 stages:
- Ingestion tool attach_and_extract(patient_id, file_path, doc_type) for lab_pdf
  and intake_form; stores source doc in OpenEMR, persists derived facts as FHIR
  resources; strict Pydantic schemas; vision extraction that NEVER fabricates —
  unreadable fields return "not found," extend the verification layer to cover
  document-sourced claims. (Note: citations only validate reliably on WIDE-format
  tool outputs — normalize extracted facts to wide format.)
- Hybrid RAG (BM25 + dense) + local rerank over a small PUBLIC, non-PHI clinical-
  guideline corpus curated to the co-pilot's use cases.
- Supervisor + 2 workers (intake-extractor, evidence-retriever) with explicit,
  logged handoffs; worker spans as children of the supervisor span (extend the
  Phase 1 correlation IDs into distributed tracing). Prefer extending the existing
  hand-rolled planner over adopting LangGraph unless the graph/trace needs force
  it — decide at Architecture Defense and record why.
- Citation contract {source_type, source_id, page_or_section, field_or_chunk_id,
  quote_or_value} on every clinical claim + a VISUAL PDF bounding-box overlay
  (click-to-source).
- 50-case golden eval set with BOOLEAN rubrics (schema_valid, citation_present,
  factually_consistent, safe_refusal, no_phi_in_logs), wired as a PR-BLOCKING CI
  gate that fails the build if any category regresses >5% or drops below
  threshold, reproducible from the repo alone (extend ollama_replay). This is the
  graded hard gate — graders inject a regression that this MUST catch, so build
  and TEST the gate deliberately against a deliberately-injected regression.
- Engineering requirements from the brief: W2_ARCHITECTURE.md (Week1-vs-Week2
  boundary, testing strategy, SLOs you set+justify — p95 ingestion latency set
  against the real single-GPU ceiling, retrieval hit rate, data model/lineage/
  access-control, incident response, backup/recovery RPO/RTO), OpenAPI 3.0 spec +
  contract tests, integration tests with fixtures/stubs that pass WITHOUT live
  API, /health vs /ready (degraded-aware), alerting, per-encounter observability
  with MEASURED cost/latency/token/retrieval-hit/extraction-confidence (Phase 1
  only ESTIMATED cost — wire real per-call token emission), NO raw PHI in logs
  (CI-verified scrubbing), a Bruno/Postman collection (Phase 1 never shipped one —
  build it), perf baselines vs Phase 1.

Also decide up front: develop against ACL-ON or ACL-OFF? Phase 1 shipped the
per-user OAuth/ACL flow built and proven live but flag-gated OFF by default
(shared "dev token bridge"). Record the choice.

Maintain prd/DECISIONS.md continuously (3 lines per non-obvious choice, at the
moment it's made). Deployment stays LOCAL-ONLY via Docker Compose (Tailscale
serve for a live URL, as Phase 1 established); keep a credible Path-to-Production
section. Synthetic/demo data only.

Start with a plan: read the three files, study the inherited codebase, then
present a phased implementation plan with per-step verification criteria BEFORE
writing code. Flag anything ambiguous or materially simpler. When the build is
hardened (Stage 4), generate DEMO_SCRIPT.md, INTERVIEW_PREP.md, and refine the
Phase 3 kickoff prompt to reference the real Phase 2 attack surface (ingestion
endpoints, worker handoffs, citation contract), then run a mock interview before
I record the video.
```
