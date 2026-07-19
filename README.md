# AgentForge Phase 2 — Multimodal Evidence Agent & Document RAG

[![copilot-ci](https://github.com/franciszver/agentforge-2-evidence-agent/actions/workflows/copilot-ci.yml/badge.svg?branch=main)](https://github.com/franciszver/agentforge-2-evidence-agent/actions/workflows/copilot-ci.yml)

> This repo *continues*
> [agentforge-1-clinical-copilot](https://github.com/franciszver/agentforge-1-clinical-copilot)
> at its v1.0 state — created as a **full-history duplicate** (not a GitHub
> fork). Phase 1's OpenEMR base is [Gauntlet-HQ/openemr-base-clean](https://github.com/Gauntlet-HQ/openemr-base-clean),
> itself derived from [OpenEMR](https://github.com/openemr/openemr) (GPL v3).

**Live operational plan:** the
[Multimodal Evidence Agent project board](https://github.com/users/franciszver/projects/3)
tracks all work — stages as milestones, tasks as issues, one issue = one
branch = one PR.

## Week-1 baseline vs Week-2 additions

Phase 1 ("Week 1") delivered a local, verification-backed clinical co-pilot
that answers questions against OpenEMR's *structured* data — medications,
labs, encounters, vitals — with every claim independently re-checked against
the raw record before it reaches the clinician. Phase 2 ("Week 2") extends
that same trust story to *unstructured source documents* (a scanned lab
report, a patient-handed intake form) and to a public clinical-guideline
corpus, without changing Week-1's behavior for existing structured-data
questions.

| Capability | Week-1 (v1.0, inherited) | Week-2 (this repo's additions) |
|---|---|---|
| Data source | Structured OpenEMR data via typed tools (meds, labs, encounters, vitals) | + Unstructured source documents (lab PDF, intake form) via local vision extraction |
| Retrieval | None — planner calls typed tools directly | + Hybrid RAG (BM25 + dense) with local rerank over a public, non-PHI guideline corpus |
| Orchestration | Single planner loop, one tool call per turn | + Supervisor delegating to 2 workers (intake-extractor, evidence-retriever), explicit logged handoffs |
| Citation contract | `{tool_call_id, record_id, field, asserted_value}` against cached tool output | + `{source_type, source_id, page_or_section, field_or_chunk_id, quote_or_value}` against source documents, with a visual PDF bounding-box overlay |
| Verification | Deterministic re-check against raw structured tool results | + Extended to re-check document-sourced claims against extracted facts; same fail-closed "not found" discipline |
| Eval gate | 31-case suite, category pass/fail, 0.8065 pass rate | + 50-case golden set, boolean rubrics (`schema_valid`, `citation_present`, `factually_consistent`, `safe_refusal`, `no_phi_in_logs`), PR-blocking CI gate |
| Observability | Correlation IDs, request/verdict-level trace store | + Per-encounter cost/latency breakdown, worker spans as children of the supervisor span, extraction-failure/retrieval-latency/eval-regression alerts |
| Models served | Qwen3-4B (planner, quarantine, extraction) via Ollama | + 7B-class VLM (document extraction), embedding model, local reranker, and an alternate `llama_server` (llama.cpp) engine for planner/extraction/rerank with a degraded-aware `/ready` |
| Access control | Per-user OAuth2/PKCE/SMART built, proven live, flag-gated OFF | Unchanged — same flag, same posture |

Full detail, schemas, SLOs, and diagrams: `docs/W2_ARCHITECTURE.md` (Week-2
architecture, extends rather than replaces `docs/ARCHITECTURE.md`, the frozen
Week-1 architecture).

## Hardening status (Stage 4 / #25)

A hardening pass was run against the Week-2 surface — ingestion, hybrid
retrieval/reranking, the extended verification layer, the supervisor/workers,
the `llama_server` engine + `/ready`, and per-patient document-fact citations
— checked against PHI/secrets-in-logs, fail-closed-on-error, exception
handling, authorization boundaries, injection resistance, resource/DoS
limits, and dependency supply-chain. **Result: every category passed clean
against the real code (file:line cited); no fixes were required.** Full
write-up: `docs/AUDIT.md` §"Phase 2 Week-2 hardening checklist (P4.1, Stage
4, issue #25)". The OpenEMR-base and Week-1-agent findings from Phase 1's own
audit are unchanged and documented in the same file.

## AgentForge series

1. [agentforge-1-clinical-copilot](https://github.com/franciszver/agentforge-1-clinical-copilot) — Clinical Co-Pilot Foundation
2. **agentforge-2-evidence-agent** — Multimodal Evidence Agent & Document RAG *(this repo)*
3. agentforge-3-redteam — Adversarial Security & Red-Team Platform

---

*Demo assets and interview prep are tracked separately (Stage 5/6 on the
project board). See `docs/DEVELOPERS_GUIDE.md` for orientation, `docs/TEST_PLAN.md`
for testing strategy, and `docs/W2_ARCHITECTURE.md` for the full Week-2
architecture.*
