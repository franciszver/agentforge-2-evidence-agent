# AgentForge Phase 2 — Multimodal Evidence Agent & Document RAG

[![copilot-ci](https://github.com/franciszver/agentforge-2-evidence-agent/actions/workflows/copilot-ci.yml/badge.svg?branch=main)](https://github.com/franciszver/agentforge-2-evidence-agent/actions/workflows/copilot-ci.yml)

> **Status: work in progress (scaffold).** This repo *continues*
> [agentforge-1-clinical-copilot](https://github.com/franciszver/agentforge-1-clinical-copilot)
> at its v1.0 state — created as a **full-history duplicate** (not a GitHub
> fork). Phase 1's OpenEMR base is [Gauntlet-HQ/openemr-base-clean](https://github.com/Gauntlet-HQ/openemr-base-clean),
> itself derived from [OpenEMR](https://github.com/openemr/openemr) (GPL v3).

## What Phase 2 adds on top of Phase 1

Phase 1 delivered a local, verification-backed clinical co-pilot on OpenEMR
(Ollama + Qwen, deterministic citation checking, eval harness, observability).
Phase 2 extends that foundation with:

- **Document ingestion** — lab PDFs and intake forms, with **local** vision
  extraction (no cloud OCR/vision; PHI never leaves the machine).
- **Hybrid RAG** — sparse (BM25) + dense retrieval with a **local** reranker
  over a small, public clinical-guideline corpus.
- **Multi-agent graph** — a supervisor plus two workers (intake-extractor,
  evidence-retriever) with explicit, logged handoffs.
- **Citation contracts** — machine-readable citations on every clinical claim,
  with visual PDF bounding-box overlays (click-to-source).
- **Eval gate** — a 50-case golden set with boolean rubrics wired as a
  PR-blocking CI gate.

See the plan in the private planning repo
(`plans/complete-agentforge-2-evidence-agent.md`) for the full scope, the
fully-local tooling decisions, and how this inherits Phase 1's as-built state.

**Live operational plan:** the
[Multimodal Evidence Agent project board](https://github.com/users/franciszver/projects/3)
tracks all work — stages as milestones, tasks as issues, one issue = one
branch = one PR.

## AgentForge series

1. [agentforge-1-clinical-copilot](https://github.com/franciszver/agentforge-1-clinical-copilot) — Clinical Co-Pilot Foundation
2. **agentforge-2-evidence-agent** — Multimodal Evidence Agent & Document RAG *(this repo)*
3. agentforge-3-redteam — Adversarial Security & Red-Team Platform

---

*Setup, architecture, and eval results will be documented here as Phase 2 is
built. Until then, the inherited Phase 1 documentation lives under `docs/`.*
