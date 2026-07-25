# AgentForge Phase 2 — Multimodal Evidence Agent & Document RAG

[![copilot-ci](https://github.com/franciszver/agentforge-2-evidence-agent/actions/workflows/copilot-ci.yml/badge.svg?branch=main)](https://github.com/franciszver/agentforge-2-evidence-agent/actions/workflows/copilot-ci.yml)

> **Status: Stage 4 (hardening + polish).** This repo *continues*
> [agentforge-1-clinical-copilot](https://github.com/franciszver/agentforge-1-clinical-copilot)
> at its v1.0 state — created as a **full-history duplicate** (not a GitHub
> fork). Phase 1's OpenEMR base is [Gauntlet-HQ/openemr-base-clean](https://github.com/Gauntlet-HQ/openemr-base-clean),
> itself derived from [OpenEMR](https://github.com/openemr/openemr) (GPL v3).

![Phase-2 evidence agent demo — three questions against the live stack: a narrative answer marked Blocked on citation-axis grounds (no single citable chart field, not a safety-conflict block), an allergy question answered against a chart with no allergies on file where the three medication claims — but not the answer's own headline claim — carry citation chips, and an honest "no recent lab results" answer whose absence itself carries a citation chip.](docs/assets/demo.gif)

*Captured from live runs against the local Phase-2 stack — a narrative
question comes back **Blocked**, which here is the citation-axis verdict
(no single citable chart field for this answer), not a safety-conflict
flag; the panel's own legend describes Blocked as "a safety conflict
stopped the answer," which does not match this particular beat — a
mismatch tracked as
[#151](https://github.com/franciszver/agentforge-2-evidence-agent/issues/151).
An allergy question comes back **Verified**: the chart has no allergies
on file, and the three medication claims each carry a citation chip
naming the record field checked, though the answer's own headline claim
("no recorded allergies...") does not. And a patient with no lab data
gets an honest "no recent lab results" answer whose absence itself
carries a citation chip.*

*Two caveats worth stating plainly rather than glossing over: verdicts
are **not deterministic** — the same question, run repeatedly against the
same stack, can return different answers, different cited claims, or a
different verdict outright. A blood-pressure question answered correctly
with a matching citation on 1 of 4 draws, and on the other 3 declined to
answer while citing an unrelated respiratory-rate claim the user never
asked about — all four still verdict `verified`
([#149](https://github.com/franciszver/agentforge-2-evidence-agent/issues/149));
the allergy question above returned `verified` on 5 of 6 draws and
`blocked` on 1, with the answer text semantically identical every time — the verdict
itself flipping
([#150](https://github.com/franciszver/agentforge-2-evidence-agent/issues/150)).

Separately, a `verified` verdict confirms that the citation attached to a
claim matches the record — not that the claim, or the answer, actually
addresses the question the user asked.*

## Week 1 vs Week 2

This project ran in two stretches, kept visibly separate here rather than
blended into one undifferentiated feature list.

### Week 1 (Phase 1 baseline, inherited unchanged at v1.0)

A verification-first clinical co-pilot embedded in OpenEMR: a physician asks
a question about the open chart and gets an answer where the verification
layer is designed to deterministically re-check every extracted factual
claim against the raw record and cite it — coverage per claim is not yet
universal in practice (see the Phase-2 demo caption above).

![Clinical Co-Pilot demo — a physician asks "What is he taking, and does anything conflict with starting ibuprofen?" on Phil Belford's chart, the answer streams in, then a Verified badge and tappable citation chips appear, each chip revealing the record value it was checked against.](docs/demo/clinical-copilot-demo.gif)

*75-second demo of this Week-1/Phase-1 baseline co-pilot — [MP4 version](docs/demo/clinical-copilot-demo.mp4)*

- **Data:** structured OpenEMR data via typed tools (medications, labs,
  encounters, vitals) — no retrieval, one tool call per planner turn.
- **Models:** local-only via Ollama (Qwen3-4B planner/extraction).
- **Trust:** a deterministic verification layer (`verification.py`) rolls
  every claim up to `verified` / `partially_verified` / `blocked`; a
  `quarantine.py` containment step keeps untrusted tool-result text from
  steering the planner's next tool call.
- **Auth:** full OAuth2 `authorization_code` + PKCE + SMART-launch +
  introspection flow, **built and proven live end-to-end** (restricted role
  403 vs admin 200 on the same endpoint) — shipped **flag-gated OFF**
  (`copilot_per_user_token_enabled`), a deliberate owner choice, dev
  token-bridge as the default.
- **Evals:** 31-case suite, category pass/fail, 0.8065 pass rate.
- Full detail: `docs/ARCHITECTURE.md`, `docs/AUDIT.md` (base-platform
  security audit), `docs/IMPLEMENTATION_PLAN.md`.

### Week 2 (Phase 2, this repo's own build)

Extends the same trust story to **unstructured source documents** and a
**public clinical-guideline corpus**, without changing the Week-1 access
model:

- **Document ingestion** — lab PDFs and intake forms, with **local** vision
  extraction (no cloud OCR/vision; PHI never leaves the machine); a strict
  "not found" (never a guessed value) contract on any illegible field.
- **Hybrid RAG** — sparse (BM25) + dense retrieval with a **local** reranker
  over a small, public, non-PHI clinical-guideline corpus.
- **Supervisor/worker orchestration** — a hand-rolled supervisor (not a
  third-party graph framework — see `docs/W2_ARCHITECTURE.md` §"Why not
  LangGraph") delegating to two workers (intake-extractor,
  evidence-retriever) with explicit, logged handoffs, worker spans parented
  under the supervisor span.
- **Citation contracts** — machine-readable citations on every
  document-sourced clinical claim (`{source_type, source_id,
  page_or_section, field_or_chunk_id, quote_or_value}`); a page-level
  click-to-source view backs every citation chip (a pixel bounding-box
  overlay was deliberately not shipped — see
  `services/copilot-agent/app/documents.py`'s module docstring for the
  honest capability call behind that decision).
- **Single-engine inference migration** — the answer/extraction/reranker LLM
  moved onto `llama-server` (llama.cpp), Qwen3-8B-Q5_K_M, with a
  degraded-aware `/ready` endpoint distinguishing "process is up" from "can
  actually serve a request" per dependency (OpenEMR, Ollama, llama-server,
  trace store).
- **Per-patient fact citations** — a bound patient's own already-ingested
  document facts are surfaced as citable evidence in live `/chat` turns
  (`DocumentFactIndex`).
- **Eval gate** — a 50-case golden set (9 Phase-1 categories + 5 new
  boolean rubrics: `schema_valid`, `citation_present`, `factually_consistent`,
  `safe_refusal`, `no_phi_in_logs`) wired as a **PR-blocking CI gate** that
  fails on any category regression.
- Full detail: `docs/W2_ARCHITECTURE.md` (target architecture, schemas,
  SLOs, data lineage), `docs/MODEL_AND_HARDWARE_SELECTION.md` (why
  Qwen3-8B-Q5_K_M, and the measured guideline-citation ceiling on
  8 GB VRAM under the tightened provenance-AND-semantic-support
  definition of "verified"), `docs/W2_AUDIT.md` (**Stage-4 hardening
  checklist results**, below).

## Stage-4 hardening (P4.1, #25)

`docs/W2_AUDIT.md` runs a hardening checklist — built from Phase 1's own
security-review conventions (`docs/AUDIT.md`'s finding categories +
`CLAUDE.md`/`docs/TEST_PLAN.md`'s security-review criteria, since Phase 1
never shipped one file literally titled "Stage-4 checklist") — against the
Week-2 surface specifically: ingestion, retrieval/reranking, the extended
verification layer, the supervisor/workers, the `llama-server` engine and
its `/ready` check, and per-patient fact citations.

**Result: 7 of 9 categories clean pass, 1 disclosed-by-design tradeoff
(fail-soft retrieval/fact-lookup on errors), and 1 real coverage gap in an
already-mitigated prompt-injection boundary — fixed in this PR** with a new
deterministic regression test proving the existing containment (an explicit
anti-injection system prompt, a tool-less extraction LLM, and fail-closed
citation verification) holds against an adversarial document-fact quote.
The remaining piece — a live-recorded eval case for the same scenario —
needs eval-schema changes plus dev-GPU recording infra and is tracked as
follow-up [#70](https://github.com/franciszver/agentforge-2-evidence-agent/issues/70).
No finding blocks Stage 4; see `docs/W2_AUDIT.md` for the full checklist,
findings, and remediation notes.

## AgentForge series

1. [agentforge-1-clinical-copilot](https://github.com/franciszver/agentforge-1-clinical-copilot) — Clinical Co-Pilot Foundation
2. **agentforge-2-evidence-agent** — Multimodal Evidence Agent & Document RAG *(this repo)*
3. agentforge-3-redteam — Adversarial Security & Red-Team Platform

**Live operational plan:** the
[Multimodal Evidence Agent project board](https://github.com/users/franciszver/projects/3)
tracks all work — stages as milestones, tasks as issues, one issue = one
branch = one PR.
