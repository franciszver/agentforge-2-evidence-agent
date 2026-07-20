# Clinical Co-Pilot — Week 2 (Phase 2) Architecture

- **Status:** Draft (P2.1). Phase 2 is mid-build; this document sets the
  target architecture, schemas, SLOs, and operational posture that the rest
  of Phase 2 (Stages 3–6) builds toward. Where a number is measured, its
  source is the Phase 1 capacity run (`docs/ARCHITECTURE.md` §"Capacity
  reality"); where it is a target set in advance of Phase 2's own
  measurement, it is marked "to be validated in P3G.4 (#24) perf baselines."
- **Related:** `docs/ARCHITECTURE.md` (Phase 1 / Week-1 architecture, frozen
  at v1.0 — this document extends it, does not replace it),
  `planning/PLAN.md` (frozen Phase 2 plan), `docs/USERS.md` (persona and
  UC1–UC5 use cases every capability below traces to), `docs/TEST_PLAN.md`
  (eval methodology, PR Definition of Done),
  `docs/MODEL_AND_HARDWARE_SELECTION.md` (how the llama.cpp answer model and
  its serving config were chosen), `docs/W2_AUDIT.md` (hardening pass over
  the Week-2 surface, including the llama-server `/ready` check).

## Summary

Phase 1 shipped a verification-first co-pilot that answers questions against
*structured* OpenEMR data — medications, labs, encounters — with every claim
independently re-checked against the raw record. Phase 2's job is to extend
that same trust story to *unstructured source documents*: a scanned lab
report or an intake form a patient hands over on paper. The architectural
bet carried forward unchanged is that a small model plus deterministic
verification beats a bigger model you trust blindly; Phase 2's addition is
that the same discipline — schema-constrained extraction, "not found" over
fabrication, and a citation that a deterministic checker can re-validate —
has to hold for *vision* extraction from a PDF, not just for text-shaped
tool output.

Two new capabilities sit on top of the Phase 1 core. First, **document
ingestion**: a local vision-language model (VLM) extracts structured facts
(lab results, intake demographics) from an uploaded PDF, stores the source
document in OpenEMR, and persists the derived facts as FHIR resources — with
a citation contract that points back to the exact page/section and quoted
value, so a clinician can click a claim and see the source pixels. Second,
**hybrid retrieval-augmented answering**: a small, public, non-PHI
clinical-guideline corpus is indexed with sparse (BM25) and dense
(embedding) retrieval, reranked locally, so the agent can answer
guideline-grounded questions ("is this within reference range for her age?")
without ever calling out to a cloud model or a cloud search API.

Both capabilities run fully local, on the same zero-egress Docker network
Phase 1 established. Inference is now split across two local engines rather
than the single Ollama instance Phase 1 used — see
§"Inference Engines: the Final Partial-Consolidation Architecture" for the
full picture and why it landed this way. Neither engine is a new trust
boundary: both run on the same internal, no-egress Docker network Phase 1
established. The two capabilities are wired together, and to the existing
single-tool-planner core, by a small in-house **supervisor** that hands work
to two workers (intake-extractor, evidence-retriever) with explicit, logged
handoffs — extending Phase 1's correlation-id tracing rather than
introducing a new orchestration framework.

## Architecture Diagram

```mermaid
flowchart TB
    subgraph net["Internal Docker network — NO internet egress (unchanged from Phase 1)"]
        direction TB

        subgraph openemr["OpenEMR (PHP)"]
            api["REST / FHIR API"]
            docstore["Source document storage"]
        end

        subgraph agent["Agent service (FastAPI)"]
            sup["Supervisor (extends Phase 1 planner loop)"]
            intake["Worker: intake-extractor"]
            retr["Worker: evidence-retriever"]
            verify["Verification layer (extended: document citations)"]
            ingest["Ingestion tool: attach_and_extract()"]
        end

        vlm["Ollama — document-ingestion VISION ONLY<br/>(qwen2.5vl:7b)"]
        llm["llama-server — answer / extract / rerank LLM<br/>(Qwen3-8B-Q5, default answer engine)"]
        embed["llama-server (2nd instance, --embedding)<br/>nomic-embed-text-v1.5, GPU"]
        idx["Hybrid index: BM25 + vector store (guideline corpus)"]
        trace["Trace store (SQLite, no PHI) — extended with worker spans"]
    end

    sup -->|handoff, logged| intake
    sup -->|handoff, logged| retr
    intake --> ingest
    ingest -->|VLM call, Ollama-only, sequential swap w/ llm on min-spec| vlm
    ingest --> docstore
    ingest -->|FHIR resource write| api
    retr --> idx
    idx --> embed
    idx -->|LLM-as-judge rerank| llm
    sup --> llm
    sup --> verify
    verify --> trace
    intake --> trace
    retr --> trace
```

This extends, not replaces, the Phase 1 diagram in `docs/ARCHITECTURE.md`
(agent internals, Ollama boundary, network egress posture, dev token bridge)
— those trust boundaries are unchanged by Phase 2 and are not repeated here.
The engine split shown here (llama.cpp for text + embeddings, Ollama for
vision only) is the **final** architecture — see
§"Inference Engines: the Final Partial-Consolidation Architecture" for how
it got here and why full consolidation onto a single engine (the original
epic #52 goal) stopped short of vision.

## Week-1 Baseline vs Week-2 New

| Capability | Week-1 (v1.0, inherited) | Week-2 (Phase 2, this document) |
|---|---|---|
| Data source | Structured OpenEMR data via typed tools (meds, labs, encounters, vitals) | + Unstructured source documents (lab PDF, intake form) via vision extraction |
| Retrieval | None — planner calls typed tools directly | + Hybrid RAG (BM25 + dense) with local rerank over a public guideline corpus |
| Orchestration | Single planner loop, one tool call per turn | + Supervisor delegating to 2 workers (intake-extractor, evidence-retriever), explicit logged handoffs |
| Citation contract | `{tool_call_id, record_id, field, asserted_value}` against cached tool output | + `{source_type, source_id, page_or_section, field_or_chunk_id, quote_or_value}` against source documents, with a visual PDF bounding-box overlay |
| Verification | Deterministic re-check against raw structured tool results | + Extended to re-check document-sourced claims against extracted facts; same fail-closed "not found" discipline for unreadable/absent fields |
| Eval gate | 31-case suite, category pass/fail, 0.8065 pass rate | + 50-case golden set, boolean rubrics (`schema_valid`, `citation_present`, `factually_consistent`, `safe_refusal`, `no_phi_in_logs`), PR-blocking CI gate |
| Observability | Correlation IDs, request/verdict-level trace store | + Per-encounter cost/latency breakdown (tool sequence, per-step latency, token usage, retrieval hits, extraction confidence), worker spans as children of the supervisor span |
| Models served | Qwen3-4B (planner, quarantine, extraction), on Ollama | Qwen3-8B-Q5 (answer/extract/rerank) + nomic-embed on **llama.cpp** (`llama-server`, two instances); + 7B-class VLM (document-ingestion vision) remains on **Ollama** — all local. See §"Inference Engines: the Final Partial-Consolidation Architecture" |
| Access control | Per-user OAuth2/PKCE/SMART built, proven live, flag-gated OFF | Unchanged — same flag, same posture (see §"Data Model, Lineage, and Access Control") |

## Phase 1 Debt Disposition

Carried forward from the Phase 1 freeze checklist (`planning/PLAN.md` §"Phase
1 freeze status"). None of these block Phase 2; each is a deliberate,
disclosed deferral:

| Item | What it is | Why deferred | When addressed |
|---|---|---|---|
| CORS TODO (`src/RestControllers/Subscriber/CORSListener.php:55-57`) | `onKernelResponse` echoes the request's `Origin` header back verbatim into `Access-Control-Allow-Origin` rather than checking it against an allowlist, with an inline `@TODO: review security implications if we need to tighten this up` | Inherited OpenEMR base behavior, needed as-is to keep public API clients (SMART apps, third-party integrations) working; tightening it is a base-platform change orthogonal to the Co-Pilot's own scope | Base-platform hardening, not scheduled within Phase 2 or Phase 3 — tracked as an OpenEMR-base concern, revisit if/when the base project addresses it upstream |
| `#185` async token introspection | `introspect_token`'s sync `httpx` call briefly occupies the FastAPI event loop when the per-user OAuth flag is on and the introspection cache misses | Low urgency: cached after first hit, and Phase 1's own capacity run showed the workload is GPU-inference-bound (~0.15 req/s), so a few-ms loop stall is negligible in practice | Before the per-user flag is ever flipped ON by default (see §"Data Model, Lineage, and Access Control") — not required for Phase 2's document/RAG work, which doesn't touch this code path |
| `#172` input-side PHI deterrent | Guidance/placeholder text on the feedback-comment field so clinicians aren't tempted to type patient identifiers into a 👎 comment (defense-in-depth on top of the existing export-side scrub from `#157`) | Non-urgent: local-first, trusted-network deployment keeps the residual risk low, and the export-side scrub already closes the public-repo leak surface | Opportunistic UI follow-up; not a Phase 2 dependency since Phase 2 introduces its own no-PHI-in-logs CI check (§"Testing Strategy", §Stage 3 gate) covering the new ingestion/retrieval surface directly |
| `#175` encounter-planner tool-selection | For a "what's going on with her right now" question over a patient with only a stale encounter (no problems/vitals), the planner never calls `get_encounters`, so the Phase 1 recency-notice mechanism has nothing to attach to; kept as an honest `xfail` | Root cause is a 4B-model tool-selection weakness, not a code defect — fixing it needs prompt work plus live re-recording with uncertain payoff on this model tier | Revisit if Phase 2's supervisor/worker split changes the planner's tool-selection prompting anyway; otherwise stays a documented, measured `xfail` per its own acceptance criteria |
| P5.6 fresh-clone validation | Confirm the stack runs end-to-end from a fresh clone using README instructions alone, no undocumented steps | N/A — already resolved | **Closed** at Phase 1 freeze; Phase 2 inherits a validated fresh-clone baseline and must keep it green as new services (VLM, embeddings, reranker, hybrid index) are added to the compose stack |

## Schemas

### Extraction Schemas

Both extraction targets follow the Phase 1 pattern established by
`ollama_client.extract` (schema-constrained decoding into Pydantic) — the
VLM's output is never accepted as free text, only as a validated instance
of one of these models. Any field the model cannot read from the document
is `None` ("not found"), never guessed; this is the extraction-side
equivalent of Phase 1's citation-checker fail-closed discipline. Like every
existing tool schema in `app/schemas/`, these extend `ToolSchemaModel`
(`app/schemas/common.py`), which provides `frozen=True` + `extra="forbid"`
so malformed extraction output fails fast rather than silently dropping
fields.

```python
class Citation(ToolSchemaModel):
    source_type: Literal["lab_pdf", "intake_form"]
    source_id: str            # stored source-document identifier
    page_or_section: str      # e.g. "page 2" or "Section: Medications"
    field_or_chunk_id: str    # which extracted field this citation backs
    quote_or_value: str       # the literal text/value the model read


class LabResultFact(ToolSchemaModel):
    test: str
    value: str | None
    unit: str | None
    reference_range: str | None
    collection_date: str | None   # ISO date if legible, else None
    abnormal_flag: Literal["H", "L", "A", "N"] | None
    citation: Citation


class IntakeFormFact(ToolSchemaModel):
    demographics: dict[str, str | None]   # name, dob, sex — as legible
    chief_concern: str | None
    medications: list[str]
    allergies: list[str]
    family_history: list[str]
    citation: Citation
```

### Citation Contract

Every clinical claim sourced from a Week-2 document — not just the raw
extraction, but any answer the agent later builds on top of it — carries the
same shape:

```python
class DocumentCitation(BaseModel):
    source_type: Literal["lab_pdf", "intake_form"]
    source_id: str
    page_or_section: str
    field_or_chunk_id: str
    quote_or_value: str
```

This is the document-sourced counterpart to Phase 1's
`{tool_call_id, record_id, field, asserted_value}` structured-data citation
(`docs/ARCHITECTURE.md` §"Verification Design (the flagship)"). The Phase 1 verification
layer (`verification.py`) is extended, not replaced: it builds its
`(tool_call_id, record_id, field) -> value` index for structured tool
results exactly as before, and gains a parallel index keyed on
`(source_id, field_or_chunk_id) -> quote_or_value` for document facts. A
claim citing a document field is verified the same way — normalized
equality against the extracted (not re-inferred) value — so the "partial
grounding is not grounding" rule applies identically to both citation
shapes.

### Pixel bbox citation grounding: page-level fallback confirmed permanent (issue #42)

P3.7 (the citation-overlay feature, `app/documents.py`) chose page-level
source navigation (open the real source PDF at the cited page, show the
citation's literal quote) over drawing a pixel bounding box on the source
image, based on a single ad hoc probe: qwen2.5vl:7b's bbox grounding was
accurate on a clean vector render of the lab-report fixture but drifted
onto the wrong table column/row once scan-realistic rotation + noise +
JPEG were applied. Issue #42 asked whether that one-fixture finding held
up against a larger, more rigorous evidence base before treating it as
permanent.

**Expanded measurement (P3.9c).** 16 fixture variants — the lab-report and
intake-form fixtures' first pages, degraded along 11 realistic-scan axes
(rotation 0.5°/1.2°/2.0°/-1.5°, Gaussian noise, salt-and-pepper noise, JPEG
q30, brightness/contrast shift, Gaussian blur, a photocopier toner-fade
band, and the original rotation+noise+JPEG combo), each rendered at the
same scale `app/ingestion.py` uses (pypdfium2, `scale=2.0`) — were sent to
qwen2.5vl:7b via the same Ollama `/api/chat` surface `OllamaClient` uses,
with the same schema-constrained-decoding mechanism `OllamaClient.extract()`
uses in production (not free-form chat, which produces malformed JSON on
this model often enough to itself be a confound). 60 field-level bbox
requests were scored against ground-truth pixel locations computed
analytically from the fixture-generation layout (row/column geometry is
known at PDF-draw time; a verified point-transform carries each
ground-truth box through rotation).

Results, by document:

| Document | Layout | Field-level bbox requests | Center-in-truth-box rate | Mean IoU | Max IoU |
|---|---|---|---|---|---|
| Lab report (6-column table) | dense, narrow columns | 48 (4 fields × 12 variants) | 6/48 = **12%** (statistically indistinguishable from chance; **0/4 = 0% even on the undegraded clean render**) | 0.005 | 0.068 |
| Intake form (2-column label:value) | simple, wide columns | 12 (3 fields × 4 variants) | 12/12 = **100%**, unaffected by degradation | 0.048 | 0.147 |

The table-layout finding is *worse* than P3.7's original one-fixture result:
grounding fails even on the clean render, not just under degradation — the
model consistently anchors the returned box's left edge on the **Test**
(label) column rather than the **Value** column it was asked for,
independent of scan noise. The simple two-column form does reliably locate
the right *row* (center-in-truth-box holds across every degradation
variant tested), but every returned box is far too small relative to the
true cell (max IoU 0.147) — even the "correct" case would render a
visibly wrong-shaped, wrong-proportioned rectangle, not a truthful tight
box around the value.

**Verdict: (b), confirmed as the permanent honest choice.** The expanded
evidence does not merely fail to overturn P3.7's finding — it strengthens
it into a documented, repeatable pattern (column-anchoring drift on
multi-column layouts, undersized boxes even where row-location succeeds)
across 4x the document/variant coverage of the original probe. Per the
project's no-fabrication thesis ("never draw a box at guessed
coordinates"), pixel-bbox citation grounding on qwen2.5vl:7b is not
implemented; the page-level fallback (`app/documents.py`'s
`GET /documents/{source_id}` + the module UI's cited-page navigation) is
confirmed as the permanent design for document citation grounding, not a
placeholder awaiting a follow-up. The measurement methodology (fixture
degradation + ground-truth transform + scored bbox probe) is committed at
`services/copilot-agent/scripts/measure_bbox_grounding.py` so a future
recommended-tier VLM swap (§"Recommended-tier revisit" below) can be
re-scored against the same fixture set and thresholds rather than starting
from scratch.

### Migration Notes

No changes are expected to any Week-1 (Phase 1) database table or schema.
Week-2 artifacts are additive:

- **Source documents** (the uploaded lab PDF / intake form) are stored in
  OpenEMR's own document-management facilities, not a new Co-Pilot-owned
  table — Phase 2 deliberately does not duplicate OpenEMR's document store.
- **Extracted facts** are persisted as standard FHIR resources
  (`Observation` for lab results, `Patient`/`Condition`/`AllergyIntolerance`
  -shaped data for intake facts) through OpenEMR's existing REST/FHIR API,
  the same system of record Phase 1's tools already read from — Phase 2
  writes through it rather than around it, consistent with Phase 1's "the
  Co-Pilot owns none of the system of record" design principle.
- The hybrid-RAG guideline corpus and its BM25/vector index are a new,
  separate, non-PHI store (public documents only) and carry no migration
  concern for existing patient data.

## Testing Strategy

Follows `docs/TEST_PLAN.md`'s red-first discipline, extended per code type:

- **Unit tests** — schema validation (malformed/partial VLM output coerces
  to `None` fields, never raises or fabricates), citation-index construction
  for the new document-keyed index, supervisor→worker handoff event
  emission. Fast, no model calls, no database.
- **Integration tests** — `attach_and_extract()` end-to-end against fixture
  PDFs (including at least one deliberately low-quality/unreadable scan, to
  prove the "not found" path), hybrid retrieval against a small fixture
  corpus, worker spans correctly parented under the supervisor span in the
  trace store.
- **Eval gate (CI-blocking)** — the 50-case golden set (§Stage 3 gate in
  `planning/PLAN.md`) with boolean rubrics (`schema_valid`,
  `citation_present`, `factually_consistent`, `safe_refusal`,
  `no_phi_in_logs`), extending Phase 1's `evals/` harness and assertion
  vocabulary rather than starting a parallel one.
- **Record/replay for model inference — never live in CI.** Exactly as
  Phase 1 established with `ollama_replay.py` for the planner/extraction
  LLM: VLM extraction calls and reranker scoring are recorded once against
  the real local models and replayed deterministically in CI. No CI run
  ever depends on a live Ollama call — this keeps the gate fast, offline,
  and reproducible from the repo alone.
- **Red-first per code type** (`docs/TEST_PLAN.md` §1): a failing PHPUnit
  case for any OpenEMR-side document storage change, a failing pytest case
  for any agent-side extraction/retrieval/verification change, a failing
  eval case for any new golden-set category, committed before the
  implementation that makes it pass.

## SLOs

Two targets, both explicitly provisional — set from Phase 1's measured
hardware ceiling and stated engineering judgment, not from Phase 2's own
measurement, which is P3G.4 (#24)'s job:

- **p95 ingestion latency for a 2-page lab PDF, minimum spec: ≤ 45 seconds.
  To be validated in P3G.4 perf baselines.** Justification: Phase 1's P5.1
  capacity run measured a single-request Qwen3-4B chat turn at 10.3s on
  this exact hardware (RTX 5060 Laptop, 8GB VRAM). A document-ingestion
  request additionally requires (a) a cross-engine swap — freeing the
  llama-server-resident answer LLM and loading Ollama's VLM — sequential
  load/unload on an 8GB card, not concurrent residency, per the
  minimum-spec operating mode in §"Reference Hardware & Model Tiers" — which is the
  dominant new cost and is budgeted generously at 10-20s given no measured
  swap time yet exists for this hardware/model pair; (b) VLM decode over a
  7B-class model, budgeted at roughly 2x a 4B call's latency for a
  same-length structured extraction (~15-20s); (c) a second swap back if the
  same turn also needs the planner/extraction LLM. 45s keeps headroom over
  the sum of these budgeted pieces while remaining inside the "a clinician
  can wait for one document to process" range this use case implies (a
  one-time per-encounter action, not a per-question latency like UC1-UC4).
- **Retrieval hit-rate for the golden queries: ≥ 80% top-5 relevant-chunk
  recall on the guideline corpus. To be validated in P3G.4 perf baselines.**
  Justification: the corpus is deliberately small and curated to the
  co-pilot's actual use cases (not general-web-scale retrieval), so a
  well-tuned hybrid BM25+dense+rerank pipeline over a few dozen documents
  should comfortably clear the bar a much larger, noisier corpus would
  struggle with; 80% is set as a defensible floor rather than an
  aspirational ceiling, leaving room to raise it once P3G.4 has real
  numbers rather than picking a number the corpus's actual size can't yet
  justify.

Both numbers remain unmeasured targets after P3G.4 (#24) — see
§"Perf Baselines vs Phase 1 (P3G.4 / #24)" below for what P3G.4 measured
instead and why these two specifically were left as documented gaps rather
than requiring a new live measurement run. This is the same posture Phase 1
took with its own capacity expectations: "research-informed priors, to be
measured in Phase 5" (`docs/ARCHITECTURE.md` §"Capacity reality") before
that run actually happened.

## Perf Baselines vs Phase 1 (P3G.4 / #24)

Same discipline as the rest of this document: a measured number is never
blended with a projected one, and every number below states which it is.
Reuses already-measured figures from `docs/ARCHITECTURE.md` §"Capacity
reality" (Phase 1) and `docs/MODEL_AND_HARDWARE_SELECTION.md` (Phase 2's own
answer-model selection) rather than standing up a new live measurement run —
per this issue's own instruction not to block on live infra.

**Measured now:**

- **Single-request answer latency, Phase 1 vs Phase 2 (same hardware: RTX
  5060 Laptop, 8 GB VRAM).** Phase 1's P5.1 capacity run measured `qwen3:4b`
  at **10.3s p50** for one live `POST /chat` request. Phase 2's chosen answer
  model — Qwen3-8B-Q5_K_M, 16k ctx, q8_0 KV, flash-attn — measures **~27s**
  per query (`docs/MODEL_AND_HARDWARE_SELECTION.md` §"The committed
  production baseline"). That is roughly **2.6× Phase 1's single-request
  baseline**: the direct, expected latency cost of moving from a 4B to an
  8B-class model on the same 8 GB card, traded for materially better
  citation reliability (Phase 2's own measured ceiling on this hardware,
  under the issue #47/#81 tightened definition — provenance AND semantic
  support — is low; see that doc's "Why the ceiling on this hardware is low,
  and getting lower under the tightened definition"). Both figures are
  measured-vs-measured, not measured-vs-projected.
- **Eval-suite citation reliability.** The `citation_present` category's
  committed production baseline (`evals/category_baseline.json`) is the one
  Phase 2 metric with no Phase 1 equivalent to compare against (Phase 1 had
  no fail-closed verbatim-citation verification layer). It is CI-guarded by
  the P3G.2 (#22) regression gate (≤5% pass-rate drop per category) and, as
  of this issue, also monitored live via the dashboard's new eval-regression
  alert (see `app.dashboard_alerts`).

**Gaps — not measured here, deliberately not blocked on:**

- **Concurrency scaling for the Phase 2 answer model.** Phase 1's P5.1 run
  measured `qwen3:4b` p50 latency scaling **10.3s → 34.0s → 59.3s** at
  1/5/10 concurrent chats, saturating at ~0.15 req/s with VRAM flat at
  ~3.2 GB (compute-bound, not memory-bound). Phase 2 has not re-run that same
  `capacity_run.py` harness against Qwen3-8B-Q5_K_M — doing so needs a live
  concurrent-load run against the dev stack, the kind of new heavy live run
  this issue's instructions say not to require. Expect the same
  compute-bound scaling shape (single-GPU inference throughput is still the
  ceiling), proportionally worse given the 8B model's higher per-token cost.
- **Document-ingestion p95 latency** (SLO above: ≤45s for a 2-page lab PDF).
  Still the budgeted target stated in §"SLOs", not a measurement. Producing
  a real number needs a live `attach_and_extract()` run against the resident
  VLM on the dev stack — not done here for the same reason as the item
  above.
- **Retrieval hit-rate** (SLO above: ≥80% top-5 relevant-chunk recall). Still
  a target, not a measurement. `evals/test_retrieved_chunks_faithful_to_corpus.py`
  checks that each eval case's canned `retrieved_chunks` fixture is
  verbatim-faithful to the real corpus — a data-integrity guard, not a
  recall measurement — and no golden-query set with labeled
  known-relevant-chunks exists yet in `evals/` to compute recall against.
  Building that golden set is new eval-fixture work, out of this issue's
  scope (documenting what's measurable now, not building new measurement
  infrastructure).

## Reference Hardware & Model Tiers

Two tiers, kept strictly separate — measured numbers are never blended with
projected ones:

| | Minimum spec (dev machine — all measured numbers) | Recommended spec |
|---|---|---|
| GPU | RTX 5060 Laptop, 8GB VRAM | Single 24-48GB+ card |
| RAM | 32GB | (not constraining at this tier) |
| Model residency | **Sequential VLM⇄LLM swap** — the Ollama-served VLM and the llama.cpp-served answer LLM (Qwen3-8B-Q5) are not resident simultaneously on the 8GB card; the llama.cpp embedding instance (nomic-embed, ~274MB) is small enough to stay GPU-resident alongside whichever of the two is currently loaded | **All models resident simultaneously** — VLM, answer/extract/rerank LLM, and embeddings all on-GPU, no swap latency |
| VLM tier | 7B-class (e.g. `qwen2.5vl:7b`, Ollama) | Larger VLM (model swapped in as a config value, not a code change) — see §"Inference Engines" for why a bigger card, not just a bigger model, is the actual unlock |
| Numbers | Measured, on this exact hardware (Phase 1's P5.1 run; Phase 2's own ingestion numbers per §"SLOs", pending P3G.4 (#24)) | Projections only — no recommended-tier hardware exists in this project; stated as directional, never presented as measured |

**Model tiering is a config value, not a code fork.** The same extraction
code, the same Pydantic schemas (§"Schemas"), and the same no-fabrication contract
("not found" over guessing) apply regardless of which VLM is configured —
swapping in a larger recommended-tier model changes extraction quality and
latency, not the safety invariant the verification layer depends on. This
mirrors Phase 1's own local-inference philosophy: the trust story lives in
the deterministic verification layer, not in betting on a particular
model's reliability.

## Inference Engines: the Final Partial-Consolidation Architecture

Epic #52 set out to retire Ollama entirely and run every model role on a
single local engine, llama.cpp (`llama-server`) — one fewer inference
engine in the supply-chain/attack surface for a hospital security review.
That consolidation happened for **text and embeddings**, but stopped short
of **vision**. This section states where it landed and why, closing out
epic #52 as a deliberate partial consolidation rather than the original
full-retirement goal.

### What runs where today

| Role | Engine | Model | Why |
|---|---|---|---|
| Answer / extraction / claim-extraction LLM | **llama.cpp** (`llama-server`) | Qwen3-8B-Q5_K_M, q8_0 KV, flash-attn | Default answer engine (owner decision, P3.10e / #73); best verified-citation config that fits the 8GB minimum spec — see `docs/MODEL_AND_HARDWARE_SELECTION.md` |
| Reranker (LLM-as-judge) | **llama.cpp** (same `llama-server` instance as above) | Qwen3-8B-Q5_K_M | Reuses the answer-LLM client rather than standing up a third model — no separate reranker model to serve |
| Embeddings (dense retrieval) | **llama.cpp** (a second, dedicated `llama-server` instance, `--embedding` mode) | nomic-embed-text-v1.5 (f16 GGUF) | Migrated off Ollama per P3.10b (#76); small enough (~274MB) to stay GPU-resident alongside whichever LLM is currently loaded |
| Document-ingestion **vision** (VLM) | **Ollama** | qwen2.5vl:7b | **Stays on Ollama** — see "Why vision stays on Ollama" below |

Both engines run on the same internal, no-egress Docker network Phase 1
established (`docker-compose.copilot.yml`); the appliance/no-egress security
thesis holds identically for both — this is a *partial engine
consolidation*, not a partial trust-boundary one. On the 8GB minimum-spec
card, the answer LLM and the VLM are not GPU-resident simultaneously (the
VLM is loaded only for the duration of a document-ingestion call, per
§"Reference Hardware & Model Tiers"); the small embedding instance fits
alongside either.

### Why vision stays on Ollama

The P3.10c spike (issue #77) tested moving the VLM onto llama.cpp's `mtmd`
multimodal stack, the last step needed for full consolidation. The verdict
was **not viable**, for a reason that overrides the supply-chain-surface
argument for consolidating in the first place:

- **Qwen2.5-VL-7B Q4_K_M on llama.cpp (build b10068, `mtmd`) reproducibly
  fabricated** on the ingestion no-fabrication contract's own test case — a
  redacted/obscured lab field (`collection_date` on a deliberately
  unreadable page-2 fixture) came back as an invented date
  (`2026-06-01`) instead of the correct `null`. This reproduced across
  three serving configurations, including `--mmproj-offload
  --image-min-tokens 1024`, ruling out a quick flag fix.
- **The same model on Ollama returns the correct `null` on that same
  fixture, every time.** The fabrication is specific to the llama.cpp
  `mtmd` code path at this quantization, not to the model weights
  themselves.
- **Q8_0 (8.1GB) does not fit the 8GB minimum-spec card**, so the spike
  could not test whether a higher-precision quantization closes the
  grounding gap — that question is open, not answered "no."
- Latency was ruled out as the blocker (fixable to ~8.3s/page via
  `--mmproj-offload`); the disqualifying finding is purely the
  no-fabrication violation.

The no-fabrication contract (`app/ingestion.py`'s "not found" over
guessing, verified end-to-end in `docs/W2_AUDIT.md` item 2) is
**non-negotiable** — it is the same fail-closed discipline the citation
verifier applies to structured claims (§"Citation Contract"), extended to
vision extraction. A VLM path that invents values on an unreadable field is
disqualified regardless of any consolidation benefit, so vision remains on
Ollama and epic #52 closes as a documented partial consolidation rather
than forcing the last step through.

### Recommended-tier revisit

This is stated as an evidence-backed limitation of the current
minimum-spec hardware and llama.cpp build, not a permanent architectural
ceiling. Two paths could reopen the question at the recommended tier
(§"Reference Hardware & Model Tiers"):

- A larger GPU with enough VRAM to hold Qwen2.5-VL at **Q8_0 or f16**
  fully resident, to test whether higher-precision weights close the
  grounding gap the 8GB card couldn't test.
- A different VLM better suited to llama.cpp's `mtmd` stack, evaluated
  against the same no-fabrication fixture before any migration is
  attempted.

Either revisit is future recommended-tier work, not scheduled Phase 2/3
work — consistent with this document's discipline of never presenting a
recommended-tier projection as a committed roadmap item.

## Orchestration

Phase 2 extends Phase 1's hand-rolled planner loop into a **supervisor**
that delegates to two workers — **intake-extractor** (drives
`attach_and_extract()` and the VLM) and **evidence-retriever** (drives
hybrid RAG) — rather than adopting a third-party graph-orchestration
framework. This is an owner decision, not a default: `planning/PLAN.md`
and issue P3.5 (#17) name extending the custom loop as the primary path, with
LangGraph as a documented fallback only if handoff/trace requirements
prove unwieldy in practice — that fallback decision point is P3.5.

### Why not LangGraph

- **Smaller supply-chain surface.** Zero third-party orchestration
  dependencies means one fewer package (and its transitive dependency
  tree) to vet for a hospital security review, and one fewer thing for
  Phase 3's red-team to have to reason about as an attack surface. The
  in-house supervisor is a few hundred lines the team wrote and can
  fully audit; a graph-orchestration library is not.
- **Fully auditable.** Phase 1's entire trust story rests on being able to
  point at deterministic, inspectable code for every safety-critical path
  (quarantine, verification, patient binding). A hand-rolled supervisor
  keeps handoffs, span parenting, and control flow in that same
  fully-owned, fully-readable code, rather than inside a framework's
  internal state machine.
- **Consistency with the Phase 1 philosophy.** `planner.py`,
  `quarantine.py`, and `verification.py` are already a deliberately
  dependency-light, hand-rolled core; introducing a heavyweight
  orchestration framework for Phase 2 alone would fragment that
  architecture rather than extend it.

LangGraph remains the documented fallback if the custom supervisor/worker
split proves unwieldy — that reassessment is P3.5's explicit decision
point, not a decision this document forecloses.

### Handoff diagram

```mermaid
flowchart LR
    U["User question"] --> S["Supervisor<br/>(extends planner.py)"]
    S -->|handoff, logged, span_id| I["Worker: intake-extractor"]
    S -->|handoff, logged, span_id| E["Worker: evidence-retriever"]
    I -->|result + citations, logged| S
    E -->|result + citations, logged| S
    S --> V["Verification layer"]
    V --> A["Answer + badge + citation chips"]
```

Every handoff is a logged event carrying a correlation id and a span id;
worker spans are recorded as **children of the supervisor span**, extending
Phase 1's correlation-id tracing (`correlation.py`) rather than introducing
a parallel tracing scheme. A live run's trace therefore shows the full
supervisor→worker→supervisor chain end-to-end from a single correlation id,
the same way Phase 1's trace store already reconstructs a full conversation
from logs alone.

## Data Model, Lineage, and Access Control

### Lineage

```
source PDF page  →  extracted field (VLM, schema-constrained)  →  FHIR resource (OpenEMR)  →  cited claim (agent answer)
```

Each arrow is independently inspectable: the source document is stored
in OpenEMR and addressable by `source_id`; the extracted field carries a
`Citation` (§"Extraction Schemas") pointing back to the exact page/section and quoted text;
the FHIR resource write carries the same field identifiers through to
OpenEMR's system of record; and any later claim built from that resource
carries a `DocumentCitation` (§"Citation Contract") that the verification layer re-checks
against the extracted value, not a re-inference. A clinician can therefore
click any citation chip and trace it all the way back to the source pixels
on the original page.

### Access control posture

Unchanged from Phase 1, restated here because it governs Week-2 artifacts
too: the full per-user OAuth2 `authorization_code` + PKCE + SMART-launch +
introspection flow is **built and proven live end-to-end**
(`docs/ARCHITECTURE.md` §"Trust Boundaries" boundary 3, issue #124) — a
restricted role gets 403 where an admin gets 200 on the identical endpoint.
It ships **flag-gated OFF** (`copilot_per_user_token_enabled`), a deliberate
owner choice to avoid a one-way door and a consent-step demo friction, with
the shared dev-token-bridge identity as the default. Week-2's ingestion and
retrieval tools call the same `OpenEmrClient` under the same flag, so they
inherit this posture unchanged — no new ACL surface, no new flag. Flipping
`copilot_per_user_token_enabled` ON is the **first item in Path to
Production** (§"Path to Production"), exactly as it was in Phase 1.

## Incident Response & Backup/Recovery

The deployment model is a single local appliance (Phase 1's "run where the
data lives" thesis, unchanged) — RPO/RTO are set for a single-node
appliance, not a multi-region service:

- **RPO: last nightly volume snapshot.** Justification: this is a
  single-developer/single-clinician appliance, not a multi-user production
  system with continuous write load from many concurrent users — a nightly
  snapshot bounds worst-case data loss to one day's uploaded documents and
  extracted facts, which is an acceptable trade for the operational
  simplicity of not running continuous replication on a laptop-class
  device.
- **RTO: time to `docker compose up` plus volume restore.** Justification:
  recovery is deliberately simple by design — no multi-node failover to
  coordinate, no external dependency to re-provision (everything, including
  the VLM/embedding/reranker models, is a local volume) — so RTO is bounded
  by container startup time plus however long the snapshot restore itself
  takes, typically minutes, not the hours a distributed system's failover
  choreography would need.
- **What gets backed up:** the OpenEMR MySQL volume (patient records, FHIR
  resources, uploaded source documents), the Ollama models volume
  (`ollamamodels` — vision-only VLM weights, per
  §"Inference Engines: the Final Partial-Consolidation Architecture"), the
  llama.cpp models volume (`llamacppmodels` — the pinned answer-LLM and
  embedding GGUFs), and the guideline corpus's hybrid index (small, non-PHI,
  but backed up for continuity rather than requiring a rebuild) — so a
  restore doesn't require re-downloading gigabytes of model weights for
  either engine.
- **Degraded modes.** Extends Phase 1's existing `/health` vs `/ready`
  distinction (`docs/ARCHITECTURE.md` §"Architecture Diagram"): `/health` reports the FastAPI process is up; `/ready` reports
  whether the process can actually serve a request end-to-end. Phase 2
  extends `/ready`'s checks to both engines — if Ollama (vision) is
  unreachable, or either `llama-server` instance (answer/extract/rerank,
  embeddings) is unreachable or hasn't finished loading its GGUF, `/ready`
  reports not-ready for the affected capability (document ingestion,
  retrieval, or answering) while the other capabilities can continue to
  report ready independently, so an outage on one engine degrades
  gracefully rather than taking down the whole agent.

## TCO

Extends Phase 1's TCO approach (`docs/ARCHITECTURE.md` §"TCO") with the
Week-2 hardware and volume dimension: **one dedicated on-prem GPU card vs a
recurring cloud vision/OCR API**, at a plausible hospital document volume.
Kept rough and honest, consistent with Phase 1's framing — these are
estimates with a stated basis, not bills.

- **Local (one-time + amortized).** A dedicated on-prem GPU sized to the
  recommended tier (§"Reference Hardware & Model Tiers", 24-48GB+, all models resident) is a one-time
  hardware cost, amortized over its useful life exactly as Phase 1's TCO
  section amortized the dev laptop — no per-document API fee, no
  per-page-processed billing, and the same zero-egress property (no PHI
  ever leaves the appliance to reach a cloud vision API) that made Phase
  1's local-inference thesis attractive in the first place.
- **Cloud comparison.** A recurring cloud document/vision API (e.g. a
  commercial OCR-plus-extraction service) typically bills per page or per
  document processed. At a plausible hospital volume — say, a few hundred
  lab reports and intake forms per day across a clinic — that recurring
  per-page cost compounds continuously, the same way Phase 1's TCO section
  found for cloud LLM tokens: the gap widens linearly with volume, and the
  local GPU's fixed cost is recovered faster the higher the document
  volume, exactly mirroring Phase 1's "weeks to a few months" break-even
  framing for the LLM side.
- **Honest caveat.** Unlike Phase 1's LLM-token estimate (which had a
  measured token-size basis from prompt templates), Phase 2 has no measured
  per-document extraction cost yet — this section is a first-principles
  estimate pending Phase 2's own cost-per-encounter instrumentation
  (`planning/PLAN.md` Stage 3 item (3e)), which is a Phase 2 requirement precisely
  because Phase 1's cost tracking was never wired into `/chat` either. Once
  that instrumentation lands, this section should be revisited with
  measured per-document token/latency figures rather than the estimate
  above.

## Path to Production

Mirrors Phase 1's Path-to-Production structure (`docs/ARCHITECTURE.md`
§"Path to Production"), reordered so the highest-narrative-weight item leads:

1. **Flip `copilot_per_user_token_enabled` ON** — see §"Data Model, Lineage,
   and Access Control" for what this unlocks and why it is off by default.
   Highest-leverage item, listed first for that reason.
2. **TLS everywhere.** Today's internal Docker network traffic (including
   the new VLM/embedding/reranker calls) is unencrypted, acceptable only
   because it never leaves a single host; a real deployment needs TLS
   termination in front of every hop.
3. **Secrets automation.** Model-serving and OpenEMR credentials move from
   local config/`.env` to a managed secrets store as part of any real
   deployment, rather than remaining developer-machine conventions.
4. **Backup automation.** The nightly-snapshot RPO in §"Incident Response &
   Backup/Recovery" is a manual/cron convention today; production hardening
   turns it into an automated, monitored, tested restore procedure rather
   than an assumed cron job.
5. **Monitoring.** `planning/PLAN.md` Stage 3 item (3e) and Stage 3 gate
   called for extending the existing hand-rolled trace-store dashboard
   (Phase 1) with extraction-failure rate, retrieval latency, and
   eval-regression alerting (>5% category regression) — P3G.4 (#24) added
   all three alert rules to `app.dashboard_alerts`, tested, with
   eval-regression wired to real committed data (`app.dashboard_eval_history`)
   and the other two implemented-but-dormant pending a future issue that
   records dedicated ingestion/retrieval trace-store spans (no such span
   type exists yet — see that module's docstring for the precedent this
   already follows for the P4.6 tool-failure-rate alert, dormant pending
   #149). Still outstanding for a real multi-node production deployment:
   BAA-covered hosting, VPC, and HA, exactly as Phase 1's own
   Path-to-Production item 4 describes.

## Use-Case Traceability

Every Week-2 capability above traces to `docs/USERS.md`'s UC1-UC5:

- **Document ingestion (§"Week-1 Baseline vs Week-2 New", §"Schemas")** —
  extends UC1 (pre-visit brief: "what changed since I last saw her" can now
  include a just-uploaded lab PDF) and UC3 (lab trend recall: a scanned
  report's values become queryable the same way structured lab-tool data
  already is).
- **Hybrid RAG over the guideline corpus (§"Week-1 Baseline vs Week-2 New",
  §"Orchestration")** — supports UC2 (medication safety: guideline-grounded
  context alongside the existing allergy/interaction check) without adding a
  new device or workflow step.
- **Citation contract + PDF overlay (§"Citation Contract", §"Data Model,
  Lineage, and Access Control")** — extends the trust mechanism UC1-UC4
  already depend on (every Phase 1 answer is citable) to document-sourced
  claims, so the "verified, before the door opens" standard from
  `docs/USERS.md`'s persona section holds for the new source type too.
- **Supervisor/worker orchestration (§"Orchestration")** — infrastructure supporting all
  of the above; no user-facing behavior change on its own, but the logged
  handoffs are what let UC1's "answer spans encounters, labs, meds, and
  notes" synthesis now also span uploaded documents coherently.
