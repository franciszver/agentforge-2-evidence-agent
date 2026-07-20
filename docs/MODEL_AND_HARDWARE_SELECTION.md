# Clinical Co-Pilot — Model & Hardware Selection (Phase 2 Evidence Agent)

- **Status:** Final for Phase 2's evidence-retrieval answer path. Documents how the local answer model and its serving configuration were chosen for the reference/minimum hardware tier, and why the verified-citation ceiling that choice produces is honestly **low** on this hardware tier — not a defect to be tuned away, but the capability boundary of an 8 GB GPU at this model class, now measured against a **tightened** definition of "verified" (see immediately below).
- **Live re-verification (issue #100):** the citation_present baseline stated throughout most of this document as "5/12" was measured by re-judging one frozen, historical answer recording per case, and that recording was never checked against what a fresh live draw actually produces today. It has now been checked: none of the 5 non-xfail cases reliably reproduce `verified` on a fresh live draw (1 pass in 42 draws total). The honest, currently live-reproducible number was **0/12** — see "Live re-verification (issue #100)" below for the full method and data before trusting any "5/12" reference elsewhere in this document as current. **Issue #108's fix (below, "Issue #108 follow-up") has since moved this number to 2/12** — `a1c-target-question` and `lithium-nsaid-question` now reliably reach `verified` live.
- **"Verified" tightened (issue #47/#81):** earlier revisions of this document measured verification as provenance only — a citation counted once its quote was confirmed a real, verbatim substring of the source. That leaves a gap: a model can pair a real quote with a claim the quote doesn't actually support. Issue #47 named this gap; issue #81 shipped the fix (`app/semantic_support.py`) ON by default and re-measured through it. **A citation now counts only when BOTH hold: the quote is verbatim-real (provenance) AND an LLM judge (the same Qwen3-8B-Q5 engine) affirms the quote actually supports the claim's prose (semantic support).** Every count below uses this tightened definition unless explicitly marked otherwise.
- **Method — two different measurements, not one, stated plainly so neither is mistaken for the other:**
  1. **The exploratory hardware-selection sweep** (below, "The ladder actually tested"): a throwaway measurement harness run once per config, best-of-attempts, used only to choose *which* model/config to adopt. Its 8B-Q5 row (4/12) predates the semantic-support gate entirely (provenance-only) and is kept here only as the historical selection rationale.
  2. **The committed, CI-reproducible production baseline** (below, "The committed production baseline"): the chosen 8B-Q5 config's citation reliability as measured through the REAL eval pipeline (`evals/cases/citation_present/`) — production `LlamaServerClient`, the gate's own effect isolated by re-judging the STABLE, already-committed answer recordings (rather than re-drawing fresh answers) with a live semantic-support judge call, fail-closed verification (gate ON) — the number `pytest evals/` reproduces on every run and the P3G.2 gate now guards.
- **Run-to-run variance is real and disclosed, not folded into the baseline.** A full fresh re-draw of the answer pipeline (planner + extraction, not just the judge) is noisy on this 8B model: repeated from-scratch re-recording attempts of all 12 cases have produced results ranging from 1/12 to 6/12 provenance-passing answers across draws, purely from planner/extraction variance unrelated to the semantic-support gate. Because of that, the committed production baseline below deliberately measures the gate's effect in isolation — by re-judging the stable, already-committed recordings rather than re-drawing new answers — so the published number is a property of one fixed, committed set of answers, not a lucky or unlucky sample of the answer pipeline itself. The variance is real and is disclosed here rather than smoothed over.
- **Related:** `docs/W2_ARCHITECTURE.md` §"Reference Hardware & Model Tiers" (the same minimum-vs-recommended framing, applied there to the VLM/ingestion path) and §"Inference Engines: the Final Partial-Consolidation Architecture" (this model is what llama.cpp serves for answer/extract/rerank, alongside embeddings — vision stays on Ollama); `docs/ARCHITECTURE.md` §"Capacity reality" (the Phase 1 precedent for measuring on real hardware before publishing a spec).

## Reference (minimum) hardware

**RTX 5060 Laptop GPU, 8 GB VRAM; 32 GB system RAM.**

This is the same dev/demo hardware Phase 1 measured its chat-latency and concurrency ceilings on (`docs/ARCHITECTURE.md`). Phase 2 keeps that precedent: the published minimum spec is whatever was actually measured on this box, not an estimate. If a configuration doesn't fit or doesn't perform acceptably here, it isn't the minimum spec — it's the recommended-tier story instead (see below).

## The ladder actually tested (exploratory hardware-selection sweep)

Four model/config combinations were run end-to-end against the same 12 eval cases, using a throwaway measurement harness (best-of-attempts, not the committed eval suite) whose sole purpose was choosing which model/config to adopt for this hardware tier:

| Model / config | Engine | Verified citations | Latency | Hardware fit |
|---|---|---|---|---|
| Qwen3-4B @ 16k ctx | Ollama | 1/12 | ~22 s | Fits GPU |
| Qwen3-8B-Q4_K_M @ 16k ctx, q8_0 KV, flash-attn | llama.cpp | 3/12 | ~23 s | Fully GPU-resident, 6.27 GB / 8 GB |
| **Qwen3-8B-Q5_K_M @ 16k ctx, q8_0 KV, flash-attn (chosen)** | llama.cpp | **4/12** (exploratory sweep) | ~27 s | Fully GPU-resident, 6.63 GB / 8 GB |
| Qwen3-30B-A3B (MoE, Q4) @ 16k ctx, q8_0 KV, expert-offload (`--cpu-moe`) | llama.cpp | 0/12* | median 152 s, max 300 s (timeout) | Experts in CPU RAM; only 1.9 GB VRAM |

The chosen row's exploratory figure (4/12) is a **provenance-only** measurement from before the semantic-support gate existed, and is superseded by the committed, CI-reproducible production baseline below, which now measures provenance AND semantic support together. The two numbers aren't contradictory: the sweep was a quick best-of-attempts comparison used only to rank the four candidates against each other, and it correctly ranked 8B-Q5 above 8B-Q4 and 4B; the production baseline is the number this project actually publishes and enforces going forward, under the tightened definition.

*The 30B-A3B row is a confound, not a citation verdict, and is called out explicitly rather than left to imply the model can't cite at all: 10 of 12 cases failed with extraction timeouts before completion — the verbose reasoner exhausted its generation budget on internal reasoning before ever emitting the constrained JSON output the verifier needs, so only 2 of 12 cases ran to completion. It is disqualified primarily on **latency**, not citation quality: 3–12× slower than the 8B, because a 30B model's experts, offloaded to CPU RAM on an 8 GB card, are memory-bandwidth-bound (roughly 36 tokens/s). No serving configuration fixes this at this hardware tier — the bottleneck is CPU↔GPU memory bandwidth, not a tunable parameter.

## The committed production baseline (superseded by issue #100 — see below)

Once 8B-Q5 was selected, its real citation reliability was measured the way every other eval category in this project is measured: the 12 `citation_present` eval cases (`evals/cases/citation_present/`) run through the actual production pipeline (planner → claim extraction → fail-closed provenance re-validation → the issue #47 semantic-support judge), against the production `LlamaServerClient` for every one of those roles. To isolate the gate's own effect from the answer-pipeline variance described above, issue #81's methodology re-judges the STABLE, already-committed answer recordings (the 6 that already passed provenance-only re-validation) rather than re-drawing fresh answers: for each, the recorded planner/extraction calls replay unchanged, and a live semantic-support judge call is made and appended to that same recording — so the full gate-ON pipeline still replays byte-for-byte offline in CI, but the number reflects the gate's effect on a fixed, known answer set rather than a fresh draw's luck.

**This methodology's blind spot, found by issue #100: a "stable" recording is only as good as the draw that produced it, and that draw is never re-checked against what the live model actually does today.** The section immediately below documents what happened when it finally was checked. The historical result is kept here for provenance, but it no longer describes current live reality — see "Live re-verification (issue #100)" for the number this project now stands behind.

**Historical result (superseded): 5 of 12 cases genuinely passed under the tightened definition, as of the #81 re-judge.** Of the 6 cases that passed provenance-only re-validation on the stable, committed recordings, 5 were judge-confirmed semantically supported and stayed non-`xfail`: `bp-stage2-question`, `a1c-target-question`, `dual-antiplatelet-question`, `hypertension-lifestyle-followup-question`, `lithium-nsaid-question`. The 6th was honestly downgraded:

- **`statin-liver-monitoring-question` — the case issue #47/#81 specifically targeted — was downgraded to `xfail`.** Its committed recording carries exactly the provenance-without-support defect issue #47 was opened over: a real, verbatim quote ("routine ongoing liver-enzyme monitoring ... is usually not needed") paired with claims asserting the opposite (monitoring "is typically recommended" / the clinician "should review the specific statin"). The live semantic-support judge correctly called **both affected claims `not_supported`**, downgrading the verdict to `partially_verified`. This was **not** counted as a 6th supported citation — the gate did exactly what it was built to do, on the exact case it was built to catch. This part of the finding is unaffected by issue #100 and still stands.
- The other **6 cases remained documented `xfail`** for reasons unrelated to the gate — the same chart-data-only / extraction-decode patterns this document has tracked since the exploratory sweep and #58 (`lipid-panel-ldl-question`, `metformin-renal-monitoring-question`, `nsaid-ace-inhibitor-question`, `renal-function-ace-question`, `statin-ck-myopathy-question`, `warfarin-antibiotic-question`) — none of these ever had a valid document citation survive to reach the semantic-support judge in the first place. This part of the finding is also unaffected by issue #100.

## Live re-verification (issue #100): the 5/12 baseline did not reflect current live behavior

A deep-dive diagnostic found that re-running `bp-stage2-question` LIVE (a genuine fresh draw through `evals/runner/pipeline.run_case`, production `LlamaServerClient`, no replay) landed on `partially_verified` with zero `document_citations` every time. Because the committed baseline above is deliberately measured against a **frozen, already-committed recording** rather than a fresh draw (precisely to insulate it from the run-to-run variance disclosed earlier in this document), that frozen recording had never been checked against what a fresh live draw of the same question actually produces today. This section closes that gap: all 5 of the baseline's non-xfail cases were re-run live, repeatedly, against current `main` (post PR #94/#95/#98/#101).

**Method.** Each case was driven through the real pipeline the same way `evals/runner/record.py` does for recording, but without committing a recording after every run — 8 to 10 independent fresh draws per case, same production `LlamaServerClient`/model/settings the committed baseline uses, against the dev stack's live `llama-server` container (Qwen3-8B-Q5_K_M).

**Result: none of the 5 reliably reproduce `verified` today.**

| Case | Live draws | `verified` | Typical outcome |
|---|---|---|---|
| `bp-stage2-question` | 10 | 0/10 (0%) | `partially_verified`, zero document citations — chart-data-only answering |
| `a1c-target-question` | 8 | 1/8 (12.5%) | `partially_verified` — guideline chunk IS cited every run, but a co-occurring chart-data claim fails verification often enough to keep the verdict below `verified` |
| `dual-antiplatelet-question` | 8 | 0/8 (0%) | `blocked` (fail-closed), zero document citations — chart-data-only answering |
| `hypertension-lifestyle-followup-question` | 8 | 0/8 (0%) | `blocked` (fail-closed), zero document citations — chart-data-only answering |
| `lithium-nsaid-question` | 8 | 0/8 (0%) | `partially_verified` every run; the guideline chunk is cited in only 4/8 of those |

**Total across all 5 cases: 1 `verified` outcome in 42 independent fresh live draws.** This is not a one-off unlucky sample — it is overwhelming, repeated evidence that the committed 5/12 number was a property of one historical, no-longer-representative recording per case, not a number a clinician using the live system today should expect to see reproduced. Three of the five (`bp-stage2-question`, `dual-antiplatelet-question`, `hypertension-lifestyle-followup-question`) fail deterministically — same outcome on every single draw — which rules out ordinary sampling noise as the explanation; the model consistently does not surface/quote the retrieved guideline text for these questions under current main. The other two (`a1c-target-question`, `lithium-nsaid-question`) are less than fully deterministic but still fail the overwhelming majority of the time (87.5–100%).

**Action taken:** all 5 recordings (`evals/recordings/{bp-stage2-question,a1c-target-question,dual-antiplatelet-question,hypertension-lifestyle-followup-question,lithium-nsaid-question}.json`) have been re-captured against current `main` with a typical (non-passing) live draw, replacing the stale recordings, and each case is now honestly marked `xfail` in its YAML with a rationale describing exactly what was observed (see `evals/cases/citation_present/`). No recording was doctored to force either a pass or a fail — each re-capture used `evals/runner/record.py` against the real live model and, where a case showed some variance (`a1c-target-question`, `lithium-nsaid-question`), the captured recording matches the statistically typical (non-passing) outcome already established across the repeated live draws above, not a cherry-picked one.

**The honest committed number, updated: 0 of 12 `citation_present` cases currently pass.** All 12 cases are now documented `xfail`. `evals/category_baseline.json`'s P3G.2 gate treats an all-`xfail` category as a vacuous pass (nothing left to regress — see `CategoryStats.pass_rate`'s documented behavior), so CI stays green; the category's `_note` has been updated to state the honest 0/12 total plainly rather than let the gate's vacuous-pass numeric value imply otherwise.

This does not change the diagnosis already given above for *why* the ceiling is low on this hardware tier (model capacity, chart-data-only answering, extraction-decode limits) — if anything it confirms that diagnosis applies more broadly than the committed baseline previously showed. It changes only how much of that ceiling this project can currently claim to have measured a live model actually clearing.

### Claim-extraction citation routing bug fixed (issue #85) — a second, distinct mechanism found

Issue #100's `bp-stage2-question` deep-dive ("the model consistently does not surface/quote the retrieved guideline text") turned out to be imprecise for that one case: issue #85's follow-up live investigation found the model DOES attempt to cite the guideline chunk every time, but qwen3-8b puts the citation in `source_refs` (reconstructing the chunk's own `<doc_id>#<section-slug>` id across `tool_call_id`/`record_id`) instead of `document_citations` — a citation-routing defect in `app.extraction.ClaimExtractor`, not a "never tries" defect. `check_source_ref` correctly failed it `unknown_tool_call`, and `check_claim`'s AND-across-citations rule dragged the whole claim down with it (including two valid chart-data citations bundled in the same claim), which is why it looked identical to "zero document_citations, chart-data-only" from the outside. Fixed: the extractor now recognizes this specific, safely-identifiable shape and reclassifies it into a real `document_citations` entry before verification.

**This does not move `bp-stage2-question` off `xfail`.** With the routing fixed, the citation now reaches the semantic-support judge for the first time (it never did before, since the claim always failed provenance first) — and the judge correctly downgrades it `not_semantically_supported`: the model's own final-answer text calls 148/94 mmHg "elevated blood pressure," but the guideline's own thresholds put that reading in "Stage 2 hypertension." That is a planner answer-composition defect (the planner composes its free-text answer before evidence retrieval ever runs — see `app.chat`'s `planner.run()` vs. `evidence_retriever()` call ordering — so it never sees the guideline thresholds it's describing), not a claim/citation-assembly defect, and is out of issue #85's scope.

**Whether this same citation-routing mechanism explains the other 4 non-`bp-stage2` xfail'd cases in this baseline was not re-verified as part of issue #85** (`a1c-target-question`, `dual-antiplatelet-question`, `hypertension-lifestyle-followup-question`, `lithium-nsaid-question`) — issue #85's scope was `bp-stage2-question` only. Given the mechanism is a generic property of `app.extraction.ClaimExtractor` (not specific to the blood-pressure corpus text), it plausibly affects any of the other 4 cases where the model attempts a guideline citation at all; this is flagged here as a candidate explanation worth checking in a follow-up, not asserted as confirmed.

### Issue #106 follow-up: the routing bug does NOT reproduce elsewhere — a different mechanism found instead

The candidate flagged directly above was checked. All 4 of the other originally-flagged cases, plus 7 further `citation_present` cases already `xfail` for older reasons, were re-run live (`evals/runner/pipeline.run_case` / `evals/runner/record.py`, production `LlamaServerClient`, no replay; 3–11 fresh draws per case). **None of the 11 reproduce the `#85` misrouted-`source_refs` shape.** Inspecting the extractor's pre-coercion output directly (before `_coerce_misrouted_guideline_refs` runs) confirms this for every case that surfaces a citation at all: the model either emits a correctly-shaped `document_citations` entry on its own, or never attempts a guideline citation in any shape. `_coerce_misrouted_guideline_refs` was therefore left unchanged — there was no shape variant observed live to generalize it against, and inventing one to "future-proof" the coercion without a live-observed trigger would risk exactly the kind of speculative broadening the function's existing safety comments warn against.

Two cases turned out to be materially affected by #85's fix anyway, just not via the routing-bug mechanism: `dual-antiplatelet-question` and `hypertension-lifestyle-followup-question` still fail 6/6 draws exactly as #100 described (chart-data-only answering, zero guideline attempt) — unchanged. But `a1c-target-question` and `lithium-nsaid-question` now reliably (6/6 and 11/11 draws respectively) get their guideline citation into `document_citations` and past provenance re-validation — a real improvement over #100's 12.5%/50% figures. Both still fail to reach `verified`, for a **newly-identified, third mechanism**: the model restates the same guideline-backed fact as two separate claims in its answer (e.g. "...above the target range..." / "...is not at target."), both citing the identical chunk — and the semantic-support judge is inconsistent across the repeat, validating one occurrence while downgrading the other to `not_semantically_supported`. This is neither the `#85` routing bug nor the `#105` planner-composition/category-mismatch defect (the category language is correct in both cases); it looks like a claim-splitting / semantic-support-judge-consistency gap and is documented as a candidate for a new follow-up issue rather than fixed here. See `evals/cases/citation_present/a1c-target-question.yaml` and `lithium-nsaid-question.yaml`'s `xfail` rationales for the full per-case detail; both recordings have been re-captured against current main with a statistically typical draw.

Separately, `statin-ck-myopathy-question`'s recording — stuck on a stale qwen3:4b-era capture since issue #58 because `LlamaServerError` didn't extend `LLMEngineError` at the time and crashed the recorder — was re-recorded successfully (4/4 draws, no crash): a later, unrelated fix (`2483e3b`) already gave `LlamaServerError` the right base class. The case shows the same chart-data-only pattern as before, now confirmed on the current production model rather than the retired 4B one.

**The `citation_present` category total is unchanged at 0/12** — this follow-up sharpens *why* 2 of the 12 cases fail (citation-routing is fixed for them; a different judge-consistency gap is the actual blocker now) without changing the honest bottom line stated above.

### Issue #105 follow-up: planner/retrieval-ordering fix — the category-mismatch mechanism is fixed, but the case still does not verify live

Issue #85's follow-up above diagnosed `bp-stage2-question`'s remaining failure as an architecture defect: `app.chat`'s guideline-corpus retrieval ran strictly *after* `planner.run()` composed the final answer text, so the planner described 148/94 mmHg as "elevated blood pressure" from its own training priors while the guideline it went on to cite categorizes that same reading as "Stage 2 hypertension" — a genuine, verbatim citation attached to prose using the wrong category name for what it cited.

**Fix implemented:** retrieval now runs *before* the planner call (`app.chat._stream_chat`), and the retrieved guideline text is threaded into the planner's own answer-composition call (`Planner.run`/`run_streaming`'s new `guideline_excerpts` parameter, consumed by `_finalize_answer_streaming`) with an explicit instruction to use the guideline's own category name rather than substitute general terminology.

**Live re-verification (11 fresh draws, production `LlamaServerClient`, `evals.runner.pipeline.run_case`, no replay):** the category-mismatch mechanism is genuinely fixed — **11/11 draws** now compose the answer as "...falls into the category of Stage 2 hypertension," matching the guideline's own category name exactly, and the claim extractor correctly attaches a real, verbatim `document_citations` entry quoting `"Stage 2 hypertension: systolic 140 mmHg or higher OR diastolic 90 mmHg or higher."` — the citation passes provenance re-validation every time.

**The case still does not reach `verified`, in all 11 draws, for a *different* reason than before.** The semantic-support judge now downgrades the citation with a new objection: *"The QUOTE defines Stage 2 hypertension but does not provide the patient's blood pressure reading, so it does not support the specific claim about the patient's condition."* The judge wants the cited guideline excerpt to *also* restate the patient's own 148/94 mmHg reading, which a category-threshold reference document by design never will — that reading only ever lives in the chart-data citation on the claim's own `source_refs`, a separate citation on the SAME claim the judge is not shown as connected context. This is a **new, narrower, and out-of-scope defect**: the semantic-support judge's per-citation evaluation doesn't appear to account for a claim being jointly supported by a chart-data citation (the value) plus a guideline citation (the category the value falls in) — it evaluates the guideline citation in isolation and correctly, on that narrow question, finds it insufficient alone. Fixing that is a claim/citation-assembly or judge-prompt change, not a planner answer-composition one, and is out of this issue's (#105) stated scope (bp-stage2's category-name mismatch specifically) — flagged here as a candidate for a new follow-up issue rather than fixed as part of this PR.

**The `citation_present` category total remains 0/12.** `bp-stage2-question` stays `xfail` — its `xfail` rationale has been updated to describe this new mechanism rather than the now-fixed category-mismatch one, so a future reader isn't misled into thinking the case is blocked on a problem that no longer exists.

### Issue #108 follow-up: duplicate-claim judge inconsistency fixed — the category total moves to 2/12

Issue #106's follow-up above identified a third mechanism blocking `a1c-target-question` and `lithium-nsaid-question`: both cases reliably got their guideline citation into `document_citations` and past provenance, but the model's answer restates the same guideline-backed fact as two separate claims (e.g. "...above the target range..." / "...is not at target." for `a1c-target-question`; the interaction caution and a separate monitoring-advice sentence for `lithium-nsaid-question`), both citing the **exact same** `DocumentCitation` (identical `source_type`/`source_id`/`field_or_chunk_id`/`quote_or_value`, byte-for-byte — confirmed by inspecting the extractor's pre-judge claim list directly). The semantic-support judge (`app.semantic_support.apply_semantic_support`) judged each claim's citation with its own independent LLM call and scored the two inconsistently — typically validating one restatement while downgrading the other (usually the terser one) to `not_semantically_supported` — purely from call-to-call variance on the paraphrased wording, not from any real difference in the underlying evidence. That stripped one of the two claims, holding the verdict at `partially_verified` in `a1c-target-question`'s case 5/6 draws and deterministically in `lithium-nsaid-question`'s case 11/11 draws (see issue #106's numbers above).

**Root cause, confirmed by tracing `app.semantic_support`, not guessed:** `apply_semantic_support` iterated per-``ClaimCheckResult`` and issued one independent judge call per claim's `DocumentCitation`, with no shared state across claims. Two claims citing the identical evidence therefore always received two independent LLM calls, with no mechanism to keep them consistent — the model's own judge nondeterminism on near-duplicate wording was the entire source of the disagreement, since the evidence backing both calls was provably byte-identical.

**Fix (`app/semantic_support.py`, PR closing issue #108):** `apply_semantic_support` now runs in two passes. The first groups every currently-`VALID` `DocumentCitation` result across ALL already-passing claims by its exact evidence identity (`source_type`, `source_id`, `field_or_chunk_id`, `quote_or_value` — a byte-for-byte match, never a fuzzy/paraphrase merge) and judges each distinct identity **exactly once**, against the combined text of every claim that cites it. The second pass applies that one shared verdict back to every citation result sharing the identity. Two claims citing genuinely identical evidence can therefore no longer land on different verdicts — and, as a side effect, fewer judge calls are made whenever this duplication occurs. Citations with distinct identities (even quoting the same source document) are still judged fully independently, exactly as before this fix; the change is scoped to this one narrow shape and does not touch `app.verification`, `app.extraction`'s claim-building, or `app.verdict`.

**Live re-verification (production `LlamaServerClient`, `evals.runner.pipeline.run_case`, no replay, inside the dev stack's `development-easy-agent-1` container against `development-easy-llama-server-1`):**

| Case | Live draws (post-fix) | `verified` |
|---|---|---|
| `a1c-target-question` | 8 | 7/8 (87.5%) |
| `lithium-nsaid-question` | 8 | 8/8 (100%) |

`a1c-target-question`'s one non-`verified` draw failed for a **different, pre-existing, unrelated mechanism**: that draw's answer never attempted the guideline citation at all (zero `document_citations`, chart-data-only) — the same planner/extraction variance already disclosed throughout this document, not a judge-consistency defect. It is not a regression introduced by this fix, and is not claimed as fixed here.

**Relationship to issue #111, checked explicitly as instructed by that issue's "possibly related" note:** #111 (`bp-stage2-question`'s semantic-support judge evaluating a guideline citation in isolation from a *different*, sibling chart-data citation on the *same* claim) is a **different root cause from #108's**, confirmed by tracing both through the same module. #108's shape is two *separate claims* citing the *identical* `DocumentCitation` object — fixable by deduplicating identical evidence before judging, with no change to what the judge sees for any single citation. #111's shape is *one claim* with *two different* citations (a category-threshold guideline citation and a chart-data value citation), where the judge only ever sees one citation at a time and has no way to know the claim's other citation already established the value — that requires the judge to see a claim's full citation set together, a materially different (and larger) change than #108's exact-identity dedup. Fixing #108 did not fix, and was never expected to fix, #111 — `bp-stage2-question` remains blocked by #111's mechanism, unchanged by this PR, and is explicitly out of this issue's scope.

**Both cases' eval YAMLs updated:** `evals/cases/citation_present/a1c-target-question.yaml` and `lithium-nsaid-question.yaml` have had their `xfail` rationale removed (replaced with a plain comment describing the fix and the live-verification numbers above) and fresh, honest `verified` recordings committed (`evals/recordings/{a1c-target-question,lithium-nsaid-question}.json`) — the first live attempt at re-recording each case landed on `verified` for both, consistent with the now-typical (87.5%/100%) live outcome, not a cherry-picked draw.

**The `citation_present` category total moves from 0/12 to 2/12.** `a1c-target-question` and `lithium-nsaid-question` are the first two cases in this document's history to reliably reach `verified` against a genuinely fresh, non-replayed live draw (not merely a stable historical recording, per issue #100's methodology critique). The other 10 cases are unaffected by this fix and remain `xfail` for the reasons already documented above (`bp-stage2-question`: #111's isolated-citation-judging mechanism; `dual-antiplatelet-question`/`hypertension-lifestyle-followup-question`: chart-data-only answering, zero guideline attempt; the remaining 6: chart-data-only/extraction-decode patterns tracked since the exploratory sweep and #58).

## Measured mid-tier addendum (RTX 3060 12GB desktop)

**This section is measured mid-tier evidence, not a minimum-spec update.** Every
number below carries the label **measured mid-tier (RTX 3060 12 GB desktop)**
and must never be blended with the RTX 5060 Laptop 8 GB numbers above — same
convention `docs/W2_ARCHITECTURE.md` uses for its own recommended-tier VLM
figures (stated directional/measured separately, never averaged into the
minimum-spec baseline). The hardware was a separate desktop-class machine
(12 GB VRAM, 128 GB system RAM); no local file paths or operator-machine
details are reproduced here beyond that.

**Fresh draw vs. stable recording, again.** The (now-superseded, see issue
#100 above) 5/12 baseline was a re-judge of stable, already-committed answer
recordings — deliberately insulated from planner/extraction variance so it
was a reproducible number the P3G.2 gate could guard. Every number in this
addendum is instead a **fresh single draw** of the full pipeline (planner →
extraction → provenance → semantic-support judge) on the desktop, run once
per case with no re-rolls — the same *kind* of measurement issue #100 later
ran repeatedly (8-10 draws/case) on the laptop tier and found mostly failing.
The two measurement styles answer different questions and are not
apples-to-apples with each other, exactly as already disclosed for the 14B
run below.

### Results table

| Model / config | Params / quant | citation_present | Mean latency | Extraction errors | VRAM / RAM |
|---|---|---|---|---|---|
| Qwen3-8B-Q5_K_M (pre-fix, 4 fresh draws) | 8B, Q5_K_M | **2/12** (identical on all 4 draws) | 26.0–30.8 s | 2–3 / 12 | 7.6 GB VRAM |
| Qwen3-14B-Q4_K_M | 14B, Q4_K_M | 4/12 | 33.3 s | 2 / 12 | 10.7 GB VRAM |
| Qwen3-30B-A3B (`--n-cpu-moe 32`) | 30B MoE, Q5_K_M | 3/12 | 57.6 s | 2 / 12 | 9.9 GB VRAM |
| Qwen3.6-35B-A3B (`--n-cpu-moe 30`) | 35B MoE, Q5_K_M | 5/12 | 81.9 s | 3 / 12 | 10.3 GB VRAM |
| gpt-oss-120b (`--n-cpu-moe 32`) | 120B MoE, Q8_K_XL | 5/12 | 220.6 s | **0 / 12** | 10.5 GB VRAM + ~60 GB CPU RAM |
| Qwen3-8B-Q5_K_M (post-fix, 1 fresh draw) | 8B, Q5_K_M | **4/12** | 25.6 s | 0 / 12 | 7.6 GB VRAM |

All rows use the same 12 `citation_present` eval cases, the same tightened
verified-citation bar (provenance AND semantic support), and the same engine
self-judging pattern as the committed baseline's methodology.

### The determinism finding

Four independent fresh draws of the production 8B-Q5 model, back-to-back on
this desktop (temp 0, `--parallel 1`, no sampling stochasticity), landed on
**exactly 2/12 every time** — the same 2 cases passed and the same 2–3 cases
hard-errored on extraction in each draw. This is a materially different
finding from the 1/12–6/12 scatter this document already discloses for
fresh full-pipeline re-draws on the laptop, and it is stated separately
rather than folded into that variance language: on this hardware/config,
repeat draws were not noisy, they were flatly reproducible at a number
*below* the committed 5/12 (which, as already explained above, is a
stable-recording re-judge, not a fresh draw — the two were never measuring
the same thing).

After PR #94/#95 merged a pipeline fix (`LlamaServerClient`'s
claim-extraction call given its own 2560-token budget, separate from chat's
1536-token budget, fixing a real truncation bug), one further fresh 8B-Q5
draw scored **4/12 at 25.6 s mean — faster and better than every pre-fix
draw**. The causal story behind that jump is not yet fully closed: 4 cases
changed outcome between the pre-fix and post-fix draws (3 improved, 1
regressed), where only 1 case had been predicted to flip from the
token-budget diagnosis alone. Issue #93 (planner tool-call nondeterminism, a
separate uncitable-claim-handling behavior) remains open and may explain
further movement in these numbers.

### Honest verdict

**No model swap tested here beat the pipeline-fixed 8B-Q5 result.** The
single most effective intervention measured this session was a bug fix, not
a bigger model — 14B, 30B-A3B, 35B-A3B, and gpt-oss-120b all either scored
lower or cost dramatically more latency (up to 9x) than the post-fix 8B-Q5's
4/12 at 25.6 s, and none reach the committed baseline's 5/12 without paying
a latency penalty that disqualifies them for interactive use. gpt-oss-120b
is the one model with zero extraction errors in this series, but its 220.6 s
mean is 7–8x over the latency envelope this project targets, disqualifying
it despite that reliability signal. The 30B-A3B result (3/12, 57.6 s, same
extraction-error signature as its original laptop rejection) reproduces the
prior rejection's failure mode on materially better hardware — this is
**confirmatory evidence for that decision**, not new information
contradicting it (see "The ladder actually tested" above).

This addendum is not a final verdict on model selection: issue #93 is open,
and these numbers may move again once it resolves. It also does not cover
the VLM/document-ingestion path — a separate Q8_0 vision-fabrication retest
(issue #90) is out of scope here and, if its result isn't already reflected
in `docs/W2_ARCHITECTURE.md`'s vision-tier section, is a follow-up for that
document, not this one.

## The two engine-level levers that decide what fits

**q8_0 KV-cache quantization + flash-attn (llama.cpp).** This is the enabling configuration for the chosen model. Without KV-cache quantization, an 8B model's key/value cache at a 16k context window does not fit alongside the model weights inside 8 GB of VRAM; q8_0 quantizes the cache to roughly a quarter of its FP16 footprint, and flash-attn keeps the attention computation itself memory-efficient at that context length. Together they're what lets Qwen3-8B-Q5_K_M run **fully GPU-resident** at 6.63 GB / 8 GB — no CPU offload, no swap latency.

**MoE expert-offload (`--cpu-moe`).** This is the lever that makes a 30B-parameter model fit on an 8 GB card at all — only 1.9 GB of VRAM is used, with the mixture-of-experts layers living in CPU RAM and dispatched per-token. It's an elegant trick for the memory constraint, but it does not solve the latency constraint: every token that routes through a CPU-resident expert pays CPU memory-bandwidth cost, and that cost dominates at this GPU tier. Fitting in VRAM and being usable for an interactive query are two different bars, and the 30B-A3B configuration clears only the first one here.

## Why the ceiling on this hardware is low, and getting lower under the tightened definition

The exploratory sweep establishes a monotonic pattern up to the point where hardware fit breaks: 4B → 1/12, 8B-Q4 → 3/12, 8B-Q5 → 4/12 (exploratory, provenance-only). Larger models cite more reliably, model capacity is the binding constraint on reliable verbatim guideline citation, and 8B-Q5 is the largest model this hardware can hold fully GPU-resident at a usable context length. Stepping up further — to the 30B-A3B — does fit in VRAM via expert-offload, but at a latency (2.5–5 minutes per query, with a third of cases timing out entirely) that makes it unusable for interactive clinical use regardless of what its citation rate would be if every case ran to completion.

Tightening "verified" to require semantic support as well as provenance (issue #47/#81) accounts for exactly **one** of the 7 non-passing cases in the *historical* committed baseline (`statin-liver-monitoring-question`, downgraded from a provenance-only pass to `partially_verified` — see above). The other 6 non-passing cases reflect the same underlying capacity ceiling this document has documented from the start — an 8B model at this quantization does not reliably surface, quote, and correctly reason about retrieved guideline text in the same turn as chart-data tool calls — independent of the gate. Issue #100's live re-verification (above) found that ceiling applies even more broadly than the historical baseline showed: the 5 cases that baseline counted as passing turn out to reproduce that same chart-data-only / co-occurring-claim failure pattern on a fresh live draw, nearly every time.

That leaves a genuinely low number as the honest ceiling for this hardware tier under the tightened definition: not a bug in the verification layer, not a tuning gap in the prompt, but the direct consequence of (a) the largest model that fits an 8 GB card at interactive latency, now also required to (b) get the SEMANTICS of a citation right, not just its provenance. The fail-closed verifier and judge are doing their job correctly here — reporting the model's real reliability at this capacity, not padding the number with unverifiable or unsupported claims.

This is stated plainly rather than downplayed: the historical committed baseline reached **5 of 12** cases under the tightened bar — down from the earlier 6/12 provenance-only figure by exactly the one case the gate was built to catch — but issue #100's live re-verification found that number does not reproduce on a fresh live draw of those same 5 cases (1 `verified` outcome across 42 fresh draws total; see "Live re-verification (issue #100)" above). The honest, currently-reproducible number was **0 of 12** until issue #108's fix (see "Issue #108 follow-up" above): with the semantic-support judge's duplicate-claim inconsistency fixed, `a1c-target-question` and `lithium-nsaid-question` now reliably reach `verified` live, moving the honest number to **2 of 12**. **Separately, and disclosed rather than hidden:** a full fresh re-draw of the answer pipeline (not just the judge) is noisy on this 8B model — repeated from-scratch re-recording attempts have landed anywhere from 1/12 to 6/12 provenance-passing answers across draws, purely from planner/extraction variance; issue #100's finding is consistent with the low end of that already-disclosed range, now measured with much more data (42 draws) than a single re-roll. The 2/12 issue #108 unlocked does not contradict that ceiling discussion — it removed a verification-LAYER inconsistency bug, not a model-capacity limitation, from two cases that were already reliably surfacing and provenance-passing their guideline citation.

## Minimum vs. recommended tier

| | Minimum spec (measured, this doc) | Recommended spec |
|---|---|---|
| GPU | RTX 5060 Laptop, 8 GB VRAM | Larger single card with enough VRAM to hold a bigger model fully resident at the same context length |
| Model | Qwen3-8B-Q5_K_M, q8_0 KV, flash-attn | A larger dense or MoE model sized to the larger card, with experts (if MoE) also GPU-resident rather than CPU-offloaded |
| Verified citations (provenance AND semantic support) | 2/12 (measured, live-reproducible — see "Issue #108 follow-up" above; was 0/12 per "Live re-verification (issue #100)" before that fix; the historical stable-recording baseline showed 5/12, but that did not survive a fresh live re-draw; fresh answer-pipeline re-draws vary 1/12–6/12, see the run-to-run variance caveat above) | Expected higher — not measured, since no recommended-tier hardware exists in this project; stated as directional only, per the same discipline `docs/W2_ARCHITECTURE.md` applies to its own projected numbers |
| Latency | ~27 s | Expected comparable or better, since the constraint driving the 30B-A3B's latency (CPU-offloaded experts) would not apply once a larger card holds the whole model in VRAM |

The mechanism is the same one `docs/W2_ARCHITECTURE.md` uses for the document-ingestion VLM: **model tier is a config value, not a code fork.** Swapping in a larger model at a larger-VRAM recommended tier changes citation reliability and latency; it does not change the verification layer's fail-closed contract. A clinician reading an answer on either tier gets the same guarantee — every surviving citation was both verbatim-verified against the guideline corpus AND judged to actually support the claim it's attached to — the two tiers differ only in how often the model succeeds at producing a citation that clears both bars in the first place.

## Summary

The chosen configuration — Qwen3-8B-Q5_K_M, 16k context, q8_0 KV-cache, flash-attn, fully GPU-resident on an 8 GB card — is the largest model this reference hardware can serve at interactive latency. Under the tightened definition of "verified" (provenance AND semantic support, issue #47/#81), a re-judge of the stable, already-committed answer recordings with a live semantic-support judge historically showed it clearing that bar on **5 of 12** guideline-citation eval cases — one case lower than the earlier 6/12 provenance-only figure, exactly the one case (`statin-liver-monitoring-question`) the gate was built to catch. **Issue #100 found that historical number does not hold up against a fresh live draw**: re-running all 5 of those cases live, repeatedly (42 draws total), reproduced `verified` only once. The honest, live-reproducible number for this hardware tier was **0 of 12** (see "Live re-verification (issue #100)" above) with all 12 `citation_present` cases documented `xfail` — until issue #108's fix (see "Issue #108 follow-up" above) resolved the semantic-support judge's duplicate-claim inconsistency, moving `a1c-target-question` and `lithium-nsaid-question` to a reliable, live-reproducible `verified` and the honest total to **2 of 12**. This sharpens rather than reverses the document's core conclusion: the capacity ceiling on this hardware tier is real for the remaining 10 cases, and it is lower than the historical baseline suggested; #108's 2 cases were never blocked by that capacity ceiling in the first place — they were blocked by a verification-layer consistency bug, now fixed. It is also the reason a recommended tier exists: a bigger GPU removes the capacity ceiling this document identifies, at the cost of hardware this project does not require to run.
