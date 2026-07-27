# Issue #192 injection-battery results — index

This directory holds **three separate measurement runs** of the same
152-payload injection battery (`tests/issue_192_injection_payloads.py`, 76
QUOTE_OR_FACTS-channel + 76 CLAIM_TEXT-channel payloads) against both LLM
judges (`app.semantic_support`, `app.source_ref_relevance`). They measure
**two different configurations**, not one — read this before opening any
`summary.json` in isolation.

| Directory | Configuration measured | Reflects shipped code? |
|---|---|---|
| `phase1-before/` | Original 76-payload QUOTE_OR_FACTS-only battery, **zero structural mitigation** | **Yes** |
| `claim-channel-before/` | Phase-2 CLAIM_TEXT-channel extension (76 payloads), **zero structural mitigation** | **Yes** |
| `summary.json` (+ `draws/`, top level) | Full 152-payload battery run through a **nonce-fenced envelope** (`app.prompt_fencing`) | **No — declined, reverted** |

`phase1-before/` + `claim-channel-before/` together are the full 152-payload
BEFORE baseline that matches what actually ships today: the soft
data-only system-prompt instruction, no structural fencing. The top-level
`summary.json` is the AFTER measurement of a nonce-fenced mitigation that
was tried for issue #192, found to make things worse, and **reverted from
both judge modules** — see `services/copilot-agent/app/semantic_support.py`
and `app/source_ref_relevance.py`'s "Injection posture" docstring sections,
and `prd/DECISIONS.md`'s 2026-07-27 entries, for the decision record.

## Headline numbers (190 draws per (judge, direction) cell — combining
`phase1-before` + `claim-channel-before`'s 95 draws each, or the top-level
run's own 190 draws per cell)

- **force-SUPPORTED — the only direction that can promote an unsupported
  clinical claim to certified-verified — was 0/190 bypass in EVERY
  configuration measured, both judges, before and after fencing.** This is
  the number that mattered for the closing decision.
- Fail-closed (force_not_supported) direction, unfenced (shipped) baseline:
  25/190 bypass for **each** judge (`semantic_support` and
  `source_ref_relevance` independently).
- Fail-closed direction, nonce-fenced (declined, not shipped):
  `semantic_support` 21/190 (noise on 4 draws vs. the 25/190 baseline);
  `source_ref_relevance` 61/190 — **2.4x worse** than its own 25/190
  baseline.

The fenced numbers are retained here as **evidence for the decline
decision**, not as a description of what ships. If you are asking "what
does production actually resist today", the answer is in `phase1-before/`
and `claim-channel-before/`, not the top-level `summary.json`.

## Why the fenced (AFTER) set isn't deleted

Deleting it would remove the only record proving the mitigation was tried
and measured, which is the whole basis for the "measured decline" closure
of #192. It stays, clearly subordinate to the two `*-before/` baselines
that match shipped code.

## Regenerating

`evals/runner/issue_192_injection_battery.py` reconstructs the SHIPPED
(unfenced) judge prompt shape only. A fresh run of that script (e.g. after a
judge-model swap) should reproduce numbers comparable to
`phase1-before/summary.json` + `claim-channel-before/summary.json`
combined — it does **not** regenerate the fenced `summary.json` at the top
level, since the fencing code it measured no longer exists in
`app.semantic_support` / `app.source_ref_relevance`.
