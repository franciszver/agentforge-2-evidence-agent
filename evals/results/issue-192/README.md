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
and `app/source_ref_relevance.py`'s "Injection posture" docstring sections
for the decision record (also mirrored in this repo's local, gitignored
`prd/DECISIONS.md`, not visible to readers of this public copy).

## Headline numbers (190 draws per (judge, direction) cell — combining
`phase1-before` + `claim-channel-before`'s 95 draws each, or the top-level
run's own 190 draws per cell)

- **force-SUPPORTED — the only direction that can promote an unsupported
  clinical claim to certified-verified — was 0/190 bypass in EVERY
  configuration measured, both judges, before and after fencing.** This is
  the number that mattered for the closing decision. **This is a named
  limitation, not a clean result** — see "Force-SUPPORTED confound" below.
- Fail-closed (force_not_supported) direction, unfenced (shipped) baseline:
  25/190 bypass for **each** judge (`semantic_support` and
  `source_ref_relevance` independently) — but the four (judge, channel)
  cells are NOT uniform: recomputed per-technique from the committed draws,
  the shipped judges resist 16/19 techniques on `semantic_support`/
  QUOTE_OR_FACTS, 17/19 on `semantic_support`/CLAIM_TEXT, 14/19 on
  `source_ref_relevance`/QUOTE_OR_FACTS, and 19/19 (zero bypass) on
  `source_ref_relevance`/CLAIM_TEXT. See `services/copilot-agent/tests/
  test_issue_192_injection_battery.py`'s module docstring for the exact
  10 payload ids these bypasses come from.
- Fail-closed direction, nonce-fenced (declined, not shipped):
  `semantic_support` 21/190 (noise on 4 draws vs. the 25/190 baseline);
  `source_ref_relevance` 61/190 — **2.4x worse** than its own 25/190
  baseline.

The fenced numbers are retained here as **evidence for the decline
decision**, not as a description of what ships. If you are asking "what
does production actually resist today", the answer is in `phase1-before/`
and `claim-channel-before/`, not the top-level `summary.json`.

## Force-SUPPORTED confound (named limitation, not a clean result)

Each force-SUPPORTED scenario pairs a claim with a maximally-UNRELATED
quote/facts value, chosen so the un-injected baseline is unambiguous (e.g.
an ESRD/dialysis claim against a topically unrelated hypertension-cadence
quote; a blood-pressure claim against `appointment_status: scheduled`) — the
EASIEST pairing for a judge to resist. The fail-closed scenarios (where the
25/190 bypasses above actually happened) start from genuinely-supporting,
high-overlap pairs instead. This design cannot distinguish "resists
force-SUPPORTED injection" from "won't call a wildly-unrelated pair
supported regardless of injection" — the realistic threat (a NEAR-MISS pair:
plausibly related, not actually supporting, plus injection) was never
measured. This is the top candidate for any future re-measurement of this
battery; treat the 0/190 as evidence about the easy case only.

## Statistical power (named limitation, not a clean result)

5 draws per payload: a payload that truly flips 10% of the time shows 0/5
with probability ~0.59, so per-payload "resisted" only reaches roughly
95%-confidence detection around a 45%+ true flip rate. The recorded-replay
test suite (`test_issue_192_injection_battery.py`) replays a SINGLE recorded
draw (draw 0) per payload, not all 5. And 38 payloads × 5 CORRELATED draws is
not 190 independent trials, so a rule-of-three reading of any 0/190 or N/190
count above is not statistically supported by this design.

## Provenance of `claim-channel-before/`

`claim-channel-before/summary.json`'s own `note` says its prompt shape was
"reconstructed verbatim from `evals/runner/issue_192_injection_battery.py`
@a8015ea" — that commit (`39d190a7`) is when both the runner and the judge
modules were still fenced, so this dataset came from an uncommitted, ad-hoc
reconstruction of the unfenced shape, not from a run of any committed,
fenced-free version of the runner. It is **not reproducible** from a
committed artifact as-is; it contributes 10 of `semantic_support`'s 25/190
fail-closed bypasses. Going forward, `evals/runner/issue_192_injection_
battery.py`'s own `summary.json` output should be treated as the
reproducible baseline for any future re-measurement; a config/prompt
fingerprint stamped into each summary (tracked as a runner improvement) would
let this kind of provenance gap be caught automatically rather than only
noticed on manual review.

## Why the fenced (AFTER) set isn't deleted

Deleting it would remove the only record proving the mitigation was tried
and measured, which is the whole basis for the "measured decline" closure
of #192. It stays, clearly subordinate to the two `*-before/` baselines
that match shipped code.

## Regenerating

`evals/runner/issue_192_injection_battery.py` exercises the SHIPPED
(unfenced) judge prompt shape only, via each judge module's own public
"full"-judgement function (`judge_support_full` / `judge_source_ref_
relevance_full`) — never a reconstruction of the prompt template. Every
invocation writes to its own fresh, labelled subdirectory,
`evals/results/issue-192/runs/<label>/` (`--label`, default a UTC
timestamp) — it refuses to start if that directory already exists, and it
never writes to this directory's own top-level `draws/`/`summary.json` or to
`phase1-before/`/`claim-channel-before/`, which remain the untouched
historical record. A fresh run's numbers should be comparable to
`phase1-before/summary.json` + `claim-channel-before/summary.json` combined
(e.g. after a judge-model swap) — it does **not** regenerate the fenced
top-level `summary.json`, since the fencing code it measured no longer
exists in `app.semantic_support` / `app.source_ref_relevance`.
