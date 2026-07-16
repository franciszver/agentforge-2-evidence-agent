"""Whole-answer verdict computation + trace-logging seam (P3.7): the
integrator of the verification layer (``docs/IMPLEMENTATION_PLAN.md``
Sec 4.4(3)), in the same family as ``app.verification`` (citation checking),
``app.rendering`` (claim stripping), ``app.allergy_check`` (P3.4), and
``app.check_drug_interactions`` (P3.6). NO LLM, NO I/O, NO clock -- a pure
function that folds the three verification signals into one
``verified | partially_verified | blocked`` verdict.

**Inputs (all already computed upstream; this module does not re-implement
any of them):**

1. Citation completeness -- ``list[app.verification.ClaimCheckResult]``
   (each ``.passed``), the same list ``app.rendering.render_answer`` consumes.
2. ``list[app.allergy_check.AllergyConflict]``.
3. ``list[app.schemas.tools.DrugInteractionItem]`` (``CheckDrugInteractionsOutput.items``).

**The decision table.** Two axes, combined with safety dominating citations.

*Citation axis* -- classifies ``claim_results`` by how many claims survived
citation re-validation (``ClaimCheckResult.passed``):

- ``ALL_VERIFIED``  -- >=1 claim, zero stripped.
- ``SOME_VERIFIED`` -- >=1 claim, some (not all) stripped.
- ``NONE_VERIFIED``  -- every claim stripped, OR zero claims at all. These two
  cases are deliberately unified: whether the answer had no checkable claims
  in it, or had claims that every single one failed re-validation, the result
  is identical from the trust story's point of view -- zero grounded evidence
  backs the answer. Collapsing "0 claims" into the same bucket as "N claims,
  N stripped" also falls out naturally from the arithmetic
  (``stripped_claim_count == total_claim_count``, vacuously true at 0/0) and
  needs no special-casing.

*Safety axis* -- classifies the domain-constraint checks:

- ``NO_VIOLATION`` -- no allergy conflicts, no drug interactions at all.
- ``WARNING``       -- no allergy conflict and no MAJOR/CONTRAINDICATED
  interaction, but >=1 MINOR/MODERATE interaction is present.
- ``BLOCKING``      -- >=1 allergy conflict, OR >=1 MAJOR/CONTRAINDICATED
  interaction.

**Severity threshold, justified.** MAJOR and CONTRAINDICATED interactions
(DDInter-style: potentially life-threatening, or a combination that should
not be given at all) must stop a user from acting on the answer without
review -- hard block. MODERATE/MINOR interactions are routinely managed
clinically via dose adjustment or monitoring; escalating those to a hard
block would train users to ignore the block, eroding the signal for the
cases that matter -- a visible warning (``partially_verified``) is the
correct amount of friction. Allergy conflicts are always ``BLOCKING``
regardless of the matched ``AllergyItem``'s own severity field: an allergy
match, unlike a drug-drug interaction, has no "this is normally fine, just
monitor it" clinical reading -- per ``app.allergy_check``'s own documented
bias, a false negative here (treating a real allergy hit as a mere warning)
is a safety miss, which is worse than the cost of over-blocking.

**Combining the axes -- safety dominates, decided cell by cell (9 cells,
exhaustive, ``_decide`` below):**

| citation \\ safety | NO_VIOLATION        | WARNING              | BLOCKING |
|--------------------|----------------------|----------------------|----------|
| ALL_VERIFIED       | verified             | partially_verified   | blocked  |
| SOME_VERIFIED      | partially_verified   | partially_verified   | blocked  |
| NONE_VERIFIED      | blocked (fail-closed)| blocked (fail-closed)| blocked  |

Rationale for the two fail-closed cells (``NONE_VERIFIED`` x
``NO_VIOLATION``/``WARNING``): an answer that has zero surviving citations
carries no confirmed factual grounding at all. Labeling that
``partially_verified`` would overstate trust -- "partial" implies *some*
of it checked out, which is false here. ``blocked`` is also consistent with
the module's UI contract: "blocked" is the verdict that forces a visible
warning the user cannot miss, which is exactly the right amount of friction
for "we could not confirm anything in this answer against the record."

``verified`` is reachable from exactly one cell (``ALL_VERIFIED`` x
``NO_VIOLATION``): every factual claim passed citation AND no domain
constraint was violated at all.

**Seam concern for the answer->claims extraction pipeline (unbuilt).** This
fold trusts only what the extraction step emits as structured claims. The plan
(Sec 4.4) names an explicit known limitation: verification covers *claims about
structured record data* and *cannot* validate free-text clinical reasoning. A
legitimately reasoning-heavy answer therefore yields few/zero structured claims
-> ``NONE_VERIFIED`` -> ``blocked`` under this table. That fail-closed floor is
deliberate here (this module must not invent trust it cannot verify), but it
means the extraction pipeline -- not this fold -- owns the decision of whether a
reasoning-only answer should carry a distinct signal (e.g. "no verifiable
claims by design" vs. "claims that failed re-validation") rather than being
folded into the same ``NONE_VERIFIED`` bucket. If that distinction is ever
wanted, it belongs upstream in extraction; changing the fold to soften
``NONE_VERIFIED`` would overstate trust for the genuinely-unverified case.

**Trace logging (this module's seam, not the durable store).** The plan
(Sec 4.4(3)) says the verdict is "logged to the trace store." The durable
trace store is P4.2 (not built) and correlation middleware is P4.1 (not
built) -- neither is built here. ``to_trace_record`` produces the
JSON-serializable record a per-turn/trace mechanism would persist (verdict +
the structured counts, mirroring how ``app.chat.Turn`` -- P2.17 -- already
keeps the shape a durable store would persist for user/patient/correlation).
It is NOT wired into ``app.chat``'s ``Turn``/SSE stream here: doing so for a
real response requires turning the planner's free-text answer into
``list[Claim]`` and "which medications are mentioned," and that
answer -> claims/meds extraction pipeline does not exist yet (the same seam
gap ``app.verification``, ``app.allergy_check``, and
``app.check_drug_interactions`` already call out). Wiring belongs with that
pipeline, not here; this module is tested against seeded
``ClaimCheckResult``/``AllergyConflict``/``DrugInteractionItem`` inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, auto

from app.allergy_check import AllergyConflict
from app.schemas.common import InteractionSeverity
from app.schemas.tools import DrugInteractionItem
from app.verification import ClaimCheckResult


class Verdict(StrEnum):
    """The whole-answer verdict badge (P3.8 UI contract)."""

    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class VerdictResult:
    """The verdict plus the structured evidence that produced it -- the
    single object P3.8 consumes to render the badge, warning banner, and
    notice count."""

    verdict: Verdict
    total_claim_count: int
    stripped_claim_count: int
    allergy_conflicts: list[AllergyConflict]
    blocking_interactions: list[DrugInteractionItem]
    warning_interactions: list[DrugInteractionItem]


class _CitationState(StrEnum):
    """Internal: which citation-axis cell ``claim_results`` falls into. Not
    exposed -- an implementation detail of the decision table."""

    ALL_VERIFIED = auto()
    SOME_VERIFIED = auto()
    NONE_VERIFIED = auto()


class _SafetyState(StrEnum):
    """Internal: which safety-axis cell the domain-constraint checks fall
    into. Not exposed -- an implementation detail of the decision table."""

    NO_VIOLATION = auto()
    WARNING = auto()
    BLOCKING = auto()


_BLOCKING_SEVERITIES = frozenset({InteractionSeverity.MAJOR, InteractionSeverity.CONTRAINDICATED})


def _citation_state(total_claim_count: int, stripped_claim_count: int) -> _CitationState:
    # ``stripped == total`` is vacuously true at 0/0, so zero-claims collapses
    # into NONE_VERIFIED here with no separate special-case (see docstring).
    if stripped_claim_count == total_claim_count:
        return _CitationState.NONE_VERIFIED
    if stripped_claim_count == 0:
        return _CitationState.ALL_VERIFIED
    return _CitationState.SOME_VERIFIED


def _is_blocking_severity(severity: InteractionSeverity) -> bool:
    match severity:
        case InteractionSeverity.MAJOR | InteractionSeverity.CONTRAINDICATED:
            return True
        case InteractionSeverity.MINOR | InteractionSeverity.MODERATE:
            return False


def _safety_state(
    allergy_conflicts: Sequence[AllergyConflict],
    blocking_interactions: Sequence[DrugInteractionItem],
    warning_interactions: Sequence[DrugInteractionItem],
) -> _SafetyState:
    if allergy_conflicts or blocking_interactions:
        return _SafetyState.BLOCKING
    if warning_interactions:
        return _SafetyState.WARNING
    return _SafetyState.NO_VIOLATION


def _decide(citation_state: _CitationState, safety_state: _SafetyState) -> Verdict:
    """The 9-cell decision table -- see module docstring for the table and
    its justification. Exhaustive `match`, no default: every
    (citation_state, safety_state) pair is enumerated explicitly."""
    match (citation_state, safety_state):
        case (_CitationState.ALL_VERIFIED, _SafetyState.NO_VIOLATION):
            return Verdict.VERIFIED
        case (_CitationState.ALL_VERIFIED, _SafetyState.WARNING):
            return Verdict.PARTIALLY_VERIFIED
        case (_CitationState.ALL_VERIFIED, _SafetyState.BLOCKING):
            return Verdict.BLOCKED
        case (_CitationState.SOME_VERIFIED, _SafetyState.NO_VIOLATION):
            return Verdict.PARTIALLY_VERIFIED
        case (_CitationState.SOME_VERIFIED, _SafetyState.WARNING):
            return Verdict.PARTIALLY_VERIFIED
        case (_CitationState.SOME_VERIFIED, _SafetyState.BLOCKING):
            return Verdict.BLOCKED
        case (_CitationState.NONE_VERIFIED, _SafetyState.NO_VIOLATION):
            return Verdict.BLOCKED
        case (_CitationState.NONE_VERIFIED, _SafetyState.WARNING):
            return Verdict.BLOCKED
        case (_CitationState.NONE_VERIFIED, _SafetyState.BLOCKING):
            return Verdict.BLOCKED


def compute_verdict(
    claim_results: Sequence[ClaimCheckResult],
    allergy_conflicts: Sequence[AllergyConflict],
    interactions: Sequence[DrugInteractionItem],
) -> VerdictResult:
    """Fold the three verification signals into one ``VerdictResult``. See
    the module docstring for the decision table. Never raises -- every input
    combination, including all-empty, maps to a verdict (fail-closed)."""
    total_claim_count = len(claim_results)
    stripped_claim_count = sum(1 for result in claim_results if not result.passed)

    blocking_interactions = [item for item in interactions if _is_blocking_severity(item.severity)]
    warning_interactions = [item for item in interactions if not _is_blocking_severity(item.severity)]

    citation_state = _citation_state(total_claim_count, stripped_claim_count)
    safety_state = _safety_state(allergy_conflicts, blocking_interactions, warning_interactions)

    return VerdictResult(
        verdict=_decide(citation_state, safety_state),
        total_claim_count=total_claim_count,
        stripped_claim_count=stripped_claim_count,
        allergy_conflicts=list(allergy_conflicts),
        blocking_interactions=blocking_interactions,
        warning_interactions=warning_interactions,
    )


def to_trace_record(result: VerdictResult) -> dict[str, object]:
    """The JSON-serializable record a per-turn trace mechanism would persist
    for this verdict -- see module docstring, "Trace logging," for what is
    and is not wired here."""
    return {
        "verdict": result.verdict.value,
        "total_claim_count": result.total_claim_count,
        "stripped_claim_count": result.stripped_claim_count,
        "allergy_conflict_count": len(result.allergy_conflicts),
        "blocking_interaction_count": len(result.blocking_interactions),
        "warning_interaction_count": len(result.warning_interactions),
    }
