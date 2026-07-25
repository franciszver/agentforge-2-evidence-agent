"""Deterministic per-tool-call citable-scope gate (issue #158).

**The gap this closes -- same root cause as issue #153, coarser fix.**
``app.extraction._build_catalog`` hands the claim extractor EVERY field of
EVERY record from EVERY tool call this turn made, and ``check_source_ref``
only re-validates a citation's ``(tool_call_id, record_id, field,
asserted_value)`` against the RAW record -- it never asks whether the
ANSWER actually discussed that record at all. A claim can therefore cite a
real, correctly-valued field from a tool call the answer never engaged with
(e.g. an answer about weight that also cites a same-turn ``get_allergies``
call the answer never mentions) and still pass provenance re-validation.

Issue #153 (``app.answer_grounding``) already closes a version of this gap,
but at CLAIM-TEXT granularity, via lexical overlap between the claim's own
words and the answer's words -- and an adversarial review found that
heuristic unfit to enable as shipped (negation-blind, short claims bypass
the ratio, wrong numeric values pass outright; see that module's docstring).
This module is a DELIBERATELY COARSER, owner-approved alternative: instead
of asking "is THIS CLAIM's text grounded in the answer" (per claim, per
field), it asks "did the answer engage with THIS TOOL CALL's data at all"
(per call, decided once per turn). Coarser means fewer failure directions to
reason about -- there is no per-claim negation/short-claim/wrong-value
surface here, because the check never looks at claim text at all -- at the
cost of precision: engaging with a call's data at all does not mean the
answer discussed every field/record within it (see "Known failure
directions" below).

**The rule.** For each tool call ``call_i`` (0-based, positional -- the same
convention ``app.verification.CacheIndex.from_raw_results`` uses), build a
"value text" by stringifying every field VALUE (never the field name) of
every record in that call's raw result, concatenated. Tokenize that value
text and the answer text with the SAME normalization
``app.answer_grounding.significant_tokens`` uses (lowercase alphanumeric,
stopwords and single-character tokens dropped -- shared, not copy-pasted, so
the two gates can never silently diverge on what counts as a "significant
word"). ``call_i`` is "engaged" iff its value-token set intersects the
answer's significant-token set -- ANY shared token is enough; there is no
overlap-ratio threshold here (unlike #153's ``_MIN_OVERLAP_RATIO``), because
a tool call's value text is typically large (many fields, many records) and
a ratio computed against it would not mean the same thing a ratio against a
single short claim's own text means.

Numbers count as significant tokens (``significant_tokens`` never treats
digit strings specially), so an answer that quotes a bare number ("Her
weight is 220 lb.") engages the vitals call whose record contains that same
number as a value (220), even with no other shared vocabulary.

**Representation contract (gate-3 review MINOR 2): this module reads
whatever ``raw_results``/``answer`` it is HANDED -- it has no opinion of its
own about normalization or notices; that is entirely the CALLER's
responsibility to get right.** ``app.extraction.run_verification`` (the one
production caller) is deliberately careful about which representation it
passes to ``engaged_call_ids``, and both choices matter:

  - It passes ``PlannerResult.raw_results`` -- PRE-normalization -- NOT the
    wide-format-``normalize_raw_results``-reshaped copy used for provenance
    re-validation. Normalization moves a vitals concept from a VALUE to a
    FIELD NAME (EAV ``{vital_type: "weight", value: 220}`` becomes
    ``{weight: 220}``), and this module deliberately excludes field names
    from engagement tokens (see "Field NAMES are deliberately excluded"
    below) -- so engaging from the NORMALIZED copy would silently lose
    "weight" as an engageable word, leaving only the call's numbers/units/
    dates to engage through. Reading the RAW EAV record instead keeps
    "weight" as a real value and restores that word as a valid engagement
    signal (a false-BLOCKING direction the owner specifically flagged as
    worth reducing).
  - It passes ``PlannerResult.answer_pre_notice`` when set, never
    ``PlannerResult.answer`` unconditionally -- see
    ``app.planner.PlannerResult.answer_pre_notice``'s docstring and this gap
    below.

``None`` field values are skipped entirely when building a call's value
text -- never stringified to the literal word "none" (see
``_call_value_tokens``'s docstring for why: "none" is not a stopword, so
that would let an answer merely containing the word "none" spuriously
engage any call with a null field). Bool values ARE kept (``True``/``False``
tokenize to "true"/"false") -- a null field asserts nothing, but a bool
field asserts something real. Nested ``dict``/``list`` values are ALSO
skipped entirely (not stringified) -- ``str()`` on a nested structure would
stringify its KEYS too, which would violate "field names are excluded" the
moment any tool schema nests a sub-object; no tool output does today, but
this is enforced structurally rather than left as a latent trap for a
future one.

**Field NAMES are deliberately excluded from the value text.** Only VALUES
are tokenized -- a field named ``respiratory_rate`` does not itself engage a
call just because the answer happens to use the word "rate" for something
else; only the call's actual DATA values (drug names, doses, numbers,
statuses, dates, ...) count. This mirrors ``app.extraction._build_catalog``'s
own value-omitted-from-catalog design in spirit (the citation catalog shown
to the model also strips values down to positions), just applied to the
opposite side of this check.

**Known failure directions -- accepted, not silently swept aside:**

  1. **Record-level granularity is lost.** A call with N records is engaged
     as a WHOLE the moment the answer's tokens hit any ONE value in any ONE
     record -- a claim citing a DIFFERENT, un-discussed record of the SAME
     engaged call still passes this gate (it would need #153's per-claim
     check, or a future per-record refinement, to catch that). This is the
     accepted coarseness the owner chose "coarse-first" for; #149's original
     motivating scenario (a claim citing a field from a tool call the answer
     never engaged with AT ALL, e.g. an unrelated ``get_allergies`` call
     bundled into a vitals question) is what this gate targets, not
     within-call precision.
  2. **A coincidental token match engages a call the answer didn't really
     discuss.** E.g. an allergy record whose ``substance`` value happens to
     equal a word the answer uses for an unrelated reason ("Sulfa" appearing
     in an unrelated context) would count as engagement. No semantic
     judgment is made about WHY a token matched -- purely lexical, same
     posture as ``app.answer_grounding``.
  3. **Value tokens that never survive `_TOKEN_RE`/stopword filtering can't
     engage a call.** A call whose every record's values are entirely short
     (length-1) tokens, punctuation, or stopword-only text can never be
     engaged even if the answer plainly discusses it in different words
     (e.g. a boolean ``true``/``false`` field the answer paraphrases as
     "yes"/"active" -- "true" itself IS a kept token, but a values set of
     e.g. only single-character codes would not be). Not expected to be
     common in this codebase's tool outputs (drug names, doses, numeric
     vitals, dates all tokenize to multi-character content words), but worth
     naming as a theoretical gap.
  4. **Whole-turn ``BLOCKED`` is the dominant flag-ON regression risk, not a
     residual edge case.** When EVERY call this turn made is unengaged (the
     answer's prose never lexically touches any of them -- plausible for a
     terse or unusual answer) AND there is no guideline/patient-fact catalog
     either, ``ClaimExtractor.extract_claims``'s existing "nothing citable"
     short-circuit (see that method's docstring) returns ``[]`` with NO
     model call at all -- zero claims, which ``app.verdict``'s
     ``NONE_VERIFIED`` row fails closed to ``blocked`` (P3.7), same as a
     genuine extraction failure. This is the CORRECT fail-closed direction
     (an answer verified against nothing citable should never read as
     "verified"), but it is the failure mode most likely to show up in
     practice if the engagement rule is ever too aggressive for a real
     conversational pattern this codebase hasn't hit yet -- worth watching
     in any per-category eval measurement before this flag is ever
     defaulted on.

**The zero-significant-token answer edge case: fail-closed, no calls
engaged.** When the answer itself has zero significant tokens (e.g. an
answer that is entirely stopwords/punctuation -- degenerate, should not
happen for a real clinical answer but not excluded by any upstream type),
``engaged_call_ids`` returns the empty set rather than treating "nothing to
compare against" as vacuous engagement of everything. This is the same
fail-closed posture ``app.answer_grounding.claim_is_grounded_in_answer``
takes for a claim with zero significant tokens, and the same posture
``check_source_ref``'s ``NO_ASSERTED_VALUE`` / ``ClaimCheckResult.passed``'s
empty-citation-list case take elsewhere in this trust layer: an inability to
establish engagement is treated as NO engagement, never as engagement of
everything.

**Two enforcement points (both required when the flag is ON; see
``Settings.copilot_extraction_tool_call_scoping_enabled``):**

  1. **PREVENTION** -- ``app.extraction.ClaimExtractor.extract_claims``
     narrows the ``_build_catalog``/``_build_tool_result_messages`` inputs to
     only engaged calls' records, via an optional ``engaged_call_ids``
     parameter on both functions. Unengaged calls are SKIPPED, never
     renumbered -- the positional ``call_i`` id scheme is load-bearing (see
     ``app.verification``'s module docstring, decision 2, and
     ``app.extraction._REAL_TOOL_CALL_ID_RE``), so call_2 must still read
     "call_2" in the catalog even if call_1 was dropped for being
     unengaged.
  2. **ENFORCEMENT** -- ``apply_tool_call_scoping`` below, invoked from
     ``app.extraction.run_verification`` exactly where
     ``apply_answer_grounding``/``apply_semantic_support`` are invoked. This
     is the enforcement that actually holds in every hermetic test in this
     codebase, since the fake extractors those tests use IGNORE the catalog
     entirely (they return a scripted claim list regardless of what
     ``extract_claims`` was called with) -- prevention alone would be
     invisible to that whole test suite.

**Scope: only re-checks currently-passing ``SourceRef`` citations.** A claim
that already failed provenance re-validation is passed through unchanged
(nothing to re-check). A ``DocumentCitation`` (guideline/patient-fact
citation) is never touched -- it has no ``tool_call_id`` and is not tied to
a tool call at all, so this gate has nothing to say about it."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.answer_grounding import significant_tokens
from app.verification import (
    CitationCheckResult,
    CitationStatus,
    ClaimCheckResult,
    DocumentCitationCheckResult,
    records_of,
)

# Free-text provenance hook present on every output item; never a citable
# value, so it is excluded from the value text a call is scored against --
# mirrors app.extraction._PROVENANCE_FIELD (kept as a separate constant
# rather than imported, to avoid a needless cross-module coupling for one
# string literal both modules already treat as a stable convention).
_PROVENANCE_FIELD = "source_refs"


def _call_value_tokens(result: dict[str, Any] | None) -> set[str]:
    """The significant tokens of every record's field VALUES (never field
    names) in one tool call's raw result -- see module docstring.

    ``None`` values are skipped entirely, never stringified -- a null field
    asserts nothing citable, and ``str(None)`` would otherwise contribute
    the literal token "none" to the set. "none" is NOT in ``_STOPWORDS``
    (only "no"/"not" are), so without this skip, ANY answer containing the
    word "none" ("No known allergies -- none noted.") would spuriously
    engage a call purely because one of its fields happened to be null,
    never because of real record data.

    ``dict``/``list`` values are ALSO skipped entirely (gate-3 review NOTE
    1) -- only SCALAR values (``str``/``int``/``float``/``bool``) are
    tokenized. ``str()`` on a nested structure stringifies its KEYS too
    (e.g. ``str({"internal_code": "x"})`` contains the literal text
    "internal_code"), which would silently violate "field names are
    excluded" the moment a tool schema nests a sub-object -- no tool output
    in this codebase does today, but this is enforced structurally rather
    than left as a latent trap for a future one.

    Note the key-leak rationale above is a DICT-specific argument -- it does
    not actually apply to ``list``: ``str(["Lisinopril", "Metformin"])``
    leaks no keys, only its own values. Lists are skipped anyway, for
    uniformity with ``dict`` and out of general conservatism about
    stringifying compound values, not because they share the key-leak
    failure mode. The accepted consequence is fail-closed, not fail-open: a
    future scalar-``list`` field (e.g. a list of drug names) would
    contribute NO engagement tokens at all rather than leak anything --
    unengageable, never wrongly engaged.

    Bool values ARE kept (``str(True)``/``str(False)`` tokenize to "true"/
    "false") -- unlike ``None``, a bool is a real, citable value (e.g. an
    active/resolved status flag), and an answer that echoes it back should
    count as genuine engagement."""
    values: list[str] = []
    for record in records_of(result):
        for field, value in record.items():
            if field == _PROVENANCE_FIELD or value is None:
                continue
            if isinstance(value, (dict, list)):
                continue
            values.append(str(value))
    return significant_tokens(" ".join(values))


def engaged_call_ids(raw_results: Sequence[dict[str, Any] | None], answer: str) -> frozenset[str]:
    """The ``call_i`` ids the answer lexically "engaged with" -- see module
    docstring for the rule. Fail-closed: an answer with zero significant
    tokens engages NO calls, never all of them."""
    answer_tokens = significant_tokens(answer)
    if not answer_tokens:
        return frozenset()
    engaged = {
        f"call_{i}"
        for i, result in enumerate(raw_results)
        if _call_value_tokens(result) & answer_tokens
    }
    return frozenset(engaged)


def apply_tool_call_scoping(
    claim_results: Sequence[ClaimCheckResult], engaged: frozenset[str]
) -> list[ClaimCheckResult]:
    """Re-check every already-passing claim's ``SourceRef`` citations against
    ``engaged`` (the ``call_i`` ids returned by ``engaged_call_ids`` above),
    downgrading any citation of an unengaged tool call to
    ``CitationStatus.TOOL_CALL_NOT_ENGAGED``. Order-preserving, one-to-one
    with ``claim_results``, same shape ``app.verification.check_claims``
    already guarantees.

    Unlike ``app.answer_grounding.apply_answer_grounding`` (which downgrades
    EVERY citation on an ungrounded claim, since that check is a whole-claim
    verdict), this downgrades only the SPECIFIC citations that name an
    unengaged call -- a claim with two citations, one to an engaged call and
    one to an unengaged call, keeps its engaged citation ``VALID`` and only
    the unengaged one is downgraded. ``ClaimCheckResult.passed``'s existing
    AND-aggregation still fails the whole claim in that case (one downgraded
    citation is enough), so the net effect on the verdict is the same shape;
    only the per-citation detail differs, for a more accurate rendered
    explanation of WHICH citation was the problem.

    A claim that already failed provenance re-validation (``.passed`` is
    ``False``) is passed through completely unchanged. A
    ``DocumentCitationCheckResult`` (guideline/patient-fact citation, no
    ``tool_call_id``) is never touched."""
    results: list[ClaimCheckResult] = []
    for claim_result in claim_results:
        if not claim_result.passed:
            results.append(claim_result)
            continue
        new_citation_results: list[CitationCheckResult | DocumentCitationCheckResult] = []
        changed = False
        for citation_result in claim_result.citation_results:
            if (
                isinstance(citation_result, CitationCheckResult)
                and citation_result.source_ref.tool_call_id not in engaged
            ):
                new_citation_results.append(
                    CitationCheckResult(
                        source_ref=citation_result.source_ref, status=CitationStatus.TOOL_CALL_NOT_ENGAGED
                    )
                )
                changed = True
            else:
                new_citation_results.append(citation_result)
        if changed:
            results.append(ClaimCheckResult(claim=claim_result.claim, citation_results=new_citation_results))
        else:
            results.append(claim_result)
    return results
