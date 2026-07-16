"""Transformation tests for claim stripping + notices (P3.3).

``app.rendering`` is the deterministic transformation that consumes the
citation checker's output (``app.verification.ClaimCheckResult``) and
produces the user-facing rendered answer: claims whose citations all passed
are kept verbatim (with their passing citations, for P3.8's citation
chips); claims whose citations did not all pass are STRIPPED and replaced
with a "not found in record" notice -- the original claim text must never
leak into the output. No LLM, no I/O, no clock -- pure function of seeded
``ClaimCheckResult`` inputs (per the P3.3 issue, this module is not wired to
the extraction pipeline yet).
"""

from __future__ import annotations

import pytest

from app.rendering import Notice, RenderedAnswer, RenderedClaim, render_answer
from app.schemas.common import SourceRef
from app.schemas.verification import Claim
from app.verification import CitationCheckResult, CitationStatus, ClaimCheckResult

# ---------------------------------------------------------------------------
# Fixtures / helpers -- seed ClaimCheckResult directly, bypassing the checker.
# ---------------------------------------------------------------------------


def _ref(*, field: str = "name", asserted_value: str | None = "x") -> SourceRef:
    return SourceRef(tool_call_id="call_0", record_id="0", field=field, asserted_value=asserted_value)


def _citation(status: CitationStatus, *, field: str = "name") -> CitationCheckResult:
    return CitationCheckResult(source_ref=_ref(field=field), status=status)


def _claim_result(text: str, statuses: list[CitationStatus]) -> ClaimCheckResult:
    """A ``ClaimCheckResult`` with one citation per status in ``statuses``."""
    claim = Claim(text=text, source_refs=[_ref(field=f"field_{i}") for i in range(len(statuses))])
    citations = [_citation(status, field=f"field_{i}") for i, status in enumerate(statuses)]
    return ClaimCheckResult(claim=claim, citation_results=citations)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_render_answer_on_empty_list_returns_empty_segments():
    result = render_answer([])

    assert result == RenderedAnswer(segments=[])


# ---------------------------------------------------------------------------
# All-pass
# ---------------------------------------------------------------------------


def test_all_passing_claims_are_kept_in_order():
    first = _claim_result("The patient is on Lisinopril.", [CitationStatus.VALID])
    second = _claim_result("Blood pressure is 120/80.", [CitationStatus.VALID, CitationStatus.VALID])

    result = render_answer([first, second])

    assert result.segments == [
        RenderedClaim(text="The patient is on Lisinopril.", source_refs=[c.source_ref for c in first.citation_results]),
        RenderedClaim(
            text="Blood pressure is 120/80.", source_refs=[c.source_ref for c in second.citation_results]
        ),
    ]


def test_kept_claim_carries_its_passing_citations_for_ui_chips():
    result_in = _claim_result("On Lisinopril 10mg.", [CitationStatus.VALID, CitationStatus.VALID])

    [segment] = render_answer([result_in]).segments

    assert isinstance(segment, RenderedClaim)
    assert segment.source_refs == [c.source_ref for c in result_in.citation_results]
    assert len(segment.source_refs) == 2


# ---------------------------------------------------------------------------
# All-fail
# ---------------------------------------------------------------------------


def test_all_failing_claims_are_replaced_with_notices():
    first = _claim_result("The patient takes Metformin.", [CitationStatus.VALUE_MISMATCH])
    second = _claim_result("The patient had surgery in 2019.", [CitationStatus.UNKNOWN_RECORD])

    result = render_answer([first, second])

    assert result.segments == [Notice(), Notice()]


def test_stripped_claims_original_text_does_not_leak_into_output():
    secret_text = "TOP SECRET UNVERIFIED CLAIM TEXT 12345"
    claim_result = _claim_result(secret_text, [CitationStatus.VALUE_MISMATCH])

    result = render_answer([claim_result])

    rendered_text = " ".join(_segment_text(s) for s in result.segments)
    assert secret_text not in rendered_text


def _segment_text(segment: RenderedClaim | Notice) -> str:
    return segment.text


# ---------------------------------------------------------------------------
# Mixed pass/fail -- order + interleaving preserved
# ---------------------------------------------------------------------------


def test_mixed_pass_fail_preserves_order_and_interleaving():
    kept_1 = _claim_result("Kept first.", [CitationStatus.VALID])
    stripped = _claim_result("Stripped middle.", [CitationStatus.VALUE_MISMATCH])
    kept_2 = _claim_result("Kept last.", [CitationStatus.VALID])

    result = render_answer([kept_1, stripped, kept_2])

    assert len(result.segments) == 3
    assert isinstance(result.segments[0], RenderedClaim)
    assert result.segments[0].text == "Kept first."
    assert isinstance(result.segments[1], Notice)
    assert isinstance(result.segments[2], RenderedClaim)
    assert result.segments[2].text == "Kept last."


# ---------------------------------------------------------------------------
# Zero-citation degenerate claim (fails closed upstream, must strip here too)
# ---------------------------------------------------------------------------


def test_claim_with_zero_citations_is_stripped():
    claim = Claim.model_construct(text="Unsupported claim.", source_refs=[])
    claim_result = ClaimCheckResult(claim=claim, citation_results=[])

    [segment] = render_answer([claim_result]).segments

    assert segment == Notice()


# ---------------------------------------------------------------------------
# Per-CitationStatus matrix -- VALID kept, every other status stripped.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        CitationStatus.UNKNOWN_TOOL_CALL,
        CitationStatus.UNKNOWN_RECORD,
        CitationStatus.UNKNOWN_FIELD,
        CitationStatus.REDACTED_FIELD,
        CitationStatus.NO_ASSERTED_VALUE,
        CitationStatus.VALUE_MISMATCH,
    ],
)
def test_each_failing_citation_status_strips_the_claim(status: CitationStatus):
    claim_result = _claim_result("An unverifiable claim.", [status])

    [segment] = render_answer([claim_result]).segments

    assert isinstance(segment, Notice)


def test_valid_citation_status_keeps_the_claim():
    claim_result = _claim_result("A verified claim.", [CitationStatus.VALID])

    [segment] = render_answer([claim_result]).segments

    assert isinstance(segment, RenderedClaim)


def test_one_failing_citation_among_several_still_strips_the_whole_claim():
    """AND semantics propagate: partial grounding is not grounding (P3.2)."""
    claim_result = _claim_result(
        "Started 2024-01-01, currently discontinued.",
        [CitationStatus.VALID, CitationStatus.VALUE_MISMATCH],
    )

    [segment] = render_answer([claim_result]).segments

    assert isinstance(segment, Notice)


# ---------------------------------------------------------------------------
# Notice wording
# ---------------------------------------------------------------------------


def test_notice_text_matches_the_required_wording():
    assert Notice().text == "Not found in record."


def test_all_notices_produced_by_render_answer_are_equal():
    # Notices carry no per-claim distinguishing state (deliberately -- see
    # module docstring): two independently-stripped claims render identically.
    first = _claim_result("Claim A.", [CitationStatus.VALUE_MISMATCH])
    second = _claim_result("Claim B.", [CitationStatus.UNKNOWN_FIELD])

    result = render_answer([first, second])

    assert result.segments[0] == result.segments[1]
