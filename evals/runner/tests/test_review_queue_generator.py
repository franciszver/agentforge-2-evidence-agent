"""RED-first tests for the P4.9 promote-to-eval generator
(``app.review_queue.generate_regression_case``).

Lives under ``evals/`` (not ``services/copilot-agent/tests/``) because the
one load-bearing property under test -- "the generator's output is a
schema-valid ``EvalCase`` the P4.7 harness actually loads" -- can only be
proven against ``runner.schema``/``runner.loader``, which live here.
``evals/conftest.py`` already puts the agent's ``app`` package on
``sys.path``, so both sides of the seam (``app.review_queue`` /
``app.trace_store`` and ``runner.loader``/``runner.schema``) are importable
from one test module without any reverse dependency from the agent package
onto ``evals/`` (the generator itself does not import ``runner.*`` -- see
``app/review_queue.py``'s module docstring).

**Why these spans have no question/answer.** The P4.2 trace store persists
NO PHI -- no question/answer text, no raw tool args/results (see
``app/trace_store.py``'s module docstring). ``generate_regression_case``
therefore can only ever emit a SKELETON seeded from what a ``Span`` actually
carries (ids, enums, counts, the feedback comment). These tests assert the
skeleton is schema-valid and carries the TODO markers a human reviewer fills
in -- not that it is a complete, runnable case (it never can be, by design).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.review_queue import generate_regression_case
from app.trace_store import FeedbackThumb, Span, SpanStatus, SpanType
from runner.loader import load_case


def _span(
    *,
    id: int,
    correlation_id: str = "corr-1",
    span_type: SpanType,
    status: SpanStatus = SpanStatus.OK,
    tool_name: str | None = None,
    args_hash: str | None = None,
    model: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    verdict: str | None = None,
    claim_count: int | None = None,
    stripped_count: int | None = None,
    feedback_thumb: FeedbackThumb | None = None,
    feedback_comment: str | None = None,
    error_category: str | None = None,
    duration_ms: float = 100.0,
) -> Span:
    """Builds a ``Span`` directly (no ``TraceStore`` needed) -- these tests
    are pure and hermetic, exercising the generator's mapping from spans to
    YAML with no SQLite involved."""
    return Span(
        id=id,
        correlation_id=correlation_id,
        span_type=span_type,
        start_ts=0.0,
        end_ts=duration_ms / 1000,
        duration_ms=duration_ms,
        status=status,
        tool_name=tool_name,
        args_hash=args_hash,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        verdict=verdict,
        claim_count=claim_count,
        stripped_count=stripped_count,
        feedback_thumb=feedback_thumb,
        feedback_comment=feedback_comment,
        error_category=error_category,
    )


def _load(tmp_path: Path, text: str, name: str = "promoted.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return load_case(path)


# --- schema validity -------------------------------------------------------


def test_thumbs_down_with_comment_produces_a_schema_valid_case(tmp_path: Path) -> None:
    spans = [
        _span(id=1, span_type=SpanType.REQUEST, status=SpanStatus.OK),
        _span(
            id=2,
            span_type=SpanType.FEEDBACK,
            feedback_thumb=FeedbackThumb.DOWN,
            feedback_comment="Missed the recent A1C value entirely.",
        ),
    ]

    text = generate_regression_case(spans)
    case = _load(tmp_path, text)

    assert case.category == "regression"
    # #157: the raw clinician comment is scrubbed from the public export; the
    # field carries a neutral non-empty TODO placeholder instead.
    assert "Missed the recent A1C value entirely." not in text
    assert case.failure_mode


def test_thumbs_down_comment_is_scrubbed_from_promoted_export(tmp_path: Path) -> None:
    """PHI-safety (#157): promoted cases are committed to the PUBLIC ``evals/``
    repo, so the raw clinician free-text comment must NEVER appear in the
    exported YAML -- a clinician who typed patient details into a thumbs-down
    comment would otherwise leak PHI into public git. ``failure_mode`` instead
    carries a neutral, non-empty TODO placeholder referencing the correlation
    id; the real comment stays only in the trusted-local review queue."""
    sentinel = "SENTINEL_COMMENT_TEXT patient details that could be PHI"
    spans = [
        _span(id=1, correlation_id="corr-scrub", span_type=SpanType.REQUEST),
        _span(
            id=2,
            correlation_id="corr-scrub",
            span_type=SpanType.FEEDBACK,
            feedback_thumb=FeedbackThumb.DOWN,
            feedback_comment=sentinel,
        ),
    ]

    text = generate_regression_case(spans)
    case = _load(tmp_path, text)

    # The sentinel (standing in for free-text PHI) is absent from the whole
    # export, not just the parsed field.
    assert "SENTINEL_COMMENT_TEXT" not in text
    # failure_mode stays a non-empty required field (#155), points at the cid,
    # and is marked as a TODO for a human to complete.
    assert case.failure_mode
    assert "TODO" in case.failure_mode
    assert "corr-scrub" in case.failure_mode


def test_verification_failure_produces_a_schema_valid_case_with_verdict_assertion(tmp_path: Path) -> None:
    spans = [
        _span(id=1, span_type=SpanType.REQUEST, status=SpanStatus.OK),
        _span(
            id=2,
            span_type=SpanType.VERIFICATION,
            status=SpanStatus.OK,
            verdict="blocked",
            claim_count=3,
            stripped_count=3,
        ),
    ]

    text = generate_regression_case(spans)
    case = _load(tmp_path, text)

    assert case.category == "regression"
    verdict_assertions = [a for a in case.assertions if a.type == "verdict"]
    assert len(verdict_assertions) == 1
    assert verdict_assertions[0].equals.value == "blocked"


def test_thumbs_up_feedback_on_an_unverified_verdict_is_not_labeled_a_failure(tmp_path: Path) -> None:
    """A trace reaches the generator on a non-verified verdict alone while its
    deduped feedback is a thumbs-UP (with a comment). ``failure_mode`` must NOT
    echo that positive comment or claim a thumbs-down -- it describes the
    verification failure that actually put the trace on the queue."""
    positive_comment = "Great, concise answer -- exactly what I needed."
    spans = [
        _span(id=1, span_type=SpanType.REQUEST),
        _span(
            id=2,
            span_type=SpanType.VERIFICATION,
            verdict="blocked",
            claim_count=2,
            stripped_count=2,
        ),
        _span(
            id=3,
            span_type=SpanType.FEEDBACK,
            feedback_thumb=FeedbackThumb.UP,
            feedback_comment=positive_comment,
        ),
    ]

    text = generate_regression_case(spans)
    case = _load(tmp_path, text)

    assert case.failure_mode is not None
    assert positive_comment not in case.failure_mode
    assert "thumbs-down" not in case.failure_mode
    assert "verdict=blocked" in case.failure_mode


def test_case_id_is_kebab_case_and_stable_for_the_same_correlation_id(tmp_path: Path) -> None:
    spans = [_span(id=1, correlation_id="corr-xyz-123", span_type=SpanType.REQUEST)]

    first = _load(tmp_path, generate_regression_case(spans), name="a.yaml")
    second = _load(tmp_path, generate_regression_case(spans), name="b.yaml")

    assert first.id == second.id
    assert "corr-xyz-123" in first.id
    assert " " not in first.id


# --- source correlation id --------------------------------------------------


def test_source_correlation_id_is_carried_on_the_case(tmp_path: Path) -> None:
    spans = [_span(id=1, correlation_id="corr-abc", span_type=SpanType.REQUEST)]

    case = _load(tmp_path, generate_regression_case(spans))

    assert case.source == "corr-abc"


# --- TODO placeholders -------------------------------------------------------


def test_output_carries_todo_placeholders_for_fields_the_trace_store_cannot_supply() -> None:
    spans = [_span(id=1, span_type=SpanType.REQUEST)]

    text = generate_regression_case(spans)
    case_dict = yaml.safe_load(text)

    assert "TODO" in text
    # No PHI ever invented to fake a complete case -- the question field is
    # never a real clinical question, just a placeholder marker.
    assert "TODO" in case_dict["question"]


def test_no_phi_and_no_fabricated_clinical_content_in_output() -> None:
    spans = [
        _span(id=1, span_type=SpanType.REQUEST),
        _span(
            id=2,
            span_type=SpanType.FEEDBACK,
            feedback_thumb=FeedbackThumb.DOWN,
            feedback_comment=None,
        ),
    ]

    text = generate_regression_case(spans)
    case_dict = yaml.safe_load(text)

    # tool_data is either absent or empty -- never invented canned patient data.
    assert not case_dict.get("tool_data")


# --- empty / malformed trace handling ---------------------------------------


def test_empty_trace_raises_value_error() -> None:
    with pytest.raises(ValueError):
        generate_regression_case([])


def test_minimal_trace_with_no_feedback_or_verification_still_produces_a_valid_skeleton(
    tmp_path: Path,
) -> None:
    """A trace with only a request span (e.g. promoted straight off a
    verification-failure-free, comment-free thumbs-down edge case) must still
    produce a loadable case -- fallback assertion + failure_mode, not a crash."""
    spans = [_span(id=1, span_type=SpanType.REQUEST, status=SpanStatus.FAIL)]

    text = generate_regression_case(spans)
    case = _load(tmp_path, text)

    assert case.category == "regression"
    assert len(case.assertions) >= 1


def test_unrecognized_span_shape_does_not_crash_the_generator() -> None:
    # A trace of only spans the generator doesn't special-case (here an LLM
    # span, no request/feedback/verification) must still return text, never
    # raise an uncaught exception -- the generator is a pure function that
    # only raises the documented ValueError on the empty-trace case.
    spans = [_span(id=1, span_type=SpanType.LLM, model="qwen3:4b", tokens_in=10, tokens_out=20)]
    text = generate_regression_case(spans)
    assert isinstance(text, str) and text.strip()


def test_output_is_well_formed_yaml_with_a_header_comment(tmp_path: Path) -> None:
    spans = [_span(id=1, span_type=SpanType.REQUEST)]
    text = generate_regression_case(spans)

    # Parses cleanly (no YAMLError) and loads via the real loader -- load_case
    # raises EvalCaseError only for genuinely broken input, never for this
    # generator's own output, so an unhandled raise here fails the test outright.
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    _load(tmp_path, text)
