"""Review queue + promote-to-eval generator, over the P4.2 trace store (P4.9).

Two pure pieces, both read-only over ``app.trace_store``:

  * :func:`list_review_queue` -- which correlation ids belong on the
    clinician-facing worklist: a thumbs-down feedback span, and/or a
    verification span whose verdict is not ``verified``. Feedback is
    deduped the same way the P4.5 dashboard does (#54): prefer the row
    WITH a comment, then the most recently written row.
  * :func:`generate_regression_case` -- trace spans -> a schema-valid
    ``EvalCase`` YAML skeleton (the P4.7 shape, ``evals/runner/schema.py``).

**Why this is a SKELETON, not a runnable case.** The P4.2 trace store
persists NO PHI -- no question/answer text, no raw tool args or results (see
``app.trace_store``'s module docstring). This module never invents that
data to fake completeness. A promoted case therefore always carries TODO
placeholders for ``question``, ``patient_id``, and ``tool_data`` -- a human
reviewer fills those in from what they actually saw before the case is a
real regression guard. The only genuinely case-specific content this module
can supply is what the trace store actually recorded: the correlation id
(as ``source:``), the observed verdict (as a starting ``verdict`` assertion,
when a verification span exists), and the feedback comment (as
``failure_mode:``, when the clinician left one -- comment text is
explicitly permitted verbatim storage, see the trace-store docstring, and is
therefore also safe to re-emit here).

**No reverse dependency on ``evals/``.** This module does not import
``runner.schema``/``runner.loader`` to self-validate its own output --
``app`` has no dependency on the ``evals/`` package (the dependency runs the
other way: ``evals/`` imports ``app.*``). The "produces schema-valid YAML"
property is proven by ``evals/runner/tests/test_review_queue_generator.py``,
which imports both sides.

**Deterministic.** No clock, no randomness: ``generate_regression_case``'s
output depends only on its ``spans`` argument, so the same trace always
promotes to byte-identical YAML.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import yaml

from app.trace_store import FeedbackThumb, Span, SpanType, TraceStore

# Demo dataset always ships this patient (see docs/TEST_PLAN.md Sec 7's
# canonical-patient table, "allergy-conflict", pubpid 1) -- used only as a
# schema-valid placeholder. The trace store carries no patient identity at
# all, so this is NEVER assumed to be the real patient the trace was about;
# the header comment and the field's own TODO wording say so explicitly.
_PLACEHOLDER_PATIENT_ID = 1

_TODO_QUESTION = (
    "TODO: fill in the clinician's actual question -- the trace store "
    "records no question/answer text by design (no PHI on disk)."
)

_TODO_PHRASE = "TODO: replace with the phrase this response should (not) contain"


@dataclass(frozen=True)
class ReviewQueueEntry:
    """One worklist row: everything the P4.9 review queue can show about a
    correlation id without touching PHI -- ids, an enum, a count, and the
    feedback comment (user-authored text about the response, not patient
    record data -- see ``app.trace_store``'s module docstring)."""

    correlation_id: str
    feedback_thumb: FeedbackThumb | None
    feedback_comment: str | None
    verdict: str | None
    tool_call_count: int
    request_duration_ms: float | None

    @property
    def has_unverified_verdict(self) -> bool:
        """A verification ran and did not return ``verified`` -- the signal that
        both puts this trace on the queue and renders its verdict badge."""
        return self.verdict is not None and self.verdict != "verified"


def _latest_feedback(feedback_spans: list[Span]) -> Span | None:
    """Same dedup rule as the P4.5 dashboard's ``_FEEDBACK_DEDUP_SQL`` (#54):
    prefer the row WITH a comment (more informative); among ties, the most
    recently written row. ``feedback_spans`` is in insertion order (as
    returned by ``TraceStore.get_spans``), so the last matching element is
    the highest ``id``."""
    if not feedback_spans:
        return None
    with_comment = [span for span in feedback_spans if span.feedback_comment is not None]
    candidates = with_comment if with_comment else feedback_spans
    return candidates[-1]


def _latest_verification(spans: list[Span]) -> Span | None:
    """The most recently written verification span in a trace (its whole-answer
    verdict), or ``None`` if the trace has none. ``spans`` is in insertion order
    (as ``TraceStore.get_spans`` returns), so the last match is the highest
    ``id``."""
    verification_spans = [span for span in spans if span.span_type == SpanType.VERIFICATION]
    return verification_spans[-1] if verification_spans else None


def _build_entry(correlation_id: str, spans: list[Span]) -> ReviewQueueEntry:
    feedback = _latest_feedback([span for span in spans if span.span_type == SpanType.FEEDBACK])
    verification = _latest_verification(spans)
    tool_call_count = sum(1 for span in spans if span.span_type == SpanType.TOOL)
    request_spans = [span for span in spans if span.span_type == SpanType.REQUEST]
    request_duration_ms = request_spans[-1].duration_ms if request_spans else None

    return ReviewQueueEntry(
        correlation_id=correlation_id,
        feedback_thumb=feedback.feedback_thumb if feedback else None,
        feedback_comment=feedback.feedback_comment if feedback else None,
        verdict=verification.verdict if verification else None,
        tool_call_count=tool_call_count,
        request_duration_ms=request_duration_ms,
    )


def _belongs_on_the_queue(entry: ReviewQueueEntry) -> bool:
    if entry.feedback_thumb == FeedbackThumb.DOWN:
        return True
    return entry.has_unverified_verdict


def _distinct_correlation_ids(db_path: str) -> list[str]:
    """Every correlation id with at least one span, via a short-lived
    connection against ``db_path`` -- same pattern as
    ``app.dashboard_metrics.compute_dashboard_metrics``, which opens its own
    connection rather than adding aggregate-query methods to ``TraceStore``
    (``TraceStore.db_path``'s docstring documents this as the intended seam
    for exactly this kind of read-only lookup)."""
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute("SELECT DISTINCT correlation_id FROM spans").fetchall()
        return [row[0] for row in rows]
    finally:
        connection.close()


def list_review_queue(trace_store: TraceStore) -> list[ReviewQueueEntry]:
    """Every correlation id with a thumbs-down and/or a non-``verified``
    verdict, most-recent activity first.

    Takes the ``TraceStore`` itself (not a bare ``db_path``) because
    building each entry reuses ``TraceStore.get_spans`` rather than
    re-deriving its query -- this queue is a small, demo-scale worklist (one
    row per reviewable trace), not a high-volume aggregate, so one
    ``get_spans`` call per candidate correlation id is the simpler choice
    over hand-rolled aggregate SQL.
    """
    correlation_ids = _distinct_correlation_ids(trace_store.db_path)
    entries = []
    for correlation_id in correlation_ids:
        spans = trace_store.get_spans(correlation_id)
        if not spans:
            continue
        entry = _build_entry(correlation_id, spans)
        if _belongs_on_the_queue(entry):
            entries.append((spans[-1].id, entry))

    entries.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _, entry in entries]


def _slugify_correlation_id(correlation_id: str) -> str:
    """Best-effort kebab-case-safe id fragment. Correlation ids are UUID4
    strings in production (``app.correlation.new_correlation_id``, already
    lowercase hex + dashes) so this is a no-op there; guards against
    whitespace/unexpected characters for any other id shape (e.g. a test
    double) without raising."""
    return "-".join(correlation_id.split())


def _failure_mode(feedback: Span | None, verification: Span | None, correlation_id: str) -> str:
    # Only a thumbs-DOWN describes a failure. A trace can reach this generator
    # on an unverified verdict alone while its deduped feedback is a thumbs-UP
    # (``_latest_feedback`` picks by comment/recency, not polarity) -- treating
    # that as a "thumbs-down" would emit praise, or a false "thumbs-down" label,
    # as the failure_mode. Fall through to the verdict-based message instead.
    if feedback is not None and feedback.feedback_thumb == FeedbackThumb.DOWN:
        if feedback.feedback_comment:
            return feedback.feedback_comment
        return f"Promoted from a thumbs-down (no comment left) on correlation id {correlation_id}."
    if verification is not None:
        return (
            f"Promoted from a verification failure (verdict={verification.verdict}) "
            f"on correlation id {correlation_id}."
        )
    return f"Promoted from correlation id {correlation_id}. TODO: describe the failure this case guards against."


_HEADER_COMMENT = """\
# AUTO-GENERATED regression-case skeleton (P4.9 promote-to-eval).
#
# The P4.2 trace store persists NO PHI -- no question/answer text, no raw
# tool args/results, no patient-record values (see
# services/copilot-agent/app/trace_store.py). This skeleton is seeded ONLY
# from what the trace store actually recorded (correlation id, verdict,
# feedback) and is NOT a runnable case as-is. Before this case is real, fill
# in:
#   - TODO question: the clinician's actual question (not recoverable from
#     the trace).
#   - TODO patient_id: the placeholder below is a valid demo id, NOT
#     necessarily the patient this trace was about -- the trace store
#     carries no patient identity at all.
#   - TODO tool_data: canned output for whichever tool(s) the question
#     will dispatch (docs/TEST_PLAN.md Sec 5).
#   - TODO assertions: the starting assertion below reflects only what was
#     OBSERVED -- add/replace with assertions that actually pin down the
#     failure this case guards against.
"""


def generate_regression_case(spans: list[Span]) -> str:
    """Build a schema-valid ``EvalCase`` YAML skeleton from one trace's spans.

    ``spans`` must be non-empty and share one ``correlation_id`` (the shape
    ``TraceStore.get_spans`` returns). Raises ``ValueError`` for an empty
    trace -- there is nothing to promote. Never raises for a trace missing
    feedback/verification spans (e.g. only a request span): the fallback
    ``failure_mode``/assertion below still produces a loadable skeleton.
    """
    if not spans:
        raise ValueError("cannot generate a regression case from an empty trace")

    correlation_id = spans[0].correlation_id
    case_id = f"promoted-{_slugify_correlation_id(correlation_id)}"

    feedback = _latest_feedback([span for span in spans if span.span_type == SpanType.FEEDBACK])
    verification = _latest_verification(spans)

    if verification is not None:
        assertions: list[dict[str, object]] = [{"type": "verdict", "equals": verification.verdict}]
    else:
        assertions = [{"type": "answer_not_contains", "phrases": [_TODO_PHRASE]}]

    case: dict[str, object] = {
        "id": case_id,
        "category": "regression",
        "failure_mode": _failure_mode(feedback, verification, correlation_id),
        "source": correlation_id,
        "question": _TODO_QUESTION,
        "patient_id": _PLACEHOLDER_PATIENT_ID,
        "tool_data": {},
        "assertions": assertions,
    }

    body = yaml.safe_dump(case, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return _HEADER_COMMENT + body
