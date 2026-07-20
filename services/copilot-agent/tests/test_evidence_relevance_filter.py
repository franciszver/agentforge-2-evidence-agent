"""Red-first test for issue #93 fix 1/4: a relevance-score floor on the
evidence chunks /chat hands to the claim extractor (`app/chat.py`'s
``_filter_by_relevance_score`` / ``_EVIDENCE_MIN_RELEVANCE_SCORE``).

Hermetic: exercises the pure filter function directly against hand-built
``RerankedChunk`` instances -- no retrieval, no LLM, no corpus. See
``_EVIDENCE_MIN_RELEVANCE_SCORE``'s docstring in ``app/chat.py`` for why 0.75
was chosen (issue #99: re-measured against qwen3-8b, the model that actually
scores relevance in production -- every deliberately-planted lexical
distractor scored <=0.65, every genuine match scored >=0.85). See
``tests/test_reranker_calibration.py`` for the fixture-backed regression
check on that gap.
"""

from __future__ import annotations

import logging

from app.chat import _EVIDENCE_MIN_RELEVANCE_SCORE, _filter_by_relevance_score
from app.schemas.reranking import RerankedChunk


def _chunk(chunk_id: str, rerank_score: float) -> RerankedChunk:
    return RerankedChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        title="title",
        section="section",
        text="text",
        scores={},
        rerank_score=rerank_score,
    )


def test_default_threshold_is_three_quarters() -> None:
    assert _EVIDENCE_MIN_RELEVANCE_SCORE == 0.75


def test_filter_drops_chunks_scoring_below_the_threshold() -> None:
    strong = _chunk("strong", 0.99)
    weak = _chunk("weak", 0.0)

    filtered = _filter_by_relevance_score([strong, weak])

    assert filtered == [strong]


def test_filter_keeps_a_chunk_scoring_exactly_at_the_threshold() -> None:
    borderline = _chunk("borderline", _EVIDENCE_MIN_RELEVANCE_SCORE)

    filtered = _filter_by_relevance_score([borderline])

    assert filtered == [borderline]


def test_filter_returns_empty_when_every_chunk_is_below_threshold() -> None:
    weak_a = _chunk("weak-a", 0.35)
    weak_b = _chunk("weak-b", 0.21)

    filtered = _filter_by_relevance_score([weak_a, weak_b])

    assert filtered == []


def test_filter_preserves_relative_order_of_surviving_chunks() -> None:
    first = _chunk("first", 0.97)
    second = _chunk("second", 0.92)
    dropped = _chunk("dropped", 0.0)

    filtered = _filter_by_relevance_score([first, second, dropped])

    assert filtered == [first, second]


def test_filter_logs_dropped_chunk_count_and_score_and_chunk_id_only(caplog) -> None:
    """Issue #98 review finding: a silent drop is indistinguishable from
    "genuinely found nothing." Assert the drop is logged, and that the
    logged payload carries only ``chunk_id``/score -- never the chunk's
    ``text`` (PHI/corpus-text discipline, same as this module's other
    logging call sites)."""
    caplog.set_level(logging.INFO)
    kept = _chunk("kept", 0.99)
    dropped = _chunk("dropped-one", 0.1)

    filtered = _filter_by_relevance_score([kept, dropped])

    assert filtered == [kept]
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("dropped" in r.getMessage() for r in info_records)
    matching = [r for r in info_records if getattr(r, "dropped_count", None) == 1]
    assert matching, "expected an INFO log recording dropped_count=1"
    record = matching[0]
    assert record.kept_count == 1
    assert record.dropped == [{"chunk_id": "dropped-one", "score": 0.1}]
    for r in caplog.records:
        assert "text" not in repr(r.__dict__.get("dropped", ""))


def test_filter_does_not_log_when_nothing_is_dropped(caplog) -> None:
    caplog.set_level(logging.INFO)
    kept = _chunk("kept", 0.99)

    filtered = _filter_by_relevance_score([kept])

    assert filtered == [kept]
    assert caplog.records == []


def test_filter_warns_when_every_chunk_is_dropped(caplog) -> None:
    """The exact silent-failure scenario the reviewer flagged: every
    retrieved chunk falls below the threshold, so zero evidence reaches
    the extractor. Must be visible/alertable (WARNING), not buried at
    INFO/DEBUG."""
    caplog.set_level(logging.INFO)
    weak_a = _chunk("weak-a", 0.35)
    weak_b = _chunk("weak-b", 0.21)

    filtered = _filter_by_relevance_score([weak_a, weak_b])

    assert filtered == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "zero evidence" in warnings[0].getMessage()
    assert warnings[0].dropped_count == 2


def test_filter_does_not_warn_when_input_is_already_empty(caplog) -> None:
    """An empty input pool (no chunks retrieved at all) is a different
    condition than "everything got filtered out" -- must not also trip
    the all-dropped WARNING."""
    caplog.set_level(logging.INFO)

    filtered = _filter_by_relevance_score([])

    assert filtered == []
    assert caplog.records == []
