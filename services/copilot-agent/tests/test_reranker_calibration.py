"""Regression test for issue #99: the evidence-relevance floor
(`app/chat.py`'s ``_EVIDENCE_MIN_RELEVANCE_SCORE``) must be calibrated
against the model that ACTUALLY scores relevance in production, not
whichever model happened to back the original fixture.

PR #98 introduced a 0.5 floor calibrated only against
``app/data/reranker_scores.json`` (model="qwen3:4b", via Ollama), while
``app.config.Settings.copilot_llm_engine`` defaults to "llama_server"
(qwen3-8b, ``llama_server_model``) -- a different model was actually scoring
in production than the one the floor was measured against. Two checks here
close that gap for good:

  1. ``test_calibration_fixture_model_matches_production_engine`` pins the
     model name stamped in ``app/data/reranker_scores_qwen3-8b.json`` (the
     issue #99 re-measurement, recorded via
     ``RERANKER_ENGINE=llama_server scripts/build_reranker_scores.py``) to
     ``Settings().llama_server_model`` -- if a future model swap changes
     ``llama_server_model`` without regenerating this fixture, this test
     fails loudly instead of the floor silently going stale again.
  2. ``test_relevance_floor_sits_above_every_measured_distractor_and_below_
     every_measured_match`` re-derives the SAME gap-based invariant
     ``_EVIDENCE_MIN_RELEVANCE_SCORE``'s docstring reasons about (every
     genuinely-relevant chunk scores above the floor, every deliberately
     wrong lexical distractor scores below it) directly from the recorded
     qwen3-8b fixture, so a future re-recording that narrows or closes the
     gap is caught here rather than discovered live.

Hermetic: reads the committed fixture and corpus text, no live LLM call.
"""

from __future__ import annotations

import json

from app.chat import _EVIDENCE_MIN_RELEVANCE_SCORE, _wants_llama_server
from app.config import Settings
from app.reranking import RERANKER_SCORES_PATH
from app.retrieval import CORPUS_DIR, chunk_text_sha256, parse_corpus
from scripts.reranker_golden_distractors import GOLDEN_DISTRACTORS
from scripts.retrieval_golden_queries import GOLDEN_QUERIES

_QWEN3_8B_FIXTURE_PATH = RERANKER_SCORES_PATH.parent / "reranker_scores_qwen3-8b.json"


def _load_fixture() -> dict[str, object]:
    return json.loads(_QWEN3_8B_FIXTURE_PATH.read_text(encoding="utf-8"))


def test_calibration_fixture_model_matches_production_engine() -> None:
    """The qwen3-8b calibration fixture must actually be stamped with the
    model production's ``copilot_llm_engine`` selects (via
    ``_wants_llama_server`` -- the single place that flag is compared, see
    its docstring) for the reranker role, NOT hardcoded to
    ``llama_server_model`` regardless of which engine is active. This is the
    exact drift issue #99 found: a fixture recorded against one model while
    a DIFFERENT model scores in production, with nothing catching the
    mismatch -- and asserting only ``llama_server_model`` (ignoring the
    engine flag) would silently stop meaning anything the day
    ``copilot_llm_engine`` defaults back to "ollama"."""
    settings = Settings()
    payload = _load_fixture()

    expected_model = settings.llama_server_model if _wants_llama_server(settings) else settings.ollama_model
    assert payload["model"] == expected_model, (
        f"reranker_scores_qwen3-8b.json is stamped model={payload['model']!r}, but "
        f"copilot_llm_engine={settings.copilot_llm_engine!r} currently selects "
        f"{expected_model!r} for the reranker role -- regenerate the fixture "
        "(scripts/build_reranker_scores.py) against whichever engine/model is actually "
        "active before trusting this calibration."
    )


def test_relevance_floor_sits_above_every_measured_distractor_and_below_every_measured_match() -> None:
    """For every golden query, its planted lexical distractor (a chunk a
    hybrid retrieval stage ranks high but that does not answer the query --
    see ``scripts/reranker_golden_distractors.py``) must score BELOW
    ``_EVIDENCE_MIN_RELEVANCE_SCORE`` in the qwen3-8b fixture, and its
    genuinely-relevant expected chunk must score AT OR ABOVE it -- the same
    invariant ``_EVIDENCE_MIN_RELEVANCE_SCORE``'s docstring measured by hand.
    """
    payload = _load_fixture()
    scores: dict[str, dict[str, float]] = payload["scores"]  # type: ignore[assignment]
    chunk_text_by_id = {chunk.chunk_id: chunk.text for chunk in parse_corpus(CORPUS_DIR)}

    for query, expected_chunk_id in GOLDEN_QUERIES:
        distractor_id = GOLDEN_DISTRACTORS[expected_chunk_id]
        query_scores = scores[query]

        gold_hash = chunk_text_sha256(chunk_text_by_id[expected_chunk_id])
        distractor_hash = chunk_text_sha256(chunk_text_by_id[distractor_id])

        gold_score = query_scores[gold_hash]
        distractor_score = query_scores[distractor_hash]

        assert gold_score >= _EVIDENCE_MIN_RELEVANCE_SCORE, (
            f"{query!r}: expected chunk {expected_chunk_id!r} scored {gold_score} -- "
            f"below the {_EVIDENCE_MIN_RELEVANCE_SCORE} floor, would be wrongly dropped"
        )
        assert distractor_score < _EVIDENCE_MIN_RELEVANCE_SCORE, (
            f"{query!r}: planted distractor {distractor_id!r} scored {distractor_score} -- "
            f"at/above the {_EVIDENCE_MIN_RELEVANCE_SCORE} floor, would be wrongly kept"
        )
