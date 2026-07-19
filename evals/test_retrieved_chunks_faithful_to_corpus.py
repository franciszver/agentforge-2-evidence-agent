"""Guard test: every eval case's canned ``retrieved_chunks`` fixture must be
verbatim-faithful to the REAL guideline corpus (integrity gate, post-hoc).

**The problem this prevents.** ``runner.pipeline.run_case`` builds the
citation-verification index (``CorpusChunkIndex``) from each case's OWN
``retrieved_chunks`` fixture field (see ``RetrievedChunkFixture.to_reranked_chunk``),
NOT from the real corpus (``app.retrieval.parse_corpus``). If a fixture's
``text`` is a paraphrase/compression of the real corpus section, a model
quote can "verify" against the doctored fixture even though it is NOT a
verbatim substring of the real corpus -- a fabricated pass, indistinguishable
from a genuine one by ``evals/test_cases.py`` alone. (A real instance of this
was found and fixed: 17 of 20 ``citation_present``/``factually_consistent``
cases had fixture text that silently dropped clauses/list-items present in
the real corpus section -- see git history for the fix.)

**What this test checks**, for EVERY case (any category) with a non-empty
``retrieved_chunks``:

  1. Each fixture's ``chunk_id`` resolves to a REAL corpus section --
     parsed the same way ``app.retrieval.parse_document`` parses it (one
     ``##`` section per chunk, keyed ``<doc_id>#<section-slug>``). A
     ``chunk_id`` that doesn't exist in the real corpus is exactly the kind
     of fabricated citation source the live checker's ``UNKNOWN_CHUNK``
     branch guards against -- a fixture referencing one would silently
     never be checkable at all.
  2. The fixture's ``text`` is verbatim-equal to that real section's text,
     whitespace-normalized consistently on both sides (every run of
     whitespace -- including a YAML folded-scalar's line-wrap newlines --
     collapsed to a single space, then stripped). This catches a
     paraphrase, a truncation, or a dropped clause/sentence/list-item: any
     of those changes the non-whitespace CHARACTER sequence, which survives
     whitespace normalization and fails the comparison -- exactly the same
     property the real ``check_document_citation`` substring check relies
     on for its own no-fabrication guarantee (see ``app/verification.py``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.retrieval import CORPUS_DIR, parse_corpus
from runner.loader import discover_case_files, load_case
from runner.schema import RetrievedChunkFixture

_CASES_DIR = Path(__file__).parent / "cases"
_REGRESSIONS_DIR = Path(__file__).parent / "regressions"

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Collapse every run of whitespace (space/tab/newline) to a single
    space and strip the ends -- absorbs a YAML folded scalar's line-wrap
    newlines and the corpus markdown's own hard-wrap newlines, while still
    requiring every non-whitespace character to match in the same order."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _real_corpus_chunk_texts() -> dict[str, str]:
    return {chunk.chunk_id: chunk.text for chunk in parse_corpus(CORPUS_DIR)}


def _cases_with_retrieved_chunks() -> list[tuple[str, RetrievedChunkFixture]]:
    """``(case_id, fixture)`` for every ``retrieved_chunks`` entry across
    every discovered case file -- one entry per fixture chunk, so a case
    with several chunks gets several independently-reported failures rather
    than one bundled assertion."""
    pairs: list[tuple[str, RetrievedChunkFixture]] = []
    for case_file in discover_case_files(_CASES_DIR, _REGRESSIONS_DIR):
        case = load_case(case_file)
        for fixture in case.retrieved_chunks:
            pairs.append((case.id, fixture))
    return pairs


_CASE_CHUNK_PAIRS = _cases_with_retrieved_chunks()
_CASE_CHUNK_IDS = [f"{case_id}::{fixture.chunk_id}" for case_id, fixture in _CASE_CHUNK_PAIRS]


def test_at_least_one_case_has_retrieved_chunks() -> None:
    """Sanity check on the test collection itself: if this ever collects
    zero pairs (e.g. a future refactor silently stops populating
    ``retrieved_chunks``), the parametrized test below would vacuously pass
    with nothing to check -- assert the corpus of cases-with-chunks is
    non-empty so that silent gap can't hide."""
    assert _CASE_CHUNK_PAIRS, "expected at least one eval case with retrieved_chunks"


@pytest.mark.parametrize("case_id, fixture", _CASE_CHUNK_PAIRS, ids=_CASE_CHUNK_IDS)
def test_retrieved_chunk_is_faithful_to_real_corpus(case_id: str, fixture: RetrievedChunkFixture) -> None:
    real_chunks = _real_corpus_chunk_texts()

    assert fixture.chunk_id in real_chunks, (
        f"case {case_id!r}: retrieved_chunks references chunk_id "
        f"{fixture.chunk_id!r}, which does not exist in the real corpus "
        "(app.retrieval.parse_corpus) -- a fixture must only reference a "
        "real corpus section."
    )

    real_text = real_chunks[fixture.chunk_id]
    assert _normalize(fixture.text) == _normalize(real_text), (
        f"case {case_id!r}: retrieved_chunks[{fixture.chunk_id!r}].text is NOT "
        "verbatim-faithful to the real corpus section (whitespace-normalized "
        "comparison) -- the fixture must carry the corpus's full, exact text, "
        "never a paraphrase, truncation, or summary. A model quote verified "
        "against a doctored fixture is a false pass: the same quote may not "
        "be a genuine substring of the real corpus section this fixture "
        "claims to represent."
    )


# ---------------------------------------------------------------------------
# Proof this guard actually catches doctoring -- exercises the same
# normalize-and-compare logic above directly against synthetic paraphrased/
# truncated text (never mutates a real case file), so the guard's own
# detection behavior is independently regression-tested.
# ---------------------------------------------------------------------------


def test_guard_detects_a_paraphrased_chunk() -> None:
    real_text = (
        "Warfarin's effect is sensitive to many interacting drugs and to "
        "vitamin K intake. Common interaction cautions include: NSAIDs and "
        "other antiplatelet agents, certain antibiotics, and amiodarone, "
        "which can meaningfully raise INR."
    )
    # The exact shape of the doctoring this guard was written to catch: a
    # compressed paraphrase that drops clauses ("NSAIDs and other
    # antiplatelet agents", "and amiodarone...") that are present in the
    # real corpus section.
    doctored_text = (
        "Common interaction cautions include certain antibiotics which can "
        "potentiate warfarin's effect."
    )
    assert _normalize(doctored_text) != _normalize(real_text)


def test_guard_detects_a_truncated_chunk() -> None:
    real_text = "- Item one.\n- Item two.\n- Item three."
    truncated_text = "- Item one."
    assert _normalize(truncated_text) != _normalize(real_text)


def test_guard_accepts_only_whitespace_differences() -> None:
    """Regression guard on the normalization itself: a fixture that only
    differs from the corpus in incidental whitespace (line-wrap position,
    a folded-scalar newline standing in for a hard-wrap newline) must still
    be accepted -- this guard is about WORD fidelity, not byte-for-byte
    formatting fidelity."""
    real_text = "borderline-high 130-159 mg/dL; high 160-189\nmg/dL or above."
    fixture_text = "borderline-high 130-159 mg/dL; high 160-189 mg/dL or above."
    assert _normalize(fixture_text) == _normalize(real_text)
