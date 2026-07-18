"""Exhaustive matrix for the P3.6 document-citation extension to the
deterministic citation checker (``app.verification``).

Every clinical claim sourced from a Week-2 document (lab-report PDF, intake
form, or a hybrid-retrieval guideline chunk) carries a ``DocumentCitation``
(``app.schemas.ingestion``) -- the document-sourced counterpart to Phase 1's
``SourceRef``. This module re-validates each ``DocumentCitation`` against the
RAW source it names:

- ``lab_pdf``/``intake_form``: re-checked against ``DocumentFactIndex``, built
  from the RAW extracted-fact ``Citation``s (what
  ``app.ingestion.LocalIngestionStore``/``FactStore`` actually persisted) --
  never a re-derived or paraphrased copy.
- ``guideline_chunk``: re-checked against ``CorpusChunkIndex``, built from the
  RAW corpus chunk text (what ``app.retrieval.parse_corpus``/
  ``HybridRetriever`` actually returns) -- the quote must appear verbatim in
  that text, not just be a plausible paraphrase.

A claim's citations (``SourceRef`` AND ``DocumentCitation``, mixed) flow
through the SAME ``ClaimCheckResult.passed`` AND-aggregation as Phase 1 -- see
``test_mixed_citation_claim_*`` below.

Hermetic and fully deterministic: no fixtures touch a real Ollama/OpenEMR/PDF.
"""

from __future__ import annotations

from app.quarantine import REDACTED_SENTINEL
from app.schemas.common import SourceRef
from app.schemas.ingestion import Citation, DocumentCitation
from app.schemas.verification import Claim
from app.verification import (
    CacheIndex,
    CitationStatus,
    CorpusChunkIndex,
    DocumentFactIndex,
    check_claim,
    check_document_citation,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _lab_citation(
    *,
    source_id: str = "doc-1",
    field_or_chunk_id: str = "Glucose",
    quote_or_value: str = "Glucose: 105 mg/dL",
) -> DocumentCitation:
    return DocumentCitation(
        source_type="lab_pdf",
        source_id=source_id,
        page_or_section="page 1",
        field_or_chunk_id=field_or_chunk_id,
        quote_or_value=quote_or_value,
    )


def _chunk_citation(
    *,
    source_id: str = "hypertension-guideline",
    field_or_chunk_id: str = "hypertension-guideline#first-line-treatment",
    quote_or_value: str = "Thiazide diuretics are first-line therapy for most patients.",
) -> DocumentCitation:
    return DocumentCitation(
        source_type="guideline_chunk",
        source_id=source_id,
        page_or_section="First-line Treatment",
        field_or_chunk_id=field_or_chunk_id,
        quote_or_value=quote_or_value,
    )


_RAW_LAB_CITATION = Citation(
    source_type="lab_pdf",
    source_id="doc-1",
    page_or_section="page 1",
    field_or_chunk_id="Glucose",
    quote_or_value="Glucose: 105 mg/dL",
)

_RAW_CHUNK_TEXT = "Thiazide diuretics are first-line therapy for most patients."
_RAW_CHUNK_ID = "hypertension-guideline#first-line-treatment"


class _FakeChunk:
    """Minimal stand-in for ``app.retrieval.Chunk``/``RetrievedChunk`` --
    ``CorpusChunkIndex.from_chunks`` only needs ``chunk_id``/``text``."""

    def __init__(self, chunk_id: str, text: str) -> None:
        self.chunk_id = chunk_id
        self.text = text


def _fact_index(*citations: Citation) -> DocumentFactIndex:
    return DocumentFactIndex.from_citations(list(citations))


def _corpus_index(*chunks: _FakeChunk) -> CorpusChunkIndex:
    return CorpusChunkIndex.from_chunks(list(chunks))


# ---------------------------------------------------------------------------
# (a) valid lab-fact citation verifies against the raw stored extraction
# ---------------------------------------------------------------------------


def test_lab_citation_matching_raw_stored_extraction_verifies():
    fact_index = _fact_index(_RAW_LAB_CITATION)
    corpus_index = _corpus_index()

    result = check_document_citation(_lab_citation(), fact_index, corpus_index)

    assert result.status is CitationStatus.VALID
    assert result.passed is True


def test_intake_citation_matching_raw_stored_extraction_verifies():
    raw = Citation(
        source_type="intake_form",
        source_id="doc-2",
        page_or_section="page 1",
        field_or_chunk_id="chief_concern",
        quote_or_value="chief_concern: chest pain",
    )
    fact_index = _fact_index(raw)

    citation = DocumentCitation(
        source_type="intake_form",
        source_id="doc-2",
        page_or_section="page 1",
        field_or_chunk_id="chief_concern",
        quote_or_value="chief_concern: chest pain",
    )
    result = check_document_citation(citation, fact_index, _corpus_index())

    assert result.status is CitationStatus.VALID


# ---------------------------------------------------------------------------
# (b) a citation whose quote does NOT match the raw source fails
# ---------------------------------------------------------------------------


def test_lab_citation_with_mismatched_quote_fails():
    fact_index = _fact_index(_RAW_LAB_CITATION)

    result = check_document_citation(
        _lab_citation(quote_or_value="Glucose: 250 mg/dL"), fact_index, _corpus_index()
    )

    assert result.status is CitationStatus.VALUE_MISMATCH
    assert result.passed is False


# ---------------------------------------------------------------------------
# (c) guideline-chunk citation: verbatim quote verifies, hallucinated
# paraphrase fails
# ---------------------------------------------------------------------------


def test_guideline_chunk_citation_with_verbatim_quote_verifies():
    corpus_index = _corpus_index(_FakeChunk(_RAW_CHUNK_ID, _RAW_CHUNK_TEXT))

    result = check_document_citation(_chunk_citation(), _fact_index(), corpus_index)

    assert result.status is CitationStatus.VALID


def test_guideline_chunk_citation_with_hallucinated_paraphrase_fails():
    corpus_index = _corpus_index(_FakeChunk(_RAW_CHUNK_ID, _RAW_CHUNK_TEXT))

    result = check_document_citation(
        _chunk_citation(quote_or_value="You should never use thiazides for hypertension."),
        _fact_index(),
        corpus_index,
    )

    assert result.status is CitationStatus.QUOTE_NOT_FOUND
    assert result.passed is False


# ---------------------------------------------------------------------------
# (d) a citation with a nonexistent source_id/chunk_id fails (no fabrication)
# ---------------------------------------------------------------------------


def test_lab_citation_nonexistent_source_id_fails():
    fact_index = _fact_index(_RAW_LAB_CITATION)

    result = check_document_citation(
        _lab_citation(source_id="doc-does-not-exist"), fact_index, _corpus_index()
    )

    assert result.status is CitationStatus.UNKNOWN_SOURCE


def test_lab_citation_nonexistent_field_fails():
    fact_index = _fact_index(_RAW_LAB_CITATION)

    result = check_document_citation(
        _lab_citation(field_or_chunk_id="Nonexistent Test"), fact_index, _corpus_index()
    )

    assert result.status is CitationStatus.UNKNOWN_FIELD


def test_guideline_chunk_citation_nonexistent_chunk_id_fails():
    corpus_index = _corpus_index(_FakeChunk(_RAW_CHUNK_ID, _RAW_CHUNK_TEXT))

    result = check_document_citation(
        _chunk_citation(field_or_chunk_id="hypertension-guideline#does-not-exist"),
        _fact_index(),
        corpus_index,
    )

    assert result.status is CitationStatus.UNKNOWN_CHUNK


def test_empty_indices_fail_every_document_citation_closed():
    result_lab = check_document_citation(_lab_citation(), _fact_index(), _corpus_index())
    result_chunk = check_document_citation(_chunk_citation(), _fact_index(), _corpus_index())

    assert result_lab.status is CitationStatus.UNKNOWN_SOURCE
    assert result_chunk.status is CitationStatus.UNKNOWN_CHUNK


# ---------------------------------------------------------------------------
# (e) mixed-citation claim: verified only if ALL citations verify -- same
# AND-aggregation as Phase 1 SourceRef claims (app.verification.ClaimCheckResult)
# ---------------------------------------------------------------------------


def _index(*raw_results: dict | None) -> CacheIndex:
    return CacheIndex.from_raw_results(list(raw_results))


_MEDS_RESULT = {
    "items": [
        {"name": "Lisinopril", "dose": "10mg", "status": "active"},
    ]
}


def test_mixed_citation_claim_all_valid_passes():
    index = _index(_MEDS_RESULT)
    fact_index = _fact_index(_RAW_LAB_CITATION)
    corpus_index = _corpus_index(_FakeChunk(_RAW_CHUNK_ID, _RAW_CHUNK_TEXT))
    claim = Claim(
        text="On Lisinopril; glucose 105 mg/dL; thiazides are first-line.",
        source_refs=[SourceRef(tool_call_id="call_0", record_id="0", field="name", asserted_value="Lisinopril")],
        document_citations=[_lab_citation(), _chunk_citation()],
    )

    result = check_claim(claim, index, fact_index, corpus_index)

    assert result.passed is True
    assert len(result.citation_results) == 3


def test_mixed_citation_claim_one_document_citation_invalid_fails_whole_claim():
    index = _index(_MEDS_RESULT)
    fact_index = _fact_index(_RAW_LAB_CITATION)
    corpus_index = _corpus_index(_FakeChunk(_RAW_CHUNK_ID, _RAW_CHUNK_TEXT))
    claim = Claim(
        text="On Lisinopril; glucose 250 mg/dL.",
        source_refs=[SourceRef(tool_call_id="call_0", record_id="0", field="name", asserted_value="Lisinopril")],
        document_citations=[_lab_citation(quote_or_value="Glucose: 250 mg/dL")],
    )

    result = check_claim(claim, index, fact_index, corpus_index)

    assert result.passed is False
    # Both citations still reported, not short-circuited.
    assert len(result.citation_results) == 2


def test_claim_with_only_document_citations_all_valid_passes():
    fact_index = _fact_index(_RAW_LAB_CITATION)
    claim = Claim(
        text="Glucose is 105 mg/dL.",
        document_citations=[_lab_citation()],
    )

    result = check_claim(claim, _index(), fact_index, _corpus_index())

    assert result.passed is True


def test_claim_with_only_document_citations_one_invalid_fails():
    fact_index = _fact_index(_RAW_LAB_CITATION)
    claim = Claim(
        text="Glucose is 250 mg/dL.",
        document_citations=[_lab_citation(quote_or_value="Glucose: 250 mg/dL")],
    )

    result = check_claim(claim, _index(), fact_index, _corpus_index())

    assert result.passed is False


# ---------------------------------------------------------------------------
# (f) regression guard: verification reads the RAW source, not a
# quarantined/cached copy
# ---------------------------------------------------------------------------


def test_document_fact_index_defensive_fail_closed_on_redacted_quote():
    # A raw fact store should NEVER contain the quarantine sentinel -- this is
    # the document-citation counterpart to
    # app.verification's SourceRef REDACTED_FIELD belt-and-suspenders branch.
    # If a caller accidentally builds the index from a quarantined/redacted
    # copy instead of the raw extracted facts, this must fail closed rather
    # than silently compare against placeholder text.
    redacted = Citation(
        source_type="lab_pdf",
        source_id="doc-1",
        page_or_section="page 1",
        field_or_chunk_id="Glucose",
        quote_or_value=REDACTED_SENTINEL,
    )
    fact_index = _fact_index(redacted)

    result = check_document_citation(_lab_citation(), fact_index, _corpus_index())

    assert result.status is CitationStatus.REDACTED_FIELD
    assert result.passed is False


def test_document_citation_verifies_against_the_true_raw_quote_not_a_paraphrased_cache():
    # The same citation, checked against two different indices: one built
    # from the TRUE raw extracted quote (verifies), one built from a
    # plausible-looking but DIFFERENT cached/paraphrased value at the same
    # (source_id, field_or_chunk_id) key (fails). This proves the checker's
    # verdict is entirely governed by whichever store it is handed -- so a
    # caller MUST hand it the raw extraction store, never a summarized or
    # re-derived cache, or a citation that is actually correct would be
    # wrongly stripped (or a wrong one wrongly kept).
    true_raw_index = _fact_index(_RAW_LAB_CITATION)
    cached_paraphrase = Citation(
        source_type="lab_pdf",
        source_id="doc-1",
        page_or_section="page 1",
        field_or_chunk_id="Glucose",
        quote_or_value="Glucose: elevated",  # a paraphrase, not the raw quote
    )
    stale_cache_index = _fact_index(cached_paraphrase)

    citation = _lab_citation()  # asserts the TRUE raw quote, "Glucose: 105 mg/dL"

    assert check_document_citation(citation, true_raw_index, _corpus_index()).status is CitationStatus.VALID
    assert (
        check_document_citation(citation, stale_cache_index, _corpus_index()).status
        is CitationStatus.VALUE_MISMATCH
    )


def test_guideline_chunk_citation_verifies_against_raw_corpus_text_not_a_reranked_summary():
    # Same invariant for guideline-chunk citations: the corpus index must be
    # built from the RAW chunk text, not a shortened/summarized stand-in a
    # reranker or retrieval layer might otherwise be tempted to substitute.
    raw_corpus_index = _corpus_index(_FakeChunk(_RAW_CHUNK_ID, _RAW_CHUNK_TEXT))
    summarized_index = _corpus_index(_FakeChunk(_RAW_CHUNK_ID, "Use thiazides."))

    citation = _chunk_citation()  # quotes the full raw sentence verbatim

    assert check_document_citation(citation, _fact_index(), raw_corpus_index).status is CitationStatus.VALID
    assert (
        check_document_citation(citation, _fact_index(), summarized_index).status
        is CitationStatus.QUOTE_NOT_FOUND
    )
