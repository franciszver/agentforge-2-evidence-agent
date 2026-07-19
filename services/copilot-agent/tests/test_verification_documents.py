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

import pytest

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
# Code-review finding: a duplicate (source_id, field_or_chunk_id)/chunk_id
# key must be a LOUD failure, not silent last-wins -- two distinct facts
# colliding on the same key would otherwise silently mis-associate a quote,
# order-dependently, with no signal anything went wrong. Proper fix (a truly
# unique id per fact/chunk) is tracked as follow-up issue #40; this is the
# cheap guard: detect and raise loudly instead of silently overwriting.
# ---------------------------------------------------------------------------


def test_from_citations_raises_on_duplicate_key_with_differing_values():
    first = Citation(
        source_type="lab_pdf",
        source_id="doc-1",
        page_or_section="page 1",
        field_or_chunk_id="Glucose",
        quote_or_value="Glucose: 105 mg/dL",
    )
    second = Citation(
        source_type="lab_pdf",
        source_id="doc-1",
        page_or_section="page 2",
        field_or_chunk_id="Glucose",  # same (source_id, field_or_chunk_id) as `first`
        quote_or_value="Glucose: 250 mg/dL",  # a DIFFERENT value -- a real collision
    )

    with pytest.raises(ValueError, match=r"doc-1.*Glucose"):
        DocumentFactIndex.from_citations([first, second])


def test_from_citations_builds_fine_with_no_colliding_keys():
    first = Citation(
        source_type="lab_pdf",
        source_id="doc-1",
        page_or_section="page 1",
        field_or_chunk_id="Glucose",
        quote_or_value="Glucose: 105 mg/dL",
    )
    second = Citation(
        source_type="lab_pdf",
        source_id="doc-1",
        page_or_section="page 1",
        field_or_chunk_id="A1c",
        quote_or_value="A1c: 5.4%",
    )

    index = DocumentFactIndex.from_citations([first, second])

    assert index.quote_for("doc-1", "Glucose") == "Glucose: 105 mg/dL"
    assert index.quote_for("doc-1", "A1c") == "A1c: 5.4%"


def test_from_chunks_raises_on_duplicate_chunk_id_with_differing_text():
    with pytest.raises(ValueError, match=_RAW_CHUNK_ID):
        CorpusChunkIndex.from_chunks(
            [
                _FakeChunk(_RAW_CHUNK_ID, _RAW_CHUNK_TEXT),
                _FakeChunk(_RAW_CHUNK_ID, "A different, colliding chunk text."),
            ]
        )


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
# (c2) P3G.1b whitespace-normalized substring check: a faithful quote that
# only differs from the raw chunk in whitespace (e.g. a line-folded hyphen)
# now verifies; a quote with an actual different/added/changed WORD still
# fails -- the no-fabrication guarantee is unaffected by normalization.
# ---------------------------------------------------------------------------

_LINE_FOLDED_CHUNK_TEXT = (
    "LDL cholesterol: optimal below 100 mg/dL; near-optimal 100-129 mg/dL; borderline-"
    " high 130-159 mg/dL; high 160-189 mg/dL; very high 190 mg/dL or above."
)
_LINE_FOLDED_CHUNK_ID = "lipid-panel-reference#general-reference-categories"


def test_guideline_chunk_citation_with_line_folded_whitespace_now_verifies():
    # The chunk stores a line-folded hyphen ("borderline-" + " high" -- an
    # extra internal space from the source's line wrap). A model that
    # faithfully quotes the same words but collapses that internal space
    # (emits "borderline-high") must now VERIFY -- this is the exact
    # lipid-panel-ldl-question near-miss this fix targets.
    corpus_index = _corpus_index(_FakeChunk(_LINE_FOLDED_CHUNK_ID, _LINE_FOLDED_CHUNK_TEXT))
    citation = _chunk_citation(
        source_id="lipid-panel-reference",
        field_or_chunk_id=_LINE_FOLDED_CHUNK_ID,
        quote_or_value="borderline-high 130-159 mg/dL",
    )

    result = check_document_citation(citation, _fact_index(), corpus_index)

    assert result.status is CitationStatus.VALID
    assert result.passed is True


def test_guideline_chunk_citation_with_extra_internal_whitespace_still_verifies():
    # The inverse direction: the quote reproduces the chunk's own extra
    # internal space verbatim -- must still verify (this already passed
    # before the fix; regression guard that normalization doesn't break it).
    corpus_index = _corpus_index(_FakeChunk(_LINE_FOLDED_CHUNK_ID, _LINE_FOLDED_CHUNK_TEXT))
    citation = _chunk_citation(
        source_id="lipid-panel-reference",
        field_or_chunk_id=_LINE_FOLDED_CHUNK_ID,
        quote_or_value="borderline- high 130-159 mg/dL",
    )

    result = check_document_citation(citation, _fact_index(), corpus_index)

    assert result.status is CitationStatus.VALID
    assert result.passed is True


def test_guideline_chunk_citation_with_changed_word_still_fails_no_fabrication():
    # No-fabrication guard, proven post-fix: a quote differing by an actual
    # WORD (not whitespace) -- "borderline-high" vs the chunk's "near-optimal"
    # range wording -- must still fail. Whitespace normalization must never
    # let a substantively different quote through.
    corpus_index = _corpus_index(_FakeChunk(_LINE_FOLDED_CHUNK_ID, _LINE_FOLDED_CHUNK_TEXT))
    citation = _chunk_citation(
        source_id="lipid-panel-reference",
        field_or_chunk_id=_LINE_FOLDED_CHUNK_ID,
        quote_or_value="borderline-high 100-129 mg/dL",  # wrong range -- not in the chunk
    )

    result = check_document_citation(citation, _fact_index(), corpus_index)

    assert result.status is CitationStatus.QUOTE_NOT_FOUND
    assert result.passed is False


def test_guideline_chunk_citation_with_added_word_still_fails_no_fabrication():
    corpus_index = _corpus_index(_FakeChunk(_LINE_FOLDED_CHUNK_ID, _LINE_FOLDED_CHUNK_TEXT))
    citation = _chunk_citation(
        source_id="lipid-panel-reference",
        field_or_chunk_id=_LINE_FOLDED_CHUNK_ID,
        quote_or_value="borderline-high and rising 130-159 mg/dL",  # words inserted mid-quote
    )

    result = check_document_citation(citation, _fact_index(), corpus_index)

    assert result.status is CitationStatus.QUOTE_NOT_FOUND
    assert result.passed is False


def test_guideline_chunk_citation_empty_quote_still_fails_after_normalization():
    # Empty/whitespace-only guard runs BEFORE normalization -- unaffected.
    corpus_index = _corpus_index(_FakeChunk(_LINE_FOLDED_CHUNK_ID, _LINE_FOLDED_CHUNK_TEXT))

    result = check_document_citation(
        _bypassed_chunk_citation("   \n\t  "), _fact_index(), corpus_index
    )

    assert result.status is CitationStatus.EMPTY_QUOTE
    assert result.passed is False


def test_guideline_chunk_citation_short_quote_still_fails_after_normalization():
    # Length-floor guard runs BEFORE normalization -- unaffected: a short
    # quote with only whitespace variation is still too short to be a
    # meaningful citation.
    corpus_index = _corpus_index(_FakeChunk(_LINE_FOLDED_CHUNK_ID, _LINE_FOLDED_CHUNK_TEXT))

    result = check_document_citation(
        _chunk_citation(
            source_id="lipid-panel-reference",
            field_or_chunk_id=_LINE_FOLDED_CHUNK_ID,
            quote_or_value="a  b",
        ),
        _fact_index(),
        corpus_index,
    )

    assert result.status is CitationStatus.EMPTY_QUOTE
    assert result.passed is False


# ---------------------------------------------------------------------------
# (c3) Narrowed whitespace normalization (security-gate finding): the P3G.1b
# fix above originally stripped ALL whitespace, which would let a quote of
# "50" match chunk text containing "5 0" -- silently collapsing two distinct
# numeric tokens into one and inventing a match that was never really there.
# The narrowed normalization (collapse whitespace runs to one space, then
# fold only whitespace immediately ADJACENT TO A HYPHEN) must still bridge
# the hyphen-fold case while refusing to bridge a plain token-separating
# space.
# ---------------------------------------------------------------------------


def test_narrowed_whitespace_normalization_does_not_collapse_distinct_tokens():
    # "5 0" (two distinct tokens, no hyphen) must NOT be matched by a quote
    # of "50" -- proves the narrowed normalization does not fabricate a
    # match by bridging a token-separating space, unlike the old
    # strip-all-whitespace behavior.
    corpus_index = _corpus_index(_FakeChunk(_RAW_CHUNK_ID, "The dose range is 5 0 mg, not otherwise specified."))
    citation = _chunk_citation(quote_or_value="50 mg")

    result = check_document_citation(citation, _fact_index(), corpus_index)

    assert result.status is CitationStatus.QUOTE_NOT_FOUND
    assert result.passed is False


def test_narrowed_whitespace_normalization_still_folds_hyphen_adjacent_whitespace():
    # "borderline-high" (no space) must still verify against the chunk's
    # line-folded "borderline- high" (one space next to the hyphen) -- the
    # narrowing must not regress the original P3G.1b fix this guards.
    corpus_index = _corpus_index(_FakeChunk(_LINE_FOLDED_CHUNK_ID, _LINE_FOLDED_CHUNK_TEXT))
    citation = _chunk_citation(
        source_id="lipid-panel-reference",
        field_or_chunk_id=_LINE_FOLDED_CHUNK_ID,
        quote_or_value="borderline-high 130-159 mg/dL",
    )

    result = check_document_citation(citation, _fact_index(), corpus_index)

    assert result.status is CitationStatus.VALID
    assert result.passed is True


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


# ---------------------------------------------------------------------------
# Security gate finding: empty/whitespace/trivially-short quote_or_value must
# NOT trivially verify a guideline_chunk citation. "".strip() in chunk_text is
# vacuously True, and a 1-2 char quote substring-matches almost any text --
# both would let a citation asserting NOTHING pass as VALID. Guarded in TWO
# places (defense in depth): the schema rejects a blank quote at
# construction (DocumentCitation), and the checker independently re-guards
# against a citation that bypassed schema validation (``model_construct``,
# same fail-closed posture as the pre-existing zero-citation Claim test).
# ---------------------------------------------------------------------------


def _bypassed_chunk_citation(quote_or_value: str) -> DocumentCitation:
    """Build a ``DocumentCitation`` bypassing pydantic validation --
    simulates a citation that reached the checker without having gone
    through the schema's non-blank guard, so the checker's OWN defensive
    guard is what's under test here."""
    return DocumentCitation.model_construct(
        source_type="guideline_chunk",
        source_id="hypertension-guideline",
        page_or_section="First-line Treatment",
        field_or_chunk_id=_RAW_CHUNK_ID,
        quote_or_value=quote_or_value,
    )


def test_guideline_chunk_citation_with_empty_quote_fails_even_though_chunk_exists():
    corpus_index = _corpus_index(_FakeChunk(_RAW_CHUNK_ID, _RAW_CHUNK_TEXT))

    result = check_document_citation(_bypassed_chunk_citation(""), _fact_index(), corpus_index)

    assert result.status is CitationStatus.EMPTY_QUOTE
    assert result.passed is False


def test_guideline_chunk_citation_with_whitespace_only_quote_fails():
    corpus_index = _corpus_index(_FakeChunk(_RAW_CHUNK_ID, _RAW_CHUNK_TEXT))

    result = check_document_citation(_bypassed_chunk_citation("   "), _fact_index(), corpus_index)

    assert result.status is CitationStatus.EMPTY_QUOTE
    assert result.passed is False


def test_guideline_chunk_citation_with_one_char_quote_fails_too_short():
    # A 1-char (or 2-char) quote substring-matches almost any real chunk
    # text -- not a meaningful citation of content. Rejected below the
    # 3-non-whitespace-char floor (see module docstring / verification.py's
    # _MIN_CHUNK_QUOTE_NON_WHITESPACE_CHARS).
    corpus_index = _corpus_index(_FakeChunk(_RAW_CHUNK_ID, _RAW_CHUNK_TEXT))

    result = check_document_citation(_chunk_citation(quote_or_value="a"), _fact_index(), corpus_index)

    assert result.status is CitationStatus.EMPTY_QUOTE
    assert result.passed is False


def test_guideline_chunk_citation_with_real_verbatim_quote_still_passes():
    # Regression guard: the empty/too-short guards must not over-reach and
    # start failing legitimate, real quotes.
    corpus_index = _corpus_index(_FakeChunk(_RAW_CHUNK_ID, _RAW_CHUNK_TEXT))

    result = check_document_citation(_chunk_citation(), _fact_index(), corpus_index)

    assert result.status is CitationStatus.VALID
    assert result.passed is True


def test_lab_citation_with_empty_quote_fails_even_though_source_and_field_exist():
    # Defense in depth on the lab_pdf/intake_form equality path too -- the
    # empty-quote guard applies to BOTH source types, though NOT the
    # length-floor (a legitimate lab value like "7" must still verify --
    # see test_lab_citation_short_numeric_quote_still_verifies below).
    fact_index = _fact_index(_RAW_LAB_CITATION)
    bypassed = DocumentCitation.model_construct(
        source_type="lab_pdf",
        source_id="doc-1",
        page_or_section="page 1",
        field_or_chunk_id="Glucose",
        quote_or_value="",
    )

    result = check_document_citation(bypassed, fact_index, _corpus_index())

    assert result.status is CitationStatus.EMPTY_QUOTE
    assert result.passed is False


def test_lab_citation_short_numeric_quote_still_verifies():
    # The length floor is specific to the guideline_chunk substring path
    # (where a short quote trivially matches almost anything); the lab/
    # intake path is exact equality against the raw stored quote, so a
    # short-but-real value ("7") must still verify.
    raw = Citation(
        source_type="lab_pdf",
        source_id="doc-3",
        page_or_section="page 1",
        field_or_chunk_id="pH",
        quote_or_value="7",
    )
    fact_index = _fact_index(raw)
    citation = DocumentCitation(
        source_type="lab_pdf",
        source_id="doc-3",
        page_or_section="page 1",
        field_or_chunk_id="pH",
        quote_or_value="7",
    )

    result = check_document_citation(citation, fact_index, _corpus_index())

    assert result.status is CitationStatus.VALID


def test_claim_with_only_an_empty_quote_document_citation_does_not_pass():
    claim = Claim.model_construct(
        text="Thiazides are first-line.",
        source_refs=[],
        document_citations=[_bypassed_chunk_citation("")],
    )
    corpus_index = _corpus_index(_FakeChunk(_RAW_CHUNK_ID, _RAW_CHUNK_TEXT))

    result = check_claim(claim, _index(), _fact_index(), corpus_index)

    assert result.passed is False
    assert result.citation_results[0].status is CitationStatus.EMPTY_QUOTE


def test_document_citation_schema_rejects_blank_quote_or_value():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DocumentCitation(
            source_type="guideline_chunk",
            source_id="hypertension-guideline",
            page_or_section="First-line Treatment",
            field_or_chunk_id=_RAW_CHUNK_ID,
            quote_or_value="",
        )


def test_document_citation_schema_rejects_whitespace_only_quote_or_value():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DocumentCitation(
            source_type="guideline_chunk",
            source_id="hypertension-guideline",
            page_or_section="First-line Treatment",
            field_or_chunk_id=_RAW_CHUNK_ID,
            quote_or_value="   ",
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
