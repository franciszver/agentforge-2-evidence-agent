"""Regression test for issue #85: live citation dropped between retrieval and
answer assembly.

**Root cause, confirmed live** (see the issue #85 PR description for the full
live-run trace, captured via ``evals/runner/pipeline.run_case`` against the
real ``llama-server`` container, 6/6 fresh draws of ``bp-stage2-question``):
qwen3-8b, when asked to cite a retrieved guideline chunk, deterministically
puts the citation in ``source_refs`` (the tool-result citation shape) instead
of ``document_citations`` (the guideline-citation shape ``_GUIDELINE_
INSTRUCTIONS`` actually asks for) -- but does so in a specific, predictable
way: it reconstructs the chunk's OWN ``<doc_id>#<section-slug>`` id across
two ``SourceRef`` fields, ``tool_call_id="<doc_id>"`` +
``record_id="<section-slug>"``, with ``field="text"`` and the verbatim quote
as ``asserted_value``. E.g. for chunk id
``"blood-pressure-categories#categories"``:

    {"tool_call_id": "blood-pressure-categories", "record_id": "categories",
     "field": "text", "asserted_value": "Stage 2 hypertension: ..."}

``check_source_ref`` correctly fails this ``UNKNOWN_TOOL_CALL`` (no real tool
call has that id) -- but ``check_claim`` ANDs across every citation on a
claim, so this ONE malformed ref drags down the WHOLE claim, including two
perfectly valid chart-data citations bundled in the same claim ("this reading
falls into the category of elevated blood pressure" cites the BP values AND
the category text together). The guideline quote itself is genuine and
verbatim -- it was simply never given a chance to verify as a
``document_citations`` entry.

The fix: recognize this specific, safely-identifiable shape at claim
extraction time (before it ever reaches ``check_claims``) and reclassify it
as a real ``DocumentCitation`` instead of dropping it. Narrowly scoped to
avoid ever touching a genuine tool citation: real tool ids are always
``call_<i>`` (``CacheIndex.from_raw_results``), which can never collide with
a corpus ``doc_id``, and the reconstructed id must exactly match a chunk this
turn actually retrieved -- never invented.
"""

from __future__ import annotations

from app.extraction import ClaimExtractor
from app.schemas.common import SourceRef
from app.schemas.planner import ToolName
from app.schemas.reranking import RerankedChunk
from app.schemas.verification import Claim, VerifiedAnswer
from app.verification import CacheIndex, CorpusChunkIndex, check_claims


class _FakeExtractOllama:
    """Scripted extraction client returning the EXACT malformed shape
    observed live (see module docstring)."""

    def __init__(self, result: VerifiedAnswer) -> None:
        self._result = result

    def extract(self, prompt_or_messages, schema, *, options=None):
        return self._result


def _bp_chunk() -> RerankedChunk:
    return RerankedChunk(
        chunk_id="blood-pressure-categories#categories",
        doc_id="blood-pressure-categories",
        title="Blood Pressure Categories and Thresholds",
        section="Categories",
        text=(
            "- Normal: systolic below 120 mmHg AND diastolic below 80 mmHg.\n"
            "- Stage 2 hypertension: systolic 140 mmHg or higher OR diastolic 90 mmHg or higher."
        ),
        scores={"hybrid": 0.9},
        rerank_score=0.9,
    )


def _live_observed_verified_answer() -> VerifiedAnswer:
    """The exact ``VerifiedAnswer`` shape captured live 6/6 times for
    ``bp-stage2-question`` -- see module docstring."""
    return VerifiedAnswer(
        claims=[
            Claim(
                text="The patient's last blood pressure reading was 148/94 mmHg.",
                source_refs=[
                    SourceRef(tool_call_id="call_0", record_id="0", field="blood_pressure_systolic", asserted_value="148.0"),
                    SourceRef(tool_call_id="call_0", record_id="1", field="blood_pressure_diastolic", asserted_value="94.0"),
                ],
            ),
            Claim(
                text="This reading falls into the category of elevated blood pressure.",
                source_refs=[
                    SourceRef(tool_call_id="call_0", record_id="0", field="blood_pressure_systolic", asserted_value="148.0"),
                    SourceRef(tool_call_id="call_0", record_id="1", field="blood_pressure_diastolic", asserted_value="94.0"),
                    SourceRef(
                        tool_call_id="blood-pressure-categories",
                        record_id="categories",
                        field="text",
                        asserted_value="Stage 2 hypertension: systolic 140 mmHg or higher OR diastolic 90 mmHg or higher.",
                    ),
                ],
            ),
        ]
    )


def test_misrouted_guideline_source_ref_is_reclassified_as_a_document_citation():
    """RED before the fix / GREEN after: a source_ref shaped like the
    reconstructed chunk id (doc_id + section-slug, matching a chunk this
    turn actually retrieved) must be pulled OUT of source_refs and turned
    into a real ``document_citations`` entry -- not silently left to fail
    ``UNKNOWN_TOOL_CALL`` and poison the whole claim."""
    chunk = _bp_chunk()
    extractor = ClaimExtractor(ollama_client=_FakeExtractOllama(_live_observed_verified_answer()))

    claims = extractor.extract_claims(
        answer="irrelevant for this test",
        tools=[ToolName.GET_VITALS],
        raw_results=[{"items": [{"blood_pressure_systolic": 148.0}, {"blood_pressure_diastolic": 94.0}]}],
        retrieved_chunks=[chunk],
    )

    category_claim = next(c for c in claims if "category" in c.text)
    assert not any(
        ref.tool_call_id == "blood-pressure-categories" for ref in category_claim.source_refs
    ), "the malformed guideline source_ref must not remain in source_refs"
    guideline_citations = [dc for dc in category_claim.document_citations if dc.source_type == "guideline_chunk"]
    assert len(guideline_citations) == 1
    assert guideline_citations[0].source_id == "blood-pressure-categories"
    assert guideline_citations[0].field_or_chunk_id == "blood-pressure-categories#categories"
    assert "Stage 2 hypertension" in guideline_citations[0].quote_or_value

    # And it must actually verify end to end against the real checker, using
    # the SAME retrieved chunk as the corpus index -- not merely be present.
    raw_results = [{"items": [{"blood_pressure_systolic": 148.0}, {"blood_pressure_diastolic": 94.0}]}]
    index = CacheIndex.from_raw_results(raw_results)
    corpus_index = CorpusChunkIndex.from_chunks([chunk])
    results = check_claims(claims, index, corpus_index=corpus_index)
    category_result = next(r for r in results if "category" in r.claim.text)
    assert category_result.passed, (
        f"expected the category claim to verify once its guideline citation is properly "
        f"shaped; statuses={[r.status.value for r in category_result.citation_results]}"
    )


def _lipid_chunk() -> RerankedChunk:
    """The exact chunk shape from issue #125's live repro
    (``lipid-panel-ldl-question``, see ``evals/cases/citation_present/
    lipid-panel-ldl-question.yaml``)."""
    return RerankedChunk(
        chunk_id="lipid-panel-reference#general-reference-categories",
        doc_id="lipid-panel-reference",
        title="Lipid Panel Reference Ranges and Follow-Up",
        section="General Reference Categories",
        text=(
            "LDL cholesterol: optimal below 100 mg/dL; near-optimal 100-129 mg/dL; "
            "borderline-high 130-159 mg/dL; high 160-189 mg/dL; very high 190 mg/dL or above."
        ),
        scores={"hybrid": 0.9},
        rerank_score=0.9,
    )


def _live_observed_doubled_doc_id_verified_answer() -> VerifiedAnswer:
    """The exact ``VerifiedAnswer`` shape captured live 10/10 times for
    ``lipid-panel-ldl-question`` (issue #125) -- the doc_id is reused for
    BOTH ``tool_call_id`` and ``record_id``, unlike #85's
    ``<doc_id>`` + ``<section-slug>`` shape."""
    return VerifiedAnswer(
        claims=[
            Claim(
                text="Her LDL cholesterol was 172 mg/dL.",
                source_refs=[
                    SourceRef(tool_call_id="call_0", record_id="0", field="value", asserted_value="172"),
                ],
            ),
            Claim(
                text="This is considered high per the lipid panel reference guideline.",
                source_refs=[
                    SourceRef(tool_call_id="call_0", record_id="0", field="value", asserted_value="172"),
                    SourceRef(
                        tool_call_id="lipid-panel-reference",
                        record_id="lipid-panel-reference",
                        field="text",
                        asserted_value=(
                            "LDL cholesterol: optimal below 100 mg/dL; near-optimal 100-129 mg/dL; "
                            "borderline-high 130-159 mg/dL; high 160-189 mg/dL; very high 190 mg/dL or above."
                        ),
                    ),
                ],
            ),
        ]
    )


def test_doubled_doc_id_misrouted_guideline_source_ref_is_reclassified_as_a_document_citation():
    """RED before the fix / GREEN after (issue #125): a source_ref where
    ``tool_call_id == record_id == doc_id`` (no section slug at all -- the
    NEW variant #85's original coercion does not recognize, since its exact
    ``f"{tool_call_id}#{record_id}"`` reconstruction produces
    ``"lipid-panel-reference#lipid-panel-reference"``, which never matches
    the real chunk id ``"lipid-panel-reference#general-reference-
    categories"``) must still be pulled OUT of source_refs and turned into a
    real ``document_citations`` entry, provided it identifies exactly one
    retrieved chunk's doc_id unambiguously."""
    chunk = _lipid_chunk()
    extractor = ClaimExtractor(ollama_client=_FakeExtractOllama(_live_observed_doubled_doc_id_verified_answer()))

    claims = extractor.extract_claims(
        answer="irrelevant for this test",
        tools=[ToolName.GET_RECENT_LABS],
        raw_results=[{"items": [{"test_name": "LDL Cholesterol", "value": "172", "unit": "mg/dL"}]}],
        retrieved_chunks=[chunk],
    )

    high_claim = next(c for c in claims if "considered high" in c.text)
    assert not any(
        ref.tool_call_id == "lipid-panel-reference" for ref in high_claim.source_refs
    ), "the malformed doubled-doc_id guideline source_ref must not remain in source_refs"
    guideline_citations = [dc for dc in high_claim.document_citations if dc.source_type == "guideline_chunk"]
    assert len(guideline_citations) == 1
    assert guideline_citations[0].source_id == "lipid-panel-reference"
    assert guideline_citations[0].field_or_chunk_id == "lipid-panel-reference#general-reference-categories"
    assert "LDL cholesterol" in guideline_citations[0].quote_or_value

    # And it must actually verify end to end against the real checker, using
    # the SAME retrieved chunk as the corpus index -- not merely be present.
    raw_results = [{"items": [{"test_name": "LDL Cholesterol", "value": "172", "unit": "mg/dL"}]}]
    index = CacheIndex.from_raw_results(raw_results)
    corpus_index = CorpusChunkIndex.from_chunks([chunk])
    results = check_claims(claims, index, corpus_index=corpus_index)
    high_result = next(r for r in results if "considered high" in r.claim.text)
    assert high_result.passed, (
        f"expected the 'considered high' claim to verify once its guideline citation is properly "
        f"shaped; statuses={[r.status.value for r in high_result.citation_results]}"
    )


def test_ambiguous_doubled_doc_id_with_multiple_matching_chunks_is_never_coerced():
    """Safety guard (issue #125): if MORE THAN ONE retrieved chunk shares the
    same doc_id, the doubled-doc_id shape must NOT be coerced -- there is no
    way to know which chunk/section the model actually meant, so guessing
    would risk fabricating a citation. The ref is left to fail
    ``UNKNOWN_TOOL_CALL`` exactly as before, same fail-closed posture as an
    unrecoverable hallucination."""
    chunk_a = RerankedChunk(
        chunk_id="lipid-panel-reference#section-a",
        doc_id="lipid-panel-reference",
        title="Lipid Panel Reference Ranges and Follow-Up",
        section="Section A",
        text="Section A text.",
        scores={"hybrid": 0.9},
        rerank_score=0.9,
    )
    chunk_b = RerankedChunk(
        chunk_id="lipid-panel-reference#section-b",
        doc_id="lipid-panel-reference",
        title="Lipid Panel Reference Ranges and Follow-Up",
        section="Section B",
        text="Section B text.",
        scores={"hybrid": 0.9},
        rerank_score=0.9,
    )
    verified = VerifiedAnswer(
        claims=[
            Claim(
                text="ambiguous claim",
                source_refs=[
                    SourceRef(
                        tool_call_id="lipid-panel-reference",
                        record_id="lipid-panel-reference",
                        field="text",
                        asserted_value="Section A text.",
                    ),
                ],
            )
        ]
    )
    extractor = ClaimExtractor(ollama_client=_FakeExtractOllama(verified))
    claims = extractor.extract_claims(
        answer="irrelevant",
        tools=[ToolName.GET_RECENT_LABS],
        raw_results=[{"items": [{"test_name": "LDL Cholesterol", "value": "172"}]}],
        retrieved_chunks=[chunk_a, chunk_b],
    )
    claim = claims[0]
    assert claim.document_citations == []
    assert len(claim.source_refs) == 1
    assert claim.source_refs[0].tool_call_id == "lipid-panel-reference"


def test_a_real_tool_call_id_is_never_mistaken_for_a_guideline_chunk():
    """Safety guard: the coercion must never touch a genuine ``call_<i>``
    source_ref -- even adversarially, when a retrieved chunk's reconstructed
    id EXACTLY collides with the real ref's ``tool_call_id#record_id``
    (``"call_0#0"`` here). Without the ``_REAL_TOOL_CALL_ID_RE`` guard, this
    exact scenario would be silently (mis)coerced into a document_citation
    and the genuine chart-data source_ref would vanish -- this test fails
    loudly if that guard is ever removed or narrowed."""
    colliding_chunk = RerankedChunk(
        chunk_id="call_0#0",
        doc_id="call_0",
        title="Adversarial collision fixture",
        section="0",
        text="This chunk's id deliberately collides with a real tool_call_id/record_id pair.",
        scores={"hybrid": 0.9},
        rerank_score=0.9,
    )
    verified = VerifiedAnswer(
        claims=[
            Claim(
                text="valid tool claim",
                source_refs=[
                    SourceRef(tool_call_id="call_0", record_id="0", field="blood_pressure_systolic", asserted_value="148.0"),
                ],
            )
        ]
    )
    extractor = ClaimExtractor(ollama_client=_FakeExtractOllama(verified))
    claims = extractor.extract_claims(
        answer="irrelevant",
        tools=[ToolName.GET_VITALS],
        raw_results=[{"items": [{"blood_pressure_systolic": 148.0}]}],
        retrieved_chunks=[colliding_chunk],
    )
    claim = claims[0]
    assert claim.document_citations == []
    assert len(claim.source_refs) == 1
    assert claim.source_refs[0].tool_call_id == "call_0"
