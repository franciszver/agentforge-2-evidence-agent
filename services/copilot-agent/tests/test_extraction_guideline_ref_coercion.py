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


def test_a_real_tool_call_id_is_never_mistaken_for_a_guideline_chunk():
    """Safety guard: the coercion must never touch a genuine ``call_<i>``
    source_ref, even if a retrieved chunk's doc_id/section happened to
    collide with something call-shaped (defensive; real tool ids are always
    exactly ``call_<i>``, which can never equal a corpus ``doc_id``)."""
    chunk = _bp_chunk()
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
        retrieved_chunks=[chunk],
    )
    claim = claims[0]
    assert claim.document_citations == []
    assert len(claim.source_refs) == 1
    assert claim.source_refs[0].tool_call_id == "call_0"
