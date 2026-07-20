"""Regression test for the claim-extraction token-truncation bug (P5.3,
issue #85).

``services/copilot-agent/app/llama_server_client.py`` used to apply ONE
hardcoded ``max_tokens: 1536`` cap to every ``LlamaServerClient`` call --
chat AND extract. The extraction prompt (``app.extraction.ClaimExtractor``)
makes the model echo each retrieved guideline/fact chunk's full text
verbatim into ``quote_or_value`` fields. When real retrieval returns
multiple full corpus sections -- unlike the eval harness's tiny
hand-authored fixtures (``evals/runner/pipeline.py`` builds
``retrieved_chunks`` straight from case-YAML, never real retrieval; see its
module docstring for the gap this closes) -- the required JSON output can
exceed the cap, llama.cpp truncates mid-string (``finish_reason: "length"``),
JSON parsing/schema validation fails, retries exhaust,
``ClaimExtractor.extract_claims`` catches the resulting ``LlamaServerError``
and returns ``[]``, and ``run_verification`` yields ``blocked`` with zero
citations -- even though retrieval succeeded and had the right chunk.

Only the ``httpx.Client`` transport is faked (``httpx.MockTransport``) --
everything else is real production code: ``app.retrieval.parse_corpus()``
against the real ``corpus/blood-pressure-categories.md``, real
``app.extraction.ClaimExtractor``/``run_verification``, real
``app.verification.check_document_citation``. This is the layer choice that
closes exactly the gap the eval harness's synthetic fixtures leave open.

The mock inspects the request's ``max_tokens`` and returns a
truncated-but-otherwise-valid-JSON-prefix, ``finish_reason: "length"``
response when it's still the OLD small cap -- reproducing the bug -- and a
complete, valid response once the cap is large enough, mirroring what
llama.cpp actually emits in each case.
"""

from __future__ import annotations

import json

import httpx

from app.extraction import ClaimExtractor, run_verification
from app.llama_server_client import LlamaServerClient
from app.planner import PlannerResult, ToolCallTrace
from app.retrieval import CORPUS_DIR, parse_corpus
from app.schemas.planner import ToolName
from app.schemas.reranking import RerankedChunk
from app.verdict import Verdict

# The OLD, too-small cap this bug shipped with -- reproduced here (not
# imported from app.llama_server_client) so this test keeps failing the same
# way even if that module's internal constant name/value changes for an
# unrelated reason; this test's job is to prove the FIX (a larger extract
# cap) resolves the failure, not to assert the exact old number.
_OLD_TOO_SMALL_CAP = 1536


def _real_bp_chunks() -> list[RerankedChunk]:
    """Top-3 real chunks from the real corpus (matches production's
    ``_EVIDENCE_RETRIEVAL_TOP_K``), reranked-shaped with a plausible score --
    NOT a synthetic stand-in for corpus text."""
    chunks = [c for c in parse_corpus(CORPUS_DIR) if c.doc_id == "blood-pressure-categories"]
    assert len(chunks) >= 3, "fixture assumption: real corpus has >=3 BP-category sections"
    return [
        RerankedChunk(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            title=c.title,
            section=c.section,
            text=c.text,
            scores={"hybrid": 0.9},
            rerank_score=0.9,
        )
        for c in chunks[:3]
    ]


def _planner_result() -> PlannerResult:
    """A realistically-sized tool-result payload, shaped like
    ``get_encounters`` -- a real chart-data record, not a synthetic
    one-field stand-in."""
    raw_result = {
        "items": [
            {
                "encounter_id": 3,
                "date": "2014-02-01",
                "blood_pressure_systolic": 148,
                "blood_pressure_diastolic": 94,
                "provider": "Dr. Smith",
                "reason": "Follow-up",
            }
        ]
    }
    return PlannerResult(
        answer=(
            "The patient's last blood pressure reading was 148 mmHg systolic and "
            "94 mmHg diastolic, recorded on 2014-02-01. This falls into Stage 2 "
            "hypertension."
        ),
        trace=[ToolCallTrace(tool=ToolName.GET_ENCOUNTERS, args={}, result=raw_result)],
        raw_results=[raw_result],
    )


def _valid_extraction_payload(chunks: list[RerankedChunk]) -> dict[str, object]:
    """A complete, schema-valid ``VerifiedAnswer`` payload citing both the
    chart data and the retrieved guideline chunk -- what llama.cpp emits once
    it has enough ``max_tokens`` budget to finish."""
    guideline_chunk = next(c for c in chunks if c.section.lower() == "summary" or "categor" in c.text.lower())
    return {
        "claims": [
            {
                "text": "The patient's last blood pressure reading was 148 mmHg systolic.",
                "source_refs": [
                    {"tool_call_id": "call_0", "record_id": "0", "field": "blood_pressure_systolic", "asserted_value": "148"}
                ],
                "document_citations": [],
            },
            {
                "text": "This falls into Stage 2 hypertension.",
                "source_refs": [],
                "document_citations": [
                    {
                        "source_type": "guideline_chunk",
                        "source_id": guideline_chunk.doc_id,
                        "page_or_section": guideline_chunk.section,
                        "field_or_chunk_id": guideline_chunk.chunk_id,
                        "quote_or_value": guideline_chunk.text,
                    }
                ],
            },
        ]
    }


def _mock_transport_handler(chunks: list[RerankedChunk]):
    """Inspects the request's ``max_tokens``: still the old small cap ->
    truncated JSON + ``finish_reason: "length"`` (reproducing the bug);
    otherwise -> a complete valid response, built from a real retrieved
    chunk's real text."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        max_tokens = body["max_tokens"]
        if max_tokens <= _OLD_TOO_SMALL_CAP:
            full_json = json.dumps(_valid_extraction_payload(chunks))
            # Truncate mid-string at the OLD cap's rough character budget --
            # ~3.5 chars/token is a reasonable approximation for this
            # payload's mix of JSON punctuation and prose/quoted-guideline
            # text -- leaving a syntactically-invalid JSON prefix, exactly
            # what llama.cpp emits when max_tokens is hit mid-generation.
            cutoff = min(len(full_json) - 40, int(max_tokens * 3.5))
            truncated = full_json[:cutoff]
            content = truncated
            finish_reason = "length"
        else:
            content = json.dumps(_valid_extraction_payload(chunks))
            finish_reason = "stop"
        payload = {
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 500, "completion_tokens": max_tokens if finish_reason == "length" else 300},
        }
        return httpx.Response(200, content=json.dumps(payload).encode())

    return handler


def test_extraction_truncates_and_fails_closed_under_the_old_small_cap():
    """RED before the fix / regression guard after: forcing the OLD
    ``max_tokens`` cap onto ``extract()`` (via ``options`` override -- the
    same seam production code uses) reproduces the truncation failure end to
    end: extraction exhausts retries, ``extract_claims`` fails closed to
    ``[]``, and ``run_verification`` yields ``blocked`` with NO
    ``guideline_chunk`` citation, despite ``retrieved_chunks`` being
    non-empty and containing the right chunk."""
    chunks = _real_bp_chunks()
    result = _planner_result()

    client = LlamaServerClient(
        base_url="http://llama-server:8080",
        client=httpx.Client(transport=httpx.MockTransport(_mock_transport_handler(chunks))),
        max_retries=2,
    )
    # Force the OLD, too-small cap via the same ``options`` override seam
    # ``_build_body`` already supports -- proves the failure is caused BY
    # the cap (not by anything else in this fixture), independent of
    # whatever ``_EXTRACT_MAX_TOKENS`` is set to after the fix.
    original_extract = client.extract

    def extract_with_old_cap(prompt_or_messages, schema, **kwargs):
        kwargs.setdefault("options", {})
        kwargs["options"]["max_tokens"] = _OLD_TOO_SMALL_CAP
        return original_extract(prompt_or_messages, schema, **kwargs)

    client.extract = extract_with_old_cap  # type: ignore[method-assign]

    extractor = ClaimExtractor(ollama_client=client)
    verdict, rendered = run_verification(extractor, result, retrieved_chunks=chunks)

    assert verdict.verdict == Verdict.BLOCKED
    assert verdict.total_claim_count == 0
    assert rendered.segments == []
    all_doc_citations = [
        dc for seg in rendered.segments for dc in getattr(seg, "document_citations", [])
    ]
    assert not any(dc.source_type == "guideline_chunk" for dc in all_doc_citations)


def test_extraction_succeeds_under_the_fixed_larger_extract_cap():
    """GREEN after the fix: extract() now sends its own larger max_tokens
    (``_EXTRACT_MAX_TOKENS``, not the old 1536 chat cap) by default -- no
    override needed. The same realistic payload now completes without
    truncation, and the resulting verdict carries a verified
    ``guideline_chunk`` citation matching the retrieved chunk's id."""
    chunks = _real_bp_chunks()
    result = _planner_result()

    client = LlamaServerClient(
        base_url="http://llama-server:8080",
        client=httpx.Client(transport=httpx.MockTransport(_mock_transport_handler(chunks))),
        max_retries=2,
    )
    extractor = ClaimExtractor(ollama_client=client)
    verdict, rendered = run_verification(extractor, result, retrieved_chunks=chunks)

    assert verdict.verdict != Verdict.BLOCKED
    guideline_citations = [
        dc
        for seg in rendered.segments
        for dc in getattr(seg, "document_citations", [])
        if dc.source_type == "guideline_chunk"
    ]
    assert guideline_citations, "expected a verified guideline_chunk citation once extraction stops truncating"
    cited_chunk_ids = {dc.field_or_chunk_id for dc in guideline_citations}
    assert cited_chunk_ids & {c.chunk_id for c in chunks}
