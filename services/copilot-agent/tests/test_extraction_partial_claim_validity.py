"""Regression test for issue #93 (Option C): one uncitable claim in an
extraction response must not discard co-occurring valid claims.

**The bug.** ``app.schemas.verification.VerifiedAnswer.claims`` is a
``list[Claim]``, and (before this fix) ``Claim`` carried a
``@model_validator(mode="after")`` requiring >=1 citation. Pydantic validates
a list of sub-models all-or-nothing: ONE claim failing its own validator
raises ``ValidationError`` for the WHOLE ``VerifiedAnswer``, not just that
claim. ``LlamaServerClient.extract()`` (``app.llama_server_client``) retries
on ``ValidationError`` up to ``max_retries`` times, but the extraction model
runs at temperature 0 -- deterministic -- so it re-emits the exact same
uncitable claim on every retry. Retries exhaust, ``LlamaServerError``
propagates, and ``ClaimExtractor.extract_claims`` (``app.extraction``)
catches it and returns ``[]`` for the ENTIRE turn: every claim is lost,
including ones that would have had valid, re-verifiable citations, solely
because one unrelated claim in the same response was uncitable. Confirmed
live as the mechanism behind the ``statin-ck-myopathy-question``
citation_present failure.

**The fix.** ``Claim`` no longer enforces the >=1-citation rule at
construction/parse time -- ``VerifiedAnswer.model_validate`` now always
succeeds regardless of how many of its claims are individually uncitable.
The rule itself is NOT weakened: it moves to ``app.verification.check_claim``
(``ClaimCheckResult.passed`` is vacuously ``False`` for a claim with zero
citation results -- already-existing logic, previously a defensive
backstop, now the actual enforcement point) and ``app.rendering
.render_answer`` (strips any failed claim to a ``Notice``, unchanged). A
claim with zero citations still never survives into the rendered answer or
counts as verified -- it just no longer poisons its siblings.

Only the ``httpx.Client`` transport is faked (``httpx.MockTransport``) --
everything else is real production code: real ``LlamaServerClient.extract``
(including its retry loop), real ``ClaimExtractor``, real
``app.verification.check_claims``, real ``app.rendering.render_answer``,
real ``app.verdict.compute_verdict``. The mock returns the SAME payload on
every attempt, mirroring the diagnosed deterministic temperature=0 behavior
that makes retrying pointless for this failure class.
"""

from __future__ import annotations

import json

import httpx

from app.extraction import ClaimExtractor, run_verification
from app.llama_server_client import LlamaServerClient
from app.planner import PlannerResult, ToolCallTrace
from app.schemas.planner import ToolName
from app.verdict import Verdict


def _raw_encounters() -> dict[str, object]:
    return {
        "items": [
            {"encounter_id": 1, "date": "2024-01-01", "provider": "Dr. Alvarez"},
            {"encounter_id": 2, "date": "2024-02-01", "provider": "Dr. Booker"},
        ]
    }


def _planner_result(raw_result: dict[str, object]) -> PlannerResult:
    return PlannerResult(
        answer=(
            "The patient's first visit was with Dr. Alvarez on 2024-01-01. "
            "The second visit was with Dr. Booker on 2024-02-01. "
            "The clinic recommends annual follow-up."
        ),
        trace=[ToolCallTrace(tool=ToolName.GET_ENCOUNTERS, args={}, result=raw_result)],
        raw_results=[raw_result],
    )


def _mixed_validity_payload() -> dict[str, object]:
    """Two claims with valid, re-verifiable citations, plus ONE claim with
    zero citations of either shape -- the exact per-response mix the
    diagnosed mechanism turns into a total loss."""
    return {
        "claims": [
            {
                "text": "The patient's first visit was with Dr. Alvarez.",
                "source_refs": [
                    {
                        "tool_call_id": "call_0",
                        "record_id": "0",
                        "field": "provider",
                        "asserted_value": "Dr. Alvarez",
                    }
                ],
                "document_citations": [],
            },
            {
                "text": "The second visit was with Dr. Booker.",
                "source_refs": [
                    {
                        "tool_call_id": "call_0",
                        "record_id": "1",
                        "field": "provider",
                        "asserted_value": "Dr. Booker",
                    }
                ],
                "document_citations": [],
            },
            {
                # Uncitable on purpose: zero source_refs AND zero
                # document_citations -- e.g. a general recommendation with
                # no tool-result or guideline backing.
                "text": "The clinic recommends annual follow-up.",
                "source_refs": [],
                "document_citations": [],
            },
        ]
    }


def _deterministic_mock_transport_handler():
    """Returns the SAME complete, valid-JSON payload on every attempt --
    modeling the diagnosed temperature=0 determinism that makes
    ``LlamaServerClient.extract``'s retry loop pointless for this failure
    class: retrying cannot produce a different claim mix."""

    def handler(request: httpx.Request) -> httpx.Response:
        content = json.dumps(_mixed_validity_payload())
        payload = {
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 400, "completion_tokens": 200},
        }
        return httpx.Response(200, content=json.dumps(payload).encode())

    return handler


def test_one_uncitable_claim_no_longer_discards_valid_co_occurring_claims():
    """GREEN after the Option C fix: a response with 2 valid claims + 1
    uncitable claim now yields a verdict carrying the 2 valid claims'
    citations, with only the uncitable one stripped -- not a total loss.

    Before the fix, ``VerifiedAnswer.model_validate`` raised
    ``ValidationError`` for the whole payload (the uncitable claim's own
    ``Claim`` validator), every retry re-raised identically (deterministic
    model), ``LlamaServerError`` propagated after exhausting retries, and
    ``extract_claims`` caught it, returning ``[]`` -- ``verdict.verdict ==
    BLOCKED`` and ``verdict.total_claim_count == 0``, losing the two claims
    that DID have valid citations. This test fails on that old behavior and
    passes once ``Claim`` no longer raises at parse time for a zero-citation
    claim (the per-claim citation bar itself is unchanged -- it just moves to
    ``app.verification``/``app.rendering``, which already strip a failed
    claim without touching its siblings)."""
    raw_result = _raw_encounters()
    result = _planner_result(raw_result)

    client = LlamaServerClient(
        base_url="http://llama-server:8080",
        client=httpx.Client(transport=httpx.MockTransport(_deterministic_mock_transport_handler())),
        max_retries=3,
    )
    extractor = ClaimExtractor(ollama_client=client)
    verdict, rendered = run_verification(extractor, result)

    assert verdict.verdict != Verdict.BLOCKED, (
        "the two valid claims should survive verification even though a third, "
        "unrelated claim in the same response was uncitable"
    )
    assert verdict.total_claim_count == 3, "all three claims parsed -- extraction must not have failed closed"
    assert verdict.stripped_claim_count == 1, "exactly the one uncitable claim should be stripped, not the other two"

    rendered_claims = [seg for seg in rendered.segments if hasattr(seg, "source_refs")]
    assert len(rendered_claims) == 2, "the two valid claims must render as real claims, not notices"
    asserted_values = {ref.asserted_value for claim in rendered_claims for ref in claim.source_refs}
    assert asserted_values == {"Dr. Alvarez", "Dr. Booker"}

    notices = [seg for seg in rendered.segments if not hasattr(seg, "source_refs")]
    assert len(notices) == 1, "the uncitable claim must still be stripped to a notice, not silently kept as fact"
