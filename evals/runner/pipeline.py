"""Runs one eval case through the REAL agent pipeline (P4.7):
planner -> (optionally) claim extraction -> verification -- deterministically,
given the case's canned tool data (``runner.tool_stub``) and an
``OllamaLike`` client (the live model in record mode, a recorded replay in
the default/CI path -- ``runner.ollama_replay``).

**Verification is lazy.** ``needs_verification`` inspects the case's
assertions: the extraction + verification stage (an extra claim-extraction
model call) only runs for cases that actually assert on ``verdict``. A plain
tool-selection case's recording therefore only has to carry the planner's own
turns, not an unused claim-extraction call -- judgment call documented in
``runner.schema``'s module docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.extraction import ClaimExtractor, run_verification
from app.openemr_client import OpenEmrClient
from app.planner import Planner, PlannerResult
from app.rendering import RenderedAnswer
from app.verdict import VerdictResult

from runner.ollama_replay import OllamaLike
from runner.schema import EvalCase, VerdictAssertion
from runner.tool_stub import build_fake_registry

_EVAL_TOKEN = "eval-harness-token"  # noqa: S105 -- not a credential, a fixed placeholder bearer value


def _offline_openemr_client() -> OpenEmrClient:
    """An ``OpenEmrClient`` that raises if it is ever actually called.

    Every tool dispatch in the eval harness is stubbed via the fake registry
    (see ``build_fake_registry``) -- the planner never reaches the real
    ``OpenEmrClient.get_rest``/``get_fhir`` methods. This stub exists purely
    as a loud tripwire: if a future change ever bypassed the fake registry,
    a real HTTP attempt here fails immediately instead of silently reaching
    the network (breaking the harness's offline guarantee).
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"eval harness attempted a real OpenEMR call: {request.url}")

    return OpenEmrClient(
        base_url="https://eval-harness.invalid",
        client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )


@dataclass(frozen=True)
class CaseResult:
    """One case's pipeline output. ``verdict_result``/``rendered`` are
    ``None`` when the case didn't need verification (see
    ``needs_verification``)."""

    planner_result: PlannerResult
    verdict_result: VerdictResult | None
    rendered: RenderedAnswer | None


def needs_verification(case: EvalCase) -> bool:
    """Whether this case's assertions require the extraction/verification
    stage (i.e. it has a ``verdict`` assertion)."""
    return any(isinstance(assertion, VerdictAssertion) for assertion in case.assertions)


def run_case(case: EvalCase, ollama_client: OllamaLike) -> CaseResult:
    """Run ``case`` end to end: the real ``Planner`` loop, then (if needed)
    the real claim-extraction + verification stack. ``ollama_client`` is
    whatever satisfies ``OllamaLike`` -- the live model, or a replay."""
    registry = build_fake_registry(case.tool_data, case.patient_id)
    planner = Planner(
        ollama_client=ollama_client,  # type: ignore[arg-type]
        openemr_client=_offline_openemr_client(),
        token=_EVAL_TOKEN,
        patient_id=case.patient_id,
        registry=registry,
    )
    planner_result = planner.run(case.question)

    if not needs_verification(case):
        return CaseResult(planner_result=planner_result, verdict_result=None, rendered=None)

    extractor = ClaimExtractor(ollama_client=ollama_client)  # type: ignore[arg-type]
    verdict_result, rendered = run_verification(extractor, planner_result)
    return CaseResult(planner_result=planner_result, verdict_result=verdict_result, rendered=rendered)
