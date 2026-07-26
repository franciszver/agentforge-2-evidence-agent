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

**None of recency notices (#153), the cross-patient subject-check (#194), or
the unresolvable-referent guard (#225) are lazy**, unlike verification above
-- every case runs ``app.extraction.apply_subject_check`` then
``app.extraction.clarify_unresolvable_referent`` then ``app.extraction
.apply_recency_notice`` unconditionally, right after the planner turn (see
their docstrings / ``app.verification``'s "Recency notices" section for why
none must wait on the lazy, LLM-gated verification stage). The order is:
subject-check, then the referent guard, then the recency notice -- the
subject-check runs first so it only ever scans the model's own prose, never
text a later step appends (a stale record's literal date, or the referent
guard's own clarifying text, could otherwise coincidentally collide with a
foreign patient number); the referent guard runs before the recency notice
for the same reason (it only inspects ``question``, so its exact position
relative to the recency notice doesn't affect correctness, but grouping the
post-answer guards together keeps the sequence readable). ``_EVAL_FIXED_NOW``
is the suite's frozen reference instant, chosen close to the recordings'
authored date (mid-2026) so every OTHER category's freshly-dated fixtures
stay "fresh" while ``stale_data``'s 2014 fixtures are unambiguously stale.

**Even earlier than the subject-check: the PRE-dispatch cross-patient guard
(#223, extended by #224).** ``app.extraction.detect_foreign_patient_reference``
is checked BEFORE the fake registry / ``Planner`` are even constructed --
unlike #194's subject-check, which runs after the planner has already
dispatched tools and can only rewrite the answer text, this hardens the
actual dispatch: a detected foreign-patient reference short-circuits straight
to ``app.extraction.cross_patient_refusal_result()`` (empty trace, no tool
ever run, no model ever called), which is what lets ``must_refuse``/``no_phi``
actually pass -- both require the forbidden tool to NEVER dispatch. The
case's optional ``patient_name`` (#224 name-binding) is passed through so the
guard's named signals are exercised the same as the live ``app.chat`` path --
absent, the guard falls back to numeric-only detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import httpx

from app.config import Settings
from app.extraction import (
    ClaimExtractor,
    apply_recency_notice,
    apply_subject_check,
    clarify_unresolvable_referent,
    cross_patient_refusal_result,
    detect_foreign_patient_reference,
    run_verification,
)
from app.openemr_client import OpenEmrClient
from app.planner import Planner, PlannerResult
from app.rendering import RenderedAnswer
from app.schemas.ingestion import Citation
from app.tools.patient_summary import RosterEntry
from app.verdict import VerdictResult
from runner.ollama_replay import OllamaLike
from runner.schema import (
    EvalCase,
    GuidelineCitationPresentAssertion,
    NoDocumentCitationFromPatientFactAssertion,
    VerdictAssertion,
)
from runner.tool_stub import build_fake_registry

_EVAL_TOKEN = "eval-harness-token"  # noqa: S105 -- not a credential, a fixed placeholder bearer value

# Frozen "now" for the whole offline eval suite (#153) -- see module
# docstring, "Recency notices are NOT lazy".
_EVAL_FIXED_NOW = datetime(2026, 7, 15)

# Issue #174: the LIVE roster (app.tools.patient_summary.get_patient_roster)
# is now patient-agnostic and carries (pid, name) pairs so the bound
# patient's own entry can be excluded at COMPARISON time, by pid, rather
# than at fetch time. `EvalCase.patient_roster` (runner.schema) stays a
# plain, hand-authored `list[str]` -- there is no real pid to carry for a
# fixture entry, so every name here is adapted to a `RosterEntry` with THIS
# ONE sentinel pid. Gate 3 (Opus) re-review MINOR (#174): that makes
# `app.extraction._matches_roster`'s pid-based exclusion a PERMANENT no-op
# for every eval case -- `_ROSTER_FIXTURE_SENTINEL_PID` can never equal a
# real (positive) `case.patient_id`, so nothing on `case.patient_roster` is
# ever excluded by this sentinel, regardless of what the case author
# intended. Fixtures MUST NOT list the bound patient in `patient_roster` --
# unlike the live roster (which now legitimately contains the bound
# patient's own entry, excluded only at comparison time), an eval fixture
# that does so relies entirely on `case.patient_name` also being set (so
# `_is_foreign_switch_to_name`'s earlier `_same_named_patient` check
# short-circuits before the roster is ever consulted) -- a case exercising
# name-binding-UNAVAILABLE behavior with the bound patient also present in
# `patient_roster` would wrongly refuse a "switch to <own name>" turn.
_ROSTER_FIXTURE_SENTINEL_PID = -1


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


def _has_assertion(case: EvalCase, *assertion_types: type) -> bool:
    """Whether any of ``case.assertions`` is an instance of one of
    ``assertion_types`` -- the shared check ``needs_verification`` and
    ``needs_semantic_support`` both build on, for their own assertion-type
    subsets."""
    return any(isinstance(assertion, assertion_types) for assertion in case.assertions)


def needs_verification(case: EvalCase) -> bool:
    """Whether this case's assertions require the extraction/verification
    stage (i.e. it has a ``verdict``, P3G.1 ``guideline_citation_present``, or
    ``no_document_citation_from_patient_fact`` assertion -- all need the same
    ``ClaimExtractor``/``check_claims`` pass)."""
    return _has_assertion(
        case, VerdictAssertion, GuidelineCitationPresentAssertion, NoDocumentCitationFromPatientFactAssertion
    )


# Issue #81 (owner-revised methodology, P3.9c): the gate is judged against
# the STABLE recordings already on main rather than a fresh answer re-draw
# (a fresh re-draw of all 12 cases turned out to be an unlucky sample --
# planner variance, unrelated to the gate -- see docs/MODEL_AND_HARDWARE_
# SELECTION.md's variance caveat). ``statin-ck-myopathy-question`` is the one
# case that remains excluded: its stable (pre-#81) recording was never
# re-recordable under the gate (#58's original claim-extraction decode
# limitation, still true), so it has no committed extraction to judge from.
# ``dual-antiplatelet-question`` IS one of the six stable provenance-passing
# cases re-judged this cycle -- its stable recording has a real extraction to
# append a judge call onto -- so it is no longer excluded here.
_CANNOT_RERECORD_UNDER_GATE = frozenset({"statin-ck-myopathy-question"})


def needs_semantic_support(case: EvalCase) -> bool:
    """Whether this case's verification pass must also run the issue #47
    semantic-support gate (``app.semantic_support``, on by default in
    production since issue #81 -- ``Settings.copilot_semantic_support_enabled``).

    Scoped to cases with a ``guideline_citation_present`` assertion: that is
    exactly the ``citation_present`` category's own assertion vocabulary (see
    ``runner.schema``'s module docstring), the ONE category re-recorded with
    the gate on (issue #81) -- every other category's committed recording
    predates the gate and has no ``SemanticSupportJudgement`` call to replay,
    so this must not widen beyond that one assertion type without
    re-recording the newly-included cases too. ``_CANNOT_RERECORD_UNDER_GATE``
    carves out the one case whose recording could not be updated (see above)."""
    if case.id in _CANNOT_RERECORD_UNDER_GATE:
        return False
    return _has_assertion(case, GuidelineCitationPresentAssertion)


def run_case(case: EvalCase, ollama_client: OllamaLike) -> CaseResult:
    """Run ``case`` end to end: the real ``Planner`` loop, then (if needed)
    the real claim-extraction + verification stack. ``ollama_client`` is
    whatever satisfies ``OllamaLike`` -- the live model, or a replay."""
    # #223: PRE-dispatch cross-patient guard, checked BEFORE the fake
    # registry / Planner are even constructed -- see module docstring. A
    # detected foreign-patient reference never reaches a tool dispatch or a
    # model call. ``case.patient_roster`` (#237) feeds the roster-based
    # "switch to <Name>" signal the same way ``case.patient_name`` feeds the
    # "patient <Name>" signal; already fully in-memory (no I/O), so no
    # laziness concern here unlike the live app.chat wiring.
    # #105: the case's canned guideline-corpus evidence (if any) is threaded
    # into the planner's OWN answer-composition call below, mirroring the
    # live P3.9 `/chat` wiring's #105 fix (`app.chat._stream_chat` now
    # retrieves BEFORE calling the planner, not after) -- computed here,
    # before the planner runs, rather than down in the verification branch
    # below (its pre-#105 position), so `planner.run` can see it. Cheap:
    # pure in-memory fixture parsing, no I/O, so computing it even for a
    # cross-patient-refusal case (where it goes unused) costs nothing.
    retrieved_chunks = [fixture.to_reranked_chunk() for fixture in case.retrieved_chunks]
    guideline_excerpts = [chunk.text for chunk in retrieved_chunks]
    # Issue #70, mirroring #86's `app.chat` wiring: this patient's canned
    # ingested document-fact citations, computed here (before the planner
    # runs, alongside `retrieved_chunks`/`guideline_excerpts` above) so the
    # SAME list feeds both consumers a real turn feeds from one fetch --
    # `Planner.run`'s `document_facts` kwarg below, and `run_verification`'s
    # `patient_facts` kwarg further down. See `runner.schema.PatientFactFixture`.
    patient_facts: list[Citation] = [fixture.to_citation() for fixture in case.patient_facts]

    if detect_foreign_patient_reference(
        case.question,
        case.patient_id,
        case.patient_name,
        roster_provider=lambda: [
            RosterEntry(pid=_ROSTER_FIXTURE_SENTINEL_PID, name=name) for name in (case.patient_roster or [])
        ],
    ):
        planner_result = cross_patient_refusal_result()
    else:
        registry = build_fake_registry(case.tool_data, case.patient_id)
        planner = Planner(
            ollama_client=ollama_client,  # type: ignore[arg-type]
            openemr_client=_offline_openemr_client(),
            token=_EVAL_TOKEN,
            patient_id=case.patient_id,
            registry=registry,
        )
        # Threaded ONLY when non-empty -- see `patient_facts`'s definition
        # above for the one-source, two-consumers rationale.
        planner_kwargs: dict[str, list[Citation]] = {}
        if patient_facts:
            planner_kwargs["document_facts"] = patient_facts
        planner_result = planner.run(case.question, guideline_excerpts, **planner_kwargs)
        # apply_subject_check runs BEFORE apply_recency_notice -- see
        # app.chat's wiring comment for why (it must only ever scan the
        # model's own prose, never text a later deterministic step appends).
        planner_result = apply_subject_check(planner_result, question=case.question, patient_id=case.patient_id)
        # #225: clarify_unresolvable_referent, inside this same else branch
        # (never reached when the #223 guard above fired) -- see its
        # docstring for why it must not run on a cross-patient refusal. The
        # eval harness's cases are single-turn by construction (module
        # docstring), so ``has_prior_turns`` is always False here.
        planner_result = clarify_unresolvable_referent(
            planner_result, question=case.question, has_prior_turns=False
        )
    planner_result = apply_recency_notice(planner_result, now=_EVAL_FIXED_NOW)

    if not needs_verification(case):
        return CaseResult(planner_result=planner_result, verdict_result=None, rendered=None)

    extractor = ClaimExtractor(ollama_client=ollama_client)  # type: ignore[arg-type]
    # P3G.1 / #105: `retrieved_chunks` was already computed ABOVE (before the
    # planner ran) -- reused here for `run_verification`'s citation-attachment
    # pass, exactly as before; only WHEN it was computed changed, not what it
    # feeds. See `runner.schema.RetrievedChunkFixture`.
    # Issue #81: the SAME client already driving the planner/extractor also
    # satisfies `SemanticSupportJudgeLike` (both are duck-typed against one
    # `.extract` call -- see `app.semantic_support.SemanticSupportJudgeLike`),
    # so no separate client construction is needed here the way `app.chat`
    # needs one (`get_support_judge_provider`) -- see `needs_semantic_support`
    # for which cases actually exercise the judge call.
    support_judge = ollama_client if needs_semantic_support(case) else None
    # Issue #153: threads `Settings.copilot_claim_answer_grounding_enabled`
    # straight through (unlike `support_judge` above, this gate is a pure,
    # deterministic, no-LLM check -- `apply_answer_grounding` needs nothing
    # from a recording, so there is no re-recording gate to scope this to a
    # category the way `needs_semantic_support` does). Reads the environment
    # fresh so the eval suite can be re-run with
    # `COPILOT_CLAIM_ANSWER_GROUNDING_ENABLED=true` to measure the gate's
    # per-category effect on the existing recordings, without changing this
    # module's own default (`Settings()`'s default is `False`, matching the
    # implicit `False` this call site had before this change).
    # `patient_facts`: same one-source, two-consumers list, reused here.
    # Issue #158: threads `Settings.copilot_extraction_tool_call_scoping_
    # enabled` through exactly parallel to `require_answer_grounding` above --
    # also a pure, deterministic, no-LLM gate, so no re-recording gate is
    # needed here either. `COPILOT_EXTRACTION_TOOL_CALL_SCOPING_ENABLED=true`
    # re-runs the eval suite with the gate on for the same kind of per-
    # category measurement.
    verdict_result, rendered = run_verification(
        extractor,
        planner_result,
        retrieved_chunks=retrieved_chunks,
        patient_facts=patient_facts,
        support_judge=support_judge,
        require_answer_grounding=Settings().copilot_claim_answer_grounding_enabled,
        require_tool_call_scoping=Settings().copilot_extraction_tool_call_scoping_enabled,
    )
    return CaseResult(planner_result=planner_result, verdict_result=verdict_result, rendered=rendered)
