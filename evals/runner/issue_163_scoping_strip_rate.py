"""Issue #163 measurement: what would #158's tool-call-scoping flag
(``Settings.copilot_extraction_tool_call_scoping_enabled``, ``app.tool_call
_scoping``) do to the committed eval suite if it were the default?

**What this measures, precisely.** For every case under ``evals/cases/``
(``discover_case_files``, same discovery ``issue_130_spike.py``/
``evals/test_cases.py`` use): ONE live planner draw (``Planner.run``, via
``runner.pipeline``'s own planner section -- see "Why the planner section is
duplicated" below), producing one ``PlannerResult``. Then, on that SAME
draw's ``PlannerResult`` -- not a fresh draw -- ``app.extraction
.run_verification`` is called TWICE: once with ``require_tool_call_scoping=
False`` (the OFF arm, today's shipped default from #162) and once with
``require_tool_call_scoping=True`` (the ON arm, #158's mechanism). Every
other ``run_verification`` argument (``support_judge``, ``require_answer_
grounding``, ``retrieved_chunks``, ``patient_facts``) is held IDENTICAL
between the two calls and mirrors exactly what ``runner.pipeline.run_case``
passes in production replay/live mode -- the ONLY thing that varies between
the two calls is the one flag. This is what "paired same-draw ON/OFF"
means: it isolates the scoping gate's own effect from planner-draw-to-draw
variance (the layer #154 measures), because the planner never runs twice.

**The ON arm re-runs claim extraction -- a second, real LLM call, by
design, not a bug.** ``require_tool_call_scoping=True`` makes ``run_
verification`` narrow the extractor's citable catalog to only the answer's
lexically-engaged tool calls (PREVENTION, ``app.tool_call_scoping``'s module
docstring) BEFORE calling ``ClaimExtractor.extract_claims`` -- so the ON
arm's extraction call sees a different (narrower) prompt than the OFF arm's
and can legitimately extract a DIFFERENT set of claims, not just re-check
the OFF arm's claims with a stricter gate. This is exactly what a real
``require_tool_call_scoping=True`` production turn would do; this harness
does not shortcut it. One consequence: this harness's per-claim comparison
between arms is NOT a claim-by-claim diff (the two arms' claim lists are not
guaranteed to line up 1:1) -- it is an AGGREGATE comparison (claim counts,
downgrade counts, whole-turn verdicts) per the report schema below, which is
still exactly what the owner's ship/don't-ship decision needs.

**What this does NOT measure (out of scope by the pairing design, spelled
out so a reader doesn't over-read the numbers):**

  * Planner tool-calling variance draw-to-draw -- that is #154's measured
    layer, deliberately untouched here (one draw per case, no re-rolls).
  * The ``require_answer_grounding``/``support_judge`` gates' own effects in
    isolation -- both are threaded through unchanged from whatever
    ``runner.pipeline.needs_semantic_support``/``Settings().copilot_claim_
    answer_grounding_enabled`` say for a real production/eval turn, exactly
    as ``runner.pipeline.run_case`` does; this harness varies ONLY
    ``require_tool_call_scoping``.
  * Cases whose assertions don't need verification at all (``runner.pipeline
    .needs_verification`` is ``False`` -- most of ``tool_selection``,
    ``safe_refusal``, ``injection``, ``authorization_probe``, etc.): #158's
    gate never runs for these turns in production either (it lives inside
    ``run_verification``), so they are reported with ``applicable=False``
    and no arms, not silently dropped -- "every case reported" (issue #163)
    means every case appears in the artifact, not that every case has a
    scoping verdict to report.

**Why the planner section is duplicated instead of reusing ``runner.pipeline
.run_case`` directly.** ``run_case`` calls the planner AND ``run_verification``
(exactly once, flag value from ``Settings()``) as one atomic unit -- there is
no seam to grab the ``PlannerResult`` in between and call verification a
second time on it. Rather than reshape ``pipeline.py`` (out of scope for a
measurement script, and every existing caller of ``run_case`` depends on its
current one-shot shape), ``_run_planner`` below duplicates exactly
``run_case``'s pre-verification section (the #223 cross-patient guard, the
fake registry + ``Planner.run`` call, ``apply_subject_check``, ``clarify_
unresolvable_referent``, ``apply_recency_notice``) -- same duplication
discipline ``issue_154_stability_harness.py``'s ``_agent_root_candidates``/
``_build_live_client`` already use for the same reason (a script's own
plumbing must not depend on another script's/module's surface staying
compatible). ``needs_verification``/``needs_semantic_support`` themselves
ARE imported and reused as-is from ``runner.pipeline`` -- no reason to
duplicate pure predicates.

**How the ON arm's per-claim ``TOOL_CALL_NOT_ENGAGED`` downgrades are
counted.** ``app.extraction.run_verification`` computes a ``claim_results:
list[ClaimCheckResult]`` internally but its PUBLIC return type
(``tuple[VerdictResult, RenderedAnswer]``) does not expose it -- ``VerdictResult``
is claim-count aggregates only, and ``RenderedAnswer``'s ``Notice`` segments
collapse every failure reason to one constant string ("Not found in
record.") by design (``app.rendering``'s module docstring: a per-reason
notice would leak information to the end user). Rather than duplicate
``run_verification``'s own extraction/check/scoping/verdict call sequence in
this script (a real risk of silently drifting from production behavior --
see ``app.extraction.run_verification`` itself for the canonical sequence),
``_capture_claim_results`` below monkeypatches ``app.extraction
.render_answer`` for the duration of one ``run_verification`` call: the
patched wrapper records its ``results`` argument (the exact ``claim_results``
``run_verification`` is about to render) and then delegates unmodified to
the real ``render_answer``. This changes NOTHING about what ``run_
verification`` computes or returns -- it is a pure observation seam, not a
reimplementation -- and is restored immediately after each call (``finally``),
so it never leaks across arms or cases.

**Live, not replay; single draw, no re-rolls (issue #163's "honest
measurement" requirement).** Mirrors ``issue_154_stability_harness.py``'s
live-client construction and dual-layout ``sys.path`` handling (#119/#135)
verbatim (duplicated for the same reason documented there). A case whose
planner draw itself raises is recorded with ``error`` set and NO verification
arms attempted (nothing to pair against). A case whose planner draw succeeds
but ONE verification arm raises (e.g. extraction retry exhaustion) is
recorded with that arm's ``error`` set and ``comparable=False`` -- the OTHER
arm's result (if it succeeded) is still reported, never discarded, and never
retried. Both are real per-case outcomes and are reported as such, not
silently retried or excluded. No case is drawn twice, and ``main`` makes
exactly one unconditional attempt per case -- there is no retry of any kind
(mechanical or otherwise) anywhere in this harness.

**Downstream-only tool data, same scope limit #154 documents.** Every case's
``tool_data`` is the case's own canned fixture (``runner.tool_stub
.build_fake_registry``) -- this harness never talks to a real OpenEMR chart.
It measures the scoping gate's effect on committed fixture data, not on
live-chart variance.

**Incremental, resumable artifacts.** Every case's ``CaseRecord`` is written
to ``evals/results/issue-163/draws/<case_id>.json`` immediately after that
case completes (crash-safe: a crash mid-run loses at most one in-flight
case). Re-running ``main()`` SKIPS any case whose draw file already exists
(resumable within one session -- see issue #163's "design the script to be
resumable" requirement) UNLESS ``--fresh`` is passed, which clears
``draws/`` first so a NEW session's artifact is never silently mixed with an
OLDER session's per-case draws (the hard rule: "in the final artifact every
case is from this session's single draw, no mixing with older runs").
``summarize()``/``--summarize-only`` re-aggregates an existing ``draws/``
directory into ``evals/results/issue-163/report.json`` without any live
call, same precedent as ``issue_154_stability_harness.py``.

**``draws/`` is the ONE full-detail source; ``report.json`` never duplicates
it.** ``report.json`` carries the per-category/total ``summary()`` output
plus a LIGHT per-case index (``case_id``, ``category``, ``comparable``,
``verdict_flip``, ``newly_blocked`` -- see ``_case_index_entry``) for a
human skimming the committed artifact to find which case ids to look up.
The full per-arm claim/citation detail lives ONLY in ``draws/<case_id>
.json`` -- ``report.json`` used to also embed a full copy of every
``CaseRecord`` in a ``"cases"`` array, which meant the same data existed in
two places that could silently diverge (e.g. a ``--fresh`` rerun that
regenerates ``draws/`` but not ``report.json``, or vice versa). There is now
exactly one place a reader goes for full case detail.

**Live-model runs stay OUT of CI.** Same three guarantees
``issue_154_stability_harness.py``'s docstring documents (collection-glob
dodge via this filename, live call only inside ``main()``/``if __name__ ==
"__main__"``, never imported by any ``evals/test_*.py`` module) -- not
repeated in full here; see that module's docstring for the complete
argument. This module's own PURE aggregation functions
(``arm_record_from_results``/``build_case_record``/``summarize``) ARE unit
tested, in CI, with no live model -- see
``evals/runner/tests/test_issue_163_scoping_strip_rate.py``.

**Run environment -- MUST run inside ``development-easy-agent-1`` with a
FRESH ``app/``+``evals/``+``corpus/`` copied in.** Identical gotcha and copy
recipe to ``issue_154_stability_harness.py``'s docstring ("Usage" section) --
not repeated here verbatim; see that module. The one operational difference:
this script also needs ``services/copilot-agent/corpus`` (``app.retrieval
.CORPUS_DIR``) for any ``citation_present`` case's guideline evidence, exactly
as #154's harness already documents.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_EVALS_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _EVALS_ROOT.parent
_MONOREPO_AGENT_ROOT = _REPO_ROOT / "services" / "copilot-agent"


def _agent_root_candidates(repo_root: Path, monorepo_agent_root: Path) -> list[Path]:
    """Same dual-layout resolution as ``issue_154_stability_harness.py`` /
    ``record.py`` (#119/#135) -- duplicated rather than imported so this
    script's ``sys.path`` setup (which must run before ANY ``app.*``/
    ``runner.*`` import) does not itself depend on an import that needs the
    path already fixed up."""
    candidates = [monorepo_agent_root, repo_root]
    return sorted(candidates, key=lambda root: not (root / "app" / "__init__.py").is_file())


for _root in reversed(_agent_root_candidates(_REPO_ROOT, _MONOREPO_AGENT_ROOT) + [_EVALS_ROOT]):
    _root_str = str(_root)
    if _root_str not in sys.path:
        sys.path.insert(0, _root_str)

import app.extraction as _extraction_module  # noqa: E402 -- monkeypatch target, see _capture_claim_results
from app.config import Settings  # noqa: E402
from app.correlation import configure_logging  # noqa: E402
from app.extraction import (  # noqa: E402
    ClaimExtractor,
    apply_recency_notice,
    apply_subject_check,
    clarify_unresolvable_referent,
    cross_patient_refusal_result,
    detect_foreign_patient_reference,
    run_verification,
)
from app.llama_server_client import LlamaServerClient  # noqa: E402
from app.ollama_client import OllamaClient  # noqa: E402
from app.openemr_client import OpenEmrClient  # noqa: E402
from app.planner import Planner, PlannerResult  # noqa: E402
from app.schemas.ingestion import Citation  # noqa: E402
from app.schemas.reranking import RerankedChunk  # noqa: E402
from app.tool_call_scoping import engaged_call_ids  # noqa: E402
from app.verdict import Verdict, VerdictResult  # noqa: E402
from app.verification import CitationCheckResult, CitationStatus, ClaimCheckResult  # noqa: E402

# Importing app.tool_call_scoping (above) succeeding at all IS the "resolved
# the live app package, not the baked image" marker the issue's run
# environment section asks for -- app.tool_call_scoping (#158) does not
# exist in any pre-#162 baked image. Printed explicitly in main() too, so a
# human watching the run log sees it without reading this comment.
_LIVE_APP_MARKER = "app.tool_call_scoping imported OK (live tree, not baked image)"

import httpx  # noqa: E402

from runner.loader import discover_case_files, load_case  # noqa: E402
from runner.ollama_replay import OllamaLike, RecordedCall, RecordingOllamaClient, save_recording  # noqa: E402
from runner.pipeline import _EVAL_FIXED_NOW, _EVAL_TOKEN, needs_semantic_support, needs_verification  # noqa: E402
from runner.schema import EvalCase  # noqa: E402
from runner.tool_stub import build_fake_registry  # noqa: E402

_CASES_DIR = _EVALS_ROOT / "cases"
_REGRESSIONS_DIR = _EVALS_ROOT / "regressions"
_RESULTS_DIR = _EVALS_ROOT / "results" / "issue-163"
_DRAWS_DIR = _RESULTS_DIR / "draws"


# --- pure report-schema types (unit tested, no live model) -----------------


@dataclass(frozen=True)
class ClaimRecord:
    """One claim's pass/fail + its citations' ``CitationStatus`` values, as
    plain strings (no claim text, no cited values -- non-PHI counts only,
    same discipline ``app.extraction.run_verification``'s own structured log
    line already uses: "counts only, never claim text or patient data")."""

    passed: bool
    citation_statuses: list[str]


@dataclass(frozen=True)
class ArmRecord:
    """One verification arm's (OFF or ON) result for one case: the whole-
    turn verdict, the claim-count aggregates ``VerdictResult`` already
    carries, the per-claim detail (``ClaimRecord``, see above), and how many
    citations were downgraded to ``TOOL_CALL_NOT_ENGAGED`` specifically --
    zero for an OFF arm by construction (that status is only ever assigned
    inside ``apply_tool_call_scoping``, which only runs when the flag is
    on). ``error`` is set (all other fields at their zero value) when this
    arm's ``run_verification`` call itself raised -- an infrastructure
    failure (e.g. extraction retry exhaustion), reported as such rather than
    retried or silently dropped."""

    verdict: str
    total_claim_count: int
    stripped_claim_count: int
    claims: list[ClaimRecord]
    downgraded_count: int
    error: str | None = None


def arm_record_from_results(verdict_result: VerdictResult, claim_results: list[ClaimCheckResult]) -> ArmRecord:
    """Build one arm's report row from the SAME objects ``run_verification``
    itself computed for that call -- ``verdict_result`` is its direct return
    value; ``claim_results`` is captured via ``_capture_claim_results``
    (see module docstring). Pure, no I/O -- unit tested directly."""
    claims = [
        ClaimRecord(passed=result.passed, citation_statuses=[c.status.value for c in result.citation_results])
        for result in claim_results
    ]
    downgraded_count = sum(
        1
        for result in claim_results
        for citation in result.citation_results
        if isinstance(citation, CitationCheckResult) and citation.status is CitationStatus.TOOL_CALL_NOT_ENGAGED
    )
    return ArmRecord(
        verdict=verdict_result.verdict.value,
        total_claim_count=verdict_result.total_claim_count,
        stripped_claim_count=verdict_result.stripped_claim_count,
        claims=claims,
        downgraded_count=downgraded_count,
        error=None,
    )


def error_arm_record(exc: BaseException) -> ArmRecord:
    """The ``ArmRecord`` for a verification call that raised -- reported,
    never silently dropped or retried (issue #163's "honest measurement"
    requirement)."""
    return ArmRecord(
        verdict="", total_claim_count=0, stripped_claim_count=0, claims=[], downgraded_count=0, error=repr(exc)
    )


@dataclass(frozen=True)
class CaseRecord:
    """One case's full report row: identity, the engaged/total tool-call-id
    sets (computed once, independent of which arm -- ``engaged_call_ids`` is
    a pure function of ``(raw_results, engagement_answer)``, unaffected by
    the flag itself), both arms (``None`` when ``applicable`` is ``False``),
    and the derived comparison fields.

    ``comparable`` is ``True`` iff both ``off`` and ``on`` are present AND
    neither carries an ``error`` -- ``verdict_flip``/``newly_blocked`` are
    only ever computed (and only ever meaningful) when this is ``True``;
    they default to ``False`` otherwise, which is a "not determined", never
    a claimed "no flip happened" -- read ``comparable`` first."""

    case_id: str
    category: str
    applicable: bool
    engaged_call_ids: list[str]
    total_call_ids: list[str]
    off: ArmRecord | None
    on: ArmRecord | None
    comparable: bool
    verdict_flip: bool
    flip_detail: str | None
    newly_blocked: bool
    error: str | None = None  # set only when the planner draw itself failed


def build_case_record(
    *,
    case_id: str,
    category: str,
    applicable: bool,
    engaged_call_ids: list[str],
    total_call_ids: list[str],
    off: ArmRecord | None,
    on: ArmRecord | None,
    draw_error: str | None = None,
) -> CaseRecord:
    """Fold one case's two arms (or ``None``s, for a not-applicable or
    draw-failed case) into its report row. Pure, no I/O -- unit tested
    directly."""
    comparable = off is not None and on is not None and off.error is None and on.error is None
    verdict_flip = False
    flip_detail = None
    newly_blocked = False
    if comparable:
        assert off is not None and on is not None  # narrows for mypy/pyright; comparable already guarantees this
        verdict_flip = off.verdict != on.verdict
        if verdict_flip:
            flip_detail = f"{off.verdict}->{on.verdict}"
        newly_blocked = off.verdict != Verdict.BLOCKED.value and on.verdict == Verdict.BLOCKED.value
    return CaseRecord(
        case_id=case_id,
        category=category,
        applicable=applicable,
        engaged_call_ids=list(engaged_call_ids),
        total_call_ids=list(total_call_ids),
        off=off,
        on=on,
        comparable=comparable,
        verdict_flip=verdict_flip,
        flip_detail=flip_detail,
        newly_blocked=newly_blocked,
        error=draw_error,
    )


def summarize(records: list[CaseRecord]) -> dict[str, Any]:
    """Per-category + total report table: cases seen, how many were
    comparable (both arms succeeded), total OFF-arm claim count (the
    denominator for a strip rate -- "claims total flag-OFF-valid" per issue
    #163), total ON-arm ``TOOL_CALL_NOT_ENGAGED`` downgrades, verdict flips,
    and newly-blocked turns (whole-turn ``blocked`` ONLY under the ON arm --
    the dominant flag-ON regression risk per ``app.tool_call_scoping``'s
    module docstring, "Known failure directions" #4). Pure, no I/O -- unit
    tested directly."""

    def _empty_bucket() -> dict[str, int]:
        return {
            "cases": 0,
            "comparable_cases": 0,
            "claims_total_off": 0,
            "claims_downgraded": 0,
            "verdict_flips": 0,
            "newly_blocked": 0,
            "errors": 0,
        }

    by_category: dict[str, dict[str, int]] = {}
    total = _empty_bucket()
    for record in records:
        bucket = by_category.setdefault(record.category, _empty_bucket())
        for target in (bucket, total):
            target["cases"] += 1
            if record.error is not None or (record.off is not None and record.off.error is not None) or (
                record.on is not None and record.on.error is not None
            ):
                target["errors"] += 1
            if record.comparable:
                target["comparable_cases"] += 1
                assert record.off is not None and record.on is not None
                target["claims_total_off"] += record.off.total_claim_count
                target["claims_downgraded"] += record.on.downgraded_count
                if record.verdict_flip:
                    target["verdict_flips"] += 1
                if record.newly_blocked:
                    target["newly_blocked"] += 1
    return {"by_category": by_category, "total": total}


# --- live orchestration (never imported/collected by pytest) ---------------


def _offline_openemr_client() -> OpenEmrClient:
    """Identical tripwire to ``runner.pipeline._offline_openemr_client`` --
    duplicated (not imported, it is a private, underscore-prefixed name) for
    the same reason the planner section below is duplicated."""

    def _handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"eval harness attempted a real OpenEMR call: {request.url}")

    return OpenEmrClient(
        base_url="https://eval-harness.invalid",
        client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )


def _run_planner(
    case: EvalCase,
    ollama_client: OllamaLike,
    *,
    retrieved_chunks: list[RerankedChunk],
    patient_facts: list[Citation],
) -> PlannerResult:
    """Exactly ``runner.pipeline.run_case``'s pre-verification section --
    see module docstring, "Why the planner section is duplicated". Issue
    #163 gate-1 review: ``retrieved_chunks``/``patient_facts`` are computed
    ONCE by ``run_one_case`` (mirroring ``pipeline.run_case``'s own
    documented one-source, two-consumers design for the SAME fixtures) and
    passed in here, rather than each of this function and
    ``_run_verification_arm`` re-deriving them from ``case`` independently."""
    if detect_foreign_patient_reference(
        case.question, case.patient_id, case.patient_name, roster_provider=lambda: case.patient_roster or []
    ):
        return apply_recency_notice(cross_patient_refusal_result(), now=_EVAL_FIXED_NOW)

    registry = build_fake_registry(case.tool_data, case.patient_id)
    planner = Planner(
        ollama_client=ollama_client,  # type: ignore[arg-type]
        openemr_client=_offline_openemr_client(),
        token=_EVAL_TOKEN,
        patient_id=case.patient_id,
        registry=registry,
    )
    guideline_excerpts = [chunk.text for chunk in retrieved_chunks]
    planner_kwargs: dict[str, list[Citation]] = {}
    if patient_facts:
        planner_kwargs["document_facts"] = patient_facts
    planner_result = planner.run(case.question, guideline_excerpts, **planner_kwargs)
    planner_result = apply_subject_check(planner_result, question=case.question, patient_id=case.patient_id)
    planner_result = clarify_unresolvable_referent(planner_result, question=case.question, has_prior_turns=False)
    planner_result = apply_recency_notice(planner_result, now=_EVAL_FIXED_NOW)
    return planner_result


@contextmanager
def _capture_claim_results() -> Iterator[list[list[ClaimCheckResult]]]:
    """Monkeypatch ``app.extraction.render_answer`` to record its input
    argument for the duration of one (or more) ``run_verification`` calls --
    see module docstring, "How the ON arm's per-claim downgrades are
    counted." Delegates to the real function unmodified; restores the
    original on exit regardless of outcome. Returns a list that accumulates
    one ``claim_results`` list per ``run_verification`` call made inside the
    ``with`` block, in order."""
    captured: list[list[ClaimCheckResult]] = []
    original = _extraction_module.render_answer

    def _wrapper(results: list[ClaimCheckResult]) -> Any:
        captured.append(results)
        return original(results)

    _extraction_module.render_answer = _wrapper  # type: ignore[assignment]
    try:
        yield captured
    finally:
        _extraction_module.render_answer = original  # type: ignore[assignment]


def _run_verification_arm(
    case: EvalCase,
    planner_result: PlannerResult,
    ollama_client: OllamaLike,
    *,
    retrieved_chunks: list[RerankedChunk],
    patient_facts: list[Citation],
    require_tool_call_scoping: bool,
) -> ArmRecord:
    """One arm's ``run_verification`` call, with claim-level detail
    captured. Exceptions are caught here (not propagated) so one failing arm
    never blocks the other arm or the rest of the run. ``retrieved_chunks``/
    ``patient_facts`` are the SAME objects ``run_one_case`` already computed
    once for ``_run_planner`` -- see that function's docstring."""
    extractor = ClaimExtractor(ollama_client=ollama_client)  # type: ignore[arg-type]
    support_judge = ollama_client if needs_semantic_support(case) else None
    try:
        with _capture_claim_results() as captured:
            verdict_result, _rendered = run_verification(
                extractor,
                planner_result,
                retrieved_chunks=retrieved_chunks,
                patient_facts=patient_facts,
                support_judge=support_judge,
                require_answer_grounding=Settings().copilot_claim_answer_grounding_enabled,
                require_tool_call_scoping=require_tool_call_scoping,
            )
    except Exception as exc:  # noqa: BLE001 -- harness run: record failure, keep going
        return error_arm_record(exc)
    claim_results = captured[0] if captured else []
    return arm_record_from_results(verdict_result, claim_results)


def run_one_case(case: EvalCase, ollama_client: OllamaLike) -> tuple[CaseRecord, list[RecordedCall]]:
    """Run ``case`` once, live: one planner draw, then (if applicable) both
    verification arms paired on that SAME draw. ``ollama_client`` is a
    ``RecordingOllamaClient`` shared across the planner AND both arms, so
    one case's full call sequence (planner + OFF extraction (+judge) + ON
    extraction (+judge)) is captured together, mirroring #154's per-draw
    recording precedent."""
    # Issue #163 gate-1 review: computed ONCE here, mirroring
    # ``pipeline.run_case``'s documented one-source-two-consumers design for
    # these SAME fixtures -- threaded into ``_run_planner`` and BOTH
    # ``_run_verification_arm`` calls below rather than each recomputing its
    # own copy from ``case``.
    retrieved_chunks = [fixture.to_reranked_chunk() for fixture in case.retrieved_chunks]
    patient_facts: list[Citation] = [fixture.to_citation() for fixture in case.patient_facts]

    recorder = RecordingOllamaClient(ollama_client)  # type: ignore[arg-type]
    try:
        planner_result = _run_planner(
            case, recorder, retrieved_chunks=retrieved_chunks, patient_facts=patient_facts  # type: ignore[arg-type]
        )
    except Exception as exc:  # noqa: BLE001 -- harness run: record failure, keep going
        record = build_case_record(
            case_id=case.id,
            category=case.category,
            applicable=needs_verification(case),
            engaged_call_ids=[],
            total_call_ids=[],
            off=None,
            on=None,
            draw_error=repr(exc),
        )
        return record, recorder.calls

    total_call_ids = [f"call_{i}" for i in range(len(planner_result.raw_results))]
    engagement_answer = (
        planner_result.answer_pre_notice if planner_result.answer_pre_notice is not None else planner_result.answer
    )
    engaged = sorted(engaged_call_ids(planner_result.raw_results, engagement_answer))

    if not needs_verification(case):
        record = build_case_record(
            case_id=case.id,
            category=case.category,
            applicable=False,
            engaged_call_ids=engaged,
            total_call_ids=total_call_ids,
            off=None,
            on=None,
        )
        return record, recorder.calls

    off = _run_verification_arm(
        case,
        planner_result,
        recorder,  # type: ignore[arg-type]
        retrieved_chunks=retrieved_chunks,
        patient_facts=patient_facts,
        require_tool_call_scoping=False,
    )
    on = _run_verification_arm(
        case,
        planner_result,
        recorder,  # type: ignore[arg-type]
        retrieved_chunks=retrieved_chunks,
        patient_facts=patient_facts,
        require_tool_call_scoping=True,
    )
    record = build_case_record(
        case_id=case.id,
        category=case.category,
        applicable=True,
        engaged_call_ids=engaged,
        total_call_ids=total_call_ids,
        off=off,
        on=on,
    )
    return record, recorder.calls


# --- incremental save / aggregate -------------------------------------------


def _draw_path(case_id: str) -> Path:
    return _DRAWS_DIR / f"{case_id}.json"


def _calls_path(case_id: str) -> Path:
    return _DRAWS_DIR / f"{case_id}.calls.json"


def save_case(record: CaseRecord, calls: list[RecordedCall]) -> tuple[Path, Path]:
    _DRAWS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = _draw_path(record.case_id)
    summary_path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    calls_path = _calls_path(record.case_id)
    save_recording(calls_path, calls)
    return summary_path, calls_path


def _record_from_payload(payload: dict[str, Any]) -> CaseRecord:
    def _arm(data: dict[str, Any] | None) -> ArmRecord | None:
        if data is None:
            return None
        return ArmRecord(
            verdict=data["verdict"],
            total_claim_count=data["total_claim_count"],
            stripped_claim_count=data["stripped_claim_count"],
            claims=[ClaimRecord(**c) for c in data["claims"]],
            downgraded_count=data["downgraded_count"],
            error=data["error"],
        )

    return CaseRecord(
        case_id=payload["case_id"],
        category=payload["category"],
        applicable=payload["applicable"],
        engaged_call_ids=payload["engaged_call_ids"],
        total_call_ids=payload["total_call_ids"],
        off=_arm(payload["off"]),
        on=_arm(payload["on"]),
        comparable=payload["comparable"],
        verdict_flip=payload["verdict_flip"],
        flip_detail=payload["flip_detail"],
        newly_blocked=payload["newly_blocked"],
        error=payload["error"],
    )


def load_draws(draws_dir: Path) -> list[CaseRecord]:
    records = []
    for path in sorted(draws_dir.glob("*.json")):
        if path.name.endswith(".calls.json"):
            continue
        records.append(_record_from_payload(json.loads(path.read_text(encoding="utf-8"))))
    return records


def _case_index_entry(record: CaseRecord) -> dict[str, Any]:
    """The LIGHT per-case row ``report.json`` carries -- just enough for a
    human skimming the committed artifact to find which case id to open in
    ``draws/`` for full detail. See module docstring, "``draws/`` is the ONE
    full-detail source" -- this is deliberately NOT ``asdict(record)``."""
    return {
        "case_id": record.case_id,
        "category": record.category,
        "comparable": record.comparable,
        "verdict_flip": record.verdict_flip,
        "newly_blocked": record.newly_blocked,
    }


def build_report(records: list[CaseRecord]) -> dict[str, Any]:
    """The full ``report.json`` payload: generated-at timestamp, case count,
    the per-category/total ``summarize()`` output, and the light per-case
    index (``_case_index_entry``) -- never a full copy of every
    ``CaseRecord`` (that lives solely in ``draws/<case_id>.json``, loaded via
    ``load_draws``)."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(records),
        "cases": [_case_index_entry(r) for r in records],
        "summary": summarize(records),
    }


def save_report(report: dict[str, Any]) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _RESULTS_DIR / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def print_table(summary: dict[str, Any]) -> None:
    """Human-facing per-category + total table. Includes an ``errors``
    column (issue #163 gate-1 review) -- ``summarize()`` already computes
    this count per bucket; previously it was silently absent from this
    printout even though it is real, non-zero-worthy signal (a case whose
    draw or a verification arm raised)."""
    header = (
        f"{'category':<20}{'cases':>7}{'claims':>9}{'downgraded':>12}{'flips':>8}{'newly_blocked':>15}{'errors':>8}"
    )
    print(header)
    print("-" * len(header))
    for category, bucket in sorted(summary["by_category"].items()):
        print(
            f"{category:<20}{bucket['cases']:>7}{bucket['claims_total_off']:>9}"
            f"{bucket['claims_downgraded']:>12}{bucket['verdict_flips']:>8}{bucket['newly_blocked']:>15}"
            f"{bucket['errors']:>8}"
        )
    print("-" * len(header))
    total = summary["total"]
    print(
        f"{'TOTAL':<20}{total['cases']:>7}{total['claims_total_off']:>9}"
        f"{total['claims_downgraded']:>12}{total['verdict_flips']:>8}{total['newly_blocked']:>15}"
        f"{total['errors']:>8}"
    )


def _build_live_client(engine: str, ollama_base_url: str) -> OllamaLike:
    """Same construction as ``issue_154_stability_harness.py``'s
    ``_build_live_client`` -- duplicated for the same reason."""
    if engine == "llama_server":
        settings = Settings(llama_server_api_timeout_seconds=180.0)
        return LlamaServerClient.from_settings(settings)  # type: ignore[return-value]
    settings = Settings(ollama_base_url=ollama_base_url, ollama_api_timeout_seconds=180.0)
    return OllamaClient.from_settings(settings)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--fresh", action="store_true", help="clear evals/results/issue-163/draws/ before running (new session)"
    )
    parser.add_argument("--summarize-only", action="store_true", help="skip live runs; re-aggregate draws/ only")
    parser.add_argument("--case-id", action="append", default=None, help="restrict to one or more case ids (repeatable)")
    args = parser.parse_args()

    configure_logging()
    print(_LIVE_APP_MARKER)

    if not args.summarize_only:
        if args.fresh and _DRAWS_DIR.exists():
            for path in _DRAWS_DIR.iterdir():
                path.unlink()

        ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        engine = os.environ.get("RECORD_ENGINE", "ollama")

        cases = [load_case(p) for p in discover_case_files(_CASES_DIR, _REGRESSIONS_DIR)]
        if args.case_id:
            wanted = set(args.case_id)
            cases = [c for c in cases if c.id in wanted]

        for case in cases:
            draw_path = _draw_path(case.id)
            if draw_path.exists():
                print(f"[issue-163] {case.id}: SKIP (already drawn, {draw_path})")
                continue
            client = _build_live_client(engine, ollama_base_url)
            record, calls = run_one_case(case, client)
            summary_path, calls_path = save_case(record, calls)
            if record.error:
                status = f"DRAW-ERROR {record.error}"
            elif not record.applicable:
                status = "not applicable (no verification needed)"
            elif not record.comparable:
                off_err = record.off.error if record.off else None
                on_err = record.on.error if record.on else None
                status = f"ARM-ERROR off={off_err!r} on={on_err!r}"
            else:
                status = (
                    f"off={record.off.verdict} on={record.on.verdict} "
                    f"downgraded={record.on.downgraded_count} flip={record.verdict_flip}"
                )
            print(f"[issue-163] {case.id} ({case.category}): {status} -> {summary_path}, {calls_path}")

    records = load_draws(_DRAWS_DIR)
    report = build_report(records)
    save_report(report)
    print_table(report["summary"])


if __name__ == "__main__":
    main()
