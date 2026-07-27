"""Issue #170 measurement: the re-measurement of #130's shadow-judge spike
under the established-facts-context fix (issues #111/#128, adapted for
SourceRef judging by ``app.source_ref_relevance``).

**Adapted from ``evals/runner/issue_130_spike.py``, not written fresh.**
#170 is explicitly #130's own pre-registered reopen path -- "apply the
established-facts-context fix mirroring #111/#128, then re-measure to zero
false-rejects" -- so the comparison is only valid if the REST of the
protocol is held constant: same 12 ``citation_present`` cases, same >=8
draws/case, same live (non-replay) pipeline run, same shadow/log-only
posture (zero verdict changes -- this script never calls
``apply_source_ref_relevance``, only ``app.source_ref_relevance``'s judge
function directly), same reconstructed #123 positive control. Only the ONE
variable under test changes: the judge call now goes through
``app.source_ref_relevance.judge_source_ref_relevance_full`` (established-
facts context, self-exclusion) instead of #130's context-free standalone
prompt. ``evals/runner/issue_163_scoping_strip_rate.py`` was considered and
rejected as the base to adapt: it measures a structurally different
mechanism (tool-CALL engagement strip rate, paired same-draw ON/OFF against
a DIFFERENT case population) for a DIFFERENT gate (#158/#163), not the
SourceRef-only shadow-judge shape #130/#170 share.

**Established-facts context, from the rendered output.** ``app.rendering
.RenderedClaim`` only ever exists for a claim that has ALREADY fully
``passed`` (``app.rendering.render_answer``), so every ``RenderedClaim``
segment in one draw's rendered answer is, by construction, an "already
established" claim -- exactly the population
``app.source_ref_relevance._established_facts_for_source_ref_claim`` draws
sibling context from. This script reconstructs that same sibling-context
set at the RENDERED-SEGMENT level (segment index used for self-exclusion,
mirroring the production module's ``other is claim_result`` check) rather
than importing the module-private ``ClaimCheckResult``-level helper, since
the live pipeline (``runner.pipeline.run_case``) only returns the rendered
answer at this call site -- see ``app.source_ref_relevance``'s module
docstring for the underlying rule this reproduces.

**What this measures.** Exactly #130's Option D shape: for every claim in
the ``citation_present`` category whose surviving citations are ALL ordinary
``SourceRef``s (zero ``DocumentCitation``s), the NEW established-facts-aware
judge is asked whether the claim's cited fields/values are RELEVANT support
for the claim's prose. The verdict is recorded, never applied -- shadow
only, exactly like #130's own spike.

**MANDATORY exposure reporting (learned from #163/#169's zero-exposure
false-clearance mistakes).** ``summarize()`` reports, per case AND overall:
how many claims were ELIGIBLE (SourceRef-only, from
``evals/runner/census_source_ref_claims.py``'s population definition), how
many were actually JUDGED, and how many false-rejects resulted.
``eligible_claim_count`` is computed in a SEPARATE structural pass over
``result.rendered.segments``, entirely BEFORE the judge loop runs (see
``run_one_draw``) -- gate-3 review (issue #170 MAJOR 3) caught an earlier
version that incremented eligible-count and appended a judge record in the
SAME loop iteration, making ``eligible == judged`` a tautology that could
never fail regardless of whether the judge call ran; the two are now
independently computed and CAN diverge (e.g. if the judge loop is ever
changed to skip or fail a subset). A case with zero eligible claims across
every draw is reported as an explicit ``INCONCLUSIVE: zero exposure`` in
the per-case summary, not silently folded into a "0 false rejects" that
could misread as clearance.

**Live, not replay.** Mirrors ``evals/runner/issue_130_spike.py``'s live
client construction and dual-layout ``sys.path`` handling verbatim.

**Incremental artifacts.** ``evals/results/issue-170/draws/*.json``,
summarized into ``evals/results/issue-170/summary.json`` -- same shape as
issue #130's own artifacts, for direct side-by-side comparison.

**Deliberate deviation from #163/#169's counts-and-enums-only artifact
convention.** Per-draw files here intentionally retain full claim text,
``field: value`` fact pairs, and the judge's rationale prose (``reason``),
not just pass/fail counts. Two reasons, both required: (1) every claim/fact
in this measurement is synthetic eval-fixture content authored for this
suite (the 12 ``citation_present`` cases' own scripted data, plus the
hardcoded ``_POSITIVE_CONTROL_CLAIM``/``_POSITIVE_CONTROL_FACTS``) -- never
real patient data, so this is not the PHI-shaped exposure #163/#169's
convention exists to prevent; and (2) a judge measurement is not auditable
without the reasoning -- stripping the rationale would leave a bare
true/false with no way to check WHY the judge decided what it decided,
making a false-reject (or a suspiciously clean zero) unfalsifiable.
``summarize()`` records this deviation explicitly in
``evals/results/issue-170/summary.json`` (the ``artifact_content_note``
key) rather than leaving it implicit.

Usage (from repo root, live model reachable):

    RECORD_ENGINE=llama_server python evals/runner/issue_170_source_ref_relevance_spike.py --draws 8
    python evals/runner/issue_170_source_ref_relevance_spike.py --summarize-only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_EVALS_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _EVALS_ROOT.parent
_MONOREPO_AGENT_ROOT = _REPO_ROOT / "services" / "copilot-agent"


def _agent_root_candidates(repo_root: Path, monorepo_agent_root: Path) -> list[Path]:
    """Same dual-layout resolution as ``evals/runner/record.py``/
    ``issue_130_spike.py`` (#119/#135) -- duplicated rather than imported so
    this script's ``sys.path`` setup (which must run before ANY
    ``app.*``/``runner.*`` import) does not itself depend on an import that
    needs the path already fixed up."""
    candidates = [monorepo_agent_root, repo_root]
    return sorted(candidates, key=lambda root: not (root / "app" / "__init__.py").is_file())


for _root in reversed(_agent_root_candidates(_REPO_ROOT, _MONOREPO_AGENT_ROOT) + [_EVALS_ROOT]):
    _root_str = str(_root)
    if _root_str not in sys.path:
        sys.path.insert(0, _root_str)

import datetime  # noqa: E402

from app.config import Settings  # noqa: E402
from app.llama_server_client import LlamaServerClient  # noqa: E402
from app.rendering import RenderedClaim  # noqa: E402
from app.schemas.common import SourceRef  # noqa: E402
from app.schemas.ingestion import DocumentCitation  # noqa: E402
from app.schemas.reranking import RerankedChunk  # noqa: E402
from app.schemas.tools import AppointmentItem, AppointmentsOutput, AppointmentStatus  # noqa: E402
from app.schemas.verification import Claim  # noqa: E402
from app.source_ref_relevance import (  # noqa: E402
    SemanticSupportJudgeLike,
    SemanticSupportJudgement,
    SupportVerdict,
    _established_facts_for_source_ref_claim,
    _is_source_ref_only_claim,
    _source_ref_facts,
    judge_source_ref_relevance_full,
)
from app.verification import CacheIndex, check_claims  # noqa: E402

from runner.loader import discover_case_files, load_case  # noqa: E402
from runner.pipeline import run_case  # noqa: E402
from runner.schema import EvalCase  # noqa: E402

_CASES_DIR = _EVALS_ROOT / "cases"
_REGRESSIONS_DIR = _EVALS_ROOT / "regressions"
_RESULTS_DIR = _EVALS_ROOT / "results" / "issue-170"
_DRAWS_DIR = _RESULTS_DIR / "draws"

_CATEGORY = "citation_present"
_DEFAULT_DRAWS = 8


def would_downgrade(judgement: SemanticSupportJudgement) -> bool:
    """Fail-closed, mirroring ``app.source_ref_relevance.judge_source_ref_
    relevance``: only an explicit ``SUPPORTED`` verdict counts as "would NOT
    downgrade"."""
    return judgement.verdict is not SupportVerdict.SUPPORTED


def _source_ref_facts_for_segment(segment: RenderedClaim) -> list[str]:
    return [f"{ref.field}: {ref.asserted_value}" for ref in segment.source_refs if ref.asserted_value is not None]


def _is_source_ref_only_exposure_claim(segment: RenderedClaim) -> bool:
    """A passed claim (``RenderedClaim`` only ever exists for passed claims)
    whose surviving citations are ALL ``SourceRef``s: zero
    ``document_citation_count`` and at least one ``SourceRef``. Exactly the
    census's exposure population (see module docstring)."""
    return segment.document_citation_count == 0 and bool(segment.source_refs)


def _established_facts_for_segment(target_index: int, segments: list[RenderedClaim]) -> list[str]:
    """Sibling-claim established facts for the segment at ``target_index``,
    reproducing ``app.source_ref_relevance._established_facts_for_source_ref
    _claim``'s rule at the rendered-segment level (module docstring,
    "Established-facts context"). Self-exclusion: the target's own index is
    always skipped -- its own facts are the primary SOURCE FACTS being
    judged, never re-added as context. Every OTHER SourceRef-only exposure
    segment in the SAME draw already satisfies "SourceRef-only AND already
    passed" (``RenderedClaim`` only exists for passed claims), so no
    additional passed/document-citation filtering is needed here beyond
    ``_is_source_ref_only_exposure_claim`` itself."""
    facts: dict[str, None] = {}
    for other_index, other in enumerate(segments):
        if other_index == target_index or not _is_source_ref_only_exposure_claim(other):
            continue
        facts.update(dict.fromkeys(_source_ref_facts_for_segment(other)))
    return list(facts)


@dataclass(frozen=True)
class ClaimJudgeRecord:
    claim_text: str
    source_ref_facts: list[str]
    context_facts: list[str]
    verdict: str
    reason: str
    would_downgrade: bool


@dataclass(frozen=True)
class DrawResult:
    case_id: str
    draw_index: int
    answer_verdict: str | None
    eligible_claim_count: int
    exposure_claim_count: int
    judge_records: list[ClaimJudgeRecord]
    exception: str | None = None


def _find_case_file(case_id: str) -> Path:
    for path in discover_case_files(_CASES_DIR, _REGRESSIONS_DIR):
        if load_case(path).id == case_id:
            return path
    raise SystemExit(f"no case with id {case_id!r} under {_CASES_DIR} or {_REGRESSIONS_DIR}")


def citation_present_case_ids() -> list[str]:
    """Every case id in the ``citation_present`` category -- same 12-case
    population #130's spike used."""
    cases: list[EvalCase] = [load_case(p) for p in discover_case_files(_CASES_DIR, _REGRESSIONS_DIR)]
    return sorted(case.id for case in cases if case.category == _CATEGORY)


def run_one_draw(case_id: str, draw_index: int, judge: SemanticSupportJudgeLike) -> DrawResult:
    """Run ``case_id`` once, live, through the real pipeline, then shadow-
    judge every SourceRef-only exposure claim in the rendered answer with
    the established-facts-aware judge. Exceptions from the pipeline itself
    are caught and recorded (never raised) as a failed :class:`DrawResult`
    so one bad draw doesn't abort the whole multi-draw session."""
    try:
        case = load_case(_find_case_file(case_id))
        result = run_case(case, judge)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 -- spike run: record failure, keep going
        return DrawResult(
            case_id=case_id,
            draw_index=draw_index,
            answer_verdict=None,
            eligible_claim_count=0,
            exposure_claim_count=0,
            judge_records=[],
            exception=repr(exc),
        )

    answer_verdict = result.verdict_result.verdict.value if result.verdict_result is not None else None
    records: list[ClaimJudgeRecord] = []
    eligible_claim_count = 0
    if result.rendered is not None:
        segments = [s for s in result.rendered.segments if isinstance(s, RenderedClaim)]
        # Structural pass, computed BEFORE any judge call and independent of
        # it (gate-3 review, issue #170 MAJOR 3): the earlier version
        # incremented `eligible_claim_count` inside the SAME loop iteration
        # that unconditionally appended a judge record, making
        # `eligible == judged` a tautology that could never fail regardless
        # of whether the judge call actually ran -- exactly the shape of
        # check this project has now caught four times. This count is a pure
        # function of the rendered segments alone; it does not change even
        # if the judge loop below is skipped, errors, or is refactored.
        eligible_claim_count = sum(1 for segment in segments if _is_source_ref_only_exposure_claim(segment))
        for index, segment in enumerate(segments):
            if not _is_source_ref_only_exposure_claim(segment):
                continue
            facts = _source_ref_facts_for_segment(segment)
            context_facts = _established_facts_for_segment(index, segments)
            judgement = judge_source_ref_relevance_full(segment.text, facts, judge, context_facts)
            records.append(
                ClaimJudgeRecord(
                    claim_text=segment.text,
                    source_ref_facts=facts,
                    context_facts=context_facts,
                    verdict=judgement.verdict.value,
                    reason=judgement.reason,
                    would_downgrade=would_downgrade(judgement),
                )
            )
    return DrawResult(
        case_id=case_id,
        draw_index=draw_index,
        answer_verdict=answer_verdict,
        eligible_claim_count=eligible_claim_count,
        exposure_claim_count=len(records),
        judge_records=records,
    )


# --- positive control: reconstructed issue #123 false-positive shape,
# --- IDENTICAL to issue_130_spike.py's, for a like-for-like comparison ------

_POSITIVE_CONTROL_CLAIM = "The patient's blood pressure was elevated."
_POSITIVE_CONTROL_FACTS = ["problem_count: 0"]

# Gate-3 review, issue #170 MAJOR 1: the context-free control above never
# exercises the ONE variable this whole re-measurement is about -- whether
# the judge can still REJECT when an ESTABLISHED FACTS block is present.
# Every rejection observed anywhere in this spike's live run was made in
# #130's context-FREE configuration (this control, and every case-draw claim
# that happened to have no eligible siblings); every WITH-context judgement
# was `supported`. That is exactly the failure mode a fix whose entire
# purpose is "stop rejecting" could produce silently: if adding context
# makes the judge broadly permissive rather than selectively so, a run with
# no with-context negative case would look IDENTICAL to a genuinely fixed
# one. This control adds a plausible, unrelated sibling context block (a
# different medication's name/dose -- topically irrelevant to a blood-
# pressure claim, same as the context-free control's own irrelevant fact) to
# the SAME #123 claim/fact pair, so a run of this spike always includes at
# least one negative-shaped judgement made WITH context present. See
# ``evals/results/issue-170/summary.json``'s
# ``positive_control_with_context_catch_rate`` for the measured result --
# read it before trusting the aggregate 0/174 false-reject number, per the
# module docstring's "MANDATORY exposure reporting" discipline.
_POSITIVE_CONTROL_WITH_CONTEXT_FACTS = ["name: Simvastatin", "dose: 40mg"]


@dataclass(frozen=True)
class PositiveControlDraw:
    draw_index: int
    verdict: str
    reason: str
    caught: bool
    # Gate-3 review, issue #170 MINOR 1: previously omitted, which meant the
    # 8 with-context control artifacts -- the ones carrying the MOST
    # evidentiary weight in the whole measurement -- were auditable only via
    # filename plus code archaeology (a reader had to re-derive
    # ``_POSITIVE_CONTROL_WITH_CONTEXT_FACTS`` from the source to confirm an
    # ESTABLISHED FACTS block was genuinely on the prompt), unlike ordinary
    # case-draw records, which always carried their own ``context_facts``.
    # Always present now (``[]`` for the context-free control), so every
    # positive-control artifact is self-attesting about what it actually
    # asked the judge -- same auditability principle as
    # ``summarize()``'s ``artifact_content_note``.
    context_facts: list[str]


def run_positive_control(draw_index: int, judge: SemanticSupportJudgeLike) -> PositiveControlDraw:
    """One draw of the scripted issue #123 shape against the SAME
    established-facts-aware judge call the live spike uses -- no sibling
    context is available for this scripted single-claim shape (matching
    #130's own positive control, which also passed no context). ``caught``
    is True iff the judge would downgrade it."""
    judgement = judge_source_ref_relevance_full(_POSITIVE_CONTROL_CLAIM, _POSITIVE_CONTROL_FACTS, judge)
    return PositiveControlDraw(
        draw_index=draw_index,
        verdict=judgement.verdict.value,
        reason=judgement.reason,
        caught=would_downgrade(judgement),
        context_facts=[],
    )


def run_positive_control_with_context(draw_index: int, judge: SemanticSupportJudgeLike) -> PositiveControlDraw:
    """The MAJOR-1 addition: the SAME #123 claim/fact pair as
    ``run_positive_control``, but WITH an established-facts context block
    present (``_POSITIVE_CONTROL_WITH_CONTEXT_FACTS`` -- a different,
    topically-irrelevant medication's name/dose). Exercises the actual
    variable under test in this re-measurement: whether the judge can still
    reject an irrelevant fact when sibling context is also on the prompt, or
    whether adding context makes it broadly permissive. ``caught`` is True
    iff the judge would still downgrade it with context present."""
    judgement = judge_source_ref_relevance_full(
        _POSITIVE_CONTROL_CLAIM, _POSITIVE_CONTROL_FACTS, judge, _POSITIVE_CONTROL_WITH_CONTEXT_FACTS
    )
    return PositiveControlDraw(
        draw_index=draw_index,
        verdict=judgement.verdict.value,
        reason=judgement.reason,
        caught=would_downgrade(judgement),
        context_facts=list(_POSITIVE_CONTROL_WITH_CONTEXT_FACTS),
    )


# --- upgrade-condition-3 tripwire scope check -------------------------------
#
# Gate-3 review, issue #170 MINOR 2: the docs' claim about the strict-xfail
# tripwire's two in-scope claims (`evals/results/issue-170/
# tripwire_scope_check.json`) had no committed generator -- reproducible
# only via ad hoc code archaeology, with no timestamp/model/commit stamp the
# way `summarize()`'s own artifacts have via `--summarize-only`. This is
# that generator: a faithful reproduction of `tests/test_extraction.py
# ._appointment_topical_irrelevance_fixture` (the #170 recorded VULN-0003
# shape), run through the SAME production functions
# `apply_source_ref_relevance` itself calls (`check_claims`,
# `_is_source_ref_only_claim`, `_source_ref_facts`,
# `_established_facts_for_source_ref_claim`, `judge_source_ref_relevance_
# full`) -- not `run_verification` end to end, so the per-claim in-scope/
# out-of-scope determination and each in-scope claim's own facts/context/
# verdict/reason are all individually visible in the output, the same
# things the docs quote.


def _tripwire_fixture_claims_and_index() -> tuple[list[Claim], CacheIndex, list[RerankedChunk]]:
    """Duplicated (not imported) from ``tests/test_extraction.py
    ._appointment_topical_irrelevance_fixture`` -- that module isn't
    importable outside a pytest run (it imports ``pytest`` at module scope,
    and the flattened container image this script also runs in has no
    ``pytest`` installed at all), so the fixture is reproduced verbatim
    here instead. Keep in sync by hand if that fixture ever changes."""
    appointment = AppointmentItem(
        date=datetime.date(2014, 1, 31), time=datetime.time(14, 30, 0),
        status=AppointmentStatus.SCHEDULED, provider="Billy Smith",
    )
    raw = AppointmentsOutput(items=[appointment]).model_dump(mode="json")
    chunk = RerankedChunk(
        chunk_id="hypertension-lifestyle#follow-up-cadence",
        doc_id="hypertension-lifestyle",
        title="Hypertension Management",
        section="Follow-Up Cadence",
        text=(
            "Elevated blood pressure or Stage 1 hypertension managed with lifestyle alone: "
            "recheck at roughly 3-6 months to assess response before deciding whether to add "
            "pharmacotherapy."
        ),
        scores={"hybrid": 0.9},
        rerank_score=0.9,
    )
    guideline_citation = DocumentCitation(
        source_type="guideline_chunk", source_id=chunk.doc_id, page_or_section=chunk.section,
        field_or_chunk_id=chunk.chunk_id, quote_or_value=chunk.text,
    )
    claims = [
        Claim(
            text=(
                "The patient has an appointment scheduled with provider Billy Smith on "
                "2014-01-31 at 14:30:00."
            ),
            source_refs=[
                SourceRef(tool_call_id="call_0", record_id="0", field="date", asserted_value="2014-01-31"),
                SourceRef(tool_call_id="call_0", record_id="0", field="time", asserted_value="14:30:00"),
                SourceRef(tool_call_id="call_0", record_id="0", field="provider", asserted_value="Billy Smith"),
            ],
        ),
        Claim(
            text="The patient's blood pressure was elevated at the last visit.",
            source_refs=[SourceRef(tool_call_id="call_0", record_id="0", field="status", asserted_value="scheduled")],
            document_citations=[guideline_citation],
        ),
        Claim(
            text="The patient is trying lifestyle changes.",
            source_refs=[SourceRef(tool_call_id="call_0", record_id="0", field="status", asserted_value="scheduled")],
            document_citations=[guideline_citation],
        ),
        Claim(
            text="It is recommended to recheck in 3-6 months to assess response.",
            source_refs=[SourceRef(tool_call_id="call_0", record_id="0", field="status", asserted_value="scheduled")],
            document_citations=[guideline_citation],
        ),
        Claim(
            text="The next follow-up should be at the scheduled appointment on 2014-01-31.",
            source_refs=[SourceRef(tool_call_id="call_0", record_id="0", field="date", asserted_value="2014-01-31")],
        ),
    ]
    index = CacheIndex.from_raw_results([raw])
    return claims, index, [chunk]


def run_tripwire_scope_check(judge: SemanticSupportJudgeLike) -> list[dict]:
    """Re-validate the #170 recorded fixture's 5 claims (provenance only --
    no document-fact/corpus index is built, so the 3 claims carrying a
    ``DocumentCitation`` resolve ``UNKNOWN_CHUNK`` on that citation and are
    irrelevant to this check either way), then run the SAME per-claim scope
    + established-facts + judge path ``apply_source_ref_relevance`` uses on
    the 2 claims that ARE in scope (pure ``SourceRef``, already passed
    provenance). Returns one dict per claim: ``in_scope=False`` for the 3
    claims carrying a ``DocumentCitation`` (out of #130-ADR scope, module
    docstring), or the claim's own facts / established-facts context /
    judge verdict / reason for the 2 that are in scope."""
    claims, index, _chunks = _tripwire_fixture_claims_and_index()
    claim_results = check_claims(claims, index)
    out: list[dict] = []
    for claim_result in claim_results:
        if not _is_source_ref_only_claim(claim_result.claim):
            out.append({"claim": claim_result.claim.text, "in_scope": False})
            continue
        own_facts = _source_ref_facts(claim_result.claim)
        context_facts = _established_facts_for_source_ref_claim(claim_result, claim_results)
        judgement = judge_source_ref_relevance_full(claim_result.claim.text, own_facts, judge, context_facts)
        out.append(
            {
                "claim": claim_result.claim.text,
                "in_scope": True,
                "own_facts": own_facts,
                "context_facts": context_facts,
                "verdict": judgement.verdict.value,
                "reason": judgement.reason,
            }
        )
    return out


def save_tripwire_scope_check(records: list[dict]) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _RESULTS_DIR / "tripwire_scope_check.json"
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return path


# --- incremental save / summarize ------------------------------------------


def _draw_path(case_id: str, draw_index: int) -> Path:
    return _DRAWS_DIR / f"{case_id}-draw{draw_index}.json"


def save_draw(draw: DrawResult) -> Path:
    _DRAWS_DIR.mkdir(parents=True, exist_ok=True)
    path = _draw_path(draw.case_id, draw.draw_index)
    path.write_text(json.dumps(asdict(draw), indent=2), encoding="utf-8")
    return path


_CONTEXT_FREE_CONTROL_PREFIX = "positive-control-draw"
_WITH_CONTEXT_CONTROL_PREFIX = "positive-control-with-context-draw"


def save_positive_control_draw(draw: PositiveControlDraw, *, with_context: bool = False) -> Path:
    _DRAWS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = _WITH_CONTEXT_CONTROL_PREFIX if with_context else _CONTEXT_FREE_CONTROL_PREFIX
    path = _DRAWS_DIR / f"{prefix}{draw.draw_index}.json"
    path.write_text(json.dumps(asdict(draw), indent=2), encoding="utf-8")
    return path


def summarize(draws_dir: Path) -> dict:
    """Aggregate every ``draws/*.json`` artifact into one summary dict.

    Per case: ``eligible`` (structural exposure -- SourceRef-only claims
    seen in the rendered answer, regardless of whether the judge call
    happened to run cleanly), ``judged`` (claims the judge actually scored),
    ``false_rejects`` (every recorded judgement here is, by construction, on
    a claim the live pipeline already scored PASSED -- i.e.
    currently-passing -- so ANY ``would_downgrade`` IS a false reject in the
    ADR's sense), ``draws``, ``exceptions``. ``status`` is
    ``"INCONCLUSIVE: zero exposure"`` when a case's ``eligible`` count is 0
    across every draw -- see module docstring, "MANDATORY exposure
    reporting" -- so a zero-power case can never silently read as
    clearance. Also stamps ``artifact_content_note`` (see module docstring,
    "Deliberate deviation from #163/#169's ... convention") so the summary
    itself, not just the module docstring, records why the per-draw files
    keep prose."""
    per_case: dict[str, dict] = {}
    total_eligible = 0
    total_judged = 0
    total_false_rejects = 0
    positive_control_draws = 0
    positive_control_caught = 0
    positive_control_with_context_draws = 0
    positive_control_with_context_caught = 0

    for path in sorted(draws_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        # MAJOR 1's with-context control filename is checked FIRST -- it is a
        # more specific prefix ("positive-control-with-context-draw") than
        # the context-free control's ("positive-control-draw"), and the two
        # do not collide (neither is a prefix of the other), but checking
        # order-independent would still be correct here; ordering it this
        # way keeps the intent explicit.
        if path.name.startswith(_WITH_CONTEXT_CONTROL_PREFIX):
            positive_control_with_context_draws += 1
            if payload["caught"]:
                positive_control_with_context_caught += 1
            continue
        if path.name.startswith(_CONTEXT_FREE_CONTROL_PREFIX):
            positive_control_draws += 1
            if payload["caught"]:
                positive_control_caught += 1
            continue
        case_id = payload["case_id"]
        bucket = per_case.setdefault(
            case_id, {"eligible": 0, "judged": 0, "false_rejects": 0, "draws": 0, "exceptions": 0}
        )
        bucket["draws"] += 1
        bucket["eligible"] += payload.get("eligible_claim_count", 0)
        total_eligible += payload.get("eligible_claim_count", 0)
        if payload.get("exception"):
            bucket["exceptions"] += 1
            continue
        for record in payload["judge_records"]:
            bucket["judged"] += 1
            total_judged += 1
            if record["would_downgrade"]:
                bucket["false_rejects"] += 1
                total_false_rejects += 1

    for bucket in per_case.values():
        bucket["status"] = "INCONCLUSIVE: zero exposure" if bucket["eligible"] == 0 else "measured"

    return {
        "per_case": per_case,
        "total_eligible_claims": total_eligible,
        "total_judged_claims": total_judged,
        "total_false_rejects": total_false_rejects,
        "positive_control_draws": positive_control_draws,
        "positive_control_caught": positive_control_caught,
        "positive_control_catch_rate": (
            positive_control_caught / positive_control_draws if positive_control_draws else None
        ),
        # MAJOR 1 (gate-3 review, issue #170): the ONLY control in this
        # summary that exercises the variable under test -- whether the
        # judge still rejects an irrelevant fact when an ESTABLISHED FACTS
        # block is present. Read this rate before trusting the aggregate
        # false-reject total: every other rejection in this run (this
        # control included) was context-free, so a run with zero
        # with-context negative evidence could look identical whether the
        # established-facts fix is selectively permissive or broadly so.
        "positive_control_with_context_draws": positive_control_with_context_draws,
        "positive_control_with_context_caught": positive_control_with_context_caught,
        "positive_control_with_context_catch_rate": (
            positive_control_with_context_caught / positive_control_with_context_draws
            if positive_control_with_context_draws
            else None
        ),
        "artifact_content_note": (
            "Per-draw files under draws/ intentionally retain full claim text, field/value "
            "facts, and judge rationale prose -- a deliberate deviation from #163/#169's "
            "counts-and-enums-only artifact convention. Safe here because every claim/fact is "
            "synthetic eval-fixture content (the 12 citation_present cases' own scripted data "
            "plus the hardcoded positive-control claim), never real patient data, unlike "
            "#163/#169's populations which could carry real claim text. The rationale is kept "
            "because a judge measurement is not auditable without it -- a bare true/false "
            "cannot be checked for WHY the judge decided what it decided."
        ),
    }


def save_summary(summary: dict) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _RESULTS_DIR / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=_DEFAULT_DRAWS, help="draws per case (default 8)")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="skip live runs; just re-aggregate evals/results/issue-170/draws/",
    )
    parser.add_argument(
        "--with-context-control-only",
        action="store_true",
        help=(
            "run ONLY the MAJOR-1 with-context positive control (--draws draws), skipping the "
            "context-free control and all 12 case-draw sets -- for a targeted top-up run against "
            "an existing draws/ directory, without re-spending the full ~96-draw session."
        ),
    )
    parser.add_argument(
        "--tripwire-scope-check",
        action="store_true",
        help=(
            "run ONLY the MINOR-2 upgrade-condition-3 tripwire scope check (one live call per "
            "in-scope claim, no --draws loop) and write evals/results/issue-170/"
            "tripwire_scope_check.json -- skips every other control and all case-draw sets."
        ),
    )
    args = parser.parse_args()

    if args.tripwire_scope_check:
        tripwire_settings = Settings(llama_server_api_timeout_seconds=180.0)
        tripwire_judge = LlamaServerClient.from_settings(tripwire_settings)
        records = run_tripwire_scope_check(tripwire_judge)
        path = save_tripwire_scope_check(records)
        print(json.dumps(records, indent=2))
        print(f"[spike] tripwire scope check -> {path}")
        return

    if not args.summarize_only:
        settings = Settings(llama_server_api_timeout_seconds=180.0)
        judge: SemanticSupportJudgeLike = LlamaServerClient.from_settings(settings)  # type: ignore[assignment]

        if not args.with_context_control_only:
            for draw_index in range(args.draws):
                control = run_positive_control(draw_index, judge)
                path = save_positive_control_draw(control)
                print(
                    f"[spike] positive-control draw {draw_index}: "
                    f"verdict={control.verdict} caught={control.caught} -> {path}"
                )

        for draw_index in range(args.draws):
            control = run_positive_control_with_context(draw_index, judge)
            path = save_positive_control_draw(control, with_context=True)
            print(
                f"[spike] positive-control-with-context draw {draw_index}: "
                f"verdict={control.verdict} caught={control.caught} -> {path}"
            )

        if not args.with_context_control_only:
            for case_id in citation_present_case_ids():
                for draw_index in range(args.draws):
                    draw = run_one_draw(case_id, draw_index, judge)
                    path = save_draw(draw)
                    status = (
                        f"EXCEPTION {draw.exception}"
                        if draw.exception
                        else f"{draw.eligible_claim_count} eligible, {draw.exposure_claim_count} judged"
                    )
                    print(f"[spike] {case_id} draw {draw_index}: {status} -> {path}")

    summary = summarize(_DRAWS_DIR)
    save_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
