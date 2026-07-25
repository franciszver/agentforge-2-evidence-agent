"""Issue #154 N-draw verdict-stability harness (diagnosed by #149/#150).

**What this measures.** #149 and #150 each independently found that running
the SAME question against the SAME patient, repeatedly, produced a
DIFFERENT whole-answer verdict (``verified`` vs ``blocked``) on
semantically-identical answer text:

  * #149 -- "What was his last blood pressure reading, and what category
    does that fall into?" (``patient_id=1``) -- the ``bp-stage2-question``
    case already committed under ``evals/cases/citation_present/``.
  * #150 -- "Is there an allergy conflict with the current medications?"
    (``patient_id=2``, Susan Underwood -- Lisinopril/Lipitor/Metformin, no
    recorded allergies) -- reconstructed here (``_ALLERGY_CONFLICT_CASE``)
    since no committed case covers this exact question/patient/data yet.
    Deliberately NOT added under ``evals/cases/`` -- that directory feeds
    ``evals/test_cases.py``'s parametrized replay suite, which requires a
    committed ``evals/recordings/<id>.json`` golden recording; this harness
    is a live, repeated-draw MEASUREMENT tool, not a new golden case, so it
    builds its ``EvalCase`` in memory instead (see ``runner.schema.EvalCase``
    -- nothing requires the case to be loaded from a YAML file on disk).

#150's own diagnosis comment traces the mechanism: ``ClaimExtractor
.extract_claims`` (``app/extraction.py``) catches ``LLMEngineError`` after
the client's internal retries and returns ``[]``, which ``app.verdict``'s
``NONE_VERIFIED`` row fail-closes to ``blocked`` regardless of the safety
axis -- INFERRED from one observed draw, not directly measured. This
harness is the measurement: it runs each question N times against the LIVE
model (never a replay -- see "Live, not replay" below) through the REAL
pipeline (``runner.pipeline.run_case``, the exact same entry point
``evals/runner/record.py`` and ``issue_130_spike.py`` use) and reports the
verdict / claim-count / answer-text distribution across draws.

**Live, not replay.** Every draw is a genuine live model call, mirroring
``evals/runner/record.py``'s live-client construction (``RECORD_ENGINE`` /
``OLLAMA_BASE_URL`` / ``LLAMA_SERVER_BASE_URL``) and dual-layout ``sys.path``
handling (#119/#135) so this script resolves the LIVE ``app`` package
whether run from a full monorepo checkout or inside the
``development-easy-agent-1`` flattened container -- same as
``issue_130_spike.py``.

**Live-model runs stay OUT of CI (hard requirement, repeated here for
anyone auditing this file in isolation).** This module:

  * is named ``issue_154_stability_harness.py`` -- it matches neither
    pytest's default ``test_*.py``/``*_test.py`` collection glob nor any
    override in ``evals/pytest.ini`` (which sets none), so
    ``pytest evals/ -m "not integration"`` (the exact invocation
    ``.github/workflows/copilot-ci.yml`` runs) never collects it, exactly
    like the pre-existing ``issue_130_spike.py``/``record.py`` in this same
    directory, neither of which CI collects either.
  * makes a live model call ONLY inside ``main()``, which runs ONLY under
    ``if __name__ == "__main__":`` -- importing this module (as a human
    might, to reuse ``summarize()`` against an existing ``draws/``
    directory) never dials out.
  * is never imported by any ``evals/test_*.py`` / ``evals/runner/tests/``
    module -- grep confirms no reference anywhere else in the suite.

**Incremental, auditable artifacts.** Every draw writes TWO files to
``evals/results/issue-154/draws/`` immediately after that draw completes (a
crash mid-run loses at most one draw): a raw model-call transcript, in the
EXACT ``RecordedCall``/``save_recording`` JSON shape ``evals/recordings/``
uses (``<case_id>-draw<N>.calls.json`` -- built via the same
``RecordingOllamaClient`` wrapper ``record.py`` uses, so a draw's full LLM
call sequence is independently inspectable/replayable with the existing
``runner.ollama_replay`` tooling), and a summary of that draw's pipeline
result (``<case_id>-draw<N>.json`` -- verdict, claim counts, answer text,
or the exception if the draw itself failed). Deliberately under
``evals/results/issue-154/``, NOT ``evals/recordings/`` itself -- the latter
holds the ONE canonical, committed golden recording per case that
``evals/test_cases.py`` replays in CI; N per-draw recordings for the same
case must not collide with, or be mistaken for, that golden file. This
mirrors ``issue_130_spike.py``'s own ``evals/results/issue-130/draws/``
precedent exactly.

``summarize()`` (also runnable standalone against an existing ``draws/``
directory via ``--summarize-only``) aggregates every draw's summary file
into ``evals/results/issue-154/summary.json``: per question, N, the verdict
distribution, the extracted (``total_claim_count``) distribution, the
extraction-failure distribution, and whether the answer text varied.
**Exact-text stability only** -- this harness does a byte-for-byte string
comparison of ``planner_result.answer`` across draws; it does NOT do
semantic/paraphrase comparison (that would need its own judge call, like
``issue_130_spike.py``'s SourceRef-relevance judge, which is out of scope
here). The summary says explicitly which kind of comparison was done rather
than implying more than was checked.

**A zero-claim ``blocked`` draw is NOT one thing.** It can mean claim
extraction's retry exhaustion fail-closed, an empty catalog with nothing to
extract from, or an answer with genuinely nothing verifiable -- three
different causes with the byte-identical ``verdict=blocked,
total_claim_count=0`` shape. See ``DrawResult.extraction_failed`` and
``summarize()``'s docstring for exactly which of the three each draw's row
lets you tell apart. This harness also calls ``app.correlation
.configure_logging()`` itself (see ``main()``) -- unlike ``app.main``, it
never imports that module, so without this the retry-exhaustion warning
would silently fall through to ``logging.lastResort`` with the correlation
id dropped, in no artifact.

**The two target questions are not like-for-like stability comparisons.**
The BP case (#149) triggers an extra judge-model call the allergy case
(#150) does not (``needs_semantic_support()`` -- see ``summarize()``'s
docstring for the mechanism). Read each question's own distribution across
its own draws; do not compare the BP question's rate against the allergy
question's rate as if they exercised the same pipeline surface.

Usage (from repo root, live model reachable -- e.g. inside
``development-easy-agent-1`` after ``docker cp``-ing fresh sources in, see
issue #140, or against a host-published bridge):

    RECORD_ENGINE=llama_server python evals/runner/issue_154_stability_harness.py --draws 8
    OLLAMA_BASE_URL=http://localhost:11435 python evals/runner/issue_154_stability_harness.py --draws 8
    python evals/runner/issue_154_stability_harness.py --summarize-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_EVALS_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _EVALS_ROOT.parent
_MONOREPO_AGENT_ROOT = _REPO_ROOT / "services" / "copilot-agent"


def _agent_root_candidates(repo_root: Path, monorepo_agent_root: Path) -> list[Path]:
    """Same dual-layout resolution as ``evals/runner/record.py`` /
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

from app.config import Settings  # noqa: E402
from app.correlation import configure_logging  # noqa: E402
from app.llama_server_client import LlamaServerClient  # noqa: E402
from app.ollama_client import OllamaClient  # noqa: E402
from app.schemas.planner import ToolName  # noqa: E402
from app.verdict import Verdict  # noqa: E402

from runner.loader import discover_case_files, load_case  # noqa: E402
from runner.ollama_replay import OllamaLike, RecordedCall, RecordingOllamaClient, save_recording  # noqa: E402
from runner.pipeline import run_case  # noqa: E402
from runner.schema import EvalCase, VerdictAssertion  # noqa: E402

_CASES_DIR = _EVALS_ROOT / "cases"
_REGRESSIONS_DIR = _EVALS_ROOT / "regressions"
_RESULTS_DIR = _EVALS_ROOT / "results" / "issue-154"
_DRAWS_DIR = _RESULTS_DIR / "draws"

_DEFAULT_DRAWS = 8

# --- the two known-unstable questions (#149, #150) -------------------------

_BP_CASE_ID = "bp-stage2-question"  # #149 -- already committed, reused as-is

# #150 -- Susan Underwood (patient_id=2): Lisinopril, Lipitor, Metformin,
# no recorded allergies. Reconstructed from #150's diagnosis comment, which
# quotes the answer text verbatim ("no recorded allergies ... Lisinopril,
# Lipitor, and Metformin") and the observed 4x verified / 1x blocked split.
# ``assertions`` only needs to be non-empty and include a VerdictAssertion so
# ``runner.pipeline.needs_verification`` triggers the extraction/verification
# stage this harness measures -- it is never evaluated as a pass/fail here
# (this harness reports the raw distribution, it does not grade draws).
_ALLERGY_CONFLICT_CASE = EvalCase(
    id="issue-150-allergy-conflict-check",
    category="constraint",
    failure_mode=(
        "#150: the whole-answer verdict (verified/blocked) and citation "
        "chips vary run-to-run for a semantically identical answer -- see "
        "issue #150's diagnosis comments for the reconstructed mechanism "
        "(claim-extraction retry exhaustion fail-closing to blocked)."
    ),
    question="Is there an allergy conflict with the current medications?",
    patient_id=2,
    tool_data={
        ToolName.GET_MEDICATIONS: {
            "items": [
                {"name": "Lisinopril", "dose": "10 mg", "route": "oral", "status": "active"},
                {"name": "Lipitor", "dose": "20 mg", "route": "oral", "status": "active"},
                {"name": "Metformin", "dose": "500 mg", "route": "oral", "status": "active"},
            ]
        },
        ToolName.GET_ALLERGIES: {"items": []},
    },
    assertions=[VerdictAssertion(type="verdict", equals=Verdict.VERIFIED)],
)

_TARGET_CASE_IDS = (_BP_CASE_ID, _ALLERGY_CONFLICT_CASE.id)


def _find_case_file(case_id: str) -> Path:
    for path in discover_case_files(_CASES_DIR, _REGRESSIONS_DIR):
        if load_case(path).id == case_id:
            return path
    raise SystemExit(f"no case with id {case_id!r} under {_CASES_DIR} or {_REGRESSIONS_DIR}")


def _load_target_case(case_id: str) -> EvalCase:
    """Resolve one of ``_TARGET_CASE_IDS`` to its ``EvalCase`` -- the #149
    case from its committed YAML file, the #150 case from the in-memory
    constant above (see module docstring for why it is not a YAML file)."""
    if case_id == _ALLERGY_CONFLICT_CASE.id:
        return _ALLERGY_CONFLICT_CASE
    return load_case(_find_case_file(case_id))


def _build_live_client(engine: str, ollama_base_url: str) -> OllamaLike:
    """Same construction as ``evals/runner/record.py``'s
    ``_build_live_client`` -- duplicated rather than imported so this
    script's own live-client wiring doesn't depend on ``record.py``'s CLI
    surface (``EXPECTED_APP_STAMP``/code-stamp gate) staying compatible."""
    if engine == "llama_server":
        settings = Settings(llama_server_api_timeout_seconds=180.0)
        return LlamaServerClient.from_settings(settings)  # type: ignore[return-value]
    settings = Settings(ollama_base_url=ollama_base_url, ollama_api_timeout_seconds=180.0)
    return OllamaClient.from_settings(settings)  # type: ignore[return-value]


@dataclass(frozen=True)
class DrawResult:
    case_id: str
    draw_index: int
    verdict: str | None
    total_claim_count: int | None
    stripped_claim_count: int | None
    answer: str | None
    exception: str | None = None
    # #154 F1: which of the THREE zero-claim causes actually happened, per
    # draw -- see ``summarize()``'s docstring for the full breakdown.
    # ``True`` iff ``app.extraction``'s retry-exhaustion warning ("claim
    # extraction exhausted retries...") fired during this draw (captured by
    # ``_RetryExhaustionWatcher`` below); ``False`` for a draw that completed
    # without it (including one whose pipeline itself raised, so it never
    # reached extraction); ``None`` only if the draw's exception happened
    # before extraction could even be attempted is still ``False`` here --
    # exception draws are excluded from this field's meaning entirely and
    # should be read via ``exception`` instead.
    extraction_failed: bool = False


class _RetryExhaustionWatcher(logging.Handler):
    """Non-pytest equivalent of ``caplog`` for one thing only: did
    ``app.extraction``'s retry-exhaustion warning (see ``app/extraction.py``'s
    ``except LLMEngineError`` block, logger ``app.extraction``, message
    ``"claim extraction exhausted retries..."``) fire during this draw?

    ``caplog`` works by temporarily attaching a handler to the root logger
    for the duration of one test; this does the same thing for the duration
    of one draw, attached directly to the ``app.extraction`` logger (matching
    ``tests/test_extraction.py``'s own ``caplog.set_level(..., logger=
    "app.extraction")`` scoping) so it does not depend on -- or interfere
    with -- whatever handlers ``configure_logging()`` installed on the root
    logger for the process's actual structured-log output. Attach with the
    ``fired`` context manager below; do not use this handler directly."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.fired = False

    def emit(self, record: logging.LogRecord) -> None:
        if "exhausted retries" in record.getMessage():
            self.fired = True


@contextmanager
def _watch_for_retry_exhaustion() -> Iterator[_RetryExhaustionWatcher]:
    watcher = _RetryExhaustionWatcher()
    logger = logging.getLogger("app.extraction")
    logger.addHandler(watcher)
    try:
        yield watcher
    finally:
        logger.removeHandler(watcher)


def run_one_draw(case: EvalCase, draw_index: int, client: OllamaLike) -> tuple[DrawResult, list[RecordedCall]]:
    """Run ``case`` once, live, through the real pipeline (``runner.pipeline
    .run_case``), wrapping ``client`` in a ``RecordingOllamaClient`` so the
    exact LLM call sequence for this one draw is capturable to disk
    alongside the summary. Exceptions from the pipeline itself are caught
    and recorded (never raised) as a failed :class:`DrawResult` so one bad
    draw doesn't abort the whole multi-draw session -- same discipline as
    ``issue_130_spike.py``'s ``run_one_draw``.

    Also watches (``_watch_for_retry_exhaustion``) whether ``app.extraction``'s
    retry-exhaustion warning fired during this draw, so the returned
    :class:`DrawResult` can distinguish "extraction infrastructure failed"
    from a merely zero-claim draw -- see ``summarize()``'s docstring."""
    recorder = RecordingOllamaClient(client)  # type: ignore[arg-type]
    with _watch_for_retry_exhaustion() as watcher:
        try:
            result = run_case(case, recorder)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 -- harness run: record failure, keep going
            return (
                DrawResult(
                    case_id=case.id,
                    draw_index=draw_index,
                    verdict=None,
                    total_claim_count=None,
                    stripped_claim_count=None,
                    answer=None,
                    exception=repr(exc),
                    extraction_failed=watcher.fired,
                ),
                recorder.calls,
            )

    verdict_result = result.verdict_result
    draw = DrawResult(
        case_id=case.id,
        draw_index=draw_index,
        verdict=verdict_result.verdict.value if verdict_result is not None else None,
        total_claim_count=verdict_result.total_claim_count if verdict_result is not None else None,
        stripped_claim_count=verdict_result.stripped_claim_count if verdict_result is not None else None,
        answer=result.planner_result.answer,
        extraction_failed=watcher.fired,
    )
    return draw, recorder.calls


# --- incremental save / summarize ------------------------------------------


def _summary_draw_path(case_id: str, draw_index: int) -> Path:
    return _DRAWS_DIR / f"{case_id}-draw{draw_index}.json"


def _calls_draw_path(case_id: str, draw_index: int) -> Path:
    return _DRAWS_DIR / f"{case_id}-draw{draw_index}.calls.json"


def save_draw(draw: DrawResult, calls: list[RecordedCall]) -> tuple[Path, Path]:
    _DRAWS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = _summary_draw_path(draw.case_id, draw.draw_index)
    summary_path.write_text(json.dumps(asdict(draw), indent=2), encoding="utf-8")
    calls_path = _calls_draw_path(draw.case_id, draw.draw_index)
    save_recording(calls_path, calls)  # same JSON shape evals/recordings/ uses
    return summary_path, calls_path


def summarize(draws_dir: Path) -> dict[str, Any]:
    """Aggregate every ``draws/<case_id>-draw*.json`` summary artifact (the
    ``.calls.json`` transcripts are audit trail only -- not read here) into
    a per-question report: N, the verdict distribution, the extracted-claim
    (``total_claim_count``) distribution, the extraction-failure distribution
    (see below), and whether the answer text varied -- EXACT string
    comparison only, explicitly labeled as such (see module docstring's
    "Exact-text stability only"). Never touches any verdict or recording --
    pure read-and-aggregate.

    **A zero-claim ``blocked`` draw has THREE possible causes** (#154 F1 --
    do not read ``total_claim_count == 0`` alone as "nothing verifiable"):

      1. **Claim-extraction retry exhaustion.** ``app.extraction.ClaimExtractor
         .extract_claims`` caught ``LLMEngineError`` after the client's own
         internal retries and gave up, returning ``[]``. This is an
         INFRASTRUCTURE failure (a flaky/overloaded model backend), not a
         judgment about the answer. Identified by ``extraction_failed: true``
         in this draw's row (captured live per-draw by
         ``_watch_for_retry_exhaustion`` watching for ``app.extraction``'s
         "claim extraction exhausted retries" warning -- see F2 below for why
         that warning would otherwise be silently dropped).
      2. **Empty catalogs.** ``ClaimExtractor.extract_claims``
         (``app/extraction.py`` ~line 356) returns ``[]`` before ever calling
         the model, when the tool-result catalog, guideline catalog, AND
         patient-fact catalog are all empty -- there was nothing to extract
         claims FROM. ``extraction_failed`` is ``False`` for this cause (no
         warning fires; extraction was never attempted).
      3. **Genuinely nothing verifiable.** The model was called, extraction
         succeeded, and it correctly found zero claims worth verifying in the
         answer text. ``extraction_failed`` is also ``False`` here.

    ``extraction_failed`` (this summary's ``extraction_failed_distribution``)
    is therefore the field that separates cause 1 from causes 2 and 3. This
    harness does NOT further distinguish cause 2 from cause 3 -- both produce
    an identical zero-claim, non-warning row; telling them apart would
    require inspecting the tool/guideline/fact inputs given to that draw
    (available in the ``.calls.json`` transcript) rather than anything this
    summary aggregates. A reader who needs that distinction should go look at
    the specific draw's recorded call inputs.

    **#154 F4 -- the two target questions' rates are not like-for-like.**
    The BP case (``bp-stage2-question``, #149) carries a
    ``GuidelineCitationPresentAssertion``, which makes
    ``runner.pipeline``'s ``needs_semantic_support()`` return ``True`` for
    it and NOT for the allergy-conflict case (#150) -- so every BP draw
    makes one extra judge-model call the allergy draw does not. Per-question
    stability (each question's own verdict/claim-count distribution across
    its own draws) is still valid to read; comparing the BP question's rate
    directly against the allergy question's rate as if they exercised the
    same pipeline surface is not."""
    per_case: dict[str, dict[str, Any]] = {}

    for path in sorted(draws_dir.glob("*.json")):
        if path.name.endswith(".calls.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_id = payload["case_id"]
        bucket = per_case.setdefault(
            case_id,
            {
                "n_draws": 0,
                "n_exceptions": 0,
                "verdict_distribution": Counter(),
                "total_claim_count_distribution": Counter(),
                "extraction_failed_distribution": Counter(),
                "answers_seen": [],
            },
        )
        bucket["n_draws"] += 1
        if payload.get("exception"):
            bucket["n_exceptions"] += 1
            continue
        bucket["verdict_distribution"][payload["verdict"]] += 1
        bucket["total_claim_count_distribution"][payload["total_claim_count"]] += 1
        bucket["extraction_failed_distribution"][bool(payload.get("extraction_failed", False))] += 1
        bucket["answers_seen"].append(payload["answer"])

    report: dict[str, Any] = {}
    for case_id, bucket in per_case.items():
        distinct_answers = list(dict.fromkeys(bucket["answers_seen"]))  # order-preserving dedupe
        n_counted = bucket["n_draws"] - bucket["n_exceptions"]
        report[case_id] = {
            "n_draws": bucket["n_draws"],
            "n_exceptions": bucket["n_exceptions"],
            # #154 F3: every *_distribution below is computed over these
            # ``n_counted`` non-exception draws only -- they do NOT sum to
            # ``n_draws`` when ``n_exceptions > 0``. n_counted = n_draws -
            # n_exceptions, spelled out explicitly so a reader never has to
            # infer it.
            "n_counted": n_counted,
            "verdict_distribution": dict(bucket["verdict_distribution"]),
            "total_claim_count_distribution": {
                str(k): v for k, v in bucket["total_claim_count_distribution"].items()
            },
            # #154 F1: True count = retry-exhaustion draws (cause 1 above);
            # False count = causes 2+3 combined (extraction not attempted, or
            # attempted and genuinely found nothing -- not distinguished).
            "extraction_failed_distribution": {
                str(k): v for k, v in bucket["extraction_failed_distribution"].items()
            },
            "answer_text_comparison": "exact",  # NOT semantic -- see module docstring
            "answer_text_varied": len(distinct_answers) > 1,
            "distinct_answer_count": len(distinct_answers),
        }
    return report


def save_summary(summary: dict[str, Any]) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _RESULTS_DIR / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=_DEFAULT_DRAWS, help="draws per question (default 8)")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="skip live runs; just re-aggregate evals/results/issue-154/draws/",
    )
    args = parser.parse_args()

    # #154 F2: ``app.main`` is the only place that previously called this
    # (at import time, module-level). This harness imports ``runner.pipeline``
    # -> ``app.extraction`` directly and never imports ``app.main``, so
    # without this the root logger has no handlers during a harness run: the
    # retry-exhaustion warning (see ``app/extraction.py``'s ``except
    # LLMEngineError`` block) falls through to ``logging.lastResort``, which
    # prints a bare, unstructured line to stderr with ``error_type`` and the
    # correlation id DROPPED, and lands in no artifact. Calling it here,
    # unconditionally (idempotent -- see its own docstring) and before any
    # draw runs, matches ``app.main``'s invocation (no arguments -> default
    # INFO level) so a harness run gets the same structured JSON log lines
    # ``app.main`` would produce in production.
    configure_logging()

    if not args.summarize_only:
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        engine = os.environ.get("RECORD_ENGINE", "ollama")

        for case_id in _TARGET_CASE_IDS:
            case = _load_target_case(case_id)
            for draw_index in range(args.draws):
                # A fresh client per draw -- mirrors a fresh POST /chat
                # invocation each time, exactly as #149/#150's own manual
                # draws did against one booted stack.
                client = _build_live_client(engine, ollama_base_url)
                draw, calls = run_one_draw(case, draw_index, client)
                summary_path, calls_path = save_draw(draw, calls)
                status = f"EXCEPTION {draw.exception}" if draw.exception else f"verdict={draw.verdict} claims={draw.total_claim_count}"
                print(f"[stability] {case.id} draw {draw_index}: {status} -> {summary_path}, {calls_path}")

    summary = summarize(_DRAWS_DIR)
    save_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
