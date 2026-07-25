"""Issue #160 live-`POST /chat` N-draw stability harness -- the layer #154's
fixed-tool-data harness cannot see, by construction.

**Why this exists.** #149 (BP question: 1/4 draws answer blood pressure, 3/4
decline) and #150 (allergy-conflict verdict 5 Verified / 1 Blocked across 6
draws) were both observed on live ``POST /chat`` draws exercising real
planner tool calls end to end against the real chart. Issue #154's harness
(``evals/runner/issue_154_stability_harness.py``) measured the SAME two
questions but with ``tool_data`` PINNED -- its own docstring calls this a
"hard scope limit": every draw's tool dispatch returns byte-identical canned
data, so #154 can only ever measure variance strictly DOWNSTREAM of tool
results (answer composition, extraction, verification). At N=8 it found
0/16 extraction failures and stable verdicts -- meaning the remaining
variance #149/#150 actually observed must originate UPSTREAM, in the
planner's live tool-calling and/or generation nondeterminism, a layer #154
never runs. This harness is that missing measurement.

**Live end to end, no shortcuts.** Every draw is a real ``POST /chat`` HTTP
request against the running ``agent`` service -- real planner tool dispatch
against the real ``OpenEmrClient`` and the real seeded chart, real answer
generation, real claim extraction/verification, over the exact same wire
protocol the browser panel uses. This module contains NO import of
``app.*``, ``runner.pipeline``, or ``runner.tool_stub`` -- it cannot
in-process-call the planner or hand it pinned ``tool_data`` even by
accident, because it has no import path to any of that machinery (see
``evals/runner/tests/test_issue_160_live_chat_harness.py``'s
``TestNoToolStub`` for the pinned-by-test version of this claim). The ONLY
way this module can produce a draw is by talking HTTP to a live server.

**What the wire protocol (SSE) actually exposes, and why the schema below
is shaped the way it is.** ``app/chat.py``'s own module docstring documents
the ``ChatEvent`` frame contract; three frames matter here:

  * ``tool_call`` -- one per planner tool dispatch, IN ORDER, carrying
    ``{"tool": str, "args": dict, "error": str | None}``. This is the
    planner's live tool-selection trace -- exactly what #154 cannot see
    (its tool layer is a fixture, never dispatched). No PHI is exposed
    here: the raw chart VALUES a tool call returns are never sent over SSE
    at all (only the client-facing ``tool_call`` frame's ``tool``/``args``/
    ``error`` -- see ``app/chat.py``'s ``ToolCallTrace`` docstring: "Raw
    record free-text must NEVER land here").
  * ``answer`` -- ``{"answer": str}``, the final verified answer text. This
    harness NEVER writes that raw string to any committed artifact -- only
    a SHA-256 hash and a character length survive into ``DrawRecord``
    (matching the #163 artifact-discipline precedent: "counts/enums/hashes/
    booleans only").
  * ``verification`` -- ``{"verdict": str, "segments": [...], "warnings":
    {...}}``. Per ``app/rendering.py``'s own module docstring, each
    ``segments`` entry is one-to-one with the underlying ``ClaimCheckResult``
    in order: a claim whose citations ALL passed survives as a
    ``{"type": "claim", ...}`` segment (carrying its citations); a claim
    that failed ANY citation is stripped and replaced with a constant
    ``{"type": "notice", "text": "Not found in record."}`` segment --
    "Notice text ... no differentiation by CitationStatus" (that
    module's own words). This means the wire protocol gives us, per claim,
    exactly a pass/fail boolean plus (for passed claims) how many citations
    it carried -- NOT the finer-grained ``CitationStatus`` enum
    (``valid``/``unknown_record``/``value_mismatch``/...), which is a
    server-internal detail the UI/wire contract deliberately never surfaces
    (see ``app/rendering.py``'s "Notice text" section for why: a per-reason
    notice would leak information about *why* a claim failed). This
    harness's ``ClaimSegmentRecord`` records exactly what the wire gives:
    ``passed`` + ``citation_count``, no more, no less. A reader who needs
    the finer enum must go look downstream, in #154 or #163's harnesses,
    which DO reach ``ClaimCheckResult`` directly (in-process).

**Mechanism attribution -- the entire point.** For a fixed question,
grouping draws by their exact ``tool_sequence`` (the ordered tuple of tool
names dispatched) and, within a sequence, by their ``answer_hash``, gives a
three-way partition of observed instability:

  * **SAME tool sequence + SAME answer hash + DIFFERENT verdict** ->
    ``downstream`` variance -- the planner's live behavior was fully
    reproducible (same tools, same words), yet extraction/verification
    itself produced a different whole-answer verdict. This is the layer
    #154 already measures in isolation (with pinned tool data); seeing it
    recur here, with REAL tool data, corroborates #154's downstream
    diagnosis rather than contradicting it.
  * **SAME tool sequence + DIFFERENT answer hash** -> ``generation
    nondeterminism`` at the planner's answer-composition step (already
    observed once on llama.cpp per issue #160's own text) -- same tools
    dispatched, same inputs, yet the model composed different words.
  * **DIFFERENT tool sequence across draws** -> ``tool-selection variance``
    -- the planner itself chose different tools (or a different order) for
    semantically identical turns, a layer #154's fixture can never
    exercise at all (its registry always returns the SAME canned payload
    regardless of what -- or whether -- the planner would have called live).

``attribute_mechanisms`` (pure, no I/O) computes all three flags plus the
full grouped distributions from a list of already-collected ``DrawRecord``s
-- see its own docstring for the exact grouping rule. These three
mechanisms are not mutually exclusive within one question's N draws; the
report surfaces all three independently rather than picking one "primary"
cause.

**Auth.** ``app.chat._default_token_validator`` (the flag-OFF default,
``copilot_per_user_token_enabled=False`` -- this dev stack's setting)
accepts any non-empty bearer token; no real OAuth dance is needed to POST
``/chat`` itself (a SEPARATE, already-bootstrapped dev-token bridge --
``scripts/bootstrap-copilot-dev-client.sh`` -- supplies the REAL OpenEMR
token the AGENT's own tool calls use server-side; that is a prerequisite
this harness assumes is already run, not something it does itself).

**Usage -- MUST run inside ``development-easy-agent-1`` (same trap #154
documents: ``copilot_internal`` is ``internal: true``, no host ports).**
Copy a fresh ``evals/runner`` + ``services/copilot-agent/app`` in first
(image-baked ``app`` may be stale -- #140) even though this module itself
never imports ``app.*``; the sibling `#154`/`#163` harnesses under the same
``evals/runner/`` package do, and this harness has no reason to duplicate
that copy step's documentation here. Then, from the repo root:

    bash scripts/bootstrap-copilot-dev-client.sh   # one-time per agent container
    python evals/fixtures/seed.py                  # confirms patients 1/2 exist
    docker exec development-easy-agent-1 mkdir -p /data/repo_ingest/evals/runner
    docker cp evals/runner development-easy-agent-1:/data/repo_ingest/evals/runner
    docker exec -w /data/repo_ingest development-easy-agent-1 \
        python evals/runner/issue_160_live_chat_harness.py --draws 8

(``httpx`` is already present in the image -- it's ``app.chat``'s/
``app.openemr_client``'s own HTTP client dependency.) The script talks to
``http://localhost:8000/chat`` -- the agent's OWN uvicorn bind, reachable
from inside its own container without going through any docker network at
all (confirmed live: ``uvicorn app.main:app --host 0.0.0.0 --port 8000`` is
the container's PID 1 command).

To re-aggregate an existing ``draws/`` output without a live run:

    python evals/runner/issue_160_live_chat_harness.py --summarize-only

**Operational gotcha -- ``docker cp`` back INTO the container leaves
root-owned files, breaking the NEXT run (confirmed live, #160 16-more-draws
follow-up).** A committed ``evals/results/issue-160/*.jsonl`` copied out to
the host for a git commit and later copied back IN (``docker cp <file>
development-easy-agent-1:/data/repo_ingest/evals/results/issue-160/<file>``
-- e.g. to seed a second batch's ``append_draw`` calls onto the first
batch's committed history) lands root-owned inside the container, because
``docker cp``'s write runs as the container's root user regardless of which
user the container's own process (``appuser``, per the entrypoint's
``HOST_UID`` adoption) runs as. The harness process itself runs AS
``appuser`` and cannot open a root-owned file in append mode --
``append_draw``'s ``path.open("a", ...)`` raises ``PermissionError``,
**silently from a polling script's point of view**: the file's line count
simply never grows again, which looks identical to "the process is just
slow," not "the process crashed on its very first write." This is exactly
what happened live: a session-2 batch launched against freshly-``docker
cp``'d-back JSONLs crashed on its first ``append_draw`` call with zero
draws written (confirmed after the fact by ``session_id`` bucket counts --
see ``by_session`` in a committed ``report.json``: a session that crashed
before its first successful append simply never appears as a key at all,
so there is no partial/mixed data to clean up, only a fully-absent batch
to re-run). **Fix: ``docker exec development-easy-agent-1 chown appuser:
appuser <path>`` on every file copied back in, BEFORE relaunching** -- do
this as routine practice after any ``docker cp ... development-easy-
agent-1:...`` (copy IN), not only when a crash is already suspected.

**Artifacts.** Per question, ``evals/results/issue-160/<question_id>.jsonl``
-- one ``DrawRecord`` (as JSON) appended per draw, immediately after that
draw completes (crash-safe, same discipline as #154's per-draw files). A
final ``evals/results/issue-160/report.json`` aggregates every question's
verdict distribution, tool-sequence distribution, answer-hash distribution,
latency stats, and the three-way mechanism attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

_EVALS_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _EVALS_ROOT / "results" / "issue-160"

_DEFAULT_DRAWS = 8
_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_TOKEN = "issue-160-harness-draw"  # nonzero-length is all the flag-OFF stub validator checks


# --- question targets (#149, #150) -------------------------------------------


@dataclass(frozen=True)
class Question:
    id: str
    patient_id: int
    question: str


BP_QUESTION = Question(
    id="issue-149-bp",
    patient_id=1,
    question="What was his last blood pressure reading, and what category does that fall into?",
)

ALLERGY_QUESTION = Question(
    id="issue-150-allergy",
    patient_id=2,
    question="Is there an allergy conflict with the current medications?",
)

TARGET_QUESTIONS = (BP_QUESTION, ALLERGY_QUESTION)


# --- draw-record schema -------------------------------------------------------


@dataclass(frozen=True)
class ToolCallRecord:
    order: int
    tool: str
    args: dict[str, Any]
    error: str | None


@dataclass(frozen=True)
class ClaimSegmentRecord:
    """One ``verification`` segment. ``passed`` mirrors the SSE wire's own
    claim/notice split (see module docstring); ``citation_count`` is 0 for a
    stripped (notice) segment by construction -- a notice carries no
    citations at all."""

    index: int
    passed: bool
    citation_count: int


@dataclass(frozen=True)
class DrawRecord:
    """One live ``POST /chat`` draw. Deliberately carries no raw answer or
    claim TEXT anywhere -- only a hash/length/counts/enums/booleans (#163
    artifact-discipline parity).

    ``session_id`` (#160 follow-up: 16-more-draws sign-off) identifies which
    booted-stack run this draw came from -- so a second batch of draws never
    silently mixes into the first without a visible label. Two batches
    against the SAME question produce draws that differ only in
    ``session_id``/``draw_index``; ``attribute_mechanisms``/
    ``summarize_question`` are session-agnostic (a caller decides whether to
    pass them one session's draws or several combined -- see
    ``build_report``'s ``combined``/``by_session`` split)."""

    question_id: str
    patient_id: int
    draw_index: int
    session_id: str
    correlation_id: str | None
    conversation_id: str | None
    tool_calls: list[ToolCallRecord]
    answer_hash: str | None
    answer_length: int | None
    claim_count: int | None
    claims: list[ClaimSegmentRecord]
    verdict: str | None
    latency_seconds: float
    http_status: int | None
    error: str | None


# --- pure SSE parsing ----------------------------------------------------------


def parse_sse_lines(lines: Iterable[str]) -> list[tuple[str, dict[str, Any]]]:
    """Turn raw SSE text lines (as ``httpx``'s ``Response.iter_lines()``
    yields them, or a literal fixture list in tests) into an ordered list of
    ``(event_name, data)`` pairs -- pure, no network. Mirrors
    ``app/chat.py``'s own ``_sse`` writer's wire format
    (``event: <name>\\ndata: <json>\\n\\n``): an ``event:`` line sets the
    pending event name, the following ``data:`` line's JSON payload closes
    it out. Blank lines and any other line shape (e.g. SSE comment lines
    starting with ``:``) are ignored, matching how a real SSE client would
    skip them."""
    events: list[tuple[str, dict[str, Any]]] = []
    pending_event: str | None = None
    for line in lines:
        if line.startswith("event:"):
            pending_event = line[len("event:") :].strip()
        elif line.startswith("data:") and pending_event is not None:
            payload = json.loads(line[len("data:") :].strip())
            events.append((pending_event, payload))
            pending_event = None
    return events


# --- pure draw-record construction --------------------------------------------


def build_draw_record(
    *,
    question: Question,
    draw_index: int,
    session_id: str,
    events: list[tuple[str, dict[str, Any]]],
    latency_seconds: float,
    http_status: int | None,
    error: str | None,
) -> DrawRecord:
    """Turn one draw's parsed SSE events into a :class:`DrawRecord`. Pure --
    no network, no hashing side effects beyond ``hashlib`` on already-
    in-memory text (never persisted). ``error`` non-``None`` means the HTTP
    call itself failed or never completed; every downstream field is then
    left ``None``/empty rather than guessing. ``session_id`` is caller-
    supplied (one value per live-run invocation -- see ``main()``), never
    derived here."""
    if error is not None:
        return DrawRecord(
            question_id=question.id,
            patient_id=question.patient_id,
            draw_index=draw_index,
            session_id=session_id,
            correlation_id=None,
            conversation_id=None,
            tool_calls=[],
            answer_hash=None,
            answer_length=None,
            claim_count=None,
            claims=[],
            verdict=None,
            latency_seconds=latency_seconds,
            http_status=http_status,
            error=error,
        )

    correlation_id: str | None = None
    conversation_id: str | None = None
    tool_calls: list[ToolCallRecord] = []
    answer_hash: str | None = None
    answer_length: int | None = None
    verdict: str | None = None
    claims: list[ClaimSegmentRecord] = []

    for event_name, data in events:
        if event_name == "conversation":
            conversation_id = data.get("conversation_id")
            correlation_id = data.get("correlation_id")
        elif event_name == "tool_call":
            tool_calls.append(
                ToolCallRecord(
                    order=len(tool_calls),
                    tool=data.get("tool", ""),
                    args=data.get("args", {}) or {},
                    error=data.get("error"),
                )
            )
        elif event_name == "answer":
            answer_text = data.get("answer", "")
            answer_hash = hashlib.sha256(answer_text.encode("utf-8")).hexdigest()
            answer_length = len(answer_text)
        elif event_name == "verification":
            verdict = data.get("verdict")
            for index, segment in enumerate(data.get("segments", [])):
                passed = segment.get("type") == "claim"
                citation_count = len(segment.get("citations", [])) if passed else 0
                claims.append(ClaimSegmentRecord(index=index, passed=passed, citation_count=citation_count))

    claim_count = len(claims) if events else None

    return DrawRecord(
        question_id=question.id,
        patient_id=question.patient_id,
        draw_index=draw_index,
        session_id=session_id,
        correlation_id=correlation_id,
        conversation_id=conversation_id,
        tool_calls=tool_calls,
        answer_hash=answer_hash,
        answer_length=answer_length,
        claim_count=claim_count,
        claims=claims,
        verdict=verdict,
        latency_seconds=latency_seconds,
        http_status=http_status,
        error=None,
    )


# --- tool-sequence helpers -----------------------------------------------------


def tool_sequence_of(draw: DrawRecord) -> tuple[str, ...]:
    return tuple(tc.tool for tc in draw.tool_calls)


def sequence_label(sequence: tuple[str, ...]) -> str:
    return " -> ".join(sequence) if sequence else "()"


# --- three-way mechanism attribution -------------------------------------------


def attribute_mechanisms(draws: list[DrawRecord]) -> dict[str, Any]:
    """The entire point of this harness (see module docstring). Groups
    non-error draws by exact ``tool_sequence``, then within each sequence by
    ``answer_hash``, and reports the three independent variance flags:

      * ``tool_selection_variance`` -- True iff more than one distinct
        ``tool_sequence`` was observed across draws.
      * ``generation_nondeterminism`` -- True iff any ONE ``tool_sequence``
        produced more than one distinct ``answer_hash``.
      * ``downstream_variance`` -- True iff any ONE (``tool_sequence``,
        ``answer_hash``) pair produced more than one distinct ``verdict``.

    These are computed independently (not mutually exclusive) -- a single
    question's draws can trip more than one flag at once. Error draws
    (``DrawRecord.error is not None``) are counted in ``n_errors`` but
    excluded from every grouping (they carry no tool sequence, answer hash,
    or verdict to group by)."""
    errors = [d for d in draws if d.error is not None]
    counted = [d for d in draws if d.error is None]

    by_sequence: dict[str, list[DrawRecord]] = defaultdict(list)
    for draw in counted:
        by_sequence[sequence_label(tool_sequence_of(draw))].append(draw)

    tool_sequence_distribution = {seq: len(group) for seq, group in by_sequence.items()}

    answer_hash_distribution: dict[str, dict[str, int]] = {}
    verdict_distribution_by_sequence_hash: dict[str, dict[str, dict[str, int]]] = {}
    generation_nondeterminism_sequences: list[str] = []
    downstream_variance_keys: list[list[str]] = []

    for seq, group in by_sequence.items():
        hash_counts: Counter[str] = Counter(d.answer_hash or "<no-answer>" for d in group)
        answer_hash_distribution[seq] = dict(hash_counts)
        if len(hash_counts) > 1:
            generation_nondeterminism_sequences.append(seq)

        by_hash: dict[str, list[DrawRecord]] = defaultdict(list)
        for draw in group:
            by_hash[draw.answer_hash or "<no-answer>"].append(draw)

        verdict_distribution_by_sequence_hash[seq] = {}
        for answer_hash, hash_group in by_hash.items():
            verdict_counts: Counter[str] = Counter(d.verdict or "<no-verdict>" for d in hash_group)
            verdict_distribution_by_sequence_hash[seq][answer_hash] = dict(verdict_counts)
            if len(verdict_counts) > 1:
                downstream_variance_keys.append([seq, answer_hash])

    verdict_distribution: Counter[str] = Counter(d.verdict or "<no-verdict>" for d in counted)

    return {
        "n_draws": len(draws),
        "n_errors": len(errors),
        "n_counted": len(counted),
        "distinct_tool_sequences": len(by_sequence),
        "tool_sequence_distribution": tool_sequence_distribution,
        "answer_hash_distribution": answer_hash_distribution,
        "verdict_distribution": dict(verdict_distribution),
        "verdict_distribution_by_sequence_hash": verdict_distribution_by_sequence_hash,
        "mechanism": {
            "tool_selection_variance": len(by_sequence) > 1,
            "generation_nondeterminism": len(generation_nondeterminism_sequences) > 0,
            "downstream_variance": len(downstream_variance_keys) > 0,
            "generation_nondeterminism_sequences": generation_nondeterminism_sequences,
            "downstream_variance_keys": downstream_variance_keys,
        },
    }


def summarize_question(question_id: str, draws: list[DrawRecord]) -> dict[str, Any]:
    """Per-question report row: mechanism attribution plus latency stats.
    Latency is measured over EVERY draw (including errors -- a timeout still
    took wall-clock time and is real data, not excluded)."""
    report = attribute_mechanisms(draws)
    latencies = [d.latency_seconds for d in draws]
    report["question_id"] = question_id
    report["latency_seconds"] = {
        "min": min(latencies) if latencies else None,
        "max": max(latencies) if latencies else None,
        "mean": mean(latencies) if latencies else None,
    }
    return report


def group_by_session(draws: list[DrawRecord]) -> dict[str, list[DrawRecord]]:
    """Split draws by ``session_id`` -- preserves first-seen session order
    (not sorted) so ``by_session`` in the report reads oldest-session-first."""
    grouped: dict[str, list[DrawRecord]] = {}
    for draw in draws:
        grouped.setdefault(draw.session_id, []).append(draw)
    return grouped


def build_report(per_question: dict[str, list[DrawRecord]]) -> dict[str, Any]:
    """Per question: a ``combined`` summary over ALL draws (every session
    mixed) plus a ``by_session`` breakdown (one summary per distinct
    ``session_id``, in first-seen order) and the explicit ``session_ids``
    list -- so a reader can never mistake a multi-session combined number for
    a single-session one, and can never miss that more than one session's
    draws went into ``combined`` (#160 16-more-draws follow-up: "no silent
    mixing")."""
    report: dict[str, Any] = {}
    for question_id, draws in per_question.items():
        by_session = group_by_session(draws)
        report[question_id] = {
            "session_ids": list(by_session.keys()),
            "combined": summarize_question(question_id, draws),
            "by_session": {
                session_id: summarize_question(question_id, session_draws)
                for session_id, session_draws in by_session.items()
            },
        }
    return report


# --- live run (network I/O -- never exercised by the unit-test suite) ---------


def _draws_path(question_id: str) -> Path:
    return _RESULTS_DIR / f"{question_id}.jsonl"


def append_draw(draw: DrawRecord) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _draws_path(draw.question_id)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(draw)) + "\n")
    return path


def load_draws(question_id: str) -> list[DrawRecord]:
    path = _draws_path(question_id)
    if not path.exists():
        return []
    draws: list[DrawRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        tool_calls = [ToolCallRecord(**tc) for tc in payload["tool_calls"]]
        claims = [ClaimSegmentRecord(**c) for c in payload["claims"]]
        payload = {**payload, "tool_calls": tool_calls, "claims": claims}
        draws.append(DrawRecord(**payload))
    return draws


def post_chat_draw(
    client: Any,
    *,
    base_url: str,
    question: Question,
    draw_index: int,
    session_id: str,
    token: str,
) -> DrawRecord:
    """Make ONE real, live ``POST /chat`` call and turn its SSE response into
    a :class:`DrawRecord`. ``client`` is an ``httpx.Client`` (typed ``Any``
    here so this module never imports ``httpx`` at module scope -- the
    no-tool-stub test asserts this module's import list stays limited to
    what pure logic needs; ``httpx`` is imported lazily inside ``main()``
    instead, alongside every other live-run-only dependency). Never raises --
    a request exception is caught and turned into an error :class:`DrawRecord`
    so one bad draw does not abort the whole run (same discipline as #154's
    ``run_one_draw``)."""
    payload = {"message": question.question, "patient_id": question.patient_id}
    headers = {"Authorization": f"Bearer {token}"}
    started = time.monotonic()
    try:
        with client.stream("POST", f"{base_url}/chat", json=payload, headers=headers) as response:
            status = response.status_code
            if status != 200:
                response.read()
                latency = time.monotonic() - started
                return build_draw_record(
                    question=question,
                    draw_index=draw_index,
                    session_id=session_id,
                    events=[],
                    latency_seconds=latency,
                    http_status=status,
                    error=f"HTTP {status}: {response.text[:500]}",
                )
            events = parse_sse_lines(response.iter_lines())
        latency = time.monotonic() - started
        return build_draw_record(
            question=question,
            draw_index=draw_index,
            session_id=session_id,
            events=events,
            latency_seconds=latency,
            http_status=status,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 -- harness run: record failure, keep going
        latency = time.monotonic() - started
        return build_draw_record(
            question=question,
            draw_index=draw_index,
            session_id=session_id,
            events=[],
            latency_seconds=latency,
            http_status=None,
            error=repr(exc),
        )


def main() -> None:
    import uuid

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=_DEFAULT_DRAWS, help="draws per question (default 8)")
    parser.add_argument("--start-index", type=int, default=0, help="starting draw_index for this session (default 0; bump for a second batch so indices don't collide)")
    parser.add_argument("--base-url", default=_DEFAULT_BASE_URL, help="agent base URL (default in-container localhost:8000)")
    parser.add_argument("--token", default=_DEFAULT_TOKEN, help="bearer token (flag-OFF stub validator accepts any non-empty value)")
    parser.add_argument("--session-id", default=None, help="label for this run's draws (default: a fresh random id) -- distinguishes this batch from any prior batch appended to the same *.jsonl, so report.json's by_session never silently mixes them")
    parser.add_argument("--summarize-only", action="store_true", help="skip live runs; just re-aggregate evals/results/issue-160/*.jsonl")
    args = parser.parse_args()

    if not args.summarize_only:
        import httpx  # live-run-only dependency -- see post_chat_draw's docstring

        session_id = args.session_id or f"session-{uuid.uuid4().hex[:12]}"
        print(f"[issue-160] session_id={session_id}")

        for question in TARGET_QUESTIONS:
            for draw_index in range(args.start_index, args.start_index + args.draws):
                with httpx.Client(timeout=180.0) as client:
                    draw = post_chat_draw(
                        client,
                        base_url=args.base_url,
                        question=question,
                        draw_index=draw_index,
                        session_id=session_id,
                        token=args.token,
                    )
                path = append_draw(draw)
                status = f"ERROR {draw.error}" if draw.error else f"verdict={draw.verdict} tools={tool_sequence_of(draw)}"
                print(f"[issue-160] {question.id} draw {draw_index} ({session_id}): {status} -> {path}")

    per_question = {q.id: load_draws(q.id) for q in TARGET_QUESTIONS}
    report = build_report(per_question)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _RESULTS_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
