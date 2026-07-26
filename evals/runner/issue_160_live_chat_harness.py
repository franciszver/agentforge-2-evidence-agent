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
    record free-text must NEVER land here"). ``args`` itself is still
    persisted VERBATIM with no allowlist/redaction (gate-2 security
    advisory, #160) -- safe only because ``TARGET_QUESTIONS``' tools are
    all argless today; extending the question set to any tool whose args
    can carry free-text/PHI-adjacent input requires adding a redaction pass
    first (see the code comment at ``build_draw_record``'s ``tool_call``
    branch).
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

**Interpretation of a committed run -- read this before citing a
``report.json`` number (gate-3/Opus MAJOR 1, corrected framing).** The
first committed run (``evals/results/issue-160/report.json``, 48 draws
across two sessions, 24 per question) found BOTH questions
``partially_verified`` on literally every draw -- the BP question answered
the blood pressure reading every time, the allergy question stripped
exactly the same 2 claims every time. **This outcome regime matches
NEITHER #149's observed mixture (1/4 answered, 3/4 declined) NOR #150's
(5 verified / 1 blocked across 6 draws).** A prior draft of this docstring
argued that non-observation of variance at N=24 was reasonably strong
evidence the ORIGINAL 1-in-6-ish instability was "still present but
unluckily not observed" -- citing a binomial miss-probability calculation
((5/6)^24 ~= 1.3%). **That argument is invalid, and the data itself is why:
a (5/6)^24 miss-probability calculation only supports "not reproduced, low
power to rule out" if the underlying MIXTURE (draws split between two or
more distinct outcomes) is still what would be sampled from.** This run
did not observe a mixture that happened to land on one side 24 times running
-- it observed no mixture at all, a completely different, single, stable
outcome from anything #149/#150 recorded. Between when #149/#150 were filed
and this branch's base commit, several changes plausibly touch exactly this
behavior: #154's retry-exhaustion instrumentation and its N=8
downstream-stability measurement (0/16 extraction failures, stable
verdicts -- see ``issue_154_stability_harness.py``), and the judge-context/
``answer_pre_notice`` fixes referenced in #149/#150's own follow-up
discussion. **The supported claim is:** on this build, config, and warm
stack, via the API path, both questions are perfectly deterministic across
24 draws each, AND the deterministic outcome observed differs from
anything #149 or #150 recorded -- i.e. the originally-reported behavior
appears **superseded by intervening changes, not merely unluckily
unobserved**. This is evidence FOR a fix having landed, not proof of one
(a single warm-stack session, one build, no controlled before/after
comparison -- see the next paragraph for a related power caveat). Do not
re-cite the (5/6)^24 framing from this run without re-deriving whether it
still applies to whatever NEW data you're looking at.

**Warm-slot caveat (gate-3/Opus MAJOR 3) -- read this before trusting a
"stable" result.** The first two committed batches ran with the DEFAULT
ordering at the time: every draw of one question, back-to-back, before
moving to the next question -- i.e. all 24 BP draws hit the SAME warm
inference slot consecutively, then all 24 allergy draws did. temp=0,
``--parallel`` effectively 1 (one draw in flight at a time, one process,
one model load). This ordering is near-zero-power against ANY
nondeterminism that only shows up on a cold/reloaded slot, a
context-window-boundary effect, or state that drifts only across a longer
gap between calls -- by construction, this loop never produces that gap
within one question's run. ``main()``'s default is now ``--interleave``
(question A draw 0, question B draw 0, question A draw 1, ... -- see its
own ``argparse`` help text): still temp=0, still one process, but
consecutive calls never share a question, which at least breaks up a
single unbroken run of identical-question calls. This does NOT eliminate
the warm-slot limitation (it's still one continuously-warm process/model
load for the whole run) -- it only weakens the SPECIFIC "same question
twenty-four times in a row" pattern. A genuinely fresh-boot-per-batch
comparison (rule out warm-state suppression entirely) is a further,
NOT-yet-run step -- see the report's own written interpretation for
whether it's warranted.

**Auth.** #168 (VULN-0001) made the flag-OFF default (
``copilot_per_user_token_enabled=False`` -- this dev stack's setting)
fail-closed: every bearer token is now rejected UNLESS
``copilot_dev_accept_any_bearer_token`` is also set. This dev stack's
compose file sets it, so ``app.chat._dev_permissive_token_validator`` is
active and any non-empty bearer token still works; no real OAuth dance is
needed to POST ``/chat`` itself (a SEPARATE, already-bootstrapped dev-token
bridge -- ``scripts/bootstrap-copilot-dev-client.sh`` -- supplies the REAL
OpenEMR token the AGENT's own tool calls use server-side; that is a
prerequisite this harness assumes is already run, not something it does
itself).

**Usage -- MUST run inside ``development-easy-agent-1`` (same trap #154
documents: ``copilot_internal`` is ``internal: true``, no host ports).**
Copy a fresh ``evals/runner`` + ``services/copilot-agent/app`` in first
(image-baked ``app`` may be stale -- #140) even though this module itself
never imports ``app.*``; the sibling `#154`/`#163` harnesses under the same
``evals/runner/`` package do, and this harness has no reason to duplicate
that copy step's documentation here. Then, from the repo root:

    bash scripts/bootstrap-copilot-dev-client.sh   # one-time per agent container
    python evals/fixtures/seed.py                  # confirms patients 1/2 exist
    docker exec development-easy-agent-1 mkdir -p /data/repo_ingest/evals
    docker cp evals/runner development-easy-agent-1:/data/repo_ingest/evals/
    docker exec -w /data/repo_ingest development-easy-agent-1 \
        python evals/runner/issue_160_live_chat_harness.py --draws 8

(Gate-3/Opus MINOR 1: the destination directory must be ``.../evals``, NOT
``.../evals/runner`` -- pre-creating ``.../evals/runner`` and THEN
``docker cp``-ing the ``evals/runner`` source directory into it lands the
contents nested one level too deep, ``.../evals/runner/runner/...``, and
silently leaves whatever was already at ``.../evals/runner`` stale/
unreplaced underneath. ``docker cp SRC DST/`` with a trailing-slash,
already-existing ``DST`` and a non-existent ``DST/basename(SRC)`` is what
correctly lands the copy at ``.../evals/runner``.)

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
latency stats, and the three-way mechanism attribution -- each gated by
``run_valid`` (gate-3/Opus MINOR 3: ``null``, not ``False``, when the run
had any error or incomplete draw -- see ``attribute_mechanisms``'s
docstring) -- plus a top-level ``run_metadata.sessions`` block (gate-3/Opus
MAJOR 2, restructured PER-SESSION per gate-3/Opus DEFECT 2 re-review: a
multi-session report has no single "the flags"/"the loop order" to report
at the top level, since different sessions may genuinely differ). Each
session's entry merges its observed wall-clock window
(``compute_session_windows``, from that session's own draws) with its
PERSISTED provenance (app git SHA, engine/model, the four feature-flag
booleans, loop order/parallelism/temperature, ``backfilled`` --
``build_run_metadata``'s docstring has the full field list). That
provenance lives durably in a SIDECAR file,
``evals/results/issue-160/session_metadata.json`` (``load_session_
metadata``/``record_session_provenance``), written ONCE by the live run
that produced each session and never re-derived by a later
``--summarize-only`` regeneration -- a session absent from the sidecar
shows ``loop_order: "unknown (no live run recorded for this session ...)"``
rather than a guessed value stamped from whatever's currently running."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
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
    ``build_report``'s ``combined``/``by_session`` split).

    ``started_at`` and ``orphan_data_line_count`` (gate-3/Opus MAJOR 2 and
    MINOR 3): added after the first two committed batches (#160
    16-more-draws follow-up). ``None`` on those pre-existing draws means
    GENUINELY UNKNOWN -- not "computed as zero/empty" -- because no run
    metadata existed at the time to derive them from; the migration that
    added these fields to the committed artifact left both explicitly
    ``null`` rather than inventing a value (see ``report.json``'s
    ``run_metadata.sessions[<session_id>].backfilled``). Every draw from a
    run using this schema version onward gets a real value for both."""

    question_id: str
    patient_id: int
    draw_index: int
    session_id: str
    started_at: str | None
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
    orphan_data_line_count: int | None
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


def count_orphan_data_lines(lines: Iterable[str]) -> int:
    """Count ``data:`` lines that ``parse_sse_lines`` silently drops because
    no ``event:`` line preceded them (gate-3/Opus MINOR 3). A well-formed
    SSE stream from ``app/chat.py``'s own ``_sse`` writer never produces
    this -- every ``data:`` line is always immediately preceded by its own
    ``event:`` line, one pair per frame -- so a nonzero count here would
    mean either a malformed/truncated response or a parsing assumption this
    module doesn't yet handle, either of which the reader should know about
    rather than have silently swallowed. Mirrors ``parse_sse_lines``'s exact
    skip condition (same ``pending_event is not None`` gate) so the two
    functions can never disagree about what counts as orphaned. Pure --
    accepts the same raw line iterable ``parse_sse_lines`` does; a caller
    that wants both must materialize ``lines`` once and pass the same list
    to each (see ``post_chat_draw``)."""
    orphan_count = 0
    pending_event: str | None = None
    for line in lines:
        if line.startswith("event:"):
            pending_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            if pending_event is None:
                orphan_count += 1
            else:
                pending_event = None
    return orphan_count


# --- pure draw-record construction --------------------------------------------


def build_draw_record(
    *,
    question: Question,
    draw_index: int,
    session_id: str,
    started_at: str | None,
    events: list[tuple[str, dict[str, Any]]],
    latency_seconds: float,
    http_status: int | None,
    orphan_data_line_count: int | None,
    error: str | None,
) -> DrawRecord:
    """Turn one draw's parsed SSE events into a :class:`DrawRecord`. Pure --
    no network, no hashing side effects beyond ``hashlib`` on already-
    in-memory text (never persisted). ``error`` non-``None`` means the HTTP
    call itself failed or never completed; every downstream field is then
    left ``None``/empty rather than guessing. ``session_id``/``started_at``/
    ``orphan_data_line_count`` are all caller-supplied (computed once per
    draw by ``post_chat_draw`` -- see its docstring), never derived here."""
    if error is not None:
        return DrawRecord(
            question_id=question.id,
            patient_id=question.patient_id,
            draw_index=draw_index,
            session_id=session_id,
            started_at=started_at,
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
            orphan_data_line_count=orphan_data_line_count,
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
            # SECURITY/PHI (gate-2 advisory, #160): args is persisted
            # VERBATIM, no allowlist or redaction. Currently benign only
            # because both TARGET_QUESTIONS' tools (get_vitals,
            # get_medications, get_allergies) are argless -- grep-confirmed
            # every committed args value under evals/results/issue-160/ is
            # {}. This does NOT generalize: before pointing this harness at
            # any tool whose args can carry free-text or PHI-adjacent input
            # (a search query, a date-range note, a name lookup, ...), add
            # an explicit allowlist/redaction pass here first -- an
            # unredacted args dict would otherwise land verbatim in a
            # committed, public-repo JSONL artifact.
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

    # Ambiguity accepted, not fixed (gate-1 simplify note): a successful
    # (error=None) draw whose ``events`` list is empty collapses claim_count
    # to None -- the same value an error draw uses -- so this field alone
    # cannot distinguish "0 events because nothing was parsed" from "0
    # events because the call itself failed." In practice this case is
    # unreachable through the live wire protocol: every real /chat response
    # always emits at least a ``conversation`` frame before anything else
    # (see app/chat.py's _stream_chat), so error=None with events=[] never
    # happens from post_chat_draw -- only a caller constructing a
    # DrawRecord by hand (as this module's own tests do for isolated
    # fixtures) could hit it. A reader who needs to be certain should check
    # ``error`` first regardless: it is already the authoritative signal for
    # "did the HTTP call itself fail," independent of this field.
    claim_count = len(claims) if events else None

    return DrawRecord(
        question_id=question.id,
        patient_id=question.patient_id,
        draw_index=draw_index,
        session_id=session_id,
        started_at=started_at,
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
        orphan_data_line_count=orphan_data_line_count,
        error=None,
    )


# --- tool-sequence helpers -----------------------------------------------------


def tool_sequence_of(draw: DrawRecord) -> tuple[str, ...]:
    return tuple(tc.tool for tc in draw.tool_calls)


def sequence_label(sequence: tuple[str, ...]) -> str:
    return " -> ".join(sequence) if sequence else "()"


# --- three-way mechanism attribution -------------------------------------------


def is_run_valid(draws: list[DrawRecord]) -> bool:
    """Gate-3/Opus MINOR 3: a run is "valid" (its mechanism flags trustworthy)
    iff there were zero errors AND every draw produced a real ``answer_hash``
    and ``verdict``. An invalid run is NOT bad data -- errors are data, never
    discarded (see the module docstring's live-run discipline) -- it just
    means the three-way mechanism BOOLEANS below cannot be read as a clean
    signal: an error draw contributes no tool_sequence/answer_hash/verdict
    to attribute against, silently shrinking the N actually compared. See
    ``attribute_mechanisms`` for how this gates the ``mechanism`` dict."""
    if any(d.error is not None for d in draws):
        return False
    return all(d.answer_hash is not None and d.verdict is not None for d in draws)


def count_duplicate_draw_indices(draws: list[DrawRecord]) -> int:
    """Gate-3/Opus MINOR 2: counts ``(session_id, draw_index)`` pairs that
    appear more than once -- a signal of an accidental relaunch/re-run
    overwriting-in-effect the same slot within one session's own index
    space. Two DIFFERENT sessions legitimately reusing the same
    ``draw_index`` (e.g. both starting at 0) is NOT counted here -- only a
    repeat WITHIN one session is, since that is the case that actually
    indicates a bug (a launch re-running indices it already covered)."""
    pair_counts = Counter((d.session_id, d.draw_index) for d in draws)
    return sum(1 for count in pair_counts.values() if count > 1)


def aggregate_orphan_data_lines(draws: list[DrawRecord]) -> dict[str, int]:
    """Gate-3/Opus MINOR 3: sums ``DrawRecord.orphan_data_line_count`` across
    draws where it's known, and separately counts how many draws don't know
    it at all (``None`` -- pre-schema/backfilled draws, see ``DrawRecord``'s
    own docstring). Keeping these separate means a reader never mistakes
    "0 known orphans, but we didn't check most draws" for "checked
    everything, found zero.\""""
    known = [d.orphan_data_line_count for d in draws if d.orphan_data_line_count is not None]
    unknown = sum(1 for d in draws if d.orphan_data_line_count is None)
    return {"total": sum(known), "draws_with_unknown_count": unknown}


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
    or verdict to group by).

    **Gate-3/Opus MINOR 3 -- ``run_valid`` gates the three booleans above.**
    If ``is_run_valid(draws)`` is ``False`` (any error draw, or any counted
    draw missing ``answer_hash``/``verdict``), ``tool_selection_variance``/
    ``generation_nondeterminism``/``downstream_variance`` are all ``None``
    (JSON ``null``), not ``False`` -- an invalid run must never render as
    "checked, found stable." The distribution dicts/lists are still
    computed and returned regardless (raw data, not a trust claim)."""
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

    run_valid = is_run_valid(draws)
    # Gate-3/Opus MINOR 3: when the run is NOT valid (any error, or any
    # counted draw missing answer_hash/verdict), the three mechanism
    # BOOLEANS report None/null rather than False -- "we observed no
    # variance" and "we can't trust what we observed" must never render
    # identically. The underlying distributions/lists above are left
    # exactly as computed either way -- they're raw counts, not a verdict
    # on trustworthiness, and a reader who wants to inspect them despite an
    # invalid run still can.
    tool_selection_variance: bool | None = len(by_sequence) > 1 if run_valid else None
    generation_nondeterminism: bool | None = len(generation_nondeterminism_sequences) > 0 if run_valid else None
    downstream_variance: bool | None = len(downstream_variance_keys) > 0 if run_valid else None

    return {
        "n_draws": len(draws),
        "n_errors": len(errors),
        "n_counted": len(counted),
        "run_valid": run_valid,
        "n_duplicate_indices": count_duplicate_draw_indices(draws),
        "orphan_data_lines": aggregate_orphan_data_lines(draws),
        "distinct_tool_sequences": len(by_sequence),
        "tool_sequence_distribution": tool_sequence_distribution,
        "answer_hash_distribution": answer_hash_distribution,
        "verdict_distribution": dict(verdict_distribution),
        "verdict_distribution_by_sequence_hash": verdict_distribution_by_sequence_hash,
        "mechanism": {
            "tool_selection_variance": tool_selection_variance,
            "generation_nondeterminism": generation_nondeterminism,
            "downstream_variance": downstream_variance,
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


def compute_session_windows(draws: list[DrawRecord]) -> dict[str, dict[str, str | None]]:
    """Derive each session's observed wall-clock window from its OWN draws'
    ``started_at`` (gate-3/Opus MAJOR 2) -- no external run-time bookkeeping
    needed, since the draws themselves carry the timestamp once a run uses
    the current schema. ``started_at`` for the window is the earliest
    draw's ``started_at``; ``ended_at`` is the latest draw's
    ``started_at + latency_seconds`` (the draw's own measured completion,
    not just when the LAST draw began -- a session's true end is when its
    slowest/last-finishing draw actually completed). A session where every
    draw has ``started_at is None`` (pre-schema/backfilled -- see
    ``DrawRecord``'s docstring) yields ``{"started_at": None, "ended_at":
    None}`` here; ``build_run_metadata``'s ``_resolve_session_window``
    falls back to a window PERSISTED directly in that session's
    ``session_metadata.json`` entry (via ``record_session_provenance``) in
    that case, and only with windows explicitly labeled approximate
    (``window_precision``)."""
    windows: dict[str, dict[str, str | None]] = {}
    for session_id, session_draws in group_by_session(draws).items():
        starts: list[datetime] = []
        ends: list[datetime] = []
        for draw in session_draws:
            if draw.started_at is None:
                continue
            start_dt = datetime.fromisoformat(draw.started_at)
            starts.append(start_dt)
            ends.append(start_dt + timedelta(seconds=draw.latency_seconds))
        windows[session_id] = {
            "started_at": min(starts).isoformat() if starts else None,
            "ended_at": max(ends).isoformat() if ends else None,
        }
    return windows


_ENV_BOOL_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_ENV_BOOL_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def _read_env_bool(name: str, default: bool) -> bool:
    """Reads an env var the same name pydantic-settings would bind
    ``app.config.Settings``'s matching field to (case-insensitive match to
    the field name) -- WITHOUT importing ``app.config`` (this module has no
    ``app.*`` import path at all, by design -- see ``TestNoToolStub``).

    Gate-3/Opus DEFECT 1: the accepted true/false string sets now MATCH
    pydantic-core's LAX bool parser (what ``BaseSettings`` actually uses to
    coerce an env-var string) -- true: ``1``/``true``/``t``/``yes``/``y``/
    ``on``; false: ``0``/``false``/``f``/``no``/``n``/``off`` (all
    case-insensitive, whitespace-stripped). A value in neither set falls
    back to ``default`` rather than raising -- this harness reads config for
    REPORTING, not for gating behavior, so a stray unparseable value should
    surface as "assumed the recorded default" rather than crash a live run
    over a provenance field. Falls back to the LITERAL default recorded here
    when the env var is unset at all, which mirrors that field's declared
    default as of this harness's own commit -- if ``app/config.py``'s
    default ever changes, this fallback can go stale; it is a recorded
    snapshot, not a live query."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _ENV_BOOL_TRUE_VALUES:
        return True
    if normalized in _ENV_BOOL_FALSE_VALUES:
        return False
    return default


def _read_env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


_LIVE_FLAGS_SOURCE = (
    "environment variable read at run time (matching app.config.Settings' pydantic-settings "
    "field-name-uppercased convention), else the field default AS RECORDED IN THIS HARNESS "
    "(snapshot of app/config.py at the harness's own commit, not a live query -- this module "
    "has zero app.* import path by design, see TestNoToolStub)"
)

# Gate-3/Opus DEFECT 2: per-session provenance NEVER re-derived at
# --summarize-only time -- a session with no persisted entry in
# session_metadata.json (see load_session_metadata/record_session_
# provenance below) gets this explicit "unknown" placeholder instead of a
# guessed/fabricated value.
_UNKNOWN_SESSION_PROVENANCE: dict[str, Any] = {
    "app_git_sha": None,
    "engine": None,
    "model": None,
    "flags": None,
    "flags_source": None,
    "loop_order": "unknown (no live run recorded for this session -- see session_metadata.json)",
    "parallel": None,
    "temperature": None,
    "backfilled": None,
}


def _session_metadata_path() -> Path:
    return _RESULTS_DIR / "session_metadata.json"


def load_session_metadata() -> dict[str, dict[str, Any]]:
    """Read the durable per-session provenance sidecar (gate-3/Opus
    DEFECT 2) -- ``evals/results/issue-160/session_metadata.json``. Returns
    ``{}`` if the file doesn't exist yet. NEVER derives or guesses a value --
    a pure read of whatever ``record_session_provenance`` previously
    persisted. Called by every report build (live or ``--summarize-only``)
    so ``build_run_metadata`` always has real, previously-captured facts to
    merge in rather than re-deriving anything from the current environment."""
    path = _session_metadata_path()
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return result


def record_session_provenance(session_id: str, provenance: dict[str, Any]) -> None:
    """Durably persist ONE session's provenance (gate-3/Opus DEFECT 2) --
    called exactly once, by a LIVE ``main()`` run, immediately after that
    session's draws are captured (NEVER by ``--summarize-only``, which only
    calls ``load_session_metadata`` -- see ``main()``). Overwrites ONLY this
    ``session_id``'s own entry; every other session's persisted provenance
    is read, kept byte-identical, and written back verbatim -- so a second
    batch's run can never clobber an earlier session's facts, and
    ``--summarize-only`` (which never calls this function at all) can never
    invent them either.

    ``provenance`` may optionally carry ``started_at``/``ended_at`` (and a
    ``window_precision`` label -- gate-3/Opus final-round MINOR 1) when the
    caller wants a window persisted alongside the rest (the ONLY current
    caller of this is the one-time backfill migration for the two
    pre-schema committed sessions, whose draws carry no real per-draw
    ``started_at`` to derive a window from at all -- see
    ``build_run_metadata``'s ``_resolve_session_window`` for exactly how a
    persisted window interacts with a draw-derived one). An ordinary live
    ``main()`` run omits these keys -- its session's window is always
    derivable from its own draws' real ``started_at``, so persisting one
    here would be redundant, and correctly never happens.

    **Atomic write (gate-3/Opus final-round MINOR 2).** Writes to a
    ``.tmp`` sibling first, then ``os.replace``s it onto the real path --
    ``os.replace`` is atomic on both POSIX and Windows (unlike a bare
    ``path.write_text``, which truncates the target before writing and
    leaves a half-written, corrupt JSON file behind if the process dies
    mid-write). This harness has already had one real mid-run crash
    (the root-owned-file ``PermissionError`` documented in this module's
    own docstring) -- the durability store this function guards must not
    be the next thing a crash corrupts."""
    path = _session_metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = load_session_metadata()
    metadata[session_id] = provenance
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def _resolve_session_window(computed: dict[str, str | None], persisted: dict[str, Any]) -> dict[str, Any]:
    """Gate-3/Opus final-round MINOR 1: prefer the window DERIVED FROM THIS
    SESSION'S OWN DRAWS (``compute_session_windows``) whenever it has real
    data -- a window measured from draws always beats a persisted one, by
    construction (this function is the ONLY place the two are compared, and
    it always checks ``computed`` first). Falls back to whatever window was
    PERSISTED in the sidecar (``record_session_provenance``) only when the
    draws themselves carry no ``started_at`` at all (pre-schema/backfilled
    draws). Returns ``window_precision`` alongside the two timestamps so a
    reader never mistakes a persisted APPROXIMATE window for an exact,
    draw-measured one."""
    if computed["started_at"] is not None:
        return {"started_at": computed["started_at"], "ended_at": computed["ended_at"], "window_precision": "exact"}
    persisted_started = persisted.get("started_at")
    if persisted_started is not None:
        return {
            "started_at": persisted_started,
            "ended_at": persisted.get("ended_at"),
            "window_precision": persisted.get("window_precision", "approximate"),
        }
    return {"started_at": None, "ended_at": None, "window_precision": None}


def build_run_metadata(
    per_question: dict[str, list[DrawRecord]],
    *,
    session_provenance: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assembles the report-level provenance block (gate-3/Opus MAJOR 2,
    restructured per DEFECT 2, window-precedence fixed per final-round
    MINOR 1): PER-SESSION, not a single top-level block -- a multi-session
    report file (this harness's normal shape once a second batch is
    appended) has no single "the flags"/"the loop order" to report at the
    top level; different sessions may genuinely have run under different
    config. Each session's entry is ``{**provenance, **window}`` -- window
    keys (``started_at``/``ended_at``/``window_precision``, from
    ``_resolve_session_window``) are applied LAST and therefore win on
    every conflicting key, so a persisted/sidecar value can never silently
    shadow a real draw-derived one (the earlier bug this precedence fixes:
    the previous ``{**window, **provenance}`` order let a provenance dict
    that happened to carry its own ``started_at``/``ended_at`` keys
    override the computed window instead of the other way around).
    ``_resolve_session_window`` itself is what supplies the actual
    precedence between "computed from draws" and "persisted in the
    sidecar" -- this function's ``{**provenance, **window}`` merge only
    ensures window fields win over any STALE copy that might otherwise live
    in ``provenance``.

    ``session_provenance`` -- see ``load_session_metadata``/
    ``record_session_provenance`` -- is NEVER re-derived or guessed here. A
    session absent from it gets the explicit ``_UNKNOWN_SESSION_PROVENANCE``
    placeholder rather than a fabricated value.

    This is what makes ``--summarize-only`` safe to run repeatedly: its
    caller (``main()``) passes whatever ``load_session_metadata()`` returns,
    verbatim, so regenerating the report NEVER stamps THIS invocation's
    live-run facts (``loop_order``, engine, ...) onto a session it did not
    itself observe -- and a session's persisted window (if any) survives
    the regeneration too, since it now lives IN ``session_provenance``
    rather than in a caller-supplied override that only the one-time
    migration script used to pass."""
    all_draws = [draw for draws in per_question.values() for draw in draws]
    windows = compute_session_windows(all_draws)

    session_ids = set(windows) | set(session_provenance)
    sessions: dict[str, Any] = {}
    for session_id in session_ids:
        computed = windows.get(session_id, {"started_at": None, "ended_at": None})
        provenance = session_provenance.get(session_id, _UNKNOWN_SESSION_PROVENANCE)
        window = _resolve_session_window(computed, provenance)
        sessions[session_id] = {**provenance, **window}

    return {
        "sessions": sessions,
        "session_windows_note": (
            "started_at/ended_at/window_precision: 'exact' means derived from this session's OWN "
            "draws (started_at = earliest draw's started_at; ended_at = latest draw's started_at + "
            "latency_seconds, i.e. when that draw actually finished). 'approximate' means no draw in "
            "this session carries a real started_at (pre-schema/backfilled draws), so the window was "
            "persisted directly into session_metadata.json instead -- see each such session's own "
            "flags_source/backfilled fields for how that approximate window was derived. A session "
            "with window_precision: null has neither a computed nor a persisted window at all."
        ),
        "session_provenance_note": (
            "app_git_sha/engine/model/flags/flags_source/loop_order/parallel/temperature/"
            "backfilled are PERSISTED per-session facts (evals/results/issue-160/"
            "session_metadata.json), captured once by the live run that produced that session "
            "and never re-derived on a later --summarize-only regeneration. A session showing "
            "loop_order starting with \"unknown\" has no persisted entry."
        ),
    }


# --- live run (network I/O -- never exercised by the unit-test suite) ---------


def _draws_path(question_id: str) -> Path:
    return _RESULTS_DIR / f"{question_id}.jsonl"


def append_draw(draw: DrawRecord) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _draws_path(draw.question_id)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(draw)) + "\n")
    return path


class DuplicateCorrelationIdError(Exception):
    """Raised by :func:`load_draws` when two non-error draws in the same
    question's ``*.jsonl`` share a ``correlation_id`` (gate-3/Opus MINOR 2).
    ``correlation_id`` is server-generated per turn (``app.correlation
    .get_correlation_id()``, a fresh UUID every ``POST /chat`` call) -- a
    duplicate is not a benign coincidence, it means either the same response
    was appended twice (an ``append_draw`` bug or a re-run over the exact
    same in-flight state) or the server itself failed to generate a fresh
    id. Either way it's data corruption worth stopping on, not silently
    aggregating over."""


def _assert_correlation_ids_unique(question_id: str, draws: list[DrawRecord]) -> None:
    seen: dict[str, int] = {}
    for draw in draws:
        if draw.error is not None or draw.correlation_id is None:
            continue
        if draw.correlation_id in seen:
            raise DuplicateCorrelationIdError(
                f"{question_id}: correlation_id {draw.correlation_id!r} appears on both "
                f"draw_index={seen[draw.correlation_id]} and draw_index={draw.draw_index} "
                "(session_id may differ) -- see DuplicateCorrelationIdError's docstring"
            )
        seen[draw.correlation_id] = draw.draw_index


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
    _assert_correlation_ids_unique(question_id, draws)
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
    ``run_one_draw``).

    ``started_at`` is stamped (UTC, ISO 8601) immediately before the request
    fires -- gate-3/Opus MAJOR 2, so FUTURE runs carry a real per-draw
    timestamp (the two already-committed batches predate this field and
    carry ``started_at: null`` -- see ``DrawRecord``'s docstring). The raw
    SSE lines are materialized into a list ONCE (``response.iter_lines()``
    is a one-shot generator) so both ``parse_sse_lines`` and
    ``count_orphan_data_lines`` (gate-3/Opus MINOR 3) can walk the identical
    line sequence without either consuming what the other needs."""
    payload = {"message": question.question, "patient_id": question.patient_id}
    headers = {"Authorization": f"Bearer {token}"}
    started_at = datetime.now(timezone.utc).isoformat()
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
                    started_at=started_at,
                    events=[],
                    latency_seconds=latency,
                    http_status=status,
                    orphan_data_line_count=None,
                    error=f"HTTP {status}: {response.text[:500]}",
                )
            lines = list(response.iter_lines())
            events = parse_sse_lines(lines)
            orphan_count = count_orphan_data_lines(lines)
        latency = time.monotonic() - started
        return build_draw_record(
            question=question,
            draw_index=draw_index,
            session_id=session_id,
            started_at=started_at,
            events=events,
            latency_seconds=latency,
            http_status=status,
            orphan_data_line_count=orphan_count,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 -- harness run: record failure, keep going
        latency = time.monotonic() - started
        return build_draw_record(
            question=question,
            draw_index=draw_index,
            session_id=session_id,
            started_at=started_at,
            events=[],
            latency_seconds=latency,
            http_status=None,
            orphan_data_line_count=None,
            error=repr(exc),
        )


def main() -> None:
    import uuid

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=_DEFAULT_DRAWS, help="draws per question (default 8)")
    parser.add_argument("--start-index", type=int, default=0, help="starting draw_index for this session (default 0; bump for a second batch so indices don't collide)")
    parser.add_argument("--base-url", default=_DEFAULT_BASE_URL, help="agent base URL (default in-container localhost:8000)")
    parser.add_argument("--token", default=_DEFAULT_TOKEN, help="bearer token (any non-empty value works with copilot_dev_accept_any_bearer_token=true, this dev stack's setting)")
    parser.add_argument("--session-id", default=None, help="label for this run's draws (default: a fresh random id) -- distinguishes this batch from any prior batch appended to the same *.jsonl, so report.json's by_session never silently mixes them")
    parser.add_argument(
        "--no-interleave",
        dest="interleave",
        action="store_false",
        default=True,
        help=(
            "gate-3/Opus MAJOR 3: by default draws are INTERLEAVED across questions "
            "(question A draw 0, question B draw 0, question A draw 1, ...) rather than run "
            "back-to-back per question -- back-to-back-per-question is a warm-single-slot loop "
            "that gives near-zero power against nondeterminism by construction (see module "
            "docstring's warm-slot caveat). Pass --no-interleave to reproduce the OLD "
            "back-to-back-per-question ordering the first two committed batches used."
        ),
    )
    parser.add_argument(
        "--app-git-sha",
        default=None,
        help="git SHA of the app tree under test (this harness has no app.* import path, so it cannot detect this itself -- pass $(git rev-parse HEAD) from the host, or set APP_GIT_SHA in the environment)",
    )
    parser.add_argument("--summarize-only", action="store_true", help="skip live runs; just re-aggregate evals/results/issue-160/*.jsonl")
    args = parser.parse_args()

    if not args.summarize_only:
        import httpx  # live-run-only dependency -- see post_chat_draw's docstring

        session_id = args.session_id or f"session-{uuid.uuid4().hex[:12]}"
        print(f"[issue-160] session_id={session_id} interleave={args.interleave}")

        draw_indices = range(args.start_index, args.start_index + args.draws)
        # gate-3/Opus MAJOR 3: interleaved is now the default ordering --
        # question, then draw_index, in the outer loop -- so consecutive
        # live calls never share a question (breaking up any single warm
        # inference slot's run of identical-question calls). --no-interleave
        # keeps the OLD per-question-back-to-back ordering for comparison.
        run_plan = (
            [(draw_index, question) for draw_index in draw_indices for question in TARGET_QUESTIONS]
            if args.interleave
            else [(draw_index, question) for question in TARGET_QUESTIONS for draw_index in draw_indices]
        )

        for draw_index, question in run_plan:
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

        # Gate-3/Opus DEFECT 2: persist THIS session's provenance durably and
        # exactly once, immediately after its draws are captured -- never
        # re-derived by a later --summarize-only regeneration (see
        # build_run_metadata's docstring). Reads env vars now, while this
        # process is still running inside the same container/session that
        # produced the draws -- a --summarize-only run later (possibly in a
        # different container, or after the environment changed) has no
        # business re-reading these and calling the result the same thing.
        engine = _read_env_str("COPILOT_LLM_ENGINE", "llama_server")
        model = (
            _read_env_str("LLAMA_SERVER_MODEL", "qwen3-8b")
            if engine == "llama_server"
            else _read_env_str("OLLAMA_MODEL", "qwen3:4b")
        )
        record_session_provenance(
            session_id,
            {
                "app_git_sha": args.app_git_sha or os.environ.get("APP_GIT_SHA"),
                "engine": engine,
                "model": model,
                "flags": {
                    "evidence_retrieval_enabled": _read_env_bool("COPILOT_EVIDENCE_RETRIEVAL_ENABLED", False),
                    "semantic_support_enabled": _read_env_bool("COPILOT_SEMANTIC_SUPPORT_ENABLED", True),
                    "answer_grounding_enabled": _read_env_bool("COPILOT_CLAIM_ANSWER_GROUNDING_ENABLED", False),
                    "tool_call_scoping_enabled": _read_env_bool(
                        "COPILOT_EXTRACTION_TOOL_CALL_SCOPING_ENABLED", False
                    ),
                },
                "flags_source": _LIVE_FLAGS_SOURCE,
                "loop_order": (
                    "interleaved (round-robin across questions per draw index)"
                    if args.interleave
                    else "sequential-per-question (all draws of one question back-to-back, warm single slot)"
                ),
                "parallel": 1,
                "temperature": "0 (llama_server_client.py source-confirmed default; not runtime-verified by this harness)",
                "backfilled": False,
            },
        )

    per_question = {q.id: load_draws(q.id) for q in TARGET_QUESTIONS}
    report = build_report(per_question)
    # Gate-3/Opus DEFECT 2: --summarize-only NEVER calls record_session_
    # provenance above, so this is a pure read of whatever was durably
    # persisted by a PAST live run -- a summarize-only regeneration can
    # never stamp fresh guesses (or this invocation's own environment) onto
    # sessions it did not itself produce.
    report["run_metadata"] = build_run_metadata(per_question, session_provenance=load_session_metadata())
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _RESULTS_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
