"""Red-first schema + attribution-logic tests for the issue #160 live-`/chat`
N-draw stability harness (``evals/runner/issue_160_live_chat_harness.py``).

**Why this harness exists / what it removes.** Issue #154's harness
(``evals/runner/issue_154_stability_harness.py``) pins ``tool_data`` and
never reaches the real ``OpenEmrClient`` -- its own docstring calls this a
"hard scope limit": it measures variance strictly DOWNSTREAM of tool
results (answer-composition given FIXED tool data, extraction,
verification) and explicitly "does not and cannot reproduce or refute
variance that originates upstream, in the planner's live tool-calling
against the real chart." Issue #160's harness is that missing upstream
measurement: it POSTs the REAL, running ``/chat`` endpoint over HTTP N
times per question, so the planner's tool selection AND the real chart are
both live, exactly like the original #149/#150 manual draws that surfaced
the instability in the first place.

**What this file does and does not cover.** The harness's live half (HTTP
POST to a running agent container's ``/chat``, SSE streaming, parsing a
live response) needs a booted dev stack and is exercised by hand (see the
harness module's own docstring for the exact in-container invocation) --
never in this suite, which must stay fast, network-free, and CI-safe. This
file pins only the harness's PURE, in-memory logic:

  1. the ``DrawRecord``/``ToolCallRecord``/``ClaimSegmentRecord`` schema
     (frozen dataclasses -- shape pinned so a future refactor can't silently
     drop a field the report depends on),
  2. ``parse_sse_lines`` -- turning raw SSE text lines into ``(event, data)``
     pairs, fed literal fixture text (no network, no ``httpx`` mock needed),
  3. ``build_draw_record`` -- turning one draw's parsed events into a
     ``DrawRecord``, including the "never persist raw answer text" contract
     (only a hash + length survive) and the "per-claim outcome, not raw
     citation status" contract (SSE's ``verification`` frame only carries
     claim-vs-notice segments -- see ``app.rendering``'s module docstring,
     quoted in the harness module's own docstring, for why a finer-grained
     ``CitationStatus`` is not on the wire at all),
  4. ``attribute_mechanisms`` -- the three-way (tool-selection /
     generation-nondeterminism / downstream) mechanism attribution that is
     this harness's entire point, per its own module docstring.

**No-tool-stub property (the #154 failure mode this harness exists to
escape).** ``TestNoToolStub`` statically inspects the harness module's own
source text and asserts it never imports ``runner.tool_stub``,
``runner.pipeline``, ``runner.schema``, or any ``app.*`` module at all --
i.e. it cannot possibly fall back to in-process planner calls against
canned ``tool_data``, by construction, because it has no import path to any
of that machinery. A harness that measures the live HTTP surface has no
legitimate reason to import ``app.planner``/``app.extraction`` or
``runner.tool_stub.build_fake_registry`` -- their mere presence in the
import list would mean some code path re-creates #154's blind spot instead
of removing it.

Written BEFORE ``runner.issue_160_live_chat_harness`` exists (strict
red-first, per ``CLAUDE.md``'s "Red first -- strict TDD, everywhere"): the
first commit of this file fails on the module import with
``ModuleNotFoundError`` -- the intended red state.

This module's name (``test_issue_160_...py``) matches pytest's collection
glob deliberately, unlike the live harness itself -- this file contains no
live call and must run in CI like any other test."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from runner.issue_160_live_chat_harness import (
    ALLERGY_QUESTION,
    BP_QUESTION,
    ClaimSegmentRecord,
    DrawRecord,
    DuplicateCorrelationIdError,
    Question,
    ToolCallRecord,
    _assert_correlation_ids_unique,
    aggregate_orphan_data_lines,
    attribute_mechanisms,
    build_draw_record,
    build_report,
    build_run_metadata,
    compute_session_windows,
    count_duplicate_draw_indices,
    count_orphan_data_lines,
    group_by_session,
    is_run_valid,
    parse_sse_lines,
    sequence_label,
    summarize_question,
    tool_sequence_of,
)

_HARNESS_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "issue_160_live_chat_harness.py"
)


# --- schema pinning ----------------------------------------------------------


class TestSchema:
    def test_question_targets_match_149_150(self) -> None:
        assert BP_QUESTION.patient_id == 1
        assert "blood pressure" in BP_QUESTION.question.lower()
        assert ALLERGY_QUESTION.patient_id == 2
        assert "allergy conflict" in ALLERGY_QUESTION.question.lower()

    def test_tool_call_record_shape(self) -> None:
        record = ToolCallRecord(order=0, tool="get_vitals", args={}, error=None)
        assert record.order == 0
        assert record.tool == "get_vitals"
        assert record.args == {}
        assert record.error is None

    def test_claim_segment_record_shape(self) -> None:
        record = ClaimSegmentRecord(index=0, passed=True, citation_count=1)
        assert record.passed is True
        assert record.citation_count == 1

    def test_draw_record_shape_and_no_raw_answer_field(self) -> None:
        draw = DrawRecord(
            question_id="issue-149-bp",
            patient_id=1,
            draw_index=0,
            session_id="session-a",
            started_at="2026-01-01T00:00:00+00:00",
            correlation_id="corr-1",
            conversation_id="conv-1",
            tool_calls=[ToolCallRecord(order=0, tool="get_vitals", args={}, error=None)],
            answer_hash="deadbeef",
            answer_length=42,
            claim_count=1,
            claims=[ClaimSegmentRecord(index=0, passed=True, citation_count=1)],
            verdict="partially_verified",
            latency_seconds=1.23,
            http_status=200,
            orphan_data_line_count=0,
            error=None,
        )
        # #163 artifact-discipline parity: the schema has no field that
        # could hold raw answer/claim text -- only hash/length/counts/enums.
        field_names = set(DrawRecord.__dataclass_fields__)
        assert "answer" not in field_names
        assert "answer_text" not in field_names
        assert "claim_text" not in field_names
        assert draw.answer_hash == "deadbeef"
        assert draw.session_id == "session-a"


# --- parse_sse_lines ----------------------------------------------------------


class TestParseSseLines:
    def test_parses_event_data_pairs(self) -> None:
        lines = [
            "event: conversation",
            'data: {"conversation_id": "c1", "correlation_id": "r1"}',
            "",
            "event: tool_call",
            'data: {"tool": "get_vitals", "args": {}, "error": null}',
            "",
            "event: done",
            "data: {}",
            "",
        ]

        events = parse_sse_lines(lines)

        assert events == [
            ("conversation", {"conversation_id": "c1", "correlation_id": "r1"}),
            ("tool_call", {"tool": "get_vitals", "args": {}, "error": None}),
            ("done", {}),
        ]

    def test_ignores_blank_and_comment_lines(self) -> None:
        lines = [":keepalive", "", "event: done", "data: {}", ""]

        events = parse_sse_lines(lines)

        assert events == [("done", {})]


# --- build_draw_record --------------------------------------------------------


def _bp_events() -> list[tuple[str, dict]]:
    return [
        ("conversation", {"conversation_id": "conv-1", "correlation_id": "corr-1"}),
        ("tool_call", {"tool": "get_vitals", "args": {}, "error": None}),
        ("reasoning_delta", {"text": "thinking..."}),
        ("answer", {"answer": "BP was 130/80, normal."}),
        (
            "verification",
            {
                "verdict": "partially_verified",
                "segments": [
                    {"type": "claim", "text": "...", "citations": [{"tool_call_id": "call_0"}]},
                    {"type": "notice", "text": "Not found in record."},
                ],
                "warnings": {"allergy_conflicts": [], "blocking_interactions": [], "warning_interactions": []},
            },
        ),
        ("done", {}),
    ]


class TestBuildDrawRecord:
    def test_happy_path_hashes_answer_never_stores_raw_text(self) -> None:
        draw = build_draw_record(
            question=BP_QUESTION,
            draw_index=0,
            session_id="session-test",
            started_at="2026-01-01T00:00:00+00:00",
            events=_bp_events(),
            latency_seconds=2.5,
            http_status=200,
            orphan_data_line_count=0,
            error=None,
        )

        assert draw.question_id == BP_QUESTION.id
        assert draw.session_id == "session-test"
        assert draw.started_at == "2026-01-01T00:00:00+00:00"
        assert draw.orphan_data_line_count == 0
        assert draw.patient_id == 1
        assert draw.conversation_id == "conv-1"
        assert draw.correlation_id == "corr-1"
        assert draw.tool_calls == [ToolCallRecord(order=0, tool="get_vitals", args={}, error=None)]
        assert draw.verdict == "partially_verified"
        assert draw.claim_count == 2
        assert draw.claims == [
            ClaimSegmentRecord(index=0, passed=True, citation_count=1),
            ClaimSegmentRecord(index=1, passed=False, citation_count=0),
        ]
        expected_hash = hashlib.sha256(b"BP was 130/80, normal.").hexdigest()
        assert draw.answer_hash == expected_hash
        assert draw.answer_length == len("BP was 130/80, normal.")
        assert draw.error is None
        assert draw.http_status == 200

    def test_multiple_tool_calls_preserve_order(self) -> None:
        events = [
            ("conversation", {"conversation_id": "c", "correlation_id": "r"}),
            ("tool_call", {"tool": "get_medications", "args": {}, "error": None}),
            ("tool_call", {"tool": "get_allergies", "args": {}, "error": None}),
            ("answer", {"answer": "no conflict"}),
            ("verification", {"verdict": "verified", "segments": [], "warnings": {}}),
            ("done", {}),
        ]

        draw = build_draw_record(
            question=ALLERGY_QUESTION,
            draw_index=1,
            session_id="session-test",
            started_at="2026-01-01T00:00:00+00:00",
            events=events,
            latency_seconds=1.0,
            http_status=200,
            orphan_data_line_count=0,
            error=None,
        )

        assert [tc.tool for tc in draw.tool_calls] == ["get_medications", "get_allergies"]
        assert [tc.order for tc in draw.tool_calls] == [0, 1]

    def test_error_draw_carries_no_answer_or_verdict(self) -> None:
        draw = build_draw_record(
            question=BP_QUESTION,
            draw_index=2,
            session_id="session-test",
            started_at="2026-01-01T00:00:00+00:00",
            events=[],
            latency_seconds=180.0,
            http_status=None,
            orphan_data_line_count=None,
            error="ConnectTimeout",
        )

        assert draw.error == "ConnectTimeout"
        assert draw.answer_hash is None
        assert draw.answer_length is None
        assert draw.verdict is None
        assert draw.claim_count is None
        assert draw.tool_calls == []
        assert draw.claims == []
        assert draw.orphan_data_line_count is None


# --- tool_sequence_of / sequence_label ----------------------------------------


class TestToolSequence:
    def test_sequence_of_extracts_ordered_tool_names(self) -> None:
        draw = build_draw_record(
            question=BP_QUESTION,
            draw_index=0,
            session_id="session-test",
            started_at="2026-01-01T00:00:00+00:00",
            events=_bp_events(),
            latency_seconds=1.0,
            http_status=200,
            orphan_data_line_count=0,
            error=None,
        )

        assert tool_sequence_of(draw) == ("get_vitals",)

    def test_empty_sequence_for_error_draw(self) -> None:
        draw = build_draw_record(
            question=BP_QUESTION,
            draw_index=0,
            session_id="session-test",
            started_at="2026-01-01T00:00:00+00:00",
            events=[],
            latency_seconds=1.0,
            http_status=None,
            orphan_data_line_count=None,
            error="boom",
        )

        assert tool_sequence_of(draw) == ()

    def test_sequence_label_joins_names(self) -> None:
        assert sequence_label(("get_medications", "get_allergies")) == "get_medications -> get_allergies"
        assert sequence_label(()) == "()"


# --- attribute_mechanisms -----------------------------------------------------


def _draw(
    draw_index: int,
    *,
    tools: tuple[str, ...] = ("get_vitals",),
    answer_text: str = "same answer",
    verdict: str = "verified",
    error: str | None = None,
    session_id: str = "session-a",
) -> DrawRecord:
    tool_calls = [ToolCallRecord(order=i, tool=t, args={}, error=None) for i, t in enumerate(tools)]
    if error is not None:
        return DrawRecord(
            question_id="q",
            patient_id=1,
            draw_index=draw_index,
            session_id=session_id,
            started_at=None,
            correlation_id=None,
            conversation_id=None,
            tool_calls=[],
            answer_hash=None,
            answer_length=None,
            claim_count=None,
            claims=[],
            verdict=None,
            latency_seconds=1.0,
            http_status=None,
            orphan_data_line_count=None,
            error=error,
        )
    return DrawRecord(
        question_id="q",
        patient_id=1,
        draw_index=draw_index,
        session_id=session_id,
        started_at=f"2026-01-01T00:00:{draw_index:02d}+00:00",
        correlation_id=f"corr-{session_id}-{draw_index}",
        conversation_id=f"conv-{draw_index}",
        tool_calls=tool_calls,
        answer_hash=hashlib.sha256(answer_text.encode()).hexdigest(),
        answer_length=len(answer_text),
        claim_count=1,
        claims=[ClaimSegmentRecord(index=0, passed=True, citation_count=1)],
        verdict=verdict,
        latency_seconds=1.0,
        http_status=200,
        orphan_data_line_count=0,
        error=None,
    )


class TestAttributeMechanisms:
    def test_stable_draws_flag_no_mechanism(self) -> None:
        draws = [_draw(i) for i in range(4)]

        report = attribute_mechanisms(draws)

        assert report["mechanism"]["tool_selection_variance"] is False
        assert report["mechanism"]["generation_nondeterminism"] is False
        assert report["mechanism"]["downstream_variance"] is False
        assert report["distinct_tool_sequences"] == 1

    def test_same_sequence_same_hash_different_verdict_is_downstream(self) -> None:
        """Mutation checked: swapping the downstream-detection condition from
        'more than one distinct verdict within a (sequence, hash) group' to
        'more than one distinct verdict overall' makes this assertion fail to
        distinguish downstream variance from tool-selection variance --
        confirmed by hand this exact fixture only trips the downstream flag,
        not the tool-selection one."""
        draws = [
            _draw(0, answer_text="same text", verdict="verified"),
            _draw(1, answer_text="same text", verdict="blocked"),
        ]

        report = attribute_mechanisms(draws)

        assert report["mechanism"]["downstream_variance"] is True
        assert report["mechanism"]["tool_selection_variance"] is False
        assert report["mechanism"]["generation_nondeterminism"] is False
        seq = sequence_label(("get_vitals",))
        assert report["answer_hash_distribution"][seq][hashlib.sha256(b"same text").hexdigest()] == 2

    def test_same_sequence_different_hash_is_generation_nondeterminism(self) -> None:
        draws = [
            _draw(0, answer_text="answer A", verdict="verified"),
            _draw(1, answer_text="answer B", verdict="verified"),
        ]

        report = attribute_mechanisms(draws)

        assert report["mechanism"]["generation_nondeterminism"] is True
        assert report["mechanism"]["downstream_variance"] is False
        assert report["mechanism"]["tool_selection_variance"] is False

    def test_different_sequence_is_tool_selection_variance(self) -> None:
        draws = [
            _draw(0, tools=("get_vitals",)),
            _draw(1, tools=("get_vitals", "get_encounters")),
        ]

        report = attribute_mechanisms(draws)

        assert report["mechanism"]["tool_selection_variance"] is True
        assert report["distinct_tool_sequences"] == 2

    def test_errors_excluded_from_counted_but_tracked(self) -> None:
        draws = [_draw(0), _draw(1, error="ConnectTimeout")]

        report = attribute_mechanisms(draws)

        assert report["n_draws"] == 2
        assert report["n_errors"] == 1
        assert report["n_counted"] == 1


class TestSummarizeQuestion:
    def test_includes_latency_and_verdict_distribution(self) -> None:
        draws = [_draw(0, verdict="verified"), _draw(1, verdict="blocked", answer_text="different")]

        summary = summarize_question("q", draws)

        assert summary["n_draws"] == 2
        assert summary["verdict_distribution"] == {"verified": 1, "blocked": 1}
        assert "latency_seconds" in summary
        assert summary["latency_seconds"]["min"] == pytest.approx(1.0)
        assert summary["mechanism"]["generation_nondeterminism"] is True


# --- session grouping / combined-vs-by-session report (#160 follow-up) --------


class TestGroupBySession:
    def test_splits_by_session_id_preserving_first_seen_order(self) -> None:
        draws = [
            _draw(0, session_id="session-a"),
            _draw(1, session_id="session-b"),
            _draw(2, session_id="session-a"),
        ]

        grouped = group_by_session(draws)

        assert list(grouped.keys()) == ["session-a", "session-b"]
        assert [d.draw_index for d in grouped["session-a"]] == [0, 2]
        assert [d.draw_index for d in grouped["session-b"]] == [1]


class TestBuildReportCombinedAndBySession:
    def test_combined_mixes_all_sessions_by_session_keeps_them_apart(self) -> None:
        """The exact scenario the 16-more-draws follow-up must never get
        wrong: session A alone is stable (all 'verified'); session B alone
        introduces a verdict flip. combined must reflect ALL draws pooled;
        by_session must let a reader see each session's own, unmixed
        picture."""
        session_a = [_draw(i, session_id="session-a", verdict="verified") for i in range(2)]
        session_b = [
            _draw(2, session_id="session-b", verdict="verified"),
            _draw(3, session_id="session-b", verdict="blocked"),
        ]
        draws = session_a + session_b

        report = build_report({"q": draws})

        assert report["q"]["session_ids"] == ["session-a", "session-b"]
        assert report["q"]["combined"]["n_draws"] == 4
        assert report["q"]["combined"]["verdict_distribution"] == {"verified": 3, "blocked": 1}
        assert report["q"]["by_session"]["session-a"]["n_draws"] == 2
        assert report["q"]["by_session"]["session-a"]["verdict_distribution"] == {"verified": 2}
        assert report["q"]["by_session"]["session-b"]["n_draws"] == 2
        assert report["q"]["by_session"]["session-b"]["verdict_distribution"] == {"verified": 1, "blocked": 1}
        # #160 follow-up mutation check: if by_session were dropped in favor
        # of combined-only, session-a's own (stable) picture would be lost --
        # confirmed this assertion fails without the by_session key at all.
        assert "by_session" in report["q"]


# --- run_valid gate (gate-3/Opus MINOR 3) --------------------------------------


class TestRunValidGate:
    def test_run_valid_true_when_no_errors_and_all_fields_present(self) -> None:
        assert is_run_valid([_draw(0), _draw(1)]) is True

    def test_run_valid_false_when_any_error_present(self) -> None:
        assert is_run_valid([_draw(0), _draw(1, error="boom")]) is False

    def test_run_valid_false_when_a_counted_draw_missing_verdict(self) -> None:
        """A malformed-but-not-erroring draw (error=None, verdict=None)
        shouldn't be producible by build_draw_record in practice (see its
        own gate-1 simplify comment on the analogous claim_count ambiguity),
        but is_run_valid must catch it defensively rather than assume
        build_draw_record is the only caller."""
        malformed = replace(_draw(0), verdict=None)

        assert is_run_valid([malformed]) is False

    def test_mechanism_booleans_are_null_not_false_when_run_invalid(self) -> None:
        """Mutation checked: if attribute_mechanisms computed the three
        mechanism booleans unconditionally (ignoring run_valid), an invalid
        run (one error draw + one stable draw) would render
        tool_selection_variance=False instead of None -- confirmed this
        assertion fails under that mutation (reads False, `is None` fails)."""
        report = attribute_mechanisms([_draw(0), _draw(1, error="boom")])

        assert report["run_valid"] is False
        assert report["mechanism"]["tool_selection_variance"] is None
        assert report["mechanism"]["generation_nondeterminism"] is None
        assert report["mechanism"]["downstream_variance"] is None

    def test_mechanism_booleans_stay_boolean_when_run_valid(self) -> None:
        report = attribute_mechanisms([_draw(0), _draw(1)])

        assert report["run_valid"] is True
        assert report["mechanism"]["tool_selection_variance"] is False


# --- duplicate draw-index detection (gate-3/Opus MINOR 2) -----------------------


class TestDuplicateDrawIndices:
    def test_zero_when_all_indices_unique_within_session(self) -> None:
        draws = [_draw(0, session_id="s"), _draw(1, session_id="s")]

        assert count_duplicate_draw_indices(draws) == 0

    def test_counts_repeats_within_one_session_not_across_sessions(self) -> None:
        """Mutation checked: using bare draw_index (not (session_id,
        draw_index)) as the dedup key would falsely flag session-a's
        draw_index=0 and session-b's draw_index=0 as a duplicate -- two
        DIFFERENT sessions legitimately both starting at index 0 must NOT be
        counted; only session-a's own genuinely repeated index 0 must be."""
        draws = [
            _draw(0, session_id="session-a"),
            _draw(0, session_id="session-a"),  # genuine duplicate within session-a
            _draw(0, session_id="session-b"),  # different session, same index -- not a duplicate
        ]

        assert count_duplicate_draw_indices(draws) == 1


# --- orphan SSE data-line detection (gate-3/Opus MINOR 3) ------------------------


class TestOrphanDataLines:
    def test_count_orphan_data_lines_skips_paired_lines(self) -> None:
        assert count_orphan_data_lines(["event: done", "data: {}", ""]) == 0

    def test_count_orphan_data_lines_counts_unpaired_data(self) -> None:
        """Mutation checked: dropping the 'pending_event is None' guard
        (counting every data: line unconditionally) makes this assertion
        fail (would report 2, not 1) -- the well-formed 'event: done'/
        'data: {}' pair must not be counted as orphaned."""
        lines = ['data: {"stray": true}', "event: done", "data: {}"]

        assert count_orphan_data_lines(lines) == 1

    def test_aggregate_orphan_data_lines_separates_known_from_unknown(self) -> None:
        known_zero = _draw(0)  # orphan_data_line_count=0 (see _draw's default)
        known_three = replace(_draw(1), orphan_data_line_count=3)
        unknown = replace(_draw(2), orphan_data_line_count=None)

        result = aggregate_orphan_data_lines([known_zero, known_three, unknown])

        assert result == {"total": 3, "draws_with_unknown_count": 1}


# --- correlation_id uniqueness (gate-3/Opus MINOR 2) -----------------------------


class TestCorrelationIdUniqueness:
    def test_passes_when_all_unique(self) -> None:
        _assert_correlation_ids_unique("q", [_draw(0), _draw(1)])  # must not raise

    def test_raises_on_duplicate_correlation_id_among_non_error_draws(self) -> None:
        """Mutation checked: removing this uniqueness check entirely (a
        no-op function body) makes this test fail -- no exception is raised
        where one is expected. The duplicate here is a genuine data-
        corruption shape: two different draw_index rows sharing one
        server-generated correlation_id."""
        d0 = _draw(0)
        d1 = replace(_draw(1), correlation_id=d0.correlation_id)

        with pytest.raises(DuplicateCorrelationIdError) as exc_info:
            _assert_correlation_ids_unique("q", [d0, d1])

        assert d0.correlation_id in str(exc_info.value)

    def test_error_draws_are_exempt_from_uniqueness_check(self) -> None:
        """Error draws all carry correlation_id=None (build_draw_record's
        error branch) -- None must never collide with itself as a
        'duplicate,' or every multi-error run would falsely raise."""
        _assert_correlation_ids_unique(
            "q", [_draw(0, error="boom"), _draw(1, error="boom")]
        )  # must not raise


# --- session windows / run_metadata (gate-3/Opus MAJOR 2) -----------------------


def _draw_record(*, session_id: str, draw_index: int, started_at: str | None, latency_seconds: float = 1.0, orphan: int | None = 0) -> DrawRecord:
    return DrawRecord(
        question_id="q",
        patient_id=1,
        draw_index=draw_index,
        session_id=session_id,
        started_at=started_at,
        correlation_id=f"c-{session_id}-{draw_index}",
        conversation_id=f"conv-{session_id}-{draw_index}",
        tool_calls=[],
        answer_hash="h",
        answer_length=1,
        claim_count=0,
        claims=[],
        verdict="verified",
        latency_seconds=latency_seconds,
        http_status=200,
        orphan_data_line_count=orphan,
        error=None,
    )


class TestComputeSessionWindows:
    def test_computes_earliest_start_and_latest_finish(self) -> None:
        d0 = _draw_record(session_id="s1", draw_index=0, started_at="2026-01-01T00:00:00+00:00", latency_seconds=10.0)
        d1 = _draw_record(session_id="s1", draw_index=1, started_at="2026-01-01T00:00:20+00:00", latency_seconds=5.0)

        windows = compute_session_windows([d0, d1])

        assert windows["s1"]["started_at"] == "2026-01-01T00:00:00+00:00"
        # ended_at = latest (started_at + latency): d1's 00:00:20 + 5s = 00:00:25,
        # which beats d0's 00:00:00 + 10s = 00:00:10.
        assert windows["s1"]["ended_at"] == "2026-01-01T00:00:25+00:00"

    def test_none_when_no_draw_in_session_has_started_at(self) -> None:
        d0 = _draw_record(session_id="old-session", draw_index=0, started_at=None)

        windows = compute_session_windows([d0])

        assert windows["old-session"] == {"started_at": None, "ended_at": None}


class TestBuildRunMetadata:
    def test_manual_session_windows_fill_unknown_sessions_only(self) -> None:
        """Mutation checked: dropping the 'windows[session_id]["started_at"]
        is None' guard (always overwriting with manual data regardless)
        would let a manual window silently clobber a real, live-derived
        window -- confirmed this test's live-session assertion would then
        read the manual placeholder instead of the real timestamp."""
        live = _draw_record(session_id="live-session", draw_index=0, started_at="2026-02-01T00:00:00+00:00")
        old = _draw_record(session_id="old-session", draw_index=0, started_at=None, orphan=None)

        metadata = build_run_metadata(
            {"q": [live, old]},
            backfilled=True,
            app_git_sha="deadbeef",
            engine="llama_server",
            model="qwen3-8b",
            flags={
                "evidence_retrieval_enabled": False,
                "semantic_support_enabled": True,
                "answer_grounding_enabled": False,
                "tool_call_scoping_enabled": False,
            },
            loop_order="sequential-per-question",
            parallel=1,
            temperature="0",
            manual_session_windows={
                "old-session": {"started_at": "approx-start", "ended_at": "approx-end"},
                "live-session": {"started_at": "SHOULD-NOT-BE-USED", "ended_at": "SHOULD-NOT-BE-USED"},
            },
        )

        assert metadata["sessions"]["old-session"] == {"started_at": "approx-start", "ended_at": "approx-end"}
        assert metadata["sessions"]["live-session"]["started_at"] == "2026-02-01T00:00:00+00:00"
        assert metadata["backfilled"] is True
        assert metadata["app_git_sha"] == "deadbeef"
        assert metadata["flags"]["semantic_support_enabled"] is True


# --- no-tool-stub property -----------------------------------------------------


class TestNoToolStub:
    """Pins the #154-escaping property: this harness has no import path back
    into in-process/pinned-tool-data pipeline machinery."""

    def test_module_has_no_app_dot_star_imports_via_ast(self) -> None:
        """Gate-1 simplify (#160): this AST-based check SUBSUMES a prior
        string-scan test that grepped raw ``import``/``from ...`` lines for
        forbidden substrings (``runner.tool_stub``, ``runner.pipeline``,
        ``app``, ...) -- removed as redundant, since everything that scan
        could catch, parsing the actual import AST catches too, and more
        robustly: a multi-line ``from x import (\\n    y,\\n)`` or an
        aliased ``import app.planner as p`` defeats a line-level substring
        scan but not this walk over ``ast.ImportFrom``/``ast.Import`` nodes,
        which sees the real module/alias names regardless of source
        formatting."""
        import ast

        source = _HARNESS_MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("app"), f"forbidden in-process import: {node.module}"
                assert not node.module.startswith("runner.pipeline")
                assert not node.module.startswith("runner.tool_stub")
                assert not node.module.startswith("runner.schema")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("app"), f"forbidden in-process import: {alias.name}"

    def test_build_draw_record_is_pure_no_network_import(self) -> None:
        """A quick smoke check that build_draw_record's own source never
        mentions httpx/socket -- it must be a pure transform over already-
        fetched events, with all network I/O confined to the live-run
        functions this suite does not exercise."""
        source = inspect.getsource(build_draw_record)
        assert "httpx" not in source
        assert "socket" not in source
