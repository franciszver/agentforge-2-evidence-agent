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
from pathlib import Path

import pytest

from runner.issue_160_live_chat_harness import (
    ALLERGY_QUESTION,
    BP_QUESTION,
    ClaimSegmentRecord,
    DrawRecord,
    Question,
    ToolCallRecord,
    attribute_mechanisms,
    build_draw_record,
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
            error=None,
        )
        # #163 artifact-discipline parity: the schema has no field that
        # could hold raw answer/claim text -- only hash/length/counts/enums.
        field_names = set(DrawRecord.__dataclass_fields__)
        assert "answer" not in field_names
        assert "answer_text" not in field_names
        assert "claim_text" not in field_names
        assert draw.answer_hash == "deadbeef"


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
            events=_bp_events(),
            latency_seconds=2.5,
            http_status=200,
            error=None,
        )

        assert draw.question_id == BP_QUESTION.id
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
            question=ALLERGY_QUESTION, draw_index=1, events=events, latency_seconds=1.0, http_status=200, error=None
        )

        assert [tc.tool for tc in draw.tool_calls] == ["get_medications", "get_allergies"]
        assert [tc.order for tc in draw.tool_calls] == [0, 1]

    def test_error_draw_carries_no_answer_or_verdict(self) -> None:
        draw = build_draw_record(
            question=BP_QUESTION,
            draw_index=2,
            events=[],
            latency_seconds=180.0,
            http_status=None,
            error="ConnectTimeout",
        )

        assert draw.error == "ConnectTimeout"
        assert draw.answer_hash is None
        assert draw.answer_length is None
        assert draw.verdict is None
        assert draw.claim_count is None
        assert draw.tool_calls == []
        assert draw.claims == []


# --- tool_sequence_of / sequence_label ----------------------------------------


class TestToolSequence:
    def test_sequence_of_extracts_ordered_tool_names(self) -> None:
        draw = build_draw_record(
            question=BP_QUESTION, draw_index=0, events=_bp_events(), latency_seconds=1.0, http_status=200, error=None
        )

        assert tool_sequence_of(draw) == ("get_vitals",)

    def test_empty_sequence_for_error_draw(self) -> None:
        draw = build_draw_record(
            question=BP_QUESTION, draw_index=0, events=[], latency_seconds=1.0, http_status=None, error="boom"
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
) -> DrawRecord:
    tool_calls = [ToolCallRecord(order=i, tool=t, args={}, error=None) for i, t in enumerate(tools)]
    if error is not None:
        return DrawRecord(
            question_id="q",
            patient_id=1,
            draw_index=draw_index,
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
            error=error,
        )
    return DrawRecord(
        question_id="q",
        patient_id=1,
        draw_index=draw_index,
        correlation_id=f"corr-{draw_index}",
        conversation_id=f"conv-{draw_index}",
        tool_calls=tool_calls,
        answer_hash=hashlib.sha256(answer_text.encode()).hexdigest(),
        answer_length=len(answer_text),
        claim_count=1,
        claims=[ClaimSegmentRecord(index=0, passed=True, citation_count=1)],
        verdict=verdict,
        latency_seconds=1.0,
        http_status=200,
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


# --- no-tool-stub property -----------------------------------------------------


class TestNoToolStub:
    """Pins the #154-escaping property: this harness has no import path back
    into in-process/pinned-tool-data pipeline machinery."""

    def test_source_has_no_tool_stub_or_pipeline_import_lines(self) -> None:
        """Scans only actual ``import``/``from ... import`` statement lines
        (not prose/comments/docstrings, which legitimately discuss #154's
        tool_stub/pipeline machinery by name when explaining why THIS module
        avoids it)."""
        source = _HARNESS_MODULE_PATH.read_text(encoding="utf-8")
        import_lines = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith(("import ", "from "))
        ]

        forbidden_substrings = [
            "runner.tool_stub",
            "build_fake_registry",
            "runner.pipeline",
            "runner.schema",
            "app",  # catches "from app..."/"import app..." on any import line
        ]
        for line in import_lines:
            for needle in forbidden_substrings:
                assert needle not in line, (
                    f"harness import line {line!r} references {needle!r} -- reintroduces #154's blind spot"
                )

    def test_module_has_no_app_dot_star_imports_via_ast(self) -> None:
        import ast

        source = _HARNESS_MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("app"), f"forbidden in-process import: {node.module}"
                assert not node.module.startswith("runner.pipeline")
                assert not node.module.startswith("runner.tool_stub")
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
