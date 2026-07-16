"""Durable SQLite trace store: spans per chat invocation (P4.2).

One row per **span** -- request / tool / LLM / verification / feedback --
keyed by the P4.1 correlation id (``app.correlation.get_correlation_id()``),
carrying timings, ok/fail status, and type-specific non-PHI columns. This is
the durable home P2.10's in-memory ``ConversationStore`` and P3.7's
``to_trace_record`` seam both called out as deferred work.

**NO PHI ON DISK -- the load-bearing property of this module.** This store
persists to a SQLite file, so anything written here is a durable liability.
Only non-PHI data is ever stored:

  * correlation id, span type, timings, ok/fail status
  * an ``args_hash`` (HMAC-SHA256 of the tool call's args, via
    :func:`hash_args`) -- never the raw args dict. Keyed (not a bare
    SHA-256): tool args are often low-entropy/enumerable (a patient id, a
    closed-set filter key, a date range), so an unkeyed hash would let
    anyone with read access to this file precompute the hash over the
    plausible candidate space and recover the original args, defeating the
    whole point of hashing instead of storing raw. The key is
    ``Settings.trace_args_hash_secret`` -- injected into ``TraceStore``
    at construction, same as ``db_path``.
  * model name, token counts, tool name (all closed-set / non-identifying)
  * verdict + claim/stripped COUNTS -- never claim text or citation values
  * feedback thumb + a user-authored comment ABOUT THE RESPONSE (explicitly
    permitted -- it is not patient record data)

Raw tool args, raw tool results, the question/answer text, and any patient
record value (drug names, allergy substances, lab values, free text) are
never passed to this module in the first place -- see ``record_tool_span``,
which accepts a raw ``args`` mapping only to immediately hash it and discard
the original.

**Schema.** A single ``spans`` table, nullable per span type (rather than
five separate tables): span count per invocation is small (4-5 rows), there
is exactly one physical shape to migrate, and ``get_spans`` returns every
span for a correlation id with one query. An index on ``correlation_id``
backs that query and the P4.5 dashboard / review-queue lookups. A JSON
``details`` blob was considered and rejected -- named columns keep the
no-PHI columns individually inspectable (and testable, see
``tests/test_trace_store.py``'s raw-bytes scan) without parsing JSON to
audit them.

**Concurrency.** ``app.chat._stream_chat`` runs in Starlette's worker-thread
pool (see ``app.correlation`` module docstring), so writes can happen from
different threads. Each ``record_*`` call opens its own short-lived
``sqlite3.connect()``, writes, and closes -- no shared connection/lock to
manage, and SQLite's own busy-timeout handles the rare write/write race.
Writes are single-row inserts; this is not a hot path.

**Timing.** No injected clock: every ``record_*`` method takes ``start_ts``/
``end_ts`` (``time.time()``-style floats) as plain arguments rather than
calling a clock internally. This is what makes the writer tests fully
deterministic (pass fixed floats, assert exact ``duration_ms``) without a
``ClockInterface``-style seam -- callers (``app.chat``) read the wall clock
at their own call sites, which is also where an injected clock would need to
live for `their` tests to be deterministic if that mattered there.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id TEXT NOT NULL,
    span_type TEXT NOT NULL,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    duration_ms REAL NOT NULL,
    status TEXT NOT NULL,
    tool_name TEXT,
    args_hash TEXT,
    model TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    verdict TEXT,
    claim_count INTEGER,
    stripped_count INTEGER,
    feedback_thumb TEXT,
    feedback_comment TEXT,
    error_category TEXT
)
"""
_INDEX = "CREATE INDEX IF NOT EXISTS idx_spans_correlation_id ON spans (correlation_id)"

_COLUMNS = (
    "id",
    "correlation_id",
    "span_type",
    "start_ts",
    "end_ts",
    "duration_ms",
    "status",
    "tool_name",
    "args_hash",
    "model",
    "tokens_in",
    "tokens_out",
    "verdict",
    "claim_count",
    "stripped_count",
    "feedback_thumb",
    "feedback_comment",
    "error_category",
)


class SpanType(StrEnum):
    """Which stage of a chat invocation a span records."""

    REQUEST = "request"
    TOOL = "tool"
    LLM = "llm"
    VERIFICATION = "verification"
    FEEDBACK = "feedback"


class SpanStatus(StrEnum):
    OK = "ok"
    FAIL = "fail"


class FeedbackThumb(StrEnum):
    UP = "up"
    DOWN = "down"


@dataclass(frozen=True)
class Span:
    """One persisted span row. Columns not meaningful for ``span_type`` are ``None``."""

    id: int
    correlation_id: str
    span_type: SpanType
    start_ts: float
    end_ts: float
    duration_ms: float
    status: SpanStatus
    tool_name: str | None
    args_hash: str | None
    model: str | None
    tokens_in: int | None
    tokens_out: int | None
    verdict: str | None
    claim_count: int | None
    stripped_count: int | None
    feedback_thumb: FeedbackThumb | None
    feedback_comment: str | None
    error_category: str | None


def hash_args(args: Mapping[str, Any], secret: str) -> str:
    """HMAC-SHA256 hex digest of ``args``, keyed by ``secret``, order-independent.

    Used everywhere a tool call's args must be persisted without ever
    storing the raw values (which may carry patient data, e.g. a
    model-supplied filter echoing record content). Keyed rather than a bare
    hash -- see module docstring's "NO PHI ON DISK" section for why an
    unkeyed hash is not a safe substitute for the raw value here.
    """
    canonical = json.dumps(dict(args), sort_keys=True, default=str)
    return hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()


_ALWAYS_COLUMNS = ("correlation_id", "span_type", "start_ts", "end_ts", "duration_ms", "status")
# Every span-type-specific column, derived from ``_COLUMNS`` (single source of
# truth) rather than re-listed -- see ``TraceStore._insert``.
_OPTIONAL_COLUMNS = tuple(c for c in _COLUMNS if c not in ("id", *_ALWAYS_COLUMNS))


def _status(ok: bool) -> SpanStatus:
    return SpanStatus.OK if ok else SpanStatus.FAIL


def _row_to_span(row: tuple[Any, ...]) -> Span:
    """Build a ``Span`` from a raw row, ``_COLUMNS``-ordered. Field names match
    ``_COLUMNS`` 1:1, so only the enum-typed columns need converting."""
    values: dict[str, Any] = dict(zip(_COLUMNS, row))
    values["span_type"] = SpanType(values["span_type"])
    values["status"] = SpanStatus(values["status"])
    if values["feedback_thumb"] is not None:
        values["feedback_thumb"] = FeedbackThumb(values["feedback_thumb"])
    return Span(**values)


class TraceStore:
    """Durable per-invocation span writer/reader, backed by a SQLite file.

    Args:
        db_path: Path to the SQLite database file. Injectable so production
            points at ``Settings.trace_db_path`` (``/data/traces.db``) and
            every test points at a ``tmp_path`` file -- see the hard
            test-isolation rule in ``docs/TEST_PLAN.md`` Sec 7.
        hash_secret: HMAC key for :func:`hash_args`. Injectable for the same
            reason as ``db_path`` -- production supplies
            ``Settings.trace_args_hash_secret``; tests supply any fixed
            string, since only self-consistency (same secret hashes the
            same args identically) matters for a test double.

    Schema creation is idempotent: safe to construct against the same path
    repeatedly (e.g. once per process, or once per test).
    """

    def __init__(self, db_path: str, *, hash_secret: str) -> None:
        self._db_path = db_path
        self._hash_secret = hash_secret
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def db_path(self) -> str:
        """The SQLite file path this store reads/writes -- exposed for the
        P4.5 dashboard's read-only aggregation queries
        (``app.dashboard_metrics.compute_dashboard_metrics``), which open
        their own connection against the same file rather than going through
        this class's per-correlation-id ``get_spans``."""
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_schema(self) -> None:
        connection = self._connect()
        try:
            connection.execute(_SCHEMA)
            connection.execute(_INDEX)
            connection.commit()
        finally:
            connection.close()

    def _insert(self, *, span_type: SpanType, correlation_id: str, start_ts: float, end_ts: float, status: SpanStatus, **type_specific: Any) -> int:
        duration_ms = (end_ts - start_ts) * 1000
        row: dict[str, Any] = dict.fromkeys(_OPTIONAL_COLUMNS)
        row.update(
            correlation_id=correlation_id,
            span_type=span_type.value,
            start_ts=start_ts,
            end_ts=end_ts,
            duration_ms=duration_ms,
            status=status.value,
        )
        row.update(type_specific)

        columns = [c for c in _COLUMNS if c != "id"]
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT INTO spans ({', '.join(columns)}) VALUES ({placeholders})"

        connection = self._connect()
        try:
            cursor = connection.execute(sql, [row[c] for c in columns])
            connection.commit()
            last_row_id = cursor.lastrowid
            if last_row_id is None:
                raise RuntimeError("INSERT into spans did not return a lastrowid")
            return last_row_id
        finally:
            connection.close()

    def record_request_span(self, *, correlation_id: str, start_ts: float, end_ts: float, ok: bool) -> int:
        """Record the whole-invocation span (one per ``POST /chat`` call)."""
        return self._insert(
            span_type=SpanType.REQUEST,
            correlation_id=correlation_id,
            start_ts=start_ts,
            end_ts=end_ts,
            status=_status(ok),
        )

    def record_tool_span(
        self,
        *,
        correlation_id: str,
        start_ts: float,
        end_ts: float,
        ok: bool,
        tool_name: str,
        args: Mapping[str, Any],
        error_category: str | None = None,
    ) -> int:
        """Record one planner tool dispatch. ``args`` is hashed via
        :func:`hash_args` and never stored raw -- see module docstring."""
        return self._insert(
            span_type=SpanType.TOOL,
            correlation_id=correlation_id,
            start_ts=start_ts,
            end_ts=end_ts,
            status=_status(ok),
            tool_name=tool_name,
            args_hash=hash_args(args, self._hash_secret),
            error_category=error_category,
        )

    def record_llm_span(
        self,
        *,
        correlation_id: str,
        start_ts: float,
        end_ts: float,
        ok: bool,
        model: str,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> int:
        """Record one Ollama call (planner turn, quarantine summary, or extraction)."""
        return self._insert(
            span_type=SpanType.LLM,
            correlation_id=correlation_id,
            start_ts=start_ts,
            end_ts=end_ts,
            status=_status(ok),
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    def record_verification_span(
        self,
        *,
        correlation_id: str,
        start_ts: float,
        end_ts: float,
        ok: bool,
        verdict: str,
        claim_count: int,
        stripped_count: int,
    ) -> int:
        """Record the ``app.verdict.compute_verdict`` fold for one response.
        ``verdict`` is the ``Verdict`` enum value string; claim counts only,
        never claim text (see ``app.verdict.to_trace_record`` for the same
        shape at the pure-function layer)."""
        return self._insert(
            span_type=SpanType.VERIFICATION,
            correlation_id=correlation_id,
            start_ts=start_ts,
            end_ts=end_ts,
            status=_status(ok),
            verdict=verdict,
            claim_count=claim_count,
            stripped_count=stripped_count,
        )

    def record_feedback_span(
        self,
        *,
        correlation_id: str,
        start_ts: float,
        end_ts: float,
        feedback_thumb: FeedbackThumb,
        feedback_comment: str | None,
    ) -> int:
        """Record clinician feedback on a response (P4.3's ``/feedback``
        endpoint seam -- not wired here). ``feedback_comment`` is
        user-authored text ABOUT THE RESPONSE, not patient record data, so
        it is stored verbatim (see module docstring). Always ``ok`` --
        writing a feedback span IS the success event; there is no
        underlying operation for it to have failed."""
        return self._insert(
            span_type=SpanType.FEEDBACK,
            correlation_id=correlation_id,
            start_ts=start_ts,
            end_ts=end_ts,
            status=SpanStatus.OK,
            feedback_thumb=feedback_thumb.value,
            feedback_comment=feedback_comment,
        )

    def get_spans(self, correlation_id: str) -> list[Span]:
        """All spans recorded for ``correlation_id``, in insertion order."""
        connection = self._connect()
        try:
            cursor = connection.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM spans WHERE correlation_id = ? ORDER BY id",
                (correlation_id,),
            )
            return [_row_to_span(row) for row in cursor.fetchall()]
        finally:
            connection.close()
