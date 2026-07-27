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
  * model name, token counts, tool name, worker name, sub-task TYPE name
    (all closed-set / non-identifying)
  * verdict + claim/stripped COUNTS -- never claim text or citation values
  * feedback thumb + a user-authored comment ABOUT THE RESPONSE -- persisting
    it here is permitted (it is not a patient RECORD value pulled from a
    tool), but the comment itself is free text a clinician typed and MAY
    CONTAIN PHI incidentally (e.g. a patient name typed inline while
    describing a failure). ``app.review_queue``'s module docstring states
    this explicitly, and issue #176 corrected a previously-false claim that
    this field carries no PHI: it is not rendered on any page for that
    reason (``app.review_page``'s ``/review`` redacts it; the P4.5 dashboard
    never rendered it; ``/review/promote`` never re-emits it into the public
    ``evals/`` repo, #157). It stays on disk here only.
  * ownership, recorded on the ``REQUEST`` and ``FEEDBACK`` spans in ONE of
    two mutually-exclusive regimes, explicitly tagged by ``owner_kind`` (see
    :class:`OwnerKind`) so a reader never has to guess which one wrote a
    given row:

      - ``owner_kind=token_hash`` (#180, the ``copilot_per_user_token_enabled``
        flag-OFF/dev-bridge regime): ``owner_token_hash`` is an HMAC-SHA256 of
        the request's bearer token, via :func:`hash_owner_token` -- never the
        raw token. Same keyed-hash discipline as ``args_hash`` (a bearer
        token is a secret-shaped value, not PHI, but read access to this file
        must not be enough to correlate/replay it), and the same
        ``Settings.trace_args_hash_secret`` key. Chosen for this regime
        because the dev bridge's ``username`` claim is forgeable (see
        :func:`hash_owner_token`'s docstring) and no other checkable
        principal is available.
      - ``owner_kind=subject`` (#185, the ``copilot_per_user_token_enabled``
        flag-ON regime): ``owner_subject`` is OpenEMR's signature-verified
        introspection ``sub`` claim (``app.openemr_auth.IntrospectionResult.sub``)
        stored VERBATIM, not hashed. Deliberately unkeyed, unlike every other
        column in this list: ``sub`` is a stable per-user identifier, not a
        secret an attacker could replay, so the keyed-hash discipline above
        buys nothing here and would actively break the very case #185 exists
        to fix -- a hash keyed by ``Settings.trace_args_hash_secret`` (which
        has no committed default and is regenerated randomly on any restart
        where it is left unset, see :func:`hash_owner_token`'s docstring)
        would make subject-based ownership rot on every such restart exactly
        like the token-hash regime does, defeating the whole point of
        binding to a durable per-user identity instead of a per-process
        secret.

    ``caller_owns_trace`` (the ``POST /feedback`` ownership check) dispatches
    on each row's own ``owner_kind`` -- see that method's docstring for the
    full mixed-regime matrix (a row written under one regime is NEVER
    claimable via the other, and an unrecorded/ambiguous owner is always
    rejected, never treated as open).

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
import logging
import sqlite3
from collections.abc import Callable, Mapping
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
    error_category TEXT,
    span_id TEXT,
    parent_span_id TEXT,
    worker_name TEXT,
    sub_task_type TEXT,
    owner_token_hash TEXT,
    owner_kind TEXT,
    owner_subject TEXT
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
    "span_id",
    "parent_span_id",
    "worker_name",
    "sub_task_type",
    "owner_token_hash",
    "owner_kind",
    "owner_subject",
)


class SpanType(StrEnum):
    """Which stage of a chat invocation a span records."""

    REQUEST = "request"
    TOOL = "tool"
    LLM = "llm"
    VERIFICATION = "verification"
    FEEDBACK = "feedback"
    # P3.8: one span per supervisor->worker handoff (app.supervisor), so the
    # per-encounter record (app.encounter_observability) can read the
    # ordered worker sequence + P3.5 span tree back out of this same durable
    # sink instead of a parallel store. ``worker_name``/``sub_task_type`` are
    # closed-set/non-identifying (a worker's stable name, a sub-task class
    # name), same discipline as ``tool_name`` -- never a sub-task field value.
    WORKER = "worker"


class SpanStatus(StrEnum):
    OK = "ok"
    FAIL = "fail"


class FeedbackThumb(StrEnum):
    UP = "up"
    DOWN = "down"


class OwnerKind(StrEnum):
    """Which ownership regime wrote a ``REQUEST``/``FEEDBACK`` span's owner
    columns (#185). Explicit and persisted -- NOT inferred from which of
    ``owner_token_hash``/``owner_subject`` happens to be non-``NULL`` -- so a
    reader can never conflate "hash of a bearer token" (``TOKEN_HASH``, #180)
    with "a signature-verified OpenEMR subject id" (``SUBJECT``, #185): the
    two are different kinds of value with different security properties
    (a token hash is keyed/secret-shaped and reissue-sensitive; a subject id
    is a stable, non-secret identifier), and overloading one column to carry
    either would make a mixed-regime DB ambiguous by construction. ``None``
    on the ``Span`` (no ``owner_kind`` column value) means either no owner
    was recorded at all, or the row predates #185 (see
    ``TraceStore.caller_owns_trace``'s docstring for how that legacy case is
    handled -- inferred as ``TOKEN_HASH`` only when ``owner_token_hash`` is
    itself present, never treated as ``SUBJECT``).
    """

    TOKEN_HASH = "token_hash"
    SUBJECT = "subject"


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
    span_id: str | None = None
    parent_span_id: str | None = None
    worker_name: str | None = None
    sub_task_type: str | None = None
    owner_token_hash: str | None = None
    owner_kind: str | None = None
    owner_subject: str | None = None


def _keyed_digest(secret: str, data: str) -> str:
    """HMAC-SHA256 hex digest of ``data``, keyed by ``secret``. The shared
    primitive behind :func:`hash_args` and :func:`hash_owner_token` -- both
    reduce to exactly this call once their own input is turned into a
    string (``hash_args`` canonicalises a ``Mapping`` to JSON first;
    ``hash_owner_token`` passes the token straight through), so the actual
    HMAC construction lives in one place rather than being duplicated
    identically in two.
    """
    return hmac.new(secret.encode(), data.encode(), hashlib.sha256).hexdigest()


def hash_args(args: Mapping[str, Any], secret: str) -> str:
    """HMAC-SHA256 hex digest of ``args``, keyed by ``secret``, order-independent.

    Used everywhere a tool call's args must be persisted without ever
    storing the raw values (which may carry patient data, e.g. a
    model-supplied filter echoing record content). Keyed rather than a bare
    hash -- see module docstring's "NO PHI ON DISK" section for why an
    unkeyed hash is not a safe substitute for the raw value here.
    """
    canonical = json.dumps(dict(args), sort_keys=True, default=str)
    return _keyed_digest(secret, canonical)


def hash_owner_token(token: str, secret: str) -> str:
    """HMAC-SHA256 hex digest of a bearer token, keyed by ``secret``.

    Issue #180: the ``/feedback`` ownership check binds a feedback write to
    the SAME bearer token that originated the trace (``record_request_span``'s
    ``owner_token`` argument), not to a claimed identity. This is the ONLY
    checkable per-user-session principal available in the
    ``copilot_per_user_token_enabled`` flag-OFF / dev-bridge regime: the dev
    token's ``username`` claim is HMAC'd with a per-session CSRF-derived
    signing key this agent never verifies (``DevAgentToken.php``,
    ``TokenBrokerController.php``), so it is forgeable and unusable for
    authorization (see ``app.feedback``'s module docstring). The front-end
    panel caches ONE token per browser session and reuses it for both
    ``/chat`` and ``/feedback`` (see
    ``interface/.../public/assets/js/copilot-chat.js``'s token-broker
    comment), so "same token" is exactly "same session that started this
    trace" -- the ownership question this regime needs answered.

    Issue #185 replaces this with subject-based ownership
    (``app.openemr_auth.IntrospectionResult.sub``, see that field's
    docstring for the signature-verification chain and OpenEMR source lines)
    once ``copilot_per_user_token_enabled`` is ON -- OpenEMR's introspection
    response DOES carry ``sub``, and it survives token reissue and service
    restart in ways a token hash structurally cannot (see below). This
    function and the token-hash regime remain the correct, and only, choice
    for the flag-off/dev-bridge case, where no verified subject is ever
    obtainable -- see ``TraceStore.record_request_span``'s docstring for how
    the two regimes are selected and kept apart on disk (``OwnerKind``).

    Keyed (not a bare hash) for the same reason as :func:`hash_args`: a
    bearer token is a secret-shaped value an attacker with read access to
    the trace store must not be able to dictionary-attack or correlate
    across records via an unkeyed digest. Inherited from that same keying
    choice: ``Settings.trace_args_hash_secret`` has no committed default
    (a random 32 bytes generated per process when unset), so if it is left
    unset a service restart rotates the key and every trace recorded before
    the restart becomes unownable -- ``caller_owns_trace`` will reject even
    the legitimate original caller's token, since the stored hash was keyed
    with the now-rotated secret. Fail-closed and consistent with
    ``caller_owns_trace``'s stated direction (an unrecorded/unverifiable
    owner is rejected, never treated as open) -- but this IS a new,
    user-visible consequence of that unset-secret default, not a pre-existing
    one: nothing reads ``args_hash`` back across a process lifetime (it is
    written once and only ever compared within the same write, never looked
    up later), so a secret rotation there was previously unobservable.
    ``owner_token_hash`` is looked up and compared on a LATER request, so an
    unset ``trace_args_hash_secret`` paired with a trace store that outlives
    the process (a persistent ``/data`` volume) is now a misconfiguration
    with a concrete, user-visible failure mode: a clinician's own legitimate
    feedback on a pre-restart answer gets rejected. Deployments that persist
    ``traces.db`` across restarts should pin ``trace_args_hash_secret``
    explicitly.
    """
    return _keyed_digest(secret, token)


def record_span_best_effort(logger: logging.Logger, operation: str, write: Callable[[], object]) -> None:
    """Emit one trace span, best-effort. Observability must NEVER crash the
    caller: a failed span write (a root-owned ``/data`` -> ``PermissionError``,
    a full disk, a locked DB) is logged (correlation-tagged by the
    ``app.correlation`` logging seam; the payload carries only the operation
    label, never PHI) and swallowed. Shared by every span-writing caller --
    ``app.chat`` (tool/llm spans) and ``app.supervisor`` (worker spans) --
    so the best-effort discipline lives in exactly one place rather than
    being reimplemented per caller. ``logger`` is the CALLER's own logger
    (not this module's), so log lines keep attributing to the module that
    triggered the write.

    Catches ``Exception`` (not ``BaseException``): a ``GeneratorExit`` raised
    while a write runs in a caller's ``finally`` block is a client
    disconnect and must keep propagating, not be swallowed here.
    """
    try:
        write()
    except Exception:
        logger.warning(
            "trace span write failed; continuing without it",
            extra={"operation": operation},
            exc_info=True,
        )


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
            self._migrate_missing_columns(connection)
            connection.commit()
        finally:
            connection.close()

    def _migrate_missing_columns(self, connection: sqlite3.Connection) -> None:
        """Idempotent migration for an EXISTING ``spans`` table predating a
        column this version of the schema adds (P3.8's ``span_id``/
        ``parent_span_id``/``worker_name``/``sub_task_type``). ``CREATE
        TABLE IF NOT EXISTS`` alone is a no-op against a pre-existing table
        with an older column set -- without this, every write naming a
        missing column raises ``sqlite3.OperationalError``, which
        ``record_span_best_effort`` swallows: tracing would silently stop
        entirely against an un-migrated production DB. SQLite's
        ``ALTER TABLE ... ADD COLUMN`` is safe here: every added column is
        nullable with no default, so existing rows are unaffected. A no-op
        on a fresh/already-migrated DB (nothing missing to add).

        Racing-migration guard (#180 review finding): ``get_trace_store``
        (``app.chat``) is a lazily-built, non-atomic check-then-set process
        global, resolved from Starlette's worker-thread pool by every
        route that depends on it -- so two concurrent FIRST requests
        against a pre-existing DB that still needs a column can both
        construct a ``TraceStore``, both read the same pre-migration
        ``PRAGMA table_info`` result here, and both attempt the SAME
        ``ALTER TABLE ... ADD COLUMN``. SQLite has no ``IF NOT EXISTS`` for
        ``ADD COLUMN``, so the loser raises
        ``sqlite3.OperationalError: duplicate column name`` -- uncaught,
        that propagates out of ``__init__`` and turns dependency resolution
        into a 500 for a request that did nothing wrong. Harmless on the
        shipped stack (a fresh, empty DB has nothing to migrate on either
        thread's first run) but live for any deployment where ``/data``
        persists a pre-#180 (or otherwise older-schema) DB across a
        restart. Caught and ignored here: the loser's column is already
        there courtesy of the winner, which is exactly the end state this
        method is trying to reach either way."""
        existing_columns = {row[1] for row in connection.execute("PRAGMA table_info(spans)")}
        for column in _COLUMNS:
            if column in ("id", *_ALWAYS_COLUMNS) or column in existing_columns:
                continue
            try:
                connection.execute(f"ALTER TABLE spans ADD COLUMN {column} TEXT")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc):
                    raise

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

    def _owner_columns(
        self, *, owner_token: str | None, owner_subject: str | None
    ) -> dict[str, str | None]:
        """Resolve the ``owner_kind``/``owner_token_hash``/``owner_subject``
        columns for a ``REQUEST`` or ``FEEDBACK`` write (#185).

        Exactly ONE regime is ever persisted per row, chosen here so
        ``record_request_span`` and ``record_feedback_span`` can't drift
        apart on the selection logic:

        * ``owner_subject`` truthy -> ``OwnerKind.SUBJECT``: stored verbatim
          (see module docstring for why this one is NOT hashed);
          ``owner_token`` is ignored even if also supplied -- a row is never
          allowed to carry both a subject and a token hash, which is exactly
          the "overloaded column" ambiguity #185 exists to rule out.
        * else ``owner_token`` truthy -> ``OwnerKind.TOKEN_HASH`` (#180):
          hashed via :func:`hash_owner_token`, never the raw token.
        * else (neither supplied) -> no owner recorded at all (``owner_kind``
          ``None``); :meth:`caller_owns_trace` treats that as UNOWNABLE, not
          open to anyone.
        """
        if owner_subject:
            return {
                "owner_kind": OwnerKind.SUBJECT.value,
                "owner_subject": owner_subject,
                "owner_token_hash": None,
            }
        if owner_token:
            return {
                "owner_kind": OwnerKind.TOKEN_HASH.value,
                "owner_subject": None,
                "owner_token_hash": hash_owner_token(owner_token, self._hash_secret),
            }
        return {"owner_kind": None, "owner_subject": None, "owner_token_hash": None}

    def record_request_span(
        self,
        *,
        correlation_id: str,
        start_ts: float,
        end_ts: float,
        ok: bool,
        owner_token: str | None = None,
        owner_subject: str | None = None,
    ) -> int:
        """Record the whole-invocation span (one per ``POST /chat`` call).

        Ownership (#180/#185) is recorded in exactly ONE of two regimes --
        see :meth:`_owner_columns` for the selection and ``OwnerKind`` /
        the module docstring for why they are never mixed on one row:

        * ``owner_subject`` (#185): OpenEMR's signature-verified
          introspection ``sub`` claim, when ``copilot_per_user_token_enabled``
          is on -- stored verbatim as ``owner_subject``.
        * ``owner_token`` (#180): the caller's raw bearer token, hashed via
          :func:`hash_owner_token` and stored as ``owner_token_hash`` --
          never the raw token. Used for the flag-off/dev-bridge regime,
          where no verified subject is available.

        This is what :meth:`caller_owns_trace` later checks a ``/feedback``
        caller's own token/subject against. Neither argument supplied
        records no owner at all -- reachable from a caller that predates
        #180 (a test, or -- deliberately -- nothing else in production), and
        ``caller_owns_trace`` treats a trace with no recorded owner as
        UNOWNABLE (rejects every claimant), not as open to anyone -- see
        that method's docstring.
        """
        return self._insert(
            span_type=SpanType.REQUEST,
            correlation_id=correlation_id,
            start_ts=start_ts,
            end_ts=end_ts,
            status=_status(ok),
            **self._owner_columns(owner_token=owner_token, owner_subject=owner_subject),
        )

    def caller_owns_trace(self, correlation_id: str, token: str, *, subject: str | None = None) -> bool:
        """#180/#185: does this caller match the principal that originated
        ``correlation_id``'s trace?

        ``token`` is always the caller's raw bearer token (the #180 proof of
        session-possession). ``subject`` is the caller's OpenEMR introspection
        ``sub`` (``app.openemr_auth.IntrospectionResult.sub``), passed by
        ``app.feedback.feedback_endpoint`` when ``copilot_per_user_token_enabled``
        is on and ``None`` otherwise (see ``app.chat.get_subject_resolver``).

        Dispatches on the ORIGINATING ``REQUEST`` span's own ``owner_kind`` --
        never on the CURRENT caller's flag state or which argument they
        supplied -- so a row written under one regime can only ever be
        claimed under that SAME regime, regardless of what the flag does
        later. This is the full mixed-regime matrix (#185):

        * ``owner_kind=subject`` row + caller has a ``subject`` (flag ON):
          compared against ``owner_subject``. Two different tokens/sessions
          for the SAME subject both match; a different subject never does.
        * ``owner_kind=subject`` row + caller has NO ``subject`` (flag turned
          back OFF): rejected outright, even for the original clinician --
          there is nothing here for the flag-off regime to safely compare
          (it has no verified subject at all), and falling back to a token
          comparison would mean this row is claimable under either regime,
          exactly the ambiguity #185 must not introduce. Fails closed.
        * ``owner_kind=token_hash`` row (#180, or #185 with the flag off) --
          ALWAYS compared via ``token``, regardless of the current flag or
          whether the caller also supplied a ``subject``. A flag flipped ON
          after this row was written does not make it subject-claimable (no
          ``owner_subject`` was ever recorded for it, so no new claimant
          becomes possible) -- it remains exactly as claimable as it always
          was: by the original token holder, and no one else.
        * ``owner_kind`` is ``None`` on the stored row (legacy): if
          ``owner_token_hash`` is present, the row PREDATES #185's
          ``owner_kind`` column (written by #180-era code, then migrated in
          -- see ``_migrate_missing_columns``) and is treated exactly as
          ``token_hash`` above, so upgrading to #185 never silently
          re-litigates ownership of rows #180 already decided. If
          ``owner_token_hash`` is ALSO absent, there is no recorded owner at
          all (no ``REQUEST`` span, a write that failed, or a caller that
          passed neither) -> rejected. An unrecorded/ambiguous originator is
          never treated as fair game, only as unclaimable.

        The two regimes use different comparisons, deliberately. ``subject``
        is a non-secret identifier (an OpenEMR ``sub`` UUID) -- it is not
        derived from anything an attacker shouldn't already know, so a plain
        ``==`` is used: nothing is protected by constant-time comparison here,
        and ``hmac.compare_digest`` requires both operands to be ASCII-only
        (``bytes``, or ``str`` restricted to the ASCII subset), which a
        non-ASCII ``subject`` would violate, raising ``TypeError`` instead of
        returning ``False``. ``owner_token_hash`` IS secret-derived (see
        ``hash_owner_token``), so that comparison stays on
        ``hmac.compare_digest`` to avoid a timing side channel on the hash.

        First REQUEST span wins, by design -- security-load-bearing, not
        arbitrary. ``correlation_id`` is attacker-influenceable (an inbound
        ``X-Correlation-ID`` header is honored verbatim, see
        ``app.correlation.CorrelationIdMiddleware``), so a caller with
        network access to this agent could ``POST /chat`` with a foreign,
        already-in-use correlation id and get a SECOND ``REQUEST`` span
        appended for it, carrying their own owner columns. Taking the LAST
        span here (mirroring ``app.review_queue``'s ``[-1]`` "most recent
        wins" convention for other span lookups) would let that second span
        silently replace the original owner -- an ownership-TAKEOVER
        primitive, not just a failed forgery. Taking the FIRST span instead
        keeps the original ``/chat`` caller authoritative forever, regardless
        of what a later request appends under the same id. Do not
        "simplify" this to match ``review_queue``'s ``[-1]`` convention
        without re-reading this paragraph first.
        """
        request_spans = [span for span in self.get_spans(correlation_id) if span.span_type == SpanType.REQUEST]
        if not request_spans:
            return False
        origin = request_spans[0]
        effective_kind = origin.owner_kind or (OwnerKind.TOKEN_HASH.value if origin.owner_token_hash else None)
        if effective_kind == OwnerKind.SUBJECT.value:
            if not subject:
                return False
            # A NULL ``origin.owner_subject`` (corrupt/hand-edited row) is not
            # special-cased: it simply fails the equality check below, same as
            # any other mismatch. ``None == subject`` is always ``False`` for
            # a non-empty ``subject`` string, so this stays fail-closed
            # without an explicit ``origin.owner_subject`` guard.
            return origin.owner_subject == subject
        if effective_kind == OwnerKind.TOKEN_HASH.value:
            if not origin.owner_token_hash:
                return False
            return hmac.compare_digest(origin.owner_token_hash, hash_owner_token(token, self._hash_secret))
        return False

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
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> int:
        """Record one planner tool dispatch. ``args`` is hashed via
        :func:`hash_args` and never stored raw -- see module docstring.
        ``span_id``/``parent_span_id`` (P3.8) carry this span's place in the
        P3.5 span tree (``app.correlation.SpanContext``), same as
        ``record_worker_span`` -- optional/nullable so a caller with no
        ambient span (nothing currently open in context) is unaffected."""
        return self._insert(
            span_type=SpanType.TOOL,
            correlation_id=correlation_id,
            start_ts=start_ts,
            end_ts=end_ts,
            status=_status(ok),
            tool_name=tool_name,
            args_hash=hash_args(args, self._hash_secret),
            error_category=error_category,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )

    def record_worker_span(
        self,
        *,
        correlation_id: str,
        start_ts: float,
        end_ts: float,
        ok: bool,
        worker_name: str,
        sub_task_type: str,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        error_category: str | None = None,
    ) -> int:
        """Record one supervisor->worker handoff (P3.5/P3.8). ``worker_name``
        (a stable worker identifier, e.g. ``"evidence-retriever"``) and
        ``sub_task_type`` (the sub-task's TYPE name, e.g.
        ``"RetrieveSubTask"``) are the same closed-set, non-PHI fields
        ``app.supervisor``'s ``_log_handoff`` already logs -- never a
        sub-task field value. ``span_id``/``parent_span_id`` carry the P3.5
        span tree (``app.correlation.SpanContext``) so the per-encounter
        record can reconstruct supervisor->worker parenting from this store
        alone."""
        return self._insert(
            span_type=SpanType.WORKER,
            correlation_id=correlation_id,
            start_ts=start_ts,
            end_ts=end_ts,
            status=_status(ok),
            worker_name=worker_name,
            sub_task_type=sub_task_type,
            span_id=span_id,
            parent_span_id=parent_span_id,
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
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> int:
        """Record one Ollama call (planner turn, quarantine summary, or
        extraction). ``span_id``/``parent_span_id`` (P3.8) carry this span's
        place in the P3.5 span tree, same as ``record_tool_span``."""
        return self._insert(
            span_type=SpanType.LLM,
            correlation_id=correlation_id,
            start_ts=start_ts,
            end_ts=end_ts,
            status=_status(ok),
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            span_id=span_id,
            parent_span_id=parent_span_id,
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
        owner_token: str | None = None,
        owner_subject: str | None = None,
    ) -> int:
        """Record clinician feedback on a response (P4.3's ``/feedback``
        endpoint seam). ``feedback_comment`` is user-authored text ABOUT THE
        RESPONSE, not a patient record value pulled from a tool, but it may
        incidentally contain PHI (a clinician typed a patient detail inline)
        -- see the module docstring, #176. It is still stored verbatim:
        persisting it here is permitted, only rendering it is not. Always
        ``ok`` -- writing a feedback span IS the success event; there is no
        underlying operation for it to have failed.

        ``owner_token``/``owner_subject`` (#180/#185): the SAME values
        ``app.feedback.feedback_endpoint`` already checked via
        ``caller_owns_trace`` before calling this -- resolved to exactly one
        regime by :meth:`_owner_columns` (same logic as
        ``record_request_span``) and stored on this FEEDBACK span too,
        purely for attribution, not enforcement (ownership was already
        decided by the time this is called). Without it, a feedback row
        would be indistinguishable across regimes/eras in a DB upgraded in
        place, so a triager reading a disputed comment has no way to tell
        which regime produced it."""
        return self._insert(
            span_type=SpanType.FEEDBACK,
            correlation_id=correlation_id,
            start_ts=start_ts,
            end_ts=end_ts,
            status=SpanStatus.OK,
            feedback_thumb=feedback_thumb.value,
            feedback_comment=feedback_comment,
            **self._owner_columns(owner_token=owner_token, owner_subject=owner_subject),
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
