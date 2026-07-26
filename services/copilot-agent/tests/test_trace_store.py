"""Hermetic tests for the durable SQLite trace store (P4.2).

Every test uses a fresh ``tmp_path`` database -- NEVER the configured
``trace_db_path`` / dev ``traces.db`` (see ``docs/TEST_PLAN.md`` Sec 7,
"agent-service tests write only to per-test temporary SQLite databases").

The no-PHI tests are the load-bearing ones here: the store persists to disk,
so anything written is a durable liability. They assert -- by inspecting the
raw database bytes, not just the typed accessors -- that a value passed as
tool ``args`` never appears verbatim anywhere in the file.
"""

from __future__ import annotations

import secrets
import sqlite3
from pathlib import Path

import pytest

from app.trace_store import FeedbackThumb, Span, SpanStatus, SpanType, TraceStore, hash_args, hash_owner_token

# The exact pre-P3.8 schema (committed on main before this branch added
# span_id/parent_span_id/worker_name/sub_task_type) -- see
# services/copilot-agent/app/trace_store.py at commit e3c8c1d.
_PRE_P38_SCHEMA = """\
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

# Derived (not a hardcoded literal) so no secret-shaped string is committed;
# stable within a run, so the store fixture and the hash-equality assertions
# below share the SAME key and still prove HMAC keying.
_TEST_HASH_KEY = secrets.token_hex(16)


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "traces.db")


@pytest.fixture
def store(db_path: str) -> TraceStore:
    return TraceStore(db_path=db_path, hash_secret=_TEST_HASH_KEY)


def test_span_constructs_without_the_p38_fields_defaulting_to_none() -> None:
    """``evals/`` (a separate pytest root, run from the repo root as
    ``pytest evals/ -m "not integration"``) constructs ``Span`` directly
    with only the pre-P3.8 field set (see
    ``evals/runner/tests/test_review_queue_generator.py``) -- it has no
    reason to know about span_id/parent_span_id/worker_name/sub_task_type.
    Those four must be defaulted (``None``), not required positional args,
    or every such construction site breaks with ``TypeError: Span.__init__()
    missing 4 required positional arguments``."""
    span = Span(
        id=1,
        correlation_id="corr-1",
        span_type=SpanType.TOOL,
        start_ts=0.0,
        end_ts=1.0,
        duration_ms=1000.0,
        status=SpanStatus.OK,
        tool_name="get_medications",
        args_hash=None,
        model=None,
        tokens_in=None,
        tokens_out=None,
        verdict=None,
        claim_count=None,
        stripped_count=None,
        feedback_thumb=None,
        feedback_comment=None,
        error_category=None,
    )
    assert span.span_id is None
    assert span.parent_span_id is None
    assert span.worker_name is None
    assert span.sub_task_type is None


def test_schema_created_idempotently(db_path: str) -> None:
    # Constructing twice against the same path must not raise.
    TraceStore(db_path=db_path, hash_secret=_TEST_HASH_KEY)
    TraceStore(db_path=db_path, hash_secret=_TEST_HASH_KEY)


def test_migrates_a_pre_p38_db_so_new_columns_are_writable(db_path: str) -> None:
    """An existing production ``traces.db`` predating this branch has the
    pre-P3.8 column set (no span_id/parent_span_id/worker_name/
    sub_task_type). ``CREATE TABLE IF NOT EXISTS`` alone is a no-op against
    such a DB -- the columns never appear, and every write that names them
    raises ``sqlite3.OperationalError``, which ``record_span_best_effort``
    swallows: tracing silently stops entirely. Constructing a ``TraceStore``
    against such a DB must migrate it (idempotently) so writes naming the
    new columns succeed."""
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(_PRE_P38_SCHEMA)
        connection.commit()
    finally:
        connection.close()

    store = TraceStore(db_path=db_path, hash_secret=_TEST_HASH_KEY)

    # Must not raise sqlite3.OperationalError: no column named span_id/...
    store.record_worker_span(
        correlation_id="corr-migrate",
        start_ts=0.0,
        end_ts=1.0,
        ok=True,
        worker_name="evidence-retriever",
        sub_task_type="RetrieveSubTask",
        span_id="span-1",
        parent_span_id="span-0",
    )
    store.record_tool_span(
        correlation_id="corr-migrate",
        start_ts=1.0,
        end_ts=2.0,
        ok=True,
        tool_name="get_medications",
        args={},
        span_id="span-2",
        parent_span_id="span-0",
    )

    spans = store.get_spans("corr-migrate")
    worker_span = next(span for span in spans if span.span_type == SpanType.WORKER)
    tool_span = next(span for span in spans if span.span_type == SpanType.TOOL)
    assert worker_span.span_id == "span-1"
    assert tool_span.span_id == "span-2"

    # Re-constructing against the now-migrated DB must still not raise.
    TraceStore(db_path=db_path, hash_secret=_TEST_HASH_KEY)


def test_migration_tolerates_a_concurrent_racing_alter_table(db_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """#180 review MINOR 6: ``get_trace_store`` (``app.chat``) is a lazily
    built, non-atomic check-then-set process global, resolved from
    Starlette's worker-thread pool by every route that depends on it. Two
    concurrent FIRST requests against a persistent DB that still needs a
    column can both read ``PRAGMA table_info`` before either ``ALTER``s,
    and the loser's ``ALTER TABLE ... ADD COLUMN`` raises
    ``sqlite3.OperationalError: duplicate column name`` -- uncaught, that
    would propagate out of ``TraceStore.__init__`` and turn dependency
    resolution into an unhandled 500 for a request that did nothing wrong.

    Reproduced deterministically (no real threads / timing dependence): a
    SECOND, independent connection is spliced in to win the race and
    commit the identical ``ALTER`` first, right as the connection under
    test is about to run the same statement -- the exact interleaving two
    real racing threads could produce.
    """
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(_PRE_P38_SCHEMA)
        connection.commit()
    finally:
        connection.close()

    # ``sqlite3.Connection`` is an immutable C type -- its methods can't be
    # monkeypatched directly. Subclassing (via the ``factory`` argument
    # ``sqlite3.connect`` already supports) and patching the module-level
    # ``sqlite3.connect`` function (an ordinary, patchable Python callable)
    # is the supported way to intercept a connection's ``execute`` calls.
    original_connect = sqlite3.connect
    already_raced = False

    class _RacingConnection(sqlite3.Connection):
        def execute(self, sql: str, *args: object, **kwargs: object) -> sqlite3.Cursor:  # type: ignore[override]
            nonlocal already_raced
            if not already_raced and "ADD COLUMN owner_token_hash" in sql:
                already_raced = True
                # A plain connection via the ORIGINAL connect -- not the
                # patched one -- so this racer's own ALTER doesn't
                # recursively re-enter this override.
                racer = original_connect(db_path)
                try:
                    racer.execute(sql)
                    racer.commit()
                finally:
                    racer.close()
            return super().execute(sql, *args, **kwargs)

    def patched_connect(path: str, *args: object, **kwargs: object) -> sqlite3.Connection:
        return original_connect(path, *args, factory=_RacingConnection, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sqlite3, "connect", patched_connect)

    # Must NOT raise sqlite3.OperationalError: duplicate column name.
    store = TraceStore(db_path=db_path, hash_secret=_TEST_HASH_KEY)

    # And the migration actually completed despite the race -- the column
    # this test targeted, and every other missing one, is usable.
    store.record_request_span(correlation_id="corr-race", start_ts=0.0, end_ts=1.0, ok=True, owner_token="tok")
    span = store.get_spans("corr-race")[0]
    assert span.owner_token_hash is not None


def test_record_request_span_write_and_read_back(store: TraceStore) -> None:
    store.record_request_span(correlation_id="corr-1", start_ts=100.0, end_ts=100.25, ok=True)

    spans = store.get_spans("corr-1")

    assert len(spans) == 1
    span = spans[0]
    assert span.correlation_id == "corr-1"
    assert span.span_type == SpanType.REQUEST
    assert span.status == SpanStatus.OK
    assert span.duration_ms == pytest.approx(250.0)


def test_record_request_span_failure_status(store: TraceStore) -> None:
    store.record_request_span(correlation_id="corr-1", start_ts=1.0, end_ts=2.0, ok=False)

    span = store.get_spans("corr-1")[0]
    assert span.status == SpanStatus.FAIL


def test_record_request_span_hashes_owner_token_not_raw(store: TraceStore) -> None:
    # #180: the raw bearer token must never be written to disk, same
    # discipline as hash_args -- only its keyed hash.
    store.record_request_span(
        correlation_id="corr-owner", start_ts=0.0, end_ts=1.0, ok=True, owner_token="super-secret-token"
    )

    span = store.get_spans("corr-owner")[0]
    assert span.owner_token_hash is not None
    assert span.owner_token_hash != "super-secret-token"
    assert len(span.owner_token_hash) == 64  # sha256 hex digest


def test_record_request_span_without_owner_token_records_no_hash(store: TraceStore) -> None:
    store.record_request_span(correlation_id="corr-no-owner", start_ts=0.0, end_ts=1.0, ok=True)

    span = store.get_spans("corr-no-owner")[0]
    assert span.owner_token_hash is None


def test_caller_owns_trace_true_for_the_originating_token(store: TraceStore) -> None:
    store.record_request_span(
        correlation_id="corr-mine", start_ts=0.0, end_ts=1.0, ok=True, owner_token="clinician-a-token"
    )

    assert store.caller_owns_trace("corr-mine", "clinician-a-token") is True


def test_caller_owns_trace_false_for_a_different_token(store: TraceStore) -> None:
    store.record_request_span(
        correlation_id="corr-mine", start_ts=0.0, end_ts=1.0, ok=True, owner_token="clinician-a-token"
    )

    assert store.caller_owns_trace("corr-mine", "attacker-token") is False


def test_caller_owns_trace_false_when_no_request_span_exists(store: TraceStore) -> None:
    # Fail-closed: an id with no originating REQUEST span at all (never
    # recorded, e.g. guessed outright) is rejected, not left open.
    assert store.caller_owns_trace("corr-never-existed", "any-token") is False


def test_caller_owns_trace_false_when_request_span_recorded_no_owner(store: TraceStore) -> None:
    # Fail-closed: a REQUEST span that predates #180 (or was written by a
    # caller that passed no owner_token) carries no owner_token_hash --
    # must reject every claimant, not fall back to "unclaimed = open."
    store.record_request_span(correlation_id="corr-legacy", start_ts=0.0, end_ts=1.0, ok=True)

    assert store.caller_owns_trace("corr-legacy", "any-token") is False


def test_caller_owns_trace_first_request_span_wins_over_a_later_appended_one(store: TraceStore) -> None:
    # #180 review finding: correlation_id is attacker-influenceable (an
    # inbound X-Correlation-ID header is honored verbatim, see
    # app.correlation.CorrelationIdMiddleware), so a caller with network
    # access to this agent could POST /chat with a foreign, already-in-use
    # correlation id and get a SECOND REQUEST span appended for it, under
    # their OWN token. caller_owns_trace deliberately keeps the FIRST
    # span's owner authoritative forever -- taking the last one (the
    # convention app.review_queue uses for OTHER span lookups) would turn
    # this check into an ownership-takeover primitive instead of a defense
    # against one. Both assertions matter: the original owner still owns
    # it, AND the second span's own token does NOT gain ownership.
    store.record_request_span(
        correlation_id="corr-shared", start_ts=0.0, end_ts=1.0, ok=True, owner_token="clinician-a-token"
    )
    store.record_request_span(
        correlation_id="corr-shared", start_ts=2.0, end_ts=3.0, ok=True, owner_token="attacker-token"
    )

    assert store.caller_owns_trace("corr-shared", "clinician-a-token") is True
    assert store.caller_owns_trace("corr-shared", "attacker-token") is False


def test_record_tool_span_write_and_read_back(store: TraceStore) -> None:
    store.record_tool_span(
        correlation_id="corr-2",
        start_ts=10.0,
        end_ts=10.5,
        ok=True,
        tool_name="get_medications",
        args={"limit": 3},
    )

    span = store.get_spans("corr-2")[0]
    assert span.span_type == SpanType.TOOL
    assert span.tool_name == "get_medications"
    assert span.error_category is None
    assert span.args_hash is not None
    assert len(span.args_hash) == 64  # sha256 hex digest


def test_record_tool_span_hashes_args_not_raw(store: TraceStore) -> None:
    raw_value = "PHI-SENTINEL-John Doe MRN 00099"
    store.record_tool_span(
        correlation_id="corr-3",
        start_ts=0.0,
        end_ts=0.1,
        ok=True,
        tool_name="get_allergies",
        args={"patient_note": raw_value},
    )

    span = store.get_spans("corr-3")[0]
    assert span.args_hash != raw_value
    assert raw_value not in span.args_hash
    # Deterministic: identical args + secret hash identically.
    assert span.args_hash == hash_args({"patient_note": raw_value}, _TEST_HASH_KEY)


def test_record_tool_span_failure_records_error_category(store: TraceStore) -> None:
    store.record_tool_span(
        correlation_id="corr-4",
        start_ts=0.0,
        end_ts=0.1,
        ok=False,
        tool_name="get_labs",
        args={},
        error_category="not_found",
    )

    span = store.get_spans("corr-4")[0]
    assert span.status == SpanStatus.FAIL
    assert span.error_category == "not_found"


def test_record_llm_span_write_and_read_back(store: TraceStore) -> None:
    store.record_llm_span(
        correlation_id="corr-5",
        start_ts=5.0,
        end_ts=6.0,
        ok=True,
        model="qwen3:4b",
        tokens_in=120,
        tokens_out=45,
    )

    span = store.get_spans("corr-5")[0]
    assert span.span_type == SpanType.LLM
    assert span.model == "qwen3:4b"
    assert span.tokens_in == 120
    assert span.tokens_out == 45


def test_record_verification_span_write_and_read_back(store: TraceStore) -> None:
    store.record_verification_span(
        correlation_id="corr-6",
        start_ts=1.0,
        end_ts=1.2,
        ok=True,
        verdict="verified",
        claim_count=3,
        stripped_count=0,
    )

    span = store.get_spans("corr-6")[0]
    assert span.span_type == SpanType.VERIFICATION
    assert span.verdict == "verified"
    assert span.claim_count == 3
    assert span.stripped_count == 0


def test_record_feedback_span_write_and_read_back(store: TraceStore) -> None:
    store.record_feedback_span(
        correlation_id="corr-7",
        start_ts=1.0,
        end_ts=1.0,
        feedback_thumb=FeedbackThumb.UP,
        feedback_comment="Helpful and accurate.",
    )

    span = store.get_spans("corr-7")[0]
    assert span.span_type == SpanType.FEEDBACK
    assert span.status == SpanStatus.OK
    assert span.feedback_thumb == FeedbackThumb.UP
    assert span.feedback_comment == "Helpful and accurate."


def test_record_feedback_span_comment_is_optional(store: TraceStore) -> None:
    store.record_feedback_span(
        correlation_id="corr-8",
        start_ts=1.0,
        end_ts=1.0,
        feedback_thumb=FeedbackThumb.DOWN,
        feedback_comment=None,
    )

    span = store.get_spans("corr-8")[0]
    assert span.feedback_thumb == FeedbackThumb.DOWN
    assert span.feedback_comment is None


def test_record_feedback_span_records_owner_token_hash(store: TraceStore) -> None:
    # #180 review MINOR 5: a FEEDBACK span records the (already-checked)
    # caller's own owner_token_hash too, so a post-#180 row is
    # distinguishable from a pre-#180 one in a DB upgraded in place -- a
    # triager reading a disputed comment can tell which regime produced it.
    store.record_feedback_span(
        correlation_id="corr-9",
        start_ts=1.0,
        end_ts=1.0,
        feedback_thumb=FeedbackThumb.DOWN,
        feedback_comment="Missed the recent A1C.",
        owner_token="clinician-a-token",
    )

    span = store.get_spans("corr-9")[0]
    assert span.owner_token_hash is not None
    assert span.owner_token_hash != "clinician-a-token"
    assert span.owner_token_hash == hash_owner_token("clinician-a-token", _TEST_HASH_KEY)


def test_record_feedback_span_without_owner_token_records_no_hash(store: TraceStore) -> None:
    # Presence pairing for the test above: omitting owner_token (every
    # existing caller before #180) still records no hash, same as
    # record_request_span's own default -- not a regression for callers
    # that don't pass it.
    store.record_feedback_span(
        correlation_id="corr-10",
        start_ts=1.0,
        end_ts=1.0,
        feedback_thumb=FeedbackThumb.UP,
        feedback_comment=None,
    )

    span = store.get_spans("corr-10")[0]
    assert span.owner_token_hash is None


def test_get_spans_filters_by_correlation_id(store: TraceStore) -> None:
    store.record_request_span(correlation_id="corr-a", start_ts=0.0, end_ts=1.0, ok=True)
    store.record_request_span(correlation_id="corr-b", start_ts=0.0, end_ts=1.0, ok=True)
    store.record_verification_span(
        correlation_id="corr-a", start_ts=1.0, end_ts=1.1, ok=True, verdict="blocked", claim_count=0, stripped_count=0
    )

    spans_a = store.get_spans("corr-a")
    spans_b = store.get_spans("corr-b")

    assert len(spans_a) == 2
    assert all(span.correlation_id == "corr-a" for span in spans_a)
    assert len(spans_b) == 1


def test_get_spans_unknown_correlation_id_returns_empty(store: TraceStore) -> None:
    assert store.get_spans("no-such-id") == []


def test_no_phi_persisted_across_all_span_types(store: TraceStore, db_path: str) -> None:
    """The rigorous no-PHI check: write one of every span type using
    record-data-shaped values, then scan the RAW database bytes on disk --
    not just the typed accessors -- for anything that looks like patient
    record content. Only the feedback comment (explicitly permitted,
    user-authored text about the response) is allowed to appear verbatim."""
    sentinel_drug_name = "Lisinopril-10mg-PATIENT-SPECIFIC"
    sentinel_note = "Patient reports chest pain since Tuesday"
    sentinel_worker_name = "evidence-retriever"
    sentinel_sub_task_type = "RetrieveSubTask"
    sentinel_bearer_token = "Bearer-Secret-Session-Token-abc123"

    store.record_request_span(
        correlation_id="corr-phi", start_ts=0.0, end_ts=1.0, ok=True, owner_token=sentinel_bearer_token
    )
    store.record_tool_span(
        correlation_id="corr-phi",
        start_ts=0.0,
        end_ts=1.0,
        ok=True,
        tool_name="get_medications",
        args={"drug_name": sentinel_drug_name, "note": sentinel_note},
    )
    store.record_llm_span(
        correlation_id="corr-phi", start_ts=0.0, end_ts=1.0, ok=True, model="qwen3:4b", tokens_in=10, tokens_out=5
    )
    store.record_verification_span(
        correlation_id="corr-phi", start_ts=0.0, end_ts=1.0, ok=True, verdict="verified", claim_count=1, stripped_count=0
    )
    store.record_feedback_span(
        correlation_id="corr-phi",
        start_ts=0.0,
        end_ts=1.0,
        feedback_thumb=FeedbackThumb.UP,
        feedback_comment="Great answer, matched the chart.",
    )
    # P3.8: worker spans (app.supervisor handoffs) belong in "one of every
    # span type" too -- exercises the WORKER span type plus its
    # worker_name/sub_task_type/span_id/parent_span_id columns. These are
    # closed-set/non-identifying by contract (never a sub-task field value --
    # see app.supervisor's module docstring), so, unlike the tool ``args``
    # above, they are EXPECTED to persist verbatim -- asserted via
    # ``get_spans`` below, not added to the not-in-raw-bytes checks.
    store.record_worker_span(
        correlation_id="corr-phi",
        start_ts=0.0,
        end_ts=1.0,
        ok=True,
        worker_name=sentinel_worker_name,
        sub_task_type=sentinel_sub_task_type,
        span_id="span-outer",
        parent_span_id="span-root",
    )

    raw_bytes = Path(db_path).read_bytes()

    assert sentinel_drug_name.encode() not in raw_bytes
    assert sentinel_note.encode() not in raw_bytes
    assert sentinel_bearer_token.encode() not in raw_bytes

    worker_span = next(span for span in store.get_spans("corr-phi") if span.span_type == SpanType.WORKER)
    assert worker_span.worker_name == sentinel_worker_name
    assert worker_span.sub_task_type == sentinel_sub_task_type
    assert worker_span.span_id == "span-outer"
    assert worker_span.parent_span_id == "span-root"


def test_hash_args_is_deterministic_and_order_independent() -> None:
    assert hash_args({"a": 1, "b": 2}, _TEST_HASH_KEY) == hash_args({"b": 2, "a": 1}, _TEST_HASH_KEY)


def test_hash_args_differs_for_different_values() -> None:
    assert hash_args({"a": 1}, _TEST_HASH_KEY) != hash_args({"a": 2}, _TEST_HASH_KEY)


def test_hash_args_is_keyed_not_a_bare_hash() -> None:
    # A bare SHA-256 would let anyone recompute the hash without the secret,
    # defeating the point of hashing low-entropy args (e.g. a patient id)
    # instead of storing them raw -- see module docstring's "NO PHI ON DISK"
    # section. Different keys over the SAME args must disagree.
    args = {"patient_id": 42}
    key_a = secrets.token_hex(16)
    key_b = secrets.token_hex(16)
    assert hash_args(args, key_a) != hash_args(args, key_b)
