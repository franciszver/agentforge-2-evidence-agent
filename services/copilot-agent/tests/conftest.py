"""Shared pytest fixtures for the copilot-agent hermetic test suite."""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from typing import Callable

import httpx
import pytest

from app.openemr_client import OpenEmrClient
from app.schemas.ingestion import ExtractedLabRow, LabPageExtraction

# Derived per run (not a hardcoded literal) so no secret-shaped string is
# committed. The isolation store below is never asserted against by hash, so
# any stable value works.
_TEST_HASH_KEY = secrets.token_hex(16)


@pytest.fixture
def make_openemr_client() -> Callable[[Callable[[httpx.Request], httpx.Response]], OpenEmrClient]:
    """Factory fixture: build an ``OpenEmrClient`` backed by an ``httpx.MockTransport``."""

    def _make(handler: Callable[[httpx.Request], httpx.Response]) -> OpenEmrClient:
        return OpenEmrClient(
            base_url="https://openemr",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    return _make


@pytest.fixture
def fixture_pdf(tmp_path):
    """A minimal single blank-page PDF, for tests that ingest a real file
    through ``attach_and_extract`` without needing its actual content (the
    scripted ``FakeVlm`` below never reads it)."""
    import pypdfium2 as pdfium

    path = tmp_path / "lab.pdf"
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(200.0, 200.0)
    pdf.save(str(path))
    pdf.close()
    return path


class FakeVlm:
    """Scripted single-page lab VLM double -- no live Ollama call. Always
    returns the same ``row`` for every page ``extract``-ed."""

    def __init__(self, row: ExtractedLabRow) -> None:
        self._row = row
        self.extract_calls: list[object] = []

    def extract(self, prompt_or_messages, schema, *, options=None, images=None):
        self.extract_calls.append(prompt_or_messages)
        return LabPageExtraction(rows=[self._row])


@pytest.fixture(scope="session")
def _isolation_trace_store(tmp_path_factory: pytest.TempPathFactory) -> object:
    """One throwaway SQLite trace store for the whole session (see
    ``_isolate_trace_store``). Session-scoped so the schema is built once, not
    per test."""
    from app.trace_store import TraceStore

    db_dir = tmp_path_factory.mktemp("trace_isolation")
    return TraceStore(db_path=str(db_dir / "traces.db"), hash_secret=_TEST_HASH_KEY)


@pytest.fixture(autouse=True)
def _isolate_trace_store(_isolation_trace_store: object) -> Iterator[None]:
    """Point ``get_trace_store`` at an isolated tmp store for EVERY test.

    Autouse so no ``/chat`` test can forget to isolate: the real
    ``get_trace_store`` builds against ``Settings.trace_db_path``
    (``/data/traces.db``), and its ``mkdir('/data')`` crashes on the
    root-owned CI runner (``PermissionError``) -- and would silently write to
    the dev instance's ``traces.db`` locally, both violating TEST_PLAN Sec 7
    ("agent-service tests write only to per-test temporary SQLite databases").
    Tests that need to INSPECT spans set their own ``get_trace_store``
    override in the test body (same dict key wins); this fixture only supplies
    a safe default and removes it on teardown.
    """
    from app.chat import get_trace_store
    from app.main import app

    app.dependency_overrides[get_trace_store] = lambda: _isolation_trace_store
    yield
    app.dependency_overrides.pop(get_trace_store, None)


@pytest.fixture(scope="session")
def _isolation_eval_history() -> list[object]:
    """One fixed in-memory eval-run history for the whole session (see
    ``_isolate_eval_history``)."""
    from app.dashboard_eval_history import EvalRunPoint

    return [
        EvalRunPoint(
            timestamp="2026-01-01T00:00:00Z",
            git_sha="isolationfixture",
            total=1,
            passed=1,
            failed=0,
            xfailed=0,
            pass_rate=1.0,
        )
    ]


@pytest.fixture(autouse=True)
def _isolate_eval_history(_isolation_eval_history: list[object]) -> Iterator[None]:
    """Point ``get_eval_history_provider`` at a fixed in-memory history for
    EVERY test.

    Autouse so no dashboard test can forget to isolate: the real
    ``get_eval_history_provider`` reads the committed, live-growing
    ``app/data/eval_history.json`` -- a future change to that file could flip
    an unrelated dashboard test that never asked to depend on it (TEST_PLAN
    Sec 7 hermeticity). Tests that need a SPECIFIC history (empty,
    multi-point) set their own ``get_eval_history_provider`` override in the
    test body (same dict key wins); this fixture only supplies a safe
    default and removes it on teardown.
    """
    from app.dashboard import get_eval_history_provider
    from app.main import app

    app.dependency_overrides[get_eval_history_provider] = lambda: (lambda: _isolation_eval_history)
    yield
    app.dependency_overrides.pop(get_eval_history_provider, None)


@pytest.fixture(autouse=True)
def _reset_process_wide_singletons() -> Iterator[None]:
    """Gate 3 (Opus) MINOR finding: ``chat._token_introspector`` and
    ``chat._dev_token_bridge`` are lazily-built, process-wide singletons
    (``get_token_introspector`` / ``get_dev_token_bridge``) baked from
    whatever ``Settings`` were live the FIRST time either is called. No
    fixture reset them, so a test that exercises the real (non-overridden)
    introspection or dev-bridge path -- e.g. ``test_chat_endpoint.py``'s flag
    precedence test, or its fail-closed-default endpoint test -- could seed a
    singleton bound to one test's env/creds that then silently leaks into a
    LATER test under ``pytest-randomly`` reordering. Mirrors
    ``test_launch_binding.py``'s ``_reset_binder_singleton`` for
    ``chat._launch_patient_binder``.

    Also resets ``chat._default_roster_cache`` (#174): ``get_roster_cache``
    is a top-level dependency of ``chat_endpoint`` (evaluated on EVERY
    ``/chat`` call, not just when a "switch to <Name>" construction fires),
    so without this reset the first test in the session to hit a real
    (non-overridden) ``get_roster_cache`` call would seed a cached roster
    that could silently leak into a later roster-assertion test under
    ``pytest-randomly`` reordering -- the same leak class as the two
    singletons above, just for a cache instead of a validator/bridge.

    Promoted to this shared conftest (F2, #185 gate) after
    ``test_subject_ownership.py::test_get_subject_resolver_flag_on_returns_a_real_resolver``
    was found seeding ``chat._token_introspector`` via the real
    ``get_subject_resolver()`` with no reset in that module -- autouse here
    so every module in this test package is covered, not just
    ``test_chat_endpoint.py``.
    """
    import app.chat as chat

    chat._token_introspector = None
    chat._dev_token_bridge = None
    chat._default_roster_cache = None
    yield
    chat._token_introspector = None
    chat._dev_token_bridge = None
    chat._default_roster_cache = None


@pytest.fixture(scope="session", autouse=True)
def _assert_default_trace_store_untouched() -> Iterator[None]:
    """Session leak guard: prove no test ever invoked the real
    ``get_trace_store`` dependency, which would build a store against the
    configured ``trace_db_path`` (``/data``). Every ``/chat`` test must use
    the isolated tmp store (see ``_isolate_trace_store``)."""
    yield
    import app.chat as chat_module

    assert chat_module._default_trace_store is None, (
        "the real get_trace_store dependency was invoked during the test suite "
        "-- a /chat test wrote to the configured trace_db_path (/data) instead "
        "of an isolated tmp store. See conftest._isolate_trace_store."
    )
