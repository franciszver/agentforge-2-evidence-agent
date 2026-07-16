"""Shared pytest fixtures for the copilot-agent hermetic test suite."""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from typing import Callable

import httpx
import pytest

from app.openemr_client import OpenEmrClient

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
