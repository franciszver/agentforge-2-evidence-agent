"""Shared pytest fixtures for the copilot-agent hermetic test suite."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from app.openemr_client import OpenEmrClient


@pytest.fixture
def make_openemr_client() -> Callable[[Callable[[httpx.Request], httpx.Response]], OpenEmrClient]:
    """Factory fixture: build an ``OpenEmrClient`` backed by an ``httpx.MockTransport``."""

    def _make(handler: Callable[[httpx.Request], httpx.Response]) -> OpenEmrClient:
        return OpenEmrClient(
            base_url="https://openemr",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    return _make
