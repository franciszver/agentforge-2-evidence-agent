"""Hermetic test for the production POST /chat planner factory wiring.

The factory builds the real ``OllamaClient``/``OpenEmrClient`` from
``Settings`` and a ``Planner`` bound to the request's ``patient_id``. Since
issue #126 (finding F4), the token the Planner uses for OpenEMR tool calls is
NOT the browser's ``DevAgentToken`` (an identity assertion, not a real OpenEMR
token) -- it is a REAL OpenEMR token obtained server-side by the
``DevTokenBridge``. This test pins that wiring: ``get_planner_factory`` pulls
the token from the bridge, and the factory chain has no access to the browser
token at all.

No real network is touched: a stub bridge supplies the token, and the http
clients the factory constructs are never invoked.
"""

from __future__ import annotations

from app.chat import _default_planner_factory, get_planner_factory
from app.planner import Planner


class _StubBridge:
    """Stand-in for ``DevTokenBridge``: returns a fixed real OpenEMR token."""

    def __init__(self, token: str) -> None:
        self._token = token
        self.calls = 0

    def get_token(self) -> str:
        self.calls += 1
        return self._token


def test_get_planner_factory_uses_the_bridge_token_for_tool_calls():
    bridge = _StubBridge("real-openemr-token")

    factory = get_planner_factory(dev_token_bridge=bridge)
    planner = factory(1)

    assert isinstance(planner, Planner)
    # The Planner must call OpenEMR tools with the REAL token from the bridge.
    assert planner._token == "real-openemr-token"
    assert bridge.calls == 1


def test_factory_binds_the_requested_patient_id():
    factory = _default_planner_factory("real-openemr-token")

    planner = factory(42)

    assert planner._patient_id == 42
    assert planner._token == "real-openemr-token"
