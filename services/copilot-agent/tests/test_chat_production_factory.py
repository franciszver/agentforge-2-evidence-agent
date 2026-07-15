"""Hermetic test for the production POST /chat planner factory wiring (P2.14).

``get_planner_factory``'s default implementation was a stub that always
raised ``NotImplementedError`` (P2.10 docstring: "wiring the real factory is
follow-up work once this endpoint is called from a real deployment"). P2.14 --
the chat panel UI -- is that real caller, so this wires it for real:
``OllamaClient``/``OpenEmrClient`` built from ``Settings``, and the request's
bearer token passed straight through for tool calls, matching the production
trust-boundary design (plan §5: "same user token" flows browser -> agent ->
OpenEMR API).

No real network is touched here -- this only checks the factory *builds* a
working ``Planner`` without raising; the http clients it constructs are
never invoked.
"""

from __future__ import annotations

from app.chat import get_planner_factory
from app.planner import Planner


def test_default_planner_factory_builds_a_real_planner_without_raising():
    factory = get_planner_factory(authorization="Bearer sometoken")

    planner = factory(1)

    assert isinstance(planner, Planner)


def test_default_planner_factory_tolerates_a_missing_authorization_header():
    # Never actually invoked on the reject path (chat_endpoint 401s before
    # calling the factory's returned closure), but the factory itself must
    # not raise while FastAPI resolves dependencies ahead of the handler body.
    factory = get_planner_factory(authorization=None)

    planner = factory(1)

    assert isinstance(planner, Planner)
