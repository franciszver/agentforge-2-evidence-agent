"""Minimal live tool-selection eval (P2.8 seed; full YAML/record-replay harness is P4.7).

Runs the REAL planner against the real qwen3:4b model (via a proxied
Ollama -- Ollama is internal-only on the dev stack's docker network; see
services/copilot-agent's ``tests/test_ollama_client.py`` for the same
``OLLAMA_BASE_URL`` bridging convention) for each case in ``cases.py``, and
asserts the FIRST tool it selects is in that case's ``acceptable`` set.

Tool *execution* is stubbed (a canned ``httpx.MockTransport`` response for
every OpenEMR endpoint) -- this eval checks tool SELECTION, not tool DATA,
so no live OpenEMR/dev-stack is needed, only a reachable Ollama. Kept
deliberately tiny per the P2.8 task: no YAML cases, no record/replay, no
scoring infra -- that is P4.7's job.

Skipped by default (``pytest -m "not integration"``). Point
``OLLAMA_BASE_URL`` at the proxied Ollama before running live.
"""

from __future__ import annotations

import os

import httpx
import pytest

from app.config import Settings
from app.ollama_client import OllamaClient
from app.openemr_client import OpenEmrClient
from app.planner import Planner
from tool_selection.cases import CASES

pytestmark = pytest.mark.integration

_PATIENT_ID = 1
_PATIENT_UUID = "eval-stub-patient-uuid"


def _stub_openemr_client() -> OpenEmrClient:
    """A patient roster + empty-everything-else stub, so every one of the 8
    tools completes without error regardless of which one the model picks."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/apis/default/api/patient":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "pid": _PATIENT_ID,
                            "fname": "Eval",
                            "lname": "Patient",
                            "uuid": _PATIENT_UUID,
                            "DOB": "1990-01-01",
                            "sex": "female",
                        }
                    ]
                },
            )
        if path.startswith("/apis/default/fhir/Observation"):
            return httpx.Response(200, json={"resourceType": "Bundle", "total": 0})
        return httpx.Response(200, json={"data": []})

    return OpenEmrClient(base_url="https://openemr", client=httpx.Client(transport=httpx.MockTransport(handler)))


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_first_selected_tool_is_acceptable_for_case(case) -> None:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    settings = Settings(ollama_base_url=base_url, ollama_api_timeout_seconds=120.0)
    ollama_client = OllamaClient.from_settings(settings)
    planner = Planner(
        ollama_client=ollama_client,
        openemr_client=_stub_openemr_client(),
        token="eval-stub-token",
        patient_id=_PATIENT_ID,
    )

    result = planner.run(case.question)

    selected = result.trace[0].tool if result.trace else None
    acceptable_names = sorted(tool.value for tool in case.acceptable)
    got_name = selected.value if selected is not None else "none (answered without calling a tool)"
    assert selected in case.acceptable, (
        f"[{case.id}] {case.use_case}: {case.question!r} -> expected "
        f"{case.expected.value} (acceptable: {acceptable_names}), got {got_name}"
    )
