"""Builds a deterministic, canned tool registry from a case's ``tool_data``
(P4.7) -- the eval-harness analogue of P2.8's OpenEMR ``httpx.MockTransport``
stub, but wired through ``app.planner.Planner``'s existing ``registry``
override seam (the same seam ``services/copilot-agent/tests/test_planner.py``
already uses for hermetic planner tests) rather than re-implementing REST-path
routing per case in YAML.

Every one of the 8 tools always completes without error, regardless of which
one the model picks: a tool named in a case's ``tool_data`` returns exactly
that canned (already schema-validated -- see ``runner.schema``) output; every
other tool returns a minimal empty default. This mirrors P2.8's
"empty-everything-else" stub philosophy so a model turn's outcome only
depends on the case's declared data, never on which tool happened to be
picked.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any

from app.planner import TOOL_REGISTRY, ToolSpec
from app.schemas.common import Sex
from app.schemas.planner import ToolName
from app.schemas.tools import PatientSummaryOutput, ToolSchemaModel

from runner.schema import OUTPUT_SCHEMAS


def _default_output(tool: ToolName, patient_id: int) -> ToolSchemaModel:
    """The empty-everything-else default for a tool not named in the case's
    ``tool_data``. ``get_patient_summary`` has no list to leave empty --
    every field is required -- so it gets placeholder demographics and
    all-zero counts instead."""
    if tool is ToolName.GET_PATIENT_SUMMARY:
        return PatientSummaryOutput(
            patient_id=patient_id,
            first_name="Eval",
            last_name="Patient",
            date_of_birth=datetime.date(2000, 1, 1),
            sex=Sex.UNKNOWN,
            medication_count=0,
            allergy_count=0,
            problem_count=0,
            recent_lab_count=0,
            vital_count=0,
            encounter_count=0,
            appointment_count=0,
        )
    return OUTPUT_SCHEMAS[tool](items=[])


def build_fake_registry(
    tool_data: Mapping[ToolName, dict[str, Any]], patient_id: int
) -> dict[ToolName, ToolSpec]:
    """One ``ToolSpec`` per registered tool, each ``func`` a closure that
    ignores its call args and always returns the case's canned (or default
    empty) output. Descriptions/input schemas are reused from the production
    ``TOOL_REGISTRY`` so the planner's system prompt is unchanged."""
    registry: dict[ToolName, ToolSpec] = {}
    for tool, spec in TOOL_REGISTRY.items():
        canned = tool_data.get(tool)
        output = OUTPUT_SCHEMAS[tool].model_validate(canned) if canned is not None else _default_output(tool, patient_id)

        def _func(client: Any, token: str, pid: int, *, _output: ToolSchemaModel = output, **kwargs: Any) -> ToolSchemaModel:
            return _output

        registry[tool] = ToolSpec(description=spec.description, input_schema=spec.input_schema, func=_func)
    return registry
