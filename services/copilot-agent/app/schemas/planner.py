"""Schemas for the single-tool-per-turn planner loop (P2.8).

``PlannerDecision`` is the schema Ollama's constrained decoding (``format``)
is pinned to on every planner turn. This structurally enforces "at most one
tool call per turn" -- there is no way to express two tool calls in one
instance of this schema -- and forces the model to choose between exactly
two actions instead of free-form tool-calling syntax a 4B model handles
unreliably.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ToolName(StrEnum):
    """The 8 patient-data tools the planner may select (P2.4-P2.6)."""

    GET_PATIENT_SUMMARY = "get_patient_summary"
    GET_MEDICATIONS = "get_medications"
    GET_ALLERGIES = "get_allergies"
    GET_PROBLEMS = "get_problems"
    GET_RECENT_LABS = "get_recent_labs"
    GET_VITALS = "get_vitals"
    GET_ENCOUNTERS = "get_encounters"
    GET_APPOINTMENTS = "get_appointments"


class PlannerAction(StrEnum):
    CALL_TOOL = "call_tool"
    ANSWER = "answer"


class PlannerDecision(BaseModel):
    """One planner turn's decision: call exactly one tool, or answer.

    ``tool``/``tool_args`` are only meaningful when ``action`` is
    ``call_tool``; ``final_answer`` only when ``action`` is ``answer``.
    ``tool_args`` is a flat string map (e.g. ``{"limit": "3"}``) rather than
    a typed per-tool schema -- constrained decoding needs one fixed schema
    for every turn regardless of which tool gets picked, and a flat string
    map keeps that schema simple for a 4B model. The planner coerces and
    validates recognized keys per-tool before dispatch (see
    ``app.planner._build_tool_kwargs``); it never includes ``patient_id``
    here -- the tool is always run against the conversation's bound patient.

    Deliberately not a ``ToolSchemaModel`` (this isn't tool I/O data) and not
    frozen: constrained decoding builds a fresh instance via
    ``model_validate`` every turn, so immutability buys nothing here.
    """

    model_config = ConfigDict(extra="forbid")

    action: PlannerAction
    tool: ToolName | None = None
    tool_args: dict[str, str] | None = None
    reason: str
    final_answer: str | None = None
