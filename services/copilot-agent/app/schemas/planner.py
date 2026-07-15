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

from app.schemas.common import SourceRef


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


class FinalAnswer(BaseModel):
    """The planner's final answer, produced by the two-call pattern (P2.9).

    Once the planner has decided to answer, it reasons in free text (a
    ``chat`` call) and then extracts *this* schema via constrained decoding (an
    ``extract`` call). Constraining only the extraction step -- not the
    reasoning -- keeps the "constraint tax" off the reasoning while still
    guaranteeing a well-formed final answer (IMPLEMENTATION_PLAN §4.3).

    ``source_refs`` is the same forward-compatible provenance hook every tool
    output carries (see ``app.schemas.common.SourceRef``); it is unpopulated
    today -- the verification layer that fills it is P3.1.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str
    source_refs: list[SourceRef] | None = None
