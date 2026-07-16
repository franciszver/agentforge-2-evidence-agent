"""YAML eval-case schema (P4.7): the case shape + assertion vocabulary.

A case file declares one clinical question against one (synthetic) patient,
the CANNED tool data every one of the 8 planner tools should return if
dispatched (so tool execution is deterministic -- no live OpenEMR), and a
list of deterministic assertions to run against the pipeline's result.

**Assertion vocabulary** (the canonical set; mirrored in
``docs/TEST_PLAN.md`` Sec 5 -- update both together when adding a type):

  * ``first_tool_in``        -- tool-selection (absorbs P2.8): the first
                                 tool the planner dispatches must be one of
                                 ``tools``.
  * ``answer_contains``      -- reference-based key-fact matching: every
                                 phrase in ``phrases`` must appear
                                 (normalized) in the planner's free-text
                                 answer.
  * ``answer_not_contains``  -- the negative form: none of ``phrases`` may
                                 appear.
  * ``verdict``               -- the whole-answer verdict
                                 (``app.verdict.Verdict``) computed by the
                                 verification layer must equal ``equals``.
  * ``must_refuse``           -- none of ``forbidden_tools`` may appear
                                 anywhere in the dispatched tool trace
                                 (authorization / injection probes that
                                 demand a specific tool call).
  * ``no_phi``                -- none of ``markers`` may appear in the
                                 final answer or the client-facing tool
                                 trace (cross-patient / leaked-secret probes).

``verdict`` (and therefore the extraction + verification pipeline stage) is
only computed for cases that actually use it -- see
``evals/runner/pipeline.py``'s ``needs_verification`` -- so a plain
tool-selection case's recording doesn't have to carry the extra claim-
extraction model call.

**Tool data validation.** ``tool_data`` maps a ``ToolName`` to a dict that
must validate against that tool's own Output schema (e.g. ``get_medications``
-> ``MedicationsOutput``) -- checked eagerly here so a malformed case fails
at load time with a clear error, not mid-run.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.schemas.planner import ToolName
from app.schemas.tools import (
    AllergiesOutput,
    AppointmentsOutput,
    EncountersOutput,
    MedicationsOutput,
    PatientSummaryOutput,
    ProblemsOutput,
    RecentLabsOutput,
    VitalsOutput,
)
from app.verdict import Verdict

# ToolName -> the Output schema its canned ``tool_data`` entry must validate
# against. Mirrors ``app.planner.TOOL_REGISTRY``.
OUTPUT_SCHEMAS: dict[ToolName, type[BaseModel]] = {
    ToolName.GET_PATIENT_SUMMARY: PatientSummaryOutput,
    ToolName.GET_MEDICATIONS: MedicationsOutput,
    ToolName.GET_ALLERGIES: AllergiesOutput,
    ToolName.GET_PROBLEMS: ProblemsOutput,
    ToolName.GET_RECENT_LABS: RecentLabsOutput,
    ToolName.GET_VITALS: VitalsOutput,
    ToolName.GET_ENCOUNTERS: EncountersOutput,
    ToolName.GET_APPOINTMENTS: AppointmentsOutput,
}


# The 8 ``docs/TEST_PLAN.md`` Sec 5 eval categories, plus ``tool_selection``
# -- the P2.8 tool-selection eval this harness absorbs. Not one of the 8
# behavioral-failure categories (it guards which tool the planner picks, not
# what it says), kept as a 9th value so the migrated P2.8 cases have a home.
_CATEGORIES = Literal[
    "hallucination_bait",
    "missing_data",
    "ambiguity",
    "authorization_probe",
    "stale_data",
    "injection",
    "constraint",
    "regression",
    "tool_selection",
]


class _AssertionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FirstToolInAssertion(_AssertionBase):
    type: Literal["first_tool_in"]
    tools: list[ToolName] = Field(min_length=1)


class AnswerContainsAssertion(_AssertionBase):
    type: Literal["answer_contains"]
    phrases: list[str] = Field(min_length=1)


class AnswerNotContainsAssertion(_AssertionBase):
    type: Literal["answer_not_contains"]
    phrases: list[str] = Field(min_length=1)


class VerdictAssertion(_AssertionBase):
    type: Literal["verdict"]
    equals: Verdict


class MustRefuseAssertion(_AssertionBase):
    type: Literal["must_refuse"]
    forbidden_tools: list[ToolName] = Field(min_length=1)


class NoPhiAssertion(_AssertionBase):
    type: Literal["no_phi"]
    markers: list[str] = Field(min_length=1)


Assertion = Annotated[
    Union[
        FirstToolInAssertion,
        AnswerContainsAssertion,
        AnswerNotContainsAssertion,
        VerdictAssertion,
        MustRefuseAssertion,
        NoPhiAssertion,
    ],
    Field(discriminator="type"),
]


class EvalCase(BaseModel):
    """One YAML eval case -- see module docstring for the assertion vocabulary."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: _CATEGORIES
    failure_mode: str | None = None
    question: str = Field(min_length=1)
    patient_id: int = Field(gt=0)
    tool_data: dict[ToolName, dict[str, Any]] = Field(default_factory=dict)
    assertions: list[Assertion] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_tool_data_against_output_schemas(self) -> EvalCase:
        for tool, canned in self.tool_data.items():
            schema = OUTPUT_SCHEMAS[tool]
            try:
                schema.model_validate(canned)
            except ValidationError as exc:
                raise ValueError(
                    f"tool_data[{tool.value!r}] does not validate against {schema.__name__}: {exc}"
                ) from exc
        return self


class EvalCaseError(Exception):
    """Raised when a case file fails to parse or validate -- a malformed
    case fails clearly rather than being silently skipped."""
