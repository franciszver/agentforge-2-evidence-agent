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
  * ``guideline_citation_present`` -- (P3G.1) at least one rendered,
                                 surviving claim carries a VERIFIED
                                 ``guideline_chunk`` ``DocumentCitation`` --
                                 needs ``retrieved_chunks`` on the case (see
                                 ``RetrievedChunkFixture``) and, like
                                 ``verdict``, the extraction/verification
                                 stage (``runner.pipeline.needs_verification``
                                 -- widened to also trigger on this
                                 assertion).

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

from app.schemas.ingestion import Citation
from app.schemas.planner import ToolName
from app.schemas.reranking import RerankedChunk
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
#
# P3G.1: 5 additional, ADDITIVE rubric categories for the Phase-2 multimodal
# surface (`docs/TEST_PLAN.md` -- eval-gate section) -- boolean, machine-
# checkable, and consumed by P3G.2's PR-blocking gate via this same
# ``category`` field (a case's schema/replay test is dynamically marked with
# ``pytest.mark.<category>`` -- see ``evals/test_cases.py`` -- so the gate can
# select ``-m schema_valid`` etc. across both YAML cases and, for the two
# categories that are deterministic pytest modules instead of YAML+recording
# cases, plain test functions carrying the same mark):
#   * ``schema_valid``           -- document extraction (LabResultFact/
#                                    IntakeFormFact) returns schema-valid
#                                    facts, incl. correctly-``None``
#                                    not-found fields and well-formed
#                                    ``Citation``/``DocumentCitation`` objects.
#   * ``citation_present``       -- a guideline-answerable question's answer
#                                    carries a VERIFIED ``guideline_chunk``
#                                    ``DocumentCitation``; a lab-value
#                                    question's answer cites the fact.
#   * ``factually_consistent``   -- the cited quote actually supports the
#                                    claim (verbatim match against the raw
#                                    source); a wrong-value citation must NOT
#                                    verify.
#   * ``safe_refusal``           -- cross-patient / out-of-scope / unreadable
#                                    -document questions are handled honestly
#                                    (not-found / refusal), never fabricated.
#   * ``no_phi_in_logs``         -- emitted logs/traces/encounter records
#                                    never carry a PHI marker threaded through
#                                    the query/document.
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
    "schema_valid",
    "citation_present",
    "factually_consistent",
    "safe_refusal",
    "no_phi_in_logs",
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


class GuidelineCitationPresentAssertion(_AssertionBase):
    """P3G.1: at least one surviving (rendered, not stripped) claim carries a
    VERIFIED ``guideline_chunk`` ``DocumentCitation`` -- distinct from
    ``verdict: verified`` alone, which a claim can satisfy via ordinary
    ``SourceRef`` chart citations with zero document citations at all. See
    ``runner.assertions._check_guideline_citation_present``."""

    type: Literal["guideline_citation_present"]


Assertion = Annotated[
    Union[
        FirstToolInAssertion,
        AnswerContainsAssertion,
        AnswerNotContainsAssertion,
        VerdictAssertion,
        MustRefuseAssertion,
        NoPhiAssertion,
        GuidelineCitationPresentAssertion,
    ],
    Field(discriminator="type"),
]


class RetrievedChunkFixture(BaseModel):
    """Canned stand-in for one ``RerankedChunk`` (P3G.1) -- the guideline-
    corpus evidence a case declares as already retrieved+reranked for its
    question, exactly the way ``tool_data`` cans a structured-tool result.
    Fed to ``app.extraction.ClaimExtractor.extract_claims``/``run_verification``
    the same as the live P3.9 ``/chat`` wiring, so a case can exercise
    guideline-citation grounding without a real retriever/reranker call.
    Field names mirror ``app.schemas.retrieval.RetrievedChunk``/
    ``app.schemas.reranking.RerankedChunk`` exactly (``chunk_id`` is the
    ``<doc_id>#<section-slug>`` id a ``DocumentCitation``'s
    ``field_or_chunk_id`` must reference to verify)."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    title: str
    section: str
    text: str = Field(min_length=1)
    rerank_score: float = 1.0

    def to_reranked_chunk(self) -> RerankedChunk:
        return RerankedChunk(
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            title=self.title,
            section=self.section,
            text=self.text,
            scores={"hybrid": self.rerank_score},
            rerank_score=self.rerank_score,
        )


class PatientFactFixture(BaseModel):
    """Canned stand-in for one patient-scoped ingested document fact (issue
    #70); fields mirror ``app.schemas.ingestion.Citation`` exactly."""

    model_config = ConfigDict(extra="forbid")

    source_type: Literal["lab_pdf", "intake_form"]
    source_id: str = Field(min_length=1)
    page_or_section: str = Field(min_length=1)
    field_or_chunk_id: str = Field(min_length=1)
    quote_or_value: str = Field(min_length=1)

    def to_citation(self) -> Citation:
        return Citation(
            source_type=self.source_type,
            source_id=self.source_id,
            page_or_section=self.page_or_section,
            field_or_chunk_id=self.field_or_chunk_id,
            quote_or_value=self.quote_or_value,
        )


class EvalCase(BaseModel):
    """One YAML eval case -- see module docstring for the assertion vocabulary."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: _CATEGORIES
    failure_mode: str = Field(min_length=1)
    source: str | None = Field(
        default=None,
        description=(
            "The P4.2 trace-store correlation id this case was promoted from "
            "(P4.9's promote-to-eval generator, app.review_queue). Absent for "
            "hand-authored cases under evals/cases/."
        ),
    )
    question: str = Field(min_length=1)
    patient_id: int = Field(gt=0)
    patient_name: str | None = Field(
        default=None,
        description=(
            "The bound patient's own display name (#224 name-binding), fed "
            "into app.extraction.detect_foreign_patient_reference's named "
            "cross-patient signals -- mirrors what app.chat resolves live via "
            "Planner.resolve_patient_name(). Absent (None) for cases that "
            "don't need it -- the guard then falls back to numeric-only "
            "detection, byte-identical to the pre-#224 harness."
        ),
    )
    patient_roster: list[str] | None = Field(
        default=None,
        description=(
            "Every OTHER patient's display name (#237 roster-based "
            "cross-patient detection), fed into app.extraction"
            ".detect_foreign_patient_reference's roster-based 'switch to "
            "<Name>' signal -- mirrors what app.chat resolves live (lazily) "
            "via Planner.resolve_patient_roster(). Absent (None) for cases "
            "that don't need it -- the signal is then skipped entirely, "
            "byte-identical to the pre-#237 harness."
        ),
    )
    tool_data: dict[ToolName, dict[str, Any]] = Field(default_factory=dict)
    retrieved_chunks: list[RetrievedChunkFixture] = Field(
        default_factory=list,
        description=(
            "P3G.1: canned guideline-corpus evidence for this case's question "
            "-- see RetrievedChunkFixture. Empty for cases with no guideline-"
            "citation surface, mirroring a chart-data-only turn that retrieves "
            "nothing on the live /chat path."
        ),
    )
    patient_facts: list[PatientFactFixture] = Field(
        default_factory=list,
        description=(
            "Issue #70: canned patient-scoped ingested document-fact "
            "citations for this case's bound patient -- see "
            "PatientFactFixture. Empty (the default) for cases with no "
            "document-fact surface, mirroring a chart-data-only turn where "
            "app.chat.get_patient_fact_provider's live fetch returns nothing "
            "for that patient -- byte-identical to the pre-#70 harness, so "
            "every existing case YAML stays valid unmodified."
        ),
    )
    assertions: list[Assertion] = Field(min_length=1)
    xfail: str | None = Field(
        default=None,
        description=(
            "Set when this case documents a KNOWN, honest failure (e.g. the "
            "4B model guesses instead of disambiguating) rather than a "
            "regression to catch. The value is the rationale, surfaced by "
            "the runner as a strict pytest xfail -- the case still runs for "
            "real every time; an unexpected PASS fails the suite loudly so a "
            "stale xfail can't rot silently (docs/TEST_PLAN.md Sec 5).\n\n"
            "**Integrity rule for the rationale text (eval-integrity fix, "
            "post-hoc -- two prior rounds drifted into embellished/inaccurate "
            "reasons, see git history).** An xfail reason must state ONLY: "
            "(a) the observed failure mode (what the recorded model/extractor "
            "actually did against the case's own, corpus-faithful fixture), "
            "and (b) the resulting CitationStatus/verdict this pipeline run "
            "produced -- both drawn directly from re-running the recording, "
            "never inferred or assumed. It must NOT: claim what any OTHER "
            "case does or doesn't do (no cross-case comparisons); use "
            "subjective quality language ('genuinely', 'correctly-grounded', "
            "'verbatim', 'measurably improved', 'answers correctly') unless "
            "that specific word is checked true against this case's own "
            "recording; or claim a case 'now passes' anywhere but in that "
            "case's own file, verified by actually running it. Keep it short."
        ),
    )

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
