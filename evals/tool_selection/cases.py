"""Tool-selection eval cases (P2.8 seed; full YAML/record-replay harness is P4.7).

Each case is a clinical question phrased the way the physician persona
would ask it (docs/USERS.md UC1-UC4), the textbook-expected first tool,
and a small ``acceptable`` set of additionally-tolerated tools. A 4B model
does not always pick the single canonical tool for an ambiguous question
(e.g. a broad "what changed" question is defensibly answered by either
``get_encounters`` or ``get_patient_summary``) -- widening ``acceptable``
for those cases avoids over-fitting the eval to one model's specific
phrasing quirks, per the P2.8 task's "widen and note it honestly" guidance.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.planner import ToolName


@dataclass(frozen=True)
class ToolSelectionCase:
    id: str
    use_case: str
    question: str
    expected: ToolName
    acceptable: frozenset[ToolName]


CASES: list[ToolSelectionCase] = [
    ToolSelectionCase(
        id="uc2-meds",
        use_case="UC2 medication safety",
        question="What meds is she on?",
        expected=ToolName.GET_MEDICATIONS,
        acceptable=frozenset({ToolName.GET_MEDICATIONS}),
    ),
    ToolSelectionCase(
        id="uc1-changed",
        use_case="UC1 pre-visit brief",
        question="What's changed since her last visit?",
        expected=ToolName.GET_ENCOUNTERS,
        acceptable=frozenset({ToolName.GET_ENCOUNTERS, ToolName.GET_PATIENT_SUMMARY}),
    ),
    ToolSelectionCase(
        id="uc3-a1c-trend",
        use_case="UC3 lab trend recall",
        question="What are her last three A1c values, and when?",
        expected=ToolName.GET_RECENT_LABS,
        acceptable=frozenset({ToolName.GET_RECENT_LABS}),
    ),
    ToolSelectionCase(
        id="uc2-allergies",
        use_case="UC2 medication safety",
        question="Does she have any allergies?",
        expected=ToolName.GET_ALLERGIES,
        acceptable=frozenset({ToolName.GET_ALLERGIES}),
    ),
    ToolSelectionCase(
        id="uc1-vitals",
        use_case="UC1 pre-visit brief",
        question="What's her blood pressure been like recently?",
        expected=ToolName.GET_VITALS,
        acceptable=frozenset({ToolName.GET_VITALS}),
    ),
    ToolSelectionCase(
        id="uc4-next-appt",
        use_case="UC4 conversational drill-down",
        question="When is her next appointment?",
        expected=ToolName.GET_APPOINTMENTS,
        acceptable=frozenset({ToolName.GET_APPOINTMENTS}),
    ),
    ToolSelectionCase(
        id="uc1-problems",
        use_case="UC1 pre-visit brief",
        question="What are her active problems?",
        expected=ToolName.GET_PROBLEMS,
        acceptable=frozenset({ToolName.GET_PROBLEMS, ToolName.GET_PATIENT_SUMMARY}),
    ),
]
