"""Pydantic v2 input/output contracts for every agent tool.

This module defines the schemas only -- no tool logic. Each tool gets a
dedicated ``*Input`` model and either a single ``*Output`` model
(``get_patient_summary``) or an ``*Output`` model wrapping a list of a
per-record ``*Item`` model (all the others).

Every patient-data output item carries an optional ``source_refs`` field
(see ``app.schemas.common.SourceRef``) as a forward-compatible hook for the
P3.1 verification layer. It is unpopulated until that layer exists.
"""

from __future__ import annotations

import datetime

from pydantic import Field, PositiveInt

from app.schemas.common import (
    AbnormalFlag,
    AllergySeverity,
    AppointmentStatus,
    EncounterType,
    InteractionSeverity,
    MedicationStatus,
    PatientId,
    ProblemStatus,
    Sex,
    SourceRef,
    ToolSchemaModel,
    VitalType,
)

# ---------------------------------------------------------------------------
# 1. get_patient_summary
# ---------------------------------------------------------------------------


class GetPatientSummaryInput(ToolSchemaModel):
    patient_id: PatientId


class PatientSummaryOutput(ToolSchemaModel):
    """Synthesis summary: demographics plus per-section record counts."""

    patient_id: PatientId
    first_name: str
    last_name: str
    date_of_birth: datetime.date
    sex: Sex
    medication_count: int = Field(ge=0)
    allergy_count: int = Field(ge=0)
    problem_count: int = Field(ge=0)
    recent_lab_count: int = Field(ge=0)
    vital_count: int = Field(ge=0)
    encounter_count: int = Field(ge=0)
    appointment_count: int = Field(ge=0)
    source_refs: list[SourceRef] | None = None


# ---------------------------------------------------------------------------
# 2. get_medications
# ---------------------------------------------------------------------------


class GetMedicationsInput(ToolSchemaModel):
    patient_id: PatientId


class MedicationItem(ToolSchemaModel):
    name: str
    dose: str
    route: str
    status: MedicationStatus
    start_date: datetime.date | None = None
    end_date: datetime.date | None = None
    source_refs: list[SourceRef] | None = None


class MedicationsOutput(ToolSchemaModel):
    items: list[MedicationItem]


# ---------------------------------------------------------------------------
# 3. get_allergies
# ---------------------------------------------------------------------------


class GetAllergiesInput(ToolSchemaModel):
    patient_id: PatientId


class AllergyItem(ToolSchemaModel):
    substance: str
    reaction: str | None = None
    severity: AllergySeverity
    source_refs: list[SourceRef] | None = None


class AllergiesOutput(ToolSchemaModel):
    items: list[AllergyItem]


# ---------------------------------------------------------------------------
# 4. get_problems
# ---------------------------------------------------------------------------


class GetProblemsInput(ToolSchemaModel):
    patient_id: PatientId


class ProblemItem(ToolSchemaModel):
    title: str
    icd_code: str | None = None
    status: ProblemStatus
    onset_date: datetime.date | None = None
    source_refs: list[SourceRef] | None = None


class ProblemsOutput(ToolSchemaModel):
    items: list[ProblemItem]


# ---------------------------------------------------------------------------
# 5. get_recent_labs
# ---------------------------------------------------------------------------


class GetRecentLabsInput(ToolSchemaModel):
    patient_id: PatientId
    limit: PositiveInt | None = None
    since: datetime.date | None = None


class LabResultItem(ToolSchemaModel):
    test_name: str
    value: str
    unit: str | None = None
    reference_range: str | None = None
    date: datetime.datetime
    abnormal_flag: AbnormalFlag
    source_refs: list[SourceRef] | None = None


class RecentLabsOutput(ToolSchemaModel):
    items: list[LabResultItem]


# ---------------------------------------------------------------------------
# 6. get_vitals
# ---------------------------------------------------------------------------


class GetVitalsInput(ToolSchemaModel):
    patient_id: PatientId
    limit: PositiveInt | None = None
    since: datetime.date | None = None


class VitalReadingItem(ToolSchemaModel):
    vital_type: VitalType
    value: float
    unit: str
    date: datetime.datetime
    source_refs: list[SourceRef] | None = None


class VitalsOutput(ToolSchemaModel):
    items: list[VitalReadingItem]


# ---------------------------------------------------------------------------
# 7. get_encounters
# ---------------------------------------------------------------------------


class GetEncountersInput(ToolSchemaModel):
    patient_id: PatientId
    start_date: datetime.date | None = None
    end_date: datetime.date | None = None
    limit: PositiveInt | None = None


class EncounterItem(ToolSchemaModel):
    encounter_id: PositiveInt
    date: datetime.datetime
    reason: str | None = None
    provider: str | None = None
    encounter_type: EncounterType
    source_refs: list[SourceRef] | None = None


class EncountersOutput(ToolSchemaModel):
    items: list[EncounterItem]


# ---------------------------------------------------------------------------
# 8. get_appointments
# ---------------------------------------------------------------------------


class GetAppointmentsInput(ToolSchemaModel):
    patient_id: PatientId
    start_date: datetime.date | None = None
    end_date: datetime.date | None = None


class AppointmentItem(ToolSchemaModel):
    date: datetime.date
    time: datetime.time
    status: AppointmentStatus
    provider: str | None = None
    source_refs: list[SourceRef] | None = None


class AppointmentsOutput(ToolSchemaModel):
    items: list[AppointmentItem]


# ---------------------------------------------------------------------------
# 9. check_drug_interactions (offline tool, no patient_id / no provenance)
# ---------------------------------------------------------------------------


class CheckDrugInteractionsInput(ToolSchemaModel):
    drugs: list[str] = Field(min_length=2)


class DrugInteractionItem(ToolSchemaModel):
    drug_a: str
    drug_b: str
    severity: InteractionSeverity
    description: str


class CheckDrugInteractionsOutput(ToolSchemaModel):
    items: list[DrugInteractionItem]
