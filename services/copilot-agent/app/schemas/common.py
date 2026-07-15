"""Shared primitives for tool I/O schemas: base config, patient id, enums, and
the provenance hook used by every patient-data output item.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# Positive OpenEMR patient record id. Used consistently as the ``patient_id``
# field type across every tool input that scopes to a patient.
PatientId = Annotated[int, Field(gt=0, description="OpenEMR patient record id")]


class ToolSchemaModel(BaseModel):
    """Base class for all tool I/O schemas.

    Immutable (``frozen``) since these are DTOs, not mutable domain objects,
    and rejects unknown fields (``extra="forbid"``) so malformed tool
    input/output fails fast at the boundary instead of silently dropping
    data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceRef(ToolSchemaModel):
    """Provenance pointer tying an output field back to its source record.

    This is a schema hook for the verification layer (P3.1), which is not
    implemented yet. No tool populates this today -- it exists so every
    patient-data output item can *optionally* carry provenance once P3.1
    lands, without another schema migration. Leave it unset (``None``)
    until then.
    """

    tool_call_id: str
    record_id: str
    field: str


class Sex(StrEnum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class MedicationStatus(StrEnum):
    ACTIVE = "active"
    DISCONTINUED = "discontinued"
    COMPLETED = "completed"
    UNKNOWN = "unknown"


class AllergySeverity(StrEnum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


class ProblemStatus(StrEnum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    INACTIVE = "inactive"
    UNKNOWN = "unknown"


class AbnormalFlag(StrEnum):
    NORMAL = "normal"
    HIGH = "high"
    LOW = "low"
    CRITICAL_HIGH = "critical_high"
    CRITICAL_LOW = "critical_low"
    UNKNOWN = "unknown"


class VitalType(StrEnum):
    BLOOD_PRESSURE_SYSTOLIC = "blood_pressure_systolic"
    BLOOD_PRESSURE_DIASTOLIC = "blood_pressure_diastolic"
    HEART_RATE = "heart_rate"
    TEMPERATURE = "temperature"
    RESPIRATORY_RATE = "respiratory_rate"
    OXYGEN_SATURATION = "oxygen_saturation"
    HEIGHT = "height"
    WEIGHT = "weight"
    BMI = "bmi"


class EncounterType(StrEnum):
    OFFICE_VISIT = "office_visit"
    TELEHEALTH = "telehealth"
    HOSPITAL = "hospital"
    PROCEDURE = "procedure"
    OTHER = "other"


class AppointmentStatus(StrEnum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    CHECKED_IN = "checked_in"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"


class InteractionSeverity(StrEnum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CONTRAINDICATED = "contraindicated"
