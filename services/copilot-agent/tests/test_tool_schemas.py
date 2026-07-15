"""Validation tests for the Pydantic v2 tool I/O schemas.

Pydantic v2's default ("lax") mode coerces some cross-type inputs at the
model boundary -- e.g. a numeric string like "42" is accepted for an ``int``
field. Rejection tests below only assert on inputs that lax mode still
rejects (non-numeric strings, out-of-range/negative numbers, too-short
lists, missing required fields, invalid enum members) rather than on
inputs lax mode would silently coerce.
"""

from __future__ import annotations

import datetime

import pytest
from pydantic import ValidationError

from app.schemas.common import (
    AbnormalFlag,
    AllergySeverity,
    AppointmentStatus,
    EncounterType,
    InteractionSeverity,
    MedicationStatus,
    ProblemStatus,
    Sex,
    SourceRef,
    VitalType,
)
from app.schemas.tools import (
    AllergiesOutput,
    AllergyItem,
    AppointmentItem,
    AppointmentsOutput,
    CheckDrugInteractionsInput,
    CheckDrugInteractionsOutput,
    DrugInteractionItem,
    EncounterItem,
    EncountersOutput,
    GetAllergiesInput,
    GetAppointmentsInput,
    GetEncountersInput,
    GetMedicationsInput,
    GetPatientSummaryInput,
    GetProblemsInput,
    GetRecentLabsInput,
    GetVitalsInput,
    LabResultItem,
    MedicationItem,
    MedicationsOutput,
    PatientSummaryOutput,
    ProblemItem,
    ProblemsOutput,
    RecentLabsOutput,
    VitalReadingItem,
    VitalsOutput,
)


# ---------------------------------------------------------------------------
# SourceRef (shared provenance hook)
# ---------------------------------------------------------------------------


def test_source_ref_round_trip():
    ref = SourceRef(tool_call_id="call-1", record_id="rec-1", field="dose")

    restored = SourceRef.model_validate(ref.model_dump())

    assert restored == ref


def test_source_ref_rejects_missing_field():
    with pytest.raises(ValidationError):
        SourceRef(tool_call_id="call-1", record_id="rec-1")


# ---------------------------------------------------------------------------
# 1. get_patient_summary
# ---------------------------------------------------------------------------


def test_get_patient_summary_input_round_trip():
    model = GetPatientSummaryInput(patient_id=42)

    restored = GetPatientSummaryInput.model_validate(model.model_dump())

    assert restored == model


def test_get_patient_summary_input_rejects_non_positive_patient_id():
    with pytest.raises(ValidationError):
        GetPatientSummaryInput(patient_id=0)


def test_get_patient_summary_input_rejects_missing_patient_id():
    with pytest.raises(ValidationError):
        GetPatientSummaryInput()


def test_get_patient_summary_input_rejects_non_numeric_patient_id():
    with pytest.raises(ValidationError):
        GetPatientSummaryInput(patient_id="not-a-number")


def _patient_summary_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "patient_id": 42,
        "first_name": "Jane",
        "last_name": "Doe",
        "date_of_birth": datetime.date(1980, 1, 1),
        "sex": Sex.FEMALE,
        "medication_count": 3,
        "allergy_count": 1,
        "problem_count": 2,
        "recent_lab_count": 5,
        "vital_count": 4,
        "encounter_count": 6,
        "appointment_count": 1,
    }
    base.update(overrides)
    return base


def test_patient_summary_output_round_trip():
    model = PatientSummaryOutput(**_patient_summary_kwargs())

    dumped = model.model_dump()
    restored = PatientSummaryOutput.model_validate(dumped)

    assert restored == model
    assert dumped["sex"] == Sex.FEMALE
    assert dumped["date_of_birth"] == datetime.date(1980, 1, 1)


def test_patient_summary_output_rejects_negative_count():
    with pytest.raises(ValidationError):
        PatientSummaryOutput(**_patient_summary_kwargs(medication_count=-1))


def test_patient_summary_output_rejects_invalid_sex():
    with pytest.raises(ValidationError):
        PatientSummaryOutput(**_patient_summary_kwargs(sex="not-a-sex"))


# ---------------------------------------------------------------------------
# 2. get_medications
# ---------------------------------------------------------------------------


def test_get_medications_input_round_trip():
    model = GetMedicationsInput(patient_id=7)

    assert GetMedicationsInput.model_validate(model.model_dump()) == model


def test_get_medications_input_rejects_missing_patient_id():
    with pytest.raises(ValidationError):
        GetMedicationsInput()


def test_medication_item_round_trip():
    item = MedicationItem(
        name="Lisinopril",
        dose="10mg",
        route="oral",
        status=MedicationStatus.ACTIVE,
        start_date=datetime.date(2024, 1, 1),
        end_date=None,
    )

    restored = MedicationItem.model_validate(item.model_dump())

    assert restored == item


def test_medications_output_round_trip():
    output = MedicationsOutput(
        items=[
            MedicationItem(
                name="Lisinopril",
                dose="10mg",
                route="oral",
                status=MedicationStatus.ACTIVE,
                start_date=datetime.date(2024, 1, 1),
                end_date=None,
            )
        ]
    )

    assert MedicationsOutput.model_validate(output.model_dump()) == output


def test_medications_output_rejects_invalid_status():
    with pytest.raises(ValidationError):
        MedicationsOutput(
            items=[
                {
                    "name": "Lisinopril",
                    "dose": "10mg",
                    "route": "oral",
                    "status": "not-a-status",
                    "start_date": datetime.date(2024, 1, 1),
                    "end_date": None,
                }
            ]
        )


# ---------------------------------------------------------------------------
# 3. get_allergies
# ---------------------------------------------------------------------------


def test_get_allergies_input_round_trip():
    model = GetAllergiesInput(patient_id=7)

    assert GetAllergiesInput.model_validate(model.model_dump()) == model


def test_get_allergies_input_rejects_missing_patient_id():
    with pytest.raises(ValidationError):
        GetAllergiesInput()


def test_allergy_item_round_trip():
    item = AllergyItem(substance="Penicillin", reaction="Rash", severity=AllergySeverity.MODERATE)

    assert AllergyItem.model_validate(item.model_dump()) == item


def test_allergies_output_rejects_invalid_severity():
    with pytest.raises(ValidationError):
        AllergiesOutput(
            items=[{"substance": "Penicillin", "reaction": "Rash", "severity": "extreme"}]
        )


def test_allergies_output_round_trip():
    output = AllergiesOutput(
        items=[AllergyItem(substance="Penicillin", reaction="Rash", severity=AllergySeverity.SEVERE)]
    )

    assert AllergiesOutput.model_validate(output.model_dump()) == output


# ---------------------------------------------------------------------------
# 4. get_problems
# ---------------------------------------------------------------------------


def test_get_problems_input_round_trip():
    model = GetProblemsInput(patient_id=7)

    assert GetProblemsInput.model_validate(model.model_dump()) == model


def test_get_problems_input_rejects_missing_patient_id():
    with pytest.raises(ValidationError):
        GetProblemsInput()


def test_problem_item_round_trip():
    item = ProblemItem(
        title="Type 2 diabetes mellitus",
        icd_code="E11.9",
        status=ProblemStatus.ACTIVE,
        onset_date=datetime.date(2020, 3, 15),
    )

    assert ProblemItem.model_validate(item.model_dump()) == item


def test_problems_output_round_trip():
    output = ProblemsOutput(
        items=[
            ProblemItem(
                title="Type 2 diabetes mellitus",
                icd_code="E11.9",
                status=ProblemStatus.ACTIVE,
                onset_date=datetime.date(2020, 3, 15),
            )
        ]
    )

    assert ProblemsOutput.model_validate(output.model_dump()) == output


def test_problems_output_rejects_invalid_status():
    with pytest.raises(ValidationError):
        ProblemsOutput(
            items=[
                {
                    "title": "Type 2 diabetes mellitus",
                    "icd_code": "E11.9",
                    "status": "not-a-status",
                    "onset_date": datetime.date(2020, 3, 15),
                }
            ]
        )


# ---------------------------------------------------------------------------
# 5. get_recent_labs
# ---------------------------------------------------------------------------


def test_get_recent_labs_input_round_trip_with_optional_fields():
    model = GetRecentLabsInput(patient_id=7, limit=10, since=datetime.date(2024, 1, 1))

    assert GetRecentLabsInput.model_validate(model.model_dump()) == model


def test_get_recent_labs_input_defaults_optional_fields_to_none():
    model = GetRecentLabsInput(patient_id=7)

    assert model.limit is None
    assert model.since is None


def test_get_recent_labs_input_rejects_negative_limit():
    with pytest.raises(ValidationError):
        GetRecentLabsInput(patient_id=7, limit=-1)


def test_get_recent_labs_input_rejects_zero_limit():
    with pytest.raises(ValidationError):
        GetRecentLabsInput(patient_id=7, limit=0)


def test_lab_result_item_round_trip():
    item = LabResultItem(
        test_name="Hemoglobin A1c",
        value="6.1",
        unit="%",
        reference_range="4.0-5.6",
        date=datetime.datetime(2024, 6, 1, 9, 30),
        abnormal_flag=AbnormalFlag.HIGH,
    )

    assert LabResultItem.model_validate(item.model_dump()) == item


def test_recent_labs_output_round_trip():
    output = RecentLabsOutput(
        items=[
            LabResultItem(
                test_name="Hemoglobin A1c",
                value="6.1",
                unit="%",
                reference_range="4.0-5.6",
                date=datetime.datetime(2024, 6, 1, 9, 30),
                abnormal_flag=AbnormalFlag.HIGH,
            )
        ]
    )

    assert RecentLabsOutput.model_validate(output.model_dump()) == output


def test_recent_labs_output_rejects_invalid_abnormal_flag():
    with pytest.raises(ValidationError):
        RecentLabsOutput(
            items=[
                {
                    "test_name": "Hemoglobin A1c",
                    "value": "6.1",
                    "unit": "%",
                    "reference_range": "4.0-5.6",
                    "date": datetime.datetime(2024, 6, 1, 9, 30),
                    "abnormal_flag": "way-off",
                }
            ]
        )


# ---------------------------------------------------------------------------
# 6. get_vitals
# ---------------------------------------------------------------------------


def test_get_vitals_input_round_trip_with_optional_fields():
    model = GetVitalsInput(patient_id=7, limit=5, since=datetime.date(2024, 1, 1))

    assert GetVitalsInput.model_validate(model.model_dump()) == model


def test_get_vitals_input_rejects_negative_limit():
    with pytest.raises(ValidationError):
        GetVitalsInput(patient_id=7, limit=-5)


def test_vital_reading_item_round_trip():
    item = VitalReadingItem(
        vital_type=VitalType.HEART_RATE,
        value=72.0,
        unit="bpm",
        date=datetime.datetime(2024, 6, 1, 9, 30),
    )

    assert VitalReadingItem.model_validate(item.model_dump()) == item


def test_vitals_output_round_trip():
    output = VitalsOutput(
        items=[
            VitalReadingItem(
                vital_type=VitalType.HEART_RATE,
                value=72.0,
                unit="bpm",
                date=datetime.datetime(2024, 6, 1, 9, 30),
            )
        ]
    )

    assert VitalsOutput.model_validate(output.model_dump()) == output


def test_vitals_output_rejects_invalid_vital_type():
    with pytest.raises(ValidationError):
        VitalsOutput(
            items=[
                {
                    "vital_type": "not-a-vital",
                    "value": 72.0,
                    "unit": "bpm",
                    "date": datetime.datetime(2024, 6, 1, 9, 30),
                }
            ]
        )


# ---------------------------------------------------------------------------
# 7. get_encounters
# ---------------------------------------------------------------------------


def test_get_encounters_input_round_trip_with_optional_fields():
    model = GetEncountersInput(
        patient_id=7,
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 6, 1),
        limit=20,
    )

    assert GetEncountersInput.model_validate(model.model_dump()) == model


def test_get_encounters_input_rejects_negative_limit():
    with pytest.raises(ValidationError):
        GetEncountersInput(patient_id=7, limit=-1)


def test_encounter_item_round_trip():
    item = EncounterItem(
        encounter_id=101,
        date=datetime.datetime(2024, 6, 1, 9, 30),
        reason="Annual physical",
        provider="Dr. Smith",
        encounter_type=EncounterType.OFFICE_VISIT,
    )

    assert EncounterItem.model_validate(item.model_dump()) == item


def test_encounters_output_round_trip():
    output = EncountersOutput(
        items=[
            EncounterItem(
                encounter_id=101,
                date=datetime.datetime(2024, 6, 1, 9, 30),
                reason="Annual physical",
                provider="Dr. Smith",
                encounter_type=EncounterType.OFFICE_VISIT,
            )
        ]
    )

    assert EncountersOutput.model_validate(output.model_dump()) == output


def test_encounters_output_rejects_non_positive_encounter_id():
    with pytest.raises(ValidationError):
        EncountersOutput(
            items=[
                {
                    "encounter_id": 0,
                    "date": datetime.datetime(2024, 6, 1, 9, 30),
                    "reason": "Annual physical",
                    "provider": "Dr. Smith",
                    "encounter_type": EncounterType.OFFICE_VISIT,
                }
            ]
        )


# ---------------------------------------------------------------------------
# 8. get_appointments
# ---------------------------------------------------------------------------


def test_get_appointments_input_round_trip_with_optional_fields():
    model = GetAppointmentsInput(
        patient_id=7,
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 6, 1),
    )

    assert GetAppointmentsInput.model_validate(model.model_dump()) == model


def test_get_appointments_input_rejects_missing_patient_id():
    with pytest.raises(ValidationError):
        GetAppointmentsInput()


def test_appointment_item_round_trip():
    item = AppointmentItem(
        date=datetime.date(2024, 6, 1),
        time=datetime.time(9, 30),
        status=AppointmentStatus.SCHEDULED,
        provider="Dr. Smith",
    )

    assert AppointmentItem.model_validate(item.model_dump()) == item


def test_appointments_output_round_trip():
    output = AppointmentsOutput(
        items=[
            AppointmentItem(
                date=datetime.date(2024, 6, 1),
                time=datetime.time(9, 30),
                status=AppointmentStatus.SCHEDULED,
                provider="Dr. Smith",
            )
        ]
    )

    assert AppointmentsOutput.model_validate(output.model_dump()) == output


def test_appointments_output_rejects_invalid_status():
    with pytest.raises(ValidationError):
        AppointmentsOutput(
            items=[
                {
                    "date": datetime.date(2024, 6, 1),
                    "time": datetime.time(9, 30),
                    "status": "not-a-status",
                    "provider": "Dr. Smith",
                }
            ]
        )


# ---------------------------------------------------------------------------
# 9. check_drug_interactions (offline tool, no patient_id)
# ---------------------------------------------------------------------------


def test_check_drug_interactions_input_round_trip():
    model = CheckDrugInteractionsInput(drugs=["Warfarin", "Aspirin"])

    assert CheckDrugInteractionsInput.model_validate(model.model_dump()) == model


def test_check_drug_interactions_input_rejects_single_drug():
    with pytest.raises(ValidationError):
        CheckDrugInteractionsInput(drugs=["Warfarin"])


def test_check_drug_interactions_input_rejects_empty_list():
    with pytest.raises(ValidationError):
        CheckDrugInteractionsInput(drugs=[])


def test_check_drug_interactions_input_rejects_missing_drugs():
    with pytest.raises(ValidationError):
        CheckDrugInteractionsInput()


def test_drug_interaction_item_round_trip():
    item = DrugInteractionItem(
        drug_a="Warfarin",
        drug_b="Aspirin",
        severity=InteractionSeverity.MAJOR,
        description="Increased bleeding risk.",
    )

    assert DrugInteractionItem.model_validate(item.model_dump()) == item


def test_check_drug_interactions_output_round_trip():
    output = CheckDrugInteractionsOutput(
        items=[
            DrugInteractionItem(
                drug_a="Warfarin",
                drug_b="Aspirin",
                severity=InteractionSeverity.MAJOR,
                description="Increased bleeding risk.",
            )
        ]
    )

    assert CheckDrugInteractionsOutput.model_validate(output.model_dump()) == output


def test_check_drug_interactions_output_rejects_invalid_severity():
    with pytest.raises(ValidationError):
        CheckDrugInteractionsOutput(
            items=[
                {
                    "drug_a": "Warfarin",
                    "drug_b": "Aspirin",
                    "severity": "catastrophic",
                    "description": "Increased bleeding risk.",
                }
            ]
        )


# ---------------------------------------------------------------------------
# SourceRef provenance hook on output items (P3.1 forward-compat)
# ---------------------------------------------------------------------------


def test_medication_item_source_refs_default_to_none():
    item = MedicationItem(
        name="Lisinopril",
        dose="10mg",
        route="oral",
        status=MedicationStatus.ACTIVE,
        start_date=datetime.date(2024, 1, 1),
        end_date=None,
    )

    assert item.source_refs is None


def test_medication_item_accepts_optional_source_refs():
    item = MedicationItem(
        name="Lisinopril",
        dose="10mg",
        route="oral",
        status=MedicationStatus.ACTIVE,
        start_date=datetime.date(2024, 1, 1),
        end_date=None,
        source_refs=[SourceRef(tool_call_id="call-1", record_id="med-1", field="dose")],
    )

    restored = MedicationItem.model_validate(item.model_dump())

    assert restored == item
    assert restored.source_refs is not None
    assert restored.source_refs[0].field == "dose"
