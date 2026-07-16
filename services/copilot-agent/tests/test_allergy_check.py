"""Exhaustive matrix for the deterministic allergy cross-check (P3.4).

``app.allergy_check`` cross-checks medication names against a patient's
recorded allergy list -- no LLM, no I/O, no clock. See the module docstring
in ``app.allergy_check`` for the matching-algorithm design (component-wise,
case/whitespace-insensitive, exact-token equality -- never substring
containment) and its precision rationale.

Hermetic and fully deterministic: no fixtures touch a real OpenEMR.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.allergy_check import AllergyConflict, check_allergy_conflicts
from app.schemas.common import AllergySeverity, MedicationStatus
from app.schemas.tools import AllergyItem, MedicationItem

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _med(name: str) -> MedicationItem:
    return MedicationItem(name=name, dose="10mg", route="oral", status=MedicationStatus.ACTIVE)


def _allergy(substance: str) -> AllergyItem:
    return AllergyItem(substance=substance, severity=AllergySeverity.MODERATE)


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------


def test_exact_name_match_conflicts():
    conflicts = check_allergy_conflicts([_med("Penicillin")], [_allergy("Penicillin")])

    assert conflicts == [AllergyConflict(medication_name="Penicillin", allergy_substance="Penicillin")]


def test_unrelated_medication_no_conflict():
    conflicts = check_allergy_conflicts([_med("Metformin")], [_allergy("Penicillin")])

    assert conflicts == []


def test_case_and_whitespace_variant_conflicts():
    conflicts = check_allergy_conflicts([_med("  penicillin  ")], [_allergy("PENICILLIN")])

    assert conflicts == [AllergyConflict(medication_name="  penicillin  ", allergy_substance="PENICILLIN")]


def test_compound_medication_name_component_conflicts():
    """A multi-drug medication conflicts if any component matches an allergy
    substance (medication compound -> simple allergy substance direction)."""
    conflicts = check_allergy_conflicts([_med("Amoxicillin/Clavulanate")], [_allergy("Amoxicillin")])

    assert conflicts == [AllergyConflict(medication_name="Amoxicillin/Clavulanate", allergy_substance="Amoxicillin")]


def test_compound_allergy_substance_component_conflicts():
    """The reverse direction: a compound allergy substance conflicts if any
    of its components matches a simple medication name."""
    conflicts = check_allergy_conflicts([_med("Aspirin")], [_allergy("Aspirin/Caffeine")])

    assert conflicts == [AllergyConflict(medication_name="Aspirin", allergy_substance="Aspirin/Caffeine")]


def test_hyphenated_compound_medication_name_conflicts():
    conflicts = check_allergy_conflicts([_med("Aspirin-Caffeine")], [_allergy("Aspirin")])

    assert conflicts == [AllergyConflict(medication_name="Aspirin-Caffeine", allergy_substance="Aspirin")]


def test_empty_allergy_list_no_conflict_no_crash():
    conflicts = check_allergy_conflicts([_med("Penicillin")], [])

    assert conflicts == []


def test_empty_medication_list_no_conflict_no_crash():
    conflicts = check_allergy_conflicts([], [_allergy("Penicillin")])

    assert conflicts == []


def test_both_lists_empty_no_conflict_no_crash():
    conflicts = check_allergy_conflicts([], [])

    assert conflicts == []


def test_multiple_medications_one_conflicting():
    medications = [_med("Metformin"), _med("Penicillin"), _med("Lisinopril")]

    conflicts = check_allergy_conflicts(medications, [_allergy("Penicillin")])

    assert conflicts == [AllergyConflict(medication_name="Penicillin", allergy_substance="Penicillin")]


def test_one_medication_conflicts_with_multiple_allergies():
    conflicts = check_allergy_conflicts([_med("Aspirin")], [_allergy("Penicillin"), _allergy("Aspirin")])

    assert conflicts == [AllergyConflict(medication_name="Aspirin", allergy_substance="Aspirin")]


def test_precision_guard_shared_substring_is_not_a_conflict():
    """"acetaminoPHEN" must not match a "PHEN..." allergy via substring --
    exact component/token equality only, per the module's conservative bias."""
    conflicts = check_allergy_conflicts([_med("Acetaminophen")], [_allergy("Phentermine")])

    assert conflicts == []


def test_seeded_demo_data_ibuprofen_conflict():
    """Pins the real demo fixture: evals/fixtures/seed.py's
    ``ALLERGY_TITLE = "Ibuprofen"`` seeds an Ibuprofen allergy for Phil
    Belford (pubpid 1), and ``app.tools.allergies._map_allergy`` maps the
    OpenEMR ``title`` column straight onto ``AllergyItem.substance`` -- so
    the seeded substance value really is the literal drug name "Ibuprofen",
    not a drug class ("NSAID"). Name-matching alone catches this demo
    conflict; no drug-class table is needed (see module docstring)."""
    conflicts = check_allergy_conflicts([_med("Ibuprofen")], [_allergy("Ibuprofen")])

    assert conflicts == [AllergyConflict(medication_name="Ibuprofen", allergy_substance="Ibuprofen")]


def test_allergy_conflict_is_frozen():
    conflict = AllergyConflict(medication_name="Penicillin", allergy_substance="Penicillin")

    with pytest.raises(dataclasses.FrozenInstanceError):
        conflict.medication_name = "Amoxicillin"  # type: ignore[misc]
