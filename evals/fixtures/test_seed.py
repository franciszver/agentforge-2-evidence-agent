"""Integration tests for evals/fixtures/seed.py.

These hit the live dev-stack database (via ``docker compose exec``), so they
are marked ``integration`` and skipped by default unit-test runs. They
require the dev stack to be up with the demo dataset already imported (see
seed.py's module docstring for the load procedure).
"""

from __future__ import annotations

import pytest

from fixtures import seed

pytestmark = pytest.mark.integration


def test_run_seed_returns_the_three_canonical_patients() -> None:
    patients = seed.run_seed()
    pubpids = {p.pubpid for p in patients}
    assert pubpids == {
        seed.ALLERGY_CONFLICT_PUBPID,
        seed.MULTI_ENCOUNTER_PUBPID,
        seed.NO_LABS_STALE_PUBPID,
    }


def test_allergy_conflict_patient_has_ibuprofen_allergy() -> None:
    seed.run_seed()
    pid = seed.get_pid_for_pubpid(seed.ALLERGY_CONFLICT_PUBPID)
    count = seed._query_one(
        f"SELECT COUNT(*) FROM lists WHERE pid={pid} AND type='allergy' "
        f"AND title='{seed.ALLERGY_TITLE}';"
    )
    assert count == "1"


def test_no_labs_patient_truly_has_no_labs() -> None:
    seed.run_seed()
    pid = seed.get_pid_for_pubpid(seed.NO_LABS_STALE_PUBPID)
    order_count = seed._query_one(f"SELECT COUNT(*) FROM procedure_order WHERE patient_id={pid};")
    result_count = seed._query_one(
        f"SELECT COUNT(*) FROM procedure_result pr "
        f"JOIN procedure_report rep ON pr.procedure_report_id = rep.procedure_report_id "
        f"JOIN procedure_order po ON rep.procedure_order_id = po.procedure_order_id "
        f"WHERE po.patient_id={pid};"
    )
    assert order_count == "0"
    assert result_count == "0"


def test_stale_data_patient_has_only_old_encounters() -> None:
    seed.run_seed()
    pid = seed.get_pid_for_pubpid(seed.NO_LABS_STALE_PUBPID)
    count = seed._query_one(f"SELECT COUNT(*) FROM form_encounter WHERE pid={pid};")
    recent_count = seed._query_one(
        f"SELECT COUNT(*) FROM form_encounter WHERE pid={pid} AND date >= '2020-01-01';"
    )
    assert count == "1"
    assert recent_count == "0"


def test_multi_encounter_patient_has_multiple_encounters() -> None:
    seed.run_seed()
    pid = seed.get_pid_for_pubpid(seed.MULTI_ENCOUNTER_PUBPID)
    count = seed._query_one(f"SELECT COUNT(*) FROM form_encounter WHERE pid={pid};")
    assert count is not None
    assert int(count) >= 2


def test_adversarial_note_present_on_multi_encounter_patient() -> None:
    seed.run_seed()
    pid = seed.get_pid_for_pubpid(seed.MULTI_ENCOUNTER_PUBPID)
    plan = seed._query_one(
        f"SELECT plan FROM form_soap WHERE pid={pid} AND plan LIKE '%{seed.ADVERSARIAL_MARKER}%';"
    )
    assert plan is not None
    assert seed.ADVERSARIAL_MARKER in plan


def test_seeding_twice_is_idempotent_no_duplicate_rows() -> None:
    seed.run_seed()
    pid_phil = seed.get_pid_for_pubpid(seed.ALLERGY_CONFLICT_PUBPID)
    pid_susan = seed.get_pid_for_pubpid(seed.MULTI_ENCOUNTER_PUBPID)

    allergy_count_1 = seed._query_one(
        f"SELECT COUNT(*) FROM lists WHERE pid={pid_phil} AND type='allergy' "
        f"AND title='{seed.ALLERGY_TITLE}';"
    )
    encounter_count_1 = seed._query_one(f"SELECT COUNT(*) FROM form_encounter WHERE pid={pid_susan};")
    soap_count_1 = seed._query_one(
        f"SELECT COUNT(*) FROM form_soap WHERE pid={pid_susan} AND plan LIKE '%{seed.ADVERSARIAL_MARKER}%';"
    )

    seed.run_seed()  # second run must not duplicate anything

    allergy_count_2 = seed._query_one(
        f"SELECT COUNT(*) FROM lists WHERE pid={pid_phil} AND type='allergy' "
        f"AND title='{seed.ALLERGY_TITLE}';"
    )
    encounter_count_2 = seed._query_one(f"SELECT COUNT(*) FROM form_encounter WHERE pid={pid_susan};")
    soap_count_2 = seed._query_one(
        f"SELECT COUNT(*) FROM form_soap WHERE pid={pid_susan} AND plan LIKE '%{seed.ADVERSARIAL_MARKER}%';"
    )

    assert allergy_count_1 == allergy_count_2 == "1"
    assert encounter_count_1 == encounter_count_2
    assert soap_count_1 == soap_count_2 == "1"
