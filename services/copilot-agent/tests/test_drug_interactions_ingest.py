"""Red-first ingest tests for P3.5's offline drug-drug interaction dataset.

Exercises ``app.data.drug_interactions`` -- the source CSV, the ingest
function that builds the SQLite artifact, and the checksum that proves the
build is reproducible from source. Does NOT test any matching/lookup logic
against ``MedicationItem`` -- that is P3.6's ``check_drug_interactions``
tool, a separate module.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.data.drug_interactions import (
    CHECKSUM_PATH,
    DB_PATH,
    SEVERITY_LEVELS,
    SOURCE_PATH,
    InteractionRow,
    build_database,
    canonical_checksum,
    canonical_pair,
    load_source_rows,
    normalize_drug_name,
)


def _query_pair(db_path: Path, drug_a: str, drug_b: str) -> sqlite3.Row | None:
    drug_lo, drug_hi = canonical_pair(drug_a, drug_b)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT * FROM interactions WHERE drug_lo = ? AND drug_hi = ?",
            (drug_lo, drug_hi),
        )
        return cursor.fetchone()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Row counts / severity enum
# ---------------------------------------------------------------------------


def test_row_count_matches_source_and_is_nonzero() -> None:
    rows = load_source_rows()
    non_comment_lines = [
        line
        for line in SOURCE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    # First non-comment line is the CSV header.
    assert len(rows) == len(non_comment_lines) - 1
    assert len(rows) > 0


def test_all_severities_within_allowed_set() -> None:
    rows = load_source_rows()
    assert SEVERITY_LEVELS == frozenset({"Major", "Moderate", "Minor"})
    for row in rows:
        assert row.severity in SEVERITY_LEVELS


def test_loader_rejects_unknown_severity(tmp_path: Path) -> None:
    bad_source = tmp_path / "bad.csv"
    bad_source.write_text(
        "drug_a,drug_b,severity,mechanism\nfoo,bar,Severe,made up\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="severity"):
        load_source_rows(bad_source)


# ---------------------------------------------------------------------------
# Known pairs present, order-independent; absent pair returns nothing
# ---------------------------------------------------------------------------


def test_known_demo_pair_ibuprofen_lisinopril_present_both_orders(tmp_path: Path) -> None:
    db_path = tmp_path / "drug_interactions.db"
    build_database(load_source_rows(), db_path)

    forward = _query_pair(db_path, "Ibuprofen", "Lisinopril")
    reverse = _query_pair(db_path, "lisinopril", "IBUPROFEN")

    assert forward is not None
    assert reverse is not None
    assert forward["severity"] in SEVERITY_LEVELS
    assert forward["drug_lo"] == reverse["drug_lo"]
    assert forward["drug_hi"] == reverse["drug_hi"]
    assert forward["severity"] == reverse["severity"]


def test_known_demo_pair_warfarin_aspirin_present() -> None:
    rows = load_source_rows()
    pair_keys = {(r.drug_lo, r.drug_hi) for r in rows}
    assert canonical_pair("warfarin", "aspirin") in pair_keys


def test_absent_pair_returns_nothing(tmp_path: Path) -> None:
    db_path = tmp_path / "drug_interactions.db"
    build_database(load_source_rows(), db_path)

    assert _query_pair(db_path, "acetaminophen", "levothyroxine") is None


# ---------------------------------------------------------------------------
# Normalization / symmetric pair key
# ---------------------------------------------------------------------------


def test_normalize_drug_name_casefolds_and_strips() -> None:
    assert normalize_drug_name("  Ibuprofen  ") == "ibuprofen"


def test_canonical_pair_is_order_independent() -> None:
    assert canonical_pair("Warfarin", "Aspirin") == canonical_pair("aspirin", "WARFARIN")


# ---------------------------------------------------------------------------
# Checksum / reproducibility
# ---------------------------------------------------------------------------


def test_checksum_is_deterministic_across_rebuilds() -> None:
    first = canonical_checksum(load_source_rows())
    second = canonical_checksum(load_source_rows())
    assert first == second
    assert len(first) == 64  # sha256 hex digest


def test_checksum_independent_of_row_order() -> None:
    rows = load_source_rows()
    reversed_rows = list(reversed(rows))
    assert canonical_checksum(rows) == canonical_checksum(reversed_rows)


def test_rebuilt_database_data_matches_across_runs(tmp_path: Path) -> None:
    rows = load_source_rows()
    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    build_database(rows, db_a)
    build_database(rows, db_b)

    def _all_rows(path: Path) -> list[tuple[str, str, str, str]]:
        conn = sqlite3.connect(path)
        try:
            return sorted(conn.execute("SELECT drug_lo, drug_hi, severity, mechanism FROM interactions").fetchall())
        finally:
            conn.close()

    assert _all_rows(db_a) == _all_rows(db_b)


def test_committed_checksum_matches_current_source() -> None:
    """The checked-in checksum file must reflect the checked-in source CSV --
    proves the committed artifact isn't stale relative to source."""
    committed = CHECKSUM_PATH.read_text(encoding="utf-8").strip()
    assert committed == canonical_checksum(load_source_rows())


def test_committed_database_matches_committed_checksum() -> None:
    """The checked-in ``.db`` -- the exact artifact P3.6 queries at runtime --
    must contain the data recorded in the committed checksum. Rebuild-from-
    source tests don't touch this file, so without this a hand-edited or
    stale committed ``.db`` would ship wrong data to runtime uncaught."""
    conn = sqlite3.connect(DB_PATH)
    try:
        db_rows = [
            InteractionRow(*record)
            for record in conn.execute(
                "SELECT drug_lo, drug_hi, severity, mechanism FROM interactions"
            )
        ]
    finally:
        conn.close()
    committed = CHECKSUM_PATH.read_text(encoding="utf-8").strip()
    assert canonical_checksum(db_rows) == committed
