"""DDInter-style offline drug-drug interaction dataset (P3.5): source data,
ingest script, and checksum for the SQLite artifact P3.6's
``check_drug_interactions`` tool will query offline (no OpenEMR call, no
network -- see ``docs/IMPLEMENTATION_PLAN.md`` Sec 4.4(2b)).

**Provenance / licensing** -- see the header comment in
``drug_interactions_source.csv`` for the full note. In short: this repo is
public, so DDInter's actual (CC BY-NC-SA 4.0) data is never redistributed
here. The source CSV is an originally curated demo dataset in DDInter's
drug-pair/severity schema style, not DDInter's data, and is not a clinical
reference.

**Why the checksum covers the DATA, not the ``.db`` file bytes.** SQLite's
on-disk format embeds page layout and free-list state that can differ
between two builds of logically identical data (page count/ordering is not
guaranteed byte-stable across runs/SQLite versions), so hashing the raw
file would make an unchanged dataset spuriously fail a reproducibility
check. ``canonical_checksum`` instead hashes a deterministic serialization
of the sorted logical rows -- reproducible by definition, and it is what
``test_rebuilt_database_data_matches_across_runs`` and
``test_committed_checksum_matches_current_source`` verify.

**Symmetric lookup.** Interactions are undirected: "ibuprofen + lisinopril"
and "lisinopril + ibuprofen" are the same fact. Rather than storing both
directions or requiring callers to try both orders, each row is stored once
under a canonical key -- the pair's two normalized names sorted
lexicographically into ``(drug_lo, drug_hi)`` (``canonical_pair``). Any
caller (P3.6 included) normalizes its two input names the same way before
querying, so lookup is order-independent by construction.

**Seam for P3.6.** This module owns the dataset (source CSV, ``DB_PATH``,
schema, normalization/canonical-key helpers). It deliberately does NOT
provide a "does this pair interact" query function -- opening the database
and matching/interpreting results against a patient's medication list is
P3.6's ``check_drug_interactions`` tool. P3.6 should reuse
``normalize_drug_name`` and ``canonical_pair`` from here rather than
re-implementing the normalization scheme, and query the ``interactions``
table at ``DB_PATH`` directly:
``SELECT severity, mechanism FROM interactions WHERE drug_lo = ? AND drug_hi = ?``
with the two input drug names run through ``canonical_pair`` first.

Run directly to (re)build the artifact from source:
``python -m app.data.drug_interactions``
"""

from __future__ import annotations

import csv
import hashlib
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
SOURCE_PATH = DATA_DIR / "drug_interactions_source.csv"
DB_PATH = DATA_DIR / "drug_interactions.db"
CHECKSUM_PATH = DATA_DIR / "drug_interactions.sha256"

# DDInter-style severity levels, enforced at load time and via the SQLite
# CHECK constraint in build_database.
SEVERITY_LEVELS: frozenset[str] = frozenset({"Major", "Moderate", "Minor"})


def normalize_drug_name(name: str) -> str:
    """Casefold + strip -- the same normalization convention
    ``app.allergy_check`` uses for medication/allergy names, so P3.6 can
    match dataset rows against ``MedicationItem.name`` with one consistent
    scheme instead of inventing a second one."""
    return name.strip().casefold()


def canonical_pair(drug_a: str, drug_b: str) -> tuple[str, str]:
    """Order-independent storage/lookup key for an interaction pair."""
    a, b = normalize_drug_name(drug_a), normalize_drug_name(drug_b)
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True)
class InteractionRow:
    """One canonicalized interaction pair, ready to insert or hash."""

    drug_lo: str
    drug_hi: str
    severity: str
    mechanism: str


def _non_comment_lines(path: Path) -> Iterator[str]:
    """Yield data lines from a source CSV, skipping ``#``-prefixed
    provenance/comment lines and blank lines (see the header comment in
    ``drug_interactions_source.csv``)."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            yield line


def load_source_rows(source_path: Path = SOURCE_PATH) -> list[InteractionRow]:
    """Parse and validate the source CSV into canonicalized
    ``InteractionRow`` objects. Raises ``ValueError`` on an unknown
    severity, a self-referential pair, or a duplicate pair."""
    reader = csv.DictReader(_non_comment_lines(source_path))
    rows: list[InteractionRow] = []
    seen: set[tuple[str, str]] = set()
    for record in reader:
        severity = record["severity"].strip()
        drug_a = record["drug_a"]
        drug_b = record["drug_b"]
        if severity not in SEVERITY_LEVELS:
            raise ValueError(
                f"Unknown severity {severity!r} for pair {drug_a!r}/{drug_b!r} "
                f"(allowed: {sorted(SEVERITY_LEVELS)})"
            )
        drug_lo, drug_hi = canonical_pair(drug_a, drug_b)
        if drug_lo == drug_hi:
            raise ValueError(f"Self-interaction pair is not valid: {drug_a!r}")
        key = (drug_lo, drug_hi)
        if key in seen:
            raise ValueError(f"Duplicate interaction pair: {drug_lo!r}/{drug_hi!r}")
        seen.add(key)
        rows.append(
            InteractionRow(
                drug_lo=drug_lo,
                drug_hi=drug_hi,
                severity=severity,
                mechanism=record["mechanism"].strip(),
            )
        )
    return rows


def build_database(rows: Iterable[InteractionRow], db_path: Path) -> None:
    """Build (overwrite) a SQLite database at ``db_path`` containing
    ``rows`` in a single ``interactions`` table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE interactions (
                drug_lo TEXT NOT NULL,
                drug_hi TEXT NOT NULL,
                severity TEXT NOT NULL CHECK (severity IN ('Major', 'Moderate', 'Minor')),
                mechanism TEXT NOT NULL,
                PRIMARY KEY (drug_lo, drug_hi)
            )
            """
        )
        conn.execute("CREATE INDEX idx_interactions_drug_lo ON interactions (drug_lo)")
        conn.execute("CREATE INDEX idx_interactions_drug_hi ON interactions (drug_hi)")
        conn.executemany(
            "INSERT INTO interactions (drug_lo, drug_hi, severity, mechanism) VALUES (?, ?, ?, ?)",
            [(row.drug_lo, row.drug_hi, row.severity, row.mechanism) for row in rows],
        )
        conn.commit()
    finally:
        conn.close()


def canonical_checksum(rows: Iterable[InteractionRow]) -> str:
    """SHA-256 hex digest of a deterministic serialization of ``rows`` --
    the reproducibility check target (see module docstring for why this is
    hashed instead of the ``.db`` file bytes)."""
    canonical_rows = sorted(rows, key=lambda r: (r.drug_lo, r.drug_hi))
    payload = "\n".join(f"{r.drug_lo}|{r.drug_hi}|{r.severity}|{r.mechanism}" for r in canonical_rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    """Rebuild ``DB_PATH`` and ``CHECKSUM_PATH`` from ``SOURCE_PATH``."""
    rows = load_source_rows()
    build_database(rows, DB_PATH)
    checksum = canonical_checksum(rows)
    CHECKSUM_PATH.write_text(checksum + "\n", encoding="utf-8")
    print(f"Built {DB_PATH} ({len(rows)} interactions); canonical data checksum {checksum}")


if __name__ == "__main__":
    main()
