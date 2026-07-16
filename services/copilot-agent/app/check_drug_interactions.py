"""Offline drug-drug interaction check (P3.6): verification-layer capability
(``docs/IMPLEMENTATION_PLAN.md`` Sec 4.4(2b)), in the same family as
``app.allergy_check`` (P3.4).

Checks every unordered pair among a list of drug names against the P3.5
SQLite dataset (``app.data.drug_interactions``) -- offline lookup only, no
OpenEMR call, no network, no client/token. Deliberately NOT a planner tool:
not in ``ToolName`` / ``TOOL_REGISTRY`` (see ``app.planner``) -- P3.7
(verdict) calls this directly with the patient's medications (+ any proposed
drug) and folds the result into warnings, the same way it folds
``app.allergy_check`` output.

**Echo vs. canonical.** ``DrugInteractionItem.drug_a``/``drug_b`` echo the
caller's original input strings (e.g. "Ibuprofen" as the user typed it), not
the normalized/canonical form -- so the UI shows names the caller recognizes.
Matching against the dataset still goes through
``app.data.drug_interactions.canonical_pair`` (casefold + strip + lexical
sort), so case/whitespace variants and pair order never affect whether a
match is found.

**Deterministic order.** Output items are sorted by canonical pair, not by
input order -- the trust layer values determinism over reflecting whatever
order the caller happened to list drugs in.

**Unknown drug is not an error.** A name with no seeded interactions simply
contributes no pairs to the result; ``check_drug_interactions`` never raises
for an unrecognized drug name.
"""

from __future__ import annotations

import sqlite3
from itertools import combinations

from app.data.drug_interactions import DB_PATH, canonical_pair
from app.schemas.common import InteractionSeverity
from app.schemas.tools import CheckDrugInteractionsInput, CheckDrugInteractionsOutput, DrugInteractionItem

# Maps the DDInter-style severity stored in the SQLite dataset (P3.5) onto
# the tool schema's InteractionSeverity enum. CONTRAINDICATED has no
# corresponding DB value -- the dataset only ever stores Major/Moderate/Minor
# (see app.data.drug_interactions.SEVERITY_LEVELS).
_SEVERITY_MAP: dict[str, InteractionSeverity] = {
    "Major": InteractionSeverity.MAJOR,
    "Moderate": InteractionSeverity.MODERATE,
    "Minor": InteractionSeverity.MINOR,
}


def check_drug_interactions(input_: CheckDrugInteractionsInput) -> CheckDrugInteractionsOutput:
    """Check every unordered pair among ``input_.drugs`` against the offline
    interaction dataset; one ``DrugInteractionItem`` per pair found, sorted
    by canonical pair. Empty (no interacting pairs) if none match."""
    db_uri = DB_PATH.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    try:
        found: list[tuple[tuple[str, str], DrugInteractionItem]] = []
        for drug_a, drug_b in combinations(input_.drugs, 2):
            drug_lo, drug_hi = canonical_pair(drug_a, drug_b)
            row = conn.execute(
                "SELECT severity, mechanism FROM interactions WHERE drug_lo = ? AND drug_hi = ?",
                (drug_lo, drug_hi),
            ).fetchone()
            if row is None:
                continue
            severity, mechanism = row
            found.append(
                (
                    (drug_lo, drug_hi),
                    DrugInteractionItem(
                        drug_a=drug_a,
                        drug_b=drug_b,
                        severity=_SEVERITY_MAP[severity],
                        description=mechanism,
                    ),
                )
            )
    finally:
        conn.close()
    found.sort(key=lambda entry: entry[0])
    return CheckDrugInteractionsOutput(items=[item for _, item in found])
