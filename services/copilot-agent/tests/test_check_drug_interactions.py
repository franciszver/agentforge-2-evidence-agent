"""Red-first tests for the offline drug-drug interaction lookup (P3.6).

``app.check_drug_interactions`` checks every unordered pair among a list of
drug names against the P3.5 SQLite dataset (``app.data.drug_interactions``)
-- no OpenEMR call, no network, no client/token. See the module docstring in
``app.check_drug_interactions`` for the matching/echo design.

Hermetic: queries the real committed ``DB_PATH`` directly (read-only), same
pattern as ``tests/test_drug_interactions_ingest.py``'s pair-lookup tests --
no mocking, no fixture DB, no running dev stack.
"""

from __future__ import annotations

from app.check_drug_interactions import check_drug_interactions
from app.data.drug_interactions import canonical_pair
from app.schemas.common import InteractionSeverity
from app.schemas.tools import CheckDrugInteractionsInput, DrugInteractionItem


def _check(*drugs: str) -> list[DrugInteractionItem]:
    return check_drug_interactions(CheckDrugInteractionsInput(drugs=list(drugs))).items


def test_known_interaction_pins_real_seeded_severity_and_description():
    """Pins the real committed DB row:
    ibuprofen,lisinopril,Moderate,"NSAID inhibits renal prostaglandins, ..."
    (see app/data/drug_interactions_source.csv)."""
    items = _check("ibuprofen", "lisinopril")

    assert items == [
        DrugInteractionItem(
            drug_a="ibuprofen",
            drug_b="lisinopril",
            severity=InteractionSeverity.MODERATE,
            description=(
                "NSAID inhibits renal prostaglandins, blunting the antihypertensive "
                "effect of ACE inhibitors and raising the risk of reduced renal "
                "function, especially with volume depletion."
            ),
        )
    ]


def test_no_interaction_between_two_real_unrelated_drugs():
    assert _check("acetaminophen", "levothyroxine") == []


def test_unknown_drug_returns_no_interactions_no_crash():
    assert _check("ibuprofen", "not-a-real-drug-xyz") == []


def test_case_insensitive_match_via_normalization():
    """Input echoed verbatim even though matching is case-insensitive."""
    items = _check("IBUPROFEN", "Lisinopril")

    assert items == [
        DrugInteractionItem(
            drug_a="IBUPROFEN",
            drug_b="Lisinopril",
            severity=InteractionSeverity.MODERATE,
            description=items[0].description,
        )
    ]


def test_multi_drug_returns_all_interacting_pairs():
    """warfarin+aspirin (Major) and warfarin+ibuprofen (Major) both interact;
    aspirin+ibuprofen is not a seeded pair."""
    items = _check("warfarin", "aspirin", "ibuprofen")

    pairs = {canonical_pair(item.drug_a, item.drug_b) for item in items}
    assert pairs == {("aspirin", "warfarin"), ("ibuprofen", "warfarin")}
    assert all(item.severity == InteractionSeverity.MAJOR for item in items)


def test_order_independence_same_pair_either_order():
    forward = _check("ibuprofen", "lisinopril")
    reverse = _check("lisinopril", "ibuprofen")

    assert len(forward) == 1
    assert len(reverse) == 1
    assert forward[0].severity == reverse[0].severity
    assert forward[0].description == reverse[0].description


def test_deterministic_output_order_sorted_by_canonical_pair():
    """Regardless of input order, the sequence of *canonical* pairs in the
    output is stable and sorted -- the trust layer values determinism. (The
    echoed ``drug_a``/``drug_b`` text can differ per call since it mirrors
    each call's own input order -- see module docstring.)"""
    items_a = _check("warfarin", "aspirin", "ibuprofen")
    items_b = _check("ibuprofen", "warfarin", "aspirin")

    canon_a = [canonical_pair(item.drug_a, item.drug_b) for item in items_a]
    canon_b = [canonical_pair(item.drug_a, item.drug_b) for item in items_b]
    assert canon_a == canon_b == sorted(canon_a)


def test_no_pairs_interact_returns_empty_list():
    assert _check("acetaminophen", "levothyroxine", "metformin") == []
