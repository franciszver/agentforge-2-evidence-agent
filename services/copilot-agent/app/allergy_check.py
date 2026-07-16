"""Deterministic allergy cross-check (P3.4): domain-constraint check (a) of
the verification layer (``docs/IMPLEMENTATION_PLAN.md`` Sec 4.4(2a)).

Cross-checks medication names already known to be mentioned against a
patient's recorded allergy list. NO LLM, NO I/O, NO clock -- a pure lookup +
comparison, in the same family as ``app.verification`` (citation checker) and
``app.rendering`` (claim stripping). P3.7 (verdict) and P3.8 (UI warning
banner) consume this module's output; neither is implemented here.

**Seam: "which medications are mentioned" is upstream, out of scope here.**
This module takes ``list[MedicationItem]`` -- already-resolved medication
records (e.g. the patient's active med list, or whatever set of medications
an upstream step decided are "mentioned" in an answer) -- not free text. That
extraction (answer text -> medication mentions) is the same kind of seam as
P3.2's claim extraction, and belongs with it, not here. This module is only
the cross-check function.

**Matching algorithm.** A medication conflicts with an allergy if the two
share at least one *component* in common, compared case/whitespace
-insensitively:

1. Normalize both strings: ``.strip().casefold()``.
2. Split each into components on common multi-drug-name separators --
   ``/``, ``+``, ``-``, ``,``, and whitespace (``_split_components``) --
   dropping empty pieces. A single-word name (no separators) yields exactly
   one component: itself. This makes a plain name-to-name match
   ("Penicillin" vs "Penicillin") just a special case of the same
   component-set check as a compound name, uniformly, in both directions:
   a compound *medication* ("Amoxicillin/Clavulanate") conflicts with the
   simple allergy substance "Amoxicillin", and a compound *allergy*
   substance ("Aspirin/Caffeine") conflicts with the simple medication name
   "Aspirin" -- the same set-intersection check handles both without
   separate code paths.
3. Conflict iff the medication's component set and the allergy's component
   set intersect.

**Precision: exact component equality, never substring containment.**
A false negative (missing a real conflict) is a safety miss; a false
positive (spurious warning) erodes trust in every other warning the system
raises. Splitting into components and requiring exact equality between them
-- rather than ``allergy_substance in medication_name`` or similar substring
checks -- is what stops "Acetaminophen" from matching a "Phentermine"
allergy just because both happen to contain "phen": neither name is split by
a separator at that point, so "phen" never becomes its own component to
compare. The tradeoff (documented, not fixed here): a shared *generic* word
component (e.g. "Vitamin" in "Vitamin B12" vs "Vitamin D") would still
intersect and register as a conflict. No stopword list is implemented --
the seeded demo data has no such case, and adding one without a real
observed false positive would be speculative scope creep. Noted as a stated
limitation.

**Drug-class mapping is OUT OF SCOPE.** An allergy recorded as a drug class
("NSAID") rather than a specific drug ("Ibuprofen") would need a
class -> drug table to catch e.g. an Ibuprofen conflict. Checked against the
actual seeded data (``evals/fixtures/seed.py``'s ``ALLERGY_TITLE =
"Ibuprofen"``, mapped onto ``AllergyItem.substance`` verbatim via
``app.tools.allergies._map_allergy``'s ``substance=record["title"]``): the
seeded and native demo allergy substances are literal drug names
("Ibuprofen", "penicillin"), never class names. Name/component matching
therefore already handles the demo's UC2 conflict case, so no class map is
implemented here. Drug-drug interactions (a genuinely different mechanism --
an offline SQLite lookup) are P3.5/P3.6, not this module.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.tools import AllergyItem, MedicationItem

_COMPONENT_SEPARATORS = re.compile(r"[/+\-,\s]+")


@dataclass(frozen=True)
class AllergyConflict:
    """One medication/allergy pair whose names share a matching component."""

    medication_name: str
    allergy_substance: str


def _components(text: str) -> set[str]:
    """Case/whitespace-insensitive component set -- see module docstring."""
    normalized = text.strip().casefold()
    return {part for part in _COMPONENT_SEPARATORS.split(normalized) if part}


def check_allergy_conflicts(
    medications: Sequence[MedicationItem], allergies: Sequence[AllergyItem]
) -> list[AllergyConflict]:
    """Cross-check ``medications`` against ``allergies``; one
    ``AllergyConflict`` per matching (medication, allergy) pair, in input
    order. Empty inputs yield an empty list."""
    conflicts: list[AllergyConflict] = []
    for medication in medications:
        medication_components = _components(medication.name)
        for allergy in allergies:
            if medication_components & _components(allergy.substance):
                conflicts.append(
                    AllergyConflict(medication_name=medication.name, allergy_substance=allergy.substance)
                )
    return conflicts
