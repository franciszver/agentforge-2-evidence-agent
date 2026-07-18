"""Golden queries for P3.3 hybrid-retrieval testing (`docs/W2_ARCHITECTURE.md`
"Testing Strategy" / SLO "Retrieval hit-rate").

Each entry is ``(query, expected_chunk_id)`` -- the chunk a correctly-working
sparse, dense, AND hybrid retriever must return somewhere in the top-k for
that query (``tests/test_retrieval.py``). Imported by both the test suite
and ``scripts/build_retrieval_embeddings.py`` (which embeds these query
strings, alongside every corpus chunk, into the committed
``app/data/retrieval_embeddings.json`` artifact) so the two never drift
apart.

One query per corpus document that has a golden-query mapping, spanning
UC1-UC3 (`docs/USERS.md`): pre-visit synthesis (blood-pressure trend),
medication safety/interactions (NSAID x ACE-inhibitor, warfarin x
antibiotics), and lab trend interpretation (A1c target, statin liver-enzyme
monitoring).
"""

from __future__ import annotations

GOLDEN_QUERIES: list[tuple[str, str]] = [
    ("What A1c target for most adults?", "a1c-targets#target-ranges"),
    ("Can I give ibuprofen with lisinopril?", "nsaid-interactions#ace-inhibitors-and-arbs"),
    ("What blood pressure counts as stage 2 hypertension?", "blood-pressure-categories#categories"),
    ("How often should I monitor liver enzymes on a statin?", "statin-monitoring#baseline-liver-function"),
    ("Does warfarin interact with antibiotics?", "anticoagulant-interactions#warfarin-specific-cautions"),
]
