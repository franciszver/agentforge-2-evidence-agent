"""Deliberately-planted lexical distractors for P3.4 reranker testing
(`tests/test_reranking.py`, `scripts/build_reranker_scores.py`).

Each entry maps a golden query (`scripts/retrieval_golden_queries.py`) to a
corpus chunk id that is lexically close to the query (shares drug names,
section headings, or vocabulary with the expected chunk) but does not
correctly answer it -- a stand-in for the kind of chunk a keyword/embedding
hybrid can rank deceptively high while a reranker, which actually reasons
about the query's intent, should not. Picked by inspecting real
``HybridRetriever.retrieve_hybrid`` output for each golden query (a lexical
near-miss already surfaced in the real fused ranking, not a contrived one):

  * ``a1c-targets#target-ranges`` -- ``lipid-panel-reference#general-reference-categories``
    shares the "reference ranges for adults" shape but is about lipids, not A1c.
  * ``nsaid-interactions#ace-inhibitors-and-arbs`` -- ``renal-function-monitoring#ace-inhibitors-and-arbs``
    shares NSAID/ACE-inhibitor/renal vocabulary almost verbatim, but answers
    "how often to recheck creatinine", not "can I give these together".
  * ``blood-pressure-categories#categories`` -- ``hypertension-lifestyle#when-pharmacotherapy-is-added-sooner``
    explicitly says "Stage 2 hypertension" but answers "when to start
    pharmacotherapy", not "what counts as Stage 2".
  * ``statin-monitoring#baseline-liver-function`` -- ``statin-monitoring#muscle-symptoms-and-ck``
    same document, same "monitoring cadence" shape, but about CK/myopathy,
    not liver enzymes.
  * ``anticoagulant-interactions#warfarin-specific-cautions`` -- ``statin-monitoring#interaction-caution-cyp3a4-inhibitors``
    shares "antibiotics" and "interaction" vocabulary (macrolide antibiotics
    raising statin levels) but is not about warfarin at all.
"""

from __future__ import annotations

GOLDEN_DISTRACTORS: dict[str, str] = {
    "a1c-targets#target-ranges": "lipid-panel-reference#general-reference-categories",
    "nsaid-interactions#ace-inhibitors-and-arbs": "renal-function-monitoring#ace-inhibitors-and-arbs",
    "blood-pressure-categories#categories": "hypertension-lifestyle#when-pharmacotherapy-is-added-sooner",
    "statin-monitoring#baseline-liver-function": "statin-monitoring#muscle-symptoms-and-ck",
    "anticoagulant-interactions#warfarin-specific-cautions": "statin-monitoring#interaction-caution-cyp3a4-inhibitors",
}
