"""Regression guard for issue #199: stale Phase-1 issue-tracker numbers must
stay marked, not silently drift back to a bare ``#N``.

**The defect this guards against.** The Phase-1 -> Phase-2 sync merge
(``chore(sync): merge Phase 1 final state (v1.0) into Phase 2 base``) carried
Phase-1 commit history straight into this repo's tree, including comments and
docstrings that cite bare ``#N`` issue numbers meaning Phase 1's OWN tracker.
Several of those numbers are now live, differently-scoped issues/PRs in
*this* repo's tracker (e.g. Phase 1's "#185" == token-introspection dispatch
perf work, vs this repo's real #185 == subject-based feedback ownership).
A bare ``#N`` that resolves cleanly to the wrong issue is worse than a
dangling link -- it reads as current and correct. Issue #199 disambiguated
every instance found by this sweep to ``Phase 1 #N``.

**What this test does NOT do.** It does not re-derive the sweep (that
required cross-referencing each number against the live GitHub tracker and
``git blame``/ancestry against the Phase-1 merge parent -- expensive, and not
hermetic). It also does not attempt to catch every future stale reference a
different PR might introduce; the collision only exists between a Phase-1
number and *whichever* Phase-2 issue happens to reuse it, which is not
something a static, offline check can determine in general (this repo's
tracker is a live GitHub project, not a file in this checkout, and issue
#199 explicitly forbids live measurements/API calls from this kind of
regression suite). Instead, this locks in the specific instances issue #199
found and fixed, so a careless future edit (a revert, a bad merge, a
copy-paste of the pre-fix docstring from history) cannot silently re-drop
the ``Phase 1`` marker without turning this test red.

**Mutation-verified:** temporarily reverting any one of the anchors below
(e.g. ``Phase 1 #185`` -> ``#185`` in ``app/chat.py``) makes the
corresponding assertion fail with a clear message naming the file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# (file relative to repo root, exact substring that must be present).
# One anchor per distinct Phase-1-number/file group fixed by issue #199 --
# not every touched line, just enough to catch a revert of any of them.
_ANCHORS: list[tuple[str, str]] = [
    # #185: cache-miss introspection dispatch (Phase 1) vs this repo's real
    # #185 (subject-based feedback ownership) -- the exact collision #199
    # was filed over.
    ("services/copilot-agent/app/chat.py", "``_validate_token`` (Phase 1 #185) looks up"),
    ("services/copilot-agent/app/chat.py", "when necessary (Phase 1 #185)."),
    ("services/copilot-agent/app/introspection.py", "        (Phase 1 #185)."),
    ("services/copilot-agent/app/introspection.py", "(hit-only, Phase 1 #185) so both stay in sync"),
    # #237: roster-based cross-patient detection (Phase 1; this repo has no
    # #237 at all).
    ("services/copilot-agent/app/tools/patient_summary.py", "``get_patient_roster`` (Phase 1 #237 -- collect"),
    ("services/copilot-agent/app/chat.py", "``_roster_provider`` enables the Phase 1 #237"),
    ("services/copilot-agent/app/extraction.py", "# Phase 1 #237 roster-based cross-patient detection"),
    # #223/#224/#225: dangling Phase-1 numbers for the cross-patient guard /
    # name-binding / unresolvable-referent mechanisms.
    ("services/copilot-agent/app/extraction.py", '"""Deterministic PRE-dispatch guard (Phase 1 #223, extended by Phase 1 #224 and Phase 1 #237):'),
    ("services/copilot-agent/app/chat.py", "falling back to Phase 1 #223's numeric-only detection."),
    ("services/copilot-agent/app/chat.py", "Deterministic unresolvable-referent guard (Phase 1 #225):"),
    # #124/#126: collide with real, unrelated PRs #124 (SourceRef fabrication
    # root-cause) and #126 (SourceRef misrouting) -- Phase 1 used both
    # numbers for the OAuth per-user-token epic and the dev-token bridge.
    ("services/copilot-agent/app/config.py", "# RFC 7662 token introspection endpoint (Phase 1 #124 Phase 4)."),
    ("services/copilot-agent/app/config.py", "# DEV-ONLY dev-token bridge (issue Phase 1 #126, finding F4)."),
    # #130/#149/#153: each has TWO live meanings in this codebase -- a
    # Phase-1 one (security boundary / span-emission / recency notice) and
    # this repo's real, differently-scoped one (SourceRef content-relevance /
    # BP-hallucination non-determinism / claim-in-answer grounding gate).
    ("services/copilot-agent/app/extraction.py", "**The security boundary (refined Phase 1 #130).**"),
    ("services/copilot-agent/app/ollama_client.py", "(P4/Phase 1 #149)."),
    ("services/copilot-agent/app/chat.py", "# Wall-clock seam for the Phase 1 #153 recency notice:"),
    # #121/#194: collide with real, unrelated issues (#121 == SourceRef
    # content-relevance census finding; #194 == stale README model line).
    ("services/copilot-agent/app/chat.py", "Phase 1 #121): a small model can verbally attribute"),
    ("services/copilot-agent/app/extraction.py", "This hardens Phase 1 #194's ``apply_subject_check``"),
    # #175/#155: collide with real, unrelated merged PRs.
    ("evals/cases/stale_data/stale-only-encounter.yaml", "Formerly xfail (Phase 1 #175 resolves it)"),
    ("evals/runner/tests/test_review_queue_generator.py", "non-empty required field (Phase 1 #155)"),
    # #209/#211/#212/#213: dangling Phase-1 streaming-epic sub-issue numbers.
    ("services/copilot-agent/tests/test_reasoning_stream.py", "sub-issue C of epic Phase 1 #209 -- Phase 1 #211 streams the SSE relay, Phase 1 #212 streams"),
]


@pytest.mark.parametrize("relpath,anchor", _ANCHORS, ids=[f"{f}::{a[:40]}" for f, a in _ANCHORS])
def test_phase1_marker_still_present(relpath: str, anchor: str) -> None:
    path = _REPO_ROOT / relpath
    text = path.read_text(encoding="utf-8")
    assert anchor in text, (
        f"{relpath} no longer contains the issue #199 Phase-1 disambiguation "
        f"marker ({anchor!r}). A bare, unmarked issue number here would "
        f"resolve to a DIFFERENT, unrelated issue in this repo's tracker "
        f"(or to nothing at all) -- see issue #199."
    )
