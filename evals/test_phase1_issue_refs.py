"""Regression guard for issue #199: stale Phase-1 issue-tracker numbers must
stay marked, not silently drift back to a bare ``#N``.

**Gate-3 MAJOR-3 follow-up.** The positive assertions below (``anchor in
text``) only prove the marked spelling still exists somewhere in the file --
they do NOT fail if the SAME line reverts to a bare ``#N`` elsewhere, or if a
brand-new bare ``#N`` citation of a Phase-1 number is introduced anywhere
else in the swept trees. That gap is exactly how issue #199 itself shipped
30 wrong references over a green suite. A blanket "no bare ``#N`` anywhere"
rule is wrong for most of these numbers: ``#130``, ``#149``, ``#153``,
``#185``, ``#176``, ``#124``, ``#140``, ``#144``, ``#157``, ``#54``, and
``#60`` all also name real, differently-scoped Phase-2 issues in *this*
repo's own tracker, and their bare (correct) citations must stay bare.
Telling those apart requires either a full allowlist of legitimate bare
sites or a snapshot of every bare occurrence -- both bigger than this
follow-up's scope.

A strict subset does NOT have that ambiguity: ``#223``, ``#224``, ``#225``,
``#237``, ``#209``, ``#211``, and ``#212``/``#213`` name Phase-1-only
mechanisms (the cross-patient guard family and the streaming epic) that this
repo's tracker has never reused -- see each anchor's comment above for the
"this repo has no #N at all" note. For those numbers, ANY bare occurrence in
the swept trees (outside this file's own commentary) is unconditionally
wrong, so a negative assertion is both cheap and safe.
``test_no_new_bare_dangling_references`` below enforces exactly that,
scoped to only those numbers. (Issue #201 closed the last carve-out: the
bare ``#237`` in ``get_patient_roster``'s own docstring in
``app/tools/patient_summary.py`` is now marked ``Phase 1 #237`` like every
other site, so no exemption remains.)

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

**The dual-meaning numbers cannot be guarded by a mechanical rule.** For
``#130``, ``#149``, ``#153``, ``#185``, ``#176``, ``#124``, ``#140``,
``#144``, ``#157``, ``#54``, and ``#60``, a bare ``#N`` is not an error --
this repo's own tracker legitimately reuses each of those numbers for a
real, differently-scoped Phase-2 issue, and most bare citations of them in
this codebase are exactly that: correct references to this repo's own
issue/PR N, not a leftover from the Phase-1 merge. No regex or line-based
check can tell those apart; it requires reading each occurrence's
surrounding prose and judging which concept it describes, then looking up
what this repo's issue/PR N actually is. That is a one-time content review,
not something this test (or any future automated sweep) re-derives or
re-verifies. The only guarantee this file's tests provide for these eleven
numbers is that the specific instances issue #199 found and marked
(``_ANCHORS`` above) stay marked -- nothing here proves no *other* bare
occurrence of one of these numbers is a mismarked Phase-1 leftover, before
or after this test was written. That correctness claim rests entirely on
the manual content-classification review recorded in issue #199's PR
description, which is not re-checked by any test in this repo and can
silently go stale if a future merge reintroduces Phase-1 history again.
The only numbers this suite mechanically guarantees have zero false
negatives are the genuinely-dangling ones in ``_DANGLING_NUMBERS`` below,
via ``test_no_new_bare_dangling_references``.

**Mutation-verified:** temporarily reverting any one of the anchors below
(e.g. ``Phase 1 #185`` -> ``#185`` in ``app/chat.py``) makes the
corresponding assertion fail with a clear message naming the file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# (file relative to repo root, exact substring that must be present).
# One anchor per distinct Phase-1 number fixed by issue #199 -- not every
# (number, file) group touched (measured: 21 of 102 such groups have an
# anchor here, and only 9 of the 46 files issue #199 changed have any anchor
# at all) and not every touched line. This proves the marker survives for
# each number somewhere, so a wholesale revert of a number's fix goes red;
# it does NOT prove every individual fixed line stays fixed, and it does NOT
# catch a brand-new bare citation elsewhere -- see
# ``test_no_new_bare_dangling_references`` below for that gap, currently
# closed only for the genuinely-dangling numbers.
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
    ("services/copilot-agent/app/tools/patient_summary.py", 'pair (Phase 1 #237\n    roster-based cross-patient detection'),
    ("services/copilot-agent/app/chat.py", "``_roster_provider`` enables the Phase 1 #237"),
    ("services/copilot-agent/app/extraction.py", "# Phase 1 #237 roster-based cross-patient detection"),
    # #223/#224/#225: dangling Phase-1 numbers for the cross-patient guard /
    # name-binding / unresolvable-referent mechanisms.
    ("services/copilot-agent/app/extraction.py", "(Phase 1 #223, extended by Phase 1 #224 and Phase 1 #237)"),
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
    ("services/copilot-agent/tests/test_reasoning_stream.py", "epic Phase 1 #209 -- Phase 1 #211 streams the SSE relay, Phase 1 #212 streams"),
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


# Numbers that name Phase-1-only mechanisms this repo's own tracker has never
# reused -- unlike #130/#149/#153/#185/#176/#124/#140/#144/#157/#54/#60, a
# bare occurrence of any of these is unconditionally wrong, never a
# legitimate same-numbered Phase-2 issue.
_DANGLING_NUMBERS: tuple[str, ...] = ("223", "224", "225", "237", "209", "211", "212", "213")

# Trees this guard sweeps -- same scope issue #199's own sweep covered.
_SWEPT_DIRS: tuple[str, ...] = ("services/copilot-agent", "evals")

# (relpath, 1-indexed line number) pairs exempt from the negative check --
# this file's own commentary above (which names the dangling numbers in
# prose, not as citations). Previously also carried the #201-owned
# ``app/tools/patient_summary.py:154`` carve-out; #201 fixed that bare
# citation, so no non-self exemption remains.
_EXEMPT: frozenset[tuple[str, int]] = frozenset()
_SELF_RELPATH = "evals/test_phase1_issue_refs.py"


def _iter_swept_files() -> list[Path]:
    files: list[Path] = []
    for tree in _SWEPT_DIRS:
        root = _REPO_ROOT / tree
        for pattern in ("*.py", "*.yaml", "*.yml", "*.md"):
            for path in root.rglob(pattern):
                if any(
                    part in {".venv", "__pycache__", ".pytest_cache", "node_modules"}
                    for part in path.parts
                ):
                    continue
                files.append(path)
    return files


def test_no_new_bare_dangling_references() -> None:
    """MAJOR-3: a bare ``#N`` for a genuinely-dangling Phase-1 number is
    unconditionally wrong anywhere in the swept trees -- this repo's tracker
    has never reused these numbers for anything else. Unlike
    ``test_phase1_marker_still_present`` (which only proves a known-good
    spelling survives), this fails on ANY bare occurrence: a revert of an
    existing anchor, or a brand-new bare citation this PR never touched.

    **Mutation-verified:** temporarily appending a bare ``#237`` anywhere in
    the swept trees turns this test red with the offending file:line quoted
    in the failure message; removing it turns the test green again.
    ``_EXEMPT`` is currently empty (see its own comment for why) but the
    per-line carve-out below is retained deliberately, for any future
    same-file, non-citation prose occurrence of a dangling number.
    """
    import re

    offenders: list[str] = []
    for number in _DANGLING_NUMBERS:
        bare_pattern = re.compile(rf"(?<!Phase 1 )#{number}\b")
        for path in _iter_swept_files():
            relpath = path.relative_to(_REPO_ROOT).as_posix()
            if relpath == _SELF_RELPATH:
                continue
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if (relpath, lineno) in _EXEMPT:
                    continue
                if bare_pattern.search(line):
                    offenders.append(f"{relpath}:{lineno}: bare #{number} -- {line.strip()!r}")

    assert not offenders, (
        "Bare reference(s) to a genuinely-dangling Phase-1-only issue number "
        "found (must read 'Phase 1 #N' -- this repo's tracker has never "
        f"reused these numbers for anything else):\n" + "\n".join(offenders)
    )
