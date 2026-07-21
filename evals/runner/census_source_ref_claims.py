"""Issue #130 measurement spike, deliverable 1: offline structural census.

Deterministic, no LLM call, no network I/O beyond reading the already-
committed golden recordings (``evals/recordings/*.json``), loaded through the
package's own ``ollama_replay.load_recording`` (the same typed record/replay
layer the rest of the eval suite uses -- see that module's docstring). For
each recording, counts claims from its ``VerifiedAnswer`` extraction call(s)
whose surviving citations are ALL ordinary ``SourceRef``s (zero
``DocumentCitation``s) -- the unjudged-relevance exposure surface a SourceRef
relevance gate would need to cover (see the issue #130 ADR: ``app.
semantic_support`` only re-judges ``DocumentCitation``-backed claims; a
SourceRef-only claim's citations are checked for provenance --
``check_source_ref`` -- but never for RELEVANCE).

**Exposure vs. uncited, the one invariant this module cares about:** a claim
counts toward ``source_ref_only_claims`` only when it carries at least one
citation and none of them is a ``DocumentCitation``. A claim with NO
citations at all has nothing to judge relevance of, so it is tracked
separately (``uncited_claims``) rather than folded into the exposure count.
See ``evals/runner/tests/test_census_source_ref_claims.py`` for the exact
boundary cases this distinction covers.

This script makes ZERO production-code changes and changes NO verdict --
purely a read-only counting pass over data already on disk. Usage (from repo
root, after activating the copilot-agent venv)::

    python evals/runner/census_source_ref_claims.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from runner.ollama_replay import RecordedCall, load_recording

_EVALS_ROOT = Path(__file__).resolve().parents[1]
_RECORDINGS_DIR = _EVALS_ROOT / "recordings"

_VERIFIED_ANSWER_SCHEMA = "VerifiedAnswer"


@dataclass(frozen=True)
class CaseCensusRow:
    """One recording's claim census (see module docstring for the
    exposure-vs.-uncited invariant)."""

    case_id: str
    total_claims: int
    source_ref_only_claims: int
    uncited_claims: int


class ClaimCounts(NamedTuple):
    """The three counts :func:`count_claims_in_calls` produces for a single
    recording, before ``census_all`` attaches the recording's ``case_id``."""

    total_claims: int
    source_ref_only_claims: int
    uncited_claims: int


def count_claims_in_calls(calls: list[RecordedCall]) -> ClaimCounts:
    """Count claims across every ``VerifiedAnswer`` extraction call in
    ``calls`` (a recording's decoded call list). A recording with no
    ``VerifiedAnswer`` call (e.g. a pure tool-selection case) reports all
    zeros -- there is nothing to census.

    This is the pure, recording-scoped counting logic under test; the
    caller, ``census_all``, attaches the recording's file-derived
    ``case_id``."""
    total_claims = 0
    source_ref_only_claims = 0
    uncited_claims = 0
    for call in calls:
        if call.schema != _VERIFIED_ANSWER_SCHEMA:
            continue
        claims = call.response.get("claims", [])
        for claim in claims:
            document_citations = claim.get("document_citations", [])
            source_refs = claim.get("source_refs", [])
            total_claims += 1
            if not document_citations and not source_refs:
                uncited_claims += 1
            elif not document_citations:
                source_ref_only_claims += 1
    return ClaimCounts(
        total_claims=total_claims,
        source_ref_only_claims=source_ref_only_claims,
        uncited_claims=uncited_claims,
    )


def census_all(recordings_dir: Path) -> list[CaseCensusRow]:
    """Walk every ``*.json`` file directly under ``recordings_dir`` and
    return one :class:`CaseCensusRow` per recording, sorted by ``case_id``
    (the filename stem) for a stable, reviewable table order."""
    rows = []
    for path in sorted(recordings_dir.glob("*.json")):
        calls = load_recording(path)
        counts = count_claims_in_calls(calls)
        rows.append(
            CaseCensusRow(
                case_id=path.stem,
                total_claims=counts.total_claims,
                source_ref_only_claims=counts.source_ref_only_claims,
                uncited_claims=counts.uncited_claims,
            )
        )
    return sorted(rows, key=lambda row: row.case_id)


def render_table(rows: list[CaseCensusRow]) -> str:
    """Render ``rows`` as a Markdown table with a trailing ``**Total**``
    row -- suitable for pasting directly into
    ``docs/MODEL_AND_HARDWARE_SELECTION.md``'s issue #130 findings section."""
    lines = [
        "| Case | Total claims | SourceRef-only (exposure) | Uncited |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row.case_id} | {row.total_claims} | {row.source_ref_only_claims} | {row.uncited_claims} |")
    total_claims = sum(row.total_claims for row in rows)
    total_exposure = sum(row.source_ref_only_claims for row in rows)
    total_uncited = sum(row.uncited_claims for row in rows)
    lines.append(f"| **Total** | {total_claims} | {total_exposure} | {total_uncited} |")
    return "\n".join(lines)


def main() -> None:
    rows = census_all(_RECORDINGS_DIR)
    print(render_table(rows))


if __name__ == "__main__":
    main()
