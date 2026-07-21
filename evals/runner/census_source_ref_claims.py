"""Issue #130 measurement spike, deliverable 1: offline structural census.

Deterministic, no LLM call, no network I/O beyond reading the already-
committed golden recordings (``evals/recordings/*.json``). For each recording,
counts claims from its ``VerifiedAnswer`` extraction call(s) whose surviving
citations are ALL ordinary ``SourceRef``s (zero ``DocumentCitation``s) -- the
unjudged-relevance exposure surface a SourceRef relevance gate would need to
cover (see the issue #130 ADR: ``app.semantic_support`` only re-judges
``DocumentCitation``-backed claims; a SourceRef-only claim's citations are
checked for provenance -- ``check_source_ref`` -- but never for RELEVANCE).

A claim counts toward ``source_ref_only_claims`` only when it carries at
least one citation and none of them is a ``DocumentCitation``. A claim with
NO citations at all has nothing to judge relevance of, so it is tracked
separately (``uncited_claims``) rather than folded into the exposure count --
see the module's test suite (``evals/runner/tests/test_census_source_ref_claims.py``)
for the exact boundary cases this distinction covers.

This script makes ZERO production-code changes and changes NO verdict --
purely a read-only counting pass over data already on disk. Usage (from repo
root, after activating the copilot-agent venv)::

    python evals/runner/census_source_ref_claims.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_EVALS_ROOT = Path(__file__).resolve().parents[1]
_RECORDINGS_DIR = _EVALS_ROOT / "recordings"

_VERIFIED_ANSWER_SCHEMA = "VerifiedAnswer"


@dataclass(frozen=True)
class CaseCensusRow:
    """One recording's claim census. ``source_ref_only_claims`` is the
    unjudged-relevance exposure count; ``uncited_claims`` is tracked
    separately (see module docstring) and is NOT part of the exposure
    surface."""

    case_id: str
    total_claims: int
    source_ref_only_claims: int
    uncited_claims: int


def count_claims_in_calls(calls: list[dict]) -> CaseCensusRow:
    """Count claims across every ``VerifiedAnswer`` extraction call in
    ``calls`` (a recording's ``"calls"`` list). A recording with no
    ``VerifiedAnswer`` call (e.g. a pure tool-selection case) reports all
    zeros -- there is nothing to census.

    ``case_id`` is left blank here (the caller, ``census_all``, knows the
    recording's file-derived id); this function is the pure, recording-scoped
    counting logic under test."""
    total_claims = 0
    source_ref_only_claims = 0
    uncited_claims = 0
    for call in calls:
        if call.get("schema") != _VERIFIED_ANSWER_SCHEMA:
            continue
        claims = call.get("response", {}).get("claims", [])
        for claim in claims:
            document_citations = claim.get("document_citations", [])
            source_refs = claim.get("source_refs", [])
            total_claims += 1
            if not document_citations and not source_refs:
                uncited_claims += 1
            elif not document_citations:
                source_ref_only_claims += 1
    return CaseCensusRow(
        case_id="",
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
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = count_claims_in_calls(payload.get("calls", []))
        rows.append(
            CaseCensusRow(
                case_id=path.stem,
                total_claims=row.total_claims,
                source_ref_only_claims=row.source_ref_only_claims,
                uncited_claims=row.uncited_claims,
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
