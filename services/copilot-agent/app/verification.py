"""Deterministic citation checker (P3.2): the trust layer's re-validation step.

Every factual ``Claim`` produced upstream (``app.schemas.verification``)
carries >=1 ``SourceRef`` pointing at a tool result already in this
conversation's cache. This module re-validates each citation independently,
with NO model call, NO clock, NO I/O -- purely a lookup + comparison over the
conversation's RAW tool results (``PlannerResult.raw_results``, the
verifier-only channel -- see decision 3). P3.3 (not implemented here) consumes
the per-claim / per-citation result this module returns to decide what to
strip from the final answer; P3.7 (also not here) rolls per-claim results into
the whole-answer verdict.

**Design decisions (P3.2), and why**

1. **Citations must carry an asserted value to be checkable.** A ``SourceRef``
   alone (``tool_call_id`` + ``record_id`` + ``field``) locates a fact but
   asserts nothing -- it cannot distinguish "cited the right dose" from
   "cited the wrong dose pointing at the same field". ``SourceRef`` gained
   one new optional field, ``asserted_value: str | None`` (see
   ``app.schemas.common``), so a citation can carry what the claim says the
   field's value is. The check is then a structured comparison --
   ``normalize(asserted) == normalize(resolved)`` -- never fuzzy
   substring/NLP matching against ``Claim.text`` prose (deliberately: prose
   matching is not deterministic enough for a trust story). The field is
   additive and optional so it does not disturb the P3.1 contract: every
   existing ``SourceRef``/``Claim`` test still passes unmodified. A citation
   that reaches this checker without an asserted value fails closed
   (``NO_ASSERTED_VALUE``) rather than being treated as a bare
   presence-check -- see ``check_source_ref``.

2. **Identity plumbing.** ``PlannerResult.raw_results`` entries carry no
   ``tool_call_id`` and tool-output records carry no ``record_id`` /
   ``uuid`` (checked: none of ``app.schemas.tools``'s ``*Item`` models expose
   one, and neither does the REST/FHIR tool layer).
   ``CacheIndex.from_raw_results`` therefore assigns both, positionally, when
   it builds the index for one conversation:
     - ``tool_call_id`` = ``f"call_{i}"`` for the 0-based order of
       ``raw_results`` (which is aligned 1:1 with the trace -- see decision
       3). An entry that produced no output (``None`` -- a binding violation
       or API error) still gets an id, so a ref to it resolves the *call*
       but finds zero records (``UNKNOWN_RECORD``) -- distinct from a ref to
       a call that was never made at all (``UNKNOWN_TOOL_CALL``).
     - ``record_id`` = the record's 0-based position (as a string) within
       the call's result: each entry of an ``items`` list for list-shaped
       tool outputs (medications, labs, ...), or ``"0"`` for a single-object
       output (``get_patient_summary``) -- treated as a one-record result so
       the scheme stays uniform. This is a positional convention, not a
       durable identity; it is stable only within one conversation (fine,
       since the cache itself only lives for one conversation).

3. **Verify against RAW record values -- and why that is safe.** The
   conversation's cached *trace* (``app.planner.ToolCallTrace.result``) is
   *post-quarantine* (``app.quarantine`` -- P2.9): every non-empty free-text
   field (medication ``name``/``dose``, problem ``title``, lab
   ``test_name``/``value``, allergy ``substance``, ...) is redacted to
   ``app.quarantine.REDACTED_SENTINEL`` there. Verifying against that
   skeleton would be a defect: the checker could never confirm the patient
   is on "Lisinopril" (the name is redacted), so P3.3 would strip the
   flagship demo's own correct answer. So this checker verifies against the
   RAW, pre-quarantine tool output instead, carried on the verifier-only
   ``PlannerResult.raw_results`` channel (never on ``ToolCallTrace``, so raw
   text never reaches the SSE stream or the observability trace).

   Using raw record text here is safe precisely because this entire path is
   deterministic -- ``CacheIndex`` build -> ``normalize`` -> equality ->
   ``CitationStatus`` -> (P3.3) stripping -- with NO LLM anywhere.
   Quarantine exists to stop injection text from steering the planner's
   *LLM* call; a deterministic ``normalize(a) == normalize(b)`` comparison
   has no such vulnerability -- an "IGNORE PREVIOUS INSTRUCTIONS" payload
   sitting in a raw drug-name field can only ever fail to equal the claim's
   asserted value. The trust story is exactly this: the planner LLM asserts
   "Lisinopril" (derived from the quarantine summary), and this checker
   deterministically confirms the RAW ``medication.name`` really is
   "Lisinopril" -> verified; a hallucinated "Metformin" mismatches ->
   stripped. ``REDACTED_FIELD`` is retained only as a defensive
   belt-and-suspenders branch (a raw result should never contain the
   sentinel); it is no longer the common path.

4. **Type-coercion / normalization rules** (``_values_match``), conservative
   by design -- a value that doesn't cleanly parse into the resolved value's
   type is a mismatch, never a coerced pass:
     - ``str`` resolved values: ``asserted_value`` compared case- and
       surrounding-whitespace-insensitively (``.strip().casefold()``).
       Covers enum values/labels ("Active" vs "active") and names
       ("Lisinopril" vs "lisinopril") the same way -- enums are already
       plain strings by the time they reach the cache (see
       ``app.quarantine``'s ``Enum`` handling).
     - ``int``/``float`` resolved values: ``asserted_value`` is parsed with
       ``float()``; a parse failure (e.g. "120 mmHg") is a mismatch, never
       a partial/lenient parse. "120" and "120.0" both match a resolved
       ``120`` since both parse to the same ``float``.
     - ``bool`` resolved values: ``asserted_value`` must casefold to exactly
       ``"true"`` or ``"false"``; anything else (including "yes"/"1") is a
       mismatch.
     - dates/times: already plain ISO strings by the time they reach the
       cache (``model_dump(mode="json")`` / ``isoformat()`` upstream), so
       they fall through the ``str`` rule above -- compared as exact ISO
       text, not date-parsed. A claim asserting only the date portion of a
       ``datetime`` field's full timestamp will therefore mismatch; no
       date-specific truncation/parsing is implemented (kept out of scope --
       conservative, and not in the requested matrix).
     - ``None`` resolved value (field present, cached as JSON ``null``):
       always a mismatch -- there is no asserted string that faithfully
       represents "no value".
     - Anything else (a resolved list/dict -- a citation pointing at a
       non-scalar field): always a mismatch; this checker only compares
       scalars.

**Seam to P3.3.** ``check_claim``/``check_claims`` return
``ClaimCheckResult``: the ``Claim``, one ``CitationCheckResult`` per
``SourceRef`` (never short-circuited -- P3.3 gets every citation's status,
not just the first failure), and ``passed`` (``bool``). ``passed`` is AND
across all of a claim's citations: a claim with N source refs is only
verified if every one of them independently checks out. Rationale: a claim
that bundles several facts (e.g. "on Lisinopril 10mg since 2024-01-01",
citing dose AND start_date) is only as trustworthy as its weakest citation --
partial grounding is not grounding. P3.3 strips/annotates on ``passed is
False``; the per-citation ``CitationStatus`` is there for a richer notice
than a bare "not found in record" if P3.3 wants one (e.g. distinguishing a
wrong value from an unresolvable citation). This module does not touch
``Claim.text`` and inserts no notices -- that's entirely P3.3.

**Recency notices (issue #153) -- an additive, separate concern from
citation re-validation above, not a change to it.**

The rule (also deterministic, no LLM): the record types the model may
present as "current" -- labs, vitals, encounters -- carry a ``date`` field.
``recency_notices`` scans every record actually returned in this turn's tool
results (``PlannerResult.raw_results``) and, for any record whose ``date``
is older than that tool's staleness threshold relative to an injected
``now``, produces a notice string naming the record's date -- so stale data
is never presented as current without its age.

**Why this scans every returned record, not per-claim citations** (a
deliberate deviation from the "for a VALID claim" framing this feature was
scoped under). The natural design would key this off ``ClaimCheckResult``
the same way citation checking does -- a notice only for records a VALID
claim actually cites. That is NOT what is implemented below: claim
extraction is itself an LLM call (``ClaimExtractor.extract_claims``), and
both the eval harness (``runner.pipeline.needs_verification``) and this
module's own citation-checking path only reach it for turns whose assertions
need a verdict. A recency check gated on claim extraction would never fire
for a turn whose recording has no extraction call -- exactly the #153
stale-data eval cases (``stale-only-lab``, ``stale-only-vitals``), whose
recordings only ever exercise ``Planner.run()``. Nor can the eval be made to
always extract: offline replay (``runner.ollama_replay.ReplayOllamaClient``)
pops exactly the calls recorded, in order; one unrecorded extra call raises
``RecordingExhaustedError`` rather than degrading gracefully. Scanning every
returned record instead needs nothing but ``Planner.run()``'s own output --
no new LLM call, ever -- so it is exactly as available against an
already-recorded run as against the live model. This is also a sound
approximation of the planner's own contract: ``app.planner``'s system prompt
already requires "Answer only from tool results already returned in this
conversation", so every record returned this turn is, by construction, in
scope for that turn's answer.

**The one clock exception.** The sections above advertise "NO model call, NO
clock, NO I/O" for citation re-validation -- that remains true for
``check_source_ref``/``check_claim``/``check_claims``, untouched by this
addition. ``recency_notices`` (and ``stale_record_date``) is the one
deliberate exception: staleness is inherently relative to "now", so it takes
``now: datetime`` as an explicit parameter and never reads the wall clock
itself -- callers own sourcing it (a fixed constant for the eval harness so
replay stays deterministic; the real wall clock read once at whatever
production call site applies it). This keeps the function itself pure and
hermetically testable with a fixed clock.

**Thresholds** (``_RECENCY_THRESHOLDS``, one clinical-cadence rationale
each): labs and vitals are expected to be re-measured at every visit, or at
least annually for chronic-disease monitoring (e.g. A1c) -- a reading over a
year old should not be presented as "current" -- so both get a 365-day
threshold. Encounter/visit history has a longer natural cadence (e.g. annual
physicals), so a visit record gets a longer, 730-day (2-year) bar for "not
current". Tools with no natural "current value" reading (medications,
allergies, problems, appointments, patient summary) have no threshold and
are never flagged, regardless of date.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any

from app.quarantine import REDACTED_SENTINEL
from app.schemas.common import SourceRef
from app.schemas.planner import ToolName
from app.schemas.verification import Claim
from app.tools._common import parse_fhir_datetime


class CitationStatus(StrEnum):
    """Why a single citation did or didn't re-validate."""

    VALID = "valid"
    UNKNOWN_TOOL_CALL = "unknown_tool_call"
    UNKNOWN_RECORD = "unknown_record"
    UNKNOWN_FIELD = "unknown_field"
    REDACTED_FIELD = "redacted_field"
    NO_ASSERTED_VALUE = "no_asserted_value"
    VALUE_MISMATCH = "value_mismatch"


@dataclass(frozen=True)
class CitationCheckResult:
    """The re-validation outcome for one ``SourceRef``."""

    source_ref: SourceRef
    status: CitationStatus

    @property
    def passed(self) -> bool:
        return self.status is CitationStatus.VALID


@dataclass(frozen=True)
class ClaimCheckResult:
    """The re-validation outcome for one ``Claim``: every citation's result,
    plus the claim-level verdict (AND across citations -- see module
    docstring)."""

    claim: Claim
    citation_results: list[CitationCheckResult]

    @property
    def passed(self) -> bool:
        # ``all([])`` is vacuously True -- guard against a degenerate claim
        # (zero citations) reaching this checker ever counting as verified.
        # A real ``Claim`` can't have zero refs (P3.1's min_length=1), but
        # nothing stops a caller from bypassing validation (e.g.
        # ``Claim.model_construct``), so this checker fails closed anyway.
        return bool(self.citation_results) and all(result.passed for result in self.citation_results)


class CacheIndex:
    """``(tool_call_id, record_id, field) -> value`` lookup over one
    conversation's RAW tool results. See module docstring, decision 2, for the
    id scheme."""

    def __init__(self, records_by_call: dict[str, list[dict[str, Any]]]) -> None:
        self._records_by_call = records_by_call

    @classmethod
    def from_raw_results(cls, raw_results: list[dict[str, Any] | None]) -> CacheIndex:
        """Build the index from ``PlannerResult.raw_results`` -- the
        verifier-only channel of un-redacted tool outputs, positionally
        aligned 1:1 with the trace (see module docstring, decision 3)."""
        records_by_call = {f"call_{i}": _extract_records(result) for i, result in enumerate(raw_results)}
        return cls(records_by_call)

    def records_for(self, tool_call_id: str) -> list[dict[str, Any]] | None:
        return self._records_by_call.get(tool_call_id)


def _extract_records(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Records within one tool call's raw result -- see module docstring,
    decision 2. ``None`` (an entry that produced no output) has zero records."""
    if result is None:
        return []
    items = result.get("items")
    if isinstance(items, list):
        return items
    return [result]


def _record_at(records: list[dict[str, Any]], record_id: str) -> dict[str, Any] | None:
    try:
        index = int(record_id)
    except ValueError:
        return None
    if index < 0 or index >= len(records):
        return None
    return records[index]


def _values_match(asserted: str, resolved: Any) -> bool:
    """Type-aware, conservative comparison -- see module docstring, decision 4."""
    if isinstance(resolved, bool):
        token = asserted.strip().casefold()
        if token not in {"true", "false"}:
            return False
        return (token == "true") is resolved
    if isinstance(resolved, (int, float)):
        try:
            parsed = float(asserted.strip())
        except ValueError:
            return False
        return parsed == float(resolved)
    if isinstance(resolved, str):
        return asserted.strip().casefold() == resolved.strip().casefold()
    return False


def check_source_ref(ref: SourceRef, index: CacheIndex) -> CitationCheckResult:
    """Re-validate one citation against ``index``. Never raises -- every
    failure mode maps to a ``CitationStatus``."""
    records = index.records_for(ref.tool_call_id)
    if records is None:
        return CitationCheckResult(source_ref=ref, status=CitationStatus.UNKNOWN_TOOL_CALL)

    record = _record_at(records, ref.record_id)
    if record is None:
        return CitationCheckResult(source_ref=ref, status=CitationStatus.UNKNOWN_RECORD)

    if ref.field not in record:
        return CitationCheckResult(source_ref=ref, status=CitationStatus.UNKNOWN_FIELD)

    resolved = record[ref.field]
    if resolved == REDACTED_SENTINEL:
        # Defensive belt-and-suspenders: raw results should never contain the
        # quarantine sentinel (this checker reads pre-quarantine values -- see
        # module docstring, decision 3). If one somehow does, fail closed
        # rather than compare an asserted value against placeholder text.
        return CitationCheckResult(source_ref=ref, status=CitationStatus.REDACTED_FIELD)

    if ref.asserted_value is None:
        return CitationCheckResult(source_ref=ref, status=CitationStatus.NO_ASSERTED_VALUE)

    if resolved is None or not _values_match(ref.asserted_value, resolved):
        return CitationCheckResult(source_ref=ref, status=CitationStatus.VALUE_MISMATCH)

    return CitationCheckResult(source_ref=ref, status=CitationStatus.VALID)


def check_claim(claim: Claim, index: CacheIndex) -> ClaimCheckResult:
    """Re-validate every citation on ``claim`` (never short-circuited)."""
    results = [check_source_ref(ref, index) for ref in claim.source_refs]
    return ClaimCheckResult(claim=claim, citation_results=results)


def check_claims(claims: list[Claim], index: CacheIndex) -> list[ClaimCheckResult]:
    """Re-validate a list of claims (the P3.3 entry point)."""
    return [check_claim(claim, index) for claim in claims]


# ---------------------------------------------------------------------------
# Recency notices (#153) -- see module docstring, "Recency notices", for the
# full rationale (why every returned record, not per-claim citations; the
# one deliberate clock exception; the threshold rationale).
# ---------------------------------------------------------------------------

LAB_RECENCY_THRESHOLD = timedelta(days=365)
VITALS_RECENCY_THRESHOLD = timedelta(days=365)
ENCOUNTER_RECENCY_THRESHOLD = timedelta(days=730)

_RECENCY_THRESHOLDS: dict[ToolName, timedelta] = {
    ToolName.GET_RECENT_LABS: LAB_RECENCY_THRESHOLD,
    ToolName.GET_VITALS: VITALS_RECENCY_THRESHOLD,
    ToolName.GET_ENCOUNTERS: ENCOUNTER_RECENCY_THRESHOLD,
}


def _as_aware_utc(value: datetime) -> datetime:
    """Normalize a datetime to timezone-aware UTC for comparison.

    Real OpenEMR/FHIR record dates can be tz-AWARE (offset-qualified) while an
    injected ``now`` may be naive (the eval's fixed clock) or aware
    (production ``datetime.now(timezone.utc)``) -- and subtracting a naive from
    an aware datetime raises ``TypeError``, which would crash a live ``/chat``
    on the first stale record. A naive datetime is interpreted as UTC (the
    zone OpenEMR stores in, and the zone production's ``now`` uses); an aware
    one is converted to UTC. Used only to make the staleness COMPARISON
    tz-safe -- never to alter a value returned to callers."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def stale_record_date(tool: ToolName, record: dict[str, Any], now: datetime) -> datetime | None:
    """The record's ``date`` field if ``tool`` has a recency threshold, the
    field parses (``app.tools._common.parse_fhir_datetime`` -- the same
    ISO-datetime parser ``get_recent_labs``/``get_vitals``/``get_encounters``
    already use for this exact field), and that date is older than the
    threshold relative to ``now`` -- else ``None``. Pure and clock-injected:
    ``now`` is always the caller's own value, never read internally (see
    module docstring, "The one clock exception").

    The staleness comparison is tz-safe: both ``now`` and the record date are
    normalized to aware-UTC (``_as_aware_utc``, naive treated as UTC) before
    subtraction, so a tz-aware record date and a naive ``now`` (or vice versa)
    compare cleanly instead of raising ``TypeError``. The datetime RETURNED is
    the raw parsed value (aware stays aware, naive stays naive) -- only the
    comparison is normalized."""
    threshold = _RECENCY_THRESHOLDS.get(tool)
    if threshold is None:
        return None
    record_date = parse_fhir_datetime(record.get("date"))
    if record_date is None:
        return None
    if _as_aware_utc(now) - _as_aware_utc(record_date) > threshold:
        return record_date
    return None


# Clinician-facing labels for the reading tools that carry a recency
# threshold -- surfaced in the notice instead of the raw snake_case enum
# value (``get_recent_labs``). MUST stay in sync with ``_RECENCY_THRESHOLDS``:
# ``_recency_notice_text`` is only ever reached for a tool that produced a
# stale date, which by construction has a threshold, so every key here that
# matters is present (a direct lookup, not ``.get``, keeps this total and the
# absence of a runtime fallback keeps the branch fully covered).
_TOOL_LABELS: dict[ToolName, str] = {
    ToolName.GET_RECENT_LABS: "lab results",
    ToolName.GET_VITALS: "vital signs",
    ToolName.GET_ENCOUNTERS: "encounter records",
}


def _recency_notice_text(tool: ToolName, record_date: datetime) -> str:
    # Wording deliberately does NOT assert the record is discussed in the
    # answer ("...data above..."): in a multi-tool turn the planner may fetch
    # a stale reading tool whose data the answer never mentions, so an
    # in-answer-placement claim would be misleading. The date stays ISO
    # (``YYYY-MM-DD``) so the year is present for the eval's
    # ``answer_contains`` check, and the phrase "may not reflect the patient's
    # current status" is kept verbatim (tests + eval semantics depend on it).
    return (
        f"Note: {_TOOL_LABELS[tool]} from {record_date.date().isoformat()} "
        "may not reflect the patient's current status."
    )


def recency_notices(
    tools: Sequence[ToolName], raw_results: Sequence[dict[str, Any] | None], now: datetime
) -> list[str]:
    """One notice per distinct stale (tool, date) actually returned this
    turn -- see module docstring, "Why this scans every returned record",
    for why this is keyed off every record ``Planner.run()`` returned rather
    than only claim-cited ones. Deduplicated (in first-seen order) so
    multiple records sharing one stale date (e.g. a systolic + diastolic
    reading from the same stale vitals check) produce one notice, not one
    per record."""
    notices: list[str] = []
    for tool, result in zip(tools, raw_results):
        for record in _extract_records(result):
            record_date = stale_record_date(tool, record, now)
            if record_date is not None:
                notices.append(_recency_notice_text(tool, record_date))
    return list(dict.fromkeys(notices))
