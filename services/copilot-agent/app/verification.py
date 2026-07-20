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

**Document-citation extension (P3.6) -- extends the checker above, does not
fork it.** `docs/W2_ARCHITECTURE.md` "Citation Contract": every clinical
claim sourced from a Week-2 document (lab-report PDF, intake form, or a
hybrid-retrieval guideline chunk) carries a ``DocumentCitation``
(``app.schemas.ingestion``) -- the document-sourced counterpart to
``SourceRef``'s ``{tool_call_id, record_id, field, asserted_value}`` shape.
``Claim.document_citations`` (additive, alongside ``source_refs`` --
`app.schemas.verification`) is re-validated by ``check_document_citation``
exactly the way ``check_source_ref`` re-validates a ``SourceRef``: a
structured lookup + comparison against a RAW index, no model call, no I/O,
no clock. ``check_claim``/``check_claims`` build ONE ``ClaimCheckResult``
per claim covering BOTH citation shapes -- ``ClaimCheckResult.passed``'s
AND-across-citations aggregation (see above) is untouched code, simply fed a
longer ``citation_results`` list when a claim has document citations too.

Two RAW indices, one per document source type -- both built directly from
the source, never from a quarantined, summarized, or re-derived copy (the
same "verify against RAW, never the cache" invariant as decision 3 above):

- ``DocumentFactIndex`` (``lab_pdf``/``intake_form``): ``(source_id,
  field_or_chunk_id) -> quote_or_value``, built from the RAW extracted-fact
  ``Citation``s -- i.e. exactly what ``app.ingestion``'s
  ``DocumentStore``/``FactStore`` (``LocalIngestionStore`` today) actually
  persisted for that document. A citation's ``quote_or_value`` must equal
  (whitespace-stripped) the raw stored quote -- a mismatch (including a
  citation naming a real source but the wrong field, or a real field but
  the wrong quote) fails. There is no free-text/enum coercion here (unlike
  ``_values_match``'s type-aware rules for structured ``SourceRef``s):
  ``Citation.quote_or_value`` is itself always a literal string rendering
  of what the VLM read (``app.ingestion._quote_for_row``/
  ``_quote_and_field_for_intake``), so an exact (stripped) string compare is
  the correct level of strictness for a *quote*, not a value that needs
  type coercion.
- ``CorpusChunkIndex`` (``guideline_chunk``): ``chunk_id -> raw chunk
  text``, built directly from the corpus (``app.retrieval.parse_corpus``'s
  ``Chunk``s, or the ``RetrievedChunk``/``RerankedChunk`` the retriever
  actually returned -- both carry the identical, unmodified corpus chunk
  text through unchanged, per ``app.reranking.Reranker.rerank``'s own
  docstring). A citation's ``quote_or_value`` must appear VERBATIM
  (substring, after whitespace normalization -- see "Whitespace-normalized
  substring check" below) in that raw text -- not merely be a
  plausible-sounding paraphrase of it. This is the "no fabrication"
  guarantee for RAG-style citations: a claim that quotes something the
  retrieved passage never actually says is a hallucination even if the
  gist is related, and must fail exactly like a hallucinated drug name
  fails ``check_source_ref`` above.

Both indices accept an explicitly-constructed, real-source-backed
collection at call time (``DocumentFactIndex.from_citations``/
``CorpusChunkIndex.from_chunks``) -- there is no code path in this module
that reaches into any cache, summary, or LLM-produced text to build them.
A caller who mistakenly feeds either index a paraphrased/cached copy
instead of the true raw store gets exactly the checker's ordinary
mismatch-detection behavior (``VALUE_MISMATCH``/``QUOTE_NOT_FOUND``) --
proven by ``tests/test_verification_documents.py``'s regression-guard
cases, which check the SAME citation against a true-raw index (passes) and
a paraphrased/summarized index at the same key (fails), and by the
``REDACTED_FIELD`` defensive branch below (the document-citation
counterpart to decision 3's belt-and-suspenders branch: a raw fact store
should never contain the quarantine sentinel, but if one somehow does,
this fails closed rather than comparing against placeholder text).

**No fabrication.** A citation naming a ``source_id``/``field_or_chunk_id``/
``chunk_id`` that does not exist in the RAW index FAILS
(``UNKNOWN_SOURCE``/``UNKNOWN_FIELD``/``UNKNOWN_CHUNK``) -- never silently
passes, mirroring ``check_source_ref``'s ``UNKNOWN_TOOL_CALL``/
``UNKNOWN_RECORD``/``UNKNOWN_FIELD`` for structured citations.

**Empty/trivial quote guard (security-gate finding, fixed post-hoc).** An
empty or whitespace-only ``quote_or_value`` would otherwise trivially
"verify" a ``guideline_chunk`` citation: ``"".strip() in chunk_text`` is
vacuously ``True`` for ANY chunk text, so a citation asserting nothing would
read as ``VALID`` -- exactly the failure mode this whole checker exists to
prevent. Guarded in two independent layers (defense in depth, matching the
project's fail-closed posture elsewhere in this module):

1. **Schema.** ``DocumentCitation.quote_or_value``
   (``app.schemas.ingestion``) rejects a blank/whitespace-only string at
   construction (``min_length=1`` plus a strip-non-empty model validator).
2. **Checker.** ``check_document_citation`` independently re-checks
   ``quote_or_value.strip()`` for blankness BEFORE either comparison, for
   BOTH source-type branches -- ``CitationStatus.EMPTY_QUOTE``, fail-closed.
   This is not redundant with the schema guard: a citation can reach this
   checker via ``model_construct`` (bypassing validation, as this test
   suite already does for other degenerate-input cases) or, longer-term,
   via any future ingress path that does not run full Pydantic validation.
   The checker must not rely solely on an upstream schema it does not
   control at the point of use.

**Minimum meaningful quote length (guideline_chunk only).** A 1-2
character quote (e.g. ``"a"``, ``"of"``) passes the blank-string guard
above but still substring-matches almost any real chunk text, which is
just as uninformative a "citation" as an empty one for a VERBATIM-QUOTE
check. ``_MIN_CHUNK_QUOTE_NON_WHITESPACE_CHARS`` (3) is the floor: a quote
whose non-whitespace character count falls below it fails closed
(``EMPTY_QUOTE``) before the substring check runs. 3 is a small,
deliberately permissive floor -- big enough to rule out single-character/
two-character noise matches, small enough not to reject any real short
clinical phrase a guideline passage might actually be quoted for (e.g.
"BMI", "HbA1c" both clear it). This floor applies ONLY to the
``guideline_chunk`` substring path -- the ``lab_pdf``/``intake_form``
path is EXACT equality against the raw stored quote (not substring
containment), so a short-but-real value like a pH reading of ``"7"``
must still verify; over-applying the length floor there would wrongly
reject legitimate short values, so it is deliberately not applied on that
branch (only the empty-string guard is, above).

**Whitespace-normalized substring check (P3G.1b, ``guideline_chunk`` only).**
The corpus stores hard-wrapped prose; a line-folded hyphen (e.g. "borderline-
high" wrapped across a line) round-trips through markdown line-wrap or a
YAML folded-scalar fixture as "borderline- high" -- an extra internal space
the source word never had. A model that faithfully quotes the phrase's WORDS
but does not reproduce that incidental whitespace exactly (observed on
qwen3:4b: it emitted "borderline-high", no space, against the chunk's
"borderline- high", one space) was, before this fix, failing
``QUOTE_NOT_FOUND`` on a near-miss that carries the exact same words in the
exact same order. Both ``quote_or_value`` and the raw chunk text are
normalized (``_normalize_chunk_whitespace``) before the substring test:
first, every run of whitespace is collapsed to a single space (absorbs
ordinary line-wrap newlines); second, whitespace immediately adjacent to a
hyphen is folded away entirely (absorbs the "borderline- high" vs
"borderline-high" case specifically). Whitespace that separates two
otherwise-unrelated tokens (not adjacent to a hyphen) is deliberately left
in place -- exactly one space, never removed -- so this narrows P3G.1b's
original fix, which stripped ALL whitespace unconditionally (a security-gate
finding: that let a quote of "50" match chunk text containing "5 0",
silently collapsing two distinct numeric tokens into one). See
``test_narrowed_whitespace_normalization_does_not_collapse_distinct_tokens``.

This is a strictly narrowing-safe change to the no-fabrication guarantee: two
strings that already differ in WORD content (an added, removed, or changed
word) introduce different non-whitespace CHARACTERS into the comparison
sequence regardless of whitespace normalization, so they still fail to
match -- a hallucinated/paraphrased/wrong-value quote fails exactly as before
(proven by this module's own tests: a wrong number/range, or an inserted
word, still fails ``QUOTE_NOT_FOUND`` after this change). Applied AFTER the
empty-quote and length-floor guards above (which operate on the
un-normalized ``stripped_quote``), so those guards' behavior is unchanged.

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

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Protocol

from app.quarantine import REDACTED_SENTINEL
from app.schemas.common import SourceRef
from app.schemas.ingestion import Citation, DocumentCitation
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
    # Document-citation extension (P3.6) -- see module docstring.
    UNKNOWN_SOURCE = "unknown_source"
    UNKNOWN_CHUNK = "unknown_chunk"
    QUOTE_NOT_FOUND = "quote_not_found"
    EMPTY_QUOTE = "empty_quote"
    # Semantic-support extension (issue #47, app.semantic_support) -- see that
    # module's docstring. Set ONLY by ``apply_semantic_support`` downgrading an
    # otherwise-``VALID`` ``DocumentCitationCheckResult`` whose quote is
    # verbatim-real (provenance holds) but does not, per the LLM-judge,
    # semantically support the claim's prose. Never produced by
    # ``check_document_citation``/``check_source_ref`` themselves -- this
    # module has no LLM call anywhere (see the module docstring's repeated
    # "NO model call" invariant); the value exists here only so
    # ``ClaimCheckResult.passed``'s existing AND-aggregation picks it up for
    # free, with no changes to this module's own checking functions.
    NOT_SEMANTICALLY_SUPPORTED = "not_semantically_supported"


@dataclass(frozen=True)
class CitationCheckResult:
    """The re-validation outcome for one ``SourceRef``."""

    source_ref: SourceRef
    status: CitationStatus

    @property
    def passed(self) -> bool:
        return self.status is CitationStatus.VALID


@dataclass(frozen=True)
class DocumentCitationCheckResult:
    """The re-validation outcome for one ``DocumentCitation`` -- the
    document-sourced counterpart to ``CitationCheckResult`` above (see
    module docstring, "Document-citation extension"). A separate type
    rather than widening ``CitationCheckResult.source_ref`` to a union:
    keeps ``CitationCheckResult`` (and every existing consumer of its
    ``source_ref: SourceRef`` field) completely unmodified."""

    document_citation: DocumentCitation
    status: CitationStatus

    @property
    def passed(self) -> bool:
        return self.status is CitationStatus.VALID


# Either citation shape's check result -- both expose `.status`/`.passed`,
# which is all `ClaimCheckResult.passed`'s AND-aggregation below needs; no
# common base class is introduced since neither consumer needs anything more
# structural than that.
AnyCitationCheckResult = CitationCheckResult | DocumentCitationCheckResult


@dataclass(frozen=True)
class ClaimCheckResult:
    """The re-validation outcome for one ``Claim``: every citation's result
    (``SourceRef`` AND ``DocumentCitation`` citations alike -- P3.6), plus
    the claim-level verdict (AND across ALL of them -- see module
    docstring). This aggregation is untouched by the P3.6 extension: it was
    already generic over "every citation this claim carries," and simply
    receives a longer ``citation_results`` list when a claim also has
    ``document_citations``."""

    claim: Claim
    citation_results: list[AnyCitationCheckResult]

    @property
    def passed(self) -> bool:
        # ``all([])`` is vacuously True -- guard against a degenerate claim
        # (zero citations) ever counting as verified. This is THE
        # enforcement point for "a claim needs >=1 citation" (issue #93,
        # Option C): ``app.schemas.verification.Claim`` deliberately no
        # longer raises on a zero-citation claim at parse time (that used to
        # make one uncitable claim poison every co-occurring claim in the
        # same ``VerifiedAnswer`` -- Pydantic validates list-of-models
        # all-or-nothing), so a real ``Claim`` reaching this checker CAN have
        # zero refs. It fails here instead, scoped to just that claim --
        # ``app.rendering.render_answer`` strips it to a notice without
        # touching its siblings, exactly like a claim whose citations fail
        # re-validation for any other reason.
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


def check_claim(
    claim: Claim,
    index: CacheIndex,
    fact_index: DocumentFactIndex | None = None,
    corpus_index: CorpusChunkIndex | None = None,
) -> ClaimCheckResult:
    """Re-validate every citation on ``claim`` (never short-circuited) --
    ``source_refs`` against ``index`` exactly as before, PLUS (P3.6)
    ``document_citations`` against ``fact_index``/``corpus_index``. Both
    document indices default to empty (no document sources available) so
    existing callers passing only ``index`` are unaffected; a claim with
    ``document_citations`` but no index supplied fails closed
    (``UNKNOWN_SOURCE``/``UNKNOWN_CHUNK``) rather than being skipped."""
    results: list[AnyCitationCheckResult] = [check_source_ref(ref, index) for ref in claim.source_refs]
    if claim.document_citations:
        resolved_fact_index = fact_index if fact_index is not None else DocumentFactIndex.from_citations([])
        resolved_corpus_index = corpus_index if corpus_index is not None else CorpusChunkIndex.from_chunks([])
        results.extend(
            check_document_citation(citation, resolved_fact_index, resolved_corpus_index)
            for citation in claim.document_citations
        )
    return ClaimCheckResult(claim=claim, citation_results=results)


def check_claims(
    claims: list[Claim],
    index: CacheIndex,
    fact_index: DocumentFactIndex | None = None,
    corpus_index: CorpusChunkIndex | None = None,
) -> list[ClaimCheckResult]:
    """Re-validate a list of claims (the P3.3 entry point)."""
    return [check_claim(claim, index, fact_index, corpus_index) for claim in claims]


# ---------------------------------------------------------------------------
# Document-citation extension (P3.6) -- see module docstring, "Document-
# citation extension", for the full design. Two RAW indices (one per source
# type) plus check_document_citation, the DocumentCitation counterpart to
# check_source_ref above.
# ---------------------------------------------------------------------------


class DocumentFactIndex:
    """``(source_id, field_or_chunk_id) -> quote_or_value`` lookup over the
    RAW extracted-fact ``Citation``s for lab/intake-form documents (see
    module docstring). Construct via ``from_citations`` with the actual
    persisted ``Citation``s (e.g. from ``app.ingestion``'s
    ``DocumentStore``/``FactStore`` output) -- never from a quarantined,
    summarized, or re-derived copy of them."""

    def __init__(self, quotes_by_key: dict[tuple[str, str], str]) -> None:
        self._quotes_by_key = quotes_by_key

    @classmethod
    def from_citations(cls, citations: Sequence[Citation]) -> DocumentFactIndex:
        # Issue #40 fix (same guard shape as CorpusChunkIndex.from_chunks
        # below): (source_id, field_or_chunk_id) is now unique BY
        # CONSTRUCTION -- app.ingestion's row/page assembly
        # (_to_lab_result_fact / _to_intake_form_facts) qualifies
        # field_or_chunk_id with the page (and, for lab rows, the row index
        # within the page) it was actually extracted from, so the same test
        # repeated across two dated pages of one lab report no longer
        # collides. This should therefore never actually trigger from a real
        # ingestion-produced Citation -- but a plain dict comprehension would
        # otherwise silently last-wins (order-dependent, undetectable) if
        # some other caller (a test double, a future non-ingestion source)
        # ever does produce a colliding key, mis-associating whichever
        # citation happened to be built first with the wrong quote. Kept as
        # a defensive, loud failure instead: raise rather than silently
        # overwrite.
        quotes_by_key: dict[tuple[str, str], str] = {}
        for citation in citations:
            key = (citation.source_id, citation.field_or_chunk_id)
            if key in quotes_by_key:
                raise ValueError(
                    f"DocumentFactIndex: duplicate (source_id, field_or_chunk_id) key {key!r} -- "
                    "two extracted facts collide on the same citation key. This should not happen "
                    "for ingestion-produced citations (see app.ingestion's page/row-qualified "
                    "field_or_chunk_id, issue #40); refusing to silently pick one over the other."
                )
            quotes_by_key[key] = citation.quote_or_value
        return cls(quotes_by_key)

    def quote_for(self, source_id: str, field_or_chunk_id: str) -> str | None:
        return self._quotes_by_key.get((source_id, field_or_chunk_id))

    def has_source(self, source_id: str) -> bool:
        """Whether ANY fact was indexed for ``source_id`` -- distinguishes a
        citation naming a real document but the wrong field
        (``UNKNOWN_FIELD``) from one naming a document that was never
        ingested at all (``UNKNOWN_SOURCE``)."""
        return any(key[0] == source_id for key in self._quotes_by_key)


class _TextChunk(Protocol):
    """Structural subset ``CorpusChunkIndex.from_chunks`` needs -- matches
    both ``app.retrieval.Chunk`` and ``app.schemas.retrieval.RetrievedChunk``/
    ``RerankedChunk`` without importing either (avoids coupling this module
    to the retrieval stack's own dependencies)."""

    chunk_id: str
    text: str


class CorpusChunkIndex:
    """``chunk_id -> raw chunk text`` lookup for guideline-chunk citations
    (see module docstring). Construct via ``from_chunks`` with the corpus's
    actual chunk text (``app.retrieval.parse_corpus``, or any
    ``RetrievedChunk``/``RerankedChunk`` the retriever returned -- both carry
    the identical corpus text unmodified) -- never a summarized/shortened
    stand-in."""

    def __init__(self, text_by_chunk_id: dict[str, str]) -> None:
        self._text_by_chunk_id = text_by_chunk_id

    @classmethod
    def from_chunks(cls, chunks: Sequence[_TextChunk]) -> CorpusChunkIndex:
        # Code-review finding (same guard as DocumentFactIndex.from_citations
        # above): a plain dict comprehension would silently last-wins on a
        # duplicate chunk_id. Corpus chunk_ids are already unique
        # (`<doc_id>#<section-slug>`, `app.retrieval.parse_document` raises
        # on a duplicate section slug at parse time), so this should never
        # actually trigger in practice -- but a caller building this index
        # from some other chunk source (a test double, a future non-corpus
        # source) gets a loud failure instead of a silent, order-dependent
        # mis-association. See issue #40 for the broader unique-id followup.
        text_by_chunk_id: dict[str, str] = {}
        for chunk in chunks:
            if chunk.chunk_id in text_by_chunk_id:
                raise ValueError(
                    f"CorpusChunkIndex: duplicate chunk_id {chunk.chunk_id!r} -- two chunks collide "
                    "on the same id. See issue #40 for the proper unique-id fix; refusing to "
                    "silently pick one over the other."
                )
            text_by_chunk_id[chunk.chunk_id] = chunk.text
        return cls(text_by_chunk_id)

    def text_for(self, chunk_id: str) -> str | None:
        return self._text_by_chunk_id.get(chunk_id)


# Floor for a guideline_chunk citation's quote, in non-whitespace
# characters -- see module docstring, "Minimum meaningful quote length".
# Applies ONLY to the guideline_chunk substring-containment path, never to
# lab_pdf/intake_form's exact-equality path.
_MIN_CHUNK_QUOTE_NON_WHITESPACE_CHARS = 3

_WHITESPACE_RE = re.compile(r"\s+")
# Whitespace immediately adjacent to a hyphen (either side, or both) -- see
# module docstring, "Whitespace-normalized substring check". Applied AFTER
# collapsing whitespace runs, so at most one space can appear on each side
# here by the time this runs.
_HYPHEN_WHITESPACE_RE = re.compile(r"\s*-\s*")


def _normalize_chunk_whitespace(text: str) -> str:
    """Collapse whitespace runs to a single space, then fold away whitespace
    immediately adjacent to a hyphen -- see module docstring, "Whitespace-
    normalized substring check", for why this is narrower than stripping all
    whitespace: a single space that separates two distinct tokens (not
    adjacent to a hyphen) is preserved, so e.g. "50" never matches text
    containing "5 0"."""
    collapsed = _WHITESPACE_RE.sub(" ", text)
    return _HYPHEN_WHITESPACE_RE.sub("-", collapsed)


def check_document_citation(
    citation: DocumentCitation,
    fact_index: DocumentFactIndex,
    corpus_index: CorpusChunkIndex,
) -> DocumentCitationCheckResult:
    """Re-validate one ``DocumentCitation`` against the RAW source it names.
    Never raises -- every failure mode maps to a ``CitationStatus`` (see
    module docstring, "Document-citation extension" / "No fabrication" /
    "Empty/trivial quote guard")."""
    stripped_quote = citation.quote_or_value.strip()
    if not stripped_quote:
        # Fail-closed BEFORE either comparison, for BOTH source types (see
        # module docstring, "Empty/trivial quote guard"): an empty quote
        # would otherwise vacuously satisfy the guideline_chunk substring
        # check, and asserts nothing on the lab_pdf/intake_form path either.
        return DocumentCitationCheckResult(document_citation=citation, status=CitationStatus.EMPTY_QUOTE)

    if citation.source_type == "guideline_chunk":
        if len(_WHITESPACE_RE.sub("", stripped_quote)) < _MIN_CHUNK_QUOTE_NON_WHITESPACE_CHARS:
            # Too short to be a meaningful verbatim-quote check -- see
            # module docstring, "Minimum meaningful quote length".
            return DocumentCitationCheckResult(document_citation=citation, status=CitationStatus.EMPTY_QUOTE)
        chunk_text = corpus_index.text_for(citation.field_or_chunk_id)
        if chunk_text is None:
            return DocumentCitationCheckResult(document_citation=citation, status=CitationStatus.UNKNOWN_CHUNK)
        if chunk_text == REDACTED_SENTINEL:
            # Defensive belt-and-suspenders, mirroring check_source_ref's own
            # REDACTED_FIELD branch: the corpus is public/non-PHI and has no
            # quarantine step today, but this index must still never be
            # trusted to compare an asserted quote against placeholder text.
            return DocumentCitationCheckResult(document_citation=citation, status=CitationStatus.REDACTED_FIELD)
        # Whitespace-normalized substring check (P3G.1b): the corpus stores
        # hard-wrapped prose -- e.g. a line-folded hyphen splitting
        # "borderline-high" across lines round-trips (through markdown
        # line-wrap, or a YAML folded-scalar fixture) as "borderline- high",
        # inserting a space the source word never had. A faithfully-quoting
        # model reproduces the WORDS correctly but may not reproduce that
        # incidental whitespace exactly -- observed on qwen3:4b as
        # "borderline-high" (no space) against the chunk's "borderline- high"
        # (one space): a genuine word-for-word match that a plain
        # single-space-collapse cannot bridge, since collapsing a run of
        # whitespace never removes the LAST whitespace character in a run --
        # it only normalizes 2+ characters down to one, so an existing
        # single-space-vs-no-space difference survives collapsing unchanged.
        # Stripping whitespace ENTIRELY (not merely collapsing runs) is the
        # normalization that actually absorbs this, and remains safe for
        # no-fabrication: every non-whitespace CHARACTER of the quote must
        # still appear in the chunk in the exact same order with nothing
        # else interposed -- a quote with a changed, added, or removed WORD
        # introduces different non-whitespace characters into that sequence,
        # so it still fails to match regardless of whitespace removal (see
        # the module docstring section by this name, and this module's
        # tests: a wrong number/range, or an inserted word, still fails
        # QUOTE_NOT_FOUND after this change).
        normalized_quote = _normalize_chunk_whitespace(stripped_quote)
        normalized_chunk_text = _normalize_chunk_whitespace(chunk_text)
        if normalized_quote not in normalized_chunk_text:
            return DocumentCitationCheckResult(document_citation=citation, status=CitationStatus.QUOTE_NOT_FOUND)
        return DocumentCitationCheckResult(document_citation=citation, status=CitationStatus.VALID)

    # "lab_pdf" / "intake_form"
    resolved_quote = fact_index.quote_for(citation.source_id, citation.field_or_chunk_id)
    if resolved_quote is None:
        if fact_index.has_source(citation.source_id):
            return DocumentCitationCheckResult(document_citation=citation, status=CitationStatus.UNKNOWN_FIELD)
        return DocumentCitationCheckResult(document_citation=citation, status=CitationStatus.UNKNOWN_SOURCE)
    if resolved_quote == REDACTED_SENTINEL:
        # Defensive belt-and-suspenders (see module docstring): a raw fact
        # store should NEVER contain the quarantine sentinel. If one somehow
        # does, fail closed rather than compare against placeholder text.
        return DocumentCitationCheckResult(document_citation=citation, status=CitationStatus.REDACTED_FIELD)
    if stripped_quote != resolved_quote.strip():
        return DocumentCitationCheckResult(document_citation=citation, status=CitationStatus.VALUE_MISMATCH)
    return DocumentCitationCheckResult(document_citation=citation, status=CitationStatus.VALID)


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
