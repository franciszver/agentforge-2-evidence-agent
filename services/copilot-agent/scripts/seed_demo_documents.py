"""Seed the P5.1 demo dry run's document-ingestion state
(``docs/DEMO_SCRIPT.md``).

Ingests the already-committed synthetic lab-report PDF
(``tests/fixtures/lab_report_synthetic.pdf``) for the canonical
allergy-conflict demo patient (Phil Belford, pubpid ``1`` --
``evals/fixtures/seed.py``'s ``ALLERGY_CONFLICT_PUBPID``) so the chat panel
has a real, citable lab-PDF fact set to answer from during the demo's
graceful-failure beat (the fixture's deliberately redacted Creatinine
collection-date field, see ``scripts/generate_lab_pdf_fixture.py``'s module
docstring). Reuses that existing fixture rather than authoring new synthetic
data -- it already carries exactly the "one field is genuinely unreadable"
story the demo needs.

This is a separate, explicit ingestion step -- not something ``POST /chat``
does implicitly (see ``app.chat._build_evidence_workers``'s docstring:
"ingesting a NEW document stays a separate concern from a chat turn").
Wiring dispatches through ``app.supervisor.IntakeExtractorWorker`` -- the
SAME worker class ``_build_evidence_workers`` constructs for ``/chat`` --
built on ``OllamaClient.from_settings(settings, model=settings.
copilot_vision_model)`` plus ``LocalIngestionStore(settings.
copilot_ingestion_base_dir)``, so this script ingests through the
identical code path production/chat reads already-ingested facts back out
of, AND gets the same vision-model default and fail-closed
vision-capability guard (issue #204) as production, decided in that one
worker class rather than re-derived here.

**Idempotent.** Before ingesting, checks
``LocalIngestionStore.list_citations_for_patient`` for an existing
``lab_pdf`` citation belonging to this patient and skips the (re-)ingest
call if one is already present -- safe to run any number of times, matching
the discipline ``evals/fixtures/seed.py`` already establishes for its own
SQL-level seeding.

Run (from ``services/copilot-agent/``, dev stack up, demo dataset + SQL
fixtures already seeded via ``python ../../evals/fixtures/seed.py``):

    OLLAMA_BASE_URL=http://localhost:11435 python -m scripts.seed_demo_documents

Requires a reachable Ollama instance serving a vision-capable model for
document-ingestion extraction (``app.chat._build_evidence_workers``'s
docstring: vision ingestion always uses Ollama, independent of
``COPILOT_LLM_ENGINE``) -- see ``docs/DEVELOPERS_GUIDE.md`` for how the dev
stack's Ollama instance is provisioned.

**P5.1 dry-run correction (issue #27):** the host-run invocation above does
NOT work against ``docker/development-easy``'s dev stack as configured --
``agent`` and ``ollama`` sit on the ``copilot_internal`` compose network,
which is declared ``internal: true`` and publishes no host ports (a
deliberate security boundary, see ``docker-compose.copilot.yml``'s
module-level comment), so ``localhost:11435`` is never reachable from the
host, and even where a host port existed, ingesting from the host would
write to the host filesystem rather than the running ``agent`` container's
``/data/ingestion`` store that ``/chat`` actually reads facts back out of.
Verified live during the P5.1 dry run. Against THIS stack, ingest from
inside the ``agent`` container instead -- see ``docs/DEMO_SCRIPT.md``'s
setup section and ``scripts/ingest_demo_pdf.py`` (a container-side runner
that dispatches through the same ``app.supervisor.IntakeExtractorWorker``
(and, beneath it, the same ``attach_and_extract``/``LocalIngestionStore``
logic) as this module, minus the host-only pubpid resolution below, which
``get_pid_for_pubpid`` performs via ``docker compose exec mysql`` -- itself
only reachable from the host, not from inside ``agent``). This module is
still correct wherever ``OLLAMA_BASE_URL`` genuinely is host-reachable
(e.g. a different network topology, or a future dev-stack change that
publishes a host port for ``ollama``) -- it is the topology assumption that
was wrong, not this function. Also host-only for a second, independent
reason: this module's ``_LAB_PDF_FIXTURE`` resolves under ``tests/``, and
``.dockerignore`` excludes ``tests/`` from every built image -- even with
network access to ``ollama`` from inside a container, this exact module
could never find its own fixture there. ``scripts/ingest_demo_pdf.py``
sidesteps this too: the fixture is `docker cp`'d in explicitly (setup step
5), not read from the image.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SERVICE_ROOT.parents[1]
_LAB_PDF_FIXTURE = _SERVICE_ROOT / "tests" / "fixtures" / "lab_report_synthetic.pdf"

# evals/fixtures/seed.py owns pubpid<->fixture-role mapping; import it
# directly rather than re-declaring the pubpid here so the two seeding
# scripts can never silently drift apart.
sys.path.insert(0, str(_REPO_ROOT / "evals"))

from app.config import get_settings  # noqa: E402
from app.ingestion import LocalIngestionStore  # noqa: E402
from app.ollama_client import OllamaClient  # noqa: E402
from app.supervisor import IngestSubTask, IntakeExtractorWorker, VisionModelMisconfiguredError  # noqa: E402
from fixtures.seed import ALLERGY_CONFLICT_PUBPID, SeedError, get_pid_for_pubpid  # noqa: E402


class DemoDocumentSeedError(RuntimeError):
    """Raised when the demo document-ingestion seed cannot be completed."""


def _already_ingested(store: LocalIngestionStore, patient_id: int) -> bool:
    """True if ``patient_id`` already has at least one ``lab_pdf`` citation
    on disk -- the idempotency check (see module docstring)."""
    return any(citation.source_type == "lab_pdf" for citation in store.list_citations_for_patient(patient_id))


def seed_demo_documents() -> int:
    """Ensure the demo patient has an ingested lab-report PDF. Returns the
    resolved patient id. Safe to call any number of times."""
    if not _LAB_PDF_FIXTURE.exists():
        raise DemoDocumentSeedError(
            f"missing fixture: {_LAB_PDF_FIXTURE} -- regenerate via "
            "`python scripts/generate_lab_pdf_fixture.py` (services/copilot-agent/)"
        )

    patient_id = get_pid_for_pubpid(ALLERGY_CONFLICT_PUBPID)

    settings = get_settings()
    store = LocalIngestionStore(settings.copilot_ingestion_base_dir)

    if _already_ingested(store, patient_id):
        return patient_id

    vision_client = OllamaClient.from_settings(settings, model=settings.copilot_vision_model)
    worker = IntakeExtractorWorker(
        ollama_client=vision_client,
        document_store=store,
        fact_store=store,
        vision_model_capability_check=settings.copilot_vision_model_capability_check,
    )
    try:
        result = worker.run(
            IngestSubTask(patient_id=patient_id, file_path=str(_LAB_PDF_FIXTURE), doc_type="lab_pdf")
        )
    except VisionModelMisconfiguredError as exc:
        raise DemoDocumentSeedError(str(exc)) from exc
    if result.failed_pages:
        raise DemoDocumentSeedError(
            f"lab PDF ingestion had failed pages {result.failed_pages} for patient_id={patient_id} "
            "-- check Ollama reachability/model before running the demo"
        )
    return patient_id


def main() -> int:
    try:
        patient_id = seed_demo_documents()
    except (DemoDocumentSeedError, SeedError) as exc:
        print(f"seed_demo_documents failed: {exc}", file=sys.stderr)
        return 1

    print(f"Demo lab-report PDF ingested for patient_id={patient_id} (pubpid={ALLERGY_CONFLICT_PUBPID}, Phil Belford).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
