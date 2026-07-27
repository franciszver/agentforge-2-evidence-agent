"""In-container ingestion runner for the P5.1 demo dry run
(``docs/DEMO_SCRIPT.md``).

Mirrors ``seed_demo_documents.py``'s ``seed_demo_documents()`` logic
exactly, EXCEPT for pubpid -> ``patient_id`` resolution: that function
transitively imports ``evals/fixtures/seed.py``, whose
``get_pid_for_pubpid`` shells out to ``docker compose exec mysql`` --
which requires a ``docker`` CLI and socket access the ``agent`` container
does not have (correctly -- it is not a docker-in-docker environment).
``patient_id`` is therefore resolved HOST-SIDE (``python
evals/fixtures/seed.py``, part of the demo's standard setup) and passed in
here as an explicit argument, so this script has no host-only dependency
and can run entirely inside the ``agent`` container -- the only place that
can reach the internal-only ``ollama`` service and write to
``LocalIngestionStore``'s on-disk state that ``/chat`` reads back out of.

Run inside the ``agent`` container (``docs/DEMO_SCRIPT.md``'s setup step 5
has the full command, including the required GPU stop/start bracketing
around this call). No ``OLLAMA_MODEL`` override is needed: this script
builds its own vision-role ``OllamaClient`` on
``settings.copilot_vision_model`` (default ``qwen2.5vl:7b``, issue #204),
the same setting/wiring ``app.chat._build_evidence_workers`` uses for
``/chat``'s intake-extractor worker, so a plain
``docker exec -w /app development-easy-agent-1 python
/data/repo_ingest/ingest_demo_pdf.py <patient_id> <pdf_path>`` already gets
a vision-capable model without any per-call env override:

    docker exec -w /app development-easy-agent-1 \\
        python /data/repo_ingest/ingest_demo_pdf.py <patient_id> <pdf_path>

The ``app.*`` imports below resolve via the ``copilot-agent`` package
installed into the image's site-packages (the ``Dockerfile``'s ``pip
install .``), not via this script's own location or the ``-w /app``
working directory -- this script can therefore be copied to and run from
anywhere writable in the container (it lives under ``/data/repo_ingest``
above, not ``/app``) with no ``PYTHONPATH``/``sys.path`` setup of its own.

**Idempotent.** Checks ``LocalIngestionStore.list_citations_for_patient``
for an existing ``lab_pdf`` citation belonging to ``patient_id`` and skips
the (re-)ingest call if one is already present.

**Vision-model fail-closed (issue #204).** Ingestion is dispatched through
``app.supervisor.IntakeExtractorWorker`` -- the SAME worker class
``app.chat._build_evidence_workers`` builds for ``/chat`` -- rather than
calling ``app.ingestion.attach_and_extract`` directly, so the vision-model
default (``settings.copilot_vision_model``) and its name-based
vision-capability guard (``app.ollama_client.is_vision_capable_model``,
``settings.copilot_vision_model_capability_check``) are decided in exactly
ONE place shared with production, not re-derived here. A misconfigured
(non-vision) model raises ``VisionModelMisconfiguredError`` before any page
is sent to the model; this script catches that and exits 1 with a
diagnostic message rather than silently producing text-model output on
page images it cannot read.
"""

from __future__ import annotations

import sys

from app.config import get_settings
from app.ingestion import IngestionError, LocalIngestionStore
from app.ollama_client import OllamaClient
from app.supervisor import IngestSubTask, IntakeExtractorWorker, VisionModelMisconfiguredError


def _already_ingested(store: LocalIngestionStore, patient_id: int) -> bool:
    return any(citation.source_type == "lab_pdf" for citation in store.list_citations_for_patient(patient_id))


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <patient_id> <pdf_path>", file=sys.stderr)
        return 2

    patient_id = int(sys.argv[1])
    pdf_path = sys.argv[2]

    settings = get_settings()
    store = LocalIngestionStore(settings.copilot_ingestion_base_dir)

    if _already_ingested(store, patient_id):
        print(f"Demo lab-report PDF already ingested for patient_id={patient_id} (idempotent skip).")
        return 0

    vision_client = OllamaClient.from_settings(settings, model=settings.copilot_vision_model)
    worker = IntakeExtractorWorker(
        ollama_client=vision_client,
        document_store=store,
        fact_store=store,
        vision_model_capability_check=settings.copilot_vision_model_capability_check,
    )
    try:
        result = worker.run(IngestSubTask(patient_id=patient_id, file_path=pdf_path, doc_type="lab_pdf"))
    except VisionModelMisconfiguredError as exc:
        print(f"lab PDF ingestion refused for patient_id={patient_id}: {exc}", file=sys.stderr)
        return 1
    except IngestionError as exc:
        # Issue #206: attach_and_extract now raises when EVERY page failed
        # extraction, distinct from the partial-failure case below (which
        # still returns a result with a non-empty failed_pages).
        print(
            f"lab PDF ingestion FAILED ENTIRELY for patient_id={patient_id} "
            f"(pages_total={exc.pages_total}, failed_pages={exc.failed_pages}): {exc}",
            file=sys.stderr,
        )
        return 1
    if result.failed_pages:
        print(
            f"lab PDF ingestion had failed pages {result.failed_pages} for patient_id={patient_id}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Demo lab-report PDF ingested for patient_id={patient_id}. "
        f"facts={len(result.facts)} pages_total={result.pages_total}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
