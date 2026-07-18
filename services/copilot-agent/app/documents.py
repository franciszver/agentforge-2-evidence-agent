"""``GET /documents/{source_id}``: serves a stored source document's raw
bytes (P3.7 citation overlay).

Honest scope. P3.1/P3.2 ingestion citations (``app.schemas.ingestion.Citation``/
``DocumentCitation``) are page-level -- ``{source_type, source_id,
page_or_section, field_or_chunk_id, quote_or_value}`` -- with NO pixel
bounding box. A P3.7 capability probe (qwen2.5vl:7b grounding against the
committed lab fixture, both a clean render and a scan-simulated degraded
render) found bbox coordinates accurate on the clean render but drifting
onto the wrong table column/row under scan-realistic noise + rotation --
unreliable enough that drawing a box from it would risk pointing at the
WRONG value on a real scanned document, violating the project's
no-fabrication thesis. This endpoint backs the honest fallback instead: the
module UI opens the real source PDF at the cited page (``#page=N``) and
shows the citation's own literal quote text alongside it -- a true page-level
"here is the source" rather than a fabricated pixel box. True bbox grounding
is tracked as a separate follow-up issue.

**Access control.** ``source_id`` is never treated as a path -- it must
match ``LocalIngestionStore``'s own ``uuid4().hex`` naming (32 lowercase hex
chars) or the request is rejected with 400 before any filesystem lookup, so
a ``"../../etc/passwd"``-shaped value can never reach the filesystem (see
``LocalIngestionStore.read_source_document``'s own independent re-check).
Gated by the same bearer-token seam as every other agent endpoint
(``app.chat``'s ``TokenValidator``) -- an unauthenticated caller cannot fetch
source documents.

**No PHI in logs.** Only ``source_id`` (an opaque, server-generated
identifier -- not a name/DOB/MRN) and outcome are logged, never file
contents.
"""

from __future__ import annotations

import logging
import re

from fastapi import Depends, Header, HTTPException, Response

from app.chat import (
    TokenValidationError,
    TokenValidator,
    extract_bearer_token,
    get_token_validator,
)
from app.config import Settings, get_settings
from app.ingestion import LocalIngestionStore

_logger = logging.getLogger(__name__)

# LocalIngestionStore.save_source_document names every stored document
# uuid4().hex -- exactly 32 lowercase hex chars, nothing else. Anything not
# matching this is rejected outright (see module docstring).
_SOURCE_ID_RE = re.compile(r"[0-9a-f]{32}")


def get_document_store(settings: Settings = Depends(get_settings)) -> LocalIngestionStore:
    """FastAPI dependency: the active document store. Override in tests."""
    return LocalIngestionStore(settings.copilot_ingestion_base_dir)


def source_document_endpoint(
    source_id: str,
    authorization: str | None = Header(default=None),
    validator: TokenValidator = Depends(get_token_validator),
    store: LocalIngestionStore = Depends(get_document_store),
) -> Response:
    try:
        token = extract_bearer_token(authorization)
        validator(token)
    except TokenValidationError as exc:
        raise HTTPException(status_code=401, detail="invalid or missing token") from exc

    if not _SOURCE_ID_RE.fullmatch(source_id):
        raise HTTPException(status_code=400, detail="invalid source_id")

    content = store.read_source_document(source_id)
    if content is None:
        _logger.warning("source document not found", extra={"source_id": source_id})
        raise HTTPException(status_code=404, detail="document not found")

    return Response(content=content, media_type="application/pdf")
