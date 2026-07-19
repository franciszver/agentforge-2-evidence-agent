"""Regenerate the pinned OpenAPI spec at ``openapi/openapi.json`` (P3G.3).

The FastAPI app's own generated schema (``app.main.app.openapi()``) is the
source of truth for the agent's HTTP contract -- this script just writes it
to a stable, diffable file so contract tests
(``tests/test_openapi_contract.py``) can detect drift and other tools/
consumers have something concrete to read.

Mutating maintenance command, same pattern as this repo's other
``scripts/generate_*`` / ``update-*-fixtures`` commands: review the diff
before committing.

Usage (from ``services/copilot-agent/``):
    python scripts/generate_openapi_spec.py
"""

from __future__ import annotations

import json
from pathlib import Path

from app.main import app

_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "openapi" / "openapi.json"


def main() -> None:
    spec = app.openapi()
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
