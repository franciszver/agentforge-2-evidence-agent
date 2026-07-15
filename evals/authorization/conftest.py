"""Makes the copilot-agent service's ``app`` package importable from this eval.

Same rationale as ``evals/injection/conftest.py``: the authorization eval
imports the real ``Planner`` / ``OllamaClient`` stack directly rather than
duplicating it, so the agent package's path is added to ``sys.path`` here
instead of requiring a system-wide install.
"""

from __future__ import annotations

import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parents[2] / "services" / "copilot-agent"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))
