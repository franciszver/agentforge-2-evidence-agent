"""Makes the copilot-agent service's ``app`` package importable from this eval.

Same rationale as ``evals/tool_selection/conftest.py``: the injection eval
imports the real ``Planner`` / ``OllamaClient`` / quarantine stack directly
rather than duplicating them, so the agent package's path is added to
``sys.path`` here instead of requiring a system-wide install.
"""

from __future__ import annotations

import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parents[2] / "services" / "copilot-agent"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))
