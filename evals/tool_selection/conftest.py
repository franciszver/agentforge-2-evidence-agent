"""Makes the copilot-agent service's ``app`` package importable from evals.

``services/copilot-agent`` is a separate Python package (its own
``pyproject.toml``) from this evals suite. The tool-selection eval imports
the real ``Planner``/``OllamaClient``/``OpenEmrClient`` directly rather than
duplicating them, so its path is added to ``sys.path`` here rather than
requiring the agent package to be installed system-wide.
"""

from __future__ import annotations

import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parents[2] / "services" / "copilot-agent"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))
