"""Makes the copilot-agent service's ``app`` package importable from every
eval under this directory (harness code + case runner alike).

Same rationale as ``evals/tool_selection/conftest.py`` and its siblings:
``services/copilot-agent`` is a separate Python package (its own
``pyproject.toml``), so its path is added to ``sys.path`` here rather than
requiring the agent package to be installed system-wide. A single root-level
conftest covers ``evals/runner/`` and the YAML case runner (``evals/test_cases.py``)
without duplicating the snippet per subdirectory.
"""

from __future__ import annotations

import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parents[1] / "services" / "copilot-agent"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))
