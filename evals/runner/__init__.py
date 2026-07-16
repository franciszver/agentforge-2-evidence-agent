"""P4.7 eval harness: YAML case schema, record/replay, and the pytest runner.

See ``evals/runner/schema.py`` (case + assertion vocabulary),
``evals/runner/ollama_replay.py`` (record/replay layer),
``evals/runner/pipeline.py`` (runs a case through the real planner/
extraction/verification stack), and ``evals/runner/assertions.py``
(evaluates a case's assertions against the pipeline's result).
"""

from __future__ import annotations
