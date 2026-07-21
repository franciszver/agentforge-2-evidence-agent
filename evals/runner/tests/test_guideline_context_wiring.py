"""Red-first test for issue #105 at the eval-harness level (bp-stage2
follow-up from #85).

Root cause (see `app.planner`'s #105 comment for the full mechanism):
`runner.pipeline.run_case` -- mirroring the live `app.chat` wiring -- used to
hand a case's `retrieved_chunks` fixture to `run_verification` ONLY, after
`planner.run(case.question)` had already composed the final answer text.
The planner itself never saw the guideline text at all, so its answer used
its own general-knowledge category language (e.g. "elevated blood
pressure") instead of the guideline's own category name (e.g. "Stage 2
hypertension") -- a genuine, verbatim citation ended up attached to prose
that used the wrong category name for what it cited.

This test proves the case's retrieved-chunk text actually reaches the
planner's answer-composition (reasoning) call, not just post-hoc
verification -- using a scripted `OllamaLike` double (not `ReplayOllamaClient`,
which ignores message content entirely) that inspects exactly what the
free-text reasoning call receives.
"""

from __future__ import annotations

from pathlib import Path

from runner.loader import load_case
from runner.pipeline import run_case
from runner.tests.conftest import ReasoningCaptureOllamaClient

_FIXTURES = Path(__file__).parent / "fixtures"
_CASES = _FIXTURES / "cases"


def test_run_case_threads_retrieved_guideline_text_into_the_reasoning_call() -> None:
    case = load_case(_CASES / "semantic-support-pass.yaml")
    assert case.retrieved_chunks, "fixture must declare at least one retrieved chunk for this test to mean anything"
    client = ReasoningCaptureOllamaClient()

    run_case(case, client)

    assert client.chat_messages, "the planner never reached its reasoning call"
    guideline_text = case.retrieved_chunks[0].text
    reasoning_messages = client.chat_messages[-1]
    joined = " ".join(message["content"] for message in reasoning_messages)
    assert guideline_text in joined, (
        "issue #105: the case's retrieved guideline-chunk text must reach the "
        "planner's answer-composition (reasoning) call, not just post-hoc "
        "run_verification citation-attachment -- otherwise the answer's "
        "category language can drift from what it ends up citing"
    )
