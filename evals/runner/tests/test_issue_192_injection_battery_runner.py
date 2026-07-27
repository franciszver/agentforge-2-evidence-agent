"""Regression guard for issue #192 gate-1 finding 1.

``evals/runner/issue_192_injection_battery.py`` used to reconstruct
``app.semantic_support.judge_support``'s message shape itself (importing its
private ``_SYSTEM_PROMPT``/``_INSTRUCTIONS_TEMPLATE``/``_CONTEXT_BLOCK_TEMPLATE``
and re-assembling ``messages``) rather than calling the production code path
-- so if that module's message assembly ever changed shape, this battery
would silently keep attacking the OLD shape, reporting resistance for a
prompt that no longer ships. The fix: the runner's ``_judge_full`` now calls
``app.semantic_support.judge_support_full`` directly, the same public seam
``judge_support`` (the actual production entry point) delegates to.

This test proves the runner's call path and the production call path are
byte-identical for the same inputs -- the guard against a future refactor
reintroducing a divergent second copy of the message assembly (e.g. if
``judge_support`` were changed to build its ``messages`` independently of
``judge_support_full`` again, this test would catch the drift; see the PR
report for the mutation proof)."""

from __future__ import annotations

import json
from typing import Any

import pytest

import runner.issue_192_injection_battery as battery
from app.semantic_support import SemanticSupportJudgement, SupportVerdict, judge_support
from runner.issue_192_injection_battery import JudgeName, _judge_full


class _MessageCapturingJudge:
    """Records the exact ``messages`` list handed to ``.extract`` (never a
    live call -- always returns the same scripted judgement)."""

    def __init__(self, response: SemanticSupportJudgement) -> None:
        self._response = response
        self.messages_seen: list[list[dict[str, str]]] = []

    def extract(self, prompt_or_messages: Any, schema: type, *, options: Any = None) -> Any:
        assert schema is SemanticSupportJudgement
        assert isinstance(prompt_or_messages, list)
        self.messages_seen.append(prompt_or_messages)
        return self._response


def test_runner_semantic_support_path_matches_production_judge_support_messages() -> None:
    claim_text = "The patient's LDL cholesterol was 165 mg/dL, above the target range."
    quote = "Lipid panel results: LDL cholesterol 165 mg/dL. Target LDL below 100 mg/dL."
    response = SemanticSupportJudgement(verdict=SupportVerdict.SUPPORTED, reason="matches")

    production_judge = _MessageCapturingJudge(response)
    judge_support(claim_text, quote, production_judge)

    runner_judge = _MessageCapturingJudge(response)
    _judge_full(runner_judge, JudgeName.SEMANTIC_SUPPORT, claim_text, quote)

    assert production_judge.messages_seen == runner_judge.messages_seen
    assert len(production_judge.messages_seen[0]) == 2  # system + user


# ---------------------------------------------------------------------------
# Regression guard for issue #192 gate-3 MAJOR-1: a re-run must not be able
# to clobber an existing labelled run's draws/summary.json.
# ---------------------------------------------------------------------------


def test_rerun_with_same_label_refuses_to_overwrite_existing_draws(tmp_path, monkeypatch) -> None:
    """A prior run's draws are evidence -- a second invocation naming the same
    ``--label`` must refuse outright (exit nonzero) rather than silently
    re-running live draws over them. This is the exact failure mode MAJOR-1
    named: identical draw filenames for any run, so an unguarded re-run would
    overwrite every prior draw and the aggregated summary."""
    monkeypatch.setattr(battery, "_RUNS_DIR", tmp_path / "runs")

    existing_run_dir = battery._run_dir("my-run")
    existing_draws_dir = existing_run_dir / "draws"
    existing_draws_dir.mkdir(parents=True)
    sentinel_draw = existing_draws_dir / "semantic_support-force_supported-CONTROL-draw0.json"
    sentinel_draw.write_text(json.dumps({"verdict": "not_supported", "sentinel": True}), encoding="utf-8")

    with pytest.raises(SystemExit):
        battery.main(["--label", "my-run", "--draws", "1"])

    # The sentinel draw must be untouched -- proves the refusal happened
    # BEFORE any live call or write, not merely that the process exited
    # after clobbering something.
    assert json.loads(sentinel_draw.read_text(encoding="utf-8")) == {"verdict": "not_supported", "sentinel": True}
    assert list(existing_draws_dir.glob("*.json")) == [sentinel_draw]


def test_summarize_only_refuses_when_labelled_run_has_no_recorded_draws(tmp_path, monkeypatch) -> None:
    """``--summarize-only`` re-aggregates an EXISTING run's own draws -- it
    must refuse (not silently produce an empty summary.json) when the named
    run was never actually recorded, rather than writing a misleadingly
    "complete" empty summary."""
    monkeypatch.setattr(battery, "_RUNS_DIR", tmp_path / "runs")

    with pytest.raises(SystemExit):
        battery.main(["--label", "never-recorded", "--summarize-only"])

    assert not (battery._run_dir("never-recorded") / "summary.json").exists()


def test_summarize_only_on_an_existing_run_only_touches_that_runs_own_summary(tmp_path, monkeypatch) -> None:
    """``--summarize-only`` against a real prior run re-aggregates ONLY that
    run's own ``draws/`` into ITS OWN ``summary.json`` -- proves the labelled
    layout keeps runs from stepping on each other, and never touches the
    historical top-level ``draws/``/``summary.json`` or ``phase1-before/``/
    ``claim-channel-before/`` (which do not exist under the patched
    ``_RUNS_DIR`` at all, so any accidental write there would raise, not
    silently succeed)."""
    monkeypatch.setattr(battery, "_RUNS_DIR", tmp_path / "runs")

    run_dir = battery._run_dir("my-run")
    draws_dir = run_dir / "draws"
    draws_dir.mkdir(parents=True)
    for judge_name in JudgeName:
        for direction in battery.Direction:
            # Control verdict must differ from its OWN direction's attempted
            # word (module docstring, "control-verdict sanity check") --
            # otherwise summarize() raises rather than silently reporting a
            # zeroed-out cell.
            control_verdict = "not_supported" if direction is battery.Direction.FORCE_SUPPORTED else "supported"
            record = battery.DrawRecord(
                subject_id=f"{judge_name.value}-{direction.value}-CONTROL",
                judge=judge_name.value,
                direction=direction.value,
                technique="CONTROL",
                draw_index=0,
                verdict=control_verdict,
                reason="scripted",
            )
            battery.save_draw(record, draws_dir)

    battery.main(["--label", "my-run", "--summarize-only"])

    summary_path = run_dir / "summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_payloads"] == 0  # only CONTROL draws were recorded above
    # Re-running --summarize-only again on the SAME label must succeed (it is
    # re-aggregating that run's own draws, not clobbering another run's) --
    # only a fresh *live* run into an existing label is refused.
    battery.main(["--label", "my-run", "--summarize-only"])
