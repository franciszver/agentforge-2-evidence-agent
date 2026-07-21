"""Red-first tests for #140: ``evals/runner/record.py`` must refuse to
record when the app code it actually resolved and imported (see #119's
``_agent_root_candidates``) has drifted from what the recording protocol
expects -- ``EXPECTED_APP_STAMP``, computed host-side before docker-exec'ing
into ``development-easy-agent-1`` and passed in as an env var (docs/
TEST_PLAN.md Sec 9).

``runner.record`` computes its own local stamp once at import/startup time
over the SAME resolved app root ``_agent_root_candidates`` picked for
``sys.path`` -- so a mismatch here means the container's baked ``app/``
genuinely differs from the tree the operator computed ``EXPECTED_APP_STAMP``
from, not a false positive from checking the wrong directory.
"""

from __future__ import annotations

import pytest

from runner import record
from runner.code_stamp import CodeStampMismatchError, compute_app_stamp
from runner.record import _RESOLVED_APP_ROOT, verify_code_stamp


def test_resolved_app_root_is_the_live_imported_app_package() -> None:
    """Sanity check binding this test file to the real resolution #119
    already proved correct -- ``_RESOLVED_APP_ROOT`` must point at an actual
    ``app`` package directory (not e.g. accidentally the repo root)."""
    assert (_RESOLVED_APP_ROOT / "__init__.py").is_file()


def test_verify_code_stamp_noop_when_env_var_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXPECTED_APP_STAMP", raising=False)

    stamp = verify_code_stamp()  # must not raise

    assert stamp == compute_app_stamp(_RESOLVED_APP_ROOT)


def test_verify_code_stamp_passes_when_env_var_matches_actual_stamp(monkeypatch: pytest.MonkeyPatch) -> None:
    actual_stamp = compute_app_stamp(_RESOLVED_APP_ROOT)
    monkeypatch.setenv("EXPECTED_APP_STAMP", actual_stamp)

    stamp = verify_code_stamp()  # must not raise

    assert stamp == actual_stamp


def test_verify_code_stamp_refuses_on_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXPECTED_APP_STAMP", "deliberately-wrong-stamp-simulating-stale-container-code")

    with pytest.raises(CodeStampMismatchError):
        verify_code_stamp()


# --- fail CLOSED under the flattened container layout (#140 gate review) --
#
# The bug this closes: an operator who forgets `docker exec -e
# EXPECTED_APP_STAMP=...` got a silent no-op (fail-OPEN) -- exactly the
# silent-stale-recording failure mode #140 exists to prevent. Under the
# flattened in-container layout (#119's `_agent_root_candidates` resolving
# to the repo root rather than the monorepo `services/copilot-agent` root
# -- see ``record._in_flattened_container_layout``), an unset/empty
# EXPECTED_APP_STAMP must ABORT record mode, not skip the check. Only the
# genuine host-monorepo layout (no container, no drift possible) may still
# treat an unset env var as a no-op.


def test_verify_code_stamp_aborts_when_container_layout_and_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXPECTED_APP_STAMP", raising=False)
    monkeypatch.setattr(record, "_in_flattened_container_layout", lambda: True)

    with pytest.raises(record.MissingExpectedStampError) as exc_info:
        verify_code_stamp()

    message = str(exc_info.value)
    assert "EXPECTED_APP_STAMP" in message


def test_verify_code_stamp_aborts_when_container_layout_and_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty-string env var (e.g. ``-e EXPECTED_APP_STAMP=`` with the
    value accidentally dropped) is treated the same as unset -- not a
    falsy-but-present value that sneaks past the check."""
    monkeypatch.setenv("EXPECTED_APP_STAMP", "")
    monkeypatch.setattr(record, "_in_flattened_container_layout", lambda: True)

    with pytest.raises(record.MissingExpectedStampError):
        verify_code_stamp()


def test_verify_code_stamp_proceeds_when_container_layout_and_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail-closed only fires on a MISSING stamp -- a genuinely supplied one
    (even a mismatching one, covered above) still goes through the normal
    mismatch/match comparison, not an unconditional abort."""
    actual_stamp = compute_app_stamp(_RESOLVED_APP_ROOT)
    monkeypatch.setenv("EXPECTED_APP_STAMP", actual_stamp)
    monkeypatch.setattr(record, "_in_flattened_container_layout", lambda: True)

    stamp = verify_code_stamp()  # must not raise

    assert stamp == actual_stamp


def test_verify_code_stamp_noop_when_host_layout_and_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The genuine host-monorepo layout (no container in the loop) keeps the
    old no-op-on-unset behavior -- there is nothing to drift from when the
    live code trivially IS the working tree."""
    monkeypatch.delenv("EXPECTED_APP_STAMP", raising=False)
    monkeypatch.setattr(record, "_in_flattened_container_layout", lambda: False)

    stamp = verify_code_stamp()  # must not raise

    assert stamp == compute_app_stamp(_RESOLVED_APP_ROOT)
