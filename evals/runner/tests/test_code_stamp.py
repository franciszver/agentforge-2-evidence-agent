"""Red-first tests for #140: live recording must fail loudly when the
in-container app code has drifted from the host working tree it's supposed
to be recording against.

Two layers, mirroring ``test_record_path_resolution.py``'s split:

* ``compute_app_stamp`` is a pure content-hash over an ``app/`` package
  directory's ``*.py`` files -- directly unit-testable against synthetic
  directory trees (no real container needed).
* ``check_code_stamp`` is the fail-loud gate ``runner.record`` calls before
  driving any case through the live model: mismatch refuses with both
  stamps + remediation named; match proceeds silently; ``expected_stamp is
  None`` (the check wasn't requested -- e.g. recording directly on host, no
  container in the loop) is a no-op, not a failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runner.code_stamp import CodeStampMismatchError, check_code_stamp, compute_app_stamp


def _write_app(root: Path, files: dict[str, str]) -> Path:
    app_dir = root / "app"
    app_dir.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = app_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return app_dir


# --- compute_app_stamp: pure content hash ---------------------------------


def test_compute_app_stamp_is_deterministic_for_identical_content(tmp_path: Path) -> None:
    app_a = _write_app(tmp_path / "a", {"__init__.py": "", "planner.py": "x = 1\n"})
    app_b = _write_app(tmp_path / "b", {"__init__.py": "", "planner.py": "x = 1\n"})

    assert compute_app_stamp(app_a) == compute_app_stamp(app_b)


def test_compute_app_stamp_changes_when_a_file_body_changes(tmp_path: Path) -> None:
    app_a = _write_app(tmp_path / "a", {"__init__.py": "", "planner.py": "x = 1\n"})
    app_b = _write_app(tmp_path / "b", {"__init__.py": "", "planner.py": "x = 2\n"})

    assert compute_app_stamp(app_a) != compute_app_stamp(app_b)


def test_compute_app_stamp_changes_when_a_file_is_added(tmp_path: Path) -> None:
    app_a = _write_app(tmp_path / "a", {"__init__.py": ""})
    app_b = _write_app(tmp_path / "b", {"__init__.py": "", "new_module.py": "y = 2\n"})

    assert compute_app_stamp(app_a) != compute_app_stamp(app_b)


def test_compute_app_stamp_ignores_pycache(tmp_path: Path) -> None:
    app_a = _write_app(tmp_path / "a", {"__init__.py": "", "planner.py": "x = 1\n"})
    app_b = _write_app(tmp_path / "b", {"__init__.py": "", "planner.py": "x = 1\n"})
    (app_b / "__pycache__").mkdir()
    (app_b / "__pycache__" / "planner.cpython-311.pyc").write_bytes(b"\x00\x01stale-bytecode")

    assert compute_app_stamp(app_a) == compute_app_stamp(app_b)


# --- check_code_stamp: fail loudly on mismatch, no-op when not requested --


def test_check_code_stamp_noop_when_expected_is_none() -> None:
    check_code_stamp("abc123", None)  # must not raise


def test_check_code_stamp_noop_when_stamps_match() -> None:
    check_code_stamp("abc123", "abc123")  # must not raise


def test_check_code_stamp_raises_on_mismatch_naming_both_stamps_and_remediation() -> None:
    with pytest.raises(CodeStampMismatchError) as exc_info:
        check_code_stamp("local-deadbeef", "expected-cafef00d")

    message = str(exc_info.value)
    assert "local-deadbeef" in message
    assert "expected-cafef00d" in message
    # Remediation must be actionable, not just "it's wrong".
    assert "docker cp" in message or "rebuild" in message.lower()
