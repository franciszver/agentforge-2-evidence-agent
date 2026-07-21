"""Red-first tests for #140: live recording must fail loudly when the
in-container app code has drifted from the host working tree it's supposed
to be recording against.

Two layers, mirroring ``test_record_path_resolution.py``'s split:

* ``compute_app_stamp`` is a pure content-hash over an ``app/`` package
  directory's files -- directly unit-testable against synthetic directory
  trees (no real container needed).
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


def _write_app(root: Path, files: dict[str, str | bytes]) -> Path:
    app_dir = root / "app"
    app_dir.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = app_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    return app_dir


# --- compute_app_stamp: pure content hash over every file under app/, not
# just *.py -- stale behavioral assets under app/data/ (drug_interactions.db,
# reranker_scores*.json, retrieval_embeddings.json, ...) must drift the stamp
# exactly as a stale .py file would (gate review on #140/PR #143: the
# original *.py-only glob let those assets rot invisibly while the guard
# reported "code matches"). ------------------------------------------------


@pytest.mark.parametrize(
    ("files_a", "files_b", "expect_equal"),
    [
        pytest.param(
            {"__init__.py": "", "planner.py": "x = 1\n"},
            {"__init__.py": "", "planner.py": "x = 1\n"},
            True,
            id="identical-content",
        ),
        pytest.param(
            {"__init__.py": "", "planner.py": "x = 1\n"},
            {"__init__.py": "", "planner.py": "x = 2\n"},
            False,
            id="body-change",
        ),
        pytest.param(
            {"__init__.py": ""},
            {"__init__.py": "", "new_module.py": "y = 2\n"},
            False,
            id="file-added",
        ),
        pytest.param(
            {"__init__.py": "", "planner.py": "x = 1\n"},
            {
                "__init__.py": "",
                "planner.py": "x = 1\n",
                "__pycache__/planner.cpython-311.pyc": b"\x00\x01stale-bytecode",
            },
            True,
            id="pycache-ignored",
        ),
        pytest.param(
            {"__init__.py": "", "data/reranker_scores.json": '{"score": 1}'},
            {"__init__.py": "", "data/reranker_scores.json": '{"score": 2}'},
            False,
            id="data-asset-change-is-not-ignored",
        ),
    ],
)
def test_compute_app_stamp(
    tmp_path: Path,
    files_a: dict[str, str | bytes],
    files_b: dict[str, str | bytes],
    expect_equal: bool,
) -> None:
    app_a = _write_app(tmp_path / "a", files_a)
    app_b = _write_app(tmp_path / "b", files_b)

    stamp_a = compute_app_stamp(app_a)
    stamp_b = compute_app_stamp(app_b)

    if expect_equal:
        assert stamp_a == stamp_b
    else:
        assert stamp_a != stamp_b


def test_compute_app_stamp_ignores_compiled_pyc_files_outside_pycache(tmp_path: Path) -> None:
    """A stray ``.pyc`` sitting next to its source (not inside a
    ``__pycache__`` dir) is still a compiled build artifact, not source --
    excluded the same way."""
    app_a = _write_app(tmp_path / "a", {"__init__.py": "", "planner.py": "x = 1\n"})
    app_b = _write_app(
        tmp_path / "b",
        {
            "__init__.py": "",
            "planner.py": "x = 1\n",
            "planner.pyc": b"\x00\x01stale-bytecode",
        },
    )

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
