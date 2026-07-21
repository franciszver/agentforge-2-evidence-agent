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
            # newline="" disables Python text-mode newline translation --
            # without it, on Windows a "\n" inside a str being written is
            # silently rewritten to os.linesep ("\r\n"), corrupting the
            # exact-bytes-on-disk fixtures the CRLF/LF normalization tests
            # below depend on (a content string already containing "\r\n"
            # would come out as "\r\r\n"). Fixtures must control their own
            # on-disk bytes precisely, independent of host platform.
            path.write_text(content, encoding="utf-8", newline="")
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


# --- CRLF/LF line-ending normalization (gate review on #143): the Windows ---
# host checkout is CRLF (autocrlf), the Linux container bakes the LF git
# blob -- a byte-exact hash of a TEXT file makes EVERY recording abort with
# CodeStampMismatchError even when the two trees are the identical logical
# source. Binary assets must NOT be normalized -- a real \r\n byte pair in a
# binary file is data, not a line ending, and collapsing it would hide
# genuine drift in exactly the assets (drug_interactions.db) the widened
# coverage exists to protect. -------------------------------------------


def test_compute_app_stamp_same_for_crlf_and_lf_text_file(tmp_path: Path) -> None:
    app_a = _write_app(tmp_path / "a", {"__init__.py": "", "planner.py": "x = 1\ny = 2\n"})
    app_b = _write_app(tmp_path / "b", {"__init__.py": "", "planner.py": "x = 1\r\ny = 2\r\n"})

    assert compute_app_stamp(app_a) == compute_app_stamp(app_b)


def test_compute_app_stamp_same_for_lone_cr_text_file(tmp_path: Path) -> None:
    """Old-Mac-style lone ``\\r`` line endings normalize to ``\\n`` too, not
    just the Windows ``\\r\\n`` pair."""
    app_a = _write_app(tmp_path / "a", {"__init__.py": "", "planner.py": "x = 1\ny = 2\n"})
    app_b = _write_app(tmp_path / "b", {"__init__.py": "", "planner.py": "x = 1\ry = 2\r"})

    assert compute_app_stamp(app_a) == compute_app_stamp(app_b)


def test_compute_app_stamp_still_detects_real_text_content_drift(tmp_path: Path) -> None:
    """Normalization must not make the guard blind to genuine source
    drift -- only line-ending differences collapse, not body changes."""
    app_a = _write_app(tmp_path / "a", {"__init__.py": "", "planner.py": "x = 1\r\n"})
    app_b = _write_app(tmp_path / "b", {"__init__.py": "", "planner.py": "x = 2\r\n"})

    assert compute_app_stamp(app_a) != compute_app_stamp(app_b)


def test_compute_app_stamp_does_not_normalize_binary_file_line_ending_bytes(tmp_path: Path) -> None:
    """A binary file (contains a NUL byte, e.g. a sqlite ``.db``) that
    happens to contain ``\\r\\n`` byte pairs must hash differently from the
    same file with those bytes collapsed to ``\\n`` -- those bytes are
    opaque binary data, not line endings, and normalizing them would mask
    real drift in a behavioral asset like ``drug_interactions.db``."""
    app_a = _write_app(tmp_path / "a", {"data/drug_interactions.db": b"\x00binary\r\npayload"})
    app_b = _write_app(tmp_path / "b", {"data/drug_interactions.db": b"\x00binary\npayload"})

    assert compute_app_stamp(app_a) != compute_app_stamp(app_b)


def test_compute_app_stamp_binary_file_identical_bytes_still_matches(tmp_path: Path) -> None:
    app_a = _write_app(tmp_path / "a", {"data/drug_interactions.db": b"\x00binary\r\npayload"})
    app_b = _write_app(tmp_path / "b", {"data/drug_interactions.db": b"\x00binary\r\npayload"})

    assert compute_app_stamp(app_a) == compute_app_stamp(app_b)


# --- runtime-mutable data files excluded from the stamp (gate review on ----
# #143): app/data/eval_history.json is REWRITTEN at runtime by
# app.dashboard_eval_history.append_eval_run (called from
# evals/runner/record_run.py after every eval run), so in a long-lived
# container it drifts from its committed state for reasons that have
# nothing to do with source code drift -- the stamp guard must not treat
# that as a mismatch. Other data/ assets the app only ever reads
# (reranker_scores.json etc.) remain real drift surface. ------------------


def test_compute_app_stamp_ignores_eval_history_json_changes(tmp_path: Path) -> None:
    app_a = _write_app(tmp_path / "a", {"__init__.py": "", "data/eval_history.json": "[]"})
    app_b = _write_app(
        tmp_path / "b",
        {"__init__.py": "", "data/eval_history.json": '[{"timestamp": "2026-01-01", "total": 1}]'},
    )

    assert compute_app_stamp(app_a) == compute_app_stamp(app_b)


def test_compute_app_stamp_still_detects_reranker_scores_changes(tmp_path: Path) -> None:
    """Sanity check that the eval-history exclusion is narrow -- a sibling
    ``data/`` asset the app only reads (never rewrites) still drifts the
    stamp as before."""
    app_a = _write_app(tmp_path / "a", {"__init__.py": "", "data/reranker_scores.json": '{"score": 1}'})
    app_b = _write_app(tmp_path / "b", {"__init__.py": "", "data/reranker_scores.json": '{"score": 2}'})

    assert compute_app_stamp(app_a) != compute_app_stamp(app_b)


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
