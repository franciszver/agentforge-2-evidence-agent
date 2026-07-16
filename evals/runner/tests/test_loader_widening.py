"""RED-first test for the P4.9 loader widening: ``discover_case_files`` must
accept multiple roots so the P4.7 runner (``evals/test_cases.py``) can
discover promoted regression cases under ``evals/regressions/`` in addition
to hand-authored cases under ``evals/cases/`` -- see
``docs/TEST_PLAN.md`` Sec 5: "P4.9 also wires that path into the P4.7
runner's case discovery, which today only scans ``evals/cases/``".

Uses throwaway ``tmp_path`` directories rather than the real
``evals/cases/``/``evals/regressions/`` trees -- this is a test of the
DISCOVERY MECHANISM, not of any real case content (mirrors
``evals/runner/tests/test_harness.py``'s use of its own isolated fixtures
rather than the real suite).
"""

from __future__ import annotations

from pathlib import Path

from runner.loader import discover_case_files


def test_single_root_still_works(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("id: a\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("id: b\n", encoding="utf-8")

    files = discover_case_files(tmp_path)

    assert [f.name for f in files] == ["a.yaml", "b.yaml"]


def test_multiple_roots_are_combined_and_sorted(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    regressions_dir = tmp_path / "regressions"
    cases_dir.mkdir()
    regressions_dir.mkdir()
    (cases_dir / "authored.yaml").write_text("id: authored\n", encoding="utf-8")
    (regressions_dir / "promoted.yaml").write_text("id: promoted\n", encoding="utf-8")

    files = discover_case_files(cases_dir, regressions_dir)

    assert {f.name for f in files} == {"authored.yaml", "promoted.yaml"}
    assert files == sorted(files)


def test_a_missing_root_is_tolerated_not_an_error(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "authored.yaml").write_text("id: authored\n", encoding="utf-8")
    missing_regressions_dir = tmp_path / "regressions"  # never created

    files = discover_case_files(cases_dir, missing_regressions_dir)

    assert [f.name for f in files] == ["authored.yaml"]


def test_the_real_suite_entry_point_scans_both_cases_and_regressions() -> None:
    """Proves the wiring, not just the mechanism: ``evals/test_cases.py``
    actually passes both real directories to ``discover_case_files``. Reads
    the source rather than importing the module -- ``evals/`` has no
    ``__init__.py`` (it is not meant to be imported as a package), so this
    avoids relying on pytest's incidental sys.path/rootdir behavior."""
    source = (Path(__file__).resolve().parents[2] / "test_cases.py").read_text(encoding="utf-8")
    assert "_REGRESSIONS_DIR" in source
    assert "discover_case_files(_CASES_DIR, _REGRESSIONS_DIR)" in source
