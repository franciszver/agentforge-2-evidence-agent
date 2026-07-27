"""Guards issue #196's fix: every declared dependency stays upper-bounded.

pyproject.toml used to declare version FLOORS ONLY (``fastapi>=0.115``, no
lockfile). Every environment -- each dev's host venv, every CI run, a
release-tag install -- could then resolve a different framework version.
#184 was a concrete instance: fastapi's own hardcoded OpenAPI schema
rendering changed between 0.124.4 and 0.139.2, and the byte-compared pinned
``openapi/openapi.json`` diverged purely on WHICH fastapi resolved.

This test does not re-check that #184's specific fastapi delta is handled
(``tests/test_openapi_contract.py`` already covers that byte-for-byte
comparison). It guards the regression this issue actually fixed: someone
adding a NEW floor-only dependency later, or stripping the ceiling off an
existing one, silently reintroducing the unbounded-resolution problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - project requires-python >=3.11
    import tomli as tomllib

PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"

# Specifier operators that establish an upper bound on the resolvable
# version. `~=` (compatible release) implies one too, but this project
# spells bounds out explicitly as `>=floor,<ceiling` (see pyproject.toml's
# issue #196 comment), so it isn't needed here -- kept anyway so a future
# `~=` entry doesn't false-positive as unbounded.
UPPER_BOUND_OPERATORS = {"<", "<=", "==", "~="}


def _load_dependency_specs() -> dict[str, str]:
    """Map each declared dependency's project name to its raw specifier string.

    Covers both ``[project.dependencies]`` and every
    ``[project.optional-dependencies]`` group (currently just ``dev``) --
    the production and dev/test surfaces both need to stay bounded, since
    CI's ``pip install -e ".[dev]"`` resolves both together.
    """
    with PYPROJECT_PATH.open("rb") as f:
        data = tomllib.load(f)

    project = data["project"]
    raw_specs: list[str] = list(project["dependencies"])
    for group_specs in project.get("optional-dependencies", {}).values():
        raw_specs.extend(group_specs)

    return {Requirement(raw).name: str(Requirement(raw).specifier) for raw in raw_specs}


def test_pyproject_declares_at_least_one_dependency() -> None:
    # Sanity check the parsing itself works, so a bug in _load_dependency_specs
    # can't masquerade as "every dependency is bounded" via an empty result.
    specs = _load_dependency_specs()
    assert len(specs) >= 10
    assert "fastapi" in specs
    assert "pydantic" in specs


def test_every_declared_dependency_has_an_upper_bound() -> None:
    specs = _load_dependency_specs()

    unbounded = [
        name
        for name, specifier in specs.items()
        if not any(spec.operator in UPPER_BOUND_OPERATORS for spec in SpecifierSet(specifier))
    ]

    assert not unbounded, (
        f"{unbounded} declare a floor with no upper bound in pyproject.toml -- "
        "this reopens issue #196 (every environment can resolve a different, "
        "untested version). Add a `<ceiling` (or `==`/`~=`) to each."
    )
