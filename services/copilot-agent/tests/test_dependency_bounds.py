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
existing one, silently reintroducing the unbounded-resolution problem -- and
someone declaring a ceiling that doesn't actually match what's installed and
tested (too tight to install, or a contradictory/empty range).
"""

from __future__ import annotations

import sys
from importlib.metadata import version as installed_version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - project requires-python >=3.11
    import tomli as tomllib

PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"

# Specifier operators that establish an upper bound on the resolvable
# version. `~=` (compatible release) implies one too, and `===` (arbitrary
# equality, a legitimate hard pin) is fully bounded by definition -- both
# are kept here so a future entry using either doesn't false-positive as
# unbounded. This project currently spells bounds out explicitly as
# `>=floor,<ceiling` (see pyproject.toml's issue #196 comment).
UPPER_BOUND_OPERATORS = {"<", "<=", "==", "===", "~="}


def _load_dependency_specs() -> list[tuple[str, str]]:
    """List each declared dependency's project name and raw specifier string.

    Covers both ``[project.dependencies]`` and every
    ``[project.optional-dependencies]`` group (currently just ``dev``) --
    the production and dev/test surfaces both need to stay bounded, since
    CI's ``pip install -e ".[dev]"`` resolves both together.

    Returns a list, not a dict keyed by name: a package re-declared across
    a base list and an extras group (or across two extras groups) must
    produce two independently-checked entries. A dict would let a bounded
    second declaration silently overwrite -- and mask -- an unbounded first
    one keyed under the same name.
    """
    with PYPROJECT_PATH.open("rb") as f:
        data = tomllib.load(f)

    project = data["project"]
    raw_specs: list[str] = list(project["dependencies"])
    for group_specs in project.get("optional-dependencies", {}).values():
        raw_specs.extend(group_specs)

    return [(Requirement(raw).name, str(Requirement(raw).specifier)) for raw in raw_specs]


def test_pyproject_declares_at_least_one_dependency() -> None:
    # Sanity check the parsing itself works, so a bug in _load_dependency_specs
    # can't masquerade as "every dependency is bounded" via an empty result.
    specs = _load_dependency_specs()
    names = {name for name, _ in specs}
    assert len(specs) >= 10
    assert "fastapi" in names
    assert "pydantic" in names


def test_every_declared_dependency_has_an_upper_bound() -> None:
    specs = _load_dependency_specs()

    unbounded = [
        name
        for name, specifier in specs
        if not any(spec.operator in UPPER_BOUND_OPERATORS for spec in SpecifierSet(specifier))
    ]

    assert not unbounded, (
        f"{unbounded} declare a floor with no upper bound in pyproject.toml -- "
        "this reopens issue #196 (every environment can resolve a different, "
        "untested version). Add a `<ceiling` (or `==`/`===`/`~=`) to each."
    )


def test_installed_versions_satisfy_declared_specifiers() -> None:
    """Every installed version must actually satisfy its own declared range.

    A ceiling below the installed/tested version (e.g. `fastapi<0.116` when
    0.139.2 is what's verified and installed) or a contradictory/empty range
    (e.g. `>=0.139,<0.115`) would both still pass the upper-bound-exists
    check above while describing a range that either can't install at all
    or doesn't match what the test suite actually ran against -- exactly
    the drift #196 exists to prevent. This is the property #196 wants:
    not just "a ceiling exists", but "the declared range matches reality".
    """
    specs = _load_dependency_specs()

    mismatched = [
        f"{name}{specifier} does not admit installed version {installed}"
        for name, specifier in specs
        for installed in [installed_version(name)]
        if not SpecifierSet(specifier).contains(installed)
    ]

    assert not mismatched, (
        f"{mismatched} -- the declared specifier in pyproject.toml doesn't match "
        "the version actually installed/tested. Either the ceiling/floor is wrong "
        "or the wrong version got installed; both reopen #196."
    )
