"""Discovers and parses YAML eval case files (P4.7)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from runner.schema import EvalCase, EvalCaseError


def discover_case_files(root: Path) -> list[Path]:
    """Every ``*.yaml`` file under ``root``, sorted for deterministic order."""
    return sorted(root.rglob("*.yaml"))


def load_case(path: Path) -> EvalCase:
    """Parse and schema-validate one case file.

    Raises :class:`EvalCaseError` with a message naming the file for any
    failure -- invalid YAML syntax, a non-mapping document, or a document
    that fails :class:`EvalCase` schema validation (including an unknown
    ``tool_data`` shape, an unknown assertion ``type``, or a missing
    required field). This is the "a malformed case fails clearly" contract.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalCaseError(f"{path}: could not read file: {exc}") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise EvalCaseError(f"{path}: malformed YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise EvalCaseError(f"{path}: case file must contain a YAML mapping at the top level")

    try:
        return EvalCase.model_validate(raw)
    except ValidationError as exc:
        raise EvalCaseError(f"{path}: case schema validation failed: {exc}") from exc
