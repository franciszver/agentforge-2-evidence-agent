"""Hermetic tests for the JSON structured-log formatter (P4.1 follow-up, #144).

``app.correlation.configure_logging`` stamps every ``LogRecord`` with a
``correlation_id`` attribute (and call sites attach ``stage``/other fields via
``extra=``), but the default ``logging.Formatter`` installed there only
renders a fixed set of named fields into plain text -- ``extra=`` attributes
never show up in the actual log output. ``JsonFormatter`` fixes that: it
serializes the standard record fields PLUS any custom ``extra=`` attributes
to a single JSON line, and must not crash on records that carry none.
"""

from __future__ import annotations

import json
import logging

from app.correlation import JsonFormatter


def _make_record(
    *, msg: str = "hello", level: int = logging.INFO, extra: dict[str, object] | None = None
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="app.test_json_formatter",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in (extra or {}).items():
        setattr(record, key, value)
    return record


def test_formats_record_with_extra_fields_as_valid_json_containing_them():
    record = _make_record(
        msg="tool_call dispatched",
        level=logging.WARNING,
        extra={"correlation_id": "abc", "stage": "planner"},
    )

    output = JsonFormatter().format(record)
    payload = json.loads(output)

    assert payload["correlation_id"] == "abc"
    assert payload["stage"] == "planner"
    assert payload["message"] == "tool_call dispatched"
    assert payload["level"] == "WARNING"


def test_formats_plain_record_with_no_extras_without_crashing():
    record = _make_record(msg="plain library log line")

    output = JsonFormatter().format(record)
    payload = json.loads(output)

    assert payload["message"] == "plain library log line"
    assert payload["level"] == "INFO"
    # No custom extras were attached -- must not be present or must not error.
    assert "correlation_id" not in payload
