"""Record/replay layer (P4.7, ``docs/TEST_PLAN.md`` Sec 9): the ONLY
non-deterministic external in the eval pipeline is the model (Ollama chat +
constrained extraction). Tool data is already deterministic -- canned per
case (see ``runner.tool_stub``) -- so recording/replaying just the ordered
sequence of Ollama responses is enough to make an entire case
(planner -> optional extraction -> verification) byte-for-byte reproducible
offline.

**Record mode** (local, opt-in, needs the live model): ``RecordingOllamaClient``
wraps a real ``OllamaClient`` and transparently forwards every ``chat``/
``extract`` call, appending each call's kind + response to an ordered list.
After a case finishes running, that list is the artifact -- write it with
``save_recording`` to a COMMITTED ``evals/recordings/<id>.json`` file.

**Replay mode** (default, CI, fully offline): ``ReplayOllamaClient`` is
constructed from a loaded recording and satisfies the same ``chat``/
``extract`` duck-typed interface the real ``OllamaClient`` does (matching the
seam ``app.planner.Planner`` and ``app.extraction.ClaimExtractor`` already
accept for hermetic tests). Each call pops the next recorded call in order
and returns its response -- no HTTP, no Ollama, nothing but list indexing and
``schema.model_validate``. A call whose kind/schema doesn't match what was
recorded next (the pipeline's behavior diverged from the recording -- e.g. a
tool_data edit changed which tool the deterministic-registry path takes) or
that runs past the end of the recording raises a CLEAR error rather than
silently returning stale/wrong data, so a broken recording never masquerades
as a pass.

**Missing recording (decided default): FAIL, not skip.** A case file that
exists but has no recording artifact is a bug in the suite (a case was
authored/edited without running it live and committing the result) -- ``docs/
TEST_PLAN.md`` Sec 9's whole point is that "a broken checker, contract, or
case still fails the PR without any inference." Skipping would let a
recording silently rot out of sync with its case; ``load_recording`` raises
:class:`RecordingNotFoundError`, which the runner surfaces as a test failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from app.ollama_client import OllamaClient


class OllamaLike(Protocol):
    """The duck-typed interface the planner/extraction pipeline needs from
    an Ollama client -- satisfied by the real ``OllamaClient``,
    ``RecordingOllamaClient``, and ``ReplayOllamaClient`` alike."""

    def chat(self, messages: Any, *, options: Any = None) -> str: ...

    def extract(self, prompt_or_messages: Any, schema: type[BaseModel], *, options: Any = None) -> Any: ...


@dataclass(frozen=True)
class RecordedCall:
    """One recorded model call, in the order it happened.

    ``schema`` is the extraction schema's class name for an ``"extract"``
    call, ``None`` for a ``"chat"`` call. ``response`` is the assembled chat
    string, or the extracted model's ``model_dump(mode="json")``.
    """

    kind: str  # "chat" | "extract"
    schema: str | None
    response: Any

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "schema": self.schema, "response": self.response}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RecordedCall:
        return cls(kind=data["kind"], schema=data.get("schema"), response=data["response"])


class RecordingNotFoundError(Exception):
    """Raised in replay mode when a case has no recording artifact -- see
    module docstring, "Missing recording"."""


class RecordingExhaustedError(Exception):
    """Raised in replay when the pipeline made more model calls than the
    recording has -- the case's behavior has diverged from the recording."""


class RecordingMismatchError(Exception):
    """Raised in replay when a call's kind/schema doesn't match what was
    recorded next in sequence -- the case's behavior has diverged from the
    recording."""


def recording_path(recordings_dir: Path, case_id: str) -> Path:
    return recordings_dir / f"{case_id}.json"


def save_recording(path: Path, calls: list[RecordedCall]) -> None:
    """Write the recording artifact. Creates parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"calls": [call.to_json() for call in calls]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_recording(path: Path) -> list[RecordedCall]:
    if not path.exists():
        raise RecordingNotFoundError(
            f"no recording at {path} -- this case has no committed model-output artifact. "
            "Run it in record mode locally against the live model and commit the result "
            "(docs/TEST_PLAN.md Sec 9); a case is never silently skipped for a missing recording."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return [RecordedCall.from_json(call) for call in data["calls"]]


class RecordingOllamaClient:
    """Wraps a real ``OllamaClient``; forwards every call and records it in
    order. RECORD mode only -- needs the live model."""

    def __init__(self, inner: OllamaClient) -> None:
        self._inner = inner
        self.calls: list[RecordedCall] = []

    def chat(self, messages: Any, *, options: Any = None) -> str:
        response = self._inner.chat(messages, options=options)
        self.calls.append(RecordedCall(kind="chat", schema=None, response=response))
        return response

    def extract(self, prompt_or_messages: Any, schema: type[BaseModel], *, options: Any = None) -> Any:
        result = self._inner.extract(prompt_or_messages, schema, options=options)
        self.calls.append(
            RecordedCall(kind="extract", schema=schema.__name__, response=result.model_dump(mode="json"))
        )
        return result


class ReplayOllamaClient:
    """Offline stand-in for ``OllamaClient``: replays a recorded call
    sequence in order. NO network, NO Ollama -- pure list playback plus
    ``schema.model_validate``. Default/CI mode."""

    def __init__(self, calls: list[RecordedCall]) -> None:
        self._calls = list(calls)
        self._index = 0

    def _next(self, kind: str, schema_name: str | None) -> RecordedCall:
        if self._index >= len(self._calls):
            raise RecordingExhaustedError(
                f"recording exhausted after {self._index} call(s) -- the pipeline requested another "
                f"{kind} call (schema={schema_name}) that the recording doesn't have; the case's "
                "behavior has diverged from what was recorded -- re-record it"
            )
        call = self._calls[self._index]
        self._index += 1
        if call.kind != kind or call.schema != schema_name:
            raise RecordingMismatchError(
                f"recording mismatch at call {self._index}: recorded kind={call.kind!r} "
                f"schema={call.schema!r}, pipeline requested kind={kind!r} schema={schema_name!r} -- "
                "the case's behavior has diverged from what was recorded -- re-record it"
            )
        return call

    def chat(self, messages: Any, *, options: Any = None) -> str:
        call = self._next("chat", None)
        return call.response

    def extract(self, prompt_or_messages: Any, schema: type[BaseModel], *, options: Any = None) -> Any:
        call = self._next("extract", schema.__name__)
        return schema.model_validate(call.response)
