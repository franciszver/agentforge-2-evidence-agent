"""Issue #81: pins the default posture of the issue #47 semantic-support gate
and its ``/chat`` provider seam (``app.chat.get_support_judge_provider``).

No existing test covered this dependency directly -- ``tests/test_semantic_support.py``
only exercises ``app.semantic_support`` in isolation (a scripted judge double,
no ``Settings``/``app.chat`` involvement at all). This file closes that gap for
the flag flip: default ON (production), and the provider still degrades to a
no-op when a caller explicitly turns the flag off.
"""

from __future__ import annotations

from app.chat import _no_op_support_judge_provider, get_support_judge_provider
from app.config import Settings
from app.llama_server_client import LlamaServerClient


def test_semantic_support_gate_defaults_to_enabled():
    assert Settings().copilot_semantic_support_enabled is True


def test_provider_builds_a_real_judge_client_when_the_flag_is_on():
    settings = Settings(copilot_semantic_support_enabled=True)

    provider = get_support_judge_provider(settings=settings)

    assert provider is not _no_op_support_judge_provider
    judge = provider()
    assert isinstance(judge, LlamaServerClient)


def test_provider_is_the_no_op_when_the_flag_is_off():
    settings = Settings(copilot_semantic_support_enabled=False)

    provider = get_support_judge_provider(settings=settings)

    assert provider is _no_op_support_judge_provider
    assert provider() is None
