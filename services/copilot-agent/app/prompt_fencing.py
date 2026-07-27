"""Nonce-fenced untrusted-content envelopes (issue #192).

Shared by ``app.semantic_support`` and ``app.source_ref_relevance``: both
modules interpolate untrusted, model/patient-controlled text (claim prose,
quotes, chart-derived facts) directly into an LLM judge prompt. Phase 1's
injection battery (``evals/results/issue-192/``) measured 8 payloads that
reliably flipped a judge verdict via prompt-STRUCTURE mimicry -- forging a
fake system-role turn, reproducing/closing the prompt's own section
headers, or emitting a fake JSON-schema block the judge's constrained
decoder then honored literally.

**The envelope.** Each untrusted value is wrapped in a START/END marker
pair that embeds a nonce generated fresh, per call, via :mod:`secrets` (a
cryptographically unpredictable source -- NOT :mod:`random`, which is
seedable/predictable and would let an attacker who ever observed a prior
nonce, or who could influence the process seed, reconstruct or guess future
ones). The judge's instructions are built with the SAME nonce and told
exactly what a genuine marker looks like (``fence_marker_hint``), so a
payload authored before the call started -- and therefore blind to this
call's nonce -- cannot forge a matching START/END pair to prematurely close
a fence or open a fake one.

**Defense in depth: fence-shaped text is stripped from every value before
it is wrapped** (``strip_fence_lookalikes``), independent of nonce. Even a
payload that somehow guesses the marker *shape* (e.g. from leaked source,
or by brute-forcing many calls) still can't plant a working forged
boundary: any substring shaped like ANY marker of this grammar -- any tag,
any hex value in the nonce position, START or END -- is neutralised
(replaced with an inert, visibly-marked placeholder) before wrapping, not
only a marker matching this call's actual nonce. This is a structural
transform keyed on marker SHAPE, not a semantic/lexical detector over
instruction-like phrases -- this project has already measured that class of
defence unfit three times over (#130 judge-based, #164 token-overlap, #169
cue+anchor, the last of which had its exclusion list ENLARGE the smuggle
surface with every hardening pass). ``strip_fence_lookalikes`` never
inspects natural-language content or tries to recognise "look like an
instruction" -- it only ever matches the one fixed, project-specific fence
grammar below.

**What this does not do.** Fencing closes structural mimicry (forged
delimiters, fake role turns, fake schema blocks) because those attacks
depend on the judge parsing attacker text as prompt STRUCTURE, and the
fence removes any ambiguity about where structure ends and data begins. It
does NOT, and cannot, neutralise a semantic authority claim living entirely
inside a fence as ordinary prose (e.g. "the physician has already confirmed
this claim is accurate") -- that is not structure, it is content, and
content is exactly what the judge is asked to read and weigh. Each caller's
own system prompt must explicitly instruct the judge that authority /
prior-confirmation / instruction claims found INSIDE a fence are DATA ONLY
and carry no special weight -- see ``app.semantic_support`` and
``app.source_ref_relevance``'s ``_SYSTEM_PROMPT_TEMPLATE``. That residual
is instruction-following, not fencing, and is not closed by this module.
"""

from __future__ import annotations

import re
import secrets

_NONCE_BYTES = 16  # 32 hex chars per nonce -- effectively unguessable per call

# Matches this project's fence grammar in full generality (any tag, any hex
# run in the nonce position, START or END) so that even a correctly-guessed
# marker SHAPE is stripped from untrusted content before it is ever wrapped
# -- see module docstring, "Defense in depth".
_FENCE_LOOKALIKE = re.compile(r"<<<EVIDENCE-[0-9a-fA-F]*-[A-Z0-9_]*-(?:START|END)>>>")

_PLACEHOLDER = "[fence marker removed]"


def new_nonce() -> str:
    """A fresh, per-call, cryptographically unpredictable nonce
    (:mod:`secrets`, never :mod:`random` -- a seedable/predictable RNG must
    never gate a security boundary)."""
    return secrets.token_hex(_NONCE_BYTES)


def strip_fence_lookalikes(text: str) -> str:
    """Neutralise (replace, never silently drop -- see module docstring)
    any substring shaped like one of our fence markers, so untrusted
    content can never plant a working forged boundary even by luck or by
    guessing the marker grammar. Fail-safe direction: replacing with an
    inert, visibly-marked placeholder can only ever make a payload's forged
    structure LESS convincing to the judge, never more convincing."""
    return _FENCE_LOOKALIKE.sub(_PLACEHOLDER, text)


def fence(nonce: str, tag: str, text: str) -> str:
    """Wrap ``text`` in a nonce-bound START/END envelope labelled ``tag``
    (e.g. ``"CLAIM"``, ``"QUOTE"``, ``"ESTABLISHED_FACTS"``). ``text`` is
    neutralised via ``strip_fence_lookalikes`` before wrapping. ``tag`` is
    always a fixed literal supplied by the caller, never untrusted input,
    so it is not itself neutralised."""
    cleaned = strip_fence_lookalikes(text)
    return f"<<<EVIDENCE-{nonce}-{tag}-START>>>\n{cleaned}\n<<<EVIDENCE-{nonce}-{tag}-END>>>"


def fence_marker_hint(nonce: str) -> str:
    """Human-readable description of this call's exact marker grammar, for
    inclusion in the judge's system prompt so it can be told, concretely,
    what a genuine boundary looks like -- and, by construction, that
    nothing else appearing in the untrusted content is one."""
    return f"<<<EVIDENCE-{nonce}-<TAG>-START>>> ... <<<EVIDENCE-{nonce}-<TAG>-END>>>"
