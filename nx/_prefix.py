"""The canonical prompt-prefix block — the one place that decides what sits
between the `SYSTEM` message and the first real turn (PERF.md PF-15, invariant
I-4).

A fresh session and a resumed one used to assemble this head their own way, and
the split was user-visible in the *cache*, not just the transcript: a fresh
session got four notes (AGENTS.md, the memory index, orientation, delegation)
appended by `_dispatch._initial_messages`, a resumed one got only two of them
rebuilt from `_resume.resumed_messages`. Any difference in these leading bytes
is a server prompt-prefix cache miss for the *entire* prompt, every turn, and it
shows up only as a mysterious `cached: 0%`.

So the block has one producer (`prefix_notes`) used by both paths, and one
pinned copy per session (`pin_prefix` / `check_prefix`) so a future regression
is a visible warning instead of a silent `cached: 0%`. The pin is checked with
`model_turn`, i.e. immediately before a request is sent — the only place a
prefix mutation can actually cost anything.

Byte-stability rules for anything added here:
  • nothing date/branch/pid/flag-dependent may be interpolated (SYSTEM is a
    static literal for exactly this reason);
  • the block is prepended/kept at the head, never rewritten — a note that
    needs refreshing is APPENDED at the tail (see _memory_cmd._refresh_memory_note)
    so the bytes before the turn boundary never move.
"""
from __future__ import annotations

import json
from typing import Any, Optional, cast

import threading

from . import _state as st

# The headers every prefix note opens with. Used to recognise the block inside a
# live message list (the notes are ordinary user-role messages, so role alone
# cannot identify them) — keep these in sync with the producers in
# prefix_notes(); a note whose header is missing here is simply not pinned.
PREFIX_MARKERS = (
    "[Project instructions from AGENTS.md",
    "[Saved memories",
    "[Orientation note:",
    "[Delegation note:",
)

# This session's pinned head: the JSON of SYSTEM followed by the prefix notes as
# they stood when the session started. None = nothing pinned (e.g. a sub-agent
# turn whose messages do not begin with the session SYSTEM, or a test that never
# called pin_prefix). Used to be one plain module global — concurrent turn loops
# in one process (parallel batch, async fan-out) clobbered each other's pin, so
# check_prefix warned spuriously and re-pinned to the wrong head (swarm audit
# P2). Now: a per-thread pin lives on _PREFIX_TLS while a loop holds it (see
# _prefix_pin_ctx); the module global remains the default store outside that —
# tests, import time, and every existing read keep working.
_PREFIX_TLS = threading.local()
pinned: Optional[list[str]] = None


def _get_pinned() -> Optional[list[str]]:
    v = getattr(_PREFIX_TLS, "pinned", _PREFIX_TLS)
    if v is not _PREFIX_TLS:
        return cast("Optional[list[str]]", v)
    return pinned


def _set_pinned(v: Optional[list[str]]) -> None:
    _PREFIX_TLS.pinned = v


class _prefix_pin_ctx:
    """Scope a per-thread pin: entered by a turn loop that pinned its own head
    (pin_prefix), left in place on exit (re-pin-to-global never happens — the
    module-global pin is only ever written by the outermost, single-session
    driver, exactly as pre-swarm)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        _set_pinned(None)
        return False


def is_prefix_note(msg: Any) -> bool:
    """True when *msg* is one of the session-start prefix notes."""
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return False
    content = msg.get("content")
    return isinstance(content, str) and content.lstrip().startswith(PREFIX_MARKERS)


def prefix_notes() -> list[dict[str, Any]]:
    """The canonical, ordered prefix block: AGENTS.md, the memory index, the
    orientation note, the delegation note — each included only when its producer
    returns one. The single source of truth for both `_initial_messages()` and
    `_resume.resumed_messages()`.

    Producers are imported lazily: this module is imported by `_dispatch`, which
    in turn is imported early, and each producer pulls optional machinery
    (sub-agent config, memory store) that must not run at import time.
    """
    from ._agents import _load_agents_md_note
    from ._memory_cmd import _load_memory_note
    from ._orientation import _load_orientation_note
    from ._subagent import _load_delegation_note

    root = getattr(st, "PROJECT_ROOT", None)
    candidates = (
        _load_agents_md_note(root),
        _load_memory_note(root),
        _load_orientation_note(),
        _load_delegation_note(),
    )
    return [n for n in candidates if n is not None]


def _head_len(messages: list[dict[str, Any]]) -> int:
    """Length of the leading SYSTEM + prefix-note run in *messages*."""
    if not messages or not isinstance(messages[0], dict) \
            or messages[0].get("role") != "system":
        return 0
    i = 1
    while i < len(messages) and is_prefix_note(messages[i]):
        i += 1
    return i


def _encode(msgs: list[dict[str, Any]]) -> list[str]:
    return [json.dumps(m, sort_keys=True, ensure_ascii=False) for m in msgs]


def pin_prefix(messages: list[dict[str, Any]]) -> bool:
    """Record this session's prefix head (SYSTEM + notes) as the bytes every
    later request must still open with. Returns True when something was pinned.

    Only pins a genuine head: it refuses a list whose first message is not the
    session system prompt (a sub-agent transcript, a test fixture), so a check
    can never fire on a list this module was never asked to guard."""
    n = _head_len(messages)
    if n == 0:
        return False
    _set_pinned(_encode(messages[:n]))
    return True


def pinned_prefix() -> list[str]:
    """The pinned head as a copy (test/debug surface). [] when nothing pinned."""
    p = _get_pinned()
    return list(p) if p else []


def _same_session_head(messages: list[dict[str, Any]]) -> bool:
    """True if *messages* opens with the same SYSTEM bytes the pin recorded.

    This is the check-side twin of pin_prefix's refusal (line 105): a list
    whose first message is not the session system prompt — a sub-agent
    transcript, a test fixture — is not guardable, and checking it against the
    parent's pin would misfire (a sub-agent head is a *different* prompt by
    design, not a bust) and then re-pin the global to the sub-agent head,
    permanently destroying the signal."""
    if not messages or not isinstance(messages[0], dict) \
            or messages[0].get("role") != "system":
        return False
    pinned = _get_pinned()
    if pinned is None:
        return False
    return _encode([messages[0]])[0] == pinned[0]


def check_prefix(messages: list[dict[str, Any]], *, warn: bool = True) -> bool:
    """Verify *messages* still opens with the pinned bytes. Returns True when
    intact (or when nothing was pinned / the list is not guardable).

    A False means the prompt prefix moved: every earlier turn's bytes changed,
    so the server's prefix cache is busted for the rest of the session. This is
    exactly the failure PERF.md I-3/I-4 forbid (a mid-history rewrite or insert),
    and it is otherwise invisible — the turn simply costs more. Warns once to
    stderr unless *warn* is False (the test surface passes False)."""
    if _get_pinned() is None:
        return True
    if not _same_session_head(messages):
        return True  # not the session head (sub-agent turn, fixture): uncheckable
    cur = _get_pinned()
    if cur is None:
        return True
    head = _encode(messages[:len(cur)])
    if head == cur:
        return True
    if warn:
        if not _warned[0]:
            _warned[0] = True
            import sys
            print("[perf] prompt prefix changed mid-session — the server prefix "
                  "cache for the rest of this session is busted (PERF.md I-3/I-4)",
                  file=sys.stderr)
    # Re-pin: the damage is done, and re-warning every turn would be noise.
    pin_prefix(messages)
    return False


# Warn at most once per process — the first bust is the informative one.
_warned = [False]
