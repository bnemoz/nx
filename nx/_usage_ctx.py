"""CTX_TAIL: the last <=1000 chars of the most recent assistant TEXT the model
emitted, remembered so a failure-path usage event (tool_call error, arg repair,
breaker trip, gate refusal) can carry the model's own words from just before
the call. That is the pair the harness-vs-model diagnosis needs: the failing
call and what the model was saying when it made it.

A single process-global snippet (truncated on write), no queue, no lock: the
worst case under swarm fan-out is a tail from a sibling agent — diagnostic
telemetry, not correctness state. Never raises."""
from __future__ import annotations

_TAIL_CHARS = 1000

_snippet = ""


def remember(text: str) -> None:
    """Record `text` as the most recent assistant text (truncated on write so
    the retained copy stays bounded). Best-effort — never raises."""
    global _snippet
    try:
        if isinstance(text, str) and text.strip():
            _snippet = text[-_TAIL_CHARS:]
    except Exception:  # noqa: BLE001 — telemetry must never break a turn
        pass


def tail() -> str:
    """The remembered snippet ("" when nothing was ever remembered)."""
    return _snippet
