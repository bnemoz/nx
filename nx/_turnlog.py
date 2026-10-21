"""Phase 3 of the explainable harness: the per-model-call event log (`/turns`).

Read-only over `st._METRICS`: the settled-call ring (`turns`) is the backbone —
one entry per model call, prompt/cache numbers already there — and the
`events` ring records notable harness interventions (see _Metrics.event()).
Nothing here mutates anything, and nothing raises: a missing/garbage ring
entry renders as "—".
"""
from __future__ import annotations

from typing import Any

from . import _state as st
from ._fullscreen import View


def _int_or(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _pct(hit: Any) -> str:
    """Cache fraction → `42%` or `—` (mirrors /ctx's cache column rule)."""
    f = _int_or(hit) / 100.0 if isinstance(hit, (int, float)) else hit
    if not isinstance(f, (int, float)) or f < 0:
        return "—"
    return f"{min(1.0, f) * 100:.0f}%"


def _event_notes(events: list[dict[str, Any]], call: int) -> list[str]:
    """The ring events attributed to settled call `call` (0-based)."""
    out = []
    for e in events:
        if isinstance(e, dict) and e.get("call") == call:
            kind = str(e.get("kind") or "?")
            detail = str(e.get("detail") or "").replace("\n", " ")
            out.append(f"{kind}: {detail}" if detail else kind)
    return out


def _turnlog_report_lines(metrics: Any = None) -> list[str]:
    """One line per settled model call + attributed harness events, plain
    text (no ANSI). `metrics` defaults to `st._METRICS`; the empty/garbage
    case renders the placeholder the command shows dim. The fullscreen view
    paints the SAME list (the /context rule: the two surfaces cannot drift).
    """
    m = metrics if metrics is not None else getattr(st, "_METRICS", None)
    turns = list(getattr(m, "turns", None) or []) if m is not None else []
    events = list(getattr(m, "events", None) or []) if m is not None else []
    if not turns:
        return ["  no settled calls yet"]
    lines = [f"  {len(turns)} settled call(s) · newest last · "
             f"{len(events)} event(s)"]
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            lines.append(f"  [{i}] —")
            continue
        prompt = _int_or(t.get("prompt_tokens"))
        cached = t.get("cache_read_tokens")
        # -1 is the _normalize_usage sentinel for "no cache breakdown at all".
        frac = (None if (not isinstance(cached, (int, float)) or cached < 0)
                else (min(1.0, cached / prompt) if prompt else 0.0))
        pct = "—" if frac is None else f"{frac * 100:.0f}%"
        gen = _int_or(t.get("gen_tokens"))
        row = f"  [{i}] prompt ~{_int_or(prompt)} · gen ~{gen} · cached {pct}"
        notes = _event_notes(events, i)
        if notes:
            row += "  · " + " · ".join(notes)
        lines.append(row)
    loose = [e for e in events
             if isinstance(e, dict) and e.get("call") is None]
    if loose:
        lines.append("  unattributed events:")
        for e in loose:
            kind = str(e.get("kind") or "?")
            detail = str(e.get("detail") or "").replace("\n", " ")
            lines.append(f"    {kind}: {detail}" if detail else f"    {kind}")
    return lines


class TurnsView(View):
    """The `/turns` view: the per-model-call event log, read-only.

    The CompressionView pattern minus the mutation — its body IS
    `_turnlog_report_lines()`, so the inline command output and this view
    agree by construction; `r` re-reads the rings, esc/q leave.
    """

    surface = "turns"
    title = "turns"

    def __init__(self, state: Any = None) -> None:
        super().__init__()
        self.state = state
        self.lines: list[str] = []
        self.refresh()

    def refresh(self) -> None:
        self.lines = _turnlog_report_lines()

    def render(self, width: int, height: int) -> list[str]:
        from ._keys import hint_line
        from ._render import RESET, panel, role
        lines = [f"{role('ACCENT')}turns{RESET}{role('MUTED')} — "
                 f"{hint_line('turns')}{RESET}"]
        budget = max(1, height - len(lines) - 2)   # hint line + box borders
        body = [f"{role('MUTED')}{ln[:width]}{RESET}"
                for ln in self.lines[:budget]]
        lines.extend(panel(
            self.title, body,
            footer=f"{len(self.lines)} line(s) · this session only · "
                   f"reread with r"))
        return lines[:height]

    def on_key(self, key: str, shell: Any) -> str:
        self.notice = ""
        if key == "r":
            self.refresh()
        if key in ("q", "esc", "ctrl-c", "ctrl-d"):
            return "quit"
        return "repaint"


def turns_view(state: Any = None) -> TurnsView:
    """The /turns view object the fullscreen shell runs."""
    return TurnsView(state)


def _cmd_turns(rest: str = "", state: Any = None) -> None:
    """/turns — per-model-call event log. Fullscreen when on a tty, the same
    plain-text report inline otherwise (the /compress --show recipe)."""
    from ._fullscreen import wants_fullscreen, open_view
    if wants_fullscreen():
        open_view(turns_view(state))
        return
    from ._render import RESET, role
    for ln in _turnlog_report_lines():
        print(f"{role('MUTED')}{ln}{RESET}")
