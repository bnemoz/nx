"""Phase 4 of the explainable harness: `/tools` — tool-gate transparency.

Read-only over `nx/_tool_gates.py`: which tool schemas the model is actually
offered this turn (active), which are withheld and WHY (the gate table +
per-gate refusal wording), and what unlocking a withheld tool would cost in
schema tokens (the same json-serialize + shared-estimate math
`_ctxreport._turn_overhead_est()` uses for the per-turn floor). Nothing here
mutates gate state, and nothing raises.
"""
from __future__ import annotations

from typing import Any

from ._fullscreen import View


def _schema_tokens(schema: dict[str, Any]) -> int:
    """Estimated schema tokens — the exact math _ctxreport's overhead line
    uses, so the unlock cost and the per-turn floor agree by construction."""
    import json as _json
    from ._tokens import estimate_tokens
    try:
        return estimate_tokens(_json.dumps(schema))
    except Exception:
        return 0


def _tools_report_lines() -> list[str]:
    """The current tool-gate state as plain text lines (no ANSI). The
    fullscreen view paints the SAME list (the /context rule: the two
    surfaces cannot drift)."""
    import nx as _m
    from ._tool_gates import (_GATE_REFUSALS, _TOOL_GATES, _active_tools)
    active = _active_tools()
    active_names = {t.get("function", {}).get("name") for t in active}
    lines = [f"  {len(active)} active tool(s) · {len(_TOOL_GATES)} gated"]
    # Unlock cost per withheld tool: schema tokens the model is NOT paying
    # this turn but would pay if the gate opened (or search_tools lifted it).
    withheld: list[tuple[str, str]] = []
    for t in getattr(_m, "TOOLS", ()) or ():
        if not isinstance(t, dict):
            continue
        name = t.get("function", {}).get("name") or "?"
        if name in active_names:
            continue
        gate = _TOOL_GATES.get(name)
        why = _GATE_REFUSALS.get(name) or (
            "malformed-args breaker dropped it this session"
            if gate is None else "withheld")
        est = _schema_tokens(t)
        withheld.append((name, f"~{est} tok to unlock · {why}"))
    if withheld:
        lines.append("  withheld this turn:")
        lines.extend(f"  · {name} — {why}" for name, why in withheld)
    else:
        lines.append("  nothing withheld — every registered schema is active")
    # PTC OS-sandbox status (0.168.7, audit P3-1): the preexec layers are
    # silent on failure, so the diagnostics surface must say which landed.
    try:
        from ._ptc_exec import sandbox_summary
        lines.append(f"  PTC sandbox: {sandbox_summary()}")
    except Exception:
        pass  # diagnostics never break the report
    return lines


class ToolsView(View):
    """The `/tools` view: the tiered gate registry state, read-only. Same
    CompressionView recipe: body IS `_tools_report_lines()`, `r` re-reads,
    esc/q leave; no mutation."""

    surface = "tools"
    title = "tools"

    def __init__(self, state: Any = None) -> None:
        super().__init__()
        self.state = state
        self.lines: list[str] = []
        self.refresh()

    def refresh(self) -> None:
        self.lines = _tools_report_lines()

    def render(self, width: int, height: int) -> list[str]:
        from ._keys import hint_line
        from ._render import RESET, panel, role
        lines = [f"{role('ACCENT')}tools{RESET}{role('MUTED')} — "
                 f"{hint_line('tools')}{RESET}"]
        budget = max(1, height - len(lines) - 2)   # hint line + box borders
        body = [f"{role('MUTED')}{ln[:width]}{RESET}"
                for ln in self.lines[:budget]]
        lines.extend(panel(
            self.title, body,
            footer=f"{len(self.lines)} line(s) · this turn's gate vector · "
                   f"reread with r"))
        return lines[:height]

    def on_key(self, key: str, shell: Any) -> str:
        self.notice = ""
        if key == "r":
            self.refresh()
        if key in ("q", "esc", "ctrl-c", "ctrl-d"):
            return "quit"
        return "repaint"


def tools_view(state: Any = None) -> ToolsView:
    """The /tools view object the fullscreen shell runs."""
    return ToolsView(state)


def _cmd_toolgates(rest: str = "", state: Any = None) -> None:
    """/toolgates — tool-gate transparency (`/tools` is taken: the legacy
    /tooloutput alias). Fullscreen when on a tty, the same plain-text report
    inline otherwise (the /turns recipe)."""
    from ._fullscreen import open_view, wants_fullscreen
    if wants_fullscreen():
        open_view(tools_view(state))
        return
    from ._render import RESET, role
    for ln in _tools_report_lines():
        print(f"{role('MUTED')}{ln}{RESET}")
