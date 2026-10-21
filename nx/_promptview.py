"""Phase 5 of the explainable harness: the annotated system-prompt floor (`/prompt`).

Read-only. Every model call pays a floor the user never sees: the pinned main
`SYSTEM` prompt, the serialized active tool schemas (`tools=`, outside the
message list), and — for one-off calls — a satellite prompt per specialized
job (risk classification, the evidence reducer, session descriptions, the
sub-agent scopes). This module renders that floor section by section, each
with the shared token estimate (`nx/_tokens.estimate_tokens`, the same
estimator `/ctx` uses), so the report answers "what does the harness say to
the model before the user says anything?"

Nothing here mutates anything and nothing raises: a prompt constant that is
missing or not a string renders as a dim-able `—` placeholder, and the tool
schemas are read the same defensive way `_ctxreport._turn_overhead_est()`
reads them (active source `None` -> schema total 0, still rendered).
"""
from __future__ import annotations

from typing import Any

from ._fullscreen import View


def _est(text: Any) -> int:
    """Shared estimator over one block of text (never raises — a garbage
    constant estimates 0 rather than breaking the report)."""
    if not isinstance(text, str) or not text:
        return 0
    try:
        from ._tokens import estimate_tokens
        return int(estimate_tokens(text))
    except Exception:
        return 0


# --- the satellite prompts (nx/_prompts.py) ----------------------------------
# (attribute, label, cadence). Cadence wording follows _prompts.py's own
# docstring: satellites "never ride the per-turn floor — they go out on
# dedicated one-off calls"; the note says which call.
_SATELLITES: list[tuple[str, str, str]] = [
    ("RISK_SYSTEM", "risk classifier", "one-off call per gated tool action"),
    ("REDUCER_SYSTEM_PROMPT", "evidence reducer",
     "one-off call per long build/test log"),
    ("DESC_SYSTEM", "session describer", "one-off call, once per session"),
    ("SUBAGENT_SYSTEM", "sub-agent (read-only)", "per read-only sub-agent call"),
    ("SUBAGENT_WRITE_SYSTEM", "sub-agent (write)", "per write-scope sub-agent call"),
    ("SUBAGENT_IMPLEMENT_SYSTEM", "sub-agent (implementer)",
     "per implementer sub-agent call"),
    ("ADVISOR_SYSTEM", "advisor", "per sub-agent consult_advisor call"),
    ("CRITIQUE_SYSTEM", "critic", "per /review critique call"),
    ("BUILD_SPEC_SYSTEM", "build: spec writer", "per /build spec phase call"),
    ("BUILD_PACKAGE_SYSTEM", "build: package implementer",
     "per /build package phase call"),
    ("BUILD_FIX_SYSTEM", "build: fixer", "per /build fix phase call"),
]


def _satellites() -> list[tuple[str, str, str, str, int]]:
    """[(attr, label, cadence, text, est)] for every satellite constant that
    exists in `nx/_prompts.py` (missing ones are skipped here and rendered as
    placeholders by the report)."""
    try:
        from . import _prompts
    except Exception:
        _prompts = None
    out = []
    for attr, label, cadence in _SATELLITES:
        text = getattr(_prompts, attr, None)
        if not isinstance(text, str) or not text:
            continue
        out.append((attr, label, cadence, text, _est(text)))
    return out


def _main_system() -> tuple[str, int]:
    """The pinned main SYSTEM prompt (lives in `nx/__init__.py`; `_prefix.py`
    pins its exact bytes as the prompt-cache head) as (text, est)."""
    try:
        from . import SYSTEM
    except Exception:
        return "", 0
    text = SYSTEM if isinstance(SYSTEM, str) else ""
    return text, _est(text)


def _tools_payload() -> tuple[Any, int]:
    """The active tool schemas + their estimated token cost — the exact call
    `_ctxreport._turn_overhead_est()` makes (`_dispatch._active_tools()` JSON
    serialized). Never raises: an active source of `None` (or a failed read)
    means an empty payload and a 0 total, which the report still renders."""
    try:
        import json as _json
        from ._dispatch import _active_tools
        tools = _active_tools()
        return tools, _est(_json.dumps(tools))
    except Exception:
        return None, 0


def _preview(text: str, n: int = 60) -> str:
    """One-line, whitespace-collapsed preview of a prompt body."""
    return " ".join(text.split())[:n]


def _prompt_report_lines() -> list[str]:
    """The annotated prompt floor, plain text (no ANSI) — the fullscreen view
    paints the SAME list (the /context rule: the two surfaces cannot drift).

    Sections: the main SYSTEM prompt, the satellite prompts from
    `nx/_prompts.py` (each marked as one-off vs per-turn), the serialized
    active tool schemas, and a footer total.
    """
    lines: list[str] = []
    system, sys_est = _main_system()
    tools, tools_est = _tools_payload()
    n_tools = len(tools) if isinstance(tools, (list, tuple, dict)) else 0

    lines.append("  per-turn floor — what goes out on EVERY model call")
    if system:
        lines.append(f"  SYSTEM  nx/__init__.py · every turn · ~{sys_est} tok")
        lines.append(f"    {_preview(system)}")
    else:
        lines.append("  SYSTEM  nx/__init__.py · every turn · ~— "
                     "(prompt unavailable)")
    if n_tools:
        lines.append(f"  TOOLS  active tool schemas (tools=, outside the message "
                     f"list) · every turn · {n_tools} schema(s) · ~{tools_est} tok")
    else:
        lines.append("  TOOLS  active tool schemas · every turn · "
                     "— no active tool registry (no source active)")

    floor = sys_est + tools_est

    sats = _satellites()
    lines.append("")
    lines.append("  satellite prompts — dedicated one-off calls, NOT on the "
                 "per-turn floor")
    if sats:
        sat_total = sum(s[4] for s in sats)
        for attr, label, cadence, text, est in sats:
            lines.append(f"  {attr}  {label} · {cadence} · ~{est} tok")
            lines.append(f"    {_preview(text)}")
        lines.append(f"  ({len(sats)} satellite prompt(s) · ~{sat_total} tok "
                     f"total, none of it per-turn)")
    else:
        lines.append("    — no satellite prompts importable")

    lines.append("")
    lines.append(f"  total per-turn floor ~{floor} tok "
                 f"(SYSTEM ~{sys_est} + tool schemas ~{tools_est})")
    return lines


class PromptView(View):
    """The `/prompt` view: the annotated prompt floor, read-only.

    The TurnsView pattern — its body IS `_prompt_report_lines()`, so the
    inline command output and this view agree by construction; `r` re-reads
    the prompts and the active tool registry, esc/q leave.
    """

    surface = "prompt"
    title = "prompt"

    def __init__(self, state: Any = None) -> None:
        super().__init__()
        self.state = state
        self.lines: list[str] = []
        self.refresh()

    def refresh(self) -> None:
        self.lines = _prompt_report_lines()

    def render(self, width: int, height: int) -> list[str]:
        from ._keys import hint_line
        from ._render import RESET, panel, role
        lines = [f"{role('ACCENT')}prompt{RESET}{role('MUTED')} — "
                 f"{hint_line('prompt')}{RESET}"]
        budget = max(1, height - len(lines) - 2)   # hint line + box borders
        body = [f"{role('MUTED')}{ln[:width]}{RESET}"
                for ln in self.lines[:budget]]
        lines.extend(panel(
            self.title, body,
            footer="static per process — prompts never vary byte-to-byte · "
                   "reread with r"))
        return lines[:height]

    def on_key(self, key: str, shell: Any) -> str:
        self.notice = ""
        if key == "r":
            self.refresh()
        if key in ("q", "esc", "ctrl-c", "ctrl-d"):
            return "quit"
        return "repaint"


def prompt_view(state: Any = None) -> PromptView:
    """The /prompt view object the fullscreen shell runs."""
    return PromptView(state)


def _cmd_prompt(rest: str = "", state: Any = None) -> None:
    """/prompt — the annotated system-prompt floor. Fullscreen when on a tty,
    the same plain-text report inline otherwise (the /turns recipe)."""
    from ._fullscreen import wants_fullscreen, open_view
    if wants_fullscreen():
        open_view(prompt_view(state))
        return
    from ._render import RESET, role
    for ln in _prompt_report_lines():
        print(f"{role('MUTED')}{ln}{RESET}")


__all__ = ["_prompt_report_lines", "PromptView", "prompt_view", "_cmd_prompt"]
