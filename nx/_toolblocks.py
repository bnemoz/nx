"""Grouped tool-call rendering for parallel dispatch (0.31.0).

`_turn._run_read_only_parallel()` runs `run_tool()` on a thread pool, and
`run_tool()` draws its own box (`┌─ name` / `│ args` / result lines / `└─`)
from whichever thread it happens to execute on, each with its own LifeSpinner
thread writing `\033[2K\r` frames. Three concurrent reads therefore produced
three half-boxes and a spinner line landing on a sibling's just-printed row:

    ┌─ read_file a.py                     ┌─ read_file b.py
    │ {"path": "a.py"}                    running read_file [▪▪▪········]
    running read_file [▪···········]      │ {"path": "b.py"}
    │ … (output collapsed …)              │ … (output collapsed …)
    └─                                    └─

The fix is ownership, not a stdout lock. A fan-out attaches a per-thread
recorder to every pool worker; `run_tool()` appends a `ToolBlock` to it
instead of printing (and starts no spinner). The dispatching thread — the
only one that owns the terminal — then renders the whole group as ONE box in
the model's call order:

    ┌─ 3 tool calls in parallel · read_file ×3
    │ ├─ read_file a.py
    │ ├─ read_file b.py · cached
    │ └─ read_file c.py
    └─ 3 calls · 0.4s · output collapsed (/tooloutput)

One collapse hint per group instead of one per call, one spinner, one elapsed
time, and a `· cached` marker on cache-hit reads — which previously printed
nothing at all, so a served-from-cache call looked like a dropped call.
"""
from __future__ import annotations

import json
import threading
from collections import Counter
from dataclasses import dataclass, field

from ._ansi import DIM, RESET, C, LifeSpinner
from ._progress import ProgressCounter
from ._rich import cprint


# Tools whose call boxes render in the MEMORY role color instead of ACCENT,
# so memory traffic stands apart from ordinary tool calls.
_MEMORY_TOOLS = frozenset({"memory_search", "memory_write", "memory_delete"})

# Tools whose call boxes render in the diff-deletion color (C.RED, the hue the
# approval diff uses for removed lines) — recall_lessons digs through PAST
# (i.e. deleted-from-context) transcripts.
_DELETED_TOOLS = frozenset({"recall_lessons"})


# Tools whose call boxes render in the SUBAGENT role color instead of ACCENT,
# so delegated (sub-agent) work stands apart from ordinary tool calls.
_SUBAGENT_TOOLS = frozenset({"spawn_agent"})


def tool_accent(name: str) -> str:
    """The rail/border color for one tool call's box: C.RED for the
    deletion-accent tools, C.MEMORY for the memory tools, C.SUBAGENT for
    spawn_agent, C.ACCENT otherwise."""
    if name in _DELETED_TOOLS:
        return C.RED
    if name in _SUBAGENT_TOOLS:
        return C.SUBAGENT
    return C.MEMORY if name in _MEMORY_TOOLS else C.ACCENT


def clip(text: str, limit: int) -> str:
    """Truncate a one-line display preview to about `limit` characters."""
    if len(text) <= limit:
        return text
    keep = max(0, limit - 4)
    return text[:keep] + "..."


def tool_display(name: str, args: dict) -> tuple[str, str, str]:
    """Shared header/args rendering for one tool call.

    Returns (header, args preview, detail). A path-taking tool puts its target
    in the header, so the file an edit touches stays legible when the preview
    (usually a big diff body) gets clipped. `detail` is the preview with the
    path key dropped: the part a header cannot show, such as a search pattern
    or a read offset, and empty for the common single-path call.

    Consumed by the sequential tool box in run_tool() (header line plus full
    preview line) and by render_group() below (header plus detail on ONE line)."""
    raw = json.dumps(args)
    preview = clip(raw, 120)
    header = name
    path_arg = args.get("path")
    if isinstance(path_arg, list):
        path_arg = path_arg[0] if path_arg else None
    if isinstance(path_arg, str) and path_arg not in (".", ""):
        header = f"{name} {path_arg}"
    rest = {k: v for k, v in args.items() if k != "path"}
    detail = clip(json.dumps(rest), 60) if rest else ""
    return (header, preview, detail)


@dataclass
class ToolBlock:
    """One recorded tool call, rendered later by the terminal-owning thread.

    `body` holds display-safe lines (the same 8-line preview the sequential box
    prints) and is EMPTY when the result body is collapsed — the group footer
    carries a single collapse hint instead of every block repeating it.
    `detail` is `tool_display`'s third value, and `note` carries either a
    failure's first line or a standalone note (a pre-hook veto)."""

    name: str
    header: str = ""            # full display header (name [+ target path])
    detail: str = ""
    body: list[str] = field(default_factory=list)
    note: str = ""
    cached: bool = False
    failed: bool = False


class _Recorder:
    __slots__ = ("blocks",)

    def __init__(self) -> None:
        self.blocks: list[ToolBlock] = []


_TLS = threading.local()


def attach_recorder() -> _Recorder:
    """Install this thread's recorder (a fan-out worker calls this before it
    dispatches anything) and return it; its `.blocks` fills in call order."""
    rec = _Recorder()
    _TLS.rec = rec
    return rec


def detach_recorder() -> None:
    _TLS.rec = None


def current_recorder() -> _Recorder | None:
    """The calling thread's recorder, or None when it owns its own output
    (the normal case: sequential dispatch, the REPL, tests)."""
    return getattr(_TLS, "rec", None)


def record_block(block: ToolBlock) -> bool:
    """Append `block` when this thread is being recorded. True if recorded."""
    rec = getattr(_TLS, "rec", None)
    if rec is None:
        return False
    rec.blocks.append(block)
    return True


def record_note(text: str) -> bool:
    """Route a standalone note line (a pre-hook veto) into the recorder
    instead of printing it from a pool thread. True if recorded."""
    return record_block(ToolBlock(name="", note=text))


def group_spinner(label: str, counter: "ProgressCounter | None" = None,
                  elapsed: bool = True) -> LifeSpinner:
    """The single activity line a parallel group shows while its calls are in
    flight — replacing one spinner thread per call fighting over one row.

    0.35.0: the line is now live, because `running 3 tool calls [▪▪▪·······]`
    looked identical whether one call had landed or two of three, and whether
    the group was 0.2s or 40s old. `counter` renders as `· 2/3` (a worker bumps
    it as its call returns) and `elapsed` as `· 4.2s`, so the row says how far
    along the turn is instead of merely that something is happening."""
    sp = LifeSpinner(label=label)
    sp.counter = counter
    sp.show_elapsed = elapsed
    return sp


def _group_title(blocks: list[ToolBlock]) -> str:
    """`3 tool calls in parallel · read_file ×2, list_dir`. Counts calls, not
    blocks: a note line (a veto) is not a call. Ties break on the tool name, so
    the title is stable for a given call set."""
    names = [b.name for b in blocks if b.name]
    counts = Counter(names)
    parts = [f"{n} ×{c}" if c > 1 else n
             for n, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    if len(parts) > 3:
        parts = parts[:3] + [f"+{len(parts) - 3} more"]
    title = f"{len(names)} tool {'call' if len(names) == 1 else 'calls'} in parallel"
    return f"{title} · {', '.join(parts)}" if parts else title


def render_group(blocks: list[ToolBlock], elapsed: float, visible: bool) -> None:
    """Draw a whole parallel group as one box, in recorded (call) order.

    `visible` is the caller's `st.TOOL_OUTPUT_VISIBLE` read at render time,
    passed in so this module stays free of state imports and the renderer can
    be tested directly."""
    if not blocks:
        return
    accent = tool_accent(next((b.name for b in blocks if b.name), ""))
    cprint(f"\n{accent}  ┌─ {_group_title(blocks)}{RESET}")
    calls = [b for b in blocks if b.name]
    for i, b in enumerate(blocks):
        branch = "└─" if i == len(blocks) - 1 else "├─"
        if not b.name:  # standalone note (pre-hook veto): no call header
            cprint(f"{accent}  │{RESET}  {accent}{branch}{RESET} {b.note}")
            continue
        b_accent = tool_accent(b.name)
        line = f"{b_accent}  │{RESET}  {b_accent}{branch} {b.header or b.name}{RESET}"
        if b.detail:
            line += f" {DIM}{b.detail}{RESET}"
        if b.cached:
            line += f" {DIM}· cached{RESET}"
        if b.failed:
            # Collapsed groups show a failure's reason inline: "· error" alone
            # would force /tooloutput to learn what nx already knows.
            line += (f" {C.ERROR}✗ {b.note}{RESET}" if (b.note and not visible)
                     else f" {C.ERROR}· error{RESET}")
        cprint(line)
        if visible and b.body:
            # Keep the box rail down the group, but not under the final entry
            # where the box has already closed.
            rail = "│ " if i < len(blocks) - 1 else "  "
            for body_line in b.body:
                cprint(f"{b_accent}  │{RESET}  {rail}{body_line}")
    n = len(calls)
    footer = f"{n} call{'' if n == 1 else 's'} · {elapsed:.1f}s"
    errs = sum(1 for b in calls if b.failed)
    if errs:
        footer += f" · {errs} error{'' if errs == 1 else 's'}"
    if not visible:
        footer += " · output collapsed (/tooloutput)"
    cprint(f"{accent}  └─{RESET} {DIM}{footer}{RESET}")
