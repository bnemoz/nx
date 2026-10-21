"""Key registry + contextual `?` overlay (UIUX_SPEC Phase C2, 0.136.0).

ONE source of truth for what every interactive surface's keys mean. The
chatbox border hint line and the `?` overlay are both rendered FROM this
registry, so they cannot drift (a test pins it: same tuples, both paths).

Surfaces:
  * "chatbox"   — the prompt editor (keys handled in `_chatbox._raw_read_key`)
  * "approval"  — the Y/n/a approval prompt (`_approval._ask_approval`)
  * "tabs"      — the full-screen /tabs dashboard (`_surfaces.TabsView`)
  * "bgjobs"    — the full-screen /bg background-jobs view
                  (`_bg_completions.BackgroundJobsView`)
  * "diff"      — the full-screen /diff working-tree review (`_review.ReviewView`)
  * "sessions"  — the full-screen /sessions browser (`_sessions_cmd.SessionsView`)
  * "todos"     — the full-screen /todo plan view (`_todo.TodoView`)
  * "fullscreen"— the fallback list for any view without its own entry

Each entry is (key, meaning); `{mode}` in a meaning is substituted per
surface rendering (the chatbox colors its own yolo/manual tail).

`?` renders a single-frame overlay (a small `_render.panel`, no alternate
screen — full-screen views arrive in Phase D), any key dismisses. Rendering
goes exclusively through `_render.py` (semantic roles, one glyph set).
"""
from __future__ import annotations

from typing import Optional

# key, meaning — ORDER is display order (hints left→right, overlay top→bottom)
KEYS: dict[str, list[tuple[str, str]]] = {
    "chatbox": [
        ("Enter", "send"),
        ("Shift+Enter / ^J", "newline"),
        ("^C", "cancel"),
        ("^L", "clear"),
        ("^Q", "exit"),
        ("^S", "compress"),
        ("^O", "jobs"),
        ("?", "keys"),
        ("shift+tab", "{mode}"),
    ],
    "approval": [
        ("Y / Enter", "allow"),
        ("n", "deny"),
        ("a", "always allow"),
        ("+", "wider diff"),
        ("-", "narrower diff"),
        ("shift+tab", "toggle yolo (allow+proceed)"),
        ("?", "keys"),
        ("esc", "back to chat"),
    ],
    # Phase D (0.137.0) — the fullscreen surfaces. Entered from the prompt,
    # Esc returns to scroll-chat, `?` paints this same overlay INSIDE the
    # alternate screen. The fullscreen shell handles ?/Esc/q itself; every
    # other key here is the view's own handler, so this list is the hint line,
    # the overlay and the key dispatch all at once.
    "tabs": [
        ("j/k", "move"),
        ("enter", "steer row"),
        ("r", "refresh"),
        ("?", "keys"),
        ("esc", "back"),
    ],
    "bgjobs": [
        ("j/k", "move"),
        ("r", "refresh"),
        ("?", "keys"),
        ("esc", "back"),
    ],
    "diff": [
        ("j/k", "move"),
        ("enter", "open diff"),
        ("tab", "back to list"),
        ("a", "commit…"),
        ("r", "refresh"),
        ("?", "keys"),
        ("esc", "back"),
    ],
    "context": [
        ("j/k", "move"),
        ("enter", "open transcript"),
        ("p/space", "pin/unpin turn"),
        ("a", "live/as-sent"),
        ("c", "last compression report"),
        ("t", "per-call event log (/turns)"),
        ("r", "refresh"),
        ("?", "keys"),
        ("esc", "back"),
    ],
    # The /compress --show ledger view: `c` in the "context" surface jumps here
    # (the registry itself has no cross-view notion — the jump is ContextView
    # pushing CompressionView onto the shell's view stack).
    "compress": [
        ("r", "refresh"),
        ("?", "keys"),
        ("esc", "back"),
    ],
    # The /turns event-log view: `t` in the "context" surface jumps here (same
    # push-onto-the-shell-stack pattern as `c` → compress).
    "turns": [
        ("r", "refresh"),
        ("?", "keys"),
        ("esc", "back"),
    ],
    "sessions": [
        ("j/k", "move"),
        ("/", "search"),
        ("enter", "preview"),
        ("R", "resume"),
        ("d", "delete"),
        ("esc", "back"),
    ],
    "memory": [
        ("j/k", "move"),
        ("enter", "open note"),
        ("tab", "back to list"),
        ("d", "delete note"),
        ("r", "refresh"),
        ("?", "keys"),
        ("esc", "back"),
    ],
    "undo": [
        ("j/k", "move"),
        ("enter", "open checkpoint"),
        ("tab", "back to list"),
        ("u", "undo newest"),
        ("r", "refresh"),
        ("?", "keys"),
        ("esc", "back"),
    ],
    "todos": [
        ("j/k", "move"),
        ("?", "keys"),
        ("esc", "back"),
    ],
    # Any fullscreen view without its own entry degrades to this one (the
    # shell's keys are identical on every surface, so a new view is never
    # hint-less).
    "fullscreen": [
        ("j/k", "move"),
        ("enter", "select"),
        ("?", "keys"),
        ("esc", "back"),
    ],
}

# Keys that open the overlay (same for every surface — one gesture).
OVERLAY_KEY = "?"


def keys_for(surface: str) -> list[tuple[str, str]]:
    """The registry entries for `surface` (the ONE list both hint lines and
    the overlay read). Unknown surface -> empty list (never raises)."""
    return KEYS.get(surface, [])


def _sub(meaning: str, subs: dict[str, str]) -> str:
    for k, v in subs.items():
        meaning = meaning.replace("{" + k + "}", v)
    return meaning


def short_key(key: str) -> str:
    """Compact form of a registry key for bracket-style prompt tails:
    "Y / Enter" -> "Y", "shift+tab" -> "shift+tab" (display-only)."""
    return key.split(" / ")[0].split(" ")[0]


def prompt_tail(surface: str) -> str:
    """The bracket-style key tail a prompt advertises, e.g. approval ->
    "Y/n/a/+/-/shift+tab/?/esc". Generated from the SAME registry the
    overlay and hint lines render (C2/C3: one source of truth)."""
    return "/".join(short_key(k) for k, _ in keys_for(surface))


def hint_line(surface: str, **subs: str) -> str:
    """The flat border-hint text for `surface` ("Enter send · ^C cancel · …").

    Exactly the `key meaning` pairs of `keys_for(surface)` joined with the
    ` · ` separator the chatbox has always used; `{placeholder}` meanings
    are substituted from `subs`. The trailing "? keys" entry is included —
    the hint that the overlay exists is part of the hint line."""
    from ._render import TAIL_BULLET
    sep = f" {TAIL_BULLET} "
    return sep.join(f"{k} {_sub(m, subs)}" for k, m in keys_for(surface))


def overlay_lines(surface: str, *, title: Optional[str] = None,
                  **subs: str) -> list[str]:
    """The `?` overlay as a list of ready-to-emit box lines (one frame, no
    alternate screen). Built from the SAME `keys_for(surface)` tuples as
    `hint_line` — hints and overlay are one source of truth by construction."""
    from ._render import panel, TAIL_BULLET
    entries = keys_for(surface)
    head = title or f"keys {TAIL_BULLET} {surface} (any key closes)"
    rows = [f"{k:<16}{_sub(m, subs)}" for k, m in entries]
    return panel(head, rows)


def print_overlay(surface: str, **subs: str) -> None:
    """Render the overlay for `surface` through `_render.commit_line` (the
    single-console discipline of SPEC_LIVEENGINE holds — no raw stdout write,
    no Live bypass)."""
    from ._render import commit_line
    for line in overlay_lines(surface, **subs):
        commit_line(line)


__all__ = ["KEYS", "OVERLAY_KEY", "keys_for", "short_key", "prompt_tail",
           "hint_line", "overlay_lines", "print_overlay"]
