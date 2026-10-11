"""Single slash-command registry (UIUX_SPEC Phase A1, 0.134.0).

One table of command metadata — name, aliases, args spec, help text,
disclosure tier, and the arg-dropdown options — from which every former
hand-synced list derives:

  * `_chatbox.SLASH_COMMANDS`   <- `menu_list()`     (completion menu)
  * `_chatbox.SLASH_ARG_OPTIONS` <- `arg_options_map()` (keyword dropdowns)
  * `_repl_settings_cmds._HELP_COMMANDS` <- `help_rows()` (/help rendering)
  * `_repl._REPL_COMMANDS`      <- `handler_map()`   (dispatch)

Handlers themselves stay where they live (`_repl.py`,
`_repl_settings_cmds.py`, `_sessions_cmd.py`, `_memory_cmd.py`, ...); the
registry holds metadata only and the handlers bind through the `command`
decorator (directly where a handler is defined in `_repl.py`, or via
`bind_handlers` for the imported ones). The old module-level names
(`SLASH_COMMANDS`, `SLASH_ARG_OPTIONS`, `_REPL_COMMANDS`, `_HELP_COMMANDS`)
are re-exported from their original homes so tests that patch them keep
working (moved-names rule).

Disclosure tiers (UIUX_SPEC UX principle 2): T0 always visible, T1 on
demand (/help), T2 dashboard (/tabs), T3 logs. Informational for now —
emission sites adopt them in later phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from . import _state as st
from ._ansi import THEMES

ArgOptions = Union[list, Callable[[], list]]


@dataclass
class CommandSpec:
    """Metadata for one slash command. `handler` binds at import of the
    module that defines it (see `command` / `bind_handlers`)."""
    name: str
    help: str = ""
    args: Optional[str] = None
    # (alias, alias help shown in menus/help rows, hidden_in_menu)
    aliases: list = field(default_factory=list)
    arg_options: Optional[ArgOptions] = None
    tier: int = 1
    hidden: bool = False
    handler: Optional[Callable] = None


# ---------------------------------------------------------------------------
# The registry. Insertion order IS the completion-menu order (it mirrors the
# old hand-synced SLASH_COMMANDS display list, which follows the README
# "Commands" table grouping). Aliases listed here dispatch like the primary
# command (they bind to the same handler) but only show in the menu when
# their alias help text is non-empty and not hidden.
# ---------------------------------------------------------------------------
COMMANDS: "dict[str, CommandSpec]" = {}

_SPECS = [
    ("/help", "list all slash commands", None, [], None, 1),
    ("/init", "generate/update AGENTS.md for this project", None, [], None, 1),
    ("/source", "list sources, switch to one, or override its model",
     "[name] [model]", [], lambda: list(st.SOURCE_ORDER), 1),
    ("/theme", "list color themes, or switch to one", "[name]", [],
     sorted(THEMES), 1),
    ("/provider", "show or set OpenRouter provider-routing order",
     "[source] [order|off]", [], None, 1),
    ("/budget", "show or set the reasoning token budget", "[tokens|off]",
     [], ["off"], 1),
    ("/yolo", "toggle auto-approve for writes and bash", None, [], None, 0),
    ("/resources", "show this nx process's CPU% and RSS memory", None,
     [("/resmon", "alias for /resources", True)], None, 1),
    ("/context", "context breakdown by message; pin/unpin turns against /compress",
     "[all|pin|unpin|clear]", [("/ctx", "alias for /context", True)],
     ["pin", "unpin", "clear"], 1),
    ("/cost", "session usage + per-source cost scoreboard", None, [], None, 1),
    ("/status", "snapshot: source, model, caps, budgets, fence, approval",
     None, [], None, 1),
    ("/bg", "list this session's background run_bash jobs and sub-agents, "
     "or /bg kill <pid>; /bg all lists every session's", "[kill <pid>|all]",
     [], ["kill", "all"], 1),
    ("/tabs", "live dashboard of running surfaces (main turn, bg jobs, agents); Ctrl-T",
     None, [], None, 2),
    ("/panel", "toggle the right status rail on the live region", None, [],
     None, 0),
    ("/approval", "show or set risk threshold (all/low/medium/high/yolo)",
     "[level]", [], ["all", "low", "medium", "high", "yolo"], 1),
    ("/sessions", "list recent sessions, or show one in full", "[n]", [],
     None, 1),
    ("/export", "write the current session to a markdown file", "[md] [path]",
     [], ["md"], 1),
    ("/resume", "resume a past session (n/id/prefix/title)", "[target]", [],
     None, 1),
    ("/save", "save the current session (optional custom title)", "[title]",
     [], None, 1),
    ("/fork", "branch the conversation into a new session (original kept)",
     "[title]", [], None, 1),
    ("/memory", "manage persistent memory",
     "[save|remember|delete|list|archived|optimize|reflect]", [],
     ["save", "remember", "delete", "list", "archived", "optimize",
      "reflect"], 1),
    ("/retro", "distill this session's friction (usage.jsonl) into gotcha memories",
     None, [], None, 1),
    ("/review", "scoped reviewer over the current work: ranked findings, no fixes (Ctrl+R)",
     "[focus]", [], None, 1),
    ("/build", "spec-driven pipeline: spec → parallel implementers → critique → fixes, on a feature branch",
     "[issue]", [], None, 1),
    ("/delete", "delete a saved session", "[target]", [], None, 1),
    ("/todo", "show the model's current todo_write list", None, [], None, 1),
    ("/wip", "snapshot work state (branch, dirty files, todos, note) into the session",
     "[note]", [], None, 1),
    ("/plan", "start/view/exit plan mode: /plan <goal>, /plan, /plan done|off|discard",
     "[goal|done|off|discard]", [], ["done", "off", "discard"], 1),
    ("/verify", "set / clear / force-run the post-edit verify command (NX_VERIFY_CMD)",
     "[cmd|off|run]", [], ["off", "run"], 1),
    ("/compress", "summarize older turns into one, keep last 2 verbatim",
     None, [("/compact", "alias for /compress", False)], None, 1),
    ("/autocompress", "show or set the auto-compress threshold",
     "[pct|off|on]", [], ["off", "on"], 1),
    ("/recover", "force a low-temp visible checkpoint after a bad stream",
     "[note]", [], None, 1),
    ("/reasoning", "on|off|last — toggle or show reasoning trace",
     "on|off|last", [], ["on", "off", "last"], 1),
    ("/tooloutput", "on|off|last — toggle or show tool-call result",
     "on|off|last", [("/tools", "legacy alias for /tooloutput", False)],
     ["on", "off", "last"], 1),
    ("/toolcalls", "tool-calling mode + breaker: text|native|auto, reset [tool], or bare to show",
     "[text|native|auto] [reset [tool]]", [],
     ["reset", "text", "native", "auto"], 1),
    ("/undo", "revert the last write_file/edit_file/multi_edit", "[--list]",
     [], None, 1),
    ("/reset", "clear conversation, start a fresh session", None,
     [("/clear", "alias for /reset", False),
      ("/new", "alias for /reset", False)], None, 1),
    ("/quit", "exit", None, [("/exit", "alias for /quit", False)], None, 0),
]

for _name, _help, _args, _alias, _argopts, _tier in _SPECS:
    COMMANDS[_name] = CommandSpec(name=_name, help=_help, args=_args,
                                  aliases=list(_alias), arg_options=_argopts,
                                  tier=_tier)
del _name, _help, _args, _alias, _argopts, _tier


def command(name: str, help: Optional[str] = None, args: Optional[str] = None,
            aliases: Optional[list] = None, arg_options: Optional[ArgOptions] = None,
            tier: Optional[int] = None):
    """Decorator binding a handler (and optionally overriding metadata) to a
    registry entry. Usage:

        @command("/tabs")
        def _cmd_tabs(user, state): ...

    Unknown names get a minimal spec so a new command is one edit here (add
    the decorated handler; add its table row for menu/help text)."""
    def _register(fn):
        spec = COMMANDS.get(name)
        if spec is None:
            spec = COMMANDS[name] = CommandSpec(name=name)
        if help is not None:
            spec.help = help
        if args is not None:
            spec.args = args
        if aliases is not None:
            spec.aliases = list(aliases)
        if arg_options is not None:
            spec.arg_options = arg_options
        if tier is not None:
            spec.tier = tier
        spec.handler = fn
        return fn
    return _register


def bind_handlers(handlers: "dict[str, Callable]") -> None:
    """Bind the handlers defined elsewhere (the bulk case: every handler
    imported into `_repl.py`) through the same `command` decorator."""
    for cmd, fn in handlers.items():
        command(cmd)(fn)


def _resolve(spec: CommandSpec) -> "list[tuple[str, str, CommandSpec]]":
    """(display name, display help, spec) for the spec and its menu-visible
    aliases, in registry order."""
    out = []
    if not spec.hidden:
        out.append((spec.name, spec.help, spec))
    for alias in spec.aliases:
        aname, adesc, hidden = alias[0], alias[1], alias[2]
        if not hidden:
            out.append((aname, adesc, spec))
    return out


def menu_list() -> "list[tuple[str, str]]":
    """The chatbox completion menu ((command, description) pairs)."""
    return [(n, h) for n, h, _ in _all_display()]


def _all_display() -> "list[tuple[str, str, CommandSpec]]":
    out = []
    for spec in COMMANDS.values():
        out.extend(_resolve(spec))
    return out


def arg_options_map() -> "dict[str, ArgOptions]":
    """Keyword dropdown options keyed by command name (primary AND aliases
    — the "/tools" legacy alias gets the "/tooloutput" options). Commands
    without an enum-like first argument are absent, never None."""
    out: "dict[str, ArgOptions]" = {}
    for spec in COMMANDS.values():
        if spec.arg_options is None:
            continue
        out[spec.name] = spec.arg_options
        for alias in spec.aliases:
            out[alias[0]] = spec.arg_options
    return out


def help_rows() -> "list[tuple[str, str]]":
    """/help rows: (usage, description). The usage column names the command
    with its args and its aliases, e.g. "/reset, /clear, /new" — the same
    shape the old hand-synced _HELP_COMMANDS used (hidden-in-menu aliases
    like /resmon still show here, as they did before)."""
    rows = []
    for spec in COMMANDS.values():
        if spec.hidden:
            continue
        label = f"{spec.name} {spec.args}".strip() if spec.args else spec.name
        for alias in spec.aliases:
            label += f", {alias[0]}"
        rows.append((label, spec.help))
    return rows


def handler_map() -> "dict[str, Callable]":
    """Dispatch map: every command and every alias (menu-hidden included,
    e.g. /resmon) to its handler. Only specs with a bound handler appear, so
    importing this before the handlers bind yields a partial map — the
    binding happens at `_repl` import, which is when this is built."""
    out: "dict[str, Callable]" = {}
    for spec in COMMANDS.values():
        if spec.handler is None:
            continue
        out[spec.name] = spec.handler
        for alias in spec.aliases:
            out[alias[0]] = spec.handler
    return out
