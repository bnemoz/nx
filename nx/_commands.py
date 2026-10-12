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
    # /help section key (see _HELP_GROUP_ORDER); defaults to "general".
    group: str = "general"
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
    # (name, help, args, aliases, arg_options, tier, group)
    ("/help", "list all slash commands", None, [], None, 1, "session"),
    ("/init", "generate/update AGENTS.md for this project", None, [], None, 1, "session"),
    ("/source", "list sources, switch to one, or override its model",
     "[name] [model]", [], lambda: list(st.SOURCE_ORDER), 1, "model"),
    ("/theme", "list color themes, or switch to one", "[name]", [],
     sorted(THEMES), 1, "display"),
    ("/provider", "show or set OpenRouter provider-routing order",
     "[source] [order|off]", [], None, 1, "model"),
    ("/budget", "show or set the reasoning token budget", "[tokens|off]",
     [], ["off"], 1, "model"),
    ("/yolo", "toggle auto-approve for writes and bash", None, [], None, 0, "session"),
    ("/resources", "show this nx process's CPU% and RSS memory", None,
     [("/resmon", "alias for /resources", True)], None, 1, "display"),
    ("/context", "context breakdown by message; pin/unpin turns against /compress",
     "[all|pin|unpin|clear]", [("/ctx", "alias for /context", True)],
     ["pin", "unpin", "clear"], 1, "context"),
    ("/cost", "session usage + per-source cost scoreboard", None, [], None, 1, "context"),
    ("/status", "snapshot: source, model, caps, budgets, fence, approval",
     None, [], None, 1, "context"),
    ("/bg", "list this session's background run_bash jobs and sub-agents, "
     "or /bg kill <pid>; /bg all lists every session's", "[kill <pid>|all]",
     [], ["kill", "all"], 1, "workflow"),
    ("/tabs", "live dashboard of running surfaces (main turn, bg jobs, agents); Ctrl-T",
     None, [], None, 2, "display"),
    ("/diff", "full-screen working-tree diff review; a commits via the reviewed git flow",
     "[path]", [], None, 1, "workflow"),
    ("/panel", "toggle the right status rail on the live region", None, [],
     None, 0, "display"),
    ("/approval", "show or set risk threshold (all/low/medium/high/yolo)",
     "[level]", [], ["all", "low", "medium", "high", "yolo"], 1, "session"),
    ("/sessions", "list recent sessions, or show one in full", "[n]", [],
     None, 1, "session"),
    ("/export", "write the current session to a markdown file", "[md] [path]",
     [], ["md"], 1, "session"),
    ("/resume", "resume a past session (n/id/prefix/title)", "[target]", [],
     None, 1, "session"),
    ("/save", "save the current session (optional custom title)", "[title]",
     [], None, 1, "session"),
    ("/fork", "branch the conversation into a new session (original kept)",
     "[title]", [], None, 1, "session"),
    ("/memory", "manage persistent memory",
     "[save|remember|delete|list|archived|optimize|reflect]", [],
     ["save", "remember", "delete", "list", "archived", "optimize",
      "reflect"], 1, "context"),
    ("/retro", "distill this session's friction (usage.jsonl) into gotcha memories",
     None, [], None, 1, "context"),
    ("/review", "scoped reviewer over the current work: ranked findings, no fixes (Ctrl+R)",
     "[focus]", [], None, 1, "workflow"),
    ("/build", "spec-driven pipeline: spec → parallel implementers → critique → fixes, on a feature branch",
     "[issue]", [], None, 1, "workflow"),
    ("/delete", "delete a saved session", "[target]", [], None, 1, "session"),
    ("/todo", "show the model's current todo_write list, full screen; Ctrl+G", None, [], None, 1, "context"),
    ("/wip", "snapshot work state (branch, dirty files, todos, note) into the session",
     "[note]", [], None, 1, "context"),
    ("/plan", "start/view/exit plan mode: /plan <goal>, /plan, /plan done|off|discard",
     "[goal|done|off|discard]", [], ["done", "off", "discard"], 1, "workflow"),
    ("/verify", "set / clear / force-run the post-edit verify command (NX_VERIFY_CMD)",
     "[cmd|off|run]", [], ["off", "run"], 1, "workflow"),
    ("/compress", "summarize older turns into one, keep last 2 verbatim",
     None, [("/compact", "alias for /compress", False)], None, 1, "context"),
    ("/autocompress", "show or set the auto-compress threshold",
     "[pct|off|on]", [], ["off", "on"], 1, "context"),
    ("/recover", "force a low-temp visible checkpoint after a bad stream",
     "[note]", [], None, 1, "context"),
    ("/reasoning", "on|off|last — toggle or show reasoning trace",
     "on|off|last", [], ["on", "off", "last"], 1, "display"),
    ("/tooloutput", "on|off|last — toggle or show tool-call result",
     "on|off|last", [("/tools", "legacy alias for /tooloutput", False)],
     ["on", "off", "last"], 1, "display"),
    ("/toolcalls", "tool-calling mode + breaker: text|native|auto, reset [tool], or bare to show",
     "[text|native|auto] [reset [tool]]", [],
     ["reset", "text", "native", "auto"], 1, "model"),
    ("/undo", "revert the last write_file/edit_file/multi_edit", "[--list]",
     [], None, 1, "workflow"),
    ("/reset", "clear conversation, start a fresh session", None,
     [("/clear", "alias for /reset", False),
      ("/new", "alias for /reset", False)], None, 1, "session"),
    ("/quit", "exit", None, [("/exit", "alias for /quit", False)], None, 0, "session"),
]

# /help section titles, in display order. A command whose group is missing
# here lands in the last (Settings & tooling) section, never dropped.
HELP_GROUP_ORDER = [
    ("session", "Session & approval"),
    ("context", "Context, memory & compression"),
    ("model", "Model & tool-calling"),
    ("display", "Display & status"),
    ("workflow", "Workflow & review"),
    ("general", "Settings & tooling"),
]

for _name, _help, _args, _alias, _argopts, _tier, _group in _SPECS:
    COMMANDS[_name] = CommandSpec(name=_name, help=_help, args=_args,
                                  aliases=list(_alias), arg_options=_argopts,
                                  tier=_tier, group=_group)
del _name, _help, _args, _alias, _argopts, _tier, _group


def command(name: str, help: Optional[str] = None, args: Optional[str] = None,
            aliases: Optional[list] = None, arg_options: Optional[ArgOptions] = None,
            tier: Optional[int] = None, group: Optional[str] = None):
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
        if group is not None:
            spec.group = group
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


def help_rows(grouped: bool = False) -> "list[tuple[str, str]] | list[tuple[str, list[tuple[str, str]]]]":
    """/help rows: (usage, description) pairs — or, with grouped=True,
    [(section title, [(usage, description), ...]), ...] in HELP_GROUP_ORDER.
    The usage column names the command with its args and its aliases, e.g.
    "/reset, /clear, /new" — the same shape the old hand-synced
    _HELP_COMMANDS used (hidden-in-menu aliases like /resmon still show
    here, as they did before). Commands with a group missing from
    HELP_GROUP_ORDER land in its last section, never dropped."""
    rows = []
    for spec in COMMANDS.values():
        if spec.hidden:
            continue
        label = f"{spec.name} {spec.args}".strip() if spec.args else spec.name
        for alias in spec.aliases:
            label += f", {alias[0]}"
        rows.append((label, spec.help, spec.group))
    if not grouped:
        return [(label, help) for label, help, _ in rows]
    known = [k for k, _ in HELP_GROUP_ORDER]
    fallback = known[-1]
    sections: "dict[str, list[tuple[str, str]]]" = {}
    for label, help, grp in rows:
        sections.setdefault(grp if grp in known else fallback, []).append((label, help))
    return [(title, sections.get(key, [])) for key, title in HELP_GROUP_ORDER]


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
