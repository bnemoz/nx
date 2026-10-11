#!/usr/bin/env python3
"""Command registry parity (UIUX_SPEC Phase A1, 0.134.0).

One registry (nx/_commands.py) replaced the three hand-synced lists
(SLASH_COMMANDS, _REPL_COMMANDS, SLASH_ARG_OPTIONS) plus _HELP_COMMANDS.
These tests pin the mechanical guarantees: every registry command has a
handler, every dispatched handler has registry metadata, every derived
list matches its registry derivation, and the registry covers every
command named in the README "Commands" table (fixes README drift
mechanically — a new README row without a registry entry fails here).
"""
import os
import re
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx  # noqa: E402,F401  (package init imports _repl, which binds handlers)
from nx import _commands as reg  # noqa: E402
from nx import _chatbox as cb  # noqa: E402
from nx import _repl_settings_cmds as rs  # noqa: E402
# nx/__init__ binds a function named `_repl` into its namespace (shadowing the
# submodule attribute), so the module must come from sys.modules.
rp = sys.modules["nx._repl"]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = open(os.path.join(REPO_ROOT, "README.md"), encoding="utf-8").read()


def _alias_owners() -> dict:
    """Every name the registry resolves to — primary AND aliases — mapped to
    the owning CommandSpec."""
    owners = {}
    for spec in reg.COMMANDS.values():
        owners[spec.name] = spec
        for alias in spec.aliases:
            owners[alias[0]] = spec
    return owners


def test_every_registry_command_has_a_handler():
    """No metadata row without a bound handler (the old failure mode: a
    list entry pointing at a command nothing dispatches)."""
    missing = [n for n, spec in reg.COMMANDS.items() if spec.handler is None]
    assert not missing, f"registry commands without a handler: {missing}"


def test_every_handler_has_registry_metadata():
    """Every dispatch key resolves through the registry: the handler on the
    map IS the owning spec's handler (aliases included, e.g. /exit -> /quit)."""
    owners = _alias_owners()
    for cmd, fn in rp._REPL_COMMANDS.items():
        spec = owners.get(cmd)
        assert spec is not None, f"dispatched but no registry metadata: {cmd}"
        assert fn is spec.handler, f"{cmd}: dispatch handler != registry handler"


def test_registry_aliases_all_dispatch():
    """Aliases in metadata dispatch to the primary's handler."""
    for spec in reg.COMMANDS.values():
        for alias, _desc, _hidden in spec.aliases:
            assert alias in rp._REPL_COMMANDS, f"alias {alias} not dispatched"
            assert rp._REPL_COMMANDS[alias] is spec.handler


def test_derived_lists_match_the_registry():
    """The re-exported module-level names are exactly the registry
    derivations (they stay patchable module attributes, but at import they
    come from the one table)."""
    assert cb.SLASH_COMMANDS == reg.menu_list()
    assert cb.SLASH_ARG_OPTIONS == reg.arg_options_map()
    assert rp._REPL_COMMANDS == reg.handler_map()
    assert rs._HELP_COMMANDS == reg.help_rows()


def _readme_command_rows() -> set:
    """Every slash-word in the README command table's first column (a cell may
    list aliases and sub-commands: '/reset /clear /new', '/memory save|remember').
    Same parsing as tests/test_docs_parity.py."""
    rows = set()
    in_table = False
    for line in README.splitlines():
        if line.startswith("| command |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            cell = line[1:].replace("\\|", "¦").split("|")[0].replace("`", "")
            rows.update(re.findall(r"/\w+", cell))
    return rows


def test_registry_covers_readme_command_table():
    """The registry covers every command name in the README table (the
    mechanical drift guard: sub-commands resolve to their head command,
    aliases resolve via the alias map)."""
    owners = _alias_owners()
    missing = sorted(w for w in _readme_command_rows() if w not in owners)
    assert not missing, (f"README command table names a command the registry "
                         f"doesn't know: {missing}")


def test_menu_help_dispatch_no_longer_hand_synced():
    """The old display list is gone from _chatbox source: the module builds
    SLASH_COMMANDS from the registry (a grep-able regression pin for the
    consolidation itself)."""
    src = open(cb.__file__, encoding="utf-8").read()
    assert "menu_list()" in src, "SLASH_COMMANDS is not registry-derived anymore"
    assert "arg_options_map()" in src, "SLASH_ARG_OPTIONS is not registry-derived"
    repl_src = open(rp.__file__, encoding="utf-8").read()
    assert "handler_map()" in repl_src, "_REPL_COMMANDS is not registry-derived"
