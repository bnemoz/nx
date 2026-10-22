#!/usr/bin/env python3
"""Docs/touchpoint parity (0.42, roadmap 1.5).

CONTRIBUTING.md documents four hand-synced command lists; drift between them
was live for releases (`/toolcalls` missing from `/help`, `/resmon` missing
from the completion dropdown and the README, a `/ctx all` hint with no
implementing branch). These tests fail on the drift instead of grepping for
it by hand.
"""
import os
import re
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _chatbox as cb  # noqa: E402
from nx import _repl_settings_cmds as rs  # noqa: E402
# nx/__init__ binds a function named `_repl` into its namespace (shadowing the
# submodule attribute), so the module must come from sys.modules.
rp = sys.modules["nx._repl"]


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The user-facing README is intentionally thin (quick start + pointers); the
# complete command/env tables live in docs/REFERENCE.md — parity checks run
# against that file, which is where `nx --dump-commands` regenerates them.
REFERENCE = open(os.path.join(REPO_ROOT, "docs", "REFERENCE.md"),
                 encoding="utf-8").read()


def _dispatch_keys() -> set[str]:
    return set(rp._REPL_COMMANDS)


def _help_names() -> set[str]:
    """Commands named by /help rows: the leading slash-word of each
    comma-separated segment. A sub-command row (approval allow, for example)
    counts as its head command."""
    names = set()
    for cmd, _ in rs._HELP_COMMANDS:
        for seg in cmd.split(","):
            mm = re.match(r"\s*(/\w+)", seg)
            if mm:
                names.add(mm.group(1))
    return names


def _menu_names() -> set[str]:
    return {cmd for cmd, _ in cb.SLASH_COMMANDS}


def test_every_dispatch_key_appears_in_help():
    missing = _dispatch_keys() - _help_names()
    assert not missing, f"dispatched but absent from /help: {sorted(missing)}"


def test_every_help_row_is_dispatchable():
    unknown = _help_names() - _dispatch_keys()
    assert not unknown, f"/help advertises a command nothing dispatches: {sorted(unknown)}"


def test_menu_never_advertises_a_dead_command():
    unknown = _menu_names() - _dispatch_keys()
    assert not unknown, f"completion menu offers a command nothing dispatches: {sorted(unknown)}"


def test_arg_options_only_for_dispatched_commands():
    stray = set(cb.SLASH_ARG_OPTIONS) - _dispatch_keys()
    assert not stray, f"arg dropdown for an undispatched command: {sorted(stray)}"


def _readme_command_rows() -> set[str]:
    """Every slash-word in the REFERENCE command table's first column (a cell
    may list aliases and sub-commands: '/reset /clear /new',
    '/memory save|remember')."""
    rows = set()
    in_table = False
    for line in REFERENCE.splitlines():
        if line.startswith("| command |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            cell = line[1:].replace("\\|", "¦").split("|")[0].replace("`", "")
            rows.update(re.findall(r"/\w+", cell))
    return rows


def test_every_dispatched_command_is_in_the_reference_table():
    """Aliases live on their primary's row (they render inside its usage
    label), so only non-alias dispatch keys need a row of their own."""
    from nx import _commands as reg
    alias_names = {a[0] for spec in reg.COMMANDS.values() for a in spec.aliases}
    missing = _dispatch_keys() - alias_names - _readme_command_rows()
    assert not missing, f"dispatched but absent from the REFERENCE command table: {sorted(missing)}"


def test_generated_commands_table_matches_the_registry():
    """The docs/REFERENCE.md command block is machine-generated
    (`nx --dump-commands`): regenerate it and it must match the registry
    byte-for-byte."""
    from nx import _cli
    expected = _cli._commands_markdown()
    mm = re.search(
        r"<!-- BEGIN GENERATED COMMANDS[^\n]*-->\n(.*?)\n<!-- END GENERATED COMMANDS -->",
        REFERENCE, re.DOTALL)
    assert mm, "REFERENCE generated-commands markers are missing"
    assert mm.group(1) == expected, (
        "docs/REFERENCE.md command table is stale — run `nx --dump-commands` and paste the "
        "output between the GENERATED COMMANDS markers")


def test_ctx_all_is_implemented_not_just_hinted():
    """The hint at the bottom of the compact /context report must point at a real
    branch (1.5: it advertised /ctx all while nothing read the argument)."""
    src = open(rs.__file__, encoding="utf-8").read()
    assert 'sub == "all"' in src, "/context all branch is gone — fix the hint or the branch"
    # and the hint still matches the branch
    assert "/context all" in src


def _documented_env_names() -> set[str]:
    """NX_* names in the REFERENCE env table, minus placeholders/families."""
    names = set(re.findall(r"NX_[A-Z][A-Z0-9_]*", REFERENCE))
    # <NAME> placeholders are conventions, not variables; a trailing
    # underscore is the regex biting `NX_PRICE_<SOURCE>_INPUT`.
    names = {n for n in names if "<" not in n and not n.endswith("_")}
    # Families resolved dynamically from a prefix, pinned separately below:
    # role colors (NX_<ROLE>_COLOR) and per-source settings (NX_SOURCE_<NAME>_*).
    return {n for n in names
            if not n.startswith("NX_SOURCE_") and not n.endswith("_COLOR")}


def _nx_source_blob() -> str:
    """Concatenated nx/ source only. A test's monkeypatch.setenv("NX_X", ...)
    does NOT count as reading NX_X — the product must read what the docs
    promise — so tests/ is deliberately excluded from this walk."""
    blob = ""
    for root, _dirs, files in os.walk(os.path.join(REPO_ROOT, "nx")):
        for f in files:
            if f.endswith(".py"):
                blob += open(os.path.join(root, f), encoding="utf-8").read()
    return blob


def test_every_documented_env_var_is_actually_read():
    """Every NX_* var named in the REFERENCE env table must be read somewhere
    in nx/ (the NX_TEMPERATURE/NX_TOP_P failure mode: documented, never
    read). Table rows may document several vars on one line. The pricing
    family (NX_PRICE_*) is resolved dynamically from a prefix in nx/_cost.py
    — documented per-model instances have no literal reader, so they're out
    of scope here (the family mechanism is pinned by
    test_dynamic_env_families_have_a_real_reader-style checks)."""
    blob = _nx_source_blob()
    missing = [n for n in sorted(_documented_env_names())
               if not n.startswith("NX_PRICE_")
               and not re.search(rf'["\']{n}["\']', blob)]
    assert not missing, f"documented in REFERENCE but read nowhere in nx/: {missing}"


def test_forward_scan_flags_a_documented_var_with_no_nx_reader(tmp_path):
    """Anti-vacuity pin (durability audit P2): the forward scan must run
    against nx/ ONLY. Run its exact detection logic over a tree that mimics
    the old bug — a planted var that a tests/ file reads (monkeypatch.setenv)
    but nx/ never does — and prove the scan still flags it as missing."""
    planted = "NX_PLANTED_TEST_ONLY_VAR"
    documented = _documented_env_names() | {planted}
    # a tests/ file "reads" it (as via monkeypatch.setenv); nx/ does not:
    test_tree = tmp_path / "tests"
    test_tree.mkdir()
    (test_tree / "fake_test.py").write_text(
        f'monkeypatch.setenv("{planted}", "1")\n', encoding="utf-8")
    test_blob = "".join(
        open(os.path.join(root, f), encoding="utf-8").read()
        for root, _d, files in os.walk(test_tree)
        for f in files if f.endswith(".py"))
    # with the tests/ blob included, the planted name would be "found":
    assert re.search(rf'["\']{planted}["\']', test_blob)
    # ...but the real scan walks nx/ only, so the name stays missing:
    blob = _nx_source_blob()
    assert not re.search(rf'["\']{planted}["\']', blob)
    missing = [n for n in sorted(documented)
               if not n.startswith("NX_PRICE_")
               and not re.search(rf'["\']{n}["\']', blob)]
    assert planted in missing


def test_dynamic_env_families_have_a_real_reader():
    """The two README families that can't be named literally (per-source
    NX_SOURCE_<NAME>_* and role colors NX_<ROLE>_COLOR) must each have a
    dynamic reader — pinned here so deleting the mechanism fails, not just
    the literal-name scan."""
    from nx import _ansi
    assert hasattr(_ansi, "ROLE_DEFAULTS") and _ansi.ROLE_DEFAULTS
    assert hasattr(_ansi, "_parse_color_override")
    src = open(os.path.join(REPO_ROOT, "nx", "_sources.py"), encoding="utf-8").read()
    assert "NX_SOURCE_" in src


def test_sampling_params_reach_the_request_payload(monkeypatch):
    """NX_TEMPERATURE/NX_TOP_P are wired, not just documented (1.5)."""
    m.st.TEMPERATURE = 0.0
    m.st.TOP_P = -1.0
    m.st.MODEL = "fake-model"
    captured = {}

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            return iter([])

    class _Cp:
        def create(self, **k):
            captured.update(k)
            return _Stream()

    class _Ch:
        completions = _Cp()

    class _Client:
        chat = _Ch()

    m.st.client = _Client()
    saved_active = m.st.ACTIVE
    m.st.ACTIVE = None
    try:
        m.open_stream([{"role": "user", "content": "hi"}])
    finally:
        m.st.ACTIVE = saved_active
        m.st.TEMPERATURE = -1.0
    assert captured.get("temperature") == 0.0
    assert "top_p" not in captured

    m.st.TEMPERATURE = -1.0
    m.st.TOP_P = 0.9
    captured.clear()
    try:
        m.open_stream([{"role": "user", "content": "hi"}])
    finally:
        m.st.TOP_P = -1.0
    assert captured.get("top_p") == 0.9
    assert "temperature" not in captured


def test_negative_sampling_values_omit_the_field(monkeypatch):
    m.st.TEMPERATURE = -1.0
    m.st.TOP_P = -1.0
    m.st.MODEL = "fake-model"
    captured = {}

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            return iter([])

    class _Cp:
        def create(self, **k):
            captured.update(k)
            return _Stream()

    class _Ch:
        completions = _Cp()

    class _Client:
        chat = _Ch()

    m.st.client = _Client()
    saved_active = m.st.ACTIVE
    m.st.ACTIVE = None
    try:
        m.open_stream([{"role": "user", "content": "hi"}])
    finally:
        m.st.ACTIVE = saved_active
    assert "temperature" not in captured
    assert "top_p" not in captured
