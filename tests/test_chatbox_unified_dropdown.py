#!/usr/bin/env python3
"""All dropdowns share the model-dropdown style (0.60.0).

Previously only the /source model dropdown scrolled; slash, arg and @file
dropdowns were hard-truncated (`[:6]` / `[:8]`), so longer lists were simply
unreachable. The match functions must now return the full candidate list and
let the shared `_dropdown_window` scroll.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-unified-dropdown-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
for k in list(os.environ):
    if k.startswith("NX_"):
        del os.environ[k]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nx import _chatbox  # noqa: E402


def test_slash_command_matches_are_not_truncated():
    got = _chatbox._match_slash_commands("/")
    assert got == _chatbox.SLASH_COMMANDS
    # A prefix narrows but never caps.
    assert len(_chatbox._match_slash_commands("/t")) <= len(got)


def test_slash_arg_matches_are_not_truncated():
    got = _chatbox._match_slash_args("/memory", "")
    assert got == [(o, "") for o in _chatbox.SLASH_ARG_OPTIONS["/memory"]]


def test_file_matches_are_not_truncated():
    # A directory with more than the old 8-entry cap, inside a temp "project"
    # with a git repo so _project_root resolves there.
    import subprocess
    d = os.path.join(_tmp, "many")
    os.makedirs(d, exist_ok=True)
    for i in range(15):
        open(os.path.join(d, f"file-{i:02d}.txt"), "w").close()
    subprocess.run(["git", "init", "-q", _tmp], check=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        _chatbox._PROJECT_ROOT_CACHE = None
        _chatbox._TREE_CACHE = None
        _chatbox._TREE_FILTER_CACHE = None
        _chatbox._TREE_RESULT_CACHE = None
        got = _chatbox._match_file_paths("file-1")
    finally:
        os.chdir(cwd)
        _chatbox._PROJECT_ROOT_CACHE = None
        _chatbox._TREE_CACHE = None
        _chatbox._TREE_FILTER_CACHE = None
        _chatbox._TREE_RESULT_CACHE = None
    # file-10 .. file-14 + the recursive basename hits, uncapped.
    assert len([g for g in got if g.startswith("many/")]) >= 5


if __name__ == "__main__":
    for _name in [n for n in sorted(globals()) if n.startswith("test_")]:
        globals()[_name]()
        print(f"  PASS {_name}")
    print("ALL PASS")
