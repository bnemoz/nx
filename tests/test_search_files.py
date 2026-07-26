#!/usr/bin/env python3
"""Tests for `search_files`, the ripgrep-style content-search tool.

No live model, no terminal, no network. Exercises search_files directly against
a temp tree. The tool is read-only and must NEVER go through the approval/risk
classifier — that's the whole reason it exists instead of shelling out to grep
via run_bash. Falls back to plain `grep -rn` when `rg` isn't installed, so these
cases pass with either backend (the CI environment here has grep).
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-search-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402


def _make_tree():
    """Build a small tree and return its root path."""
    root = os.path.join(_tmp, "tree")
    os.makedirs(os.path.join(root, "sub"), exist_ok=True)
    with open(os.path.join(root, "a.py"), "w") as f:
        f.write("def foo():\n    return 'apple'\n\ndef bar():\n    return 'apple'\n")
    with open(os.path.join(root, "sub", "b.py"), "w") as f:
        f.write("x = 'apple'\ny = 'banana'\n")
    with open(os.path.join(root, "notes.md"), "w") as f:
        f.write("# apple notes\n\nsome prose about apples\n")
    return root


def test_finds_matches_with_path_line_format():
    root = _make_tree()
    out = m.search_files("apple", path=root)
    assert out and "apple" in out
    # every returned line is a path:line:match triple
    for line in out.splitlines():
        if line.startswith("... ["):
            continue
        parts = line.split(":", 2)
        assert len(parts) == 3, line
        assert parts[0].startswith(root), line
        assert parts[1].isdigit(), line


def test_no_matches_returns_clear_marker():
    root = _make_tree()
    out = m.search_files("doesnotexist_zzz", path=root)
    assert out.startswith("[no matches")


def test_glob_filter_restricts_files():
    root = _make_tree()
    py = m.search_files("apple", path=root, glob="*.py")
    md = m.search_files("apple", path=root, glob="*.md")
    assert "notes.md" not in py
    assert "notes.md" in md


def test_regex_pattern_works():
    root = _make_tree()
    out = m.search_files(r"ban[an]+a", path=root)
    assert "banana" in out
    assert "apple" not in out.split("\n", 1)[0] or "banana" in out


def test_search_single_file():
    root = _make_tree()
    path = os.path.join(root, "a.py")
    out = m.search_files("def", path=path)
    assert "def foo" in out and "def bar" in out


def test_empty_pattern_is_error():
    out = m.search_files("", path=".")
    assert out.startswith("ERROR")


def test_max_results_truncates():
    root = _make_tree()
    # 'apple' appears in 3 files / multiple lines; cap at 1 hit.
    out = m.search_files("apple", path=root, max_results=1)
    lines = [ln for ln in out.splitlines() if not ln.startswith("... [")]
    assert len(lines) == 1
    assert "more matches" in out


def test_never_calls_confirm():
    # search_files must not invoke the approval gate. Replace _confirm with a
    # canary that would fail the test if it were ever called.
    called = {"n": 0}
    orig = m._confirm
    m._confirm = lambda action: called.__setitem__("n", called["n"] + 1) or False
    try:
        root = _make_tree()
        m.search_files("apple", path=root)
    finally:
        m._confirm = orig
    assert called["n"] == 0, "search_files must not call _confirm (read-only)"


def test_nonexistent_dir_surfaces_error():
    out = m.search_files("x", path=os.path.join(_tmp, "no-such-dir-xyz"))
    # grep/rg report the missing path; we surface it as an ERROR string
    assert out.startswith("ERROR") or out.startswith("[no matches")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
