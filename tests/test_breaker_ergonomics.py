#!/usr/bin/env python3
"""0.29.9: breaker ergonomics.

- _malformed_args_nudge derives its fallback suggestion from the dropped tool
  (FALLBACK_FOR), not a hardcoded edit_file/write_file line.
- The breaker decays: one clean dispatch of the tool clears its streak and
  re-arms it (NX_MALFORMED_TOOL_BREAKER default now 3).
- /toolcalls shows streaks; /toolcalls reset [tool] re-arms.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY"):
    os.environ.pop(k, None)

import nx as m  # noqa: E402
from nx._dispatch import run_tool, _breaker_decay  # noqa: E402
from nx._turn import _malformed_args_nudge, FALLBACK_FOR  # noqa: E402


def test_nudge_fallback_map_covers_edit_siblings():
    assert FALLBACK_FOR["multi_edit"] == "edit_file"
    # A stale apply_patch name in the breaker still nudges somewhere useful:
    # edit_file's DIFF mode (not plain edit_file, which would invite a
    # literal-mode call).
    assert FALLBACK_FOR["apply_patch"].startswith("edit_file")
    assert "diff" in FALLBACK_FOR["apply_patch"]
    assert FALLBACK_FOR["read_image"] == "read_file"


def test_nudge_for_dropped_edit_file_names_a_real_alternative():
    out = _malformed_args_nudge("edit_file", broken_now=True)
    assert "edit_file has now been removed" in out
    assert "edit_file calls" not in out  # the old hardcoded line
    assert "use one of:" in out  # degrade: live remaining tools
    assert "read_file" in out.split("use one of:")[1]


def test_nudge_for_multi_edit_still_says_edit_file():
    out = _malformed_args_nudge("multi_edit", broken_now=True)
    assert "use edit_file for this step instead" in out


def test_breaker_default_is_three():
    assert m.MALFORMED_TOOL_BREAKER_LIMIT == 3


def test_clean_dispatch_clears_the_streak():
    st = m._state
    st._MALFORMED_TOOL_FAILURES["read_file"] = 2
    try:
        run_tool("read_file", {"path": "README.md"})
        assert st._MALFORMED_TOOL_FAILURES.get("read_file") is None
    finally:
        st._MALFORMED_TOOL_FAILURES.pop("read_file", None)


def test_broken_tool_stays_refused_until_reset():
    """A broken tool is refused before dispatch (0.28.1), so it can't earn its
    own re-arm — the streak decays only for tools still being offered; a
    broken one waits for /toolcalls reset (or /reset)."""
    st = m._state
    st._MALFORMED_TOOL_FAILURES["list_dir"] = 3
    st._MALFORMED_TOOL_BROKEN.add("list_dir")
    try:
        out = run_tool("list_dir", {"path": "."})
        assert "removed from the tool" in out
        assert "list_dir" in st._MALFORMED_TOOL_BROKEN  # refusal is not a clean dispatch
    finally:
        st._MALFORMED_TOOL_FAILURES.pop("list_dir", None)
        st._MALFORMED_TOOL_BROKEN.discard("list_dir")


def test_breaker_decay_helper_rearms():
    """_breaker_decay is what /toolcalls reset runs: pop the streak, drop the
    broken mark — the next schema list offers the tool again."""
    st = m._state
    st._MALFORMED_TOOL_FAILURES["list_dir"] = 3
    st._MALFORMED_TOOL_BROKEN.add("list_dir")
    try:
        _breaker_decay("list_dir")
        assert "list_dir" not in st._MALFORMED_TOOL_BROKEN
        assert "list_dir" not in st._MALFORMED_TOOL_FAILURES
    finally:
        st._MALFORMED_TOOL_FAILURES.pop("list_dir", None)
        st._MALFORMED_TOOL_BROKEN.discard("list_dir")


def test_errored_dispatch_does_not_decay():
    st = m._state
    st._MALFORMED_TOOL_FAILURES["read_file"] = 2
    try:
        run_tool("read_file", {"path": "/definitely/not/here/xyz"})
        # the call errored (file missing) — the streak survives
        assert st._MALFORMED_TOOL_FAILURES.get("read_file") == 2
    finally:
        st._MALFORMED_TOOL_FAILURES.pop("read_file", None)


def test_cache_hit_counts_as_clean_dispatch():
    """A cache-hit read is still a successful dispatch of that tool: the
    miss decays the streak, and the hit path runs the decay too (a no-op by
    then, but the ordering is pinned so a cache hit never resurrects)."""
    st = m._state
    st._TOOL_RESULT_CACHE.clear()
    st._MALFORMED_TOOL_FAILURES["git_status"] = 2
    try:
        run_tool("git_status", {"path": "."})   # miss: populates + decays
        run_tool("git_status", {"path": "."})   # hit: decay runs again
        assert st._MALFORMED_TOOL_FAILURES.get("git_status") is None
    finally:
        st._MALFORMED_TOOL_FAILURES.pop("git_status", None)
        st._TOOL_RESULT_CACHE.clear()


def test_breaker_check_beats_decay_but_a_reoffer_runs():
    """run_tool refuses a broken tool BEFORE dispatch (so the model can't use
    it), but a direct re-offer via /toolcalls reset lets the next call through
    and that clean dispatch keeps it armed-in."""
    st = m._state
    st._MALFORMED_TOOL_FAILURES["list_dir"] = 3
    st._MALFORMED_TOOL_BROKEN.add("list_dir")
    try:
        out = run_tool("list_dir", {"path": "."})
        assert "removed from the tool" in out          # refused while broken
        assert "list_dir" in st._MALFORMED_TOOL_BROKEN  # and still broken
        _breaker_decay("list_dir")                      # what /toolcalls reset does
        out2 = run_tool("list_dir", {"path": "."})
        assert not out2.startswith("ERROR: list_dir is not available")
    finally:
        st._MALFORMED_TOOL_FAILURES.pop("list_dir", None)
        st._MALFORMED_TOOL_BROKEN.discard("list_dir")


def test_toolcalls_cmd_shows_and_resets(capsys):
    from nx._repl_settings_cmds import _cmd_toolcalls

    class _S:  # minimal _ReplState stand-in
        messages = []

    st = m._state
    st._MALFORMED_TOOL_FAILURES = {"multi_edit": 2, "read_file": 1}
    st._MALFORMED_TOOL_BROKEN = {"multi_edit"}
    try:
        _cmd_toolcalls("", _S())
        out = capsys.readouterr().out
        assert "multi_edit: 2" in out and "[broken" in out
        assert "read_file: 1" in out
        _cmd_toolcalls("reset multi_edit", _S())
        capsys.readouterr()
        assert "multi_edit" not in st._MALFORMED_TOOL_BROKEN
        assert "multi_edit" not in st._MALFORMED_TOOL_FAILURES
        assert st._MALFORMED_TOOL_FAILURES == {"read_file": 1}  # others untouched
        _cmd_toolcalls("reset", _S())
        capsys.readouterr()
        assert st._MALFORMED_TOOL_FAILURES == {}
        assert st._MALFORMED_TOOL_BROKEN == set()
        _cmd_toolcalls("", _S())
        assert "no malformed-args failures" in capsys.readouterr().out
    finally:
        st._MALFORMED_TOOL_FAILURES = {}
        st._MALFORMED_TOOL_BROKEN = set()


def test_toolcalls_registered_everywhere():
    from nx._repl import _REPL_COMMANDS
    from nx._chatbox import SLASH_COMMANDS, SLASH_ARG_OPTIONS
    assert _REPL_COMMANDS.get("/toolcalls") is not None
    assert any(c[0] == "/toolcalls" for c in SLASH_COMMANDS)
    assert SLASH_ARG_OPTIONS.get("/toolcalls") == ["reset", "text", "native", "auto"]
