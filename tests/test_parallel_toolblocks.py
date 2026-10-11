"""0.31.0: parallel tool calls render as ONE group box, not N interleaved ones.

Before this, a turn with 2+ read-only calls ran run_tool() on a thread pool
non-quiet: every worker printed its own `┌─ name` / args box while its own
LifeSpinner thread cleared lines a sibling had just drawn, producing the
garbled transcript (half-boxes, a bare `running read_file [▪▪··]` row, and N
collapse hints). Workers now run quiet and record a ToolBlock
(nx/_toolblocks); the dispatching thread renders the group.

Run: uv run pytest tests/test_parallel_toolblocks.py -q  (also standalone)
"""
import io
import os
import sys
import threading
import time
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for _k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR",
           "NX_TOOL_OUTPUT_VISIBLE", "NX_PRE_TOOL_HOOK", "NX_POST_TOOL_HOOK"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    monkeypatch.chdir(tmp_path)
    import nx
    yield nx, tmp_path
    nx._state._TOOL_RESULT_CACHE.clear()
    for _k in ("NX_HOME", "NX_SESSIONS_DIR"):
        os.environ.pop(_k, None)


def _files(tmp, n=3):
    out = []
    for i in range(n):
        p = tmp / f"f{i}.txt"
        p.write_text(f"body {i}\n")
        out.append(str(p))
    return out


def _capture(fn):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rv = fn()
    return rv, buf.getvalue()


def test_group_is_one_box_in_call_order(nx_fresh):
    """Three slow reads, faked so they finish OUT OF order: the group prints
    one frame, its calls in the model's original order."""
    nx, tmp = nx_fresh
    paths = _files(tmp, 3)
    real = nx._dispatch.DISPATCH["read_file"]

    def staggered(path, **kw):
        # the LAST call returns first — ordering must come from call order
        time.sleep(0.25 if paths.index(path) == 0 else 0.01)
        return real(path, **kw)

    nx._dispatch.DISPATCH["read_file"] = staggered
    try:
        res, out = _capture(lambda: nx._turn._run_read_only_parallel(
            [("read_file", {"path": p}) for p in paths]))
    finally:
        nx._dispatch.DISPATCH["read_file"] = real

    assert all(f"body {i}" in res[i] for i in range(3))  # results still correct
    plain = _strip(out)
    # The parallel-group box is drawn by nx/_toolblocks.py, which is OUTSIDE
    # this change's file scope, so it still uses the square corner (reported as
    # a Phase B deviation; the sequential tool box, chatbox frame, rail and
    # footer all moved to _render.BOX_*).
    assert plain.count("┌─") == 1                      # exactly one box frame
    assert "3 tool calls in parallel" in plain
    assert plain.count("output collapsed") == 1         # ONE hint, not three
    # call order survives the pool finishing them out of order
    at = [plain.index(f"f{i}.txt") for i in range(3)]
    assert at == sorted(at), f"calls out of order:\n{plain}"
    assert "▪" not in plain and "running read_file" not in plain


def test_expanded_group_bodies_and_error_marker(nx_fresh):
    nx, tmp = nx_fresh
    paths = _files(tmp, 2)
    nx._state.TOOL_OUTPUT_VISIBLE = True
    res, out = _capture(lambda: nx._turn._run_read_only_parallel(
        [("read_file", {"path": paths[0]}),
         ("read_file", {"path": str(tmp / "gone.txt")})]))
    plain = _strip(out)
    assert "body 0" in plain                 # expanded group shows result bodies
    assert "ERROR: FileNotFoundError" in plain
    assert "1 error" in plain                # footer counts the failure
    assert res[1].startswith("ERROR")


def test_cache_hit_shows_marker(nx_fresh):
    """A served-from-cache read used to render NOTHING in a fan-out (and
    looked like a dropped call); it now shows its line with a · cached tag."""
    nx, tmp = nx_fresh
    from nx._dispatch import run_tool
    p = _files(tmp, 1)[0]
    run_tool("read_file", {"path": p})       # warm the cache sequentially
    _, out = _capture(lambda: nx._turn._run_read_only_parallel(
        [("read_file", {"path": p}), ("list_dir", {"path": str(tmp)})]))
    plain = _strip(out)
    assert "cached" in plain
    assert p.rsplit("/", 1)[1] in plain


def test_non_path_calls_keep_their_args(nx_fresh):
    """search_files' pattern (and any non-path arg) must stay visible — the
    header only carries the path."""
    nx, tmp = nx_fresh
    (tmp / "x.py").write_text("needle here\n")
    _, out = _capture(lambda: nx._turn._run_read_only_parallel(
        [("search_files", {"pattern": "needle", "path": str(tmp)}),
         ("list_dir", {"path": str(tmp)})]))
    plain = _strip(out)
    assert "needle" in plain


def test_quiet_caller_renders_nothing(nx_fresh):
    """An async agent / batch task fans out on a quiet thread: results flow
    back to its transcript and the shared terminal stays clean."""
    nx, tmp = nx_fresh
    from nx._subagent import _ASYNC_QUIET_SLOTS
    paths = _files(tmp, 2)
    _ASYNC_QUIET_SLOTS.quiet = True
    try:
        res, out = _capture(lambda: nx._turn._run_read_only_parallel(
            [("read_file", {"path": p}) for p in paths]))
    finally:
        _ASYNC_QUIET_SLOTS.quiet = False
    assert all("body" in res[i] for i in res)
    assert out == ""


def test_group_line_reports_completed_calls_not_submitted(nx_fresh):
    """0.35.0: while a group runs, its line shows elapsed + n/m so the user can
    tell 2-of-3 from 0-of-3. The count must track ACTUAL completions: with a
    slow first call, ex.map would have the line sitting at 0/3 after two calls
    had already landed."""
    nx, tmp = nx_fresh
    paths = _files(tmp, 3)
    real = nx._dispatch.DISPATCH["read_file"]
    seen = {}

    def slow_first(path, **kw):
        time.sleep(0.4 if path == paths[0] else 0.02)
        return real(path, **kw)

    class _Probe:
        """Stands in for the group spinner. Samples its counter from a SEPARATE
        thread: the counter is bumped by the dispatching thread as it consumes
        completed futures, so anything sampled on that thread (including from
        inside start()) would always read 0."""

        def __init__(self, label, counter=None, elapsed=True):
            seen["label"], seen["counter"], seen["elapsed"] = label, counter, elapsed

        def start(self):
            def _sample():
                time.sleep(0.15)     # calls 2 and 3 land well before this
                seen["mid"] = seen["counter"].done
            threading.Thread(target=_sample, daemon=True).start()

        def stop(self):
            pass

    nx._dispatch.DISPATCH["read_file"] = slow_first
    saved = nx._toolblocks.group_spinner
    nx._toolblocks.group_spinner = _Probe
    try:
        nx._turn._run_read_only_parallel(
            [("read_file", {"path": p}) for p in paths])
    finally:
        nx._dispatch.DISPATCH["read_file"] = real
        nx._toolblocks.group_spinner = saved

    assert seen["elapsed"] is True, "the group line stopped tracking elapsed"
    assert seen.get("mid") == 2, (
        f"count did not track real completions (saw {seen.get('mid')}); "
        "ex.map yields in submission order and would report 0 here")
    assert str(seen["counter"]) == "3/3"


def test_prefetch_still_silent(nx_fresh):
    """The mixed-turn prefetch stays invisible even though the group renderer
    now exists (it attaches no recorder)."""
    nx, tmp = nx_fresh
    a, b = _files(tmp, 2)
    msgs = [{"role": "system", "content": "s"}]
    tcs = {0: {"id": "c0", "name": "write_file",
               "args": '{"path": "w.txt", "content": "x"}'},
           1: {"id": "c1", "name": "read_file", "args": f'{{"path": "{a}"}}'},
           2: {"id": "c2", "name": "read_file", "args": f'{{"path": "{b}"}}'}}
    nx.YOLO = True
    try:
        _, out = _capture(lambda: nx._turn._handle_native_tool_calls(
            msgs, tcs, "", ["tool_calls"], 0))
    finally:
        nx.YOLO = False
    plain = _strip(out)
    # the sequential boxes render (write + 2 reads), but never a group frame
    assert "in parallel" not in plain


def _strip(text):
    import re
    return re.sub(r"\033\[[0-?]*[ -/]*[@-~]", "", text)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
