"""Regression tests: mtime-aware read-result cache.

run_bash previously cleared st._TOOL_RESULT_CACHE wholesale, so any shell call
(banner `ls`, git, a build) forced every cached read_file/list_dir to re-execute
for the rest of the session — re-pay tokens and wall-clock. The cache now stores
os.stat().st_mtime_ns with each entry and re-stats on hit: a hit whose mtime is
unchanged survives bash; only changed files are dropped. That also fixes a
staleness bug for files edited outside nx (user's editor) mid-session.
"""
import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR"):
    os.environ.pop(k, None)

import pytest  # noqa: E402 — repo-root sys.path insert must come first


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    nx.YOLO = True
    yield nx, tmp_path
    nx.YOLO = False
    nx._state._TOOL_RESULT_CACHE.clear()
    nx._state._SESSION_HAS_BACKGROUNDED = False
    nx._state._PENDING_BACKGROUND_NOTICE = None
    for k in ("NX_HOME", "NX_SESSIONS_DIR"):
        os.environ.pop(k, None)


def _stat_calls(monkeypatch, nx, counter):
    real_stat = os.stat

    def counting_stat(p, *a, **kw):
        counter[0] += 1
        return real_stat(p, *a, **kw)

    monkeypatch.setattr(os, "stat", counting_stat)
    return counter


def test_bash_keeps_unchanged_entries_drops_changed(nx_fresh):
    """After run_bash, a read_file whose file is untouched stays cached; one
    whose file changed (as by bash touching it) is dropped and re-read."""
    nx, tmp = nx_fresh
    keep = tmp / "keep.txt"
    keep.write_text("keep me\n")
    change = tmp / "change.txt"
    change.write_text("before\n")
    from nx._dispatch import run_tool, _tool_cache_key
    from nx import _state as st

    r1 = run_tool("read_file", {"path": str(keep)})
    assert "keep me" in r1
    r2 = run_tool("read_file", {"path": str(change)})
    assert "before" in r2
    keep_key = _tool_cache_key("read_file", {"path": str(keep)})
    change_key = _tool_cache_key("read_file", {"path": str(change)})
    assert keep_key in st._TOOL_RESULT_CACHE and change_key in st._TOOL_RESULT_CACHE

    # change.txt changes on disk "behind nx's back" (editor-like write); wait
    # out filesystem timestamp granularity so the mtime really differs
    time.sleep(0.01)
    change.write_text("after\n")
    entry = st._TOOL_RESULT_CACHE[_tool_cache_key("read_file", {"path": str(change)})]
    for _ in range(200):
        if os.stat(str(change)).st_mtime_ns != entry.mtime_ns:
            break
        time.sleep(0.005)
    assert os.stat(str(change)).st_mtime_ns != entry.mtime_ns, "test void: mtime did not change"

    # a trivial bash call drops ONLY the changed entry, not the whole cache
    run_tool("run_bash", {"command": "true"})
    assert keep_key in st._TOOL_RESULT_CACHE, "run_bash dropped an unchanged entry"
    assert change_key not in st._TOOL_RESULT_CACHE, "run_bash kept a changed (stale) entry"

    # the unchanged file still returns the cached string...
    assert st._TOOL_RESULT_CACHE[keep_key].result == r1
    # ...and the changed file re-reads fresh content
    fresh = run_tool("read_file", {"path": str(change)})
    assert "after" in fresh
    assert fresh != r2


def test_cache_hit_after_bash_does_not_reexecute_tool(nx_fresh, monkeypatch):
    """Prove the post-bash hit returns the CACHED string without re-running
    the underlying read_file (the dispatched fn would count 1)."""
    nx, tmp = nx_fresh
    f = tmp / "f.txt"
    f.write_text("stable\n")
    import nx._dispatch as disp
    from nx import _state as st

    first = disp.run_tool("read_file", {"path": str(f)})
    assert "stable" in first

    calls = []
    real_read = disp.DISPATCH["read_file"]

    def counting_read(path, **kw):
        calls.append(path)
        return real_read(path, **kw)

    monkeypatch.setitem(disp.DISPATCH, "read_file", counting_read)
    assert disp.run_tool("read_file", {"path": str(f)}) == first
    assert calls == [], "cache hit re-executed the read"

    # the sweep re-stats each entry exactly once (1 entry in cache)
    counter = [0]
    _stat_calls(monkeypatch, nx, counter)
    disp.run_tool("run_bash", {"command": "true"})
    entry = st._TOOL_RESULT_CACHE[("read_file", '{"path": "' + str(f) + '"}')]
    assert entry in st._TOOL_RESULT_CACHE.values()
    monkeypatch.undo()

    # post-bash read of the untouched file is still a cache hit
    calls.clear()
    monkeypatch.setitem(disp.DISPATCH, "read_file", counting_read)
    assert disp.run_tool("read_file", {"path": str(f)}) == first
    assert calls == [], "post-bash read re-executed for an unchanged file"


def test_mtime_change_invalidates_even_without_bash(nx_fresh):
    """Staleness fix: a cache hit re-stats, so an external editor's write to a
    cached file is visible on the next read even with no run_bash in between."""
    nx, tmp = nx_fresh
    f = tmp / "f.txt"
    f.write_text("v1\n")
    from nx._dispatch import run_tool

    first = run_tool("read_file", {"path": str(f)})
    assert "v1" in first
    time.sleep(0.01)
    f.write_text("v2\n")
    for _ in range(200):
        if os.stat(str(f)).st_mtime_ns > 0:
            break
        time.sleep(0.005)
    second = run_tool("read_file", {"path": str(f)})
    assert "v2" in second, "stale cache entry served after external edit"


def test_write_file_invalidates_its_own_path(nx_fresh):
    nx, tmp = nx_fresh
    f = tmp / "f.txt"
    f.write_text("orig\n")
    from nx._dispatch import run_tool

    run_tool("read_file", {"path": str(f)})
    run_tool("write_file", {"path": str(f), "content": "new\n"})
    assert all(e.path != os.path.abspath(str(f)) for e in nx._state._TOOL_RESULT_CACHE.values())
    assert "new" in run_tool("read_file", {"path": str(f)})


def test_prefetch_and_store_shape(nx_fresh):
    """A mixed turn (2 reads + a write) stores the same mtime-aware entry
    shape the all-read-only path does — the prefetch dispatches through
    run_tool(), which is the cache's single writer."""
    nx, tmp = nx_fresh
    paths = []
    for tag in ("a", "b"):
        p = tmp / f"{tag}.txt"
        p.write_text(f"{tag.upper()}\n")
        paths.append(p)
    calls = [{"id": f"c{i}", "name": "read_file", "args": {"path": str(p)}}
             for i, p in enumerate(paths)]
    calls.append({"id": "c2", "name": "write_file",
                  "args": {"path": "w.txt", "content": "x"}})
    tcs = {i: {**c, "args": json.dumps(c["args"])} for i, c in enumerate(calls)}
    messages = [{"role": "system", "content": "s"}]
    from nx._turn import _handle_native_tool_calls
    status = _handle_native_tool_calls(messages, tcs, "", ["tool_calls"], 0)
    assert status == nx.TURN_TOOL
    cache = nx._state._TOOL_RESULT_CACHE
    assert len(cache) == 2
    for entry in cache.values():
        assert entry.result
        assert os.path.isabs(entry.path)
        assert entry.mtime_ns == os.stat(entry.path).st_mtime_ns


def test_delete_file_drops_its_entry(nx_fresh):
    nx, tmp = nx_fresh
    f = tmp / "f.txt"
    f.write_text("x\n")
    from nx._dispatch import run_tool

    run_tool("read_file", {"path": str(f)})
    assert len(nx._state._TOOL_RESULT_CACHE) == 1
    run_tool("delete_file", {"path": str(f)})
    assert len(nx._state._TOOL_RESULT_CACHE) == 0


def test_memory_search_entry_survives_bash(nx_fresh, monkeypatch):
    """A cached memory_search entry survives run_bash when nothing it reflects
    changed. memory_search has no path arg, so its entry is anchored at the
    cwd — and pytest's tmp_path directory mtime moves when pytest's own
    bookkeeping (or the .pytest cache) touches it, which would flakily drop
    the entry. To test exactly what this regression is about (bash must not
    clear an untouched entry), pin the anchor to a directory nothing else
    writes to."""
    nx, tmp = nx_fresh
    from nx._dispatch import run_tool, _tool_cache_key
    from nx import _state as st

    anchor = tmp / "stable-anchor"
    anchor.mkdir()
    real_abspath = os.path.abspath

    def pinned_abspath(p):
        # memory_search has no path arg -> cache_store_entry anchors it at
        # abspath("."). Pin "." to a directory pytest never touches so the
        # test exercises exactly the regression (bash must not drop an
        # untouched entry) instead of pytest's own tmp-dir churn.
        return str(anchor) if p == "." else real_abspath(p)

    monkeypatch.setattr(os.path, "abspath", pinned_abspath)

    out = run_tool("memory_search", {"query": "nothing-matches-this-query"})
    key = _tool_cache_key("memory_search", {"query": "nothing-matches-this-query"})
    assert key in st._TOOL_RESULT_CACHE
    assert st._TOOL_RESULT_CACHE[key].path == str(anchor)
    run_tool("run_bash", {"command": "true"})
    assert key in st._TOOL_RESULT_CACHE, "bash dropped an unchanged non-file entry"
    assert run_tool("memory_search", {"query": "nothing-matches-this-query"}) == out
