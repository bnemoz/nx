"""F9: parallel dispatch of all-read-only tool-call turns.

When every call in a turn is a cacheable read-only tool (read_file, search,
git_status, ...), they execute concurrently; mixed/mutating turns keep the
strict sequential contract. Results appear in the model's original order.
"""
import importlib
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR"):
    os.environ.pop(k, None)

import pytest


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


def _make_files(tmp, n=4):
    for i in range(n):
        (tmp / f"f{i}.txt").write_text(f"content {i}\n" + "x" * 200)
    return [str(tmp / f"f{i}.txt") for i in range(n)]


class _ConcurrencyProbe:
    """Records max simultaneous executions of the wrapped fn. Proves actual
    overlap directly, so the test needs no wall-clock bound (which flakes on
    loaded runners while a sequential run can never show overlap anyway)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active = 0
        self.max_concurrent = 0

    def wrap(self, fn):
        def inner(*a, **kw):
            with self._lock:
                self._active += 1
                self.max_concurrent = max(self.max_concurrent, self._active)
            try:
                return fn(*a, **kw)
            finally:
                with self._lock:
                    self._active -= 1
        return inner

    def assert_parallel(self, expected_floor=2):
        assert self.max_concurrent >= expected_floor, (
            f"max concurrent executions {self.max_concurrent} < {expected_floor} "
            "— dispatch ran sequentially")


def test_all_read_only_detector(nx_fresh):
    nx, tmp = nx_fresh
    from nx._turn import _all_read_only
    assert _all_read_only([("read_file", {"path": "a"}), ("git_status", {})]) is True
    assert _all_read_only([]) is False
    assert _all_read_only([("read_file", {}), ("run_bash", {"command": "ls"})]) is False
    assert _all_read_only([("write_file", {"path": "x", "content": "y"})]) is False


def test_parallel_turn_beats_sequential_wallclock(nx_fresh, tmp_path):
    """4 reads must actually overlap: the probe inside the fake read records
    max concurrency — 1 would mean sequential, >=2 proves the fan-out."""
    nx, tmp = nx_fresh
    paths = _make_files(tmp)
    real_read_file = nx.read_file
    probe = _ConcurrencyProbe()

    def slow_read(path, **kw):
        time.sleep(0.4)
        return real_read_file(path, **kw)

    nx._dispatch.DISPATCH["read_file"] = probe.wrap(slow_read)
    try:
        calls = [("read_file", {"path": p}) for p in paths]
        from nx._turn import _run_read_only_parallel
        results = _run_read_only_parallel(calls)
    finally:
        nx._dispatch.DISPATCH["read_file"] = real_read_file
    assert len(results) == 4
    assert all(f"content {i}" in results[i] for i in range(4))  # order preserved
    probe.assert_parallel()


def test_native_path_parallel_and_order(nx_fresh, tmp_path, monkeypatch):
    nx, tmp = nx_fresh
    paths = _make_files(tmp, 3)
    real_read_file = nx.read_file
    probe = _ConcurrencyProbe()

    def slow_read(path, **kw):
        time.sleep(0.3)
        return real_read_file(path, **kw)
    monkeypatch.setitem(nx._dispatch.DISPATCH, "read_file", probe.wrap(slow_read))
    # route native tool_calls through the handler with a fake streamed turn
    tcs = {i: {"id": f"c{i}", "name": "read_file", "args": f'{{"path": "{p}"}}'}
           for i, p in enumerate(paths)}
    messages = [{"role": "system", "content": "s"}]
    from nx._turn import _handle_native_tool_calls
    status = _handle_native_tool_calls(messages, tcs, "", ["tool_calls"], 0)
    assert status == nx.TURN_TOOL
    probe.assert_parallel()
    # assistant tool_calls turn + 3 tool results in order
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 3
    for i, m in enumerate(tool_msgs):
        assert f"content {i}" in m["content"], f"tool result {i} out of order"
    assert tool_msgs[0]["tool_call_id"] == "c0"


def test_mixed_turn_stays_sequential(nx_fresh, tmp_path, monkeypatch):
    nx, tmp = nx_fresh
    order = []
    real_read, real_write = nx._dispatch.DISPATCH["read_file"], nx._dispatch.DISPATCH["write_file"]

    def spy_read(path, **kw):
        order.append(f"read:{path}")
        return real_read(path, **kw)

    def spy_write(path, content, **kw):
        order.append(f"write:{path}")
        return real_write(path, content, **kw)

    monkeypatch.setitem(nx._dispatch.DISPATCH, "read_file", spy_read)
    monkeypatch.setitem(nx._dispatch.DISPATCH, "write_file", spy_write)
    tcs = {0: {"id": "c0", "name": "write_file", "args": '{"path": "w.txt", "content": "x"}'},
           1: {"id": "c1", "name": "read_file", "args": '{"path": "w.txt"}'}}
    messages = [{"role": "system", "content": "s"}]
    from nx._turn import _handle_native_tool_calls
    status = _handle_native_tool_calls(messages, tcs, "", ["tool_calls"], 0)
    assert status == nx.TURN_TOOL
    assert order == ["write:w.txt", "read:w.txt"]  # strictly sequential


def test_text_protocol_parallel_turn(nx_fresh, tmp_path, monkeypatch):
    nx, tmp = nx_fresh
    paths = _make_files(tmp, 3)
    real_read_file = nx._dispatch.DISPATCH["read_file"]
    probe = _ConcurrencyProbe()

    def slow_read(path, **kw):
        time.sleep(0.3)
        return real_read_file(path, **kw)

    monkeypatch.setitem(nx._dispatch.DISPATCH, "read_file", probe.wrap(slow_read))
    calls = [("read_file", {"path": p}) for p in paths]
    from nx._turn import _run_read_only_parallel
    results = _run_read_only_parallel(calls)
    assert all(f"content {i}" in results[i] for i in range(3))
    probe.assert_parallel()


def test_cache_invalidation_still_consistent_after_parallel(nx_fresh, tmp_path):
    nx, tmp = nx_fresh
    p = tmp / "c.txt"
    p.write_text("v1")
    from nx._dispatch import run_tool
    r1 = run_tool("read_file", {"path": str(p)})
    assert "v1" in r1
    # a write invalidates the cache; next read sees v2
    run_tool("write_file", {"path": str(p), "content": "v2"})
    r2 = run_tool("read_file", {"path": str(p)})
    assert "v2" in r2
