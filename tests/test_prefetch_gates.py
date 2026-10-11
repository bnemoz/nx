"""Prefetch gate parity for mixed tool-call turns (reads + a mutating call).

The mixed-turn prefetch used to call the tool bodies directly from its thread
pool, which executed reads outside run_tool() — no plan gate, no NX_PRE_TOOL_HOOK
veto, no NX_POST_TOOL_HOOK audit record, and an unvetted entry materialized into
the shared st._TOOL_RESULT_CACHE. The prefetch now dispatches through run_tool(),
so it inherits every gate and hook the sequential pass applies.
"""
import importlib
import json
import os
import stat
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR",
          "NX_PRE_TOOL_HOOK", "NX_POST_TOOL_HOOK"):
    os.environ.pop(k, None)

import pytest

import nx


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    monkeypatch.chdir(tmp_path)
    importlib.reload(nx)
    nx.YOLO = True
    nx._state._TOOL_RESULT_CACHE.clear()
    yield nx, tmp_path
    nx.YOLO = False
    nx._state._TOOL_RESULT_CACHE.clear()
    for k in ("NX_HOME", "NX_SESSIONS_DIR", "NX_BASE_URL", "NX_MODEL", "NX_API_KEY",
              "NX_PRE_TOOL_HOOK", "NX_POST_TOOL_HOOK"):
        os.environ.pop(k, None)


def _script(tmp_path, body, name="hook.sh"):
    p = tmp_path / name
    p.write_text(body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _mixed_turn(nx, tmp_path, read_paths, write_path="w.txt"):
    """One native mixed turn (reads + a write_file) through the sequential
    dispatch path — the shape that triggers the prefetch."""
    calls = [{"id": f"c{i}", "name": "read_file", "args": {"path": p}}
             for i, p in enumerate(read_paths)]
    calls.append({"id": f"c{len(read_paths)}", "name": "write_file",
                  "args": {"path": write_path, "content": "x"}})
    tcs = {i: {**c, "args": json.dumps(c["args"])} for i, c in enumerate(calls)}
    messages = [{"role": "system", "content": "s"}]
    from nx._turn import _handle_native_tool_calls
    status = _handle_native_tool_calls(messages, tcs, "", ["tool_calls"], 0)
    assert status == nx.TURN_TOOL
    return messages


def _tool_results(messages):
    return [m["content"] for m in messages if m.get("role") == "tool"]


def test_pre_hook_vetoes_prefetched_reads(nx_fresh):
    """A policy blocking reads of a marker path vetoes them on a mixed turn,
    the file is never read, and nothing lands in the shared cache."""
    nx, tmp = nx_fresh
    from nx import _dispatch
    secret = tmp / "secret.txt"
    secret.write_text("sensitive\n")
    other = tmp / "other.txt"
    other.write_text("fine\n")
    os.environ["NX_PRE_TOOL_HOOK"] = _script(
        tmp,
        "#!/bin/sh\n"
        "payload=$(cat)\n"
        'echo "$payload" | grep -q secret && exit 1\n'
        "exit 0\n",
        name="blocksecret.sh")
    messages = _mixed_turn(nx, tmp, [str(secret), str(other)])
    results = _tool_results(messages)
    assert any("blocked by NX_PRE_TOOL_HOOK" in r for r in results)
    cache = _dispatch.st._TOOL_RESULT_CACHE
    secret_key = ("read_file", json.dumps({"path": str(secret)}))
    assert secret_key not in cache, "vetoed read materialized into the cache"
    # the allowed read went through normally and was cached
    other_key = ("read_file", json.dumps({"path": str(other)}))
    assert other_key in cache
    assert "fine" in cache[other_key].result


def test_post_hook_sees_every_call_in_mixed_turn(nx_fresh):
    """Audit completeness: the post-hook observes all four calls of a mixed
    turn (3 reads + 1 write). Prefetch used to run the first two reads outside
    run_tool(), so the audit trail silently lost exactly those reads."""
    nx, tmp = nx_fresh
    paths = []
    for i in range(3):
        p = tmp / f"f{i}.txt"
        p.write_text(f"content {i}\n")
        paths.append(str(p))
    audit_dir = tmp / "audit"
    audit_dir.mkdir()
    os.environ["NX_POST_TOOL_HOOK"] = _script(
        tmp,
        "#!/bin/sh\n"
        "cat > " + str(audit_dir) + "/$$.json\n",
        name="audit.sh")
    _mixed_turn(nx, tmp, paths)
    tools = []
    read_paths = []
    for f in sorted(audit_dir.glob("*.json")):
        payload = json.loads(f.read_text())
        tools.append(payload["tool"])
        if payload["tool"] == "read_file":
            read_paths.append(payload["args"]["path"])
    # every call of the turn is audited: all 3 read paths + the write
    # (misses and cache hits fire separately, so compare distinct paths)
    assert sorted(set(read_paths)) == sorted(paths)
    assert tools.count("write_file") == 1


def test_prefetched_read_executes_exactly_once(nx_fresh, monkeypatch):
    """No double execution: the prefetch runs the read once and the sequential
    pass must take the cache hit — the read_file body executes exactly once,
    preserving the 'reads stay instant' contract for mixed turns."""
    nx, tmp = nx_fresh
    from nx import _dispatch
    f = tmp / "once.txt"
    f.write_text("only once\n")
    g = tmp / "twice_never.txt"
    g.write_text("support read\n")
    body_calls = []
    real_read = _dispatch.DISPATCH["read_file"]

    def counting_read(path, **kw):
        body_calls.append(path)
        return real_read(path, **kw)

    monkeypatch.setitem(_dispatch.DISPATCH, "read_file", counting_read)
    _mixed_turn(nx, tmp, [str(f), str(g)])
    # each distinct read ran its body exactly once — the sequential pass hit
    # the cache rather than re-executing (double execution would double-count)
    assert body_calls.count(str(f)) == 1, f"read body executed {len(body_calls)} times"
    assert body_calls.count(str(g)) == 1


def test_prefetch_quiet_in_quiet_slot(nx_fresh, capsys):
    """With the quiet slot set (async agent / batch worker), the prefetch emits
    nothing to stdout: no tool box, no spinner line, no pre-hook veto line —
    pool threads don't inherit thread-local quiet, so it must be re-applied."""
    nx, tmp = nx_fresh
    from nx._subagent import _ASYNC_QUIET_SLOTS
    a = tmp / "qa.txt"
    a.write_text("qa\n")
    b = tmp / "qb.txt"
    b.write_text("qb\n")
    # a vetoing pre-hook makes the pool thread's would-be printed line obvious
    os.environ["NX_PRE_TOOL_HOOK"] = _script(
        tmp, "#!/bin/sh\ncat >/dev/null\necho policy >&2\nexit 1\n",
        name="blockall.sh")

    captured = {}
    real = nx._turn._prefetch_cacheable_tools

    def spy(ordered, parsed_args):
        captured["slots"] = bool(getattr(_ASYNC_QUIET_SLOTS, "quiet", False))
        return real(ordered, parsed_args)

    def quiet_caller():
        _ASYNC_QUIET_SLOTS.quiet = True
        try:
            spy([{"name": "read_file"}, {"name": "read_file"}],
                [{"path": str(a)}, {"path": str(b)}])
        finally:
            _ASYNC_QUIET_SLOTS.quiet = False

    t = threading.Thread(target=quiet_caller)
    t.start()
    t.join()
    assert captured["slots"] is True
    out = capsys.readouterr().out
    assert "policy" not in out
    assert "╭─" not in out and "╰─" not in out


def test_prefetch_gates_plan_writes_not_reads(nx_fresh):
    """Reads stay allowed under an active plan (only PLAN_GATE_TOOLS are
    gated), so a mixed turn inside plan mode still prefetches them."""
    nx, tmp = nx_fresh
    from nx import _dispatch
    a = tmp / "pa.txt"
    a.write_text("pa\n")
    b = tmp / "pb.txt"
    b.write_text("pb\n")
    nx._state._PLAN_ACTIVE = True
    nx._state._PLAN_GOAL = "test"
    nx._state._PLAN_SLUG = "test"
    try:
        messages = _mixed_turn(nx, tmp, [str(a), str(b)])
    finally:
        nx._state._PLAN_ACTIVE = False
    cache = _dispatch.st._TOOL_RESULT_CACHE
    keys = list(cache)
    read_keys = [k for k in keys if k[0] == "read_file"]
    assert len(read_keys) == 2
    assert "Plan mode is active" in _tool_results(messages)[2]  # the write is gated
