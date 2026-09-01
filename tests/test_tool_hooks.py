"""F5: NX_PRE_TOOL_HOOK / NX_POST_TOOL_HOOK — JSON-on-stdin policy hooks with
veto semantics on the pre-hook and observe-only semantics on the post-hook.
"""
import importlib
import json
import os
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR",
          "NX_PRE_TOOL_HOOK", "NX_POST_TOOL_HOOK"):
    os.environ.pop(k, None)

import pytest


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    os.environ["NX_SUBAGENT_ASYNC"] = "0"  # these tests exercise the foreground sub-loop; the 0.18.8 default is background (covered in test_subagent_async.py)
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    yield nx, tmp_path
    os.environ.pop("NX_PRE_TOOL_HOOK", None)
    os.environ.pop("NX_POST_TOOL_HOOK", None)
    for k in ("NX_HOME", "NX_SESSIONS_DIR", "NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SUBAGENT_ASYNC"):
        os.environ.pop(k, None)


def _script(tmp_path, body, name="hook.sh"):
    p = tmp_path / name
    p.write_text(body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _mixed_turn(nx, tmp, read_paths, write_name="w.txt"):
    """Run one native mixed turn (read-only calls + one write_file) through
    the sequential dispatch path, exactly as model_turn would for a turn whose
    tool-call batch combines reads with a mutating call."""
    tcs = {i: {"id": f"c{i}", "name": "read_file", "args": json.dumps({"path": p})}
           for i, p in enumerate(read_paths)}
    tcs[len(read_paths)] = {
        "id": f"c{len(read_paths)}", "name": "write_file",
        "args": json.dumps({"path": write_name, "content": "x"})}
    messages = [{"role": "system", "content": "s"}]
    from nx._turn import _handle_native_tool_calls
    status = _handle_native_tool_calls(messages, tcs, "", ["tool_calls"], 0)
    assert status == nx.TURN_TOOL
    return messages


def test_no_hooks_configured_zero_path(nx_fresh):
    nx, tmp = nx_fresh
    from nx._dispatch import run_tool
    out = run_tool("list_dir", {"path": "."})
    assert not out.startswith("ERROR")


def test_pre_hook_allow_and_veto(nx_fresh, capsys):
    nx, tmp = nx_fresh
    from nx._dispatch import run_tool
    # allow (exit 0): tool runs normally
    os.environ["NX_PRE_TOOL_HOOK"] = _script(tmp, '#!/bin/sh\nexit 0\n')
    out = run_tool("list_dir", {"path": "."})
    assert not out.startswith("ERROR")
    # veto: tool never runs, model gets an actionable error
    os.environ["NX_PRE_TOOL_HOOK"] = _script(
        tmp, '#!/bin/sh\njq -r .tool # consume stdin\nif [ "$1" = "x" ]; then exit 0; fi\necho "no writes on fridays" >&2\nexit 3\n',
        name="veto.sh")
    out = run_tool("list_dir", {"path": "."})
    assert out.startswith("ERROR: blocked by NX_PRE_TOOL_HOOK")
    assert "no writes on fridays" in out
    assert "fridays" in capsys.readouterr().out


def test_pre_hook_receives_json_payload(nx_fresh):
    nx, tmp = nx_fresh
    seen = {}
    os.environ["NX_PRE_TOOL_HOOK"] = _script(
        tmp, '#!/bin/sh\ncat > ' + str(tmp / "payload.json") + '\nexit 0\n', name="capture.sh")
    from nx._dispatch import run_tool
    run_tool("list_dir", {"path": "some/dir"})
    payload = json.loads((tmp / "payload.json").read_text())
    assert payload["tool"] == "list_dir"
    assert payload["args"] == {"path": "some/dir"}


def test_pre_hook_cannot_be_bypassed_by_cache(nx_fresh):
    """A policy hook must veto even cache-hit reads (checked before the cache)."""
    nx, tmp = nx_fresh
    from nx import _dispatch
    _dispatch.st._TOOL_RESULT_CACHE.clear()
    os.environ["NX_PRE_TOOL_HOOK"] = _script(tmp, '#!/bin/sh\ncat >/dev/null\nexit 1\n', name="blockall.sh")
    out1 = _dispatch.run_tool("git_status", {"path": "."})
    out2 = _dispatch.run_tool("git_status", {"path": "."})  # second call would be a cache hit
    assert out1.startswith("ERROR: blocked by NX_PRE_TOOL_HOOK")
    assert out2.startswith("ERROR: blocked by NX_PRE_TOOL_HOOK")


def test_post_hook_observes_but_never_blocks(nx_fresh):
    nx, tmp = nx_fresh
    os.environ["NX_POST_TOOL_HOOK"] = _script(
        tmp, '#!/bin/sh\ncat > ' + str(tmp / "post.json") + '\necho "hook exploded" >&2\nexit 7\n',
        name="post.sh")
    from nx._dispatch import run_tool
    out = run_tool("list_dir", {"path": "."})
    assert not out.startswith("ERROR")  # post-hook failure can't affect the result
    payload = json.loads((tmp / "post.json").read_text())
    assert payload["tool"] == "list_dir"
    assert "result" in payload
    assert "cached" not in payload  # cache miss: no marker


def test_post_hook_fires_on_cache_hit_with_marker(nx_fresh):
    """Audit-trail contract: a cached read still fires the post-hook, tagged
    "cached": true — otherwise reads appear in the trail only once (the miss),
    making the observer non-deterministic."""
    nx, tmp = nx_fresh
    from nx import _dispatch
    _dispatch.st._TOOL_RESULT_CACHE.clear()
    stable = tmp / "stable"
    stable.mkdir()
    os.environ["NX_POST_TOOL_HOOK"] = _script(
        tmp, '#!/bin/sh\ncat > ' + str(tmp / "post.json") + '\nexit 0\n', name="post2.sh")
    # Read a stable dir rather than the cwd: the hook's own post.json write and
    # this test's unlink would bump the cwd's mtime, turning the second call
    # into a fresh miss before the cache-hit path is ever reached.
    args = {"path": "stable"}
    _dispatch.run_tool("git_status", args)   # miss: populates the cache
    first = json.loads((tmp / "post.json").read_text())
    assert "cached" not in first
    (tmp / "post.json").unlink()
    _dispatch.run_tool("git_status", args)   # hit: hook still fires
    payload = json.loads((tmp / "post.json").read_text())
    assert payload["cached"] is True
    assert payload["tool"] == "git_status"
    assert payload["result"]  # cached entries store the sanitized result string


def test_missing_hook_binary_vetoes_rather_than_skips(nx_fresh):
    nx, tmp = nx_fresh
    os.environ["NX_PRE_TOOL_HOOK"] = str(tmp / "definitely-not-here")
    from nx._dispatch import run_tool
    out = run_tool("list_dir", {"path": "."})
    assert out.startswith("ERROR: blocked by NX_PRE_TOOL_HOOK")
    # sh reports 'not found' (127) or the OSError path — either way it vetoes
    assert ("not found" in out) or ("hook could not run" in out)


def test_hung_hook_times_out(nx_fresh):
    nx, tmp = nx_fresh
    os.environ["NX_PRE_TOOL_HOOK"] = _script(tmp, '#!/bin/sh\nsleep 30\n', name="slow.sh")
    from nx._dispatch import run_tool
    out = run_tool("list_dir", {"path": "."})
    assert out.startswith("ERROR: blocked by NX_PRE_TOOL_HOOK")
    assert "timed out" in out


def test_hook_runs_inside_subagent_loop_too(nx_fresh, monkeypatch):
    """Sub-agents bypass approval but NOT org policy hooks."""
    nx, tmp = nx_fresh
    os.environ["NX_PRE_TOOL_HOOK"] = _script(tmp, '#!/bin/sh\ncat >/dev/null\nexit 1\n', name="block.sh")
    from nx._subagent import spawn_agent
    from nx._dispatch import run_tool

    def fake_loop(messages, *a, **kw):
        out = run_tool("list_dir", {"path": "."})
        assert out.startswith("ERROR: blocked by NX_PRE_TOOL_HOOK")
        messages.append({"role": "assistant", "content": "blocked"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx, "_run_model_turn_loop", fake_loop)
    result = spawn_agent("list the files")
    assert result == "blocked"
