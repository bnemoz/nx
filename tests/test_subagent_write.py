"""Write-tier sub-agents (NX_SUBAGENT_WRITE=1): the four file-mutation tools
are offered inside spawn_agent and resolve to confirm-through wrappers that
run the parent's real body — so the parent's _confirm() (risk classifier +
Y/n prompt with a diff preview) gates every sub-agent write, the tool box
header is tagged (sub-agent), and no write-tier agent ever runs concurrently
with another."""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
# Clear ambient NX_* config so a developer's ~/.env / shell (real endpoints,
# keys, NX_VERIFY_CMD, ...) can't affect results — per repo test guidelines.
_AMBIENT_NX = [k for k in os.environ if k == "NX_ENV_FILE" or k.startswith("NX_")]
for k in _AMBIENT_NX:
    os.environ.pop(k, None)
os.environ["NX_ENV_FILE"] = "/dev/null"
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR",
          "NX_SUBAGENTS", "NX_SUBAGENT_WRITE"):
    os.environ.pop(k, None)

import pytest

# 0.96.0: foreground spawn results are head+report-path by default; these
# tests assert on the full inline answer, so opt into the legacy inline mode.
os.environ["NX_SUBAGENT_SYNC_INLINE"] = "1"


@pytest.fixture(autouse=True)
def _sync_inline(monkeypatch):
    import nx as _nx
    monkeypatch.setattr(_nx, "NX_SUBAGENT_SYNC_INLINE", 1, raising=False)
    yield


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    for k in [k for k in os.environ if k.startswith("NX_")]:
        os.environ.pop(k, None)
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    os.environ["NX_SUBAGENT_ASYNC"] = "0"  # these tests exercise the foreground sub-loop; the 0.18.8 default is background (covered in test_subagent_async.py)
    # These tests dispatch REAL write_file/edit_file/delete_file calls to tmp_path,
    # outside the freshly-reloaded nx's project root (this repo). The shipped
    # fence (2.1) would refuse them before the body runs — and the behavior
    # under test is the approval/tag/wrapper layer, not the fence (pinned in
    # tests/test_write_fence.py). Re-enable the fence per-test where needed:
    # tests/test_write_fence.py's dispatch tests monkeypatch st.PROJECT_ROOT
    # already, and the default-on posture is asserted there.
    os.environ["NX_WRITE_FENCE"] = "0"
    monkeypatch.setattr(sys, "argv", ["nx"])
    import nx
    importlib.reload(nx)
    yield nx
    for k in ("NX_HOME", "NX_SESSIONS_DIR", "NX_BASE_URL", "NX_MODEL", "NX_API_KEY",
              "NX_SUBAGENTS", "NX_SUBAGENT_WRITE", "NX_SUBAGENT_ASYNC", "NX_WRITE_FENCE"):
        os.environ.pop(k, None)


def test_write_tier_off_by_default(nx_fresh):
    """Without NX_SUBAGENT_WRITE=1 the sub-loop is exactly the read-only one:
    write tools are neither offered nor dispatched."""
    from nx._subagent import _subagent_tools, _read_only_dispatch
    names = {t["function"]["name"] for t in _subagent_tools()}
    assert not (names & {"write_file", "edit_file", "multi_edit", "sed_edit"})
    table = _read_only_dispatch()
    for name in ("write_file", "edit_file", "multi_edit", "sed_edit"):
        assert name not in table


def test_write_tier_off_write_still_refused(nx_fresh, monkeypatch):
    from nx import _dispatch
    from nx._subagent import spawn_agent

    def fake_turn(messages, *a, **kw):
        out = _dispatch.run_tool("write_file", {"path": "evil.txt", "content": "x"})
        assert "not available to the read-only sub-agent" in out
        assert not os.path.exists("evil.txt")
        messages.append({"role": "assistant", "content": "ro"})
        return nx_fresh.TURN_DONE

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)
    assert spawn_agent("q") == "ro"


def test_write_tier_on_offers_and_wraps(nx_fresh, monkeypatch, tmp_path):
    """With NX_SUBAGENT_WRITE=1 the four schemas are offered, resolve to the
    confirm-through wrappers, and each body runs the parent's real tool with
    the sub-agent TLS context torn down around it (so _confirm sees the
    parent's approval posture, not the sub-loop's)."""
    from nx import _dispatch
    from nx._subagent import (_subagent_tools, _write_tier_dispatch,
                              _write_tier_enabled, spawn_agent, _SUBAGENT_TLS)
    nx_fresh.NX_SUBAGENT_WRITE = 1
    assert _write_tier_enabled() is True
    names = {t["function"]["name"] for t in _subagent_tools()}
    assert {"write_file", "edit_file", "multi_edit", "sed_edit"} <= names

    table = _write_tier_dispatch()
    for name in ("write_file", "edit_file", "multi_edit", "sed_edit"):
        assert table[name] is not _dispatch.DISPATCH[name]  # wrapped, not the raw body
    # the parent's table stays untouched
    for name in ("write_file", "edit_file", "multi_edit", "sed_edit"):
        assert _dispatch.DISPATCH[name] is getattr(nx_fresh, name)

    seen_ctx = {}

    def fake_confirm(action, preview=None):
        seen_ctx["in_confirm_ctx"] = _SUBAGENT_TLS.ctx is not None
        seen_ctx["action"] = action
        return True  # approve — this is what the parent's _confirm() returns

    monkeypatch.setattr(nx_fresh, "_confirm", fake_confirm)

    def fake_turn(messages, *a, **kw):
        assert messages[0]["content"] == nx_fresh._subagent.SUBAGENT_WRITE_SYSTEM
        out = _dispatch.run_tool("write_file", {"path": str(tmp_path / "w.txt"), "content": "hi"})
        assert out.startswith("wrote "), out
        # the body ran OUTSIDE the sub-agent context (confirm-through)
        assert seen_ctx["in_confirm_ctx"] is False
        # while back inside the sub-loop, the context is restored and still
        # flagged as write-capable (drives the (sub-agent) box tag)
        ctx = _SUBAGENT_TLS.ctx
        assert ctx is not None and ctx.write is True
        out2 = _dispatch.run_tool("delete_file", {"path": str(tmp_path / "w.txt")})
        assert "not available to the read-only sub-agent" in out2  # not tier-eligible
        messages.append({"role": "assistant", "content": "wrote it"})
        return nx_fresh.TURN_DONE

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)
    assert spawn_agent("write a file") == "wrote it"
    assert (tmp_path / "w.txt").read_text() == "hi"


def test_write_tier_edit_denied_by_parent_confirm(nx_fresh, monkeypatch, tmp_path):
    """The sub-agent's edit is denied when the parent's _confirm() says no —
    the gate is the parent's, not a YOLO override."""
    from nx._subagent import spawn_agent
    nx_fresh.NX_SUBAGENT_WRITE = 1

    def fake_confirm(action, preview=None):
        assert preview is not None  # diff preview reached the gate
        return False

    monkeypatch.setattr(nx_fresh, "_confirm", fake_confirm)

    def fake_turn(messages, *a, **kw):
        from nx import _dispatch
        target = tmp_path / "f.txt"
        target.write_text("alpha\n")
        out = _dispatch.run_tool("edit_file", {"path": str(target), "old": "alpha", "new": "beta"})
        assert out == "DENIED by user", out
        assert target.read_text() == "alpha\n"  # nothing was written
        messages.append({"role": "assistant", "content": "denied"})
        return nx_fresh.TURN_DONE

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)
    assert spawn_agent("try to edit") == "denied"
    assert (tmp_path / "f.txt").read_text() == "alpha\n"


def test_write_tier_tool_box_tagged_sub_agent(nx_fresh, monkeypatch, tmp_path, capsys):
    """A write-tier sub-agent's file mutation is tagged (sub-agent) in the
    tool box header; the same call outside a sub-agent is not."""
    from nx import _dispatch
    from nx._subagent import _SubagentContext, _write_tier_dispatch, _SUBAGENT_TLS
    nx_fresh.NX_SUBAGENT_WRITE = 1
    monkeypatch.setattr(nx_fresh, "_confirm", lambda action, preview=None: True)
    target = tmp_path / "tagged.txt"
    args = {"path": str(target), "content": "x"}

    _SUBAGENT_TLS.ctx = _SubagentContext(_write_tier_dispatch(), write=True)
    try:
        _dispatch.run_tool("write_file", args)
    finally:
        _SUBAGENT_TLS.ctx = None
    out1 = capsys.readouterr().out
    assert "write_file " + str(target) + " (sub-agent)" in out1, out1

    _dispatch.run_tool("write_file", args)
    out2 = capsys.readouterr().out
    assert "(sub-agent)" not in out2, out2


def test_write_tier_boxes_readonly_calls_untagged(nx_fresh, monkeypatch, tmp_path):
    """Read-only calls inside a write-tier agent keep the plain header — the
    tag is only for the mutations that can prompt."""
    import contextlib
    import io
    from nx import _dispatch
    from nx._subagent import _SubagentContext, _write_tier_dispatch, _SUBAGENT_TLS
    nx_fresh.NX_SUBAGENT_WRITE = 1
    _SUBAGENT_TLS.ctx = _SubagentContext(_write_tier_dispatch(), write=True)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # unique path: "." would hit the cross-test read-only result cache
            # (list_dir entries persist on the shared _state module) and return
            # before any box is printed
            _dispatch.run_tool("list_dir", {"path": str(tmp_path)})
    finally:
        _SUBAGENT_TLS.ctx = None
    out = buf.getvalue()
    assert f"list_dir {tmp_path} (sub-agent)" not in out
    assert "┌─ list_dir" in out  # plain header, no tag


def test_write_tier_never_parallel_fanout(nx_fresh, monkeypatch, tmp_path):
    """Two write-capable spawn_agent calls stay sequential: their bodies hit
    the real _confirm(), and two prompts must never interleave."""
    from nx._turn import _should_parallelize
    nx_fresh.NX_SUBAGENT_WRITE = 1
    calls = [("spawn_agent", {"instruction": "A"}), ("spawn_agent", {"instruction": "B"})]
    assert _should_parallelize(calls) is False
    assert _should_parallelize(calls + [("read_file", {"path": "a"})]) is False
    # read-only spawns still fan out when the tier is on but unused
    nx_fresh.NX_SUBAGENT_WRITE = 0
    assert _should_parallelize(calls) is True


def test_write_tier_no_bash_widening(nx_fresh, monkeypatch, tmp_path):
    """The write tier widens nothing but the four file mutations: bash stays
    allowlist-gated, delete/wait_background/memory_write stay unavailable."""
    from nx import _dispatch
    from nx._subagent import _SubagentContext, _write_tier_dispatch, _SUBAGENT_TLS
    nx_fresh.NX_SUBAGENT_WRITE = 1
    table = _write_tier_dispatch()
    from nx._subagent import _subagent_bash
    assert table["run_bash"] is _subagent_bash  # bash stays allowlist-gated
    ran = []
    monkeypatch.setattr(nx_fresh._bash_jobs, "run_bash", lambda command="", **kw: ran.append(command) or "ran")
    _SUBAGENT_TLS.ctx = _SubagentContext(table, write=True)
    try:
        out = _dispatch.run_tool("run_bash", {"command": "rm -rf x"})
        assert out.startswith("ERROR") and "read-only" in out
        out2 = _dispatch.run_tool("memory_write", {"title": "t", "body": "b"})
        assert "not available to the read-only sub-agent" in out2
        out3 = _dispatch.run_tool("wait_background", {"pid": 1})
        assert "not available to the read-only sub-agent" in out3
    finally:
        _SUBAGENT_TLS.ctx = None
    assert ran == []  # nothing reached bash


def test_write_tier_nested_spawn_refused(nx_fresh, monkeypatch):
    """A write-tier agent cannot spawn another agent — only the parent holds
    the approval gate."""
    from nx._subagent import spawn_agent
    nx_fresh.NX_SUBAGENT_WRITE = 1

    def fake_turn(messages, *a, **kw):
        from nx import _dispatch
        out = _dispatch.run_tool("spawn_agent", {"instruction": "x"})
        assert out.startswith("ERROR") and "not available to the read-only sub-agent" in out
        messages.append({"role": "assistant", "content": "flat"})
        return nx_fresh.TURN_DONE

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)
    assert spawn_agent("q") == "flat"


def test_write_tier_yolo_inherited_not_forced(nx_fresh, monkeypatch, tmp_path):
    """The sub-loop is not forced YOLO when the write tier is on: the parent's
    approval posture is inherited (a YOLO parent's sub-agent writes run
    unattended; a prompting parent's sub-agent writes prompt)."""
    from nx import _state as st
    from nx._subagent import spawn_agent
    nx_fresh.NX_SUBAGENT_WRITE = 1
    seen = {}

    def fake_turn(messages, *a, **kw):
        seen["yolo"] = st.YOLO
        messages.append({"role": "assistant", "content": "ok"})
        return nx_fresh.TURN_DONE

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)
    st.YOLO, st.APPROVE_LEVEL = False, "medium"
    try:
        spawn_agent("q")
        assert seen["yolo"] is False
        assert st.YOLO is False and st.APPROVE_LEVEL == "medium"  # restored
    finally:
        st.YOLO, st.APPROVE_LEVEL = False, "high"


def test_write_tier_wrappers_refuse_outside_context(nx_fresh):
    """A wrapper escaping its sub-agent context (defensive) errors instead of
    running a silent write."""
    from nx._subagent import _write_tier_dispatch
    table = _write_tier_dispatch()
    out = table["write_file"](path="x.txt", content="y")
    assert out.startswith("ERROR") and "outside a sub-agent" in out
    assert not os.path.exists("x.txt")
