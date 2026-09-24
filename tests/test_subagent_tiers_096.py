"""0.96.0 subagent orchestration: tier param, scope gate, lean sync results,
TLS-aware sub-loop schemas, explicit (never silent) refusals."""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import nx
import nx._subagent as S
from nx._subagent_tiers import (
    _SubagentContext, _normalize_tier, _scope_check, _scope_target,
    _subagent_tools, _SUBAGENT_TLS,
)


def _setup(tmp_path, monkeypatch):
    nx.SESSION_HOME = str(tmp_path)
    nx._state.SESSION_HOME = str(tmp_path)
    nx._state.PROJECT_ROOT = str(tmp_path)
    nx.NX_SUBAGENT_WRITE = 1
    nx.NX_SUBAGENT_SYNC_INLINE = 0
    monkeypatch.setattr(S, "_run_sync_agent_body",
                        lambda instruction, max_turns, tier="read", scope=None:
                        "did the thing" * 200)


def test_normalize_tier(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    assert _normalize_tier(None) == "write"      # env default
    assert _normalize_tier("read") == "read"
    assert _normalize_tier(" IMPLEMENT ") == "implement"
    assert _normalize_tier("bogus") == ""
    nx.NX_SUBAGENT_WRITE = 0
    assert _normalize_tier(None) == "read"


def test_write_tier_requires_env(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    nx.NX_SUBAGENT_WRITE = 0
    out = S.spawn_agent("x", tier="write")
    assert "NX_SUBAGENT_WRITE=1" in out and out.startswith("ERROR:")


def test_write_tier_refused_async_explicitly(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    out = S.spawn_agent("x", tier="implement", **{"async": "true"})
    assert out.startswith("ERROR:") and "async:false" in out


def test_write_tier_async_allowed_under_yolo(tmp_path, monkeypatch):
    # 0.96.1: a yolo session's approval gate auto-approves quiet writes
    # (_approval._confirm's quiet path returns True on YOLO), so a background
    # write agent is honest there — the spawn must NOT be refused.
    _setup(tmp_path, monkeypatch)
    nx._state.YOLO = True
    try:
        out = S.spawn_agent("x", tier="implement", **{"async": "true"})
        assert not out.startswith("ERROR:"), out
    finally:
        nx._state.YOLO = False
    # and without yolo the refusal names both fixes (async:false / --yolo)
    out = S.spawn_agent("x", tier="write", **{"async": "true"})
    assert "async:false" in out and "--yolo" in out


def test_bad_tier_and_scope_refused(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    assert S.spawn_agent("x", tier="nope").startswith("ERROR:")
    assert S.spawn_agent("x", scope="not-a-list tier? no", tier="read",
                         **{"async": "false"}).startswith("ERROR:")


def test_sync_result_is_lean_head_plus_path(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    out = S.spawn_agent("investigate the flurb", tier="read",
                        **{"async": "false"})
    assert out.startswith("[sub-agent finished]")
    assert "Full report: " in out
    assert "read_file it for the rest" in out
    assert len(out) < 900  # head, not the 2600-char answer
    # and the report file exists
    path = out.split("Full report: ", 1)[1].split(" —", 1)[0].strip()
    assert os.path.isfile(path) and "did the thing" in open(path).read()


def test_sync_inline_escape_hatch(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    nx.NX_SUBAGENT_SYNC_INLINE = 1
    out = S.spawn_agent("x", **{"async": "false"})
    assert out == "did the thing" * 200


def test_scope_gate_refuses_outside(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    ctx = _SubagentContext({}, write=True, scope=["nx/_turn*.py"])
    _SUBAGENT_TLS.ctx = ctx
    try:
        assert _scope_check(os.path.join(str(tmp_path), "nx", "_turn.py")) is None
        assert _scope_check(os.path.join(str(tmp_path), "nx", "_turn_retry.py")) is None
        refusal = _scope_check(os.path.join(str(tmp_path), "nx", "_dispatch.py"))
        assert refusal and refusal.startswith("ERROR:") and "scope" in refusal
        # no scope = unrestricted
        _SUBAGENT_TLS.ctx = _SubagentContext({}, write=True)
        assert _scope_check("/etc/passwd") is None
    finally:
        _SUBAGENT_TLS.ctx = None


def test_scope_target_extracts_paths(tmp_path, monkeypatch):
    assert _scope_target("write_file", {"path": "a/b.py"}) == "a/b.py"
    assert _scope_target("edit_file", {"file": "c.py"}) == "c.py"
    assert _scope_target("run_tests", {}) is None


def test_subloop_schemas_match_tier(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    # read-tier context: no write schemas
    _SUBAGENT_TLS.ctx = _SubagentContext({}, write=False)
    try:
        names = {t["function"]["name"] for t in _subagent_tools()}
        assert "write_file" not in names and "read_file" in names
    finally:
        _SUBAGENT_TLS.ctx = None
    # write-tier context: write schemas present
    _SUBAGENT_TLS.ctx = _SubagentContext({}, write=True)
    try:
        names = {t["function"]["name"] for t in _subagent_tools()}
        assert {"write_file", "edit_file", "run_tests"} <= names
    finally:
        _SUBAGENT_TLS.ctx = None
