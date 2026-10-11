#!/usr/bin/env python3
"""self-retro (nx #1): collect_friction grouping/cap/dedupe and run_retro
non-interactive note writing, against a seeded usage.jsonl and a temp
project memories dir. Follows test_memory_optimize.py's state-restore
pattern (st.PROJECT_ROOT / st.SESSION_HOME swapped in a fixture; conftest
rewinds nx module dicts between tests anyway).

The session-end retro is ON by default (a design change after the first
spec: the user must not have to remember to ask for it), so the hook tests
below assert the opt-OUT — only an explicit NX_RETRO_ON_EXIT=0 silences it.
"""
import importlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _retro  # noqa: E402

_root = tempfile.mkdtemp(prefix="nx-test-retro-")


def _rec(event, **data):
    return {"ts": 0.0, "dir": "usage", "event": event, "data": data}


def _write_usage(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _seed_records(sid="sess-1"):
    """Two edit_file repairs, three breaker trips on multi_edit, two failed
    tool_calls on search_files, one lone gate (below threshold), one ok
    tool_call, one model_call — plus a foreign-session retry pair."""
    return [
        _rec("repair", kind="salvaged_truncated_args", tool="edit_file",
             what="edit_file: closed brackets", session=sid,
             ctx_tail="first\nlet me close the brackets"),
        _rec("repair", kind="malformed_args", tool="edit_file",
             what="edit_file: bad json", session=sid),
        _rec("breaker", tool="multi_edit", trip_count=3, session=sid),
        _rec("breaker", tool="multi_edit", trip_count=4, session=sid),
        _rec("breaker", tool="multi_edit", trip_count=5, session=sid),
        _rec("tool_call", tool="search_files", outcome="error",
             args={"pattern": "x"}, session=sid, error_kind="io"),
        _rec("tool_call", tool="search_files", outcome="error",
             args={"pattern": "y"}, session=sid, error_kind="io"),
        _rec("tool_call", tool="read_file", outcome="ok",
             args={"path": "a"}, session=sid),
        _rec("gate", tool="memory_write", reason="no schema", session=sid),
        _rec("model_call", duration_ms=100, session=sid),
        _rec("retry", attempt=1, error_class="timeout", session="other"),
        _rec("retry", attempt=2, error_class="timeout", session="other"),
    ]


def _seed(tmp=None, records=None):
    """Point SESSION_HOME/PROJECT_ROOT at a fresh temp dir and seed its
    usage.jsonl. Returns (usage_path, project_root)."""
    tmp = tmp or tempfile.mkdtemp(prefix="nx-test-retro-")
    root = os.path.join(tmp, "proj")
    os.makedirs(root, exist_ok=True)
    usage = os.path.join(tmp, "logs", "usage.jsonl")
    _write_usage(usage, records if records is not None else _seed_records())
    m.st.SESSION_HOME = tmp
    m.st.PROJECT_ROOT = root
    return usage, root


def _memdir():
    return os.path.join(m.st.PROJECT_ROOT, ".nx", "memories")


def _notes():
    d = _memdir()
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


# --- collect_friction --------------------------------------------------------

def test_collect_friction_groups_sorts_and_caps():
    usage, _root = _seed()
    pats = _retro.collect_friction("sess-1", path=usage)
    # 3 breaker + 2 repair(edit_file) + 2 tool_call(search_files) qualify;
    # the lone gate and the foreign-session retries do not. Cap keeps 3.
    assert len(pats) == 3
    assert pats[0] == {"event": "breaker", "subject": "multi_edit",
                       "count": 3, "detail": ""}
    assert [(p["event"], p["subject"]) for p in pats[1:]] == [
        ("repair", "edit_file"), ("tool_call", "search_files")]
    # detail: sample ctx_tail line from the group
    repair = next(p for p in pats if p["event"] == "repair")
    assert repair["detail"] == "let me close the brackets"


def test_collect_friction_cap_at_three():
    usage = os.path.join(tempfile.mkdtemp(prefix="nx-test-retro-"), "u.jsonl")
    recs = []
    for tool in ("a", "b", "c", "d", "e"):
        recs += [_rec("breaker", tool=tool), _rec("breaker", tool=tool)]
    _write_usage(usage, recs)
    pats = _retro.collect_friction(None, path=usage)
    assert len(pats) == 3


def test_collect_friction_skips_malformed_and_ok_events():
    usage = os.path.join(tempfile.mkdtemp(prefix="nx-test-retro-"), "u.jsonl")
    os.makedirs(os.path.dirname(usage), exist_ok=True)
    with open(usage, "w", encoding="utf-8") as f:
        f.write("not json\n")
        f.write(json.dumps(_rec("tool_call", tool="x", outcome="ok",
                                args={})) + "\n")
        f.write(json.dumps(_rec("tool_call", tool="x", outcome="err",
                                args="not-a-dict")) + "\n")
    assert _retro.collect_friction(None, path=usage) == []


def test_collect_friction_missing_file():
    missing = os.path.join(tempfile.mkdtemp(prefix="nx-test-retro-"),
                           "nope.jsonl")
    assert _retro.collect_friction(None, path=missing) == []


# --- run_retro (the /retro body) ---------------------------------------------

def test_run_retro_non_interactive_writes_gotcha_notes():
    _seed()
    summary = _retro.run_retro("sess-1", interactive=False)
    assert summary.startswith("retro: 3 friction patterns, 3 notes written:")
    notes = _notes()
    assert len(notes) == 3
    with open(os.path.join(_memdir(), notes[0]), encoding="utf-8") as f:
        body = f.read()
    assert "_kind: gotcha" in body
    assert "# Retro " in body and "repeated" in body


def test_run_retro_dedupes_against_existing_notes():
    _seed()
    from nx import _memory_store
    # an existing note mentioning breaker + multi_edit covers one pattern
    _memory_store._write_memory_file(
        "multi_edit keeps tripping the breaker",
        "the multi_edit tool trips the malformed-args breaker a lot",
        scope="project", gate=False)
    summary = _retro.run_retro("sess-1", interactive=False)
    assert "3 friction patterns, 2 notes written" in summary
    assert "already noted" not in summary
    # 3 notes on disk: the pre-existing one + the 2 the retro just wrote
    assert len(_notes()) == 3


def test_run_retro_all_noted_summary():
    _seed()
    from nx import _memory_store
    for combo in (("breaker", "multi_edit"), ("repair", "edit_file"),
                  ("tool_call", "search_files")):
        _memory_store._write_memory_file(
            f"{combo[0]} on {combo[1]}", f"seen {combo[1]} {combo[0]} here",
            scope="project", gate=False)
    summary = _retro.run_retro("sess-1", interactive=False)
    assert summary == ("retro: 3 friction patterns, "
                       "3 already noted — nothing new written")


def test_run_retro_no_usage_data():
    tmp = tempfile.mkdtemp(prefix="nx-test-retro-")
    m.st.SESSION_HOME = tmp
    m.st.PROJECT_ROOT = tmp
    assert _retro.run_retro("sess-x", interactive=False) == "retro: no usage data"


def test_run_retro_no_significant_friction():
    usage, _root = _seed(records=[_rec("gate", tool="memory_write")])
    assert usage and os.path.exists(usage)
    assert _retro.run_retro("sess-1", interactive=False) == \
        "retro: no significant friction"


def test_run_retro_interactive_requires_confirmation(monkeypatch):
    _seed()
    calls = []

    def deny(action, preview=None):
        calls.append(action)
        return False

    monkeypatch.setattr(m, "_confirm", deny)
    summary = _retro.run_retro("sess-1", interactive=True)
    assert "0 notes written" in summary
    assert len(calls) == 3  # one confirm per pattern
    assert not _notes()


# --- the session-end hook (automatic by default) -----------------------------

def test_retro_on_exit_defaults_on_and_optout(monkeypatch):
    """The design change: no env var at all must still fire the retro;
    NX_RETRO_ON_EXIT=0 is the only silence."""
    _seed(records=_seed_records(sid="sess-e1"))
    _retro._reset_retro_done()
    monkeypatch.delenv("NX_RETRO_ON_EXIT", raising=False)
    _retro.retro_on_exit("sess-e1")
    assert len(_notes()) == 3

    _seed(records=_seed_records(sid="sess-e2"))
    _retro._reset_retro_done()
    monkeypatch.setenv("NX_RETRO_ON_EXIT", "0")
    _retro.retro_on_exit("sess-e2")
    assert _notes() == []
    _retro._reset_retro_done()


def test_retro_on_exit_once_per_session_id(monkeypatch):
    _seed(records=_seed_records(sid="sess-e3"))
    _retro._reset_retro_done()
    monkeypatch.delenv("NX_RETRO_ON_EXIT", raising=False)
    try:
        _retro.retro_on_exit("sess-e3")
        assert len(_notes()) == 3
        # second call for the same id: guarded, no second batch
        _retro.retro_on_exit("sess-e3")
        assert len(_notes()) == 3
    finally:
        _retro._reset_retro_done()


def test_retro_on_exit_never_raises(monkeypatch):
    """A broken memory store must not turn an exit into a traceback."""
    _seed(records=_seed_records(sid="sess-e4"))
    _retro._reset_retro_done()
    monkeypatch.delenv("NX_RETRO_ON_EXIT", raising=False)

    def boom(*a, **k):
        raise RuntimeError("no store")

    retro = importlib.import_module("nx._retro")
    monkeypatch.setattr(retro, "run_retro", boom)
    retro.retro_on_exit("sess-e4")  # must not raise
    retro._reset_retro_done()


# --- exit-path wiring --------------------------------------------------------

def test_repl_exit_hook_calls_retro_and_swallows(monkeypatch):
    """The choke point both clean exits (`/quit`, EOF/Ctrl-D) call: it hands
    st.SESSION_ID to `_retro.retro_on_exit` and never raises."""
    # Import live: conftest rewinds/pops nx.* modules between tests, so a
    # module captured at test-module import time may not be the one the lazy
    # `from ._retro import ...` inside the hook resolves at call time.
    _repl = importlib.import_module("nx._repl")
    retro = importlib.import_module("nx._retro")
    seen = []
    monkeypatch.setattr(m.st, "SESSION_ID", "sess-hook")
    monkeypatch.setattr(retro, "retro_on_exit", lambda sid: seen.append(sid))
    _repl._retro_on_exit_safe()
    assert seen == ["sess-hook"]

    def boom(sid):
        raise RuntimeError("retro crashed")

    monkeypatch.setattr(retro, "retro_on_exit", boom)
    _repl._retro_on_exit_safe()  # must not raise


def test_cmd_quit_runs_the_exit_retro(monkeypatch):
    """`/quit` (the documented exit path) calls the retro hook and still
    returns "break"."""
    _rsc = importlib.import_module("nx._repl_settings_cmds")
    retro = importlib.import_module("nx._retro")
    seen = []
    monkeypatch.setattr(retro, "retro_on_exit", lambda sid: seen.append(sid))
    monkeypatch.setattr(m.st, "SESSION_ID", "sess-quit")

    class _S:
        session_dirty = False

    assert _rsc._cmd_quit("/quit", _S()) == "break"
    assert seen == ["sess-quit"]


def test_retro_registered_everywhere():
    """/retro is dispatchable, listed in /help, and offered in the chatbox
    dropdown (the registration trio every command keeps in sync)."""
    _repl = importlib.import_module("nx._repl")
    _rsc = importlib.import_module("nx._repl_settings_cmds")
    _chatbox = importlib.import_module("nx._chatbox")
    assert _repl._REPL_COMMANDS.get("/retro") is _rsc._cmd_retro
    assert any(c[0] == "/retro" for c in _rsc._HELP_COMMANDS)
    assert any(cmd == "/retro" for cmd, _ in _chatbox.SLASH_COMMANDS)
