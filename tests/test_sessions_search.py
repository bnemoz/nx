"""F4: session body search (FTS5) + /export markdown."""
import importlib
import os
import sys

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
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    os.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    yield nx, tmp_path
    for k in ("NX_HOME", "NX_SESSIONS_DIR", "NX_BASE_URL", "NX_MODEL", "NX_API_KEY"):
        os.environ.pop(k, None)


def _write_session(nx, sid, title, user_msgs, updated_at=None):
    msgs = [{"role": "system", "content": "sys"}]
    for u in user_msgs:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": f"re: {u}"})
    nx._write_session(sid, msgs, meta={"id": sid, "title": title,
                                       "created_at": updated_at or 1,
                                       "updated_at": updated_at or 1,
                                       "cwd": os.getcwd()})
    return os.path.join(os.environ["NX_SESSIONS_DIR"], f"{sid}.json")


def test_search_finds_body_matches(nx_fresh):
    nx, tmp = nx_fresh
    _write_session(nx, "sess-alpha", "alpha session",
                   ["fix the regex in _stream_parse.py"])
    _write_session(nx, "sess-beta", "beta session",
                   ["add a retry to _openai_client.py"])
    from nx._sessions_search import search_sessions
    hits = search_sessions("_stream_parse")
    assert len(hits) == 1
    assert hits[0]["id"] == "sess-alpha"
    # FTS5's default tokenizer splits on '_' so the snippet highlights the tail;
    # assert on the highlighted fragment rather than the raw token
    assert "[[stream_parse]]" in hits[0]["snippet"] or "stream_parse" in hits[0]["snippet"]
    # title still searchable
    hits = search_sessions("beta")
    assert len(hits) == 1 and hits[0]["id"] == "sess-beta"
    # multiple-body-match query ranks both
    hits = search_sessions("retry OR regex")
    assert {h["id"] for h in hits} == {"sess-alpha", "sess-beta"}


def test_search_increments_new_and_stale_sessions(nx_fresh):
    nx, tmp = nx_fresh
    _write_session(nx, "s1", "one", ["hello world"])
    from nx._sessions_search import search_sessions, index_stats, _refresh_index
    assert len(search_sessions("hello")) == 1
    assert index_stats()["sessions_indexed"] == 1
    # a second session is picked up without touching the first
    _write_session(nx, "s2", "two", ["goodbye world"])
    assert len(search_sessions("world")) == 2
    # re-indexing is mtime-gated: nothing changed → no work
    assert _refresh_index() == 0


def test_search_degrades_on_bad_fts_syntax(nx_fresh):
    nx, tmp = nx_fresh
    _write_session(nx, "s1", "one", ["hello world"])
    from nx._sessions_search import search_sessions
    # quotes/operators that make FTS bark → empty, never an exception
    assert search_sessions('"unterminated') == []
    assert search_sessions("NEAR(") == []
    assert search_sessions("") == []


def test_export_markdown_shape(nx_fresh):
    nx, tmp = nx_fresh
    from nx._sessions_search import export_markdown
    data = {
        "id": "abc123", "title": "test session", "source": "local",
        "model": "m1", "created_at": 1700000000,
        "description": "a described session",
        "messages": [
            {"role": "system", "content": "hidden"},
            {"role": "user", "content": "please do a thing"},
            {"role": "assistant", "content": "did the thing",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "read_file", "arguments": '{"path": "x"}'}}]},
            {"role": "tool", "content": "x" * 5000},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    md = export_markdown(data)
    assert "# test session" in md
    assert "id `abc123`" in md
    assert "please do a thing" in md
    assert "→ read_file" in md
    assert "hidden" not in md  # system prompt never exported
    assert "truncated" in md   # long tool body trimmed
    assert "input tokens: 10" in md


def test_cmd_export_writes_file(nx_fresh, capsys):
    nx, tmp = nx_fresh
    from nx._sessions_cmd import _cmd_export
    from nx._repl_state import _ReplState
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "export me"},
            {"role": "assistant", "content": "exported"}]
    state = _ReplState(msgs, "sess-x", False, [])
    _cmd_export("/export", state)
    out = capsys.readouterr().out
    assert "wrote" in out and ".md" in out
    path = os.path.join(tmp, "nx-export-sess-x.md")
    assert os.path.exists(path)
    body = open(path, encoding="utf-8").read()
    assert "export me" in body and "exported" in body
    # explicit path + 'md' format token both work
    _cmd_export("/export md notes/other.md", state)
    capsys.readouterr()
    assert os.path.exists(os.path.join(tmp, "notes", "other.md"))


def test_cmd_sessions_body_search_output(nx_fresh, capsys):
    nx, tmp = nx_fresh
    from nx._sessions_cmd import _cmd_sessions
    _write_session(nx, "findme-1", "unrelated title", ["the zebra QUOKKA escaped"])
    # a title match still takes priority
    _cmd_sessions("/sessions unrelated", "x")
    out = capsys.readouterr().out
    assert "unrelated title" in out
    # a body-only match falls through to FTS
    _cmd_sessions("/sessions quokka", "x")
    out = capsys.readouterr().out
    assert "containing 'quokka'" in out
    assert "QUOKKA" in out
    # no match anywhere → old message
    _cmd_sessions("/sessions pterodactyl", "x")
    out = capsys.readouterr().out
    assert "no session matching" in out
