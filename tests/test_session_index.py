#!/usr/bin/env python3
"""Tests for the sidecar listing index (`.index.json`) that lets
`_list_sessions()` / `nx sessions [query]` list and search saved
sessions without json.load-ing every full transcript on disk.
"""
import os
import shutil
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-session-index-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m


def _use_test_sessions_dir():
    os.environ["NX_SESSIONS_DIR"] = _tmp
    os.environ["NX_HOME"] = _tmp
    return m._sessions_dir()


def setup_function(_function):
    _use_test_sessions_dir()


def _clear_test_sessions_dir():
    d = _use_test_sessions_dir()
    assert os.path.basename(d).startswith("nx-session-index-"), d
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
    return d


def _msg(role, content):
    return {"role": role, "content": content}


def test_write_session_populates_index_entry():
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx001"
    m._write_session(sid, [_msg("user", "hello there")], {"title": "greeting"})
    idx = m._load_index()
    assert sid in idx
    assert idx[sid]["title"] == "greeting"
    assert idx[sid]["preview"] == "hello there"
    assert idx[sid]["n"] == 1


def test_index_dir_is_excluded_from_session_listing():
    _clear_test_sessions_dir()
    m._write_session("20250101-120000-idx002", [_msg("user", "hi")])
    files = m._session_files_newest()
    assert all(not f.startswith(".") for f in files)
    assert ".index" not in files


def test_list_sessions_uses_index_even_if_file_is_corrupted():
    """The whole point of the index: listing must not require a full parse
    of every transcript. Prove it by corrupting the on-disk file directly
    (bypassing nx) after the index has a valid entry, and checking the
    listing is unaffected."""
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx003"
    m._write_session(sid, [_msg("user", "important preview text")],
                      {"title": "corruption-proof"})
    path = m._session_path(sid)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json at all")

    sessions = m._list_sessions()
    match = [s for s in sessions if s["id"] == sid]
    assert len(match) == 1
    assert match[0]["title"] == "corruption-proof"
    assert match[0]["preview"] == "important preview text"


def test_missing_index_entry_falls_back_to_full_parse():
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx004"
    m._write_session(sid, [_msg("user", "fallback preview")],
                      {"title": "fallback-title"})
    # Simulate an index written by an older nx (or lost/corrupted) by
    # dropping this session's entry.
    idx = m._load_index()
    idx.pop(sid, None)
    m._write_index(idx)

    sessions = m._list_sessions()
    match = [s for s in sessions if s["id"] == sid]
    assert len(match) == 1
    assert match[0]["title"] == "fallback-title"
    assert match[0]["preview"] == "fallback preview"


def test_delete_session_removes_index_entry():
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx005"
    m._write_session(sid, [_msg("user", "bye")])
    assert sid in m._load_index()
    assert m._delete_session(sid) is True
    assert sid not in m._load_index()


def test_missing_index_dir_degrades_gracefully():
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx006"
    m._write_session(sid, [_msg("user", "still works")], {"title": "no-index"})
    shutil.rmtree(m._index_dir())

    sessions = m._list_sessions()
    match = [s for s in sessions if s["id"] == sid]
    assert len(match) == 1
    assert match[0]["title"] == "no-index"
    # a fresh shard should now exist again after the next write
    m._write_session(sid, [_msg("user", "still works")], {"title": "no-index"})
    assert os.path.exists(m._index_entry_path(sid))


def test_corrupt_index_shard_degrades_gracefully():
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx007"
    m._write_session(sid, [_msg("user", "resilient")], {"title": "resilient-title"})
    with open(m._index_entry_path(sid), "w", encoding="utf-8") as f:
        f.write("not json")

    sessions = m._list_sessions()
    match = [s for s in sessions if s["id"] == sid]
    assert len(match) == 1
    assert match[0]["title"] == "resilient-title"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
