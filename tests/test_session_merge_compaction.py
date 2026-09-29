#!/usr/bin/env python3
"""Regression tests for the session-save merge vs. context compression.

_write_session() re-reads the on-disk transcript under the lock and merges
the caller's snapshot into it. The merge assumed both writers only ever
append; compress() instead *shortens* messages in place (summary turn +
verbatim tail), so every save after a /compress used to splice the
summarized-away history back in front of the summary and duplicate the
verbatim tail — once per autosave step, growing the file unboundedly.

These tests pin the shortening-rewrite detection
(nx._sessions_fs._is_compaction_rewrite) and the two-writer append behavior
that must keep working (a genuinely concurrent second REPL).
"""
import os
import shutil
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-session-merge-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
for k in list(os.environ):
    if k.startswith("NX_") and k not in ("NX_HOME", "NX_SESSIONS_DIR"):
        del os.environ[k]
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m


def _clear_test_sessions_dir():
    d = os.environ["NX_SESSIONS_DIR"]
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def _msg(role, content):
    return {"role": role, "content": content}


def _pre_compress_history(n_user=20):
    """[system] + n user/assistant turn pairs, with 'u<i>' markers in the
    body text so content-duplication checks can find them."""
    msgs = [{"role": "system", "content": "You are nx."}]
    for i in range(n_user):
        msgs.append(_msg("user", f"u{i}"))
        msgs.append(_msg("assistant", f"a{i}"))
    return msgs


def _compress_snapshot(existing, keep=4):
    """Reproduce the shape compress() writes via messages[:] = sys + [summary]
    + tail (nx/_compress.py, end of compress())."""
    sys_msg = [existing[0]] if existing[0].get("role") == "system" else []
    summarized = len(existing) - len(sys_msg) - keep
    summary = _msg(
        "user",
        f"[Compressed context — {summarized} earlier turns summarized;"
        f" last {keep} turns kept verbatim]\n\nearlier work summarized here",
    )
    return sys_msg + [summary] + existing[-keep:]


# --- 1. the reproduced regression ------------------------------------------

def test_compaction_merge_does_not_resurrect_history():
    existing = _pre_compress_history()
    incoming = _compress_snapshot(existing, keep=4)
    assert len(incoming) < len(existing)
    merged = m._merge_session_messages(existing, incoming)
    assert merged == incoming
    assert len(merged) == len(incoming)


def test_compaction_merge_without_system_prompt():
    existing = _pre_compress_history()[1:]
    incoming = _compress_snapshot(existing, keep=4)
    merged = m._merge_session_messages(existing, incoming)
    assert merged == incoming


# --- 2. iterated saves stay stable ------------------------------------------

def test_repeated_writes_keep_file_stable():
    _clear_test_sessions_dir()
    sid = "20250101-120000-merge1"
    base = _pre_compress_history()
    incoming = _compress_snapshot(base, keep=4)
    m._write_session(sid, base)  # pre-compress state hits disk first
    for _ in range(5):
        m._write_session(sid, incoming)
        on_disk = m._load_session(sid)["messages"]
        assert len(on_disk) == len(incoming), (len(on_disk), len(incoming))
    body = "\n".join(
        str(t.get("content")) for t in on_disk if t.get("role") == "user")
    assert body.count("u19") == 1, body


# --- 3. genuine two-writer append still merges ------------------------------

def test_two_writer_append_still_survives():
    base = _pre_compress_history()
    other = base + [_msg("user", "from the other process")]
    merged = m._merge_session_messages(base, other)
    assert any(
        t.get("content") == "from the other process" for t in merged)
    # and a stale snapshot written after that must not clobber the other
    # writer's turn (existing longer, incoming is its prefix)
    assert m._merge_session_messages(other, base) is other or \
        m._merge_session_messages(other, base) == other


def test_compaction_then_other_writer_append():
    existing = _pre_compress_history()
    incoming = _compress_snapshot(existing, keep=4)
    first = m._merge_session_messages(existing, incoming)
    assert first == incoming
    # another process appends one turn after the compaction landed
    grew = first + [_msg("user", "appended after compress")]
    assert m._merge_session_messages(first, grew) == grew


# --- 4. non-compaction shortening keeps the fork path -----------------------

def test_plain_shortening_without_header_still_forks():
    existing = _pre_compress_history()
    incoming = existing[:len(existing) // 2]  # same prefix, no summary header
    merged = m._merge_session_messages(existing, incoming)
    assert merged is existing or merged == existing


def test_fork_splice_unchanged():
    existing = _pre_compress_history() + [_msg("user", "branch A")]
    incoming = _pre_compress_history() + [_msg("user", "branch B")]
    merged = m._merge_session_messages(existing, incoming)
    assert merged == _pre_compress_history() + \
        [_msg("user", "branch A"), _msg("user", "branch B")]


# --- 5. structural detection is specific ------------------------------------

def test_compaction_merge_with_a_few_unsaved_trailing_turns():
    # The regression behind the 600K-message store bloat: autosaves land
    # between the compress() snapshot and the post-turn save, so the
    # incoming tail is missing a couple of just-saved turns. That must
    # still count as a compaction rewrite, not a fork that re-appends the
    # whole compressed snapshot on every save.
    existing = _pre_compress_history() + \
        [_msg("user", "just-saved turn"), _msg("assistant", "ack")]
    incoming = _compress_snapshot(existing, keep=4)
    incoming = incoming[:-2]  # snapshot predates those two turns
    merged = m._merge_session_messages(existing, incoming)
    assert merged == incoming


def test_summary_header_without_matching_tail_is_not_compaction():
    existing = _pre_compress_history()
    incoming = _compress_snapshot(existing, keep=4)
    # break the tail so it is not a suffix of existing at any slack offset
    incoming[-1] = _msg("user", "tail rewritten after snapshot")
    merged = m._merge_session_messages(existing, incoming)
    assert merged != incoming  # falls through to the fork splice


def test_round_trip_resume_has_no_duplicate_turns():
    _clear_test_sessions_dir()
    sid = "20250101-120000-merge2"
    base = _pre_compress_history()
    incoming = _compress_snapshot(base, keep=4)
    m._write_session(sid, base)
    m._write_session(sid, incoming)
    resumed = m._load_session(sid)["messages"]
    assert resumed == incoming
    contents = [str(t.get("content")) for t in resumed]
    assert len(contents) == len(set(contents))
    # and a further real compression (incoming strictly shorter) keeps the
    # file monotonic instead of splicing the old summary back in
    again = _compress_snapshot(resumed, keep=2)
    assert len(again) < len(resumed)
    m._write_session(sid, again)
    assert m._load_session(sid)["messages"] == again
