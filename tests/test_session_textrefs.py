#!/usr/bin/env python3
"""0.116.0: big-text sidecar offload — payloads over NX_SESSION_TEXT_REF_MIN
live in <sessions>/_textrefs/<sha256>.txt; events carry a text_ref with a
512-char preview; _load_session rehydrates; GC removes unreferenced files.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-session-textrefs-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
for k in list(os.environ):
    if k.startswith("NX_") and k not in ("NX_HOME", "NX_SESSIONS_DIR"):
        del os.environ[k]
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _sessions_fs as fs  # noqa: E402
from nx import _sessiondb as sdb  # noqa: E402


def _clear():
    for f in os.listdir(_tmp):
        p = os.path.join(_tmp, f)
        if os.path.isdir(p):
            import shutil
            shutil.rmtree(p)
        else:
            os.remove(p)


BIG = "X" * 100000  # > 64 KB default threshold
SMALL = "a short turn"


def test_store_form_offloads_and_load_rehydrates():
    _clear()
    msgs = [{"role": "user", "content": BIG},
            {"role": "assistant", "content": SMALL}]
    store = fs._store_form_messages(msgs)
    # store form carries a ref + preview, not the payload
    ref = store[0]["content"][0]
    assert ref["type"] == "text_ref" and ref["str"] is True
    assert len(ref["text"]) == 513 and ref["bytes"] == len(BIG)
    assert store[1]["content"] == SMALL  # under threshold: untouched
    # sidecar exists, content-addressed
    side = os.path.join(fs._textrefs_dir(), ref["sha256"] + ".txt")
    assert os.path.exists(side)
    with open(side, encoding="utf-8") as f:
        assert f.read() == BIG
    # live form round-trips
    assert fs._inflate_text_refs(store) >= 1
    assert store[0]["content"] == BIG
    # the caller's live history was never mutated
    assert msgs[0]["content"] == BIG


def test_list_part_offload_and_missing_sidecar_degrade():
    _clear()
    msgs = [{"role": "tool", "content": [{"type": "text", "text": BIG}]}]
    store = fs._store_form_messages(msgs)
    ref = store[0]["content"][0]
    assert ref["type"] == "text_ref" and ref["str"] is False
    # missing sidecar → degrade to the preview note, not an exception
    os.remove(os.path.join(fs._textrefs_dir(), ref["sha256"] + ".txt"))
    n = fs._inflate_text_refs(store)
    assert n == 1
    assert store[0]["content"][0]["type"] == "text"
    assert "payload missing" in store[0]["content"][0]["text"]


def test_full_write_read_cycle_through_the_store():
    _clear()
    sid = "20250101-120000-tref1"
    msgs = [{"role": "user", "content": BIG}, {"role": "assistant", "content": "done"}]
    m._write_session(sid, msgs, {"title": "tref"})
    # the stored event carries a ref, not the payload
    conn = sdb._open()
    blob = conn.execute(
        "SELECT msg FROM session_events WHERE session_id = ? AND seq = 0",
        (sid,)).fetchone()[0]
    conn.close()
    import json
    ev = json.loads(blob)
    content = ev["content"]
    assert isinstance(content, list) and content[0]["type"] == "text_ref"
    assert len(json.dumps(content)) < 1000  # the event is tiny
    # load rehydrates
    d = m._load_session(sid)
    assert d["messages"][0]["content"] == BIG
    assert d["messages"][1]["content"] == "done"


def test_gc_removes_unreferenced_payloads():
    _clear()
    msgs = [{"role": "user", "content": BIG}]
    store = fs._store_form_messages(msgs)
    sha = store[0]["content"][0]["sha256"]
    # an unreferenced sibling payload
    orphan = os.path.join(fs._textrefs_dir(), ("f" * 64) + ".txt")
    with open(orphan, "w", encoding="utf-8") as f:
        f.write("orphan")
    fs._gc_textrefs({sha}, min_age_s=0.0)
    assert os.path.exists(os.path.join(fs._textrefs_dir(), sha + ".txt"))
    assert not os.path.exists(orphan)


def test_disable_flag_keeps_text_inline():
    _clear()
    os.environ["NX_SESSION_TEXT_REFS"] = "0"
    try:
        store = fs._store_form_messages([{"role": "user", "content": BIG}])
        assert store[0]["content"] == BIG  # inline, no sidecar written
    finally:
        os.environ.pop("NX_SESSION_TEXT_REFS", None)
    assert not os.path.isdir(fs._textrefs_dir()) or \
        not os.listdir(fs._textrefs_dir())


def test_threshold_env_override():
    _clear()
    os.environ["NX_SESSION_TEXT_REF_MIN"] = "10"
    try:
        store = fs._store_form_messages([{"role": "user", "content": SMALL}])
        assert store[0]["content"][0]["type"] == "text_ref"  # 14 bytes > 10
    finally:
        os.environ.pop("NX_SESSION_TEXT_REF_MIN", None)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS —", name)
