"""0.39.0: session image storage — payloads in a content-addressed sidecar
(<sessions>/_images/<sha256>.b64), messages carry image_ref parts; resume
expands them; GC prunes to the union live-set; NX_SESSION_IMAGES=0 restores
inline-base64 storage.
"""
import base64
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR",
          "NX_VISION", "NX_IMAGE_MAX_BYTES", "NX_SESSION_IMAGES"):
    os.environ.pop(k, None)

import pytest

PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
PNG_BYTES = base64.b64decode(PNG_B64)
SHA = "6d1f8f5a4e51b3e5cb8e0a13a2d4b1f8c3b9a6d5e7f1a2b3c4d5e6f7a8b9c0d1"  # placeholder, computed in fixtures


def _parts(path="~/shot.png"):
    return [
        {"type": "text", "text": f"[image: {path}, 70 bytes]"},
        {"type": "image_url", "image_url":
            {"url": f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"}},
    ]


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    yield nx, tmp_path
    os.environ.pop("NX_SESSION_IMAGES", None)


def _msgs():
    return [
        {"role": "user", "content": "look at this"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "read_image", "arguments": '{"path": "shot.png"}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": _parts()},
    ]


def _sha():
    import hashlib
    return hashlib.sha256(PNG_BYTES).hexdigest()


def _images_dir(nx):
    return os.path.join(nx._sessions_fs._sessions_dir(), "_images")

def test_store_form_moves_payload_to_sidecar(nx_fresh):
    nx, tmp = nx_fresh
    msgs = _msgs()
    store = nx._sessions_fs._store_form_messages(msgs)
    side = os.path.join(_images_dir(nx), f"{_sha()}.b64")
    assert os.path.exists(side)  # payload written once
    with open(side) as f:
        assert f.read() == base64.b64encode(PNG_BYTES).decode()
    # live history untouched
    assert msgs[2]["content"][1]["type"] == "image_url"
    # store copy carries a ref with provenance from the sibling text part
    ref = store[2]["content"][1]
    assert ref["type"] == "image_ref"
    assert ref["sha256"] == _sha()
    assert ref["mime"] == "image/png" and ref["bytes"] == len(PNG_BYTES)
    assert ref["path"] == "~/shot.png"
    # and the file itself is payload-free
    assert PNG_B64 not in json.dumps(store)


def test_write_then_load_roundtrip(nx_fresh):
    nx, tmp = nx_fresh
    import nx._sessions_fs as fs
    msgs = _msgs()
    fs._write_session("sess-imgs", msgs, {"title": "t"})
    # the stored transcript carries refs, not payloads (read via the store)
    raw = fs._load_session("sess-imgs")
    import json as _json
    stored_blob = _json.dumps(raw)
    assert PNG_B64 not in stored_blob and "image_ref" in stored_blob
    # load returns the store form untouched (expansion is resume-only)
    data = raw
    assert data["messages"][2]["content"][1]["type"] == "image_ref"
    # resume-path expansion restores the live form
    assert fs._deserialize_image_parts(data["messages"]) == 1
    live = data["messages"][2]["content"][1]
    assert live["type"] == "image_url"
    assert base64.b64decode(live["image_url"]["url"].split(",", 1)[1]) == PNG_BYTES
    # ...and the next save round-trips through store form again (idempotent)
    fs._write_session("sess-imgs", data["messages"])
    reloaded = _json.dumps(fs._load_session("sess-imgs"))
    assert "image_ref" in reloaded


def test_missing_sidecar_degrades_to_note(nx_fresh):
    nx, tmp = nx_fresh
    import nx._sessions_fs as fs
    ref = {"type": "image_ref", "sha256": "f" * 64, "mime": "image/png",
           "bytes": 70, "path": "gone.png"}
    msgs = [{"role": "tool", "tool_call_id": "t", "content": [ref]}]
    fs._deserialize_image_parts(msgs)
    p = msgs[0]["content"][0]
    assert p["type"] == "text" and "missing" in p["text"]


def test_traversal_sha_refused(nx_fresh):
    nx, tmp = nx_fresh
    from nx._images import _ref_to_data_url_part
    for bad in ("../evil", "..\\evil", "a.b", "a/b", ""):
        assert _ref_to_data_url_part({"type": "image_ref", "sha256": bad}) is None


def test_bad_session_images_off_restores_inline(nx_fresh, monkeypatch):
    nx, tmp = nx_fresh
    import nx._sessions_fs as fs
    monkeypatch.setenv("NX_SESSION_IMAGES", "0")
    msgs = _msgs()
    store = fs._store_form_messages(msgs)
    assert store[2]["content"][1]["type"] == "image_url"  # inline, 0.38.x form
    assert not os.path.exists(_images_dir(nx))


def test_dedup_two_messages_same_image(nx_fresh):
    nx, tmp = nx_fresh
    import nx._sessions_fs as fs
    msgs = _msgs() + [{"role": "tool", "tool_call_id": "t2", "content": _parts()}]
    store = fs._store_form_messages(msgs)
    refs = [store[i]["content"][1] for i in (2, 3)]
    assert refs[0]["sha256"] == refs[1]["sha256"]
    assert len(os.listdir(_images_dir(nx))) == 1  # one payload file


def test_gc_keeps_referenced_and_recent_orphans(nx_fresh):
    nx, tmp = nx_fresh
    import nx._sessions_fs as fs
    from nx._images import _gc_image_sidecars
    fs._write_session("sess-gc", _msgs())
    d = _images_dir(nx)
    live = f"{_sha()}.b64"
    assert live in os.listdir(d)
    # an orphan older than the grace window goes away
    orphan = os.path.join(d, "a" * 64 + ".b64")
    with open(orphan, "w") as f:
        f.write("AAAA")
    old = __import__("time").time() - 120
    os.utime(orphan, (old, old))
    live_set = fs._scan_session_image_refs()
    removed = _gc_image_sidecars(live_set, min_age_s=60.0)
    assert removed == 1 and not os.path.exists(orphan)
    # a fresh orphan within the grace window survives
    fresh = os.path.join(d, "b" * 64 + ".b64")
    with open(fresh, "w") as f:
        f.write("AAAA")
    assert _gc_image_sidecars(live_set, min_age_s=60.0) == 0
    assert os.path.exists(fresh)
    # the referenced payload is never touched
    assert os.path.exists(os.path.join(d, live))


def test_gc_across_sessions_union(nx_fresh):
    nx, tmp = nx_fresh
    import nx._sessions_fs as fs
    from nx._images import _gc_image_sidecars
    fs._write_session("sess-a", _msgs())
    fs._write_session("sess-b", _msgs())  # same image, second session
    # this session's live set alone must not doom the other session's ref:
    # the GC scans all session files, so the shared payload survives
    live = fs._scan_session_image_refs()
    assert _gc_image_sidecars(live, min_age_s=60.0) == 0
    assert os.path.exists(os.path.join(_images_dir(nx), f"{_sha()}.b64"))


def test_flatten_uses_ref_path(nx_fresh):
    from nx._sessions_io import _flatten_message_text
    ref = {"type": "image_ref", "sha256": "a" * 64, "mime": "image/png",
           "bytes": 70, "path": "shot.png"}
    out = _flatten_message_text({"role": "tool", "tool_call_id": "t",
                                 "content": [ref]})
    assert "shot.png" in out


def test_preview_counts_refs(nx_fresh):
    from nx._images import _parts_preview
    ref = {"type": "image_ref", "sha256": "a" * 64, "mime": "image/png",
           "bytes": 70, "path": "shot.png"}
    out = _parts_preview([ref])
    assert "image attached" in out and "base64" in out
