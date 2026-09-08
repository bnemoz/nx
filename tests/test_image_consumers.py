"""0.38.0: image-payload containment at every text channel a parts tool
result can reach — previews, the overrun auto-compress retry, batch stdout,
trace, and session flattening — plus call-time resolution of
NX_IMAGE_MAX_BYTES.
"""
import base64
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR",
          "NX_VISION", "NX_IMAGE_MAX_BYTES"):
    os.environ.pop(k, None)

import pytest

PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
PNG_BYTES = base64.b64decode(PNG_B64)


def _parts():
    return [
        {"type": "text", "text": "[image: /tmp/shot.png, 70 bytes]"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
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


def test_parts_preview_keeps_marker_drops_payload(nx_fresh):
    from nx._images import _parts_preview
    out = _parts_preview(_parts())
    assert isinstance(out, str) and out.startswith("[image: /tmp/shot.png")
    assert "image attached" in out
    assert PNG_B64 not in out and "data:image" not in out


def test_parts_preview_image_only(nx_fresh):
    from nx._images import _parts_preview
    out = _parts_preview([_parts()[1]])
    assert out.startswith("[image attached") and PNG_B64 not in out


def test_strip_image_parts_evicts_in_place(nx_fresh):
    from nx._images import _strip_image_parts
    msgs = [
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "read_image", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": _parts()},
        {"role": "tool", "tool_call_id": "t2", "content": "plain string"},
        {"role": "user", "content": [{"type": "text", "text": "parts user turn"}]},
    ]
    assert _strip_image_parts(msgs) == 1
    c = msgs[2]["content"]
    assert isinstance(c, list) and all(p.get("type") == "text" for p in c)
    assert "evicted" in c[1]["text"] and PNG_B64 not in str(c)
    # untouched messages really untouched
    assert msgs[3]["content"] == "plain string"
    assert msgs[4]["content"][0]["text"] == "parts user turn"


def test_overrun_compress_strips_images_even_without_fold(nx_fresh, monkeypatch):
    nx, tmp = nx_fresh
    calls = []

    def fake_compress(messages, keep=None, auto=False):
        calls.append(auto)
        return None  # nothing to fold — short history

    monkeypatch.setattr(nx, "compress", fake_compress)
    from nx._compress import _compress_for_overrun
    msgs = [{"role": "tool", "tool_call_id": "t1", "content": _parts()}]
    # strip alone must count as progress, or the retry never happens and the
    # turn dies with "could not free enough room" on an image-caused overrun
    assert _compress_for_overrun(msgs) is True
    assert calls == [True]
    assert all(p.get("type") == "text" for p in msgs[0]["content"])


def test_overrun_compress_still_folds(nx_fresh, monkeypatch):
    nx, tmp = nx_fresh
    monkeypatch.setattr(nx, "compress",
                        lambda messages, keep=None, auto=False: (1, 2, 10))
    from nx._compress import _compress_for_overrun
    assert _compress_for_overrun([{"role": "user", "content": "hi"}]) is True


def test_batch_trim_result_value_renders_parts(nx_fresh):
    from nx._batch import _trim_result_value
    out = _trim_result_value(_parts(), 500)
    assert isinstance(out, str) and "image attached" in out
    assert PNG_B64 not in out
    # strings still head+tail trimmed
    assert "elided" in _trim_result_value("x" * 2000, 100)


def test_flatten_names_image_only_tool_result(nx_fresh):
    from nx._sessions_io import _flatten_message_text
    blank = _flatten_message_text({"role": "tool", "tool_call_id": "t",
                                   "content": [_parts()[1]]})
    assert "image" in blank
    marked = _flatten_message_text({"role": "tool", "tool_call_id": "t",
                                    "content": _parts()})
    assert "shot.png" in marked
    assert _flatten_message_text({"role": "user", "content": "plain"}) == "plain"


def test_image_cap_resolved_at_call_time(nx_fresh, tmp_path, monkeypatch):
    nx, tmp = nx_fresh
    from nx._images import read_image
    big = tmp / "big.png"
    big.write_bytes(b"x" * 100)
    # env var overrides live — no module reload, no attribute patch
    monkeypatch.setenv("NX_IMAGE_MAX_BYTES", "10")
    out = read_image(str(big))
    assert out.startswith("ERROR") and "over the" in out
    monkeypatch.setenv("NX_IMAGE_MAX_BYTES", "not-an-int")  # bad value → default
    assert isinstance(read_image(str(big)), list)
