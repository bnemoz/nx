"""F7: read_image — content-parts tool result, gating (NX_VISION + native
tool_calls confirmed), and the display path treating parts results opaquely.
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

# 1x1 red PNG
PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
PNG_BYTES = base64.b64decode(PNG_B64)


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
    for k in ("NX_HOME", "NX_SESSIONS_DIR", "NX_BASE_URL", "NX_MODEL", "NX_API_KEY",
              "NX_VISION", "NX_IMAGE_MAX_BYTES"):
        os.environ.pop(k, None)
    nx._state._NATIVE_TOOL_CALLS_CONFIRMED = False
    nx._state._SESSION_HAS_BACKGROUNDED = False


def test_read_image_returns_parts_array(nx_fresh):
    nx, tmp = nx_fresh
    img = tmp / "shot.png"
    img.write_bytes(PNG_BYTES)
    from nx._images import read_image
    out = read_image(str(img))
    assert isinstance(out, list) and len(out) == 2
    assert out[0]["type"] == "text" and "shot.png" in out[0]["text"]
    assert out[1]["type"] == "image_url"
    assert out[1]["image_url"]["url"].startswith("data:image/png;base64,")
    # the payload decodes back to the original bytes
    payload = out[1]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(payload) == PNG_BYTES


def test_read_image_rejects_non_images(nx_fresh):
    nx, tmp = nx_fresh
    from nx._images import read_image
    assert read_image("missing.png").startswith("ERROR")
    bad = tmp / "run.exe"
    bad.write_bytes(b"MZ")
    assert read_image(str(bad)).startswith("ERROR")
    assert read_image("").startswith("ERROR")
    # a text file with .png extension still reads (extension-trusted, like read_file)
    ok = tmp / "fake.png"
    ok.write_bytes(b"not really a png")
    out = read_image(str(ok))
    assert isinstance(out, list)


def test_read_image_size_cap(nx_fresh):
    nx, tmp = nx_fresh
    import nx._images as imgs
    img = tmp / "big.png"
    img.write_bytes(b"x" * 100)
    saved = imgs.IMAGE_MAX_BYTES
    imgs.IMAGE_MAX_BYTES = 10  # module-level, like other NX_* resolved caps
    try:
        out = imgs.read_image(str(img))
        assert out.startswith("ERROR") and "over the" in out
    finally:
        imgs.IMAGE_MAX_BYTES = saved


def test_schema_gating_vision_and_native(nx_fresh):
    nx, tmp = nx_fresh
    from nx._dispatch import _active_tools
    names = lambda: [t["function"]["name"] for t in _active_tools()]
    # default: no vision, no native confirmation → hidden
    assert "read_image" not in names()
    # vision on but no native tool_calls yet → still hidden
    nx.NX_VISION = 1
    assert "read_image" not in names()
    # native confirmed → visible
    nx._state._NATIVE_TOOL_CALLS_CONFIRMED = True
    assert "read_image" in names()
    # vision off again → hidden
    nx.NX_VISION = 0
    assert "read_image" not in names()
    nx.NX_VISION = 1
    # schema always present in the full TOOLS list (like wait_background)
    assert any(t["function"]["name"] == "read_image" for t in nx.TOOLS)


def test_run_tool_passthrough_of_parts_result(nx_fresh, capsys):
    nx, tmp = nx_fresh
    nx.NX_VISION = 1  # run_tool() refuses a gated read_image (0.28.1); this tests the parts passthrough
    nx._state._NATIVE_TOOL_CALLS_CONFIRMED = True
    nx._state.TOOL_OUTPUT_VISIBLE = True  # render the body so the display path is exercised
    img = tmp / "shot.png"
    img.write_bytes(PNG_BYTES)
    from nx._dispatch import run_tool
    from nx import _state as st
    out = run_tool("read_image", {"path": str(img)})
    assert isinstance(out, list)  # passes through unsanitized
    assert out[1]["image_url"]["url"].startswith("data:image/png;base64,")
    screen = capsys.readouterr().out
    assert "image attached" in screen  # display path shows a hint, not the b64
    assert "iVBOR" not in screen        # never dumps the payload to the terminal
    # /tooloutput last stash holds the display hint, not megabytes of base64
    assert "data:image" not in st._LAST_TOOL_OUTPUT
    # error results stay strings
    out2 = run_tool("read_image", {"path": str(tmp / "nope.png")})
    assert isinstance(out2, str) and out2.startswith("ERROR")


def test_read_image_not_available_to_subagent(nx_fresh):
    nx, tmp = nx_fresh
    from nx._subagent import _subagent_tools
    names = {t["function"]["name"] for t in _subagent_tools()}
    assert "read_image" not in names  # parts results are a parent-only capability
