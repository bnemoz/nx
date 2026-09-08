"""0.39.0: @image mentions — an @shot.png token in chat input attaches the
picture as content parts (NX_VISION on), names the gate (NX_VISION off), and
leaves non-image mentions untouched.
"""
import base64
import importlib
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
    os.environ.pop("NX_VISION", None)


def _img(tmp, name="shot.png"):
    p = tmp / name
    p.write_bytes(PNG_BYTES)
    return p


def test_image_mention_attaches_parts(nx_fresh):
    nx, tmp = nx_fresh
    _img(tmp)
    nx.NX_VISION = 1
    from nx._mentions import expand_at_mentions
    out = expand_at_mentions("what is in @shot.png?")
    assert isinstance(out, list)
    text_part = out[0]
    assert text_part["type"] == "text" and "what is in @shot.png?" in text_part["text"]
    img_part = next(p for p in out if p.get("type") == "image_url")
    assert base64.b64decode(img_part["image_url"]["url"].split(",", 1)[1]) == PNG_BYTES


def test_image_mention_without_vision_names_gate(nx_fresh):
    nx, tmp = nx_fresh
    _img(tmp)
    nx.NX_VISION = 0
    from nx._mentions import expand_at_mentions
    out = expand_at_mentions("look at @shot.png")
    assert isinstance(out, str)
    assert "NX_VISION" in out and "@shot.png" in out


def test_non_image_mentions_unchanged(nx_fresh, tmp_path):
    nx, tmp = nx_fresh
    (tmp / "a.txt").write_text("hello there\n")
    from nx._mentions import expand_at_mentions
    out = expand_at_mentions("see @a.txt please")
    assert isinstance(out, str)
    assert "hello there" in out and "--- @a.txt ---" in out


def test_mixed_mentions_str_plus_image(nx_fresh):
    nx, tmp = nx_fresh
    _img(tmp)
    (tmp / "a.txt").write_text("hello there\n")
    nx.NX_VISION = 1
    from nx._mentions import expand_at_mentions
    out = expand_at_mentions("compare @a.txt with @shot.png")
    assert isinstance(out, list)
    assert "compare @a.txt with @shot.png" in out[0]["text"]
    assert "--- @a.txt ---" in out[0]["text"]          # text block kept inline
    assert any(p.get("type") == "image_url" for p in out)


def test_failed_image_read_falls_back_to_error_text(nx_fresh):
    nx, tmp = nx_fresh
    # extension says png, size 0 → read_image refuses; mention must degrade
    (tmp / "empty.png").write_bytes(b"")
    nx.NX_VISION = 1
    from nx._mentions import expand_at_mentions
    out = expand_at_mentions("look at @empty.png")
    assert isinstance(out, str) and "ERROR" in out


def test_email_and_scope_guards_still_apply(nx_fresh, tmp_path):
    nx, tmp = nx_fresh
    _img(tmp, "pic.png")
    from nx._mentions import expand_at_mentions
    # email-like token never mentions
    assert expand_at_mentions("mail me at bob@x.com") == "mail me at bob@x.com"
    # outside project/home scope is not resolved (parent of the project dir)
    outside = tmp_path.parent / "elsewhere.png"
    outside.write_bytes(PNG_BYTES)
    assert expand_at_mentions(f"see @{outside}") == f"see @{outside}"
    # unknown token untouched
    assert expand_at_mentions("see @nope.png") == "see @nope.png"
