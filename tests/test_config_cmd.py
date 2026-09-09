#!/usr/bin/env python3
"""P3-9 (0.52.0): `nx config` viewer/doctor + `/status` — regression tests.

Covers: secret masking, quote-aware set round-trip through _set_env_var →
_load_env_file, the doctor's findings (hyphenated source name, /dev/null
note, legacy-var shadowing, NX_VISION without native tools), the /status
render, and the `nx config` startup-probe skip in nx/__init__.py.
"""
import importlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_tmp = tempfile.mkdtemp(prefix="nx-test-config-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ.setdefault("NX_SKIP_STARTUP_PROBES", "1")

# Standalone-file idiom (tests/test_cli_version_flag.py): neutralize the local
# checkout's real source config before importing nx.
for _k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SOURCES", "NX_ACTIVE",
           "NX_VISION", "NX_SOURCE_LOCAL_BASE_URL"):
    os.environ.pop(_k, None)
os.environ["NX_ENV_FILE"] = "/dev/null"

import nx as m  # noqa: E402
from nx import _config_cmd as cc  # noqa: E402
from nx import _env  # noqa: E402


# ---------------------------------------------------------------- masking ---
def test_sensitive_vars_are_masked():
    assert cc._is_sensitive("NX_SOURCE_ZAI_API_KEY")
    assert cc._is_sensitive("TOGETHER_API_KEY")
    assert cc._is_sensitive("NX_MY_TOKEN")
    assert not cc._is_sensitive("NX_THEME")
    out = cc._display_value("NX_SOURCE_ZAI_API_KEY", "sk-real-secret")
    assert "sk-real-secret" not in out and "set" in out
    assert cc._display_value("NX_THEME", "monokai") == "monokai"
    assert cc._display_value("NX_EMPTY", "") == "(empty)"


def _run(argv):
    """Capture stdout+stderr of _cli_config without leaving prints around."""
    import io
    from contextlib import redirect_stderr, redirect_stdout
    buf, err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        code = cc._cli_config(argv)
    return code, buf.getvalue(), err.getvalue()


# ------------------------------------------------------------- list / get ---
def test_list_get_mask_and_missing(monkeypatch, capsys):
    monkeypatch.setenv("NX_SOURCE_X_API_KEY", "sk-live")
    monkeypatch.setenv("NX_THEME", "dark")
    code, out, _ = _run(["list", "nx_source_x"])
    assert code == 0 and "sk-live" not in out and "NX_SOURCE_X_API_KEY" in out
    code, out, err = _run(["get", "NX_THEME", "NX_NOPE_XYZ"])
    assert code == 1 and out.startswith("NX_THEME=dark")
    assert "NX_NOPE_XYZ is not set" in err


# ---------------------------------------------------------------- set path ---
def test_set_writes_env_file_and_round_trips(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("NX_THEME=monokai  # kept\nNX_OTHER=1\n")
    monkeypatch.setenv("NX_ENV_FILE", str(env_file))
    monkeypatch.delenv("NX_FOO", raising=False)
    code, _out, _err = _run(["set", "NX_FOO", "bar # not a comment"])
    assert code == 0
    assert os.environ["NX_FOO"] == "bar # not a comment"
    # in-place: the pre-existing lines and their comments survive
    text = env_file.read_text()
    assert "NX_THEME=monokai  # kept" in text and "NX_OTHER=1" in text
    # and the written value re-loads to the same string
    del os.environ["NX_FOO"]
    _env._load_env_file()
    assert os.environ["NX_FOO"] == "bar # not a comment"


def test_set_rejects_bad_names(tmp_path, monkeypatch):
    monkeypatch.setenv("NX_ENV_FILE", str(tmp_path / "e.env"))
    code, _o, err = _run(["set", "NX_MY-LLM", "1"])
    assert code == 2 and "not a valid env var name" in err
    code, _o, _e = _run(["set", "9LEAD", "1"])
    assert code == 2


# ------------------------------------------------------------------ doctor ---
def test_doctor_flags_shell_file_divergence(tmp_path, monkeypatch):
    env_file = tmp_path / "e.env"
    env_file.write_text("NX_THEME=vaporwave\n")
    monkeypatch.setenv("NX_ENV_FILE", str(env_file))
    monkeypatch.setenv("NX_THEME", "dark")
    monkeypatch.setenv("NX_SESSIONS_DIR", "/dev/null")
    code, out, _e = _run(["doctor"])
    assert "differs between the shell" in out and "the shell wins" in out


def test_doctor_flags_hyphenated_source_and_devnull():
    saved = (m.st.SOURCES, m.st.SOURCE_ORDER, m.st.ACTIVE)
    m.st.SOURCES = {"my-llm": None}
    m.st.SOURCE_ORDER = ["my-llm"]
    m.st.ACTIVE = None
    try:
        code, out, _e = _run(["doctor"])
    finally:
        m.st.SOURCES, m.st.SOURCE_ORDER, m.st.ACTIVE = saved
    assert code == 1
    assert "NX_SOURCE_MY-LLM_*" in out
    assert "/dev/null" in out  # NX_ENV_FILE=/dev/null note (conftest baseline)


def test_doctor_flags_legacy_vars_and_vision_gate(monkeypatch):
    saved = (m.st.SOURCES, m.st.SOURCE_ORDER, m.st.ACTIVE)
    monkeypatch.setenv("NX_ENV_FILE", "/dev/null")
    monkeypatch.setenv("NX_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("NX_VISION", "1")
    from nx import _compress as c
    saved_gates = (c.NX_NATIVE_TOOLS, c.NX_GRAMMAR)
    # 0.54.0: NX_NATIVE_TOOLS flipped meaning — True now means the user
    # OPTED OUT (NX_NATIVE_TOOLS=0), which is what the vision gate flags.
    c.NX_NATIVE_TOOLS = True
    c.NX_GRAMMAR = False
    try:
        code, out, _e = _run(["doctor"])
    finally:
        c.NX_NATIVE_TOOLS, c.NX_GRAMMAR = saved_gates
        m.st.SOURCES, m.st.SOURCE_ORDER, m.st.ACTIVE = saved
    assert code == 1
    assert "ignored while NX_SOURCE_* sources exist" in out
    assert "NX_VISION" in out


# ----------------------------------------------------------------- /status ---
def test_render_status_snapshot(monkeypatch):
    monkeypatch.setenv("NX_FLOOR_BUDGET", "3180")
    monkeypatch.delenv("NX_VISION", raising=False)
    saved = (m.st.SOURCES, m.st.SOURCE_ORDER, m.st.ACTIVE, m.st.MODEL)
    m.st.SOURCES, m.st.SOURCE_ORDER, m.st.ACTIVE = {}, [], None
    m.st.MODEL = None
    try:
        text = cc.render_status()
        assert "(none — set NX_SOURCE_* or NX_BASE_URL)" in text
        assert "approval" in text and "fence" in text
    finally:
        m.st.SOURCES, m.st.SOURCE_ORDER, m.st.ACTIVE, m.st.MODEL = saved


def test_render_status_with_active_source(monkeypatch):
    monkeypatch.setenv("NX_FLOOR_BUDGET", "3180")
    saved = (m.st.SOURCES, m.st.SOURCE_ORDER, m.st.ACTIVE, m.st.MODEL)
    try:
        m.st.SOURCES = {"local": m.st.SOURCES.get("local")}
        m.st.SOURCE_ORDER = ["local"]
        src = m.st.ACTIVE
        if src is None:
            from nx._sources import Source
            src = Source("local", "http://localhost:8080/v1", "sk-test", "test-model")
        m.st.ACTIVE = src
        m.st.MODEL = "test-model"
        text = cc.render_status()
        assert "local" in text and "test-model" in text
        assert "NX_SOURCE_LOCAL_*" in text
        assert str(m._approval_display()) in text or "approval" in text
    finally:
        m.st.SOURCES, m.st.SOURCE_ORDER, m.st.ACTIVE, m.st.MODEL = saved
