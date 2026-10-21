#!/usr/bin/env python3
"""Behavioral tests for the /model command (model-first switching).

/model shares the switch machinery with /source but resolves by model id:
bare /model lists ids grouped by source, /model <id> finds the unique
source serving it, ambiguity and miss cases must NOT switch, and the
source-name / "default" escape hatches reuse /source's switch path.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-model-cmd-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
for k in list(os.environ):
    if k.startswith("NX_"):
        del os.environ[k]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx  # noqa: E402
from nx import _state as st  # noqa: E402
from nx._repl_settings_cmds import _cmd_model, _fuzzy_model_hits  # noqa: E402


class _FakeModels:
    def __init__(self, ids):
        self.data = [type("M", (), {"id": i})() for i in ids]

    def list(self, timeout=None):
        return self


class _FakeClient:
    def __init__(self, ids):
        self.models = _FakeModels(ids)


def _make_source(name, ids):
    from nx._sources import Source
    src = Source(name, "http://localhost:9/v1", "sk-noop",
                 model="glm-5.3" if name == "a" else None)
    src.client = _FakeClient(ids)
    return src


def _setup():
    st.SOURCES = {"a": _make_source("a", ["glm-5.3", "glm-5.3-flash"]),
                  "b": _make_source("b", ["qwen3.8-27b"])}
    st.SOURCE_ORDER = ["a", "b"]
    st.ACTIVE = st.SOURCES["a"]
    st.MODEL = "glm-5.3"


def _cleanup():
    st.SOURCES = {}
    st.SOURCE_ORDER = []
    st.ACTIVE = None
    st.MODEL = None


def test_bare_lists_ids_grouped_by_source():
    _setup()
    try:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _cmd_model("/model", None)
        out = buf.getvalue()
        assert "glm-5.3" in out and "qwen3.8-27b" in out
        # starred current model
        assert out.count("★") >= 1
    finally:
        _cleanup()


def test_switch_by_model_id_pins_and_switches(monkeypatch):
    _setup()
    try:
        monkeypatch.setattr(nx, "_banner", lambda: "", raising=False)
        called = []
        monkeypatch.setattr("nx._repl_settings_cmds.switch_source",
                            lambda name, model_override=None:
                            called.append((name, model_override)) or True)
        st_obj = type("S", (), {"session_dirty": False})()
        _cmd_model("/model qwen3.8-27b", st_obj)
        assert called == [("b", "qwen3.8-27b")]
        assert st_obj.session_dirty is True
    finally:
        _cleanup()


def test_id_on_default_source_pins_nothing(monkeypatch):
    _setup()
    try:
        monkeypatch.setattr(nx, "_banner", lambda: "", raising=False)
        called = []
        monkeypatch.setattr("nx._repl_settings_cmds.switch_source",
                            lambda name, model_override=None:
                            called.append((name, model_override)) or True)
        _cmd_model("/model glm-5.3", type("S", (), {"session_dirty": False})())
        # glm-5.3 IS source a's configured default → bare switch, no pin
        assert called == [("a", None)]
    finally:
        _cleanup()


def test_ambiguous_id_does_not_switch():
    _setup()
    try:
        st.SOURCES["b"] = _make_source("b", ["qwen3.8-27b", "glm-5.3"])
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _cmd_model("/model glm-5.3", type("S", (), {"session_dirty": False})())
        out = buf.getvalue()
        assert "several sources" in out
        assert st.ACTIVE.name == "a"  # no switch happened
    finally:
        _cleanup()


def test_unknown_id_does_not_switch():
    _setup()
    try:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _cmd_model("/model nope", type("S", (), {"session_dirty": False})())
        assert "no source serves" in buf.getvalue()
        assert st.ACTIVE.name == "a"
    finally:
        _cleanup()


def test_source_name_escape_hatch_switches_to_default(monkeypatch):
    _setup()
    try:
        monkeypatch.setattr(nx, "_banner", lambda: "", raising=False)
        called = []
        monkeypatch.setattr("nx._repl_settings_cmds.switch_source",
                            lambda name, model_override=None:
                            called.append((name, model_override)) or True)
        _cmd_model("/model b", type("S", (), {"session_dirty": False})())
        assert called == [("b", None)]
    finally:
        _cleanup()


def test_default_keyword_reswitches_active_source(monkeypatch):
    _setup()
    try:
        monkeypatch.setattr(nx, "_banner", lambda: "", raising=False)
        called = []
        monkeypatch.setattr("nx._repl_settings_cmds.switch_source",
                            lambda name, model_override=None:
                            called.append((name, model_override)) or True)
        _cmd_model("/model default", type("S", (), {"session_dirty": False})())
        assert called == [("a", None)]
    finally:
        _cleanup()


def test_two_arg_form_delegates_to_source(monkeypatch):
    _setup()
    try:
        monkeypatch.setattr(nx, "_banner", lambda: "", raising=False)
        called = []
        monkeypatch.setattr("nx._repl_settings_cmds.switch_source",
                            lambda name, model_override=None:
                            called.append((name, model_override)) or True)
        _cmd_model("/model a glm-5.3-flash",
                   type("S", (), {"session_dirty": False})())
        assert called == [("a", "glm-5.3-flash")]
    finally:
        _cleanup()


def test_fuzzy_model_hits_skips_ambiguous_ids():
    _setup()
    try:
        st.SOURCES["b"] = _make_source("b", ["qwen3.8-27b", "glm-5.3-flash"])
        # unique id → (source, id)
        assert _fuzzy_model_hits("qwen") == [("b", "qwen3.8-27b")]
        # id served by both sources → excluded
        assert _fuzzy_model_hits("glm-5.3-flash") == []
    finally:
        _cleanup()


def test_registry_has_model_and_source_rows():
    from nx._commands import COMMANDS, arg_options_map
    assert "/model" in COMMANDS and "/source" in COMMANDS
    assert COMMANDS["/model"].handler is not None
    assert COMMANDS["/model"].group == "model"
    # arg options callable resolves without sources configured
    opts = arg_options_map()
    assert "/model" in opts and "/source" in opts
    assert callable(opts["/model"])


if __name__ == "__main__":
    for _name in [n for n in sorted(globals()) if n.startswith("test_")]:
        globals()[_name]()
        print(f"  PASS {_name}")
