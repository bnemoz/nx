#!/usr/bin/env python3
"""0.40 (ROADMAP 1.4): one resume path for `nx --resume` and `/resume`.

The two entry points used to load a session independently, and the split was
user-visible:

- `/resume` handed the API store-form `image_ref` parts — the CLI path expanded
  them — which the API rejects with a 400 on the next turn.
- provider routing (an OpenRouter `/provider` choice) was re-applied only on the
  CLI path.
- `cwd` was saved on every autosave but restored by neither path.

They now share `nx/_resume.resume_session()`. These tests pin that the shared
loader does all four fixups, and that BOTH entry points route through it.

Standalone-runnable (python3 tests/test_resume_parity.py) and pytest-collectable.
"""
import base64
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("NX_SKIP_STARTUP_PROBES", "1")
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import nx as m  # noqa: E402
import nx._resume as resume  # noqa: E402
from nx._repl_state import _ReplState  # noqa: E402

# The package attribute `nx._repl` / `nx._sessions_cmd` is shadowed by the
# _repl FUNCTION of the same name, so monkeypatch targets must be the module
# objects, resolved via sys.modules.
_repl_mod = sys.modules["nx._repl"]
_cmd_mod = sys.modules["nx._sessions_cmd"]

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def _state(messages=None, session_id="start-id"):
    return _ReplState(messages or [{"role": "system", "content": "sys"}],
                      session_id, False, [])


def _isolate(tmp_path, monkeypatch):
    """Sessions + images under tmp_path (env-driven, no reload), and a project
    root with no memories so the resume note machinery is a no-op."""
    sdir = tmp_path / "sessions"
    monkeypatch.setenv("NX_SESSIONS_DIR", str(sdir))
    monkeypatch.setenv("NX_HOME", str(tmp_path))
    monkeypatch.setenv("NX_RESUME_CWD", "0")
    monkeypatch.setattr(m._state, "SESSION_HOME", str(tmp_path))
    empty_root = tmp_path / "proj"
    empty_root.mkdir(exist_ok=True)
    monkeypatch.setattr(m._state, "PROJECT_ROOT", str(empty_root))
    return sdir


def _seed_session(sid="sess-parity", *, cwd=None, source=None,
                  model=None, provider=None):
    """Write a session whose transcript carries a store-form image_ref part."""
    import nx._sessions_fs as fs
    content = [{"type": "text", "text": "[image: shot.png, 70 bytes]"},
               {"type": "image_url",
                "image_url": {"url": "data:image/png;base64,"
                              + base64.b64encode(_PNG).decode()}}]
    msgs = [{"role": "user", "content": "look at this"},
            {"role": "tool", "tool_call_id": "t1", "content": content}]
    meta = {"title": "parity"}
    if cwd is not None:
        meta["cwd"] = cwd
    if source is not None:
        meta["source"] = source
    if model is not None:
        meta["model"] = model
    if provider is not None:
        meta["provider"] = provider
    fs._write_session(sid, msgs, meta)
    # confirm the on-disk form really is a store-form ref (else the test is vacuous)
    data = fs._load_session(sid)
    assert data["messages"][1]["content"][1]["type"] == "image_ref", \
        "session was not written in store form — nothing to expand"
    return sid


# --- the shared loader does every fixup --------------------------------------
def test_resumed_messages_expands_image_ref_and_swaps_system(tmp_path, monkeypatch):
    """The 400-bug: a resumed transcript must be request-ready, not carry
    `image_ref` parts to the API."""
    _isolate(tmp_path, monkeypatch)
    sid = _seed_session()
    import nx._sessions_fs as fs
    data = fs._load_session(sid)
    msgs = resume.resumed_messages(data)
    tool = next(x for x in msgs if x.get("role") == "tool")
    parts = {p.get("type") for p in tool["content"]}
    assert "image_ref" not in parts, f"store-form ref survived: {tool['content']}"
    assert "image_url" in parts, f"image not expanded: {tool['content']}"
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == m.SYSTEM


def test_restore_cwd_chdirs_when_dir_exists(tmp_path, monkeypatch):
    here = tmp_path / "here"
    here.mkdir()
    target = tmp_path / "proj"
    target.mkdir()
    monkeypatch.chdir(here)
    monkeypatch.setattr(m._state, "PROJECT_ROOT", str(here))
    monkeypatch.setenv("NX_RESUME_CWD", "1")
    cwd, warning = resume.restore_cwd({"cwd": str(target)})
    assert cwd == str(target) and warning is None
    assert os.path.realpath(os.getcwd()) == os.path.realpath(str(target))
    # PROJECT_ROOT is recomputed for the new directory (every path fence,
    # allow-rule scope and memory scope keys off it)
    assert m._state.PROJECT_ROOT != str(here)


def test_restore_cwd_warns_for_missing_dir_and_stays(tmp_path, monkeypatch):
    here = tmp_path / "here"
    here.mkdir()
    monkeypatch.chdir(here)
    monkeypatch.setenv("NX_RESUME_CWD", "1")
    cwd, warning = resume.restore_cwd({"cwd": str(tmp_path / "gone")})
    assert cwd is None and "no longer exists" in warning
    assert os.path.realpath(os.getcwd()) == os.path.realpath(str(here))


def test_restore_cwd_respects_disable(tmp_path, monkeypatch):
    target = tmp_path / "proj"
    target.mkdir()
    here = tmp_path / "here"
    here.mkdir()
    monkeypatch.chdir(here)
    monkeypatch.setenv("NX_RESUME_CWD", "0")
    cwd, warning = resume.restore_cwd({"cwd": str(target)})
    assert cwd is None and warning is None
    assert os.path.realpath(os.getcwd()) == os.path.realpath(str(here))


def test_restore_endpoint_calls_restore_source_and_provider(monkeypatch):
    calls = {}
    src = types.SimpleNamespace(name="or")
    resume.st.SOURCES = {"or": src}
    resume.st.ACTIVE = src

    def fake_restore(name, model=None):
        calls["restore"] = (name, model)
        return True

    def fake_order(source, order):
        calls["order"] = (source, order)

    monkeypatch.setattr("nx._sources.restore_source", fake_restore)
    monkeypatch.setattr("nx._sources._set_provider_order", fake_order)
    out = resume.restore_endpoint({"source": "or", "model": "glm-x",
                                   "provider": ["io-net", "together"]})
    assert out == "or"
    assert calls["restore"] == ("or", "glm-x")
    assert calls["order"] == (src, ["io-net", "together"])


def test_restore_endpoint_no_provider_without_saved_list(monkeypatch):
    src = types.SimpleNamespace(name="local")
    resume.st.SOURCES = {"local": src}
    resume.st.ACTIVE = src
    monkeypatch.setattr("nx._sources.restore_source", lambda n, model=None: True)
    called = []
    monkeypatch.setattr("nx._sources._set_provider_order",
                        lambda *a: called.append(a))
    assert resume.restore_endpoint({"source": "local"}) == "local"
    assert called == [], "provider routing must not be touched with no saved list"


# --- resume_session wires the whole thing onto the state ---------------------
def test_resume_session_populates_state(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    sid = _seed_session()
    st = _state()
    ok = resume.resume_session(sid, st, show_transcript=False)
    assert ok is True
    assert st.session_id == sid
    assert st.session_dirty is False
    assert st.messages[0]["role"] == "system"
    tool = next(x for x in st.messages if x.get("role") == "tool")
    assert "image_ref" not in {p.get("type") for p in tool["content"]}
    from nx._metrics import _Metrics
    assert isinstance(m._state._METRICS, _Metrics)
    assert m._state._METRICS.session_id == sid


def test_resume_session_restores_cwd_end_to_end(tmp_path, monkeypatch):
    """The half neither old path implemented: the recorded cwd actually lands."""
    _isolate(tmp_path, monkeypatch)
    target = tmp_path / "recorded"
    target.mkdir()
    here = tmp_path / "here"
    here.mkdir()
    monkeypatch.chdir(here)
    monkeypatch.setattr(m._state, "PROJECT_ROOT", str(here))
    monkeypatch.setenv("NX_RESUME_CWD", "1")
    sid = _seed_session(sid="sess-cwd", cwd=str(target))
    st = _state()
    assert resume.resume_session(sid, st, show_transcript=False) is True
    assert os.path.realpath(os.getcwd()) == os.path.realpath(str(target))


def test_resume_session_missing_is_false_and_leaves_state(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    st = _state(messages=[{"role": "system", "content": "keepme"}])
    ok = resume.resume_session("does-not-exist", st, announce_missing=False)
    assert ok is False
    assert st.messages[0]["content"] == "keepme"
    assert st.session_id == "start-id"


# --- parity: BOTH entry points route through resume_session ------------------
def test_repl_cli_resume_calls_shared_loader(monkeypatch, tmp_path):
    """`nx --resume` must go through resume_session(), not re-implement it."""
    _isolate(tmp_path, monkeypatch)
    seen = {}

    def fake_resume(sid, st, **kw):
        seen["sid"] = sid
        return True

    monkeypatch.setattr(_repl_mod, "resume_session", fake_resume)
    monkeypatch.setattr(_repl_mod, "_initial_messages",
                        lambda: [{"role": "system", "content": "s"}])
    monkeypatch.setattr(_repl_mod, "_load_input_history", lambda: [])
    monkeypatch.setattr(_repl_mod, "_ensure_ctx_probe", lambda *a, **k: None)
    monkeypatch.setattr(m._state, "ACTIVE", None)
    monkeypatch.setattr(sys, "argv", ["nx", "--resume", "sess-xyz"])
    monkeypatch.setattr(m, "read_multiline",
                        lambda history=None, initial=None, **kw: "/quit")
    monkeypatch.setattr(m, "_banner", lambda: "")
    import builtins
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)
    m._repl()
    assert seen.get("sid") == "sess-xyz", "CLI --resume bypassed resume_session()"


def test_slash_resume_handler_calls_shared_loader(monkeypatch, tmp_path):
    """/resume must go through the SAME loader."""
    _isolate(tmp_path, monkeypatch)
    seen = {}

    def fake_resume(sid, st, **kw):
        seen["sid"] = sid
        return True

    monkeypatch.setattr(resume, "resume_session", fake_resume)
    monkeypatch.setattr(_cmd_mod, "_resolve_session", lambda t, s: "sess-abc")
    monkeypatch.setattr(_cmd_mod, "_list_sessions", lambda **k: [{"id": "sess-abc"}])
    monkeypatch.setattr(_cmd_mod, "_save_current", lambda *a, **k: None)
    import builtins
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)
    st = _state(session_id="current-id")
    _cmd_mod._cmd_resume_handler("/resume sess-abc", st)
    assert seen.get("sid") == "sess-abc", "/resume bypassed resume_session()"


def test_both_paths_share_one_implementation():
    """Structural parity guard: neither entry point keeps its own loader.
    If someone re-inlines the fixups in either place, this fails even when the
    behavioral tests above still pass on the happy path."""
    import inspect
    cli_src = inspect.getsource(_repl_mod._repl)
    slash_src = inspect.getsource(_cmd_mod._cmd_resume_handler)
    for src, label in ((cli_src, "nx --resume"), (slash_src, "/resume")):
        assert "resume_session(" in src, f"{label} no longer calls resume_session()"
        assert "_deserialize_image_parts(" not in src, \
            f"{label} re-implements image expansion instead of sharing it"
        assert "restore_source(" not in src, \
            f"{label} re-implements endpoint restore instead of sharing it"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
