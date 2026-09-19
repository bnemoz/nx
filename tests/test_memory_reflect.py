#!/usr/bin/env python3
"""Note kinds (0.68.0) + `/memory reflect`.

Kinds: the `_kind:` footer, its parsing, memory_write's kind param, and the
preference-kind injection in the session-start note. Reflect: the plan-line
parser and the no-model / thin-store guards (the model round-trips themselves
are stubbed, same convention as test_memory_tools.py).
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-reflect-")
os.environ["NX_HOME"] = _tmp
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _memory_cmd, _memory_reflect  # noqa: E402

_confirm_calls = []


def _fake_confirm(action, preview=None):
    _confirm_calls.append(action)
    return True


_saved_confirm = None
_saved_home = None


def _setup_function():
    global _saved_confirm, _saved_home
    _confirm_calls.clear()
    _saved_confirm = m._confirm
    _saved_home = m.st.SESSION_HOME
    m._confirm = _fake_confirm
    m.st.SESSION_HOME = _tmp


def _teardown_function():
    global _saved_confirm, _saved_home
    if _saved_confirm is not None:
        m._confirm = _saved_confirm
    if _saved_home is not None:
        m.st.SESSION_HOME = _saved_home
    _saved_confirm = None
    _saved_home = None


setup_function = lambda _function: _setup_function()
teardown_function = lambda _function: _teardown_function()


def _write_memory(title, body, scope="project", project_root=None, kind=""):
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = project_root
    try:
        return _memory_cmd.memory_write(title=title, body=body, scope=scope, kind=kind)
    finally:
        m.st.PROJECT_ROOT = old


def _fresh_root(prefix):
    root = tempfile.mkdtemp(prefix=prefix)
    m.st.SESSION_HOME = root  # isolate the global scope per test too
    return root


# --- kinds ---------------------------------------------------------------------

def test_write_with_kind_adds_footer():
    root = _fresh_root("nx-kind-")
    res = _write_memory("Kinded note", "body", project_root=root, kind="preference")
    assert "saved" in res
    path = _memory_cmd._iter_notes(root + "/.nx/memories")[0][1]
    content = open(path).read()
    assert "_kind: preference" in content
    assert _memory_cmd._note_kind(content) == "preference"


def test_write_without_kind_keeps_old_footer():
    root = _fresh_root("nx-nokind-")
    _write_memory("Plain note", "body", project_root=root)
    path = _memory_cmd._iter_notes(root + "/.nx/memories")[0][1]
    content = open(path).read()
    assert "_kind:" not in content
    assert _memory_cmd._note_kind(content) == ""


def test_unrecognized_kind_stored_kind_free():
    root = _fresh_root("nx-badkind-")
    _write_memory("Odd note", "body", project_root=root, kind="nonsense")
    path = _memory_cmd._iter_notes(root + "/.nx/memories")[0][1]
    assert "_kind:" not in open(path).read()


def test_note_kind_tolerates_old_and_hand_edited():
    assert _memory_cmd._note_kind("# T\n\nbody\n") == ""
    assert _memory_cmd._note_kind("# T\n\nbody\n\n---\n_scope: global  _saved: x\n") == ""
    assert _memory_cmd._note_kind("# T\n\n---\n_scope: global  _saved: x  _kind: gotcha\n") == "gotcha"


def test_preference_kind_injected_always_apply():
    root = _fresh_root("nx-prefinj-")
    # A preference older than 5 newer untagged notes: pre-0.68 injection
    # (newest-first cap 5) would have hidden it.
    _write_memory("User prefers uv", "always use uv", scope="global", project_root=None, kind="preference")
    for i in range(7):
        _write_memory(f"Project fact {i}", f"fact body {i}", project_root=root)
    note = _memory_cmd._load_memory_note(root)
    assert note is not None
    assert "User preferences (always apply):" in note["content"]
    assert "[global] User prefers uv" in note["content"]
    # the plain list is still capped at 5
    plain = [ln for ln in note["content"].split("\n") if ln.startswith("[project]")]
    assert len(plain) == 5, note["content"]


def test_no_preferences_no_section():
    root = _fresh_root("nx-nopref-")
    _write_memory("Only a fact", "body", project_root=root)
    note = _memory_cmd._load_memory_note(root)
    assert note is not None
    assert "User preferences" not in note["content"]


def test_kind_schema_present():
    import nx as _m
    schema = next(t for t in _m.TOOLS if t["function"]["name"] == "memory_write")
    props = schema["function"]["parameters"]["properties"]
    assert "kind" in props
    assert props["kind"]["enum"] == list(_memory_cmd.MEMORY_KINDS)


# --- reflect ---------------------------------------------------------------------

def test_reflect_parse_plan():
    known = {"a.md": None, "b.md": None}
    plan = _memory_reflect._parse_reflection(
        "NOTE a.md\nNOTE b.md\nWRITE global preference\nWRITE project bogus\n"
        "PROPOSE note a contradicts note b\nrandom chatter\n", known)
    assert plan.writes == [("global", "preference")]
    assert plan.sources == ["a.md", "b.md"]
    assert any("PROPOSE" in a for a in plan.advice)
    assert any("random chatter" in a for a in plan.advice)


def test_reflect_needs_model():
    root = _fresh_root("nx-refmodel-")
    _write_memory("A", "a", project_root=root)
    _write_memory("B", "b", project_root=root)
    old_client, old_model = m.st.client, m.st.MODEL
    m.st.client, m.st.MODEL = None, None
    try:
        _memory_reflect._cmd_memory_reflect("")  # prints, must not raise
    finally:
        m.st.client, m.st.MODEL = old_client, old_model


def test_reflect_thin_store_no_call():
    # Note files from other tests ride in via the shared global home, so pin
    # the collector: exactly one note in the index must short-circuit.
    root = _fresh_root("nx-refthin-")
    _write_memory("Only one", "body", project_root=root)
    old_collect = _memory_reflect._collect_reflection
    _memory_reflect._collect_reflection = lambda cap: old_collect(cap)[:1]
    old_client, old_model = m.st.client, m.st.MODEL
    m.st.client, m.st.MODEL = object(), "fake"
    called = []
    old_chat = _memory_reflect._reflect_chat
    _memory_reflect._reflect_chat = lambda *a, **k: called.append(1)
    try:
        _memory_reflect._cmd_memory_reflect("")
        assert called == [], "thin store must not reach the model"
    finally:
        _memory_reflect._collect_reflection = old_collect
        _memory_reflect._reflect_chat = old_chat
        m.st.client, m.st.MODEL = old_client, old_model


def test_reflect_end_to_end_stubbed():
    root = _fresh_root("nx-refe2e-")
    _write_memory("P1 uses uv", "project one prefers uv run", project_root=root)
    _write_memory("P2 uses uv", "project two also prefers uv run", project_root=root)
    old_client, old_model = m.st.client, m.st.MODEL
    m.st.client, m.st.MODEL = object(), "fake"
    calls = []

    def fake_chat(prompt, purpose):
        calls.append(purpose)
        if purpose == "memory_reflect_plan":
            return "NOTE 1.md\nWRITE global preference\nPROPOSE dup-ish notes"
        return "# Cross-project uv habit\n\nThe user prefers `uv run` everywhere.\n"

    old_chat = _memory_reflect._reflect_chat
    _memory_reflect._reflect_chat = fake_chat
    try:
        _memory_reflect._cmd_memory_reflect("")
        assert calls[0] == "memory_reflect_plan"
        assert any(c.startswith("memory_reflect_write_") for c in calls)
    finally:
        _memory_reflect._reflect_chat = old_chat
        m.st.client, m.st.MODEL = old_client, old_model
    # SESSION_HOME is a shared global other tests mutate; the run wrote into
    # the fresh root's memories dir, not _tmp's.
    gdir = os.path.join(root, "memories")
    notes = _memory_cmd._iter_notes(gdir)
    assert any("uv habit" in t for _n, _p, _m2, t, _c in notes), notes
    hit = [c for _n, _p, _m2, _t, c in notes if "uv habit" in _t][0]
    assert _memory_cmd._note_kind(hit) == "preference"
    assert "_scope: global" in hit


def test_reflect_no_signal_writes_nothing():
    root = _fresh_root("nx-refnone-")
    _write_memory("A", "alpha body", project_root=root)
    _write_memory("B", "beta body", project_root=root)
    old_client, old_model = m.st.client, m.st.MODEL
    m.st.client, m.st.MODEL = object(), "fake"
    old_chat = _memory_reflect._reflect_chat
    _memory_reflect._reflect_chat = lambda *a, **k: "PROPOSE nothing really"
    try:
        _memory_reflect._cmd_memory_reflect("")
    finally:
        _memory_reflect._reflect_chat = old_chat
        m.st.client, m.st.MODEL = old_client, old_model
    assert not os.path.isdir(os.path.join(_tmp, "memories"))
