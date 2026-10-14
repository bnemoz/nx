"""Plan mode (nx/_plan.py): the /plan state machine, the run_tool write gate,
and the plan_edit tool — tested without a running model.

Plan mode exists because "write code only after thinking" enforced by prompt
nudges fails constantly (the reasoning-stall nudges fight it every turn); a
state machine that refuses mutating tool calls until /plan done is structural.
These tests pin the state machine: activation writes .nx/<slug>.md and injects
a plan block, the gate refuses mutating tools with the gate message but never
read-only ones, plan_edit edits the plan file (and only the plan file),
/plan done releases, and the state round-trips through the session JSON like
st._TODOS does.
"""
import io
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-plan-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m
from nx import _state as st
from nx import _plan
from nx._plan import slugify
from nx._repl_state import _ReplState, _save_current, _reload_metrics_totals


captured = io.StringIO()
real_print = print


def fake_print(*a, **kw):
    captured.write(" ".join(str(x) for x in a) + (kw.get("end", "\n")))


def cap():
    """Swap in fake_print, return the buffer (already truncated by callers)."""
    return captured


def reset_plan_state():
    st._PLAN_ACTIVE = False
    st._PLAN_GOAL = None
    st._PLAN_SLUG = None
    st._PLAN_STARTED = None
    st.PROJECT_ROOT = None


# --- slugify -----------------------------------------------------------------
assert slugify("Fix the login bug, quickly!") == "fix-the-login-bug-quickly"
assert slugify("3 bugs and 42 regressions") == "bugs-and-regressions", \
    "leading digits must not survive into the slug"
assert slugify("!!!") == "plan", "all-punctuation goal falls back to 'plan'"
assert slugify("one two three four five six seven eight nine ten") == \
    "one-two-three-four-five-six-seven-eight", "default cap is 8 words"
m.NX_PLAN_MAX_WORDS = 2
try:
    assert slugify("one two three four") == "one-two"
finally:
    m.NX_PLAN_MAX_WORDS = 8

# --- activation: file + message block ----------------------------------------
reset_plan_state()
root = tempfile.mkdtemp(prefix="nx-test-plan-root-")
st.PROJECT_ROOT = root
state = _ReplState([{"role": "system", "content": "sys"}], "sess-plan", False, [])
captured.truncate(0); captured.seek(0)
_real_print = print
builtins_print = __builtins__
import builtins
builtins.print = fake_print
try:
    _plan._cmd_plan("/plan implement the plan gate", state)
finally:
    builtins.print = _real_print
out = cap().getvalue()
assert "plan mode" in out and "implement-the-plan-gate" in out, out
pdir = os.path.join(root, ".nx")
ppath = os.path.join(pdir, "implement-the-plan-gate.md")
assert os.path.isfile(ppath), f"plan file not written: {ppath}"
with open(ppath) as f:
    body = f.read()
assert "implement the plan gate" in body
assert st._PLAN_ACTIVE and st._PLAN_SLUG == "implement-the-plan-gate"
assert state.messages[-1]["role"] == "user"
assert "Plan mode active" in state.messages[-1]["content"]
assert "plan_edit" in state.messages[-1]["content"]
assert "/plan done" in state.messages[-1]["content"]
assert m.plan_active() and m.plan_path() == ppath

# --- gate: mutating tools refused, read-only pass -----------------------------
for tool in ("write_file", "edit_file", "sed_edit",
             "delete_file", "run_bash", "memory_write", "memory_delete"):
    msg = m.check_gate(tool)
    assert msg and "Plan mode is active" in msg, f"{tool} must be gated"
    assert "/plan done" in msg and "plan_edit" in msg
for tool in ("read_file", "list_dir", "search_files", "git_status", "git_diff",
             "git_log", "memory_search", "plan_edit", "todo_write"):
    assert m.check_gate(tool) is None, f"{tool} must NOT be gated"
# run_tool itself refuses before dispatch (no classifier call, no approval):
result = m.run_tool("write_file", {"path": os.path.join(root, "evil.txt"), "content": "x"})
assert result.startswith("[Plan mode is active"), result
assert not os.path.exists(os.path.join(root, "evil.txt"))
assert m.check_gate("unknown_tool") is None, "unknown-tool handling stays with dispatch"

# --- plan_edit: the pressure valve --------------------------------------------
captured.truncate(0); captured.seek(0)
builtins.print = fake_print
try:
    r = m.plan_edit(old="(working — edit freely with plan_edit)",
                    new="step 1: gate; step 2: release")
finally:
    builtins.print = _real_print
assert r.startswith("plan updated (edited)"), r
with open(ppath) as f:
    assert "step 1: gate; step 2: release" in f.read()
# old must match exactly once
captured.truncate(0); captured.seek(0)
builtins.print = fake_print
try:
    r = m.plan_edit(old="step", new="X")
finally:
    builtins.print = _real_print
assert r.startswith("ERROR") and "2 time(s)" in r, r
# old_lines splice
captured.truncate(0); captured.seek(0)
builtins.print = fake_print
try:
    r = m.plan_edit(old_lines="1", new="# Plan: replaced headline")
finally:
    builtins.print = _real_print
assert r.startswith("plan updated (replaced lines 1-1)"), r
with open(ppath) as f:
    assert f.read().startswith("# Plan: replaced headline")
# full rewrite
captured.truncate(0); captured.seek(0)
builtins.print = fake_print
try:
    r = m.plan_edit(write="# v2 plan\n\n- only step")
finally:
    builtins.print = _real_print
assert r.startswith("plan updated (rewrote)"), r
# out-of-range old_lines and no-mode error
captured.truncate(0); captured.seek(0)
builtins.print = fake_print
try:
    r1 = m.plan_edit(old_lines="99", new="x")
    r2 = m.plan_edit(new="x")
finally:
    builtins.print = _real_print
assert r1.startswith("ERROR") and "out of range" in r1, r1
assert r2.startswith("ERROR") and "old" in r2, r2

# --- release (/plan done) and /plan show --------------------------------------
captured.truncate(0); captured.seek(0)
builtins.print = fake_print
try:
    _plan._cmd_plan("/plan", state)
finally:
    builtins.print = _real_print
assert "── plan ──" in cap().getvalue()
captured.truncate(0); captured.seek(0)
builtins.print = fake_print
try:
    _plan._cmd_plan("/plan done", state)
finally:
    builtins.print = _real_print
out = cap().getvalue()
assert "plan released" in out and "writes and bash are unlocked" in out, out
assert not st._PLAN_ACTIVE and m.plan_path() is None
assert os.path.isfile(ppath), "/plan done keeps the plan file"
assert m.check_gate("write_file") is None
# a fresh mutation is no longer refused by the *gate* — but still crosses the
# normal approval path, so just verify the gate no longer fires on it:
assert not m.run_tool("write_file", {"path": os.path.join(root, "ok.txt"),
                                     "content": "x"}).startswith("[Plan mode")

# --- /plan off alias + discard -------------------------------------------------
builtins.print = fake_print
try:
    _plan._cmd_plan("/plan second goal", state)
    _plan._cmd_plan("/plan off", state)
finally:
    builtins.print = _real_print
assert not st._PLAN_ACTIVE
_path2 = os.path.join(pdir, "second-goal.md")
assert os.path.isfile(_path2)
builtins.print = fake_print
try:
    _plan._cmd_plan("/plan second goal", state)  # re-activates the existing file
    _plan._cmd_plan("/plan discard", state)
finally:
    builtins.print = _real_print
assert not st._PLAN_ACTIVE
assert not os.path.exists(_path2), "discard deletes the plan file"

# --- /plan with no plan active; /plan twice is idempotent ----------------------
reset_plan_state()
captured.truncate(0); captured.seek(0)
builtins.print = fake_print
try:
    _plan._cmd_plan("/plan", state)
    _plan._cmd_plan("/plan done", state)
finally:
    builtins.print = _real_print
assert "no plan active" in cap().getvalue()

# --- _active_tools: plan_edit only while a plan is active ----------------------
names = [t["function"]["name"] for t in m._active_tools()]
assert "plan_edit" not in names
_plan_state = _ReplState([], "s2", False, [])
builtins.print = fake_print
reset_plan_state()
st.PROJECT_ROOT = tempfile.mkdtemp(prefix="nx-test-plan-gate-")  # never the cwd/repo
try:
    _plan._cmd_plan("/plan gate test", _plan_state)
finally:
    builtins.print = _real_print
names = [t["function"]["name"] for t in m._active_tools()]
assert "plan_edit" in names
_plan._cmd_plan("/plan done", _plan_state)
assert "plan_edit" not in [t["function"]["name"] for t in m._active_tools()]

# --- session persistence round-trip (like st._TODOS) ---------------------------
saved_meta = {}
class _FakeState:
    messages = []
    session_id = "sess-plan-persist"
    session_dirty = False
    history = []
st._METRICS = None
st._TODOS = []
st._PLAN_ACTIVE = True
st._PLAN_GOAL = "persist me"
st._PLAN_SLUG = "persist-me"
st._PLAN_STARTED = 123.5
_save_current(_FakeState())
m._PLAN_ACTIVE = False
m._PLAN_GOAL = None
m._PLAN_SLUG = None
m._PLAN_STARTED = None
_reload_metrics_totals("sess-plan-persist")
assert st._PLAN_ACTIVE and st._PLAN_GOAL == "persist me" and st._PLAN_SLUG == "persist-me"
assert st._PLAN_STARTED == 123.5
reset_plan_state()

# --- compress auto-pins the latest plan_edit turn ------------------------------
from nx._compress import compress  # noqa: E402
msgs = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "goal"},
    {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
        "function": {"name": "plan_edit", "arguments": "{\"new\":\"x\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "plan updated"},
    {"role": "user", "content": "old context that should fold"},
    {"role": "assistant", "content": "recent tail"},
]
st._PLAN_ACTIVE = True
st._TODOS = []
try:
    m.compress = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network"))
    # stub the summary call itself
    class _Resp:
        class choices(list):
            def __init__(self):
                super().__init__([type("M", (), {"message": type("Msg", (), {"content": "summary"})()})()])
        choices = choices()
        usage = None
    # compress() reaches st.client — fake it
    class _FakeCompletions:
        def create(self, **kw):
            # real assert: the plan-compress prompt must carry the plan_edit
            # marker (the old `or True` made this a tautology)
            assert "plan_edit" in kw["messages"][0]["content"]
            return _Resp()
    class _FakeClient:
        base_url = "http://x"
        chat = type("Chat", (), {"completions": _FakeCompletions()})()
    st.client = _FakeClient()
    st.MODEL = "fake"
    # keep=1 → head = all but the last turn; the plan_edit pair is in head
    n = len(msgs)
    compress(msgs, keep=2)
    # the plan_edit assistant turn must have survived the fold verbatim
    surviving = [x for x in msgs if x.get("role") == "assistant"
                 and any((tc.get("function") or {}).get("name") == "plan_edit"
                         for tc in (x.get("tool_calls") or []))]
    assert surviving, "plan_edit turn was folded away — compress must auto-pin it"
finally:
    st._PLAN_ACTIVE = False
    st.client = None

print("test_plan_mode: all assertions passed")


def test_plan_mode_state_machine():
    """pytest shim: every assertion above ran at import (standalone-assert
    style, like tests/test_sources.py) — a failure surfaces as a collection
    error. Re-assert the closing invariants so the suite counts this file."""
    assert not st._PLAN_ACTIVE
    assert m.check_gate("write_file") is None
