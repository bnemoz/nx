#!/usr/bin/env python3
"""Tests for auto context-compression (NX_AUTOCOMPRESS_PERCENT + /autocompress).

Covers:
  - compress(messages, auto=True) keeps ~1/3 of the body verbatim (more than
    the manual COMPRESS_KEEP=2), so in-progress work survives the fold.
  - _maybe_autocompress() fires only past the threshold, is a no-op below it,
    and is a no-op when the context window is unknown.
  - /autocompress bare shows the current %, /autocompress <n> sets it,
    /autocompress off disables, /autocompress on re-enables.
  - the REPL loop calls _maybe_autocompress after a settled turn and triggers
    a compress when over threshold.

Offline: compress() and model_turn() are stubbed so no network is needed.

Tier A (docs/test-architecture.md §4): nx is imported ONCE and never reloaded.
Every nx global this file used to assign at module scope (client/MODEL/ACTIVE/
AUTOCOMPRESS_PERCENT/compress/NX_ASYNC_COMPRESS/YOLO) is now monkeypatched per
test, so conftest's module-dict rewind (`globals` marker) is no longer needed.
The one reload that existed for a real reason — re-running nx/__init__.py's
import-time NX_AUTOCOMPRESS_PERCENT resolution to check env seeding + clamping —
is replaced by exec'ing the SHIPPED block, sliced verbatim from nx/__init__.py
between two unique anchors (asserted at collection time so the slice can never
silently drift), into a throwaway namespace.
"""
import builtins
import inspect
import io
import os
import sys
import tempfile

import pytest

_tmp = tempfile.mkdtemp(prefix="nx-autocomp-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
# conftest.py sets this for pytest runs; set it here too so the standalone
# invocation (python3 tests/test_autocompress.py) skips nx's startup probes —
# they would otherwise dial the configured endpoint(s) for seconds.
os.environ.setdefault("NX_SKIP_STARTUP_PROBES", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
import nx._state as st  # noqa: E402

# ---------------------------------------------------------------------------
# The import-time auto-compress knob resolution, sliced verbatim out of
# nx/__init__.py between two unique anchors. If the block moves, this assert
# fails loudly at collection instead of exec'ing a stale copy.
# ---------------------------------------------------------------------------
_BLOCK_START = "# --- auto context-compression ---"
_BLOCK_END = "# --- compression v2 (2.4, 0.47.0)"


def _resolve_env_int():
    """The `_env_int` helper the sliced block calls (it lives on the package or
    on _state depending on the split)."""
    for _mod in (m, st):
        _fn = getattr(_mod, "_env_int", None)
        if _fn is not None:
            return _fn
    raise AssertionError("nx._env_int not found on nx or nx._state — the "
                         "auto-compress env block's helper moved")


_SRC = inspect.getsource(m)
assert _BLOCK_START in _SRC and _BLOCK_END in _SRC, (
    "auto-compress block anchors moved in nx/__init__.py; update the slice "
    "here (its point is to exec the shipped block verbatim)")
_AUTOCOMPRESS_BLOCK = _SRC[_SRC.index(_BLOCK_START):_SRC.index(_BLOCK_END)]
_ENV_INT = _resolve_env_int()


def _exec_autocompress_block():
    """Re-run nx's import-time NX_AUTOCOMPRESS_PERCENT resolution (read, clamp
    0–100, derive the `/autocompress on` default) against the current env."""
    exec(_AUTOCOMPRESS_BLOCK, {"st": st, "os": os, "_env_int": _ENV_INT})


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMsg(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]

    def model_dump(self):
        return {"choices": [{"message": {"content": self.choices[0].message.content}}]}


class _FakeCompletions:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return _FakeResp("fake summary of the conversation")


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


class _FakeSource:
    def __init__(self, ctx_window):
        self._context_window = ctx_window
        self.name = "fake"

    def resolve_context_window(self, model=None, force=False):
        return self._context_window


@pytest.fixture
def fake_net(monkeypatch):
    """Pin client/MODEL to the fake so the REAL compress() never dials out, and
    ACTIVE to None so auto's keep count stays count-based (no token target
    derived from an ambient source's window)."""
    client = _FakeClient()
    monkeypatch.setattr(m, "client", client)
    monkeypatch.setattr(m, "MODEL", "test-model")
    monkeypatch.setattr(m, "ACTIVE", None)
    return client


def _make_stub_compress(calls):
    """A stub compress used by the guard/REPL tests. It records (keep, auto) and
    mimics the keep-count logic so _maybe_autocompress's return value is
    realistic."""
    def _stub_compress(msgs, keep=m.COMPRESS_KEEP, auto=False):
        calls.append((keep, auto))
        sys_msg = msgs[0] if msgs and msgs[0].get("role") == "system" else None
        body = msgs[1:] if sys_msg else msgs
        kept = max(keep, len(body) // 3) if auto else keep
        if len(body) <= kept:
            return None
        summarized = len(body) - kept
        msgs[:] = ([sys_msg] if sys_msg else []) + [
            {"role": "user", "content": "[Compressed context — fake summary]"}] + body[-kept:]
        return (kept, summarized, 42)

    return _stub_compress


def _boom(*a, **k):
    raise AssertionError("should not compress")


def _body_turns(n):
    return [{"role": "user" if j % 2 == 0 else "assistant", "content": f"turn {j // 2}"}
            for j in range(n)]


# ---------------------------------------------------------------------------
# 1. compress(auto=True) keeps a third of the body, not just COMPRESS_KEEP.
#    Uses the REAL compress() with a faked network call so we test the actual
#    keep-computation logic, not a stub.
# ---------------------------------------------------------------------------
def test_compress_auto_keeps_a_third(fake_net):
    # 30 body turns (60 messages) + system = 61. auto keep = 60 // 3 = 20.
    msgs_auto = [{"role": "system", "content": m.SYSTEM}] + _body_turns(60)
    result = m.compress(msgs_auto, auto=True)
    assert result is not None
    kept_n, summarized_n, _ = result
    assert kept_n == 20, f"auto should keep 20 (1/3 of 60), got {kept_n}"
    assert summarized_n == 40, f"auto should summarize 40, got {summarized_n}"

    # Same messages with auto=False (manual) → keeps only COMPRESS_KEEP (2)
    msgs_manual = [{"role": "system", "content": m.SYSTEM}] + _body_turns(60)
    result = m.compress(msgs_manual, auto=False)
    assert result is not None
    kept_n, summarized_n, _ = result
    assert kept_n == m.COMPRESS_KEEP, \
        f"manual should keep COMPRESS_KEEP ({m.COMPRESS_KEEP}), got {kept_n}"

    # A dangling tool result at the start of the requested tail cannot stay
    # verbatim, but it must be included in the summary rather than silently lost.
    msgs_dangling_tail = [
        {"role": "system", "content": m.SYSTEM},
        {"role": "user", "content": "older request"},
        {"role": "assistant", "content": "older response"},
        {"role": "tool", "tool_call_id": "call-1", "content": "critical tool result"},
        {"role": "user", "content": "latest request"},
    ]
    result = m.compress(msgs_dangling_tail)
    kept_n, summarized_n, _ = result
    summary_request = fake_net.chat.completions.requests[-1]["messages"][0]["content"]
    assert "critical tool result" in summary_request, "dropped tail tool result must be summarized"
    assert kept_n == 1 and summarized_n == 3, (kept_n, summarized_n)
    assert "last 1 turns kept verbatim" in msgs_dangling_tail[1]["content"], msgs_dangling_tail[1]

    # Small conversation: auto keep (body//3) should not exceed body size → None
    msgs_small = [{"role": "system", "content": m.SYSTEM},
                  {"role": "user", "content": "hi"},
                  {"role": "assistant", "content": "hello"}]
    result = m.compress(msgs_small, auto=True)
    assert result is None, f"too-small conversation should return None, got {result}"


# ---------------------------------------------------------------------------
# 3. _maybe_autocompress guards: disabled, unknown window, below threshold.
# ---------------------------------------------------------------------------
def test_maybe_autocompress_guards(monkeypatch):
    # Set up a fake ACTIVE with a known 10000-token window.
    monkeypatch.setattr(m, "ACTIVE", _FakeSource(10000))
    monkeypatch.setattr(m, "AUTOCOMPRESS_PERCENT", 85)
    # With the 0.103.5 async fold on by default the crossing STARTS a background
    # fold (returns False); the guard test pins the synchronous contract, so
    # disable async for it (read via getattr(nx, "NX_ASYNC_COMPRESS") first).
    monkeypatch.setattr(m, "NX_ASYNC_COMPRESS", False, raising=False)
    calls = []

    # A realistic message list (system + several turns) so the stub compress has
    # something to "fold".
    test_msgs = [{"role": "system", "content": m.SYSTEM}] + _body_turns(20)

    # (a) below threshold → no compress
    monkeypatch.setattr(m, "compress", _boom)
    fired = m._maybe_autocompress(test_msgs, prompt_tokens=5000)  # 50% < 85%
    assert fired is False, "should not fire below threshold"

    # (b) at/above threshold → compress fires
    monkeypatch.setattr(m, "compress", _make_stub_compress(calls))
    fired = m._maybe_autocompress(test_msgs, prompt_tokens=8500)  # 85% ≥ 85%
    assert fired is True, "should fire at threshold"
    assert calls and calls[-1][1] is True, "should call compress(auto=True)"

    # (c) disabled (0%) → no compress even at 100%
    monkeypatch.setattr(m, "AUTOCOMPRESS_PERCENT", 0)
    monkeypatch.setattr(m, "compress", _boom)
    fired = m._maybe_autocompress(test_msgs, prompt_tokens=10000)
    assert fired is False, "should not fire when disabled"

    # (d) unknown window (None) → no compress
    monkeypatch.setattr(m, "AUTOCOMPRESS_PERCENT", 85)
    monkeypatch.setattr(m, "ACTIVE", _FakeSource(None))
    fired = m._maybe_autocompress(test_msgs, prompt_tokens=9999)
    assert fired is False, "should not fire with unknown window"

    # (e) window = 0 → no compress
    monkeypatch.setattr(m, "ACTIVE", _FakeSource(0))
    fired = m._maybe_autocompress(test_msgs, prompt_tokens=9999)
    assert fired is False, "should not fire with zero window"

    # (f) ACTIVE = None → no compress
    monkeypatch.setattr(m, "ACTIVE", None)
    fired = m._maybe_autocompress(test_msgs, prompt_tokens=9999)
    assert fired is False, "should not fire with no active source"

    # (g) prompt_tokens = 0 / negative → no compress
    monkeypatch.setattr(m, "ACTIVE", _FakeSource(10000))
    assert m._maybe_autocompress(test_msgs, prompt_tokens=0) is False
    assert m._maybe_autocompress(test_msgs, prompt_tokens=-1) is False


# ---------------------------------------------------------------------------
# 2b. Auto-pin: the latest todo_write turn survives a fold verbatim.
#     todo_write output lives in the message body, so without this a
#     compression landing mid multi-step task folds the plan away and the
#     model re-derives or forgets it. compress() re-injects the most recent
#     todo_write call+result pair into the kept tail via the /ctx pin
#     machinery.
# ---------------------------------------------------------------------------
def test_todo_write_turn_auto_pinned(fake_net):
    msgs_todo = [{"role": "system", "content": m.SYSTEM}]
    for i in range(15):
        msgs_todo.append({"role": "user", "content": f"turn {i}"})
        if i == 3:  # a todo_write call + its rendered-list result, mid-history
            msgs_todo.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-todo-1", "type": "function", "function": {
                    "name": "todo_write", "arguments": '{"todos": []}'}}]})
            msgs_todo.append({"role": "tool", "tool_call_id": "call-todo-1",
                              "content": "[ ] step one\n[~] step two\n[ ] step three"})
        else:
            msgs_todo.append({"role": "assistant", "content": f"reply {i}"})
    # 31 body messages (14 plain turns + one call+result pair); manual keep=2 →
    # tail = last 2 (user+assistant of turn 14), so the todo pair (body idx 7,8)
    # lands in the foldable head.
    result = m.compress(msgs_todo)
    assert result is not None
    kept_n, summarized_n, _ = result
    # pair moved into the tail → tail = 2 verbatim + 2 pinned
    assert kept_n == 4, f"todo pair should be re-injected into tail (2+2), got {kept_n}"
    assert summarized_n == 31 - 4, f"expected 27 summarized, got {summarized_n}"
    # verbatim: the todo_write call and its full rendered list survive the fold
    tail_pair = msgs_todo[-2:]
    assert tail_pair[0].get("tool_calls") and \
        tail_pair[0]["tool_calls"][0]["function"]["name"] == "todo_write", tail_pair[0]
    assert "[~] step two" in tail_pair[1]["content"], tail_pair[1]

    # A todo_write already inside the kept tail is left alone (no duplicate pin,
    # no pin-skipped warning).
    msgs_todo_tail = [{"role": "system", "content": m.SYSTEM}]
    for i in range(15):
        msgs_todo_tail.append({"role": "user", "content": f"turn {i}"})
        if i == 14:  # in the last 2 body messages → stays verbatim on its own
            msgs_todo_tail.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-todo-2", "type": "function", "function": {
                    "name": "todo_write", "arguments": '{"todos": []}'}}]})
            msgs_todo_tail.append({"role": "tool", "tool_call_id": "call-todo-2",
                                   "content": "[ ] only step"})
        else:
            msgs_todo_tail.append({"role": "assistant", "content": f"reply {i}"})
    result = m.compress(msgs_todo_tail)
    assert result is not None
    kept_n, summarized_n, _ = result
    assert kept_n == 2, f"tail-dwelling todo pair should not be re-pinned, got kept={kept_n}"
    assert msgs_todo_tail[-2]["tool_calls"][0]["function"]["name"] == "todo_write"

    # No todo_write in the foldable head → nothing pinned, counts unchanged.
    msgs_no_todo = [{"role": "system", "content": m.SYSTEM}] + _body_turns(60)
    result = m.compress(msgs_no_todo)
    assert result is not None
    kept_n, summarized_n, _ = result
    assert kept_n == m.COMPRESS_KEEP, f"no todo → plain manual keep, got {kept_n}"


# ---------------------------------------------------------------------------
# 4. /autocompress command: show, set, off, on, invalid.
# ---------------------------------------------------------------------------
def _run_repl_commands(mp, commands):
    """Feed a list of commands to main() and return captured stdout."""
    captured = io.StringIO()

    def fake_print(*a, **kw):
        captured.write(" ".join(str(x) for x in a) + (kw.get("end", "\n")))

    mp.setattr(builtins, "print", fake_print)
    # Pin argv: under pytest, sys.argv is pytest's argv and nx's main() reads
    # it (it would die with "unknown command 'tests/...'").
    mp.setattr(sys, "argv", ["nx"])
    prompts = iter(list(commands) + ["/quit"])
    mp.setattr(m, "read_multiline", lambda history=None, initial=None: next(prompts))
    mp.setattr(m, "open_stream", lambda *a, **k: iter([]))  # no network
    mp.setattr(m, "model_turn", lambda *a, **k: m.TURN_DONE)
    mp.setattr(m, "compress", _make_stub_compress([]))  # /compress never hits network
    mp.setattr(m, "YOLO", True)
    m.main()
    return captured.getvalue()


def test_autocompress_command_parsing(monkeypatch):
    """The whole TEST 4 sequence runs through one test function so module
    import never calls m.main() (under pytest, sys.argv is pytest's argv —
    nx's main() would read it and fail with 'unknown command')."""
    monkeypatch.setattr(m, "ACTIVE", _FakeSource(10000))
    # Start from a known state.
    monkeypatch.setattr(m, "AUTOCOMPRESS_PERCENT", 85)
    monkeypatch.setattr(m, "NX_ASYNC_COMPRESS", False, raising=False)

    # bare /autocompress → shows current %
    out = _run_repl_commands(monkeypatch, ["/autocompress"])
    assert "auto-compress: 85%" in out, f"bare /autocompress should show 85%: {out!r}"

    # /autocompress 70 → sets to 70
    out = _run_repl_commands(monkeypatch, ["/autocompress 70"])
    assert "auto-compress: 70%" in out, f"/autocompress 70 should set 70%: {out!r}"
    assert m.AUTOCOMPRESS_PERCENT == 70

    # /autocompress off → disables
    out = _run_repl_commands(monkeypatch, ["/autocompress off"])
    assert "auto-compress: off" in out, f"/autocompress off should disable: {out!r}"
    assert m.AUTOCOMPRESS_PERCENT == 0

    # bare /autocompress when off → shows "off"
    out = _run_repl_commands(monkeypatch, ["/autocompress"])
    assert "auto-compress: off" in out, f"bare when off should show off: {out!r}"

    # /autocompress on → re-enables at the configured default (85 here)
    out = _run_repl_commands(monkeypatch, ["/autocompress on"])
    assert f"auto-compress: {m.AUTOCOMPRESS_DEFAULT}%" in out, \
        f"/autocompress on should restore the configured default: {out!r}"
    assert m.AUTOCOMPRESS_PERCENT == m.AUTOCOMPRESS_DEFAULT

    # /autocompress 0 → disables (alias for off)
    out = _run_repl_commands(monkeypatch, ["/autocompress 0"])
    assert "auto-compress: off" in out, f"/autocompress 0 should disable: {out!r}"
    assert m.AUTOCOMPRESS_PERCENT == 0

    # /autocompress 999 → out of range
    monkeypatch.setattr(m, "AUTOCOMPRESS_PERCENT", 50)
    out = _run_repl_commands(monkeypatch, ["/autocompress 999"])
    assert "out of range" in out, f"999 should be out of range: {out!r}"
    assert m.AUTOCOMPRESS_PERCENT == 50, "should be unchanged after invalid input"

    # /autocompress abc → usage message
    out = _run_repl_commands(monkeypatch, ["/autocompress abc"])
    assert "usage" in out.lower(), f"abc should show usage: {out!r}"


# ---------------------------------------------------------------------------
# 5. ENV var NX_AUTOCOMPRESS_PERCENT seeds the default.
# ---------------------------------------------------------------------------
def test_autocompress_env_seeds_default(monkeypatch):
    """TEST 5: NX_AUTOCOMPRESS_PERCENT env var seeds the default (runs inside a
    test function for the same argv reason as test_autocompress_command_parsing).

    Tier A: instead of importlib.reload(nx) (which re-executed ~100 modules to
    re-read three lines), the shipped import-time block is exec'd verbatim; the
    state it writes is registered with monkeypatch so teardown restores it (that
    is what the Tier-B module-dict rewind used to do). The NX_* env scrubbing is
    monkeypatch.delenv/setenv, whose undo is the universal env rewind."""
    for k in [k for k in os.environ if k.startswith("NX_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("NX_ENV_FILE", "/dev/null")
    monkeypatch.setenv("NX_SOURCE_LOCAL_BASE_URL", "http://localhost:8080/v1")
    # the del-loop above wiped conftest's NX_SKIP_STARTUP_PROBES; re-pin so the
    # REPL run below doesn't dial localhost:8080 (~3.5s per probe)
    monkeypatch.setenv("NX_SKIP_STARTUP_PROBES", "1")
    monkeypatch.setenv("NX_AUTOCOMPRESS_PERCENT", "60")
    # Pin the module state the exec'd block writes.
    monkeypatch.setattr(st, "AUTOCOMPRESS_PERCENT", st.AUTOCOMPRESS_PERCENT)
    monkeypatch.setattr(st, "AUTOCOMPRESS_DEFAULT", st.AUTOCOMPRESS_DEFAULT)
    monkeypatch.setattr(m, "ACTIVE", _FakeSource(10000))
    monkeypatch.setattr(m, "NX_ASYNC_COMPRESS", False, raising=False)

    _exec_autocompress_block()
    assert m.AUTOCOMPRESS_PERCENT == 60, f"env 60 → {m.AUTOCOMPRESS_PERCENT}"

    # /autocompress on restores the *configured* default (60), not a hardcoded 85.
    assert m.AUTOCOMPRESS_DEFAULT == 60, \
        f"AUTOCOMPRESS_DEFAULT should track the env default, got {m.AUTOCOMPRESS_DEFAULT}"
    out = _run_repl_commands(monkeypatch, ["/autocompress off", "/autocompress on"])
    assert "auto-compress: 60%" in out, \
        f"/autocompress on should restore the configured 60%, not 85: {out!r}"
    assert m.AUTOCOMPRESS_PERCENT == 60, \
        f"on should restore configured 60, got {m.AUTOCOMPRESS_PERCENT}"

    # clamp upper bound
    monkeypatch.setenv("NX_AUTOCOMPRESS_PERCENT", "150")
    _exec_autocompress_block()
    assert m.AUTOCOMPRESS_PERCENT == 100, \
        f"env 150 should clamp to 100, got {m.AUTOCOMPRESS_PERCENT}"

    # clamp lower bound
    monkeypatch.setenv("NX_AUTOCOMPRESS_PERCENT", "-5")
    _exec_autocompress_block()
    assert m.AUTOCOMPRESS_PERCENT == 0, \
        f"env -5 should clamp to 0, got {m.AUTOCOMPRESS_PERCENT}"
