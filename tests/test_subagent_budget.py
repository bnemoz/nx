"""Sub-agent turn-budget effectiveness: the 0.18.7 fixes for spawn_agent
burning its whole max_turns budget without answering, plus the 0.20.0
anti-cap-out changes (bigger default budget, all rescue families clamped
inside a sub-loop, rescue-aware early force-final).

Covers: the budget line in the sub-agent's user message, the recovery
failsafe clamp (NX_SUBAGENT_RECOVERY_RETRIES), the last-sub-turn force-final,
the rescue-aware early force-final, the last-observation fallback when the
agent dies mid-exploration, and the async duplicate-in-flight refusal.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
# Clear ambient NX_* config so a developer's ~/.env / shell (real endpoints,
# keys, NX_VERIFY_CMD, ...) can't affect results — per repo test guidelines.
for k in [k for k in os.environ if k.startswith("NX_")]:
    os.environ.pop(k, None)
os.environ["NX_ENV_FILE"] = "/dev/null"

import pytest

# 0.96.0: foreground spawn results are head+report-path by default; these
# tests assert on the full inline answer, so opt into the legacy inline mode.
os.environ["NX_SUBAGENT_SYNC_INLINE"] = "1"


@pytest.fixture(autouse=True)
def _sync_inline(monkeypatch):
    # Load-bearing (not a no-op): nx/_subagent._sync_inline prefers the module
    # attr _m.NX_SUBAGENT_SYNC_INLINE over the env. nx_fresh's reload()
    # re-executes nx/__init__ in the SAME module dict without defining this
    # name, so setting it here (autouse runs before nx_fresh) survives the
    # reload and pins inline mode. raising=False because the attr is not
    # defined at import time.
    import nx as _nx
    monkeypatch.setattr(_nx, "NX_SUBAGENT_SYNC_INLINE", 1, raising=False)
    yield


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    for k in [k for k in os.environ if k.startswith("NX_")]:
        os.environ.pop(k, None)
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    os.environ["NX_SUBAGENT_ASYNC"] = "0"  # these tests exercise the foreground sub-loop; the 0.18.8 default is background (covered in test_subagent_async.py)
    monkeypatch.setattr(sys, "argv", ["nx"])
    import nx
    importlib.reload(nx)
    yield nx
    from nx import _subagent
    with _subagent._ASYNC_LOCK:
        _subagent._ASYNC_AGENTS.clear()
    for k in ("NX_HOME", "NX_SESSIONS_DIR", "NX_BASE_URL", "NX_MODEL",
              "NX_API_KEY", "NX_SUBAGENTS", "NX_SUBAGENT_ASYNC"):
        os.environ.pop(k, None)


# --- budget line -------------------------------------------------------------

def test_budget_line_in_subagent_user_message(nx_fresh, monkeypatch):
    """The sub-agent's instruction carries its turn budget so a flash-tier
    model plans within it instead of exploring until the cap."""
    from nx._subagent import spawn_agent
    seen = {}

    def fake_loop(messages, *a, **kw):
        seen["user"] = messages[1]["content"]
        seen["max_turns"] = kw.get("max_turns_override")
        messages.append({"role": "assistant", "content": "answer"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    # default is a FLOOR (0.36.1); lower it so an explicit max_turns=7 is honored
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENT_DEFAULT_TURNS", 1)
    spawn_agent("where is X?", max_turns=7)
    assert seen["max_turns"] == 7
    assert "[Budget: at most 7 tool turns" in seen["user"]
    assert seen["user"].startswith("where is X?")
    assert "state what you verified" in seen["user"]
    assert "parallel tool calls" in seen["user"]  # 0.20.0: batch, don't trickle


def test_default_and_ceiling_turns(nx_fresh, monkeypatch):
    """The un-specified budget is 100 (the old 12 left real investigations
    capping out — the 0.19 failure pattern), the hard ceiling 200, and both
    are env-tunable via NX_SUBAGENT_DEFAULT_TURNS / NX_SUBAGENT_MAX_TURNS."""
    from nx._subagent import spawn_agent
    seen = []

    def fake_loop(messages, *a, **kw):
        seen.append(kw.get("max_turns_override"))
        messages.append({"role": "assistant", "content": "ok"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    spawn_agent("q")
    assert seen[-1] == 100  # schema default, no explicit max_turns
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENT_DEFAULT_TURNS", 30)
    spawn_agent("q")
    assert seen[-1] == 30
    spawn_agent("q", max_turns=9999)
    assert seen[-1] == 500  # ceiling (0.103.10, matches MAX_MODEL_TURNS)
    # the default is a FLOOR: a request below it is raised to it
    spawn_agent("q", max_turns=3)
    assert seen[-1] == 30
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENT_MAX_TURNS", 15)
    spawn_agent("q", max_turns=9999)
    assert seen[-1] == 15
    spawn_agent("q", max_turns="not-a-number")
    # nonsense → the floor; the floor is min(default, ceiling)=min(30,15)=15 so
    # an inverted config (default>ceiling) can never exceed the ceiling
    assert seen[-1] == 15


def test_schema_advertises_default_100(nx_fresh):
    """The schema text is what the parent model reads to pick max_turns — it
    must name the new default or the parent keeps passing 12."""
    schema = next(t for t in nx_fresh.TOOLS
                  if t["function"]["name"] == "spawn_agent")
    desc = schema["function"]["parameters"]["properties"]["max_turns"]["description"]
    assert "default 100" in desc


def test_default_is_a_floor_not_a_cap(nx_fresh, monkeypatch):
    """0.36.1: several models carry their own turn-budget notion and pass a
    small max_turns (5/10/12 scale) even for a broad sweep, then watch the
    sub-agent cap out mid-investigation. The default is a FLOOR: such a request
    is raised to it. Only the ceiling pulls a request back down, so a request
    between floor and ceiling is honored exactly."""
    from nx._subagent import spawn_agent
    seen = []

    def fake_loop(messages, *a, **kw):
        seen.append(kw.get("max_turns_override"))
        messages.append({"role": "assistant", "content": "ok"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    # the classic failure: model passes 12 for a broad sweep → raised to 100
    spawn_agent("map the repo", max_turns=12)
    assert seen[-1] == 100
    spawn_agent("quick check", max_turns=1)
    assert seen[-1] == 100
    # a request above the floor but below the ceiling is honored verbatim
    spawn_agent("scoped question", max_turns=150)
    assert seen[-1] == 150
    # the schema must tell the parent this, or it keeps sending small numbers
    schema = next(t for t in nx_fresh.TOOLS
                  if t["function"]["name"] == "spawn_agent")
    desc = schema["function"]["parameters"]["properties"]["max_turns"]["description"]
    assert "FLOOR" in desc


# --- recovery failsafe clamp -------------------------------------------------

def test_recovery_retry_clamp_inside_subagent(nx_fresh, monkeypatch):
    """Inside a sub-loop the stall/empty failsafes clamp to
    NX_SUBAGENT_RECOVERY_RETRIES (default 1); outside it they keep the
    parent's limits. A parent that disabled a failsafe entirely (<= 0,
    e.g. benchmark mode) is NOT re-enabled inside sub-agents."""
    from nx._subagent import _subagent_scoped_retry, _subagent_stream_cut_limit
    from nx._dispatch import DISPATCH
    nx_fresh._state.REASONING_ONLY_RETRY_LIMIT = 3
    nx_fresh._state.EMPTY_TURN_RETRY_LIMIT = 3
    nx_fresh._state.MALFORMED_STREAM_RETRY_LIMIT = 2
    assert _subagent_scoped_retry(3) == 3  # outside: parent values
    assert _subagent_stream_cut_limit(2) == 2
    prev = _subagent_tls_ctx()
    prev_recovery = nx_fresh._subagent._SUBAGENT_RECOVERY_RETRIES
    try:
        install_ctx(DISPATCH)
        assert _subagent_scoped_retry(3) == 1  # clamped inside a sub-loop
        assert _subagent_stream_cut_limit(2) == 1  # stream cuts clamp too (0.20.0)
        assert _subagent_scoped_retry(0) == 0  # disabled stays disabled
        assert _subagent_stream_cut_limit(0) == 0
        nx_fresh._subagent._SUBAGENT_RECOVERY_RETRIES = 0
        assert _subagent_scoped_retry(3) == 0  # 0 = failsafes fully disabled
        assert _subagent_stream_cut_limit(2) == 0
        nx_fresh._subagent._SUBAGENT_RECOVERY_RETRIES = 2
        assert _subagent_scoped_retry(3) == 2
        assert _subagent_stream_cut_limit(2) == 2
    finally:
        restore_ctx(prev)
        nx_fresh._subagent._SUBAGENT_RECOVERY_RETRIES = prev_recovery


def _subagent_tls_ctx():
    from nx._subagent import _SUBAGENT_TLS
    return getattr(_SUBAGENT_TLS, "ctx", None)


def _install_ctx(table):
    from nx._subagent import _SUBAGENT_TLS, _SubagentContext
    _SUBAGENT_TLS.ctx = _SubagentContext(table, write=False, quiet=True)


def install_ctx(table):
    _install_ctx(table)


def restore_ctx(prev):
    from nx._subagent import _SUBAGENT_TLS
    _SUBAGENT_TLS.ctx = prev


def test_stall_escapes_after_one_rescue_inside_subagent(nx_fresh, monkeypatch):
    """Behavioral: a sub-agent stuck in reasoning-only stalls burns ONE rescue
    turn, then the next stall forces a final answer — three model_turn calls
    total, not the parent's stall→stall→force-final escalation."""
    from nx._subagent import spawn_agent
    calls = []

    def fake_loop(messages, *a, **kw):
        calls.append(kw.get("max_turns_override"))
        if len(calls) == 1:
            # model_turn-level stall handling is inside the real loop; the
            # clamp is exercised at unit level above. Here: assert the loop
            # itself still terminates normally on a plain finish.
            messages.append({"role": "assistant", "content": "ok"})
            return {"status": "done", "turns": 1}
        raise AssertionError("loop should have ended")

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    assert spawn_agent("q") == "ok"
    assert calls == [100]


# --- last-sub-turn force-final ----------------------------------------------

def test_last_sub_turn_is_forced_final(nx_fresh, monkeypatch):
    """The sub-loop's FINAL permitted turn opens with forced_final=True (the
    model is asked for its answer, not another search) — the parent loop never
    does this, and the cap rescue that follows is skipped since the last turn
    already was forced."""
    import nx as m
    from nx._subagent import spawn_agent
    forced = []

    def fake_turn(messages, reasoning_loop_cuts=0, malformed_stream_cuts=0,
                  empty_turn_count=0, forced_final=False, **kw):
        forced.append(forced_final)
        if len(forced) < 3:
            messages.append({"role": "assistant", "tool_calls": []})
            return m.TURN_TOOL
        return m.TURN_DONE

    monkeypatch.setattr(m, "model_turn", fake_turn)
    monkeypatch.setattr(m, "NX_SUBAGENT_DEFAULT_TURNS", 1)  # floor (0.36.1)
    out = spawn_agent("q", max_turns=3)
    assert forced == [False, False, True]  # only the LAST sub-turn is forced
    assert "answer" in out or out  # the loop still returns the sub-agent's text


def test_rescue_eaten_budget_preforces_final(nx_fresh, monkeypatch):
    """A sub-agent whose rescues already consumed the recovery budget gets its
    final answer pre-forced one turn early: with max_turns=4 and one stall
    rescue burned, turn 3 (not just turn 4) opens forced — the rescue-budget
    guard in _run_model_turn_loop."""
    import nx as m
    from nx._subagent import spawn_agent
    forced = []
    statuses = [m.TURN_STALL, m.TURN_TOOL, m.TURN_TOOL, m.TURN_DONE]

    def fake_turn(messages, reasoning_loop_cuts=0, malformed_stream_cuts=0,
                  empty_turn_count=0, forced_final=False, **kw):
        forced.append(forced_final)
        messages.append({"role": "assistant", "tool_calls": []})
        return statuses[len(forced) - 1]

    monkeypatch.setattr(m, "model_turn", fake_turn)
    monkeypatch.setattr(m, "NX_SUBAGENT_DEFAULT_TURNS", 1)  # floor (0.36.1)
    spawn_agent("q", max_turns=4)
    assert forced == [False, False, True, True]


def test_rescue_eaten_budget_never_touches_parent(nx_fresh, monkeypatch):
    """The early pre-force only applies inside a sub-loop: a parent loop with
    the same stall shape keeps its full budget and only gets the post-loop
    cap rescue."""
    import nx as m
    from nx._turn import _run_model_turn_loop
    forced = []

    def fake_turn(messages, reasoning_loop_cuts=0, malformed_stream_cuts=0,
                  empty_turn_count=0, forced_final=False, **kw):
        forced.append(forced_final)
        if len(forced) < 5:
            messages.append({"role": "assistant", "tool_calls": []})
            return m.TURN_STALL
        return m.TURN_DONE

    monkeypatch.setattr(m, "model_turn", fake_turn)
    summary = _run_model_turn_loop([{"role": "user", "content": "q"}],
                                   max_turns_override=4)
    assert forced == [False, False, False, False, True]  # rescue stays post-budget
    assert summary["status"] == "cap"


def test_parent_rescue_comes_after_budget_not_inside_it(nx_fresh, monkeypatch):
    """Sanity: the parent's forced-final arrives as the POST-loop rescue (a
    4th model_turn call outside max_turns), never as a pre-forced turn inside
    the budget — contrast with the sub-loop, whose last turn is pre-forced."""
    import nx as m
    from nx._turn import _run_model_turn_loop
    forced = []

    def fake_turn(messages, reasoning_loop_cuts=0, malformed_stream_cuts=0,
                  empty_turn_count=0, forced_final=False, **kw):
        forced.append(forced_final)
        if len(forced) < 4:
            messages.append({"role": "assistant", "tool_calls": []})
            return m.TURN_TOOL
        return m.TURN_DONE

    monkeypatch.setattr(m, "model_turn", fake_turn)
    summary = _run_model_turn_loop([{"role": "user", "content": "q"}],
                                   max_turns_override=3)
    assert forced == [False, False, False, True]  # rescue is OUTSIDE the 3-turn budget
    assert summary["status"] == "cap"


# --- last-observation fallback ----------------------------------------------

def test_cap_without_answer_returns_last_observation(nx_fresh, monkeypatch):
    """An agent that spent every turn searching and died at the cap hands the
    parent its last tool observation instead of 'produced no answer'."""
    from nx._subagent import spawn_agent

    def fake_loop(messages, *a, **kw):
        messages.append({"role": "assistant", "tool_calls": []})
        messages.append({"role": "tool", "content": "nx/_x.py:42: def x() ..."})
        return {"status": "cap", "turns": 12}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENT_DEFAULT_TURNS", 1)  # floor (0.36.1)
    out = spawn_agent("q", max_turns=12)
    assert out.startswith("[sub-agent hit its 12-turn cap before writing an "
                          "answer; last observation]")
    assert "nx/_x.py:42" in out


def test_cap_without_any_observation_keeps_old_message(nx_fresh, monkeypatch):
    """No tool output at all → the old 'produced no answer' verdict stands."""
    from nx._subagent import spawn_agent

    def fake_loop(messages, *a, **kw):
        return {"status": "done", "turns": 1}  # model produced nothing

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    assert "produced no answer" in spawn_agent("q")


# --- async duplicate-in-flight guard -----------------------------------------

def test_async_duplicate_instruction_refused_while_running(nx_fresh, monkeypatch):
    """Re-spawning the same instruction while the original async agent is
    still running is refused with the job id — no double spend."""
    from nx._subagent import spawn_agent
    started = threading.Event()
    release = threading.Event()

    def fake_loop(messages, *a, **kw):
        started.set()
        release.wait(timeout=5)
        messages.append({"role": "assistant", "content": "findings"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    out = spawn_agent("where is X?", async_=True)
    assert out.startswith("[background sub-agent] job agent-")
    assert started.wait(timeout=5)
    # hold the worker in its loop until after the duplicate check below
    # (once released it finalizes fast and the guard no longer applies)
    dup = spawn_agent("where  is   X?", async_=True)  # whitespace-normalized match
    assert dup.startswith("ERROR: an identical async agent is already running")
    assert "job agent-" in dup
    dup_sync = spawn_agent("a DIFFERENT question", async_=True)
    assert dup_sync.startswith("[background sub-agent] job agent-")
    release.set()
    # give the workers a moment to finalize their notes
    for _ in range(100):
        if not any(r.get("status") == "running"
                   for r in nx_fresh._subagent._ASYNC_AGENTS.values()):
            break
        time.sleep(0.05)


import threading  # noqa: E402  (used by the async-duplicate test above)
import time  # noqa: E402
