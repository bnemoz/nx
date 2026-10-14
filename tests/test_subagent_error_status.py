"""Early-death honesty (0.121.1): a sub-agent whose turn loop returns
status="error" (API error, context overrun, stream timeout) used to be
announced to the parent as "done" — the reverse scan handed over whatever
prose preceded the break and `_answer_completion` classified it "done".

Covers:
  (a) status=error with prior assistant text → FAILED, never done/finished;
  (b) the "produced no answer after N turn(s); status=..." message is
      classified failure/partial (not announced as finished);
  (c) the quiet-thread rescue give-ups log an event even when quiet.
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

import pytest  # noqa: E402

# 0.96.0: foreground spawn results are head+report-path by default; these
# tests assert on the full inline answer, so opt into the legacy inline mode.
os.environ["NX_SUBAGENT_SYNC_INLINE"] = "1"


@pytest.fixture(autouse=True)
def _sync_inline(monkeypatch):
    # Load-bearing (see tests/test_subagent.py): nx/_subagent._sync_inline
    # prefers the module attr over the env. Patched in nx_fresh AFTER the
    # reload too (a reload re-resolves the attr from the scrubbed env, wiping
    # this patch); the autouse pre-patch covers the Tier-A files that don't.
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
    os.environ["NX_SUBAGENT_ASYNC"] = "0"
    monkeypatch.setattr(sys, "argv", ["nx"])
    import nx
    importlib.reload(nx)
    # Reload re-resolves NX_SUBAGENT_SYNC_INLINE from the scrubbed env (= 0),
    # wiping the autouse pre-patch — re-apply after (pattern of
    # test_subagent_exit_state.py).
    monkeypatch.setattr(nx, "NX_SUBAGENT_SYNC_INLINE", 1, raising=False)
    yield nx
    from nx import _subagent
    with _subagent._ASYNC_LOCK:
        _subagent._ASYNC_AGENTS.clear()
    for k in ("NX_HOME", "NX_SESSIONS_DIR", "NX_BASE_URL", "NX_MODEL",
              "NX_API_KEY", "NX_SUBAGENTS", "NX_SUBAGENT_ASYNC"):
        os.environ.pop(k, None)


def _tag(answer):
    """The tag _lean_result would stamp on this answer (failure surfaces)."""
    from nx._subagent import _answer_completion
    return {"error": "failed", "partial": "stopped early",
            "done": "finished"}[_answer_completion(answer)]


# --- (a) loop status=error with prior assistant text -------------------------

def test_error_status_with_prior_prose_is_reported_failed(nx_fresh, monkeypatch):
    """The agent emitted prose, then the loop died. The prose must NOT be
    passed off as the answer — the parent sees a failure with the reason."""
    from nx._subagent import spawn_agent, _answer_completion

    def fake_loop(messages, *a, **kw):
        messages.append({"role": "assistant",
                         "content": "I looked at nx/_todo.py and it does X."})
        return {"status": "error",
                "error": "API request failed: connection refused",
                "turns": 3}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    out = spawn_agent("q")
    assert out.startswith("ERROR: sub-agent failed")
    assert "API request failed: connection refused" in out
    assert ("[sub-agent stopped: API request failed: connection refused "
            "after 3 turn(s)]") in out
    # the pre-error prose survives as evidence only, under the failure header
    assert "I looked at nx/_todo.py" in out
    # classification: failure, never done
    assert _answer_completion(out) == "error"
    assert _tag(out) == "failed"


def test_error_status_without_prose_returns_failure_marker(nx_fresh, monkeypatch):
    """No prose at all: still a failure (with the last observation if any),
    never the generic 'produced no answer' misread as a clean finish."""
    from nx._subagent import spawn_agent, _answer_completion

    def fake_loop(messages, *a, **kw):
        messages.append({"role": "tool", "content": "nx/_x.py:42: def x() ..."})
        return {"status": "error", "error": "stream timeout: no chunks",
                "turns": 9}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    out = spawn_agent("q")
    assert out.startswith("ERROR: sub-agent failed: stream timeout")
    assert "[sub-agent stopped: stream timeout: no chunks after 9 turn(s)]" in out
    assert "Last observation:" in out and "nx/_x.py:42" in out
    assert _answer_completion(out) == "error"


def test_error_status_async_note_says_failed_not_finished(nx_fresh, monkeypatch):
    """The async [Runtime note] for a dead agent must say FAILED with the
    reason and the inline-retry hint — never 'finished'."""
    import time
    from nx._subagent import spawn_agent

    def fake_loop(messages, *a, **kw):
        messages.append({"role": "assistant", "content": "interim findings"})
        return {"status": "error",
                "error": "API request failed: 500 from endpoint",
                "turns": 2}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    out = spawn_agent("q", async_=True)
    assert out.startswith("[background sub-agent] job agent-")
    deadline = time.time() + 30
    note = None
    while note is None and time.time() < deadline:
        with nx_fresh._subagent._ASYNC_LOCK:
            rec = next(iter(nx_fresh._subagent._ASYNC_AGENTS.values()))
            if rec.get("status") != "running":
                note = getattr(nx_fresh.st, "_PENDING_SUBAGENT_NOTICE", None)
        if note is None:
            time.sleep(0.02)
    assert note, "the failed agent must stage a runtime note"
    assert "FAILED" in note and "finished" not in note
    assert "API request failed: 500 from endpoint" in note
    assert "Retry the task inline (async=false)" in note
    assert "[sub-agent stopped: API request failed: 500 from endpoint " \
           "after 2 turn(s)]" in note


def test_error_status_marks_marker_partial_without_error_prefix():
    """The in-band marker alone (no ERROR: prefix) must classify as failure —
    mirrors the existing cap/stopped-early marker scheme."""
    from nx._subagent import _answer_completion
    assert _answer_completion(
        "[sub-agent stopped: context window overflow after 4 turn(s)]\npartial"
    ) == "partial"


# --- (b) no-answer message ----------------------------------------------------

def test_no_answer_message_classified_partial(nx_fresh, monkeypatch):
    """'produced no answer after N turn(s); status=...' is a failure verdict —
    it used to fall through _answer_completion to 'done' and be announced as
    'finished' in the [Runtime note]."""
    from nx._subagent import spawn_agent, _answer_completion

    def fake_loop(messages, *a, **kw):
        return {"status": "done", "turns": 5}  # model produced nothing at all

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    out = spawn_agent("q")
    assert out.startswith("[sub-agent produced no answer after 5 turn(s); "
                          "status=done")
    assert _answer_completion(out) == "partial"
    assert _tag(out) == "stopped early"


def test_no_answer_marker_partial_without_prefix():
    from nx._subagent import _answer_completion
    assert _answer_completion(
        "[sub-agent produced no answer after 2 turn(s); status=error — "
        "try a narrower instruction]") == "partial"


# --- (c) quiet give-ups log an event -----------------------------------------

def test_reasoning_stall_give_up_logs_event_when_quiet(nx_fresh, monkeypatch):
    """The rescue-exhaustion give-up printed only when not quiet, so a
    background agent surrendered silently. It must reach the event log."""
    import nx
    from nx import _turn
    events = []
    monkeypatch.setattr(_turn, "_subagent_quiet", lambda: True)
    monkeypatch.setattr("nx._logging._log_event",
                        lambda d, p: events.append((d, p)))
    monkeypatch.setattr(nx.st, "REASONING_ONLY_RETRY_LIMIT", 1)
    messages = [{"role": "user", "content": "q"}]
    status = _turn._handle_loop_cut(messages, False, 1, 0, 0, None, 5, [], 0)
    # 0.154.6: rescue exhaustion with nothing usable produced is the honest
    # TURN_GAVE_UP (was TURN_DONE, which headless callers scored as a pass).
    assert status == nx.TURN_GAVE_UP
    assert events, "give-up must be logged even on a quiet thread"
    direction, payload = events[-1]
    assert direction == "turn"
    assert payload["kind"] == "give_up"
    assert payload["failure"] == "reasoning_only_stall"
    assert payload["attempts"] == 1 and payload["retry_limit"] == 1


def test_repeat_collapse_give_up_logs_event_when_quiet(nx_fresh, monkeypatch):
    """The special-token/repetition-collapse give-up is the third silent one."""
    from nx._turn import _handle_loop_cut
    from nx._special_tokens import LEAKED_TOKEN_RUN_LIMIT
    events = []
    monkeypatch.setattr(nx_fresh.st, "MALFORMED_STREAM_RETRY_LIMIT", 0)
    monkeypatch.setattr("nx._logging._log_event",
                        lambda d, p: events.append((d, p)))
    messages = [{"role": "user", "content": "q"}]
    status = _handle_loop_cut(messages, False, 0, LEAKED_TOKEN_RUN_LIMIT, 0,
                              None, 5, [], 0)
    # 0.154.6: collapse give-up discards the degenerated output — honest
    # TURN_GAVE_UP, not TURN_DONE.
    assert status == nx_fresh.TURN_GAVE_UP
    give_ups = [e for e in events if e[1].get("kind") == "give_up"]
    assert give_ups, "collapse give-up must be logged even when quiet"
    assert give_ups[-1][1]["failure"] == "special-token collapse"


def test_reasoning_stall_limit_zero_marks_failsafes_disabled(nx_fresh, monkeypatch):
    """retry_limit == 0 means failsafes are deliberately disabled (benchmark
    mode), not a rescue exhaustion — the event says so instead of claiming a
    rescue failure, and the ending is still logged (it ends a turn)."""
    from nx import _turn
    events = []
    monkeypatch.setattr(_turn, "_subagent_quiet", lambda: True)
    monkeypatch.setattr("nx._logging._log_event",
                        lambda d, p: events.append((d, p)))
    monkeypatch.setattr(nx_fresh.st, "REASONING_ONLY_RETRY_LIMIT", 0)
    messages = [{"role": "user", "content": "q"}]
    status = _turn._handle_loop_cut(messages, False, 1, 0, 0, None, 5, [], 0)
    assert status == nx_fresh.TURN_DONE
    give_ups = [e for e in events if e[1].get("kind") == "give_up"]
    assert give_ups, "the turn ending must be logged even when quiet"
    assert give_ups[-1][1]["failure"] == "reasoning_only_stall"
    assert give_ups[-1][1]["failsafes_disabled"] is True
    assert give_ups[-1][1]["retry_limit"] == 0
