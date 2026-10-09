"""EXIT_STATE convention for write/implement-tier sub-agents.

The agent's final message ends with `EXIT_STATE: <STATE>`; the orchestrator
gets the machine state surfaced on the [Runtime note] and the /bg listing,
and the line is stripped from the prose. Missing/unrecognized lines fall back
to a heuristic (failure wording -> BLOCKED, else NEEDS_REVIEW).

Covers:
  (a) each of the 4 explicit states parsed (case-insensitive) and stripped;
  (b) markdown-bold tolerance (**EXIT_STATE:** DONE);
  (c) heuristic fallback — missing line, failure wording -> BLOCKED;
  (d) state recorded on the job record / list_async_agents.
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
    os.environ["NX_SUBAGENT_WRITE"] = "1"
    os.environ["NX_SUBAGENT_ASYNC"] = "0"
    os.environ["NX_SUBAGENT_SYNC_INLINE"] = "1"
    monkeypatch.setattr(sys, "argv", ["nx"])
    import nx
    importlib.reload(nx)
    import nx as _nx
    monkeypatch.setattr(_nx, "NX_SUBAGENT_SYNC_INLINE", 1, raising=False)
    yield nx
    from nx import _subagent
    with _subagent._ASYNC_LOCK:
        _subagent._ASYNC_AGENTS.clear()
    for k in [k for k in os.environ if k.startswith("NX_")]:
        os.environ.pop(k, None)
    os.environ["NX_ENV_FILE"] = "/dev/null"


# --- (a) _extract_exit_state: all four states, stripped, case-insensitive ----

def test_extract_explicit_states_stripped():
    from nx._subagent import _extract_exit_state
    for state in ("DONE", "DONE_WITH_DEVIATIONS", "BLOCKED", "NEEDS_REVIEW"):
        state_, prose, origin = _extract_exit_state(
            f"Report body.\nEXIT_STATE: {state}")
        assert state_ == state
        assert prose == "Report body."
        assert origin == "explicit"


def test_extract_case_insensitive_and_whitespace():
    from nx._subagent import _extract_exit_state
    state, prose, origin = _extract_exit_state(
        "Body text.\n\n   exit state:   done  \n")
    assert (state, prose, origin) == ("DONE", "Body text.", "explicit")


def test_extract_markdown_bold_tolerated():
    from nx._subagent import _extract_exit_state
    state, prose, origin = _extract_exit_state(
        "All edits verified.\n**EXIT_STATE:** DONE")
    assert (state, prose, origin) == ("DONE", "All edits verified.", "explicit")


def test_extract_unrecognized_state_falls_back():
    from nx._subagent import _extract_exit_state
    # unrecognized value: the line is NOT stripped, heuristic applies
    state, prose, origin = _extract_exit_state(
        "Everything worked.\nEXIT_STATE: SORT_OF_DONE")
    assert state == "NEEDS_REVIEW"
    assert origin == "heuristic"
    assert "SORT_OF_DONE" in prose


def test_extract_missing_line_needs_review():
    from nx._subagent import _extract_exit_state
    state, prose, origin = _extract_exit_state("Clean report, no marker.")
    assert (state, prose, origin) == ("NEEDS_REVIEW",
                                      "Clean report, no marker.", "heuristic")


def test_extract_failure_wording_blocked():
    from nx._subagent import _extract_exit_state
    state, prose, origin = _extract_exit_state(
        "The build failed and I could not fix it.")
    assert (state, prose, origin) == ("BLOCKED",
                                      "The build failed and I could not fix "
                                      "it.", "heuristic")


# --- (b) end-to-end: foreground write-tier spawn -----------------------------

def _fake_loop(nx_fresh, final_text):
    def fake(messages, *a, **kw):
        messages.append({"role": "assistant", "content": final_text})
        return {"status": "done", "turns": 2}
    return fake


def test_write_tier_sync_answer_stripped(nx_fresh, monkeypatch):
    from nx._subagent import spawn_agent
    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop",
                        _fake_loop(nx_fresh, "Edited nx/x.py.\nEXIT_STATE: DONE"))
    out = spawn_agent("task", tier="write")
    assert "EXIT_STATE" not in out
    assert "Edited nx/x.py." in out


def test_read_tier_left_alone(nx_fresh, monkeypatch):
    from nx._subagent import spawn_agent
    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop",
                        _fake_loop(nx_fresh, "Findings.\nEXIT_STATE: DONE"))
    out = spawn_agent("q")  # read tier
    assert "EXIT_STATE: DONE" in out  # prose untouched


# --- (d) async worker: record, report file, [Runtime note], /bg listing ------

def _run_job(nx_fresh, monkeypatch, final_text="Edited nx/x.py.\nEXIT_STATE: DONE",
             tier="write"):
    from nx._subagent import (_new_job_id, _ASYNC_AGENTS, _ASYNC_LOCK,
                              _run_async_agent)
    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop",
                        _fake_loop(nx_fresh, final_text))
    jid = _new_job_id()
    with _ASYNC_LOCK:
        _ASYNC_AGENTS[jid] = {"instruction": "task", "status": "running",
                              "answer": None, "started": 0.0,
                              "announced": False, "max_turns": 10,
                              "tier": tier, "turns": 0}
    _run_async_agent(jid, "task", 10, tier=tier)
    return jid


def test_async_record_and_report_carry_state(nx_fresh, monkeypatch, tmp_path):
    import nx._state as st
    from nx._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
    st.SESSION_HOME = str(tmp_path / "sh")
    jid = _run_job(nx_fresh, monkeypatch)
    with _ASYNC_LOCK:
        rec = _ASYNC_AGENTS[jid]
    assert rec["exit_state"] == "DONE"
    assert rec["exit_origin"] == "explicit"
    assert "EXIT_STATE" not in rec["answer"]
    assert rec["report"] and os.path.exists(rec["report"])
    body = open(rec["report"]).read()
    assert "- exit_state: DONE" in body
    assert "EXIT_STATE:" not in body.split("## Answer", 1)[1]


def test_runtime_note_and_listing_carry_state(nx_fresh, monkeypatch, tmp_path):
    import nx._state as st
    from nx._notices import get as notice_get
    from nx._subagent import (_PENDING_NOTICE_KEY, _stage_subagent_answer,
                              list_async_agents)
    st.SESSION_HOME = str(tmp_path / "sh")
    jid = _run_job(nx_fresh, monkeypatch)
    _stage_subagent_answer(jid)
    note = notice_get(_PENDING_NOTICE_KEY) or ""
    assert "[exit: DONE]" in note and "explicit" in note
    entry = [e for e in list_async_agents() if e["id"] == jid][0]
    assert entry["exit_state"] == "DONE"
    assert entry["exit_origin"] == "explicit"


def test_heuristic_blocked_recorded(nx_fresh, monkeypatch, tmp_path):
    import nx._state as st
    from nx._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
    st.SESSION_HOME = str(tmp_path / "sh")
    jid = _run_job(nx_fresh, monkeypatch,
                   final_text="The build failed and I could not fix it.")
    with _ASYNC_LOCK:
        rec = _ASYNC_AGENTS[jid]
    assert rec["exit_state"] == "BLOCKED"
    assert rec["exit_origin"] == "heuristic"
