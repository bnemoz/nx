"""P1/P2 AUTOWAKE_SPEC Phase E reproducer: per-task NOTICE OWNERSHIP under
`nx run --parallel 2`.

With --parallel > 1 the pending-note channel was process-global: an async
sub-agent finishes on its own worker thread, which has no notice frame, so its
`[Runtime note]` landed in the module store — where whichever sibling task
flushed first drained it into ITS transcript. The owning task's `announced`
mark was already burned at staging, so it could never re-receive the answer
(the failure _batch.py's docstring documented and this phase fixes).

The fix has two halves, both exercised here:
1. a batch task installs its own notice frame for its whole task (not just one
   turn loop) and PUBLISHES it under its thread ident (_notices.publish);
2. the async agent's spawn records the spawning turn's thread ident, and its
   finalize stages through _notices.stage_for_owner, so the note lands in the
   OWNER's frame only.

Offline: the agent's sub-loop is faked; only the staging path is real.
"""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SOURCES", "NX_ACTIVE"):
    os.environ.pop(k, None)

import pytest


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    monkeypatch.chdir(tmp_path)
    import nx
    from nx._sources import _discover_sources, switch_source
    _discover_sources()
    assert switch_source(nx._state.SOURCE_ORDER[0]), "fixture source wiring failed"
    yield nx, tmp_path


def _write_tasks(tmp, lines):
    p = tmp / "tasks.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def _capture(monkeypatch):
    import io
    buf, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    monkeypatch.setattr(sys, "stderr", err)
    return buf, err


def _fake_agent_worker(nx, monkeypatch, holder):
    """Replace the async agent's BODY with one that finishes immediately and
    stages its answer through the real ownership-aware staging path."""
    from nx import _subagent as sa

    def fake_run(job_id, instruction, max_turns, tier="read", scope=None, *,
                 worktree=None, announce=True, handle=None):
        time.sleep(holder.get("delay", 0.0))
        with sa._ASYNC_LOCK:
            rec = sa._ASYNC_AGENTS.get(job_id) or {}
            rec.update({"status": "done", "answer": f"answer for {instruction}"})
            sa._ASYNC_AGENTS[job_id] = rec
        sa._stage_notice_owner(rec, sa._PENDING_NOTICE_KEY,
                               f"[Runtime note: agent for {instruction} done]")
        if handle is not None:
            handle.done.set()

    monkeypatch.setattr(sa, "_run_async_agent", fake_run)
    return sa


# --- the reproducer -----------------------------------------------------------

def test_parallel_batch_async_note_lands_only_in_its_own_task(nx_fresh, tmp_path,
                                                             monkeypatch):
    """Two batch tasks run concurrently; each spawns one async agent. Each
    task's transcript must contain ONLY its own agent's note (P1)."""
    nx, tmp = nx_fresh
    from nx import _subagent as sa
    sa._ASYNC_AGENTS.clear()

    holder = {"delay": 0.05}
    _fake_agent_worker(nx, monkeypatch, holder)

    src = nx._sources.Source("fake", "http://127.0.0.1:9/v1", "k", "fake-model")
    monkeypatch.setattr(nx.st, "SOURCES", {**nx.st.SOURCES, "fake": src})
    monkeypatch.setattr(nx.st, "SOURCE_ORDER", list(nx.st.SOURCE_ORDER) + ["fake"])
    monkeypatch.setattr(nx.st, "ACTIVE", src)

    transcripts = {}
    lock = threading.Lock()

    def fake_loop(messages, force_final=False, recovery_sampling=False,
                  on_step=None, max_turns_override=None):
        task = nx._batch._batch_task()
        tid = task.get("id")
        sa.spawn_agent_job("instr-" + str(tid), 5)
        # let both tasks' agents finish, then flush (the shape of a real
        # turn's first-tool-of-turn flush)
        time.sleep(0.3)
        nx._runtime_notes._flush_pending_background_notice(messages)
        with lock:
            transcripts[tid] = [m.get("content") for m in messages
                                if isinstance(m.get("content"), str)
                                and m.get("content").startswith(
                                    "[Runtime note: agent for")]
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx, "_run_model_turn_loop", fake_loop)

    f = _write_tasks(tmp, [
        '{"instruction": "alpha", "id": "alpha"}',
        '{"instruction": "beta", "id": "beta"}',
    ])
    buf, err = _capture(monkeypatch)
    from nx._batch import run_batch
    code = run_batch(["-f", f, "--parallel", "2"])
    assert code == 0, f"exit={code} stderr={err.getvalue()!r}"

    assert set(transcripts) == {"alpha", "beta"}, transcripts
    for tid, notes in transcripts.items():
        assert notes, f"{tid} never received its own agent's note: {transcripts}"
        assert all(tid in n for n in notes), \
            f"{tid} received a SIBLING's note: {notes}"


def test_publish_stage_and_unpublish_roundtrip(nx_fresh):
    """The frame-routing primitive itself: a published frame receives notes
    from a frameless thread; an unpublished ident returns False (the caller
    must fall back to the module store), and unpublish is idempotent."""
    nx, _ = nx_fresh
    from nx import _notices as nz
    prev_cur = nz._CURRENT.get()
    saved = {s: getattr(nx.st, s, None) for s in nz.SLOTS}
    try:
        nz._CURRENT.set(None)
        for s in nz.SLOTS:
            setattr(nx.st, s, None)
        assert nz.stage_for_owner(threading.get_ident(),
                                  "_PENDING_SUBAGENT_NOTICE",
                                  "[Runtime note: nobody home]") is False
        assert nx.st._PENDING_SUBAGENT_NOTICE is None  # no stray write

        fr = nz.install_notice_ctx()
        ident = nz.publish(fr)
        try:
            results = []

            def worker():
                ok = nz.stage_for_owner(ident, "_PENDING_SUBAGENT_NOTICE",
                                        "[Runtime note: owner-only]")
                ok2 = nz.stage_for_owner(ident, "_PENDING_SUBAGENT_NOTICE",
                                         "[Runtime note: owner-only]")
                results.append((ok, ok2))

            t = threading.Thread(target=worker)
            t.start()
            t.join(5)
            assert results == [(True, True)]
            # Landed in the owner's frame, exactly once (deduped), and NOT in
            # the module store (where a sibling task's flush would find it).
            assert nz.get("_PENDING_SUBAGENT_NOTICE") == \
                "[Runtime note: owner-only]"
            assert nx.st._PENDING_SUBAGENT_NOTICE is None
        finally:
            nz.unpublish(ident, fr)
            nz._CURRENT.set(fr)
            nz.pop_notice_ctx(fr)
        # After unpublish the owner's frame is invisible to foreign threads.
        assert nz.stage_for_owner(ident, "_PENDING_SUBAGENT_NOTICE",
                                  "[Runtime note: too late]") is False
    finally:
        nz._CURRENT.set(prev_cur)
        for s, v in saved.items():
            setattr(nx.st, s, v)


# --- P2: per-task background-job registry -------------------------------------

def test_batch_task_sees_only_its_own_bg_jobs(nx_fresh, tmp_path, monkeypatch):
    """Two batch tasks, each 'launching' one bg job. The completion sweep run
    for a task must announce only THAT task's job — the process-global
    owner_pid filter alone lets either task sweep either job, and a chained
    task would re-announce a sibling's exit line forever."""
    nx, tmp = nx_fresh
    from nx import _bg_completions as bg

    src = nx._sources.Source("fake", "http://127.0.0.1:9/v1", "k", "fake-model")
    monkeypatch.setattr(nx.st, "SOURCES", {**nx.st.SOURCES, "fake": src})
    monkeypatch.setattr(nx.st, "SOURCE_ORDER", list(nx.st.SOURCE_ORDER) + ["fake"])
    monkeypatch.setattr(nx.st, "ACTIVE", src)

    bg_dir = tmp / "bglogs"
    bg_dir.mkdir()
    logs = {}

    def fake_list_bg_jobs(mine_only: bool = True):
        out = []
        for pid, owner in (("11111", "alpha"), ("22222", "beta")):
            log = logs[owner]
            out.append({"pid": pid, "owner_pid": os.getpid(), "alive": False,
                        "log_path": str(log), "command": f"echo {owner}",
                        "tail": f"{owner} done", "started": time.time()})
        return out

    monkeypatch.setattr("nx._bash_jobs.list_bg_jobs", fake_list_bg_jobs)
    monkeypatch.setattr("nx._verify._is_verify_job", lambda j: False)
    monkeypatch.setattr("nx._bash_jobs._read_bg_exit_status",
                        lambda pid: (True, 0, ""))

    # Both job logs exist before either task can sweep (the fake registry
    # lists both jobs for every caller, exactly as the real shared pid->log
    # map does — that shared visibility is the bug being pinned).
    for tid in ("alpha", "beta"):
        log = tmp / f"{tid}.log"
        log.write_text("ok\n", encoding="utf-8")
        logs[tid] = log

    swept = {}
    lock2 = threading.Lock()

    def fake_loop(messages, force_final=False, recovery_sampling=False,
                  on_step=None, max_turns_override=None):
        task = nx._batch._batch_task()
        tid = task.get("id")
        pid = "11111" if tid == "alpha" else "22222"
        nx._batch._bg_task_scopes().add(pid)
        # The sweep THIS task's turn runs. Called directly so the test pins
        # the per-task SCOPE filter, not the gate that decides whether to run
        # the sweep at all (that gate is test_batch_run's contract).
        bg._run_bg_completion_scan()
        from nx import _notices as nz
        with lock2:
            swept[tid] = str(nz.get("_PENDING_BG_COMPLETION_NOTICE") or "")
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx, "_run_model_turn_loop", fake_loop)

    f = _write_tasks(tmp, [
        '{"instruction": "alpha", "id": "alpha"}',
        '{"instruction": "beta", "id": "beta"}',
    ])
    buf, err = _capture(monkeypatch)
    from nx._batch import run_batch
    code = run_batch(["-f", f, "--parallel", "2"])
    assert code == 0, f"exit={code} stderr={err.getvalue()!r}"

    assert set(swept) == {"alpha", "beta"}, swept
    assert "pid 11111" in swept["alpha"] and "pid 22222" not in swept["alpha"], swept
    assert "pid 22222" in swept["beta"] and "pid 11111" not in swept["beta"], swept


def test_autowake_off_task_delivers_note_staged_in_its_own_frame(
        nx_fresh, tmp_path, monkeypatch):
    """0.163.1 review fix 1: a note an async agent staged into the TASK's frame
    during the FINAL turn loop must reach the task's transcript even with
    autowake OFF.

    The driver publishes F0 for the whole task and every turn loop nests its own
    F1 seeded from it; the agent stages into F0 by ident, so only a LATER frame
    (a next turn loop, or the wake loop's flush between loops) can read it. With
    the flag off and one turn there is no such reader, and the finally discards
    F0 without folding — pre-0.163 the worker wrote the module store and the next
    drain delivered it, so the note was simply lost. The driver now drains its
    own frame once into the transcript before the discard."""
    nx, tmp = nx_fresh
    from nx import _subagent as sa
    sa._ASYNC_AGENTS.clear()

    # The agent finishes shortly after the spawn, i.e. while the turn is still
    # running — the common single-agent shape.
    _fake_agent_worker(nx, monkeypatch, {"delay": 0.05})

    src = nx._sources.Source("fake", "http://127.0.0.1:9/v1", "k", "fake-model")
    monkeypatch.setattr(nx.st, "SOURCES", {**nx.st.SOURCES, "fake": src})
    monkeypatch.setattr(nx.st, "SOURCE_ORDER", list(nx.st.SOURCE_ORDER) + ["fake"])
    monkeypatch.setattr(nx.st, "ACTIVE", src)

    def fake_loop(messages, force_final=False, recovery_sampling=False,
                  on_step=None, max_turns_override=None):
        sa.spawn_agent_job("instr-solo", 5)
        # The agent stages into F0 here. Deliberately NO flush: this is the
        # text-only final turn that used to end with the note stranded.
        time.sleep(0.3)
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx, "_run_model_turn_loop", fake_loop)

    saved = {}
    real_save = nx._batch._save_task_session

    def spy_save(session_id, messages):
        saved["messages"] = [dict(m) for m in messages]
        return real_save(session_id, messages)

    monkeypatch.setattr(nx._batch, "_save_task_session", spy_save)

    f = _write_tasks(tmp, ['{"instruction": "solo", "id": "solo"}'])
    buf, err = _capture(monkeypatch)
    from nx._batch import run_batch
    code = run_batch(["-f", f])
    assert code == 0, f"exit={code} stderr={err.getvalue()!r}"

    contents = [m.get("content") for m in saved.get("messages", [])
                if isinstance(m.get("content"), str)]
    notes = [c for c in contents if c.startswith("[Runtime note: agent for")]
    assert notes, f"the agent's note never reached the task transcript: {contents}"
    assert all("instr-solo" in n for n in notes), notes


def test_async_agent_bg_job_joins_the_parent_task_scope(nx_fresh, tmp_path,
                                                       monkeypatch):
    """0.163.1 review fix 2: a bg job an ASYNC AGENT backgrounds (its own worker
    thread) must join the SPAWNING TASK's job scope.

    The scope is a thread-local installed by the task driver, and `run_bash`
    records its pid on the CALLING thread — so on the agent's loop thread
    `bg_scope_add` found no scope and did nothing: the job was invisible to the
    task's sweep narrowing (its exit note was staged to nobody, and never marked
    seen) and never reaped at task end (an orphaned live process). The spawn site
    now hands the worker the parent's live pid set and the worker installs it."""
    nx, tmp = nx_fresh
    from nx import _subagent as sa
    from nx import _bg_completions as bg
    sa._ASYNC_AGENTS.clear()

    src = nx._sources.Source("fake", "http://127.0.0.1:9/v1", "k", "fake-model")
    monkeypatch.setattr(nx.st, "SOURCES", {**nx.st.SOURCES, "fake": src})
    monkeypatch.setattr(nx.st, "SOURCE_ORDER", list(nx.st.SOURCE_ORDER) + ["fake"])
    monkeypatch.setattr(nx.st, "ACTIVE", src)

    # One finished bg job with a readable log — the registry shape the real
    # shared pid->log map gives every task (which is why the SCOPE, not the
    # owner_pid filter, is what decides who may announce it).
    log = tmp / "agentbg.log"
    log.write_text("done\n", encoding="utf-8")

    def fake_list_bg_jobs(mine_only: bool = True):
        return [{"pid": "44444", "owner_pid": os.getpid(), "alive": False,
                 "log_path": str(log), "command": "sleep 999 &",
                 "tail": "done", "started": time.time()}]

    monkeypatch.setattr("nx._bash_jobs.list_bg_jobs", fake_list_bg_jobs)
    monkeypatch.setattr("nx._verify._is_verify_job", lambda j: False)
    monkeypatch.setattr("nx._bash_jobs._read_bg_exit_status",
                        lambda pid: (True, 0, ""))

    killed = []
    monkeypatch.setattr("nx._bash_jobs.cancel_kill_group",
                        lambda pid, command="": killed.append(str(pid)))
    monkeypatch.setattr("nx._bash_jobs._pid_is_alive",
                        lambda pid, expected=None: True)

    def fake_run(job_id, instruction, max_turns, tier="read", scope=None, *,
                 worktree=None, announce=True, handle=None):
        # Exactly what `_bash_jobs._run_bash` does with the pid it just
        # backgrounded, from the AGENT's thread.
        bg.bg_scope_add("44444")
        with sa._ASYNC_LOCK:
            rec = sa._ASYNC_AGENTS.get(job_id) or {}
            rec.update({"status": "done", "answer": "agent done"})
            sa._ASYNC_AGENTS[job_id] = rec
        sa._stage_notice_owner(rec, sa._PENDING_NOTICE_KEY,
                               "[Runtime note: agent with the bg job done]")
        if handle is not None:
            handle.done.set()

    monkeypatch.setattr(sa, "_run_async_agent", fake_run)

    scope_seen = {}

    def fake_loop(messages, force_final=False, recovery_sampling=False,
                  on_step=None, max_turns_override=None):
        if not scope_seen:
            sa.spawn_agent_job("instr-bg", 5)
        mine = bg.bg_scope()
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if mine is not None and "44444" in mine:
                break
            time.sleep(0.02)
        scope_seen["in_task_scope"] = bool(mine is not None
                                           and "44444" in mine)
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx, "_run_model_turn_loop", fake_loop)

    saved = {}
    real_save = nx._batch._save_task_session

    def spy_save(session_id, messages):
        saved["messages"] = [dict(m) for m in messages]
        return real_save(session_id, messages)

    monkeypatch.setattr(nx._batch, "_save_task_session", spy_save)

    f = _write_tasks(tmp, ['{"instruction": "x", "id": "t1"}'])
    buf, err = _capture(monkeypatch)
    from nx._batch import run_batch
    code = run_batch(["-f", f, "--autowake", "--timeout", "30"])
    assert code == 0, f"exit={code} stderr={err.getvalue()!r}"

    # 1. the agent's launch joined the task's scope (the thread-local hop) —
    #    the only reason any of the rest can happen;
    assert scope_seen.get("in_task_scope") is True, \
        "the agent's job never entered the parent task's bg-job scope"
    # 2. so the task's own completion sweep announced it (in scope is what lets
    #    the sweep stage the note AND mark the pid seen; out of scope it was
    #    skipped for nobody) and a wake turn carried it into the transcript;
    line = json.loads(buf.getvalue().strip())
    assert line.get("wakes"), \
        f"the agent's finished job never woke its owning task: {line}"
    contents = [m.get("content") for m in saved.get("messages", [])
                if isinstance(m.get("content"), str)]
    assert any(c.startswith("[Runtime note: background job(s)") and "pid 44444" in c
               for c in contents), \
        f"the agent's job was never announced to the task: {contents}"
    # 3. and it got reaped when the task ended.
    assert killed == ["44444"], f"the agent's job was not reaped at task end: {killed}"


def test_batch_task_end_kills_its_live_bg_jobs(nx_fresh, tmp_path, monkeypatch):
    """A task that ends with a live detached job it LAUNCHED must have it
    killed and reaped — otherwise a chained task accumulates detached jobs
    across its own turns and a --parallel run leaks a process tree per task."""
    nx, tmp = nx_fresh
    src = nx._sources.Source("fake", "http://127.0.0.1:9/v1", "k", "fake-model")
    monkeypatch.setattr(nx.st, "SOURCES", {**nx.st.SOURCES, "fake": src})
    monkeypatch.setattr(nx.st, "SOURCE_ORDER", list(nx.st.SOURCE_ORDER) + ["fake"])
    monkeypatch.setattr(nx.st, "ACTIVE", src)

    killed = []
    monkeypatch.setattr("nx._bash_jobs.cancel_kill_group",
                        lambda pid, command="": killed.append((str(pid), command)))

    def fake_loop(messages, **kw):
        nx._batch._bg_task_scopes().add("33333")
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx, "_run_model_turn_loop", fake_loop)
    monkeypatch.setattr("nx._bash_jobs._pid_is_alive", lambda pid, expected=None: True)

    f = _write_tasks(tmp, ['{"instruction": "x", "id": "t1"}'])
    buf, err = _capture(monkeypatch)
    from nx._batch import run_batch
    assert run_batch(["-f", f]) == 0
    assert [p for p, _ in killed] == ["33333"], killed
