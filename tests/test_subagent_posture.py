"""Thread-local approval posture for sub-agents (0.155.0, docs/YOLO_POSTURE_SPEC.md).

The pins this file holds:

1. §5.2 the leak window — a read-only sync agent whose body dies between the
   OLD swap site and the OLD `try` (spinner stop / model-target install) must
   leave `st.YOLO`/`st.APPROVE_LEVEL` exactly as the parent had them, must not
   leave a thread-local override behind, and a LATER `_confirm()` must prompt
   instead of riding a leaked yolo fast path. Pre-0.155.0 the swap sat before
   the `try`, so the raise skipped the restore for good.
2. §5.3 async isolation — while a read-only async agent runs, the PARENT
   thread's posture reads (and the agent's, from the worker's TLS) are correct.
3. §5.4 resolver parity — `_posture()` == globals on a fresh thread, == the
   override inside an agent body, and a parent mid-turn global mutation is
   visible to the parent's `_posture()` on the next call (Shift+Tab contract).
4. §5.4 the delegation note describes the AGENT's posture while an agent builds
   one (`_delegation_note._write_bg_sentence`, pinned by the maintainer).
5. §5.5 the guard matrix still refuses/prompts on the agent path.
"""

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

# Mirror tests/test_subagent_write.py: scrub ambient NX_* config, then opt into
# the legacy full-text-inline foreground result.
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for _k in [k for k in os.environ if k == "NX_ENV_FILE" or k.startswith("NX_")]:
    os.environ.pop(_k, None)
os.environ["NX_ENV_FILE"] = "/dev/null"
for _k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR",
           "NX_SUBAGENTS", "NX_SUBAGENT_WRITE"):
    os.environ.pop(_k, None)
os.environ["NX_SUBAGENT_SYNC_INLINE"] = "1"


@pytest.fixture(autouse=True)
def _sync_inline(monkeypatch):
    # _subagent._sync_inline prefers the module attr over the env; pin it.
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
    os.environ["NX_WRITE_FENCE"] = "0"
    monkeypatch.setattr(sys, "argv", ["nx"])
    import nx
    from nx._sources import _discover_sources, switch_source
    _discover_sources()
    assert switch_source(nx._state.SOURCE_ORDER[0]), "fixture source wiring failed"
    monkeypatch.setattr(nx, "NX_SUBAGENT_ASYNC", 0, raising=False)
    monkeypatch.setattr(nx, "NX_SUBAGENT_WRITE", 0, raising=False)
    yield nx


# ---------------------------------------------------------------------------
# 1. the leak window (§5.2)
# ---------------------------------------------------------------------------

def test_failed_readonly_agent_leaves_no_forced_posture(nx_fresh, monkeypatch):
    """§5.2 — the leak window.

    A read-only sync agent that raises between the old swap site and the old
    `try` (injected via POSTURE_RAISE_AT=spinner|install) must leave the
    process globals exactly as the parent had them AND leave no thread-local
    override behind; a later `_confirm()` must PROMPT, not auto-allow.

    The env var is deliberately NOT named NX_*: this module (and conftest)
    scrub every NX_* var at import, which would erase the injection.
    """
    from nx import _approval
    from nx import _state as st
    from nx._subagent import spawn_agent

    raise_at = os.environ.get("POSTURE_RAISE_AT", "")
    inject = raise_at in ("spinner", "install")
    monkeypatch.setattr(st, "YOLO", False)
    monkeypatch.setattr(st, "APPROVE_LEVEL", "medium")

    if raise_at == "install":
        def _boom_install(ctx):
            raise RuntimeError("model-target install failed (injected)")
        monkeypatch.setattr("nx._subagent_model.install", _boom_install)
    elif raise_at == "spinner":
        class _BoomSpinner:
            def stop(self):
                raise RuntimeError("spinner stop failed (injected)")

            def start(self):
                pass

            def note(self, *a, **k):
                pass
        monkeypatch.setattr(st, "_ACTIVE_SPINNER", _BoomSpinner())

    fired = []

    def fake_turn(messages, *a, **kw):
        fired.append(1)
        messages.append({"role": "assistant", "content": "unreachable"})
        return nx_fresh.TURN_DONE

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)

    if inject:
        with pytest.raises(RuntimeError):
            spawn_agent("probe the leak window", tier="read")
    else:
        # The raise-hook is active but nothing injected on this run: the agent
        # completes normally, so the appends are moot (kept so the list is
        # always defined for the assert below).
        out = spawn_agent("probe the leak window", tier="read")
        if out == "unreachable":
            assert fired, "the sub-loop never ran"

    assert st.YOLO is False, (
        f"a failed read-only spawn leaked the forced posture "
        f"(raise_at={raise_at!r}): st.YOLO={st.YOLO!r} "
        f"st.APPROVE_LEVEL={st.APPROVE_LEVEL!r}")
    assert st.APPROVE_LEVEL == "medium"
    _tls = getattr(st, "_POSTURE_TLS", None)
    assert _tls is None or dict(vars(_tls)) == {}, (
        f"posture override leaked on this thread: "
        f"{dict(vars(_tls)) if _tls is not None else None!r}")

    # The next decision must be a real prompt: with the parent's posture
    # (yolo off, threshold medium) a HIGH-risk action cannot auto-allow.
    monkeypatch.setattr(st, "_ACTIVE_SPINNER", None)
    monkeypatch.setattr(nx_fresh, "_assess_risk",
                        lambda action: ("high", "injected risk"))
    prompted = []
    monkeypatch.setattr(nx_fresh, "_ask_approval",
                        lambda prompt, allow_hint=None, **kw:
                        (prompted.append(prompt) or "n"))
    allowed = _approval._confirm("write_file path=leak-probe-0.155.0.txt")
    assert allowed is False
    assert prompted, ("_confirm auto-allowed after the failed spawn — a forced "
                      "posture leaked into the next decision")


# ---------------------------------------------------------------------------
# 2. cross-thread isolation (§5.3)
# ---------------------------------------------------------------------------

def test_readonly_agent_posture_is_thread_local_and_invisible_to_parent(
        nx_fresh, monkeypatch):
    """§5.3 — while a read-only agent runs, its forced posture lives on the
    agent's thread only: the parent thread's `_posture()` and the process
    globals never move. (Pre-0.155.0 the swap wrote the globals, so a parent
    `_confirm()` mid-agent read the agent's forced posture — for an async
    worker that lasted the agent's whole, minutes-long run.)

    The spawn runs on a worker thread (exactly the async-worker shape: a bare
    thread running the same body), and the parent thread reads DURING the body,
    held open by an Event. The async worker's own version of this pin lives in
    tests/test_subagent_async.py.
    """
    from nx import _state as st
    from nx._subagent import spawn_agent

    monkeypatch.setattr(st, "YOLO", False)
    monkeypatch.setattr(st, "APPROVE_LEVEL", "medium")
    seen: dict = {}
    out: dict = {}
    entered = threading.Event()
    release = threading.Event()

    def fake_turn(messages, *a, **kw):
        # Defensive read so this pin also RUNS against pre-0.155.0 code (where
        # `_posture` does not exist) and reports the leak `st.YOLO`-shaped
        # instead of dying in the stub.
        seen["agent_posture"] = (st._posture() if hasattr(st, "_posture")
                                 else (st.YOLO, st.APPROVE_LEVEL))
        seen["agent_globals"] = (st.YOLO, st.APPROVE_LEVEL)
        entered.set()
        assert release.wait(30), "release never signalled"
        messages.append({"role": "assistant", "content": "ok"})
        return nx_fresh.TURN_DONE

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)

    def _run():
        out["answer"] = spawn_agent("q", tier="read")

    t = threading.Thread(target=_run, name="posture-parent-probe")
    t.start()
    try:
        assert entered.wait(30), "the agent body never started"
        # The PARENT thread's reads, taken while the agent body is running.
        # Globals first: this is the pre-0.155.0 leak, observable without any
        # new API (`_posture()` did not exist then).
        assert (st.YOLO, st.APPROVE_LEVEL) == (False, "medium"), (
            "an agent body wrote the process globals")
        assert st._posture() == (False, "medium"), (
            "the parent thread saw the agent's forced posture: "
            f"{st._posture()!r}")
        _tls = getattr(st, "_POSTURE_TLS", None)
        assert _tls is None or dict(vars(_tls)) == {}, (
            "the parent thread inherited an agent's posture override")
    finally:
        release.set()
        t.join(30)

    assert out.get("answer") == "ok", out
    assert seen["agent_posture"] == (True, "high"), (
        "a read-only agent must see its own forced posture through _posture(): "
        f"{seen['agent_posture']!r}")
    assert seen["agent_globals"] == (False, "medium"), (
        "the agent's forced posture must NOT be written to the process globals: "
        f"{seen['agent_globals']!r}")
    assert st._posture() == (False, "medium")     # restored after the spawn
    _tls = getattr(st, "_POSTURE_TLS", None)
    assert _tls is None or dict(vars(_tls)) == {}


# ---------------------------------------------------------------------------
# 3. resolver parity (§5.4)
# ---------------------------------------------------------------------------

def test_posture_resolver_parity_and_live_read(nx_fresh, monkeypatch):
    """§5.4 — `_posture()` == globals on a fresh thread; == the override inside
    an agent body; and a parent mid-turn mutation of the globals is visible to
    the parent's very next `_posture()` call (the Shift+Tab-at-a-prompt and
    /yolo //approval live-read contract)."""
    from nx import _state as st
    from nx._subagent import spawn_agent

    monkeypatch.setattr(st, "YOLO", False)
    monkeypatch.setattr(st, "APPROVE_LEVEL", None)
    assert st._posture() == (False, None)
    assert st._posture() == (st.YOLO, st.APPROVE_LEVEL)

    # live read: a mid-turn mutation by the parent shows up immediately
    monkeypatch.setattr(st, "YOLO", True)
    assert st._posture() == (True, None)
    monkeypatch.setattr(st, "YOLO", False)
    monkeypatch.setattr(st, "APPROVE_LEVEL", "high")
    assert st._posture() == (False, "high")

    # inside an agent body: the override, and it must not touch the globals
    seen = {}

    def fake_turn(messages, *a, **kw):
        seen["agent"] = st._posture()
        # a mid-turn parent mutation of the globals stays invisible to the
        # agent (its posture is installed), and the agent's override stays
        # invisible to the parent.
        st.YOLO = False
        st.APPROVE_LEVEL = None
        messages.append({"role": "assistant", "content": "ok"})
        return nx_fresh.TURN_DONE

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)
    assert spawn_agent("q", tier="read") == "ok"
    assert seen["agent"] == (True, "high")
    assert st._posture() == (False, None)

    # a fresh thread (no override ever installed) falls back to the globals
    fresh = {}
    t = threading.Thread(target=lambda: fresh.update(p=st._posture()))
    t.start()
    t.join()
    assert fresh["p"] == (False, None), fresh

    # a write-tier agent INHERITS: nothing installed, so the globals are read
    # live (tests/test_subagent_write.py's contract, pinned here at the
    # resolver level too).
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENT_WRITE", 1, raising=False)
    monkeypatch.setattr(st, "YOLO", True)
    monkeypatch.setattr(st, "APPROVE_LEVEL", None)
    inherited = {}

    def fake_turn_write(messages, *a, **kw):
        inherited["agent"] = st._posture()
        _tls = getattr(st, "_POSTURE_TLS", None)
        inherited["tls"] = {} if _tls is None else dict(vars(_tls))
        messages.append({"role": "assistant", "content": "ok"})
        return nx_fresh.TURN_DONE

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn_write)
    assert spawn_agent("q", tier="write") == "ok"
    assert inherited["agent"] == (True, None)
    assert inherited["tls"] == {}, "a write-tier agent must install nothing"


# ---------------------------------------------------------------------------
# 4. the delegation note (§5.4, maintainer-pinned: the AGENT's posture)
# ---------------------------------------------------------------------------

def test_delegation_note_describes_the_agents_posture(nx_fresh, monkeypatch):
    """The note text must describe the posture of the loop that builds it: an
    agent body runs yolo-forced (read-only non-quiet), so a note built INSIDE
    an agent says the yolo sentence even when the parent is not in yolo mode."""
    from nx import _state as st
    from nx._subagent import spawn_agent
    from nx._delegation_note import _write_bg_sentence, _WRITE_BG_YOLO

    monkeypatch.setattr(st, "YOLO", False)
    monkeypatch.setattr(st, "APPROVE_LEVEL", "medium")
    assert _write_bg_sentence() != _WRITE_BG_YOLO   # parent: non-yolo wording

    seen = {}

    def fake_turn(messages, *a, **kw):
        seen["note"] = _write_bg_sentence()
        messages.append({"role": "assistant", "content": "ok"})
        return nx_fresh.TURN_DONE

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)
    assert spawn_agent("q", tier="read") == "ok"
    assert seen["note"] == _WRITE_BG_YOLO, (
        "the delegation note built inside an agent body must describe the "
        f"AGENT's posture: {seen['note'][:60]!r}")
    assert _write_bg_sentence() != _WRITE_BG_YOLO   # parent wording back


# ---------------------------------------------------------------------------
# 5. guard matrix under agent posture (§5.5)
# ---------------------------------------------------------------------------

def test_quiet_readonly_agent_refuses_quiet_write_tier(nx_fresh, monkeypatch):
    """§5.5 — a QUIET thread in a non-yolo session still refuses a write-tier
    spawn: quiet threads install nothing (they inherit), so the gate reads the
    parent's real posture and the refusal message is unchanged."""
    from nx import _state as st
    from nx._subagent import _ASYNC_QUIET_SLOTS, spawn_agent

    monkeypatch.setattr(st, "YOLO", False)
    monkeypatch.setattr(st, "APPROVE_LEVEL", "high")
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENT_WRITE", 1, raising=False)
    _ASYNC_QUIET_SLOTS.quiet = True
    try:
        out = spawn_agent("q", tier="write")
    finally:
        _ASYNC_QUIET_SLOTS.quiet = False
    assert out.startswith("ERROR: write-tier spawn_agent refused"), out


def test_git_index_guard_still_prompts_under_agent_posture(nx_fresh, monkeypatch,
                                                          tmp_path):
    """§5.5 (mirrors tests/test_git_index_guard.py:70): the git-index guard
    forces its own prompt even when the thread's posture is the AGENT's
    yolo-forced one, so a read-only agent's `git commit` over a dirty shared
    index cannot ride an auto-allow. Refused outright on a quiet (detached)
    agent thread — also pinned here."""
    import subprocess

    from nx import _state as st
    from nx._bash_jobs import run_bash

    repo = tmp_path / "repo"
    repo.mkdir()
    env = dict(os.environ)
    env.update({"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"})
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "seed.txt"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True,
                   env=env)
    # another session's WIP, staged and never committed
    (repo / "other.txt").write_text("wip\n", encoding="utf-8")
    subprocess.run(["git", "add", "other.txt"], cwd=repo, check=True, env=env)

    monkeypatch.setattr(st, "YOLO", False)
    monkeypatch.setattr(st, "APPROVE_LEVEL", None)
    asked = []
    monkeypatch.setattr(nx_fresh, "_ask_approval",
                        lambda prompt, allow_hint=None, **kw:
                        (asked.append(prompt) or "n"))

    # (a) the agent thread is NOT quiet (a sync read-only spawn): it can prompt.
    out = _run_bash_under_forced_posture(
        st, run_bash, f"git -C {repo} commit -m x", quiet=False)
    assert out == "DENIED by user", out
    assert asked and "staged" in asked[0], asked

    # (b) a quiet (detached) agent thread cannot prompt: refusal, not auto-allow
    del asked[:]
    out = _run_bash_under_forced_posture(
        st, run_bash, f"git -C {repo} commit -m x", quiet=True)
    assert out.startswith("ERROR:") and "staged" in out, out
    assert not asked, asked


def test_secret_guard_still_prompts_under_agent_posture(nx_fresh, monkeypatch):
    """§5.5 (mirrors tests/test_bash_watchdog.py:98): NX_BASH_PROMPT_SECRETS
    forces its own prompt on the agent path too — the forced agent posture
    cannot short-circuit it."""
    from nx import _state as st
    from nx._bash_jobs import run_bash

    os.environ["NX_BASH_PROMPT_SECRETS"] = "1"
    monkeypatch.setattr(st, "YOLO", False)
    monkeypatch.setattr(st, "APPROVE_LEVEL", "high")
    asked = []
    monkeypatch.setattr(nx_fresh, "_ask_approval",
                        lambda prompt, allow_hint=None, **kw:
                        (asked.append(prompt) or "n"))
    # A LITERAL token, built here so no secret-shaped literal sits in the
    # source — the pattern on the command line is what has to match.
    tok = "sk-" + ("abcdefghij" * 2)
    out = _run_bash_under_forced_posture(
        st, run_bash, f"curl -s -H 'x-api-key: {tok}' http://x")
    assert out == "DENIED by user", out
    assert asked and "possible secret" in asked[0], asked


def _run_bash_under_forced_posture(st, run_bash, command, quiet=False):
    """Run `run_bash` with the READ-ONLY AGENT posture installed on this thread
    (exactly what `_run_sync_agent_body` installs for a read-only agent) — the
    agent path a guard must keep guarding."""
    from nx._state import _posture_scope
    from nx._subagent import _ASYNC_QUIET_SLOTS
    _ASYNC_QUIET_SLOTS.quiet = quiet
    try:
        with _posture_scope((True, "high")):
            return run_bash(command)
    finally:
        _ASYNC_QUIET_SLOTS.quiet = False
