#!/usr/bin/env python3
"""Background-probe backoff must not leak into the user's session (0.53.2).

A dead endpoint used to keep talking after the user switched away from it: the
capability / thinking / context-window probes run in daemon threads with a
captured Source + model id, so each one re-printed "nx: POST /v1/chat/completions
failed ... retrying n/2" from a background thread — landing over a chatbox frame
that had already moved on, and naming the abandoned model — then slept out the
whole curve while holding STARTUP_PROBE_LOCK, which the source the user switched
TO needed to run its own probes.

_openai_client.probe_scope() makes a probe thread's retry loop silent and
abortable on staleness. Offline: httpx.MockTransport, time.sleep recorded.
"""
import os
import sys
import tempfile
import threading
import time

_tmp = tempfile.mkdtemp(prefix="nx-probebackoff-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_HOME"] = _tmp
for k in ("NX_HTTP_RETRIES", "NX_HTTP_BACKOFF_BASE", "NX_PROBE_QUIET",
          "NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SOURCES", "NX_ACTIVE"):
    os.environ.pop(k, None)
# switch_source() must not spawn live probes from inside these tests.
os.environ["NX_SKIP_STARTUP_PROBES"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402
import pytest  # noqa: E402

import nx._state as st  # noqa: E402
from nx._openai_client import (  # noqa: E402
    APIError, _ProbeAbort, OpenAI, probe_scope,
)


@pytest.fixture()
def sleeps(monkeypatch):
    seen = []
    monkeypatch.setattr("nx._openai_client.time.sleep", seen.append)
    return seen


# Ceiling for cross-thread waits (join / Event.wait / lock acquire). These wait
# on a daemon thread that only has to be *scheduled* -- not to hit a latency
# budget -- so a loaded box may need far more than a few seconds. Nothing below
# asserts how fast a thread wakes, only that it eventually finishes.
_WAIT = 30.0


def _always_failing_client(counter):
    def handler(request):
        counter["n"] += 1
        return httpx.Response(500, json={"error": {"message": "connection error"}})
    return OpenAI(base_url="http://test/v1", api_key="sk-test",
                  transport=httpx.MockTransport(handler))


# --- the client-level contract -------------------------------------------------

def test_outside_probe_scope_nothing_changes(sleeps, capsys):
    """A user turn is unaffected: same attempts, same notices. Guards against
    probe plumbing leaking into the main thread."""
    calls = {"n": 0}
    with pytest.raises(APIError):
        _always_failing_client(calls).models.list()
    assert calls["n"] == 3                       # 1 attempt + 2 retries
    assert sleeps == [0.5, 1.0]
    assert "retrying 1/2" in capsys.readouterr().err


def test_probe_scope_silences_notices(sleeps, capsys):
    calls = {"n": 0}
    with probe_scope(lambda: False):
        with pytest.raises(APIError):
            _always_failing_client(calls).models.list()
    assert "retrying" not in capsys.readouterr().err


def test_probe_scope_is_cleared_on_exit(sleeps, capsys):
    """No sticky silencing: a later user turn must print its notices again."""
    with probe_scope(lambda: False):
        pass
    calls = {"n": 0}
    with pytest.raises(APIError):
        _always_failing_client(calls).models.list()
    assert "retrying 1/2" in capsys.readouterr().err


def test_stale_probe_aborts_instead_of_sleeping(sleeps):
    """A probe whose source was switched away from stops at its next backoff
    boundary: one request, no sleeps waited, no further attempts."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        stale["v"] = True          # the user switches source mid-backoff
        return httpx.Response(500, json={"error": {"message": "connection error"}})

    c = OpenAI(base_url="http://test/v1", api_key="sk-test",
               transport=httpx.MockTransport(handler))
    stale = {"v": False}
    with probe_scope(lambda: stale["v"]):
        with pytest.raises(_ProbeAbort):
            c.models.list()
    assert stale["v"] is True
    assert calls["n"] == 1                       # first attempt, then abandoned
    assert sleeps == []                          # no backoff was waited


def test_fresh_probe_still_runs_the_full_retry_curve(sleeps):
    """The abort is not "probes retry less": never-stale == old behavior, minus
    the noise (asserted by test_probe_scope_silences_notices)."""
    calls = {"n": 0}
    with probe_scope(lambda: False):
        with pytest.raises(APIError):
            _always_failing_client(calls).models.list()
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.0]


def test_abort_is_per_thread(sleeps, capsys):
    """threading.local, not a global flag: a probe running in a worker must not
    silence or cancel the main thread's own request."""
    gate = threading.Event()
    release = threading.Event()
    main_errs = []

    def worker():
        def handler(request):
            gate.set()
            release.wait(_WAIT)
            return httpx.Response(500, json={"error": {"message": "down"}})
        c = OpenAI(base_url="http://test/v1", api_key="sk-test",
                   transport=httpx.MockTransport(handler))
        try:
            with probe_scope(lambda: False):
                c.models.list()
        except Exception as e:                    # noqa: BLE001 - recorded below
            main_errs.append(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert gate.wait(_WAIT), "worker never sent its first request"
    try:
        calls = {"n": 0}
        with pytest.raises(APIError):             # main thread retries normally
            _always_failing_client(calls).models.list()
        assert calls["n"] == 3
        assert "retrying 1/2" in capsys.readouterr().err
    finally:
        release.set()
        t.join(_WAIT)
    assert not t.is_alive(), "worker did not finish within the wait ceiling"
    assert len(main_errs) == 1 and isinstance(main_errs[0], APIError)


def test_scope_is_reentrant(sleeps):
    """A probe that probes (resolve_model's /v1/models inside a probe) nests."""
    calls = {"n": 0}
    with probe_scope(lambda: False):
        with probe_scope(lambda: False):
            with pytest.raises(APIError):
                _always_failing_client(calls).models.list()
    assert calls["n"] == 3


# --- the switch that motivated this -------------------------------------------

def test_switch_during_probe_backoff_releases_the_probe_lock(monkeypatch):
    """The real failure mode: an old source's probe parks itself on
    STARTUP_PROBE_LOCK and then sleeps out its backoff against a dead endpoint,
    so the source just switched TO cannot start its own probes."""
    from nx._capability import STARTUP_PROBE_LOCK
    from nx._sources import switch_source

    monkeypatch.setenv("NX_HTTP_RETRIES", "5")
    monkeypatch.setenv("NX_HTTP_BACKOFF_BASE", "1.5")
    monkeypatch.setattr("nx._sources.Source.resolve_model",
                        lambda self: self.model or "local-model")

    gen = st.PROBE_GENERATION          # what this probe was armed "for"
    stuck = threading.Event()
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500, json={"error": {"message": "connection error"}})

    dead = OpenAI(base_url="http://test/v1", api_key="sk-test",
                  transport=httpx.MockTransport(handler))

    def held_probe():
        with STARTUP_PROBE_LOCK:
            stuck.set()
            with probe_scope(lambda: st.PROBE_GENERATION != gen):
                try:
                    dead.models.list()
                except Exception:                 # noqa: BLE001 - as in probes
                    pass

    t = threading.Thread(target=held_probe, daemon=True, name="nx-old-probe")
    t.start()
    assert stuck.wait(_WAIT), "old probe never entered its request loop"
    deadline = time.monotonic() + _WAIT
    while calls["n"] < 1 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert calls["n"] >= 1, (
        f"old probe never sent a request within {_WAIT:.0f}s "
        f"(calls={calls['n']})")            # probe is in (or at) backoff

    # The probe holds the lock; the user switches source mid-backoff.
    st.SOURCES["other"] = st.ACTIVE
    st.SOURCE_ORDER.append("other")
    assert switch_source("other") is True

    t.join(_WAIT)                 # one 1.5s sleep remains; then the abort lands
    assert not t.is_alive(), (
        "old probe did not abort within the wait ceiling after the switch")
    assert calls["n"] == 1    # no further attempts against the dead endpoint
    acquired = STARTUP_PROBE_LOCK.acquire(timeout=_WAIT)
    assert acquired, "aborted probe still holds STARTUP_PROBE_LOCK"
    STARTUP_PROBE_LOCK.release()


def test_generation_bumped_only_on_a_real_switch(monkeypatch):
    """An unknown source name must not invalidate probes that are still valid,
    and a switch that keeps the same source still counts (a model override may
    have changed)."""
    monkeypatch.setenv("NX_SKIP_STARTUP_PROBES", "1")
    if st.ACTIVE is None:
        pytest.skip("no active source in this environment")
    from nx._sources import switch_source
    monkeypatch.setattr("nx._sources.Source.resolve_model",
                        lambda self: self.model or "local-model")
    before = st.PROBE_GENERATION
    assert switch_source("no-such-source-anywhere") is False
    assert st.PROBE_GENERATION == before
    name = st.ACTIVE.name
    assert switch_source(name) is True
    assert st.PROBE_GENERATION == before + 1
