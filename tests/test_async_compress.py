#!/usr/bin/env python3
"""Tests for the async autocompress fold (0.103.5).

Covers:
  - threshold crossing starts a background fold and returns immediately
    (compress runs on the fold thread, not the caller's);
  - a finished fold is applied at the next _maybe_autocompress check;
  - a fold whose snapshot no longer fingerprint-matches the live transcript
    is discarded (stale-snapshot safety);
  - only one fold is in flight at a time;
  - NX_ASYNC_COMPRESS=0 restores the synchronous fold;
  - _collect_async_fold(wait=N) joins a still-running fold.

Offline: compress() is stubbed (the same _FakeResp pattern as
tests/test_autocompress.py) so no network is needed.
"""
import os
import sys
import tempfile
import threading
import time

_tmp = tempfile.mkdtemp(prefix="nx-asynccomp-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ.setdefault("NX_SKIP_STARTUP_PROBES", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _compress as c  # noqa: E402


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
        return {"choices": [{"message": {"content": "s"}}]}


class _FakeSource:
    def __init__(self, ctx_window):
        self._context_window = ctx_window
        self.name = "fake"

    def resolve_context_window(self, model=None, force=False):
        return self._context_window

    def extra_request_kwargs(self):
        return {}


_saved_active = None


def _stub_compress(messages, keep=None, auto=False, delay=0.0):
    """A compress() stand-in with the real mutation semantics folded in:
    summarizes the head into one digest turn so the fold result is a real
    splice, not a None. Optional delay simulates the model round-trip."""
    if delay:
        time.sleep(delay)
    sys_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    body = messages[1:] if sys_msg else messages
    keep_n = max(2, len(body) // 3)
    head, tail = body[:-keep_n], body[-keep_n:]
    digest = {"role": "user",
              "content": f"[Compressed context — {len(head)} earlier turns "
                         f"summarized; last {len(tail)} turns kept verbatim]\nsummary"}
    messages[:] = (([sys_msg] if sys_msg else []) + [digest] + tail)
    return len(tail), len(head), 7


def _msgs():
    out = [{"role": "system", "content": m.SYSTEM}]
    for i in range(12):
        out.append({"role": "user", "content": f"turn {i}"})
        out.append({"role": "assistant", "content": f"reply {i}"})
    return out


def _reset():
    with c._FOLD_LOCK:
        c._ASYNC_FOLD.clear()


# ---------------------------------------------------------------------------
print("TEST 1: threshold crossing starts a background fold, returns immediately")
_reset()
c._LAST_DISCARD.clear()
_saved_active = m.ACTIVE
m.ACTIVE = _FakeSource(10000)
m.AUTOCOMPRESS_PERCENT = 85
_real = m.compress
calls = []
started = threading.Event()


def _slow_compress(messages, **kw):
    calls.append(messages)
    started.set()
    time.sleep(0.4)  # longer than the caller's budget
    return _stub_compress(messages, **kw)


m.compress = _slow_compress
msgs = _msgs()
t0 = time.monotonic()
fired = m._maybe_autocompress(msgs, prompt_tokens=9000)
elapsed = time.monotonic() - t0
assert fired is False, "async fold in flight → not applied yet"
assert elapsed < 0.3, f"start must not block on the fold (took {elapsed:.2f}s)"
assert started.wait(1.0), "fold must run in the background"
assert calls and calls[0] is not msgs, "fold must run over a snapshot, not the live list"
m.compress = _real
_reset()
c._LAST_DISCARD.clear()
print("  → PASS")

print("TEST 2: finished fold is applied at the next threshold check")
_reset()
c._LAST_DISCARD.clear()
m.compress = lambda messages, **kw: _stub_compress(messages, **kw)
msgs = _msgs()
m._maybe_autocompress(msgs, prompt_tokens=9000)          # starts the fold
deadline = time.monotonic() + 2.0
while time.monotonic() < deadline:
    with c._FOLD_LOCK:
        rec = c._ASYNC_FOLD.get("rec")
    if rec and rec.get("done"):
        break
    time.sleep(0.01)
assert m._maybe_autocompress(msgs, prompt_tokens=9100) is True, "finished fold applies"
assert any(x.get("role") == "user" and "Compressed context" in str(x.get("content"))
           for x in msgs), "digest turn spliced into the live list"
m.compress = _real
_reset()
c._LAST_DISCARD.clear()
print("  → PASS")

print("TEST 3: stale snapshot fold is discarded")
_reset()
c._LAST_DISCARD.clear()
m.compress = lambda messages, **kw: _stub_compress(messages, **kw)
msgs = _msgs()
m._maybe_autocompress(msgs, prompt_tokens=9000)          # starts the fold
msgs.append({"role": "user", "content": "new work while folding"})  # region changes
deadline = time.monotonic() + 2.0
while time.monotonic() < deadline:
    with c._FOLD_LOCK:
        rec = c._ASYNC_FOLD.get("rec")
    if rec and rec.get("done"):
        break
    time.sleep(0.01)
# The next check sees a fingerprint mismatch → discard → sync fold over the
# live list (the stub succeeds) → True overall.
assert m._maybe_autocompress(msgs, prompt_tokens=9100) is True
n_digests = sum(1 for x in msgs
                if "Compressed context" in str(x.get("content")))
assert n_digests == 1, f"stale fold discarded, exactly one digest applied (got {n_digests})"
m.compress = _real
_reset()
c._LAST_DISCARD.clear()
print("  → PASS")

print("TEST 4: only one fold in flight")
_reset()
c._LAST_DISCARD.clear()
gate = threading.Event()


def _gated_compress(messages, **kw):
    gate.wait(2.0)
    return _stub_compress(messages, **kw)


m.compress = _gated_compress
msgs = _msgs()
assert m._start_async_fold(msgs) is True, "fold A starts"
assert m._start_async_fold(msgs) is False, "a second fold cannot start while A is in flight"
with c._FOLD_LOCK:
    rec = c._ASYNC_FOLD.get("rec")
assert rec is not None and not rec.get("done"), "A still in flight"
gate.set()
deadline = time.monotonic() + 2.0
while time.monotonic() < deadline:
    if rec.get("done"):
        break
    time.sleep(0.01)
assert rec.get("done"), "fold A finished"
_reset()
c._LAST_DISCARD.clear()
m.compress = _real
print("  → PASS")

print("TEST 5: kill switch restores the synchronous fold")
_reset()
c._LAST_DISCARD.clear()
m.NX_ASYNC_COMPRESS = False  # the lazy reader getattr()s this module attr
sync_calls = []


def _sync_compress(messages, **kw):
    sync_calls.append(1)
    return _stub_compress(messages, **kw)


m.compress = _sync_compress
msgs = _msgs()
t0 = time.monotonic()
fired = m._maybe_autocompress(msgs, prompt_tokens=9000)
elapsed = time.monotonic() - t0
assert fired is True, "sync fold applies inline"
assert elapsed < 0.3, "kill switch → no background thread involved"
assert sync_calls, "compress ran synchronously on the caller thread"
assert not c._ASYNC_FOLD, "no fold state recorded when disabled"
m.compress = _real
_reset()
c._LAST_DISCARD.clear()
os.environ.pop("NX_ASYNC_COMPRESS", None)
m.NX_ASYNC_COMPRESS = True  # restore the default for the later tests
print("  → PASS")

print("TEST 6: _collect_async_fold(wait=N) joins a running fold")
_reset()
c._LAST_DISCARD.clear()
release = threading.Event()


def _gated_compress(messages, **kw):
    release.wait(2.0)
    return _stub_compress(messages, **kw)


m.compress = _gated_compress
msgs = _msgs()
m._maybe_autocompress(msgs, prompt_tokens=9000)          # fold in flight
t0 = time.monotonic()


def _releaser():
    time.sleep(0.3)
    release.set()


threading.Thread(target=_releaser, daemon=True).start()
assert m._collect_async_fold(msgs, wait=1.5) is True, "wait>0 joins the fold"
assert time.monotonic() - t0 >= 0.25, "genuinely waited for the worker"
m.compress = _real
_reset()
c._LAST_DISCARD.clear()
print("  → PASS")

print("\nALL ASYNC COMPRESS TESTS PASSED")

# Leave the module as we found it: the fake ACTIVE must not leak into a later
# test module's import (compress() reads st.ACTIVE.extra_request_kwargs() at
# request time, and a FakeSource without it turns every fold into a silent
# None).
m.ACTIVE = _saved_active

