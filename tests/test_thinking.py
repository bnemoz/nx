#!/usr/bin/env python3
"""Tests for nx/_thinking.py — the thinking-separation auto-probe.

glm-5.3-flash behind a vLLM-backed proxy streams its private reasoning as
plain `content` with no separator, so nx's reasoning-block rendering and the
reasoning-only stall guards never fire (the reasoning runs straight into the
visible answer: "...No need for tools.Hey! I'm ready..."). The server only
separates them when `chat_template_kwargs: {"thinking": true}` rides in the
request body.

_thinking._probe_kwargs_for() resolves that kwarg by asking the server (one
baseline request + one per candidate toggle) and caches it on the Source,
model-keyed; the kwarg is then attached to every request via
Source.extra_request_kwargs().

Offline: fake clients record requests and return canned responses. The suite
must not depend on a live model endpoint.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Scrub env so _discover_sources builds a clean single 'local' source, and no
# local credentials can affect results.
for k in list(os.environ):
    if k.startswith("NX_"):
        del os.environ[k]
os.environ.pop("TOGETHER_API_KEY", None)
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SOURCE_LOCAL_BASE_URL"] = "http://localhost:8080/v1"
sys.argv = ["nx.py"]
import nx as m  # noqa: E402
import nx._thinking as th  # noqa: E402


# --- canned non-streaming responses ------------------------------------------
class _Resp:
    """Shape of our _openai_client._Obj for a non-streaming completion."""

    def __init__(self, reasoning=None, content="hi", raw=None):
        if raw is not None:
            self._raw = raw
            return
        msg = {"role": "assistant", "content": content}
        if reasoning is not None:
            msg["reasoning_content"] = reasoning
        self._raw = {"choices": [{"index": 0, "message": msg,
                                 "finish_reason": "stop"}]}

    def model_dump(self):
        return self._raw


class _Recorder:
    """Fake OpenAI-ish client that records every create() call. `handler`
    receives the kwargs and returns a _Resp (or raises)."""

    def __init__(self, handler):
        self.requests = []
        self._handler = handler
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.requests.append(kwargs)
                return outer._handler(kwargs)

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        class _Models:
            def list(self, **kw):
                return None

        self.chat = _Chat()
        self.models = _Models()


def _make_source(client):
    src = m.Source("test", "http://localhost:8080/v1", "sk-noop", "test-model")
    src.client = client
    return src


def _ctk(kwargs):
    """chat_template_kwargs dict of a recorded request, or {}."""
    eb = (kwargs.get("extra_body") or {})
    return eb.get("chat_template_kwargs") or {}


# --- Test 1: merged server → probe resolves the thinking kwarg ----------------
print("TEST 1: probe resolves chat_template_kwargs.thinking on a merged server")


def merged_handler(kwargs):
    # glm-5.3-flash's observed proxy behavior: no separation unless asked.
    if _ctk(kwargs).get("thinking") is True:
        return _Resp(reasoning="Let me think…", content="Hello!")
    if _ctk(kwargs).get("enable_thinking") is True:
        return _Resp(reasoning="Let me think…", content="Hello!")
    return _Resp(content="Let me think…Hello!")


cli = _Recorder(merged_handler)
src = _make_source(cli)
kw = th._probe_kwargs_for(src, "glm-5.3-flash")
print(f"  resolved: {kw}")
assert kw == {"chat_template_kwargs": {"thinking": True}}, f"unexpected kw: {kw}"
# One baseline + exactly one candidate request (the first candidate hits).
assert len(cli.requests) == 2, f"expected 2 requests, got {len(cli.requests)}"
assert cli.requests[0].get("extra_body") is None, "baseline must be bare"
print("  PASS\n")

# --- Test 2: already-separating server gets NO kwarg attached -----------------
print("TEST 2: already-separated → None (never attach anything)")


def separating_handler(kwargs):
    return _Resp(reasoning="already separate", content="Hello!")


cli = _Recorder(separating_handler)
src = _make_source(cli)
kw = th._probe_kwargs_for(src, "glm-5.3")
print(f"  resolved: {kw}")
assert kw is None, f"expected None for already-separating server, got {kw}"
assert len(cli.requests) == 1, "baseline only — must stop after a separated answer"
print("  PASS\n")

# --- Test 3: second candidate (enable_thinking) used when thinking fails -----
print("TEST 3: falls through to the enable_thinking candidate")


def qwen_handler(kwargs):
    if _ctk(kwargs).get("enable_thinking") is True:
        return _Resp(reasoning="hmm", content="ok")
    return _Resp(content="hmm ok")


cli = _Recorder(qwen_handler)
src = _make_source(cli)
kw = th._probe_kwargs_for(src, "qwen-x")
print(f"  resolved: {kw}")
assert kw == {"chat_template_kwargs": {"enable_thinking": True}}, f"unexpected kw: {kw}"
assert len(cli.requests) == 3
print("  PASS\n")

# --- Test 4: rejecting server → no kwarg, no crash ---------------------------
print("TEST 4: server rejects the extra field → None")


def reject_handler(kwargs):
    if kwargs.get("extra_body"):
        raise RuntimeError("Error code: 400 - Unrecognized request argument")
    return _Resp(content="merged, no toggle accepted")


cli = _Recorder(reject_handler)
src = _make_source(cli)
kw = th._probe_kwargs_for(src, "gpt-x")
print(f"  resolved: {kw}")
assert kw is None
assert len(cli.requests) == 3, "baseline + both candidates tried, all rejected"
print("  PASS\n")

# --- Test 5: extra_request_kwargs() attaches the probed kwarg -----------------
print("TEST 5: extra_request_kwargs deep-merges the probed kwarg")
cli = _Recorder(merged_handler)
src = _make_source(cli)
src._thinking_kwargs["glm-5.3-flash"] = {"chat_template_kwargs": {"thinking": True}}
src.extra_body = {"provider": {"order": ["x"]},  # must survive alongside
                  "chat_template_kwargs": {"other": 1}}
m.st.MODEL = "glm-5.3-flash"
m.st._METRICS = None  # PF-10: no session recorder -> no routing hint attached
try:
    out = src.extra_request_kwargs()
finally:
    m.st.MODEL = None
print(f"  merged: {out}")
eb = out.get("extra_body") or {}
assert eb.get("provider") == {"order": ["x"]}, "configured EXTRA_BODY must survive"
assert eb.get("chat_template_kwargs") == {"other": 1, "thinking": True}, \
    "thinking must deep-merge into an existing chat_template_kwargs dict"
assert src.extra_body["chat_template_kwargs"] == {"other": 1}, \
    "the merge must not pollute src.extra_body's nested dicts in place"
# …and a source with no probe result yet passes its configured body untouched.
src2 = _make_source(_Recorder(merged_handler))
src2.extra_body = {"temperature": 0.2}
m.st._METRICS = None  # PF-10: the routing hint is session-scoped
assert src2.extra_request_kwargs() == {"extra_body": {"temperature": 0.2}}
print("  PASS\n")

# --- Test 6: probe never runs when NX_THINKING=off ---------------------------
print("TEST 6: THINKING_MODE=off skips the probe entirely")
cli = _Recorder(merged_handler)
src = _make_source(cli)
m.st.THINKING_MODE = "off"
try:
    th.ensure_thinking_probe(src, "glm-5.3-flash")
    time.sleep(0.2)  # background thread: give it a chance to (wrongly) fire
    assert not cli.requests, f"probe fired despite THINKING_MODE=off: {cli.requests}"
    assert src._thinking_kwargs == {}
finally:
    m.st.THINKING_MODE = "auto"
print("  PASS\n")

# --- Test 7: ensure_thinking_probe populates the cache in background ---------
print("TEST 7: ensure_thinking_probe caches per model id, no re-probe")
cli = _Recorder(merged_handler)
src = _make_source(cli)
m.st.MODEL = "glm-5.3-flash"
# Hermetic lock: the real STARTUP_PROBE_LOCK serializes this background
# probe against capability/context probes. In a full-suite run, earlier
# modules (test_sources.py's import-time switch_source calls) leave a live
# probe holding that lock for real network-timeout seconds, which
# intermittently starved the 5s cache wait below — a cross-module race,
# not a thinking-probe behavior. Swap in a no-op lock for this test only;
# the probe talks solely to the fake client above either way.
import nx._capability as _cap  # noqa: E402


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_real_lock = _cap.STARTUP_PROBE_LOCK
_cap.STARTUP_PROBE_LOCK = _NullLock()
try:
    th.ensure_thinking_probe(src, "glm-5.3-flash")
    for _ in range(100):
        if "glm-5.3-flash" in src._thinking_kwargs:
            break
        time.sleep(0.05)
    got = src._thinking_kwargs.get("glm-5.3-flash")
    print(f"  cached: {got}")
    assert got == {"chat_template_kwargs": {"thinking": True}}
    n_req = len(cli.requests)
    th.ensure_thinking_probe(src, "glm-5.3-flash")
    time.sleep(0.15)
    assert len(cli.requests) == n_req, "re-probing an already-cached model"
    assert n_req == 2, f"probe should have made 2 requests, made {n_req}"
    # A different model id on the SAME source is probed separately.
    th.ensure_thinking_probe(src, "glm-5.3")
    for _ in range(100):
        if "glm-5.3" in src._thinking_kwargs:
            break
        time.sleep(0.05)
    assert src._thinking_kwargs.get("glm-5.3") == {"chat_template_kwargs": {"thinking": True}}
finally:
    _cap.STARTUP_PROBE_LOCK = _real_lock
    m.st.MODEL = None
print("  PASS\n")

# --- Test 8: malformed response shapes degrade to None -----------------------
print("TEST 8: malformed response shapes degrade to None")
cli = _Recorder(lambda kwargs: _Resp(raw={"nope": "no choices here"}))
src = _make_source(cli)
kw = th._probe_kwargs_for(src, "weird")
print(f"  resolved: {kw}")
assert kw is None
print("  PASS\n")

print("ALL TESTS PASSED")
