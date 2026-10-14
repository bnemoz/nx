"""Per-subagent model routing (0.86.0) — NX_SUBAGENT_SOURCE / NX_SUBAGENT_MODEL.

Tests the same contract as the compress-fold routing (test_undo_compress_v2.
test_compress_call_is_bounded_and_routable): the resolved (client, model) is
used as-is for the sub-agent's model calls, unknown sources fall back to the
parent's active source with a one-time warning, and nothing is swapped
process-globally (concurrent agents must not race).

Offline: open_stream() is exercised through a fake client via the thread-local
_SubagentContext, never through a real endpoint.
"""
import importlib
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR",
          "NX_SUBAGENT_SOURCE", "NX_SUBAGENT_MODEL"):
    os.environ.pop(k, None)

import pytest


@pytest.fixture()
def nx_env(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    yield nx, tmp_path


class _FakeResp:
    def __init__(self):
        self.choices = []
        self.usage = None

    def model_dump(self):
        return {}


class _FakeCompletions:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return _FakeResp()


class _FakeClient:
    def __init__(self):
        self.base_url = "http://fake"
        self.chat = type("H", (), {})()
        self.chat.completions = _FakeCompletions()


def test_default_inherits_parent(nx_env):
    nx, tmp = nx_env
    from nx._subagent_model import subagent_call_target
    assert subagent_call_target() is None


def test_model_alone_keeps_parent_client(nx_env):
    nx, tmp = nx_env
    from nx._subagent_model import subagent_call_target
    nx._state.client = _FakeClient()
    nx._state.MODEL = "parent-model"
    nx._state.ACTIVE = None
    nx.NX_SUBAGENT_MODEL = "glm-5.3-flash"
    t = subagent_call_target()
    assert t is not None
    assert t[0] is nx._state.client and t[1] == "glm-5.3-flash"


def test_source_routes_to_that_source(nx_env):
    nx, tmp = nx_env
    from nx._subagent_model import subagent_call_target
    from nx._sources import Source
    cheap_client = _FakeClient()
    src = Source("flash", "http://localhost:9/v1", "k", model="glm-5.3-flash")
    src.client = cheap_client
    nx._state.SOURCES["flash"] = src
    nx._state.client = _FakeClient()
    nx._state.MODEL = "parent-model"
    nx._state.ACTIVE = None
    nx.NX_SUBAGENT_SOURCE = "flash"
    t = subagent_call_target()
    assert t is not None
    assert t[0] is cheap_client and t[1] == "glm-5.3-flash"
    # explicit model overrides the source's default
    nx.NX_SUBAGENT_MODEL = "qwen3.8-flash-next"
    t2 = subagent_call_target()
    assert t2[1] == "qwen3.8-flash-next"


def test_model_pool_round_robin(nx_env):
    nx, tmp = nx_env
    from nx._subagent_model import subagent_call_target
    nx._state.client = _FakeClient()
    nx._state.MODEL = "parent-model"
    nx._state.ACTIVE = None
    os.environ["NX_SUBAGENT_MODEL_POOL"] = "a-model,flash:b-model"
    from nx._sources import Source
    src = Source("flash", "http://localhost:9/v1", "k", model="glm-5.3-flash")
    flash_client = _FakeClient()
    src.client = flash_client
    nx._state.SOURCES["flash"] = src
    seen = [subagent_call_target() for _ in range(4)]
    assert [t[1] for t in seen] == ["a-model", "b-model", "a-model", "b-model"]
    assert seen[0][0] is nx._state.client      # bare model → parent's client
    assert seen[1][0] is flash_client          # source:model → that endpoint
    os.environ.pop("NX_SUBAGENT_MODEL_POOL")


def test_unknown_source_warns_once_and_falls_back(nx_env):
    nx, tmp = nx_env
    from nx._subagent_model import subagent_call_target
    nx._state.client = _FakeClient()
    nx._state.MODEL = "parent-model"
    nx._state.ACTIVE = None
    nx.NX_SUBAGENT_SOURCE = "nope"
    assert subagent_call_target() is None
    assert nx._state._SUBAGENT_SOURCE_WARNED is True


def test_open_stream_uses_thread_local_target(nx_env):
    """open_stream() reads the target from THIS thread's _SubagentContext —
    a sibling thread with a different (or no) target is unaffected."""
    nx, tmp = nx_env
    from nx import _compress as cp
    from nx._subagent import _SubagentContext, _SUBAGENT_TLS

    parent_client = _FakeClient()
    cheap_client = _FakeClient()
    nx._state.client = parent_client
    nx._state.MODEL = "parent-model"
    ctx = _SubagentContext({}, quiet=True)
    # duck-typed source stand-in with its own ladder store
    class _Src:
        _request_shapes = {}
    install_ctx = ctx
    install_ctx.model_target = (cheap_client, "flash-model", {}, _Src())
    _SUBAGENT_TLS.ctx = install_ctx
    try:
        cp.open_stream([{"role": "user", "content": "hi"}], tools=[])
    finally:
        _SUBAGENT_TLS.ctx = None
    reqs = cheap_client.chat.completions.requests
    assert len(reqs) == 1 and reqs[0]["model"] == "flash-model"
    assert not parent_client.chat.completions.requests


def test_concurrent_agents_do_not_clobber(nx_env):
    nx, tmp = nx_env
    from nx import _compress as cp
    from nx._subagent import _SubagentContext, _SUBAGENT_TLS

    client_a, client_b = _FakeClient(), _FakeClient()
    nx._state.client = _FakeClient()
    nx._state.MODEL = "parent-model"
    results = {}

    def _run(name, client, model):
        ctx = _SubagentContext({}, quiet=True)
        ctx.model_target = (client, model, {}, None)
        _SUBAGENT_TLS.ctx = ctx
        try:
            cp.open_stream([{"role": "user", "content": "hi"}], tools=[])
            results[name] = client.chat.completions.requests[0]["model"]
        finally:
            _SUBAGENT_TLS.ctx = None

    threads = [threading.Thread(target=_run, args=("a", client_a, "model-a")),
               threading.Thread(target=_run, args=("b", client_b, "model-b"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {"a": "model-a", "b": "model-b"}


def test_install_resolves_env_knobs(nx_env):
    nx, tmp = nx_env
    from nx._subagent import _SubagentContext
    from nx._subagent_model import install
    from nx._sources import Source
    src = Source("flash", "http://localhost:9/v1", "k", model="glm-5.3-flash")
    src.client = _FakeClient()
    nx._state.SOURCES["flash"] = src
    nx._state.client = _FakeClient()
    nx._state.MODEL = "parent-model"
    nx._state.ACTIVE = None
    nx.NX_SUBAGENT_SOURCE = "flash"
    ctx = _SubagentContext({}, quiet=True)
    install(ctx)
    assert ctx.model_target is not None
    assert ctx.model_target[1] == "glm-5.3-flash"
    # unset → inherit
    nx.NX_SUBAGENT_SOURCE = ""
    ctx2 = _SubagentContext({}, quiet=True)
    install(ctx2)
    assert ctx2.model_target is None


# --- 0.156.0 phase M: the per-TASK call target (batch tasks) -------------------
# `call_target_scope` is the batch-task sibling of NX_SUBAGENT_*: the task's
# resolved 4-tuple lives on the worker thread and open_stream() prefers it over
# the process globals. Same fake-client pattern as above — nothing hits a wire.


def test_call_target_scope_sets_and_restores(nx_env):
    """Scope installs the target for the body and DELETES it afterwards, so
    'no target' stays observable (the _posture_scope contract)."""
    from nx._subagent_model import call_target, call_target_scope
    t = (_FakeClient(), "task-model", {}, None)
    assert call_target() is None
    with call_target_scope(t):
        assert call_target() is t
    assert call_target() is None
    assert not hasattr(__import__("nx._subagent_model", fromlist=["x"])._CALL_TARGET_TLS,
                       "target")


def test_call_target_scope_nests_and_restores_prev(nx_env):
    """scope inside scope restores the OUTER target, not None — and None means
    INHERIT (nothing installed), so a nested None body keeps seeing the outer."""
    from nx._subagent_model import call_target, call_target_scope
    inner_client, outer_client = _FakeClient(), _FakeClient()
    outer = (outer_client, "outer-model", {}, None)
    inner = (inner_client, "inner-model", {}, None)
    with call_target_scope(outer):
        assert call_target() is outer
        with call_target_scope(inner):
            assert call_target() is inner
        assert call_target() is outer
        with call_target_scope(None):
            assert call_target() is outer  # None = inherit, installs nothing
        assert call_target() is outer
    assert call_target() is None


def test_call_target_scope_restores_on_exception(nx_env):
    from nx._subagent_model import call_target, call_target_scope
    t = (_FakeClient(), "task-model", {}, None)
    with pytest.raises(RuntimeError):
        with call_target_scope(t):
            raise RuntimeError("boom")
    assert call_target() is None


def test_open_stream_prefers_ctx_over_call_target(nx_env):
    """Resolution order ctx > batch TLS: a sub-agent body running INSIDE a batch
    task must ride its OWN target, never the task's."""
    nx, tmp = nx_env
    from nx import _compress as cp
    from nx._subagent import _SubagentContext, _SUBAGENT_TLS
    from nx._subagent_model import call_target_scope

    parent_client, task_client, agent_client = _FakeClient(), _FakeClient(), _FakeClient()
    nx._state.client = parent_client
    nx._state.MODEL = "parent-model"
    ctx = _SubagentContext({}, quiet=True)
    ctx.model_target = (agent_client, "agent-model", {}, None)
    with call_target_scope((task_client, "task-model", {}, None)):
        _SUBAGENT_TLS.ctx = ctx
        try:
            cp.open_stream([{"role": "user", "content": "hi"}], tools=[])
        finally:
            _SUBAGENT_TLS.ctx = None
    assert [r["model"] for r in agent_client.chat.completions.requests] == ["agent-model"]
    assert not task_client.chat.completions.requests
    assert not parent_client.chat.completions.requests


def test_open_stream_uses_call_target_over_globals(nx_env):
    """A batch-task target wins over the globals — and its own extra body is
    what rides the wire (the tuple's call_extra, not st.ACTIVE's)."""
    nx, tmp = nx_env
    from nx import _compress as cp
    from nx._subagent_model import call_target_scope

    parent_client, task_client = _FakeClient(), _FakeClient()
    nx._state.client = parent_client
    nx._state.MODEL = "parent-model"
    task_extra = {"extra_body": {"provider": {"order": ["task-endpoint"]}}}
    with call_target_scope((task_client, "task-model", task_extra, None)):
        cp.open_stream([{"role": "user", "content": "hi"}], tools=[])
    reqs = task_client.chat.completions.requests
    assert len(reqs) == 1 and reqs[0]["model"] == "task-model"
    assert reqs[0]["extra_body"]["provider"] == {"order": ["task-endpoint"]}
    assert not parent_client.chat.completions.requests


def test_open_stream_call_target_with_unconfigured_globals(nx_env):
    """The guard fix: a task target reaches the wire even when st.client/
    st.MODEL are unset (the old guard demanded globals and silently fell back)."""
    nx, tmp = nx_env
    from nx import _compress as cp
    from nx._subagent_model import call_target_scope

    task_client = _FakeClient()
    nx._state.client = None
    nx._state.MODEL = None
    nx._state.ACTIVE = None
    with call_target_scope((task_client, "task-model", {}, None)):
        cp.open_stream([{"role": "user", "content": "hi"}], tools=[])
    assert [r["model"] for r in task_client.chat.completions.requests] == ["task-model"]


def test_open_stream_without_any_target_uses_globals(nx_env):
    """Unrouted path unchanged: no ctx, no scope → globals, exactly as before."""
    nx, tmp = nx_env
    from nx import _compress as cp

    parent_client = _FakeClient()
    nx._state.client = parent_client
    nx._state.MODEL = "parent-model"
    nx._state.ACTIVE = None
    cp.open_stream([{"role": "user", "content": "hi"}], tools=[])
    assert [r["model"] for r in parent_client.chat.completions.requests] == ["parent-model"]


def test_bare_model_agent_inside_scope_rides_scope_client(nx_env):
    """Inherit-chain fix: a sub-agent with only a bare model id inside a batch
    task rides the TASK's client (run → task → agent), not the run's globals."""
    nx, tmp = nx_env
    from nx._subagent_model import call_target_scope, subagent_call_target
    from nx._subagent import _SubagentContext
    from nx._sources import Source

    parent_client, task_client = _FakeClient(), _FakeClient()
    nx._state.client = parent_client
    nx._state.MODEL = "parent-model"
    task_src = Source("task", "http://task/v1", "k", model="task-src-model")
    task_src.client = task_client
    ctx = _SubagentContext({}, quiet=True)
    nx.NX_SUBAGENT_MODEL = "pooled-model"
    with call_target_scope((task_client, "task-model", {}, task_src)):
        t = subagent_call_target()
    nx.NX_SUBAGENT_MODEL = ""
    assert t is not None
    assert t[0] is task_client and t[1] == "pooled-model"
    assert t[3] is task_src  # ladder memory stays the task endpoint's


def test_bare_model_agent_outside_scope_still_rides_parent(nx_env):
    """The pre-0.156.0 default is untouched: no scope → parent globals."""
    nx, tmp = nx_env
    from nx._subagent_model import subagent_call_target

    parent_client = _FakeClient()
    nx._state.client = parent_client
    nx._state.MODEL = "parent-model"
    nx._state.ACTIVE = None
    nx.NX_SUBAGENT_MODEL = "glm-5.3-flash"
    try:
        t = subagent_call_target()
    finally:
        nx.NX_SUBAGENT_MODEL = ""
    assert t is not None and t[0] is parent_client


class _WindowSrc:
    """Duck-typed source stand-in whose window the clamp must follow."""

    def __init__(self, window):
        self._window = window
        self._request_shapes = {}
        self._capability = None

    def resolve_context_window(self):
        return self._window

    def extra_request_kwargs(self, model=None):
        return {}


def test_routed_target_clamp_follows_target_window(nx_env):
    """The _request_opts wart fix: a routed target is clamped by ITS source's
    window, not the process default's. Two sources, two windows — the wire's
    max_tokens must follow the TARGET."""
    nx, tmp = nx_env
    from nx import _compress as cp
    from nx._subagent_model import call_target_scope

    parent_client, small_client = _FakeClient(), _FakeClient()
    nx._state.client = parent_client
    nx._state.MODEL = "parent-model"
    nx._state.ACTIVE = _WindowSrc(200_000)      # roomy process default
    small_src = _WindowSrc(4_000)               # cramped task endpoint
    # a prompt that cannot fit the 4k task window but sits far inside 200k:
    # routed → floored at MIN_COMPLETION_TOKENS, unrouted → MAX_COMPLETION_TOKENS
    msgs = [{"role": "user", "content": "x" * 20_000}]
    with call_target_scope((small_client, "task-model", {}, small_src)):
        cp.open_stream(msgs, tools=[])
    sent = small_client.chat.completions.requests[0]["max_tokens"]
    expected = max(1, nx._state.MIN_COMPLETION_TOKENS)
    assert sent == expected, (sent, expected)
    assert not parent_client.chat.completions.requests
    # and the unrouted path is unchanged: the roomy source never clamps
    nx._state.ACTIVE = _WindowSrc(200_000)
    cp.open_stream(msgs, tools=[])
    assert parent_client.chat.completions.requests[-1]["max_tokens"] == cp.MAX_COMPLETION_TOKENS


def test_grammar_attach_follows_routed_source(nx_env):
    """The grammar half of the wart: the CAPABILITY verdict of the ROUTED
    source decides the attach, not the process default's."""
    nx, tmp = nx_env
    from nx import _compress as cp
    from nx._subagent_model import call_target_scope
    from nx._capability import CAPABILITY_GRAMMAR, CAPABILITY_NATIVE

    task_client = _FakeClient()
    nx._state.ACTIVE = _WindowSrc(0)
    nx._state.ACTIVE._capability = CAPABILITY_NATIVE   # globals: native → no grammar
    routed = _WindowSrc(0)
    routed._capability = CAPABILITY_GRAMMAR            # task: grammar helps → attach
    with call_target_scope((task_client, "task-model", {}, routed)):
        cp.open_stream([{"role": "user", "content": "hi"}], tools=list(cp.TOOLS))
    body = task_client.chat.completions.requests[0].get("extra_body") or {}
    assert "grammar" in body, task_client.chat.completions.requests[0]


def test_sibling_threads_do_not_clobber_call_target(nx_env):
    """Mirror of test_concurrent_agents_do_not_clobber for call_target_scope:
    two threads in their own scopes each reach the wire on their own client."""
    nx, tmp = nx_env
    from nx import _compress as cp
    from nx._subagent_model import call_target, call_target_scope

    client_a, client_b = _FakeClient(), _FakeClient()
    nx._state.client = _FakeClient()
    nx._state.MODEL = "parent-model"
    results = {}

    def _run(name, client, model):
        with call_target_scope((client, model, {}, None)):
            cp.open_stream([{"role": "user", "content": "hi"}], tools=[])
            results[name] = client.chat.completions.requests[0]["model"]
        assert call_target() is None  # scope restored inside the worker thread

    threads = [threading.Thread(target=_run, args=("a", client_a, "model-a")),
               threading.Thread(target=_run, args=("b", client_b, "model-b"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {"a": "model-a", "b": "model-b"}


def test_scope_never_touches_subagent_ctx(nx_env):
    """Guard (spec: 'batch never sets _SUBAGENT_TLS.ctx'): the batch slot is a
    full agent — entering a call_target_scope must leave the sub-agent ctx
    exactly as it was."""
    nx, tmp = nx_env
    from nx._subagent_model import call_target_scope
    from nx._subagent import _SUBAGENT_TLS

    assert getattr(_SUBAGENT_TLS, "ctx", None) is None
    with call_target_scope((_FakeClient(), "task-model", {}, None)):
        assert getattr(_SUBAGENT_TLS, "ctx", None) is None
    assert getattr(_SUBAGENT_TLS, "ctx", None) is None
