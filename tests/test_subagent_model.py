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
    from nx._subagent_model import install

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
