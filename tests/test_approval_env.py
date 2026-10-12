#!/usr/bin/env python3
"""Approval default resolution: repo default, env file, env var, CLI overrides.

Tier A: nx is imported ONCE and never reloaded. Tests 1-6 cover the import-time
approval precedence chain (env file -> NX_APPROVAL -> --approval/--yolo);
instead of importlib.reload(nx) (which re-executed ~100 modules to re-read a
handful of names), the `resolve_approval` fixture re-executes the EXACT
resolution block from nx/__init__.py — sliced between anchors so the shipped
block itself stays under test, no drift — in a throwaway namespace. The
module-dict rewind that used to undo each run's writes to the approval state is
replaced by monkeypatch (see _pin_approval_state); the NX_* env rewind is
universal via conftest.
"""
import inspect
import os
import sys
import types

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nx  # noqa: E402
import nx._state as st  # noqa: E402
from nx._env import _load_env_file  # noqa: E402

# The import-time approval resolution block, sliced verbatim out of
# nx/__init__.py between two unique anchors (comment header start, next
# section header end). If the block moves, the assert fails at collection —
# loud — rather than silently exec'ing a stale copy.
_START = "# Approval default resolution (highest precedence wins):"
_END = "# --- --prompt-file / --prompt: non-interactive single-shot mode"
_SRC = inspect.getsource(nx)
assert _START in _SRC and _END in _SRC, (
    "approval-resolution block anchors moved in nx/__init__.py; update the "
    "slice here (its point is to exec the shipped block verbatim)")
_APPROVAL_BLOCK = _SRC[_SRC.index(_START):_SRC.index(_END)]


@pytest.fixture(autouse=True)
def _pin_approval_state(monkeypatch):
    # resolve_approval's exec() writes st.YOLO/APPROVE_LEVEL/
    # DEFAULT_APPROVE_LEVEL directly; registering the entry values through
    # monkeypatch makes undo() restore them at teardown — this replaces the
    # Tier-B module-dict rewind for the state these tests mutate.
    monkeypatch.setattr(st, "YOLO", st.YOLO)
    monkeypatch.setattr(st, "APPROVE_LEVEL", st.APPROVE_LEVEL)
    monkeypatch.setattr(st, "DEFAULT_APPROVE_LEVEL", st.DEFAULT_APPROVE_LEVEL)


@pytest.fixture
def resolve_approval(monkeypatch):
    """Re-run nx's import-time approval resolution under a scripted env/argv,
    replacing the old reload_nx(argv, env, env_file) helper."""

    def resolve(argv=None, env=None, env_file="/dev/null"):
        argv = ["nx.py"] if argv is None else argv
        env = {} if env is None else env
        # Same clean-env view the old reload_nx scrubbed to (env rewind is
        # universal in conftest, so no manual restore on teardown).
        for k in [k for k in os.environ if k.startswith("NX_")]:
            del os.environ[k]
        os.environ["NX_ENV_FILE"] = env_file
        os.environ.update(env)
        monkeypatch.setattr(sys, "argv", list(argv))
        # Mirror nx/__init__.py's import-time order: env-file load first,
        # then the resolution block itself, exec'd verbatim.
        _load_env_file()
        exec(_APPROVAL_BLOCK, {
            "st": st,
            "os": os,
            "sys": sys,
            "_normalize_approval": nx._normalize_approval,
            "_apply_approval": nx._apply_approval,
            "_approval_display": nx._approval_display,
        })
        return nx

    return resolve


def test_default_prompts_for_everything(resolve_approval):
    m = resolve_approval()
    assert m.YOLO is False
    assert m.APPROVE_LEVEL is None
    assert m.DEFAULT_APPROVE_LEVEL is None
    assert "prompt:all" in m._banner()


def test_nx_approval_env_sets_persistent_default(resolve_approval):
    m = resolve_approval(env={"NX_APPROVAL": "medium"})
    assert m.YOLO is False
    assert m.APPROVE_LEVEL == "medium"
    assert m.DEFAULT_APPROVE_LEVEL == "medium"


def test_nx_approval_can_come_from_env_file(resolve_approval, tmp_path):
    env_path = tmp_path / "nx.env"
    env_path.write_text("NX_APPROVAL=low\n", encoding="utf-8")
    m = resolve_approval(env_file=str(env_path))
    assert m.YOLO is False
    assert m.APPROVE_LEVEL == "low"
    assert m.DEFAULT_APPROVE_LEVEL == "low"


def test_cli_approval_overrides_env_yolo(resolve_approval):
    m = resolve_approval(
        argv=["nx.py", "--approval", "low"],
        env={"NX_APPROVAL": "yolo"},
    )
    assert m.YOLO is False
    assert m.APPROVE_LEVEL == "low"
    assert m.DEFAULT_APPROVE_LEVEL == "low"


def test_cli_yolo_overrides_approval_level(resolve_approval):
    m = resolve_approval(
        argv=["nx.py", "--approval", "medium", "--yolo"],
        env={"NX_APPROVAL": "low"},
    )
    assert m.YOLO is True
    assert m.APPROVE_LEVEL is None
    assert m.DEFAULT_APPROVE_LEVEL == "medium"


def test_prompt_all_aliases_are_accepted(resolve_approval):
    m = resolve_approval(argv=["nx.py", "--approval", "all"],
                         env={"NX_APPROVAL": "medium"})
    assert m.YOLO is False
    assert m.APPROVE_LEVEL is None
    assert m.DEFAULT_APPROVE_LEVEL is None
    assert m._normalize_approval("strict") == ("prompt_all", None)


def test_risk_classifier_retries_connection_failures(monkeypatch):
    # The old test reloaded nx and save/restored m.client/_log_event/knobs by
    # hand; Tier A patches the SAME names through st (the single state home;
    # nx.<name> proxies here) and the consumer module's _log_event binding, all
    # auto-restored by monkeypatch.
    calls = {"n": 0}
    request = httpx.Request("POST", "http://test/v1/chat/completions")

    class Resp:
        choices = [types.SimpleNamespace(
            message=types.SimpleNamespace(
                content='{"level":"low","reason":"retry recovered"}'))]

        def model_dump(self):
            return {"ok": True}

    def create(**_):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise nx.APIConnectionError(message="test connection failure",
                                        request=request)
        return Resp()

    monkeypatch.setattr(st, "client", types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create))))
    monkeypatch.setattr(st, "MODEL", st.MODEL or "test-model")
    monkeypatch.setattr(st, "RISK_CONNECTION_RETRIES", 3)
    monkeypatch.setattr(st, "RISK_CONNECTION_RETRY_SECONDS", 0)
    # _risk did `from ._logging import _log_event` — patch the name it reads.
    monkeypatch.setattr("nx._risk._log_event", lambda *_: None)

    assert nx._assess_risk("run: true") == ("low", "retry recovered")
    assert calls["n"] == 4


def test_open_stream_only_sets_sampler_params_for_recovery(monkeypatch):
    calls = []

    class Resp:
        def __iter__(self):
            return iter(())

    def create(**kwargs):
        calls.append(kwargs)
        return Resp()

    monkeypatch.setattr(st, "client", types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create))))
    monkeypatch.setattr(st, "MODEL", st.MODEL or "test-model")
    monkeypatch.setattr(st, "RECOVERY_TEMPERATURE", 0.2)
    monkeypatch.setattr(st, "RECOVERY_TOP_P", 0.95)
    # _compress did `from ._logging import _log_event, _llog` — patch the
    # names it actually reads (the old test's m._log_event/m._llog patches
    # silently did nothing; the module-level rewind hid that).
    monkeypatch.setattr("nx._compress._log_event", lambda *_: None)

    nx.open_stream([{"role": "user", "content": "normal"}])
    nx.open_stream([{"role": "user", "content": "recover"}],
                   recovery_sampling=True)

    assert "temperature" not in calls[0]
    assert "top_p" not in calls[0]
    assert calls[1]["temperature"] == 0.2
    assert calls[1]["top_p"] == 0.95
