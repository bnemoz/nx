#!/usr/bin/env python3
"""Approval default resolution: repo default, env file, env var, CLI overrides."""
import importlib
import os
import sys
import types

import httpx


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def reload_nx(argv=None, env=None, env_file="/dev/null"):
    argv = ["nx.py"] if argv is None else argv
    env = {} if env is None else env
    for k in list(os.environ):
        if k.startswith("NX_"):
            del os.environ[k]
    os.environ["NX_ENV_FILE"] = env_file
    os.environ.update(env)
    sys.argv = argv
    import nx
    return importlib.reload(nx)


def test_default_prompts_for_everything():
    m = reload_nx()
    assert m.YOLO is False
    assert m.APPROVE_LEVEL is None
    assert m.DEFAULT_APPROVE_LEVEL is None
    assert "prompt:all" in m._banner()


def test_nx_approval_env_sets_persistent_default():
    m = reload_nx(env={"NX_APPROVAL": "medium"})
    assert m.YOLO is False
    assert m.APPROVE_LEVEL == "medium"
    assert m.DEFAULT_APPROVE_LEVEL == "medium"


def test_nx_approval_can_come_from_env_file(tmp_path):
    env_path = tmp_path / "nx.env"
    env_path.write_text("NX_APPROVAL=low\n", encoding="utf-8")
    m = reload_nx(env_file=str(env_path))
    assert m.YOLO is False
    assert m.APPROVE_LEVEL == "low"
    assert m.DEFAULT_APPROVE_LEVEL == "low"


def test_cli_approval_overrides_env_yolo():
    m = reload_nx(
        argv=["nx.py", "--approval", "low"],
        env={"NX_APPROVAL": "yolo"},
    )
    assert m.YOLO is False
    assert m.APPROVE_LEVEL == "low"
    assert m.DEFAULT_APPROVE_LEVEL == "low"


def test_cli_yolo_overrides_approval_level():
    m = reload_nx(
        argv=["nx.py", "--approval", "medium", "--yolo"],
        env={"NX_APPROVAL": "low"},
    )
    assert m.YOLO is True
    assert m.APPROVE_LEVEL is None
    assert m.DEFAULT_APPROVE_LEVEL == "medium"


def test_prompt_all_aliases_are_accepted():
    m = reload_nx(argv=["nx.py", "--approval", "all"],
                      env={"NX_APPROVAL": "medium"})
    assert m.YOLO is False
    assert m.APPROVE_LEVEL is None
    assert m.DEFAULT_APPROVE_LEVEL is None
    assert m._normalize_approval("strict") == ("prompt_all", None)


def test_risk_classifier_retries_connection_failures():
    m = reload_nx()
    saved = (
        m.client,
        m._log_event,
        m.RISK_CONNECTION_RETRIES,
        m.RISK_CONNECTION_RETRY_SECONDS,
    )
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
            raise m.APIConnectionError(message="test connection failure",
                                       request=request)
        return Resp()

    try:
        m.RISK_CONNECTION_RETRIES = 3
        m.RISK_CONNECTION_RETRY_SECONDS = 0
        m._log_event = lambda *_: None
        m.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)))

        assert m._assess_risk("run: true") == ("low", "retry recovered")
        assert calls["n"] == 4
    finally:
        (
            m.client,
            m._log_event,
            m.RISK_CONNECTION_RETRIES,
            m.RISK_CONNECTION_RETRY_SECONDS,
        ) = saved


def test_open_stream_only_sets_sampler_params_for_recovery():
    m = reload_nx()
    saved = (
        m.client,
        m._log_event,
        m._llog,
        m.RECOVERY_TEMPERATURE,
        m.RECOVERY_TOP_P,
    )
    calls = []

    class Resp:
        def __iter__(self):
            return iter(())

    def create(**kwargs):
        calls.append(kwargs)
        return Resp()

    try:
        m.RECOVERY_TEMPERATURE = 0.2
        m.RECOVERY_TOP_P = 0.95
        m._log_event = lambda *_: None
        m._llog = None
        m.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)))

        m.open_stream([{"role": "user", "content": "normal"}])
        m.open_stream([{"role": "user", "content": "recover"}],
                      recovery_sampling=True)

        assert "temperature" not in calls[0]
        assert "top_p" not in calls[0]
        assert calls[1]["temperature"] == 0.2
        assert calls[1]["top_p"] == 0.95
    finally:
        (
            m.client,
            m._log_event,
            m._llog,
            m.RECOVERY_TEMPERATURE,
            m.RECOVERY_TOP_P,
        ) = saved
