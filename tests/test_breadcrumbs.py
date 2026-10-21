#!/usr/bin/env python3
"""Phase 6 decision breadcrumbs (docs/EXPLAINABLE_HARNESS_SPEC.md).

note() gates on the setting AND the quiet flag; the three instrumented
decision sites (malformed-args salvage, retry ladder, breaker trip/reset)
emit through it; /breadcrumbs flips the state like the other settings
toggles. Default is ON."""
import os
import sys

_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_breadcrumbs")
os.makedirs(_tmp, exist_ok=True)
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ.pop("NX_BREADCRUMBS", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nx import _breadcrumbs as bc  # noqa: E402
from nx import _repl_settings_cmds as rs  # noqa: E402


def _on():
    bc.set_enabled(True)


def _off():
    bc.set_enabled(False)


def test_default_is_on(monkeypatch):
    monkeypatch.delenv("NX_BREADCRUMBS", raising=False)
    assert bc.ENABLED is True


def test_note_prints_when_on(capsys):
    _on()
    bc.note("salvaged malformed tool args (attempt 2/3)")
    out = capsys.readouterr().out
    assert "[harness] salvaged malformed tool args (attempt 2/3)" in out


def test_note_silent_when_off(capsys):
    _off()
    bc.note("salvaged malformed tool args (attempt 2/3)")
    assert "[harness]" not in capsys.readouterr().out


def test_note_silent_when_quiet(capsys, monkeypatch):
    _on()
    monkeypatch.setattr(bc, "_quiet", lambda: True)
    bc.note("retrying after timeout (ladder step 2)")
    assert "[harness]" not in capsys.readouterr().out


def test_salvage_site_emits_breadcrumb(capsys):
    _on()
    from nx._argsalvage import _salvage_truncated_args
    # a truncated read-tool payload: cut inside a nested object
    raw = '{"path": "docs/REFERENCE.md", "opts": {"offset": 1'
    obj = _salvage_truncated_args(raw)
    assert obj is not None and obj["path"] == "docs/REFERENCE.md"
    assert "[harness] salvaged malformed tool args" in capsys.readouterr().out
    # and no breadcrumb when the salvage refuses (odd quote count)
    capsys.readouterr()
    assert _salvage_truncated_args('{"path": "a/b') is None
    assert "[harness]" not in capsys.readouterr().out


def test_retry_site_emits_breadcrumb(capsys, monkeypatch):
    _on()
    monkeypatch.setattr("nx._logging.usage_retry", lambda *a, **k: None)
    from nx._turn_retry import _retry_log
    _retry_log(2, "timeout")
    assert "[harness] retrying after timeout (ladder step 2)" in capsys.readouterr().out


def test_breaker_trip_emits_breadcrumb(capsys, monkeypatch):
    _on()
    from nx import _state as st
    from nx import _breaker as br
    monkeypatch.setattr("nx._logging.usage_breaker", lambda *a, **k: None)
    st._MALFORMED_TOOL_FAILURES.clear()
    st._MALFORMED_TOOL_BROKEN.clear()
    br._TRIP_ANNOUNCED.clear()
    br.rearm_run_tools_trip_watch()
    try:
        st._MALFORMED_TOOL_FAILURES["multi_edit"] = 3
        st._MALFORMED_TOOL_BROKEN.add("multi_edit")
        assert br._breaker_broken("multi_edit") is True
        assert "[harness] breaker tripped: multi_edit dropped" in capsys.readouterr().out
        # second read announces nothing further (once per open)
        assert br._breaker_broken("multi_edit") is True
        assert capsys.readouterr().out.count("breaker tripped") == 0
        # clean execution resets and emits the reset line
        st._MALFORMED_TOOL_FAILURES["multi_edit"] = 3
        br._breaker_decay("multi_edit")
        assert "breaker reset: multi_edit re-armed" in capsys.readouterr().out
    finally:
        st._MALFORMED_TOOL_FAILURES.clear()
        st._MALFORMED_TOOL_BROKEN.clear()
        br._TRIP_ANNOUNCED.clear()
        br.rearm_run_tools_trip_watch()


def test_breaker_silent_when_off(capsys, monkeypatch):
    _off()
    from nx import _state as st
    from nx import _breaker as br
    monkeypatch.setattr("nx._logging.usage_breaker", lambda *a, **k: None)
    st._MALFORMED_TOOL_FAILURES.clear()
    st._MALFORMED_TOOL_BROKEN.clear()
    br._TRIP_ANNOUNCED.clear()
    br.rearm_run_tools_trip_watch()
    try:
        st._MALFORMED_TOOL_BROKEN.add("multi_edit")
        assert br._breaker_broken("multi_edit") is True
        assert "[harness]" not in capsys.readouterr().out
    finally:
        st._MALFORMED_TOOL_FAILURES.clear()
        st._MALFORMED_TOOL_BROKEN.clear()
        br._TRIP_ANNOUNCED.clear()
        br.rearm_run_tools_trip_watch()


def test_breadcrumbs_toggle_flips_state(capsys, monkeypatch, tmp_path):
    _on()
    # /breadcrumbs persists to ~/.env — point that write at a throwaway file
    # via the sanctioned NX_ENV_FILE override so the test never touches the
    # real one (cross-session tree-clean rule).
    monkeypatch.setenv("NX_ENV_FILE", str(tmp_path / "env"))
    rs._cmd_breadcrumbs("/breadcrumbs off", None)
    assert bc.ENABLED is False
    assert "breadcrumbs: off" in capsys.readouterr().out
    rs._cmd_breadcrumbs("/breadcrumbs on", None)
    assert bc.ENABLED is True
    assert "breadcrumbs: on" in capsys.readouterr().out
    # bare shows state
    rs._cmd_breadcrumbs("/breadcrumbs", None)
    assert "breadcrumbs: on" in capsys.readouterr().out
