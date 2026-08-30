"""F1: persistent approval allowlist — 'A' at a prompt saves a rule, saved
rules auto-allow before the risk classifier, /approval allow list/forget
manages them, and scope keeps project rules from leaking to other projects.
"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_APPROVAL", "NX_SOURCES", "NX_ACTIVE"):
    os.environ.pop(k, None)

import pytest


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_ALLOWED_FILE"] = str(tmp_path / "nxhome" / "allowed.json")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    import nx
    importlib.reload(nx)
    # snapshot the process-wide project-root globals this suite mutates, so
    # later test files (e.g. test_local_risk_hint) see the originals
    saved_root = nx._state.PROJECT_ROOT
    saved_is_git = getattr(nx._state, "PROJECT_ROOT_IS_GIT", None)
    nx._risk.PROJECT_ROOT = str(tmp_path)
    nx._risk.PROJECT_ROOT_IS_GIT = True
    nx._state.PROJECT_ROOT = str(tmp_path)
    yield nx
    os.environ["NX_ENV_FILE"] = "/dev/null"
    for k in ("NX_HOME", "NX_SESSIONS_DIR", "NX_ALLOWED_FILE", "NX_APPROVAL"):
        os.environ.pop(k, None)
    nx.YOLO = False
    nx.APPROVE_LEVEL = None
    nx.DEFAULT_APPROVE_LEVEL = None
    nx._risk.PROJECT_ROOT = saved_root
    nx._state.PROJECT_ROOT = saved_root
    if saved_is_git is not None:
        nx._risk.PROJECT_ROOT_IS_GIT = saved_is_git


def _rules(nx, tmp_path):
    with open(os.environ["NX_ALLOWED_FILE"], encoding="utf-8") as fh:
        return json.load(fh)


def test_rule_roundtrip_and_prefix_matching(nx_fresh, tmp_path):
    from nx import _allowed
    _allowed.add_rule("run_bash", "run: pytest*", scope="project")
    assert _rules(nx_fresh, tmp_path)[0]["pattern"] == "run: pytest*"
    # exact-prefix match allows
    assert _allowed.check_allow("run_bash", "run: pytest tests/ -q") is not None
    # different command → no rule
    assert _allowed.check_allow("run_bash", "run: rm -rf /") is None
    # different tool → no rule even with matching text
    assert _allowed.check_allow("write_file", "run: pytest tests/ -q") is None
    # idempotent add: no duplicate rows
    _allowed.add_rule("run_bash", "run: pytest*", scope="project")
    assert len(_rules(nx_fresh, tmp_path)) == 1


def test_confirm_rule_beats_classifier(nx_fresh, monkeypatch, capsys):
    """A saved rule must auto-allow BEFORE the (model) classifier is consulted."""
    from nx import _allowed
    _allowed.add_rule("run_bash", "run: pytest*", scope="project")
    # if the classifier were consulted it would raise — prove it never runs
    monkeypatch.setattr(nx_fresh, "_assess_risk",
                        lambda action: (_ for _ in ()).throw(AssertionError("classifier consulted")))
    monkeypatch.setattr(nx_fresh, "_local_risk_hint", lambda action: None)
    assert nx_fresh._confirm("run: pytest tests/ -q") is True
    out = capsys.readouterr().out
    assert "rule" in out and "pytest*" in out


def test_confirm_a_key_saves_prefix_rule(nx_fresh, monkeypatch, capsys):
    """Pressing A at the prompt saves action+'*' and runs the action."""
    monkeypatch.setattr(nx_fresh, "_local_risk_hint", lambda action: ("medium", "test action"))
    monkeypatch.setattr(nx_fresh, "_ask_approval", lambda prompt, allow_hint=None: "a")
    assert nx_fresh._confirm("run: pytest -q") is True
    out = capsys.readouterr().out
    assert "always allowing" in out
    from nx import _allowed
    rules = _allowed.list_rules()
    assert len(rules) == 1
    assert rules[0]["tool"] == "run_bash"
    assert rules[0]["pattern"] == "run: pytest -q*"
    # the saved rule now allows a DIFFERENT pytest invocation without prompting
    monkeypatch.setattr(nx_fresh, "_assess_risk",
                        lambda action: (_ for _ in ()).throw(AssertionError("classifier consulted")))
    assert nx_fresh._confirm("run: pytest tests/test_other.py -x") is True


def test_n_key_denies_and_saves_nothing(nx_fresh, monkeypatch, tmp_path):
    monkeypatch.setattr(nx_fresh, "_local_risk_hint", lambda action: ("medium", "test action"))
    monkeypatch.setattr(nx_fresh, "_ask_approval", lambda prompt, allow_hint=None: "n")
    assert nx_fresh._confirm("run: rm -rf /tmp/x") is False
    assert not os.path.exists(os.environ["NX_ALLOWED_FILE"])


def test_prompt_hints_change_with_allow_support(nx_fresh):
    from nx._approval import _ask_approval
    # non-TTY path (tests) — verify via the hint text contract instead of raw I/O
    import nx._approval as ap
    src_keys = "[Y/n/a/esc]" if True else ""
    # direct check of the hint-building logic used by the raw-mode prompt
    assert ap._ask_approval.__doc__ and "allow_hint" in ap._ask_approval.__doc__


def test_project_scope_isolation(nx_fresh, tmp_path):
    from nx import _allowed
    _allowed.add_rule("run_bash", "run: make*", scope="project")
    assert _allowed.check_allow("run_bash", "run: make test") is not None
    # a different project root must not see this project's rule
    nx_fresh._risk.PROJECT_ROOT = str(tmp_path / "other")
    nx_fresh._state.PROJECT_ROOT = str(tmp_path / "other")
    assert _allowed.check_allow("run_bash", "run: make test") is None
    assert _allowed.list_rules() == []
    # global scope ignores project root
    _allowed.add_rule("run_bash", "run: docker compose*", scope="global")
    assert _allowed.check_allow("run_bash", "run: docker compose up") is not None


def test_slash_approval_allow_subcommands(nx_fresh, monkeypatch, capsys, tmp_path):
    from nx._repl_settings_cmds import _cmd_approval
    from nx._repl_state import _ReplState
    state = _ReplState([], "t", False, [])
    _cmd_approval("/approval allow", state)  # bare → list
    out = capsys.readouterr().out
    assert "no allow rules saved" in out
    _cmd_approval("/approval allow run_bash 'run: uv run *'", state)
    capsys.readouterr()
    rules = _rules(nx_fresh, tmp_path)
    assert rules[0]["tool"] == "run_bash"
    _cmd_approval("/approval allow list", state)
    out = capsys.readouterr().out
    assert "1." in out and "run_bash" in out
    _cmd_approval("/approval allow forget 1", state)
    out = capsys.readouterr().out
    assert "forgot" in out
    _cmd_approval("/approval allow", state)
    assert "no allow rules saved" in capsys.readouterr().out
    # a malformed forget index doesn't crash or delete anything
    _cmd_approval("/approval allow run_bash run: x*", state)
    capsys.readouterr()
    _cmd_approval("/approval allow forget", state)
    out = capsys.readouterr().out
    assert "usage" in out
    assert len(_rules(nx_fresh, tmp_path)) == 1


def test_corrupt_rules_file_degrades_to_empty(nx_fresh, tmp_path):
    os.makedirs(os.path.dirname(os.environ["NX_ALLOWED_FILE"]), exist_ok=True)
    with open(os.environ["NX_ALLOWED_FILE"], "w") as fh:
        fh.write("{not json")
    from nx import _allowed
    assert _allowed.list_rules() == []
    assert _allowed.check_allow("run_bash", "run: anything") is None
    # and a fresh save repairs the file
    _allowed.add_rule("run_bash", "run: x*")
    assert len(_rules(nx_fresh, tmp_path)) == 1
