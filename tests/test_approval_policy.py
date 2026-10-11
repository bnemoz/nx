"""C4 (0.135.1): NX_APPROVAL_POLICY=headless-allow:<glob> — headless approval
policy pre-check.

On headless surfaces (Telegram gateway, async sub-agent bash, `nx run` batch)
`_approval._confirm` auto-DENIES gated calls unless YOLO is on (the
`_subagent_quiet()` branch). The new policy env var lets an operator
pre-approve TOOL names matching a glob on those surfaces; everything else —
and every other env value — keeps the deny.

Run: uv run pytest tests/test_approval_policy.py -q  (also standalone)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for _k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR",
           "NX_APPROVAL_POLICY"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NX_APPROVAL_POLICY", raising=False)
    import nx
    yield nx, tmp_path
    nx._state._TOOL_RESULT_CACHE.clear()
    for _k in ("NX_HOME", "NX_SESSIONS_DIR"):
        os.environ.pop(_k, None)


# --- the policy module in isolation ------------------------------------------

def test_unset_policy_denies(monkeypatch):
    from nx import _approval_policy as ap
    monkeypatch.delenv(ap.ENV_VAR, raising=False)
    assert ap.headless_allow("bash git status") is False


def test_exact_tool_name_is_allowed(monkeypatch):
    from nx import _approval_policy as ap
    monkeypatch.setenv(ap.ENV_VAR, "headless-allow:edit_file")
    assert ap.headless_allow("edit_file /tmp/a.py") is True


def test_glob_matches_tool_name_only(monkeypatch):
    from nx import _approval_policy as ap
    monkeypatch.setenv(ap.ENV_VAR, "headless-allow:bash*")
    assert ap.headless_allow("bash git status") is True
    # the match is on the FIRST word, never the whole action
    assert ap.headless_allow("edit_file bash git") is False


def test_non_matching_glob_denies(monkeypatch):
    from nx import _approval_policy as ap
    monkeypatch.setenv(ap.ENV_VAR, "headless-allow:edit_file")
    assert ap.headless_allow("bash rm -rf /") is False


def test_comma_separated_globs(monkeypatch):
    from nx import _approval_policy as ap
    monkeypatch.setenv(ap.ENV_VAR, "headless-allow: bash* , edit_file")
    assert ap.headless_allow("bash ls") is True
    assert ap.headless_allow("edit_file x.py") is True
    assert ap.headless_allow("git_commit m") is False


def test_empty_glob_list_denies(monkeypatch):
    from nx import _approval_policy as ap
    monkeypatch.setenv(ap.ENV_VAR, "headless-allow:")
    assert ap.headless_allow("bash ls") is False


def test_other_or_malformed_values_denies(monkeypatch):
    from nx import _approval_policy as ap
    for val in ("yolo", "allow-all:bash", "headless-allow-all:bash", "", "  "):
        monkeypatch.setenv(ap.ENV_VAR, val)
        assert ap.headless_allow("bash ls") is False, f"policy={val!r}"


# --- wired into the headless deny branch of _confirm -------------------------

def test_quiet_confirm_deny_by_default(nx_fresh):
    nx, _ = nx_fresh
    nx._subagent._ASYNC_QUIET_SLOTS.quiet = True
    nx._state.YOLO = False
    try:
        assert nx._approval._confirm("bash echo hi") is False
    finally:
        nx._subagent._ASYNC_QUIET_SLOTS.quiet = False
        nx._state.YOLO = False


def test_quiet_confirm_policy_allows_matching_tool(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    monkeypatch.setenv("NX_APPROVAL_POLICY", "headless-allow:echo_tool")
    nx._subagent._ASYNC_QUIET_SLOTS.quiet = True
    nx._state.YOLO = False
    try:
        assert nx._approval._confirm("echo_tool hello") is True
        assert nx._approval._confirm("bash echo hi") is False
    finally:
        nx._subagent._ASYNC_QUIET_SLOTS.quiet = False
        nx._state.YOLO = False


def test_yolo_still_wins_on_headless(nx_fresh):
    nx, _ = nx_fresh
    nx._subagent._ASYNC_QUIET_SLOTS.quiet = True
    nx._state.YOLO = True
    try:
        assert nx._approval._confirm("bash echo hi") is True
    finally:
        nx._subagent._ASYNC_QUIET_SLOTS.quiet = False
        nx._state.YOLO = False


def test_interactive_confirm_ignores_the_policy(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    monkeypatch.setenv("NX_APPROVAL_POLICY", "headless-allow:bash")
    # not quiet -> the normal flow; stub the risk assessment and the prompt so
    # no model round-trip / keypress is needed and the prompt is what decides.
    import nx as _m
    monkeypatch.setattr(_m, "_assess_risk", lambda action: ("high", "stub"))
    monkeypatch.setattr(_m, "_ask_approval", lambda *a, **k: "n")
    nx._state.YOLO = False
    nx._state.APPROVE_LEVEL = None
    assert nx._approval._confirm("bash echo hi") is False  # prompt said n
