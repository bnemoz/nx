#!/usr/bin/env python3
"""Tests for the run_tests tool (0.49.0, ROADMAP P3-5).

Offline by construction: no test here executes a real test suite. The body's
subprocess call runs a stub `uv`/`pytest` script that emits canned runner
output, so the detection → confirm → execute → parse pipeline is exercised
end-to-end without uv, a venv, or the network.
"""
import os
import stat
import sys
import tempfile
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest  # noqa: E402

import nx as m  # noqa: E402
from nx import _testrunner as tr  # noqa: E402

_tmp = tempfile.mkdtemp(prefix="nx-runtests-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp


@pytest.fixture
def pyproj(tmp_path):
    """A minimal pytest project: marker file + one passing/one failing test."""
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.0.0'\n", encoding="utf-8")
    tdir = tmp_path / "tests"
    tdir.mkdir()
    (tdir / "test_demo.py").write_text(
        "def test_ok():\n    assert 1\n\ndef test_bad():\n    assert 0\n",
        encoding="utf-8")
    return tmp_path


@pytest.fixture
def stub_bin(tmp_path):
    """A directory for stub executables, made prependable to PATH."""
    d = tmp_path / "bin"
    d.mkdir()
    return d


def _write_stub(d, name, body):
    p = d / name
    p.write_text(body, encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def test_detect_pytest_from_markers(tmp_path):
    for marker in ("pytest.ini", "conftest.py", "tox.ini", "setup.cfg",
                   "pyproject.toml"):
        d = tmp_path / ("d_" + marker.replace(".", "_"))
        d.mkdir()
        (d / marker).write_text("", encoding="utf-8")
        assert tr._detect_runner(str(d)) == "pytest", marker


def test_detect_tests_dir_counts_as_pytest(tmp_path):
    (tmp_path / "tests").mkdir()
    assert tr._detect_runner(str(tmp_path)) == "pytest"


def test_detect_go_beats_package_json(tmp_path):
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    assert tr._detect_runner(str(tmp_path)) == "go"


def test_detect_cargo_npm(tmp_path):
    (tmp_path / "Cargo.toml").write_text("", encoding="utf-8")
    assert tr._detect_runner(str(tmp_path)) == "cargo"
    d2 = tmp_path / "n"
    d2.mkdir()
    (d2 / "package.json").write_text("{}", encoding="utf-8")
    assert tr._detect_runner(str(d2)) == "npm"


def test_detect_none_on_empty_dir(tmp_path):
    assert tr._detect_runner(str(tmp_path)) is None
    assert tr._detect_runner(str(tmp_path / "missing")) is None


# ---------------------------------------------------------------------------
# command shape
# ---------------------------------------------------------------------------
def test_command_shape_pytest_without_uv_uses_python_dash_m(monkeypatch):
    monkeypatch.setattr(tr.shutil, "which", lambda _: None)
    cmd = tr._test_command("pytest", "")
    assert cmd == [sys.executable, "-m", "pytest", "-q"]
    cmd2 = tr._test_command("pytest", "tests/test_demo.py")
    assert cmd2 == [sys.executable, "-m", "pytest", "-q", "tests/test_demo.py"]


def test_command_shape_pytest_uses_uv_path(monkeypatch, stub_bin):
    fake = _write_stub(stub_bin, "uv", "#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(tr.shutil, "which", lambda n: fake if n == "uv" else None)
    cmd = tr._test_command("pytest", "")
    assert cmd[:3] == [fake, "run", "pytest"]


def test_command_shape_go_cargo_npm():
    assert tr._test_command("go", "") == ["go", "test", "./..."]
    assert tr._test_command("cargo", "") == ["cargo", "test"]
    assert tr._test_command("npm", "")[1:] == ["test", "--", "--run"]


# ---------------------------------------------------------------------------
# summarize (pure parsing, canned runner output)
# ---------------------------------------------------------------------------
PYTEST_OUT = textwrap.dedent("""\
    _______________________________ test_bad ________________________________

    tests/test_demo.py:4: in test_bad
        assert 0
    E   assert 0
    ======================== short test summary info ========================
    FAILED tests/test_demo.py::test_bad - assert 0
    =========================== 1 failed in 0.01s ===========================
    """)


def test_summarize_pytest_failure_keeps_block_and_summary():
    s = tr._summarize("pytest", PYTEST_OUT, 1)
    assert "test_bad" in s
    assert "E   assert 0" in s
    assert "FAILED tests/test_demo.py::test_bad" in s
    assert "1 failed in 0.01s" in s


def test_summarize_pytest_pass_is_summary_only():
    out = ("tests/test_demo.py .                                                     "
           "[100%]\n========================= 1 passed in 0.01s =========================\n")
    s = tr._summarize("pytest", out, 0)
    assert "1 passed" in s
    assert "traceback lines folded" not in s


def test_summarize_pytest_q_pass_is_summary_only():
    # -q green output: the summary line has no '=' padding at all.
    out = "23 passed in 1.07s\n"
    assert tr._summarize("pytest", out, 0) == "23 passed in 1.07s"


def test_summarize_pytest_long_traceback_folds_middle():
    mid = "\n".join(f"    frame {i}" for i in range(60))
    out = ("_______________________________ test_x _______________________________\n"
           + mid + "\nE   boom\n"
           "=========================== 1 failed in 0.2s =========================\n")
    s = tr._summarize("pytest", out, 1)
    assert "traceback lines folded" in s
    assert "frame 10" not in s  # middle is gone
    assert "frame 2" in s       # head kept (header + first frames)
    assert "frame 59" in s      # tail kept — assertions live at the end
    assert len(s.splitlines()) < 30


def test_summarize_go_cargo():
    go = "--- FAIL: TestAdd (0.00s)\n    add_test.go:12: 1 != 2\nFAIL\n" \
         "FAIL\texample.com/x\t0.01s\n"
    assert "TestAdd" in tr._summarize("go", go, 1)
    cargo = ("failures:\n\n---- src/lib.rs stdout ----\npanicked at 'boom'\n\n"
             "test result: FAILED. 0 passed; 1 failed; 0 ignored\n")
    s = tr._summarize("cargo", cargo, 1)
    assert "test result: FAILED" in s
    assert "panicked" in s


# ---------------------------------------------------------------------------
# the body, end-to-end via a stub uv
# ---------------------------------------------------------------------------
def test_run_tests_success_body(monkeypatch, stub_bin, pyproj):
    fake = _write_stub(stub_bin, "uv", "#!/bin/sh\n"
                       "echo '========================= 1 passed in 0.01s ====='"
                       "\nexit 0\n")
    monkeypatch.setattr(tr.shutil, "which", lambda n: fake if n == "uv" else None)
    monkeypatch.setattr(m, "_confirm", lambda action, preview=None: True)
    res = m.run_tool("run_tests", {})
    assert res.startswith("[run_tests: pytest]")
    assert "All tests passed. (exit 0)" in res
    assert "read_file(" not in res  # nothing folded, nothing to log


def test_run_tests_failure_body_parses_and_offers_full_log(monkeypatch, stub_bin, pyproj):
    long_out = ("_______________________________ test_x _______________________________\n"
                + "\n".join(f"    frame {i}" for i in range(60))
                + "\nE   boom\nFAILED tests/test_demo.py::test_x - boom\n"
                "=========================== 1 failed in 0.2s =========================\n")
    fake = _write_stub(stub_bin, "uv", "#!/bin/sh\ncat <<'OUT'\n"
                       + long_out + "OUT\nexit 1\n")
    monkeypatch.setattr(tr.shutil, "which", lambda n: fake if n == "uv" else None)
    monkeypatch.setattr(m, "_confirm", lambda action, preview=None: True)
    monkeypatch.setattr(tr, "_LOG_MIN_LINES", 0)
    res = m.run_tool("run_tests", {})
    assert "FAILED tests/test_demo.py::test_x" in res
    assert "1 failed in 0.2s" in res
    assert "traceback lines folded" in res
    assert "read_file(" in res  # the raw output is one read_file away


def test_run_tests_denied_does_not_execute(monkeypatch, stub_bin, pyproj):
    marker = pyproj / "ran"
    fake = _write_stub(stub_bin, "uv", f"#!/bin/sh\ntouch {marker}\nexit 0\n")
    monkeypatch.setattr(tr.shutil, "which", lambda n: fake if n == "uv" else None)
    monkeypatch.setattr(m, "_confirm", lambda action, preview=None: False)
    res = m.run_tool("run_tests", {})
    assert res == "DENIED by user"
    assert not marker.exists()


def test_run_tests_action_string_hits_the_local_risk_fast_path(monkeypatch, stub_bin, pyproj):
    seen = {}
    def fake_confirm(action, preview=None):
        seen["action"] = action
        return False
    monkeypatch.setattr(m, "_confirm", fake_confirm)
    m.run_tool("run_tests", {})
    assert m._local_risk_hint(seen["action"]) == ("medium", "contained test run")
    assert seen["action"].startswith("run tests: ")


def test_run_tests_unknown_project_is_an_error(tmp_path):
    res = m.run_tool("run_tests", {"targets": str(tmp_path)})
    assert res.startswith("ERROR: run_tests found no test project")
    assert "run_bash" in res  # names the escape hatch


def test_run_tests_missing_binary_is_an_error(monkeypatch, pyproj):
    monkeypatch.setattr(tr.shutil, "which", lambda _: None)
    monkeypatch.setattr(m, "_confirm", lambda action, preview=None: True)
    # sys.executable always exists, so force the FileNotFoundError instead:
    def boom(*a, **k):
        raise FileNotFoundError("nope")
    monkeypatch.setattr(tr.subprocess, "run", boom)
    res = m.run_tool("run_tests", {})
    assert res.startswith("ERROR: run_tests:")
    assert "not found on PATH" in res


def test_run_tests_timeout(monkeypatch, stub_bin, pyproj):
    fake = _write_stub(stub_bin, "uv", "#!/bin/sh\nsleep 30\nexit 0\n")
    monkeypatch.setattr(tr.shutil, "which", lambda n: fake if n == "uv" else None)
    monkeypatch.setattr(m, "_confirm", lambda action, preview=None: True)
    monkeypatch.setenv("NX_TESTRUNNER_TIMEOUT", "1")
    res = m.run_tool("run_tests", {})
    assert "TIMED OUT after 1s" in res
    assert "run_bash" in res


def test_run_tests_reads_targets_from_args(monkeypatch, stub_bin, pyproj):
    fake = _write_stub(stub_bin, "uv", "#!/bin/sh\necho '=== 1 passed ==='\nexit 0\n")
    monkeypatch.setattr(tr.shutil, "which", lambda n: fake if n == "uv" else None)
    monkeypatch.setattr(m, "_confirm", lambda action, preview=None: True)
    res = m.run_tool("run_tests", {"targets": str(pyproj / "tests")})
    assert "All tests passed. (exit 0)" in res


# ---------------------------------------------------------------------------
# session-gate posture
# ---------------------------------------------------------------------------
def test_plan_mode_gates_run_tests(monkeypatch):
    monkeypatch.setattr(m.st, "_PLAN_ACTIVE", True)
    from nx._plan import check_gate
    assert check_gate("run_tests") is not None
    assert check_gate("read_file") is None


def test_run_tests_in_plan_gate_and_write_tier_sets():
    from nx._plan import PLAN_GATE_TOOLS
    from nx._subagent import _WRITE_TIER_TOOL_NAMES, _subagent_tools
    assert "run_tests" in PLAN_GATE_TOOLS
    assert "run_tests" in _WRITE_TIER_TOOL_NAMES
    # read-only agents: withheld from the schema list entirely
    m.NX_SUBAGENT_WRITE = 0
    try:
        assert "run_tests" not in [t["function"]["name"] for t in _subagent_tools()]
    finally:
        m.NX_SUBAGENT_WRITE = 0
    # write-tier agents: offered, and dispatching routes to the real body
    m.NX_SUBAGENT_WRITE = 1
    try:
        assert "run_tests" in [t["function"]["name"] for t in _subagent_tools()]
        from nx._subagent import _write_tier_dispatch
        assert _write_tier_dispatch().get("run_tests") is not None
    finally:
        m.NX_SUBAGENT_WRITE = 0


def test_read_only_subagent_refusal_names_the_gate():
    from nx._subagent import _SubagentContext, _SUBAGENT_TLS
    # A truthy read-only table (what spawn_agent actually installs) — an EMPTY
    # dict falls back to DISPATCH by design (_subagent_table() or DISPATCH).
    _SUBAGENT_TLS.ctx = _SubagentContext(table={"read_file": lambda **_: "x"})
    try:
        res = m.run_tool("run_tests", {})
        assert res.startswith("ERROR: run_tests is not available to the read-only sub-agent")
    finally:
        _SUBAGENT_TLS.ctx = None
