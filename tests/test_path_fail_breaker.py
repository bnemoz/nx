#!/usr/bin/env python3
"""Same-failing-path loop breaker (0.129.7): run_tool appends a cwd-carrying
nudge to the error result on the 2nd+ identical existence-type failure, never
on transient failures (claims, denials), and any other outcome resets the run.
"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    for k in [k for k in os.environ if k.startswith("NX_")]:
        os.environ.pop(k, None)
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = "/dev/null"
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    from nx import _state as st
    st._PATH_FAIL_LAST = None
    st._PATH_FAIL_RUN = 0
    yield nx, tmp_path


def test_second_identical_missing_path_gets_nudge(nx_fresh, capsys):
    nx, tmp = nx_fresh
    from nx._dispatch import run_tool
    p = str(tmp / "nope.txt")
    first = run_tool("read_file", {"path": p})
    capsys.readouterr()
    second = run_tool("read_file", {"path": p})
    capsys.readouterr()
    assert first.startswith("ERROR")
    assert "has failed" not in first
    assert "has failed" in second
    assert "failed 2 times" in second
    assert os.getcwd() in second  # the cwd rides the nudge
    assert "list_dir" in second


def test_third_failure_escalates_count(nx_fresh, capsys):
    nx, tmp = nx_fresh
    from nx._dispatch import run_tool
    p = str(tmp / "nope.txt")
    for _ in range(3):
        run_tool("read_file", {"path": p})
        capsys.readouterr()
    out = run_tool("read_file", {"path": p})
    capsys.readouterr()
    assert "failed 4 times" in out


def test_different_path_restarts_run(nx_fresh, capsys):
    nx, tmp = nx_fresh
    from nx._dispatch import run_tool
    run_tool("read_file", {"path": str(tmp / "a.txt")})
    capsys.readouterr()
    second = run_tool("read_file", {"path": str(tmp / "b.txt")})
    capsys.readouterr()
    # a NEW path is failure #1 for that key — no nudge
    assert "Runtime note" not in second


def test_success_resets_the_run(nx_fresh, capsys):
    nx, tmp = nx_fresh
    from nx._dispatch import run_tool
    bad = str(tmp / "nope.txt")
    run_tool("read_file", {"path": bad})
    capsys.readouterr()
    run_tool("read_file", {"path": str(tmp / "README.md")})  # ok (missing → error but different)
    capsys.readouterr()
    # write a real file and read it to reset via success
    (tmp / "real.txt").write_text("hi")
    run_tool("read_file", {"path": str(tmp / "real.txt")})
    capsys.readouterr()
    again = run_tool("read_file", {"path": bad})
    capsys.readouterr()
    assert "Runtime note" not in again


def test_claim_refusal_never_nudges(nx_fresh, capsys, monkeypatch):
    """A file-claim refusal is transient — retrying is correct, so the loop
    breaker must stay silent (and NOT consume the run counter)."""
    nx, tmp = nx_fresh
    from nx._dispatch import run_tool
    import nx._edit_tools as et

    def _fake_claim_error(path):
        return ("ERROR: claimed — refusing; retry is the correct move")

    monkeypatch.setattr(et, "_claim_error", _fake_claim_error)
    p = str(tmp / "claimed.txt")
    (tmp / "claimed.txt").write_text("x")
    out1 = run_tool("edit_file", {"path": p, "old": "x", "new": "y"})
    capsys.readouterr()
    out2 = run_tool("edit_file", {"path": p, "old": "x", "new": "y"})
    capsys.readouterr()
    assert out1.startswith("ERROR") and out2.startswith("ERROR")
    assert "Runtime note" not in out2


def test_is_a_directory_failure_nudges_on_repeat(nx_fresh, capsys):
    nx, tmp = nx_fresh
    from nx._dispatch import run_tool
    d = str(tmp)  # a directory, not a file
    run_tool("read_file", {"path": d})
    capsys.readouterr()
    second = run_tool("read_file", {"path": d})
    capsys.readouterr()
    assert "has failed" in second


def test_state_resets_between_tools(nx_fresh, capsys):
    """read_file failing then search_files failing the same path: different
    tool keys — the search failure is its own run #1."""
    nx, tmp = nx_fresh
    from nx._dispatch import run_tool
    d = str(tmp)
    run_tool("read_file", {"path": str(tmp / "nope.txt")})
    capsys.readouterr()
    out = run_tool("search_files", {"path": str(tmp / "elsewhere"), "pattern": "x"})
    capsys.readouterr()
    assert "Runtime note" not in out
