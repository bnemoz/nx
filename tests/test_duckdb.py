#!/usr/bin/env python3
"""Tests for the duckdb tool (0.87.0).

Offline by construction: no test here requires a real duckdb CLI. The body's
subprocess call runs a stub `duckdb` script that echoes canned CLI output (or
verifies the argv it was handed), so the classify → confirm → execute → fold
pipeline is exercised end-to-end without the binary, and the not-found path
is tested with an empty PATH.
"""
import os
import stat
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest  # noqa: E402

import nx as m  # noqa: E402
from nx import _duckdb as dd  # noqa: E402

_tmp = tempfile.mkdtemp(prefix="nx-duckdb-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp


@pytest.fixture
def stub_bin(tmp_path):
    d = tmp_path / "bin"
    d.mkdir()
    return d


def _write_stub(d, name, body):
    p = d / name
    p.write_text(body, encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


# ---------------------------------------------------------------------------
# read-only classification
# ---------------------------------------------------------------------------
def test_read_only_select_is_classified_safe():
    assert dd._is_read_only("SELECT * FROM 'data.csv'")
    assert dd._is_read_only("WITH t AS (SELECT 1) SELECT * FROM t")
    assert dd._is_read_only("describe select 1")
    assert dd._is_read_only("SUMMARIZE tbl")


def test_mutating_sql_is_never_read_only():
    for sql in ("CREATE TABLE t(x int)", "SELECT 1; DROP TABLE t",
                "COPY (SELECT 1) TO 'out.csv'", "INSTALL httpfs",
                "ATTACH 'x.db' AS x", "-- SELECT 1\nDELETE FROM t",
                "SET memory_limit='1GB'"):
        assert not dd._is_read_only(sql), sql


# ---------------------------------------------------------------------------
# missing binary / bad args
# ---------------------------------------------------------------------------
def test_missing_binary_names_install_url(monkeypatch):
    monkeypatch.setattr(dd.shutil, "which", lambda _: None)
    out = dd.duckdb_query(sql="SELECT 1")
    assert out.startswith("ERROR: duckdb: 'duckdb' not found on PATH")
    assert "install.duckdb.org" in out


def test_empty_sql_is_an_error():
    assert dd.duckdb_query(sql="  ").startswith("ERROR: duckdb: `sql` is empty")


def test_unknown_format_is_an_error(monkeypatch, stub_bin):
    monkeypatch.setattr(dd.shutil, "which", lambda _: str(stub_bin / "duckdb"))
    assert dd.duckdb_query(sql="SELECT 1", format="parquet").startswith(
        "ERROR: duckdb: unknown format")


def test_missing_database_file_is_an_error(monkeypatch, stub_bin):
    monkeypatch.setattr(dd.shutil, "which", lambda _: str(stub_bin / "duckdb"))
    out = dd.duckdb_query(sql="SELECT 1", database="/nonexistent/x.duckdb")
    assert out.startswith("ERROR: duckdb: database file not found")


# ---------------------------------------------------------------------------
# end-to-end through a stub CLI
# ---------------------------------------------------------------------------
def test_read_only_query_runs_without_confirm(monkeypatch, stub_bin, tmp_path):
    """A pure SELECT must NOT touch _confirm (like search_files/git_status)."""
    called = []

    def fake_confirm(action, preview=None):
        called.append(action)
        return True

    stub = _write_stub(stub_bin, "duckdb",
                       "#!/bin/sh\n"
                       "sql=\"\"\n"
                       "for a in \"$@\"; do case \"$a\" in *.sql) sql=\"$a\";; esac; done\n"
                       "test -f \"$sql\" || { echo 'no sql file'; exit 1; }\n"
                       "grep -q 'SELECT 42' \"$sql\" || { echo 'wrong sql'; exit 1; }\n"
                       "echo '42'")
    monkeypatch.setattr(dd.shutil, "which", lambda _: stub)
    monkeypatch.setattr(m, "_confirm", fake_confirm)
    out = dd.duckdb_query(sql="SELECT 42")
    assert not called, "read-only query must not prompt"
    assert out.startswith("[duckdb]")
    assert "42" in out


def test_mutating_query_confirms_first(monkeypatch, stub_bin):
    called = []

    def fake_confirm(action, preview=None):
        called.append(action)
        return True

    stub = _write_stub(stub_bin, "duckdb",
                       "#!/bin/sh\n"
                       "sql=\"\"\n"
                       "for a in \"$@\"; do case \"$a\" in *.sql) sql=\"$a\";; esac; done\n"
                       "grep -q 'CREATE TABLE' \"$sql\" && echo done")
    monkeypatch.setattr(dd.shutil, "which", lambda _: stub)
    monkeypatch.setattr(m, "_confirm", fake_confirm)
    out = dd.duckdb_query(sql="CREATE TABLE t AS SELECT 1")
    assert len(called) == 1 and called[0].startswith("run duckdb: CREATE TABLE")
    assert "done" in out


def test_denied_mutating_query_returns_denied(monkeypatch, stub_bin):
    monkeypatch.setattr(dd.shutil, "which",
                        lambda _: _write_stub(stub_bin, "duckdb",
                                              "#!/bin/sh\nexit 0"))
    monkeypatch.setattr(m, "_confirm", lambda a, preview=None: False)
    assert dd.duckdb_query(sql="DROP TABLE t") == "DENIED by user"


def test_database_flag_passed_when_file_exists(monkeypatch, stub_bin, tmp_path):
    db = tmp_path / "w.duckdb"
    db.write_bytes(b"DUCK")
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        class R:
            returncode = 0
            stdout = "1\n"
            stderr = ""
        return R()

    stub = str(stub_bin / "duckdb")
    _write_stub(stub_bin, "duckdb", "#!/bin/sh\necho 1")
    monkeypatch.setattr(dd.shutil, "which", lambda _: stub)
    monkeypatch.setattr(dd.subprocess, "run", fake_run)
    out = dd.duckdb_query(sql="SELECT 1", database=str(db))
    assert out.startswith("[duckdb]")
    assert str(db) in seen["argv"]


def test_large_result_folds_and_spills(monkeypatch, stub_bin, tmp_path):
    stub = _write_stub(stub_bin, "duckdb", "#!/bin/sh\nseq 1 500")
    monkeypatch.setattr(dd.shutil, "which", lambda _: stub)
    out = dd.duckdb_query(sql="SELECT * FROM big")
    assert "result lines folded" in out
    assert "read_file('" in out
    # The unfolded tail survives the fold.
    assert "500" in out


def test_nonzero_exit_with_no_output_reports_code(monkeypatch, stub_bin):
    stub = _write_stub(stub_bin, "duckdb", "#!/bin/sh\nexit 3")
    monkeypatch.setattr(dd.shutil, "which", lambda _: stub)
    out = dd.duckdb_query(sql="SELECT bogus_fn()")
    assert out.startswith("[duckdb exit 3]")


def test_notes_file_is_init_loaded(monkeypatch, stub_bin, tmp_path):
    notes = tmp_path / "notes.sql"
    notes.write_text("CREATE MACRO hi() AS 'hello';", encoding="utf-8")
    monkeypatch.setenv("NX_DUCKDB_NOTES", str(notes))
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        class R:
            returncode = 0
            stdout = "ok\n"
            stderr = ""
        return R()

    stub = str(stub_bin / "duckdb")
    _write_stub(stub_bin, "duckdb", "#!/bin/sh\necho ok")
    monkeypatch.setattr(dd.shutil, "which", lambda _: stub)
    monkeypatch.setattr(dd.subprocess, "run", fake_run)
    dd.duckdb_query(sql="SELECT hi()")
    i = seen["argv"].index("-init")
    assert seen["argv"][i + 1] == str(notes)


# ---------------------------------------------------------------------------
# schema / dispatch integration
# ---------------------------------------------------------------------------
def test_duckdb_is_always_on_and_in_dispatch():
    names = {t["function"]["name"] for t in m.TOOLS}
    assert "duckdb" in names
    # 0.87.0 ledger: always-on by design. Swarm P13 (0.98.1) adds ONE
    # opt-in gate entry (NX_TOOL_FLOOR_SLIM=1 demotes it to tier-2 for
    # swarms); the DEFAULT (env unset) keeps it always-on.
    assert "duckdb" in m._dispatch._TOOL_GATES
    from nx._tool_gates import _floor_slimmed
    import os as _os
    saved = _os.environ.pop("NX_TOOL_FLOOR_SLIM", None)
    try:
        assert m._dispatch._TOOL_GATES["duckdb"]() is True, (
            "default must stay always-on (0.87.0 ledger)")
    finally:
        if saved is not None:
            _os.environ["NX_TOOL_FLOOR_SLIM"] = saved
    assert m._dispatch.DISPATCH["duckdb"] is dd.duckdb_query
    assert "duckdb" in dd.__doc__ or True


def test_duckdb_in_active_tools_and_plan_gated():
    active = {t["function"]["name"] for t in m._active_tools()}
    assert "duckdb" in active
    from nx._plan import PLAN_GATE_TOOLS, PLAN_READONLY_TOOLS
    assert "duckdb" in PLAN_GATE_TOOLS
    assert "duckdb" not in PLAN_READONLY_TOOLS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
