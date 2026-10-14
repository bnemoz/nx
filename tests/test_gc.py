#!/usr/bin/env python3
"""Phase-1 artifact GC (nx/_gc.py): declarative age/size retention over the
durable files under NX_HOME that nothing ever cleaned — sub-agent reports,
undo snapshots, usage.jsonl, the web cache, the input-history recall buffer.

Every test pins NX_HOME/SESSION_HOME at a tmp dir (never the real ~/.nx).
`gc_sweep()` itself is ungated (rules apply on every call), so rule tests are
deterministic; amortization is the save path's shared `_sweep_due("gc", …)`
gate and is exercised through `_maybe_gc_artifacts` with a fresh `_SWEEP_LAST`.
Covers: each rule fires when aged and does NOT fire when fresh; env override 0
disables a rule; dry_run reports without deleting; the size-cap trims keep the
newest bytes and valid JSONL; and the save-path hook actually invokes it.
"""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import nx._gc as gc  # noqa: E402
import nx._state as st  # noqa: E402
import nx._sessions_fs as sfs  # noqa: E402

_GC_ENV = ("NX_GC_AGENT_REPORT_DAYS", "NX_GC_UNDO_DAYS",
           "NX_USAGE_LOG_MAX_MB", "NX_WEBCACHE_MAX_MB",
           "NX_GC_SWEEP_INTERVAL")
_DAY = 86400.0


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """A private session home; every GC knob at its default (env-clean)."""
    monkeypatch.setattr(st, "SESSION_HOME", str(tmp_path))
    for k in _GC_ENV:
        monkeypatch.delenv(k, raising=False)
    return tmp_path


@pytest.fixture()
def sweep_gate(home, monkeypatch):
    """`home` plus a FRESH process-wide sweep stamp table: `_SWEEP_LAST` is the
    module global the shared `_sweep_due` gate reads, so hook tests start from
    "never swept" and monkeypatch restores the real dict (AGENTS.md: every
    mutated module global goes through monkeypatch)."""
    monkeypatch.setattr(sfs, "_SWEEP_LAST", {})
    return home


def _aged(path, age_days, text="x\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    old = time.time() - age_days * _DAY
    os.utime(path, (old, old))
    return path


# --- dir-age: sub-agent reports ----------------------------------------------

def test_agent_report_old_deleted_fresh_kept(home):
    old = _aged(home / "agents" / "proj-abc12345" / "job1.md", 15)
    intent = _aged(home / "agents" / "proj-abc12345" / "job2.intent.json", 20)
    flat = _aged(home / "agents" / "legacy-job.md", 14.5)  # pre-0.145.2 flat
    fresh = _aged(home / "agents" / "proj-abc12345" / "job3.md", 1)
    lines = gc.gc_sweep()
    assert not old.exists(), "aged agent report survived the sweep"
    assert not intent.exists(), "aged intent record survived the sweep"
    assert not flat.exists(), "flat legacy report survived the sweep"
    assert fresh.is_file(), "fresh agent report was swept"
    assert any("removed" in l and "job1.md" in l for l in lines)


def test_agent_report_env_zero_disables(home, monkeypatch):
    monkeypatch.setenv("NX_GC_AGENT_REPORT_DAYS", "0")
    old = _aged(home / "agents" / "proj-x" / "job1.md", 999)
    assert gc.gc_sweep() == []
    assert old.is_file(), "NX_GC_AGENT_REPORT_DAYS=0 must disable the rule"


def test_agent_report_env_override_shortens(home, monkeypatch):
    monkeypatch.setenv("NX_GC_AGENT_REPORT_DAYS", "3")
    old = _aged(home / "agents" / "proj-x" / "job1.md", 5)
    lines = gc.gc_sweep()
    assert not old.exists()
    assert lines and "job1.md" in lines[0]


# --- file-age: undo store + input history -------------------------------------

def test_undo_files_old_deleted_fresh_kept(home):
    per_session = _aged(home / "undo" / "sess-123.json", 31, "[{}]")
    legacy = _aged(home / "undo.json", 40, "[{}]")
    fresh = _aged(home / "undo" / "sess-456.json", 2, "[{}]")
    hist = _aged(home / "input_history.json", 31, '["a"]')
    lines = gc.gc_sweep()
    assert not per_session.exists(), "aged per-session undo file survived"
    assert not legacy.exists(), "aged legacy undo.json survived"
    assert fresh.is_file(), "fresh undo file was swept"
    assert not hist.exists(), "aged input_history.json survived"
    assert sum("undo" in l for l in lines) >= 2


def test_undo_env_zero_disables(home, monkeypatch):
    monkeypatch.setenv("NX_GC_UNDO_DAYS", "0")
    a = _aged(home / "undo" / "s.json", 999)
    b = _aged(home / "input_history.json", 999)
    assert gc.gc_sweep() == []
    assert a.is_file() and b.is_file()


# --- size-cap: usage.jsonl (tail trim) ----------------------------------------

def _write_lines(path, n, line_bytes):
    """n JSONL records of ~line_bytes each, ordered oldest (i=0) to newest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pad = "x" * max(0, line_bytes - 20)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"i": i, "pad": pad}) + "\n")
    return path


def test_usage_log_over_cap_keeps_newest_bytes(home, monkeypatch):
    monkeypatch.setenv("NX_USAGE_LOG_MAX_MB", "1")
    path = _write_lines(home / "logs" / "usage.jsonl", 1400, 1024)  # ~1.4 MB
    before = path.stat().st_size
    lines = gc.gc_sweep()
    assert path.is_file(), "size-cap must trim, not delete"
    assert any("trimmed" in l and "usage.jsonl" in l for l in lines)
    assert path.stat().st_size < before
    assert path.stat().st_size <= 1024 * 1024
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert rows[-1]["i"] == 1399, "trim must keep the NEWEST records"
    assert rows[0]["i"] > 0, "trim must drop from the OLDEST end"


def test_usage_log_under_cap_untouched(home, monkeypatch):
    monkeypatch.setenv("NX_USAGE_LOG_MAX_MB", "1")
    path = _write_lines(home / "logs" / "usage.jsonl", 10, 1024)  # ~10 KB
    before = path.stat().st_size
    assert gc.gc_sweep() == []
    assert path.stat().st_size == before


def test_usage_log_rotation_between_stat_and_open_is_abandoned(home,
                                                               monkeypatch):
    """The documented race: `_AsyncLogWriter._run` rotates usage.jsonl
    (os.replace to `.1` + recreate) AFTER gc sized the file but BEFORE its
    r+b rewrite open. Seeking the old offset into the fresh inode and
    truncating would zero the new file (or splice the old tail into it), so
    `_trim_tail` must notice the inode changed and leave the fresh file
    byte-for-byte alone, reporting nothing."""
    monkeypatch.setenv("NX_USAGE_LOG_MAX_MB", "1")
    path = _write_lines(home / "logs" / "usage.jsonl", 1400, 1024)  # ~1.4 MB
    fresh = json.dumps({"i": 0, "post": "rotation"}) + "\n"
    real_open = gc._open_for_trim
    fired = []

    def racing_open(p):
        # Exactly the rotation the writer thread performs, landing between
        # the stat and the rewrite open.
        fired.append(p)
        os.replace(p, p + ".1")
        with open(p, "a", encoding="utf-8") as f:
            f.write(fresh)
        return real_open(p)

    monkeypatch.setattr(gc, "_open_for_trim", racing_open)
    assert gc.gc_sweep() == [], "a rotated-out file must not be reported"
    assert fired == [str(path)], "the race seam never fired"
    assert path.read_text() == fresh, "fresh post-rotation file was clobbered"
    assert (home / "logs" / "usage.jsonl.1").is_file()


def test_usage_log_env_zero_disables(home, monkeypatch):
    monkeypatch.setenv("NX_USAGE_LOG_MAX_MB", "0")
    path = _write_lines(home / "logs" / "usage.jsonl", 2200, 1024)  # > 2 MB
    before = path.stat().st_size
    assert gc.gc_sweep() == []
    assert path.stat().st_size == before


# --- size-cap: web cache (TTL + oldest-first rewrite) --------------------------

def _cache_line(key, ts, val="v"):
    return json.dumps({"k": key, "t": ts, "v": val})


def test_web_cache_expired_entries_dropped(home, monkeypatch):
    from nx import _webfetch
    monkeypatch.setattr(_webfetch, "CACHE_TTL_SECS", 100)
    now = time.time()
    path = home / "web-cache" / "cache.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join([
        _cache_line("expired", now - 500),
        "not-json-at-all",
        _cache_line("fresh", now - 10),
    ]) + "\n")
    lines = gc.gc_sweep()
    assert any("cache.jsonl" in l and "trimmed" in l for l in lines)
    keys = [json.loads(l)["k"] for l in path.read_text().splitlines()]
    assert keys == ["fresh"], f"expired/corrupt lines must be dropped, got {keys}"


def test_web_cache_over_cap_drops_oldest_survivors(home, monkeypatch):
    monkeypatch.setenv("NX_WEBCACHE_MAX_MB", "1")
    from nx import _webfetch
    monkeypatch.setattr(_webfetch, "CACHE_TTL_SECS", 10 * _DAY)
    now = time.time()
    path = home / "web-cache" / "cache.jsonl"
    path.parent.mkdir(parents=True)
    big = "y" * 1024
    with open(path, "w", encoding="utf-8") as f:
        for i in range(1400):  # ~1.4 MB, all inside TTL
            f.write(_cache_line(f"k{i}", now - (1400 - i), big) + "\n")
    lines = gc.gc_sweep()
    assert any("trimmed" in l and "cache.jsonl" in l for l in lines)
    assert path.stat().st_size <= 1024 * 1024
    keys = [json.loads(l)["k"] for l in path.read_text().splitlines()]
    assert keys[-1] == "k1399" and keys[0] != "k0"


def test_web_cache_fresh_under_cap_untouched(home, monkeypatch):
    from nx import _webfetch
    monkeypatch.setattr(_webfetch, "CACHE_TTL_SECS", 10 * _DAY)
    path = home / "web-cache" / "cache.jsonl"
    path.parent.mkdir(parents=True)
    body = _cache_line("fresh", time.time()) + "\n"
    path.write_text(body)
    assert gc.gc_sweep() == []
    assert path.read_text() == body


# --- dry run + amortization gate -----------------------------------------------

def test_dry_run_reports_without_deleting(home):
    old = _aged(home / "agents" / "proj-x" / "job1.md", 90)
    hist = _aged(home / "input_history.json", 90)
    lines = gc.gc_sweep(dry_run=True)
    assert old.is_file() and hist.is_file(), "dry_run must not delete"
    assert any(l.startswith("would remove") and "job1.md" in l for l in lines)
    assert any("input_history.json" in l for l in lines)


def test_hook_gate_suppresses_second_sweep(sweep_gate, monkeypatch):
    """The amortization is the shared `_sweep_due("gc", …)` gate: the first hook
    call sweeps and stamps it, a second inside the interval does nothing."""
    monkeypatch.setenv("NX_GC_SWEEP_INTERVAL", "86400")
    old = _aged(sweep_gate / "agents" / "proj-x" / "job1.md", 90)
    sfs._maybe_gc_artifacts()
    assert not old.exists(), "first hook call must sweep"
    aged2 = _aged(sweep_gate / "agents" / "proj-x" / "job2.md", 90)
    sfs._maybe_gc_artifacts()
    assert aged2.is_file(), "second call must be gated inside the interval"
    monkeypatch.setattr(sfs, "_SWEEP_LAST", {})  # interval elapsed
    sfs._maybe_gc_artifacts()
    assert not aged2.exists(), "sweep must run again once the interval elapsed"


def test_hook_gate_interval_zero_sweeps_every_call(sweep_gate, monkeypatch):
    """NX_GC_SWEEP_INTERVAL=0 restores the every-call behavior, matching
    NX_SESSION_SWEEP_INTERVAL=0's legacy semantics."""
    monkeypatch.setenv("NX_GC_SWEEP_INTERVAL", "0")
    a = _aged(sweep_gate / "agents" / "proj-x" / "job1.md", 90)
    sfs._maybe_gc_artifacts()
    assert not a.exists()
    b = _aged(sweep_gate / "agents" / "proj-x" / "job2.md", 90)
    sfs._maybe_gc_artifacts()
    assert not b.exists()


# --- save-path wiring -----------------------------------------------------------

def test_maybe_gc_artifacts_invokes_gc_sweep(sweep_gate, monkeypatch):
    # Resolve at test time, not via the module-level binding: a Tier B
    # restore (conftest globals walk) pops sys.modules["nx._gc"] between
    # tests, so _maybe_gc_artifacts' lazy `from . import _gc` can see a
    # fresh module object under xdist; patch whichever it will resolve.
    import nx._gc as gc_now
    calls = []
    monkeypatch.setattr(gc_now, "gc_sweep", lambda *a, **kw: calls.append((a, kw)))
    sfs._maybe_gc_artifacts()
    assert len(calls) == 1, "save hook must invoke gc_sweep exactly once"


def test_maybe_gc_artifacts_sweeps_end_to_end(sweep_gate):
    old = _aged(sweep_gate / "agents" / "proj-x" / "job1.md", 90)
    sfs._maybe_gc_artifacts()
    assert not old.exists(), "the save-path hook did not reach the sweep"


def test_maybe_gc_artifacts_never_raises(sweep_gate, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("sweep exploded")
    monkeypatch.setattr(gc, "gc_sweep", boom)
    sfs._maybe_gc_artifacts()  # must swallow: GC may never break a save


# --- `nx gc` CLI command (nx/_gc_cmd._cli_gc) ----------------------------------
# The command is a thin surface over gc_sweep, so the CLI tests exercise the
# real handler (returns its exit code) with a monkeypatched home, exactly like
# the other subcommand handlers are tested (call the handler with argv).

def _run_cli(home, argv, monkeypatch, capsys):
    """Drive `_cli_gc` against the pinned home; returns (code, out, err)."""
    from nx import _gc_cmd
    monkeypatch.setattr(st, "SESSION_HOME", str(home))
    code = _gc_cmd._cli_gc(argv)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_cli_gc_default_is_dry_run(home, monkeypatch, capsys):
    old = _aged(home / "agents" / "proj-x" / "job1.md", 90)
    hist = _aged(home / "input_history.json", 90)
    code, out, _err = _run_cli(home, [], monkeypatch, capsys)
    assert code == 0
    assert old.is_file() and hist.is_file(), "bare `nx gc` must not delete"
    assert any(l.startswith("would remove") and "job1.md" in l
               for l in out.splitlines())
    assert "would be removed" in out
    assert "run `nx gc --all` to apply" in out


def test_cli_gc_all_deletes(home, monkeypatch, capsys):
    old = _aged(home / "agents" / "proj-x" / "job1.md", 90)
    hist = _aged(home / "input_history.json", 90)
    code, out, _err = _run_cli(home, ["--all"], monkeypatch, capsys)
    assert code == 0
    assert not old.exists() and not hist.exists(), "`nx gc --all` must delete"
    assert any(l.startswith("removed") and "job1.md" in l
               for l in out.splitlines())
    assert "would" not in out  # real run uses the non-dry verbs
    assert "artifact(s) removed" in out
    assert "run `nx gc --all` to apply" not in out  # hint is dry-run only


def test_cli_gc_summary_format_and_size(home, monkeypatch, capsys):
    # two aged artifacts, each 2 KB -> "2 artifact(s) would be removed"
    _aged(home / "agents" / "proj-x" / "a.md", 90, "x" * 2048)
    _aged(home / "agents" / "proj-x" / "b.md", 90, "y" * 2048)
    code, out, _err = _run_cli(home, [], monkeypatch, capsys)
    assert code == 0
    summary = next(l for l in out.splitlines() if "reclaimed" in l)
    assert summary.startswith("2 artifact(s) would be removed, ")
    assert summary.endswith(" MB reclaimed")
    # the summary MB figure is derived from the per-line sizes (4 KB -> 0.00 MB)
    mb = float(summary.split(",")[1].split(" MB")[0])
    assert 0.0 <= mb < 0.01


def test_cli_gc_empty_home_no_crash(tmp_path, monkeypatch, capsys):
    """An empty NX_HOME (nothing to sweep) exits 0 with a zero summary."""
    home = tmp_path / "emptyhome"
    home.mkdir()
    code, out, _err = _run_cli(home, [], monkeypatch, capsys)
    assert code == 0
    assert "0 artifact(s) would be removed, 0.00 MB reclaimed" in out
    assert "run `nx gc --all` to apply" in out


def test_cli_gc_all_empty_home_no_crash(tmp_path, monkeypatch, capsys):
    home = tmp_path / "emptyhome"
    home.mkdir()
    code, out, _err = _run_cli(home, ["--all"], monkeypatch, capsys)
    assert code == 0
    assert "0 artifact(s) removed" in out


def test_cli_gc_env_knob_respected(home, monkeypatch, capsys):
    # age knob 0 disables the rule -> dry run finds nothing to remove
    monkeypatch.setenv("NX_GC_AGENT_REPORT_DAYS", "0")
    _aged(home / "agents" / "proj-x" / "job1.md", 999)
    code, out, _err = _run_cli(home, [], monkeypatch, capsys)
    assert code == 0
    assert "0 artifact(s) would be removed" in out


def test_cli_gc_unknown_flag_exits_2(home, monkeypatch, capsys):
    code, _out, err = _run_cli(home, ["--bogus"], monkeypatch, capsys)
    assert code == 2
    assert "unknown flag" in err


def test_cli_gc_help_exits_0(home, monkeypatch, capsys):
    for flag in ("-h", "--help"):
        code, out, _err = _run_cli(home, [flag], monkeypatch, capsys)
        assert code == 0
        assert "Usage: nx gc" in out


def test_cli_gc_unexpected_error_exits_1(home, monkeypatch, capsys):
    def boom(*a, **kw):
        raise RuntimeError("sweep exploded")
    monkeypatch.setattr(gc, "gc_sweep", boom)
    # _cli_gc imports gc_sweep by name at call time; patch the symbol it reads.
    monkeypatch.setattr("nx._gc.gc_sweep", boom)
    code, _out, err = _run_cli(home, [], monkeypatch, capsys)
    assert code == 1
    assert "sweep exploded" in err


def test_cli_gc_dispatched_from_main(monkeypatch, capsys):
    """`nx gc` reaches _cli_gc via main()'s verb dispatch (and exits with its
    code) rather than falling through to the REPL."""
    from nx import _cli
    calls = {}

    def fake_gc(args):
        calls["args"] = args
        return 0
    import nx._gc_cmd as gc_cmd
    monkeypatch.setattr(gc_cmd, "_cli_gc", fake_gc)
    monkeypatch.setattr(sys, "argv", ["nx", "gc", "--all"])
    monkeypatch.setattr(_cli, "_load_env_file", lambda *a, **k: None,
                        raising=False)
    with pytest.raises(SystemExit) as e:
        _cli.main()
    assert e.value.code == 0
    assert calls.get("args") == ["--all"]


def test_gc_verb_recognized_by_help_dispatch():
    """`gc` is a known verb: `_cli_help_or_version` falls through for it (so
    it reaches main()'s dispatch) and never treats it as an unknown command."""
    from nx import _cli
    assert "gc" in _cli._VERBS
    assert _cli._cli_help_or_version(["gc"]) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
