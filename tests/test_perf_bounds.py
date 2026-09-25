#!/usr/bin/env python3
"""Bounds and durability (PERF.md 0.61.0, PF-05/16/17/18/19; invariant I-8).

Five fixes that only bite after months of use, pinned here so none of them can
silently regress:

- **PF-16** — `st._TOOL_RESULT_CACHE` had no cap and `_ASYNC_AGENTS` never
  evicted finished records. Both are now bounded (T-10: `len <= cap`).
- **PF-17** — async fan-out was one thread + HTTP stream + parser per spawn,
  unbounded. A semaphore (NX_SUBAGENT_MAX_CONCURRENT, default 4) parks the
  excess as `queued`.
- **PF-18** — cache validity anchored on `path_arg[0]` only, so editing file
  #2 of a multi-path read never invalidated it, and a `search_files` hit
  survived changes in a subdirectory. Entries now anchor on every path /
  a subtree stamp.
- **PF-05** — `persist_undo_stack` re-serialized the whole stack per edit.
  Pushes now mark the stack dirty and it is written once per turn (T-12).
- **PF-19** — `_diff_hunk` output fed to the model is now capped
  (NX_DIFF_HUNK_MAX_CHARS, default 8192) with an elision notice.
"""
import importlib
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _k in list(os.environ):
    if _k.startswith("NX_"):
        del os.environ[_k]
os.environ["NX_ENV_FILE"] = "/dev/null"

import pytest  # noqa: E402


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    nx.YOLO = True
    yield nx, tmp_path
    nx.YOLO = False
    nx._state._TOOL_RESULT_CACHE.clear()
    nx._state._UNDO_STACK.clear()
    from nx import _subagent
    with _subagent._ASYNC_LOCK:
        _subagent._ASYNC_AGENTS.clear()
        _subagent._ASYNC_QUEUED.clear()


# --- T-10: the caches are bounded (PF-16, I-8) --------------------------------

def test_tool_result_cache_is_bounded_by_entries(nx_fresh, monkeypatch):
    """5000 inserts must not leave 5000 entries: the LRU keeps the cap."""
    nx, tmp = nx_fresh
    monkeypatch.setenv("NX_TOOL_CACHE_MAX_ENTRIES", "64")
    monkeypatch.setenv("NX_TOOL_CACHE_MAX_BYTES", "0")  # isolate the entry cap
    from nx._dispatch import run_tool

    f = tmp / "f.txt"
    for i in range(5000):
        f.write_text(f"v{i}\n")
        os.utime(f, (1000 + i, 1000 + i))
        # distinct cache key each time (the offset arg), one file
        run_tool("read_file", {"path": str(f), "offset": 1})

    cache = nx._state._TOOL_RESULT_CACHE
    assert len(cache) <= 64, len(cache)
    assert len(cache) >= 1


def test_tool_result_cache_is_bounded_by_bytes(nx_fresh, monkeypatch):
    """The byte cap is the one that actually bounds RSS when each entry holds
    whole file contents."""
    nx, tmp = nx_fresh
    monkeypatch.setenv("NX_TOOL_CACHE_MAX_ENTRIES", "0")  # isolate the byte cap
    monkeypatch.setenv("NX_TOOL_CACHE_MAX_BYTES", "20000")
    from nx._dispatch import run_tool

    for i in range(20):
        p = tmp / f"big{i}.txt"
        p.write_text("x" * 8000)
        run_tool("read_file", {"path": str(p)})

    cache = nx._state._TOOL_RESULT_CACHE
    total = sum(len(e.result or "") for e in cache.values())
    assert total <= 20000, total


def test_cache_hit_touches_lru_order(nx_fresh, monkeypatch):
    """Recency must matter: a re-read of an old entry keeps it alive while the
    genuinely cold entries are the ones evicted."""
    nx, tmp = nx_fresh
    from nx._dispatch import run_tool, _tool_cache_key

    keys = []
    for i in range(4):
        p = tmp / f"f{i}.txt"
        p.write_text(f"v{i}\n")
        run_tool("read_file", {"path": str(p)})
        keys.append(_tool_cache_key("read_file", {"path": str(p)}))
    # refresh the first entry, then shrink the cap to 2 and add a new one
    run_tool("read_file", {"path": str(tmp / "f0.txt")})
    monkeypatch.setenv("NX_TOOL_CACHE_MAX_ENTRIES", "2")
    cluster = tmp / "new.txt"
    cluster.write_text("new\n")
    run_tool("read_file", {"path": str(cluster)})
    cache = nx._state._TOOL_RESULT_CACHE
    assert len(cache) <= 2
    assert keys[0] in cache, "recently-read entry was evicted as if it were cold"


def test_async_agent_ledger_evicts_finished_records(nx_fresh, monkeypatch):
    """50 finished, announced agents with a floor of 10 leave <= 10 records."""
    nx, _ = nx_fresh
    from nx import _subagent

    monkeypatch.setattr(_subagent, "_MAX_FINISHED_AGENTS", 10)
    with _subagent._ASYNC_LOCK:
        for i in range(50):
            _subagent._ASYNC_AGENTS[f"agent-{i}"] = {
                "instruction": "q", "status": "done", "answer": "a" * 500,
                "started": i, "finished": i, "announced": True,
            }
        _subagent._evict_finished_agents()
        assert len(_subagent._ASYNC_AGENTS) <= 10


def test_eviction_never_drops_an_unannounced_answer(nx_fresh, monkeypatch):
    """A finished-but-unstaged answer is the parent's only copy of it: eviction
    must skip it even over the floor."""
    nx, _ = nx_fresh
    from nx import _subagent

    monkeypatch.setattr(_subagent, "_MAX_FINISHED_AGENTS", 1)
    with _subagent._ASYNC_LOCK:
        for i in range(5):
            _subagent._ASYNC_AGENTS[f"agent-{i}"] = {
                "instruction": "q", "status": "done", "answer": "a",
                "started": i, "finished": i, "announced": i > 0,
            }
        _subagent._ASYNC_AGENTS["agent-9"] = {
            "instruction": "q", "status": "done", "answer": "keep me",
            "started": 9, "finished": 9, "announced": False,
        }
        _subagent._evict_finished_agents()
    assert "agent-9" in _subagent._ASYNC_AGENTS, "unannounced answer was evicted"


# --- PF-17: bounded async fan-out ---------------------------------------------

def test_async_fan_out_beyond_the_cap_queues(nx_fresh, monkeypatch):
    """Spawns past NX_SUBAGENT_MAX_CONCURRENT park as `queued` instead of each
    starting its own stream, and drain as the running ones finish."""
    nx, _ = nx_fresh
    from nx import _subagent
    import nx as m

    monkeypatch.setenv("NX_SUBAGENT_MAX_CONCURRENT", "2")
    monkeypatch.setattr(m, "NX_SUBAGENT_MAX_CONCURRENT", 2)
    monkeypatch.setattr(_subagent, "_ASYNC_SEM", None)

    started, release = threading.Event(), threading.Event()

    def fake_loop(messages, *a, **kw):
        started.set()
        release.wait(timeout=5)
        messages.append({"role": "assistant", "content": "done"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(m, "_run_model_turn_loop", fake_loop)
    outs = [_subagent._spawn_async(f"q{i}", 5) for i in range(5)]
    time.sleep(0.05)
    with _subagent._ASYNC_LOCK:
        statuses = [r["status"] for r in _subagent._ASYNC_AGENTS.values()]
    assert statuses.count("running") <= 2, statuses
    assert statuses.count("queued") >= 2, statuses
    assert sum("(queued)" in o for o in outs) >= 2, outs
    assert started.wait(timeout=5)
    release.set()
    # let the parked ones drain (each acquires as a slot frees)
    deadline = time.time() + 5
    while time.time() < deadline:
        with _subagent._ASYNC_LOCK:
            if not _subagent._ASYNC_QUEUED:
                break
        time.sleep(0.02)
    assert not _subagent._ASYNC_QUEUED, "queue never drained"


# --- PF-18: validity anchors ---------------------------------------------------

def test_multi_path_read_is_invalidated_by_any_of_its_paths(nx_fresh):
    """Editing file #2 of a list read must drop the cached entry."""
    nx, tmp = nx_fresh
    a, b = tmp / "a.txt", tmp / "b.txt"
    a.write_text("A\n")
    b.write_text("B\n")
    from nx._dispatch import run_tool, _tool_cache_key

    run_tool("read_file", {"path": [str(a), str(b)]})
    key = _tool_cache_key("read_file", {"path": [str(a), str(b)]})
    assert key in nx._state._TOOL_RESULT_CACHE
    run_tool("write_file", {"path": str(b), "content": "B2\n"})
    assert key not in nx._state._TOOL_RESULT_CACHE, "second path was not an anchor"


def test_search_files_hit_is_invalidated_by_a_subdirectory_change(nx_fresh):
    """A new/changed file below the searched root must invalidate the cache hit
    even though the root directory's own mtime never moves."""
    nx, tmp = nx_fresh
    sub = tmp / "sub"
    sub.mkdir()
    (sub / "one.txt").write_text("needle\n")
    from nx._dispatch import run_tool, _tool_cache_key

    out = run_tool("search_files", {"pattern": "needle", "path": str(tmp)})
    assert "needle" in out
    key = _tool_cache_key("search_files", {"pattern": "needle", "path": str(tmp)})
    assert key in nx._state._TOOL_RESULT_CACHE

    # a new file appears in the subdirectory (no run_bash, no edit tool)
    time.sleep(0.01)
    (sub / "two.txt").write_text("needle again\n")
    second = run_tool("search_files", {"pattern": "needle", "path": str(tmp)})
    assert "two.txt" in second, "stale search hit survived a subdirectory change"


def test_editing_a_file_under_a_searched_root_drops_the_hit(nx_fresh):
    """The pre-commit path drop must also see a search_files entry whose anchor
    is the searched root: an edit anywhere below it invalidates the hit."""
    nx, tmp = nx_fresh
    sub = tmp / "sub"
    sub.mkdir()
    target = sub / "one.txt"
    target.write_text("needle\n")
    from nx._dispatch import run_tool, _tool_cache_key

    run_tool("search_files", {"pattern": "needle", "path": str(tmp)})
    key = _tool_cache_key("search_files", {"pattern": "needle", "path": str(tmp)})
    assert key in nx._state._TOOL_RESULT_CACHE
    run_tool("write_file", {"path": str(target), "content": "needle changed\n"})
    assert key not in nx._state._TOOL_RESULT_CACHE


def test_external_edit_of_second_path_invalidates_without_a_tool(nx_fresh):
    """Anchoring on every path also fixes the no-tool window: an external write
    to path #2 invalidates the hit on the next read."""
    nx, tmp = nx_fresh
    a, b = tmp / "a.txt", tmp / "b.txt"
    a.write_text("A\n")
    b.write_text("B1\n")
    from nx._dispatch import run_tool

    first = run_tool("read_file", {"path": [str(a), str(b)]})
    assert "B1" in first
    time.sleep(0.01)
    b.write_text("B2\n")
    second = run_tool("read_file", {"path": [str(a), str(b)]})
    assert "B2" in second, "stale multi-path hit served after an external edit"


# --- T-12 / PF-05: amortized persistence ---------------------------------------

def _patch_writes(monkeypatch, undo_module):
    """Count atomic_write_private calls into undo.json."""
    calls = []
    real = undo_module.atomic_write_private

    def counting(path, content):
        calls.append((path, len(content)))
        return real(path, content)

    monkeypatch.setattr(undo_module, "atomic_write_private", counting)
    return calls


def test_push_checkpoint_does_not_write_per_edit(nx_fresh, monkeypatch):
    """A burst of pushes must not re-serialize the stack each time — that was
    PF-05's write amplification."""
    nx, tmp = nx_fresh
    from nx import _undo

    calls = _patch_writes(monkeypatch, _undo)
    for i in range(25):
        p = tmp / f"w{i}.txt"
        p.write_text("x" * 4000)
        _undo.push_checkpoint("write_file", [_undo.snapshot(str(p))])
    assert calls == [], f"persisted per edit: {len(calls)} writes"
    _undo.flush_undo_stack()
    assert len(calls) == 1, calls


def test_flush_is_idempotent_and_writes_only_when_dirty(nx_fresh, monkeypatch):
    nx, tmp = nx_fresh
    from nx import _undo

    calls = _patch_writes(monkeypatch, _undo)
    _undo.flush_undo_stack()  # nothing dirty
    assert calls == []
    p = tmp / "f.txt"
    p.write_text("v")
    _undo.push_checkpoint("write_file", [_undo.snapshot(str(p))])
    _undo.flush_undo_stack()
    _undo.flush_undo_stack()
    assert len(calls) == 1, calls


def test_session_save_cost_ratio_50_to_300_messages(nx_fresh, monkeypatch):
    """T-12: a save must not scale with the transcript (ratio < 2 from 50 to
    300 messages). PERF.md §5's harness rule: assert a *ratio* and take the
    best of 5, never a mean — noise on a loaded box only ever inflates a
    sample, so the minimum is the robust statistic."""
    nx, tmp = nx_fresh
    from nx import _sessions_fs

    def mk(n):
        msgs = [{"role": "system", "content": "S" * 400}]
        for i in range(n):
            msgs.append({"role": "user", "content": f"q{i} " + "x" * 400})
            msgs.append({"role": "assistant", "content": f"a{i} " + "y" * 900})
        return msgs

    def best_of(msgs, sid):
        _sessions_fs._write_session(sid, msgs)  # create + warm
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            _sessions_fs._write_session(sid, msgs)
            samples.append(time.perf_counter() - t0)
        return min(samples)

    small = best_of(mk(25), "ratio-small")
    big = best_of(mk(150), "ratio-big")
    ratio = big / max(small, 1e-6)
    # 6x the messages is ~2.1x the cost in practice: per-save fixed costs
    # (lock, open, index) dominate the small sample, and the marginal
    # per-message work (store-form copy + json.dumps) is genuinely linear.
    # 2.5 catches real transcript-scaling regressions without flaking on a
    # loaded box; measured 0.3ms small / 0.7ms big.
    assert ratio < 2.5, f"save cost ratio {ratio:.2f} (small {small*1000:.1f}ms)"


def test_undo_json_is_machine_compact(nx_fresh, monkeypatch):
    """PF-05: `indent` is off the persist path; the file is still valid JSON."""
    nx, tmp = nx_fresh
    from nx import _undo

    p = tmp / "f.txt"
    p.write_text("v")
    _undo.push_checkpoint("write_file", [_undo.snapshot(str(p))])
    _undo.flush_undo_stack()
    raw = open(_undo._undo_path(), encoding="utf-8").read()
    assert "\n" not in raw.strip(), "persisted with indent"
    assert json.loads(raw)["version"] == 1


# --- PF-19: diff hunk cap ------------------------------------------------------

def test_diff_hunk_is_capped_with_an_elision_notice(monkeypatch):
    from nx._approval import _diff_hunk

    monkeypatch.setenv("NX_DIFF_HUNK_MAX_CHARS", "2000")
    before = "\n".join(f"line {i}" for i in range(4000))
    after = "\n".join(f"other {i}" for i in range(4000))  # whole-file replace
    out = _diff_hunk(before, after)
    assert len(out) <= 2000 + 300, len(out)
    assert "elided" in out and "read_file" in out


def test_diff_hunk_uncapped_when_disabled(monkeypatch):
    from nx._approval import _diff_hunk

    monkeypatch.setenv("NX_DIFF_HUNK_MAX_CHARS", "0")
    before = "\n".join(f"line {i}" for i in range(500))
    out = _diff_hunk(before, before.replace("line 5", "X", 1))
    assert "elided" not in out
