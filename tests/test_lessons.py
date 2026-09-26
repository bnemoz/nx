#!/usr/bin/env python3
"""recall_lessons (0.102.0): BM25 cross-session learning recall over the
session store's FTS index (nx/_lessons.py).

Pins: BM25 ranking (a body-heavy match outranks a title-only match), the
NX_LESSONS_LIMIT cap, scope=project cwd filtering, the degradation contract
(bad FTS syntax / locked store / empty store → no hits, never a raise), and
the read-only posture (no gate entry, in the plan read-only set, cacheable).
"""
import os
import sqlite3
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-lessons-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest  # noqa: E402
import nx as m  # noqa: E402
from nx import _lessons  # noqa: E402
from nx import _sessiondb  # noqa: E402


def _clear():
    d = m._sessions_dir()
    assert os.path.basename(d).startswith("nx-lessons-"), d
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isdir(p):
            import shutil
            shutil.rmtree(p)
        else:
            os.remove(p)
    _sessiondb._MIGRATED.clear()


def _seed(sid, text, cwd=None, title=None):
    msgs = [{"role": "user", "content": text}]
    meta = {"title": title}
    if cwd:
        meta["cwd"] = cwd
    m._write_session(sid, msgs, meta)


# --- the happy path -----------------------------------------------------------

def test_recall_finds_and_ranks_by_bm25():
    _clear()
    # s1 mentions quokka once; s2 discusses it at length — BM25 must put s2
    # first (more term occurrences in a comparable-length body).
    _seed("20260101-120000-lsa1", "the quokka escaped")
    _seed("20260101-120000-lsa2",
          "quokka quokka quokka habitat study quokka feeding habits")
    out = _lessons.recall_lessons("quokka")
    assert out.startswith("past sessions matching")
    hits = _lessons.search_lessons("quokka")
    assert [h["id"] for h in hits][0] == "20260101-120000-lsa2"
    # the snippet carries matched body text
    assert "quokka" in hits[0]["snippet"].lower()


def test_recall_misses_quietly():
    _clear()
    _seed("20260101-120000-lsb1", "ordinary debugging session")
    out = _lessons.recall_lessons("pterodactyl")
    assert out.startswith("[no past session matching")
    assert _lessons.search_lessons("") == []


def test_multi_term_query_is_anded():
    _clear()
    _seed("20260101-120000-lsc1", "retry ladder around the timeout")
    _seed("20260101-120000-lsc2", "the timeout fired")
    hits = _lessons.search_lessons("retry timeout")
    assert [h["id"] for h in hits] == ["20260101-120000-lsc1"]
    # OR widens
    hits_or = _lessons.search_lessons("retry OR fired")
    assert {h["id"] for h in hits_or} == {
        "20260101-120000-lsc1", "20260101-120000-lsc2"}


def test_bad_fts_syntax_degrades():
    _clear()
    _seed("20260101-120000-lsd1", "hello world")
    assert _lessons.search_lessons('"unterminated') == []
    assert _lessons.search_lessons("NEAR(") == []
    assert _lessons.recall_lessons("NEAR(").startswith("[no past session")


# --- the cap + scope ----------------------------------------------------------

def test_limit_cap_and_env(monkeypatch):
    _clear()
    for i in range(8):
        _seed(f"20260101-12000{i}-lse{i}", f"shared marker word {i}")
    hits = _lessons.search_lessons("shared marker")
    assert len(hits) == 5  # DEFAULT_LIMIT
    assert len(_lessons.recall_lessons("shared marker").splitlines()) > 5
    monkeypatch.setenv("NX_LESSONS_LIMIT", "2")
    assert len(_lessons.search_lessons("shared marker")) == 2
    monkeypatch.setenv("NX_LESSONS_LIMIT", "0")
    assert len(_lessons.search_lessons("shared marker")) == 8


def test_scope_project_filters_by_cwd():
    _clear()
    here = os.path.realpath(_tmp)
    _seed("20260101-120000-lsf1", "deploy the worker pool", cwd=here)
    _seed("20260101-120000-lsf2", "deploy the worker pool",
          cwd="/elsewhere/project")
    old_root = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = here
    try:
        hits = _lessons.search_lessons("deploy worker", scope="project")
    finally:
        m.st.PROJECT_ROOT = old_root
    assert [h["id"] for h in hits] == ["20260101-120000-lsf1"]
    # scope=all (default) sees both
    assert len(_lessons.search_lessons("deploy worker")) == 2
    # 'global' is an accepted alias of 'all'
    assert len(_lessons.search_lessons("deploy worker", scope="global")) == 2


def test_scope_project_with_no_root_sees_everything(monkeypatch):
    _clear()
    _seed("20260101-120000-lsg1", "cache warmup details")
    monkeypatch.setattr(m.st, "PROJECT_ROOT", None)
    assert len(_lessons.search_lessons("cache warmup", scope="project")) == 1


# --- degradation / store hygiene ----------------------------------------------

def test_corrupt_store_degrades_not_raises():
    _clear()
    _seed("20260101-120000-lsh1", "quorum tuning notes")
    with open(_sessiondb._store_path(), "w") as f:
        f.write("not a database at all")
    assert _lessons.search_lessons("quorum") == []  # never raises
    assert _lessons.recall_lessons("quorum").startswith("[no past session")


def test_deleted_session_is_not_recalled():
    _clear()
    sid = "20260101-120000-lsi1"
    _seed(sid, "unique lemur facts")
    assert _lessons.search_lessons("lemur")
    m._delete_session(sid)
    assert _lessons.search_lessons("lemur") == []


def test_lessons_never_serve_a_store_only_body():
    """A session saved with update_index=False (autosave) has no FTS row
    until the next search backfills it — recall must backfill too."""
    _clear()
    _seed("20260101-120000-lsj1", "otter sliding behavior")
    import sqlite3 as _sq
    conn = _sq.connect(_sessiondb._store_path())
    conn.execute("DELETE FROM sessions_fts WHERE session_id = ?",
                 ("20260101-120000-lsj1",))
    conn.commit()
    conn.close()
    assert _lessons.search_lessons("otter sliding")  # backfilled on the call


# --- registry posture ---------------------------------------------------------

def test_lessons_is_read_only_in_every_registry():
    names = {t["function"]["name"] for t in m.TOOLS}
    assert "recall_lessons" in names
    assert "recall_lessons" in m._dispatch.DISPATCH
    assert "recall_lessons" in m._dispatch._CACHEABLE_READ_TOOLS
    from nx._plan import PLAN_READONLY_TOOLS, PLAN_GATE_TOOLS
    assert "recall_lessons" in PLAN_READONLY_TOOLS
    assert "recall_lessons" not in PLAN_GATE_TOOLS
    # always-on: no gate entry, offered on turn 1
    assert "recall_lessons" not in m._dispatch._TOOL_GATES
    assert "recall_lessons" in [t["function"]["name"] for t in m._active_tools()]
    # and it joins the read-only sub-agent tier
    from nx._subagent_tiers import _READ_ONLY_TOOL_NAMES
    assert "recall_lessons" in _READ_ONLY_TOOL_NAMES


def test_run_tool_recalls_without_approval():
    _clear()
    _seed("20260101-120000-lsk1", "marinated deployment recipe")
    out = m.run_tool("recall_lessons", {"query": "marinated deployment"})
    assert "marinated" in out
    assert not out.startswith("ERROR")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
