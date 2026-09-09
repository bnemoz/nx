#!/usr/bin/env python3
"""2.6 memory retrieval v2: the FTS5 accelerator (`nx/_memindex.py`), the
one-scanner rewire, the `/memory list` scope fix, the no-clobber memory
writer and the optimize oldest-tail selection.

No model/network round-trip — `_confirm` is stubbed and everything touches
temp dirs. The FTS tests skip (never fail) when this sqlite build lacks the
trigram tokenizer, mirroring the production fallback.
"""
import os
import sys
import tempfile

os.environ["NX_HOME"] = tempfile.mkdtemp(prefix="nx-test-memidx-")
os.environ["NX_SESSIONS_DIR"] = os.environ["NX_HOME"]
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _memindex  # noqa: E402
from nx import _memory_cmd  # noqa: E402
from nx import _memory_optimize as opt  # noqa: E402
from nx._private import atomic_write_private  # noqa: E402

try:
    import sqlite3 as _sq
    _c = _sq.connect(":memory:")
    _c.execute("CREATE VIRTUAL TABLE _p USING fts5(x, tokenize='trigram')")
    _c.close()
    HAVE_FTS = True
except Exception:  # pragma: no cover - only on odd builds
    HAVE_FTS = False


_saved_confirm = None
_saved_home = None
_saved_root = None
_OUR_HOME = os.environ["NX_HOME"]


def _setup_function():
    global _saved_confirm, _saved_home, _saved_root
    _saved_confirm = m._confirm
    _saved_home = m.st.SESSION_HOME
    _saved_root = m.st.PROJECT_ROOT
    m._confirm = lambda action, preview=None: True
    m.st.SESSION_HOME = _OUR_HOME
    m.st.PROJECT_ROOT = None


def _teardown_function():
    global _saved_confirm, _saved_home, _saved_root
    m._confirm = _saved_confirm
    m.st.SESSION_HOME = _saved_home
    m.st.PROJECT_ROOT = _saved_root
    _saved_confirm = None


setup_function = lambda _fn: _setup_function()  # noqa: E731 (file-local convention)
teardown_function = lambda _fn: _teardown_function()  # noqa: E731


def _note(d, name, body):
    """Write one note directly (full control over name/timestamp)."""
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    atomic_write_private(p, body)
    return p


def _proj_dir():
    root = tempfile.mkdtemp(prefix="nx-memidx-store-")
    d = os.path.join(root, ".nx", "memories")
    m.st.PROJECT_ROOT = root
    return d


# --- the accelerator -----------------------------------------------------------

def test_search_dir_finds_by_body_and_title_terms():
    d = _proj_dir()
    _note(d, "20260101-000001_alpha.md",
          "# Alpha Note\n\nThe API needs X-Header: 1 for auth.\n")
    _note(d, "20260101-000002_beta.md",
          "# Beta Note\n\nsomething about renderer internals\n")
    rows = _memindex.search_dir(d, "X-Header")
    if not HAVE_FTS:
        assert rows is None
        return
    assert rows is not None and [r[0] for r in rows] == ["20260101-000001_alpha.md"], rows
    name, mtime, title, body, snip = rows[0]
    assert title == "Alpha Note" and "X-Header" in body
    assert snip  # FTS snippet around the hit
    rows2 = _memindex.search_dir(d, "renderer internals")
    assert rows2 is not None and [r[0] for r in rows2] == ["20260101-000002_beta.md"]
    # no hits: the caller falls back (empty list, not None, means "index worked")
    rows3 = _memindex.search_dir(d, "zzz-totally-absent")
    assert rows3 == [], rows3


def test_reconcile_indexes_new_and_prunes_deleted():
    d = _proj_dir()
    p1 = _note(d, "20260101-000001_keep.md", "# Keep\n\nstays indexed\n")
    rows = _memindex.search_dir(d, "stays")
    assert rows and rows[0][0].endswith("keep.md")
    # new note appears without touching the index
    _note(d, "20260101-000002_new.md", "# New\n\nfreshly written fact\n")
    # deleted note vanishes
    os.remove(p1)
    rows2 = _memindex.search_dir(d, "stays")
    assert rows2 == [], rows2
    rows3 = _memindex.search_dir(d, "freshly")
    assert rows3 and rows3[0][0].endswith("new.md")
    # and a content change is picked up (mtime bumped)
    _note(d, "20260101-000002_new.md", "# New\n\nrewritten with uvloop\n")
    rows4 = _memindex.search_dir(d, "uvloop")
    assert rows4 and rows4[0][0].endswith("new.md")


def test_search_dir_none_when_disabled_or_noterms():
    d = _proj_dir()
    _note(d, "20260101-000001_a.md", "# A\n\nbody one\n")
    os.environ["NX_MEMORY_FTS"] = "0"
    try:
        assert _memindex.search_dir(d, "body") is None
        assert _memindex.enabled() is False
    finally:
        os.environ.pop("NX_MEMORY_FTS", None)
    assert _memindex.enabled() is True
    # empty query terms -> caller must use its own scan
    assert _memindex.search_dir(d, "") is None
    assert _memindex.search_dir(d, "   ") is None


def test_short_terms_use_the_plain_scan():
    d = _proj_dir()
    _note(d, "20260101-000001_a.md", "# A\n\nti-ny term\n")
    # "ti" has no trigram; _memory_scan must route it to the plain scan and
    # still find it (never through the accelerator).
    hits, _of, _tot = _memory_cmd._memory_scan("ti", "project", 0)
    assert [h.name for h in hits] == ["20260101-000001_a.md"], hits


def test_fts_off_results_match_fts_on_results():
    """The accelerator must not change the answer: same names, same order,
    same snippets, with and without it (the _rank_hit scoring is shared)."""
    d = _proj_dir()
    _note(d, "20260101-000001_quirk.md",
          "# Quirk about the renderer\n\nunrelated body words here\n")
    _note(d, "20260101-000002_plain.md",
          "# Unrelated note\n\nsomething about renderer internals\n")
    def _results():
        hits, _of, tot = _memory_cmd._memory_scan("renderer", "project", 0)
        return [(h.name, h.title, round(h.score, 3)) for h in hits], tot
    with_fts = _results()
    assert len(with_fts[0]) == 2
    os.environ["NX_MEMORY_FTS"] = "0"
    try:
        without = _results()
    finally:
        os.environ.pop("NX_MEMORY_FTS", None)
    assert with_fts == without, (with_fts, without)
    # title hit ranks above body hit on both paths
    assert with_fts[0][0][0].endswith("quirk.md")


def test_corrupt_index_is_set_aside_and_rebuilt():
    d = _proj_dir()
    _note(d, "20260101-000001_a.md", "# A\n\nsearchable body\n")
    rows = _memindex.search_dir(d, "searchable")
    if not HAVE_FTS:
        assert rows is None
        return
    assert rows
    db = os.path.join(d, _memindex.INDEX_FILENAME)
    with open(db, "wb") as f:
        f.write(b"this is not a database at all")
    rows2 = _memindex.search_dir(d, "searchable")  # must not raise
    assert rows2 and rows2[0][0].endswith("a.md")
    assert os.path.exists(db + ".corrupt"), os.listdir(d)


def test_memory_search_end_to_end_hits_index_and_reports_dirs():
    d = _proj_dir()
    _note(d, "20260101-000001_m.md", "# Path memo\n\nmarkerxyz body\n")
    out = _memory_cmd.memory_search(query="markerxyz", scope="project")
    if HAVE_FTS:
        # served from the index: dir hint still correct, hit present
        assert "Path memo" in out and d in out, out
        assert os.path.exists(os.path.join(d, _memindex.INDEX_FILENAME))
    # and the write tool keeps the index coherent through its own search path
    res = _memory_cmd.memory_write(title="Second memo", body="secondmemo body",
                                   scope="project")
    assert res.startswith("saved project memory"), res
    out2 = _memory_cmd.memory_search(query="secondmemo", scope="project")
    assert "Second memo" in out2, out2


# --- the one scanner ------------------------------------------------------------

def test_iter_notes_is_the_single_scanner():
    d = _proj_dir()
    _note(d, "20260101-000002_b.md", "# B\n\ntwo\n")
    _note(d, "20260101-000001_a.md", "# A\n\none\n")
    os.makedirs(os.path.join(d, "archive", "20260101-000000"), exist_ok=True)
    _note(os.path.join(d, "archive", "20260101-000000"),
          "20260101-000000_old.md", "# Old\n\narchived\n")
    with open(os.path.join(d, "20260101-000003_bad.md"), "wb") as f:
        f.write(b"\xff\xfe not utf-8")
    notes = _memory_cmd._iter_notes(d)
    assert [n[0] for n in notes] == ["20260101-000002_b.md", "20260101-000001_a.md"], notes
    names = [n[0] for n in notes]
    titles = [n[3] for n in notes]
    assert titles == ["B", "A"]
    assert "20260101-000003_bad.md" not in names  # unreadable: skipped
    assert not any("archive" in n for n in names)
    # delete resolver (mtime-newest-first, unlike the name-ordered scanner)
    # + list + memory note all see exactly these names
    entry_names = sorted(h.name for h in _memory_cmd._scan_memory_entries("project"))
    assert entry_names == sorted(names), entry_names
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _memory_cmd._cmd_memory_list()
    out = buf.getvalue()
    assert "[project] B" in out and "[project] A" in out, out
    assert "archived" not in out


# --- /memory list scope fix -----------------------------------------------------

def test_memory_list_shows_project_only_store():
    d = _proj_dir()  # project scope only; global NX_HOME has no memories
    _note(d, "20260101-000001_solo.md", "# Solo note\n\nbody\n")
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _memory_cmd._cmd_memory_list()
    out = buf.getvalue()
    assert "no memories saved yet" not in out, out
    assert "Solo note" in out and "[project]" in out, out


# --- writer race -----------------------------------------------------------------

def test_write_memory_file_never_clobbers_same_name():
    """Two writers that both pick the same (timestamp, slug) name must both
    survive with distinct suffixes — the pre-2.6 claim was exists-then-write
    with an os.replace publication, so the loser's bytes vanished."""
    d = _proj_dir()
    m.st.PROJECT_ROOT = os.path.dirname(os.path.dirname(d))
    a = _memory_cmd._write_memory_file("Clash Title", "first body", "project")
    b = _memory_cmd._write_memory_file("Clash Title", "second body", "project")
    assert a and b and a[0] != b[0], (a, b)
    fa, fb = open(a[0]).read(), open(b[0]).read()
    assert "first body" in fa and "second body" in fb
    # both are valid notes the scanner resolves
    titles = sorted(n[3] for n in _memory_cmd._iter_notes(d))
    assert titles == ["Clash Title", "Clash Title"]


def test_write_memory_file_suffix_walk_under_parallel_writers():
    """10 threads racing the writer: exactly 10 notes, none lost, none
    truncated, all readable."""
    d = _proj_dir()
    errs = []

    def w(i):
        try:
            r = _memory_cmd._write_memory_file(f"Race {i}", f"race body {i}", "project")
            if r is None:
                errs.append(f"writer {i} returned None")
        except Exception as e:  # pragma: no cover
            errs.append(f"{type(e).__name__}: {e}")

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(w, range(10)))
    assert not errs, errs
    notes = _memory_cmd._iter_notes(d)
    assert len(notes) == 10, [n[0] for n in notes]
    bodies = sorted(open(n[1]).read() for n in notes)
    assert len({b for b in bodies}) == 10
    for i in range(10):
        assert any(f"race body {i}" in b for b in bodies), i


def test_atomic_write_private_noreplace_contract():
    from nx._private import atomic_write_private_noreplace
    d = tempfile.mkdtemp(prefix="nx-noreplace-")
    p = os.path.join(d, "x.md")
    atomic_write_private_noreplace(p, "first")
    try:
        atomic_write_private_noreplace(p, "second")
        raise AssertionError("expected FileExistsError")
    except FileExistsError:
        pass
    assert open(p).read() == "first"
    assert os.listdir(d) == ["x.md"]  # no tmp left behind
    mode = oct(os.stat(p).st_mode & 0o777)
    assert mode == "0o600", mode


# --- optimize oldest-tail --------------------------------------------------------

def _opt_client_text(text):
    class _Msg:
        content = text
    class _Choice:
        message = _Msg()
        finish_reason = "stop"
    class _Resp:
        choices = [_Choice()]
        usage = None
        def model_dump(self):
            return {}
    class _Completions:
        def create(self, **kwargs):
            return _Resp()
    class _Chat:
        completions = _Completions()
    class _Client:
        chat = _Chat()
        base_url = "http://localhost:9/v1"
    return _Client()


def test_optimize_max_notes_reviews_oldest_tail():
    root = tempfile.mkdtemp(prefix="nx-opt-tail-")
    m.st.PROJECT_ROOT = root
    d = os.path.join(root, ".nx", "memories")
    # 6 notes; the two NEWEST are what a session note shows and must stay
    # reviewable while the cap bites.
    stamps = ["20260101-00000%d" % i for i in range(1, 7)]
    for i, s in enumerate(stamps):
        _note(d, f"{s}_n{i}.md", f"# Note {i}\n\nbody {i}\n")
    os.utime(os.path.join(d, f"{stamps[0]}_n0.md"), (1_100_000_000, 1_100_000_000))
    os.utime(os.path.join(d, f"{stamps[1]}_n1.md"), (1_100_000_001, 1_100_000_001))
    os.utime(os.path.join(d, f"{stamps[2]}_n2.md"), (1_100_000_002, 1_100_000_002))
    os.utime(os.path.join(d, f"{stamps[3]}_n3.md"), (1_100_000_003, 1_100_000_003))
    os.utime(os.path.join(d, f"{stamps[4]}_n4.md"), (1_100_000_004, 1_100_000_004))
    os.utime(os.path.join(d, f"{stamps[5]}_n5.md"), (1_100_000_005, 1_100_000_005))
    real_plan_prompt = opt._plan_prompt
    seen_names: list[str] = []

    def _spy(chunk, focus, carry=None):
        seen_names.extend(n.name for n in chunk)
        return real_plan_prompt(chunk, focus, carry)

    opt._plan_prompt = _spy
    saved = os.environ.get("NX_MEMORY_OPTIMIZE_MAX_NOTES")
    os.environ["NX_MEMORY_OPTIMIZE_MAX_NOTES"] = "4"
    try:
        opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")
    finally:
        opt._plan_prompt = real_plan_prompt
        if saved is None:
            os.environ.pop("NX_MEMORY_OPTIMIZE_MAX_NOTES", None)
        else:
            os.environ["NX_MEMORY_OPTIMIZE_MAX_NOTES"] = saved
    assert len(seen_names) == 4, seen_names
    # the OLDEST four (n0..n3) were reviewed; the newest tail was not
    assert sorted(seen_names) == [
        "20260101-000001_n0.md", "20260101-000002_n1.md",
        "20260101-000003_n2.md", "20260101-000004_n3.md"], seen_names


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
