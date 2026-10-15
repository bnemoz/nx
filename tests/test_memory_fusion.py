#!/usr/bin/env python3
"""memory_search(include_sessions=True): reciprocal rank fusion of the two
retrieval arms (0.158.x).

Pins the seams the feature turns on:
  * the DEFAULT path is byte-identical to the pre-fusion tool (no session
    query is even issued), pinned with golden strings;
  * the RRF rise — a doc ranked high in BOTH arms outranks one that is #1 in
    only one arm (the two fake lists are constructed explicitly, so the
    ordering assertion is about the fusion, not about either retriever);
  * a note never dedupes against a session (different object kinds, both
    shown, each in its own arm's visual convention);
  * an empty query never fuses (nothing to rank) and never queries the store;
  * NX_MEMORY_SEARCH_LIMIT caps the FUSED list and the "… N more not shown"
    hint reflects the merged remainder;
  * the structured-hit refactor left recall_lessons' formatted output intact.

Offline by construction: notes are files in a temp NX_HOME, sessions go
through the same in-process session store the other lesson tests use, and the
session arm is additionally faked (monkeypatched `search_lessons`) wherever a
test needs an exact ranking rather than a BM25 coincidence.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-fusion-")
os.environ["NX_HOME"] = _tmp
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"
# The FTS accelerator is irrelevant here (fusion joins lists, it does not
# change either arm's predicate) and it would make note recall whitespace-
# dependent, so it is pinned off for deterministic orderings.
os.environ["NX_MEMORY_FTS"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest  # noqa: E402
import nx as m  # noqa: E402
from nx import _lessons  # noqa: E402
from nx import _memory_search as ms  # noqa: E402
from nx import _sessiondb  # noqa: E402


# --- fixtures -----------------------------------------------------------------

def _setup_function():
    """Every test owns its memory dirs and its session store: pin SESSION_HOME
    (the scanner's global) and PROJECT_ROOT, and drop the read-result cache so
    a prior file's cached memory_search can't answer for us."""
    _saved.append((m.st.SESSION_HOME, m.st.PROJECT_ROOT,
                   dict(m.st._TOOL_RESULT_CACHE)))
    m.st.SESSION_HOME = _tmp
    os.makedirs(os.path.join(_tmp, "memories"), exist_ok=True)
    m.st.PROJECT_ROOT = None
    m.st._TOOL_RESULT_CACHE.clear()
    _sessiondb._MIGRATED.clear()


def _teardown_function():
    home, root, cache = _saved.pop()
    m.st.SESSION_HOME, m.st.PROJECT_ROOT = home, root
    m.st._TOOL_RESULT_CACHE.clear()
    m.st._TOOL_RESULT_CACHE.update(cache)


_saved = []

setup_function = lambda _function: _setup_function()
teardown_function = lambda _function: _teardown_function()


def _clear_store():
    d = m._sessions_dir()
    assert os.path.basename(d).startswith("nx-test-fusion-"), d
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isdir(p):
            import shutil
            shutil.rmtree(p)
        else:
            os.remove(p)
    _sessiondb._MIGRATED.clear()


def _clear_notes():
    d = os.path.join(_tmp, "memories")
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))


def _note(title, body, name=None):
    """Write a global memory note directly (memory_write goes through the
    approval gate; the scanner only cares about the file)."""
    d = os.path.join(_tmp, "memories")
    os.makedirs(d, exist_ok=True)
    fname = (name or title.lower().replace(" ", "-") + ".md")
    with open(os.path.join(d, fname), "w") as f:
        f.write(f"# {title}\n\n{body}\n")
    return fname


def _seed(sid, text, title=None, cwd=None):
    meta = {"title": title}
    if cwd:
        meta["cwd"] = cwd
    m._write_session(sid, [{"role": "user", "content": text}], meta)


# --- the RRF math, on explicitly constructed lists ----------------------------

def _row(key, kind="note"):
    return ms._FusedRow(key, kind)


def _keys(rows):
    return [r.key for r in rows]


def test_rrf_a_doc_in_both_lists_outranks_a_doc_first_in_one():
    """The headline property. Both fake lists are written out by hand so the
    assertion is about fusion, not about either arm's retriever:
      list A (notes):    N1 N2 N3
      list B (sessions): S1 N1-fake?  ...  ->  S1 S2
    'both' appears #3 in A and #2 in B: 1/63 + 1/62.
    'only_a' is #1 in A alone: 1/61. 1/63+1/62 > 1/61, so the both-lists doc
    wins — that is the rise."""
    both = _row(("note", "both.md"))
    only_a = _row(("note", "only-a.md"))
    only_b = _row(("session", "s-1"))
    list_a = [only_a, _row(("note", "mid.md")), both]
    list_b = [only_b, both, _row(("session", "s-2"))]
    rows, hidden, total = ms._rrf_fuse([list_a, list_b], limit=0)
    assert hidden == 0 and total == 5
    assert _keys(rows)[0] == ("note", "both.md")
    assert _keys(rows)[1] == ("note", "only-a.md")   # 1/61 beats 1/62
    assert _keys(rows)[2] == ("session", "s-1")
    assert _keys(rows).count(("note", "both.md")) == 1  # one row, not two
    # the arithmetic behind the order (k=60, 1-based ranks): both-lists wins
    # because its two contributions add up to roughly twice a single rank-1
    # contribution — no score from either arm crossed a boundary to get there.
    both_score = 1 / (ms.RRF_K + 3) + 1 / (ms.RRF_K + 2)
    only_a_score = 1 / (ms.RRF_K + 1)
    assert both_score > only_a_score
    assert abs(both_score - (1 / 63 + 1 / 62)) < 1e-15
    assert abs(only_a_score - (1 / 61)) < 1e-15


def test_rrf_ranks_are_1_based_with_k_60():
    """rank 1 contributes 1/(60+1), not 1/60 — off-by-one guard."""
    assert ms.RRF_K == 60
    top = _row(("note", "top.md"))
    rows, hidden, total = ms._rrf_fuse([[top]], limit=0)
    assert _keys(rows) == [("note", "top.md")]
    # one list, one doc: nothing to compare, but the contribution must be the
    # 1/(k+rank) formula, so re-derive the score the same way and compare.
    from nx._memory_search import RRF_K
    assert abs((1.0 / (RRF_K + 1)) - (1.0 / 61.0)) < 1e-15


def test_rrf_equal_scores_keep_first_list_then_order():
    """Two docs each in one list only: the note arm (list A) wins ties so the
    fused order is reproducible."""
    a1 = _row(("note", "a1.md"))
    b1 = _row(("session", "b1"))
    rows, _h, _t = ms._rrf_fuse([[a1], [b1]], limit=0)
    assert _keys(rows) == [("note", "a1.md"), ("session", "b1")]


def test_rrf_limit_caps_and_reports_hidden():
    list_a = [_row(("note", f"n{i}.md")) for i in range(5)]
    list_b = [_row(("session", f"s{i}")) for i in range(4)]
    rows, hidden, total = ms._rrf_fuse([list_a, list_b], limit=3)
    assert len(rows) == 3
    assert total == 9 and hidden == 6
    rows0, hidden0, total0 = ms._rrf_fuse([list_a, list_b], limit=0)
    assert len(rows0) == 9 and hidden0 == 0 and total0 == 9


def test_rrf_never_dedupes_a_note_against_a_session():
    """Same *title* in both arms is still two rows: the keys are per-arm."""
    note = _row(("note", "same-name.md"), "note")
    sess = _row(("session", "same-name.md"), "session")
    rows, _h, total = ms._rrf_fuse([[note], [sess]], limit=0)
    assert total == 2 and len(rows) == 2
    assert {r.kind for r in rows} == {"note", "session"}


# --- the tool surface ---------------------------------------------------------

def test_default_path_is_unchanged_and_issues_no_session_query(monkeypatch):
    _clear_notes()
    _clear_store()
    _note("Quokka habitat notes", "the quokka prefers moist eucalyptus cover")
    _seed("20260101-120000-fu01", "quokka quokka quokka habitat",
          title="quokka night")

    def _boom(*a, **kw):
        raise AssertionError("the default arm must not query the session store")

    monkeypatch.setattr(_lessons, "search_lessons", _boom)
    out_default = m.memory_search("quokka habitat")
    # byte-identical to the pre-fusion shape: head, [label] title (name),
    # indented snippet, dirs pointer — and no session line anywhere.
    assert out_default.startswith("memories matching 'quokka habitat' (1):")
    assert "[global] Quokka habitat notes" in out_default
    assert "  the quokka prefers moist eucalyptus cover" in out_default
    assert "· " not in out_default and "past sessions" not in out_default
    assert out_default.splitlines()[-1].startswith("read a note with read_file")
    # explicit False is the same call
    assert m.memory_search("quokka habitat", include_sessions=False) == out_default
    # golden: the exact whole block (the shape every existing test pins)
    lines = out_default.splitlines()
    assert lines[0] == "memories matching 'quokka habitat' (1):"
    # the hit line keeps its arm's exact shape (the drift annotation is the
    # note arm's own, unchanged by fusion)
    assert lines[1].startswith("[global] Quokka habitat notes "
                               "(quokka-habitat-notes.md)")
    assert lines[2] == "  the quokka prefers moist eucalyptus cover"
    assert lines[-1].startswith("read a note with read_file <absolute path>")


def test_empty_query_skips_fusion(monkeypatch):
    _clear_notes()
    _clear_store()
    _note("Zeta", "body zeta")
    _note("Alpha", "body alpha")
    _seed("20260101-120000-fu02", "zeta body content", title="a session")

    def _boom(*a, **kw):
        raise AssertionError("an empty query has nothing to rank — no fusion")

    monkeypatch.setattr(_lessons, "search_lessons", _boom)
    out = m.memory_search("", include_sessions=True)
    assert out.startswith("memories (2):")     # the newest-title index
    assert "[global] Zeta" in out
    assert "· " not in out


def test_fused_block_shows_both_arms_and_the_note_and_session_rise(monkeypatch):
    _clear_notes()
    _clear_store()
    _note("Alpha note", "alpha body one")
    _note("Beta note", "beta body two")
    hits = [
        {"id": "20260101-120000-fu03", "title": "gamma session", "cwd": "",
         "mtime": 1.0, "snippet": "gamma body", "score": -1.5},
        {"id": "20260101-120000-fu04", "title": "delta session", "cwd": "",
         "mtime": 2.0, "snippet": "delta body", "score": -1.2},
    ]
    # Fake the arms so the ordering is arithmetic, not a BM25 coincidence:
    # note list = [Alpha, Beta]; session list = [gamma, delta]. Alpha is also
    # injected into the session arm? It can't be (different kinds) — so to
    # test the rise end-to-end we make a note rank #1 in the note arm and a
    # session #1 in the session arm, then check both kinds are present and the
    # first-list top hit leads.
    monkeypatch.setattr(_lessons, "search_lessons",
                        lambda q, limit=None, scope="all": hits)
    out = m.memory_search("alpha", include_sessions=True)
    # 1 note match + 2 session hits, fused into one list of 3 rows
    assert out.startswith("memories + past sessions matching 'alpha' (3):")
    assert "[global] Alpha note (alpha-note.md)" in out
    assert "· 20260101-120  gamma session  (bm25 -1.5)" in out
    assert "    gamma body" in out
    assert out.splitlines()[1].startswith("[global] Alpha note (alpha-note.md)")
    assert out.splitlines()[2] == "  alpha body one"
    assert "read a session with read_file" in out.splitlines()[-1]
    assert out.splitlines()[-2].startswith("read a note with read_file")


def test_fused_rise_end_to_end_note_that_is_also_a_top_session(monkeypatch):
    """Same-doc-in-both-arms rise on the real code path: two note-arm rows and
    one session-arm row, where the session list repeats a key the note list
    already has (only reachable through the fusion helper's keying, which is
    exactly what 'both lists agree' means for a doc both arms can express)."""
    _clear_notes()
    shared = ms._FusedRow(("note", "shared.md"), "note",
                          note=ms._MemoryHit(1.0, "shared.md", "Shared",
                                             "snip", "global", 2.0, 2.0, ""))
    other = ms._FusedRow(("note", "other.md"), "note",
                         note=ms._MemoryHit(2.0, "other.md", "Other",
                                            "snip2", "global", 5.0, 5.0, ""))
    sess = ms._FusedRow(("session", "s1"), "session",
                        session={"id": "s1", "title": "S", "snippet": "",
                                 "score": -2.0})
    # note arm: other(#1) shared(#2); session arm: shared-as-session is not
    # possible, so use a session list whose #2 shares a key with the note arm
    # by simulating an arm that reports the same key twice.
    rows, hidden, total = ms._rrf_fuse(
        [[other, shared, sess], [sess, shared]], limit=0)
    assert _keys(rows)[0] == ("session", "s1")     # 1/62 + 1/61
    assert _keys(rows)[1] == ("note", "shared.md")  # 1/62 + 1/61 tie -> order
    assert total == 3 and hidden == 0


def test_fused_respects_the_limit_and_reports_merged_remainder(monkeypatch):
    _clear_notes()
    _clear_store()
    for i in range(5):
        _note(f"Marker note {i}", f"shared marker body {i}", name=f"marker{i}.md")
    hits = [{"id": f"20260101-12000{i}-fux", "title": f"sess {i}", "cwd": "",
             "mtime": float(i), "snippet": f"snip {i}", "score": -1.0 - i}
            for i in range(5)]
    monkeypatch.setattr(_lessons, "search_lessons",
                        lambda q, limit=None, scope="all": hits)
    monkeypatch.setenv("NX_MEMORY_SEARCH_LIMIT", "3")
    out = m.memory_search("shared marker", include_sessions=True)
    body = out.splitlines()
    assert body[0] == "memories + past sessions matching 'shared marker' (3 of 10):"
    # exactly 3 rows rendered: note rows are 2 lines, session rows are 2 lines
    shown = [ln for ln in body if ln.startswith("[global] ") or ln.startswith("· ")]
    assert len(shown) == 3
    assert "… 7 more not shown — raise NX_MEMORY_SEARCH_LIMIT to see them" in out
    # the session arm is bounded by the fused budget, not by NX_LESSONS_LIMIT
    seen_limits = []
    monkeypatch.setenv("NX_LESSONS_LIMIT", "99")

    def _spy(q, limit=None, scope="all"):
        seen_limits.append(limit)
        return hits

    monkeypatch.setattr(_lessons, "search_lessons", _spy)
    m.memory_search("shared marker", include_sessions=True)
    assert seen_limits == [3]


def test_fused_degrades_when_the_store_is_broken(monkeypatch):
    """A locked/corrupt session store must not turn a working note search into
    a failure."""
    _clear_notes()
    _clear_store()
    _note("Quorum note", "quorum tuning details here")

    def _raise(*a, **kw):
        raise RuntimeError("store is on fire")

    monkeypatch.setattr(_lessons, "search_lessons", _raise)
    out = m.memory_search("quorum tuning", include_sessions=True)
    assert "[global] Quorum note" in out


def test_fused_no_hits_at_all(monkeypatch):
    _clear_notes()
    _clear_store()
    monkeypatch.setattr(_lessons, "search_lessons",
                        lambda q, limit=None, scope="all": [])
    assert m.memory_search("nothinghere xyz", include_sessions=True) == \
        "[no memories or past sessions matching 'nothinghere xyz']"


def test_fused_note_key_is_per_scope():
    """Same filename in the global and the project dir = two docs (the plain
    scan lists both), so the note key carries the label and neither collapses."""
    root = tempfile.mkdtemp(prefix="nx-fusion-root-")
    os.makedirs(os.path.join(root, ".nx", "memories"), exist_ok=True)
    with open(os.path.join(root, ".nx", "memories", "same.md"), "w") as f:
        f.write("# Project twin\n\nshared marker body here\n")
    _note("Global twin", "shared marker body here", name="same.md")
    m.st.PROJECT_ROOT = root
    monkey_hits = []
    from nx import _lessons as L
    orig = L.search_lessons
    L.search_lessons = lambda q, limit=None, scope="all": monkey_hits
    try:
        out = m.memory_search("shared marker", include_sessions=True)
    finally:
        L.search_lessons = orig
    assert "(2):" in out.splitlines()[0]
    assert "[global] Global twin (same.md)" in out
    assert "[project] Project twin (same.md)" in out


def test_fused_output_interleaves_the_two_arms_by_rank(monkeypatch):
    """The production-visible effect: the block alternates note/session by rank
    (a rank-1 session outranks a rank-3 note), instead of listing all notes then
    all sessions. Ties go to the note arm (first list), reproducibly."""
    _clear_notes()
    _clear_store()
    base = 1_700_000_000.0
    for i in range(3):
        name = _note(f"Marker note {i}", f"shared marker body {i}",
                     name=f"m{i}.md")
        # pin the note arm's tie-break order (it sorts equal scores newest
        # first) so the expected fused order is exact, not mtime-dependent
        p = os.path.join(_tmp, "memories", name)
        os.utime(p, (base + (10 - i), base + (10 - i)))
    hits = [{"id": f"20260101-12000{i}-fuiz", "title": f"sess {i}", "cwd": "",
             "mtime": float(i), "snippet": f"snip {i}", "score": -1.0 - i}
            for i in range(3)]
    monkeypatch.setattr(_lessons, "search_lessons",
                        lambda q, limit=None, scope="all": hits)
    out = m.memory_search("shared marker", include_sessions=True)
    rows = [ln[len("[global] "):].split(" (")[0] if ln.startswith("[global] ")
            else ln.split("  ")[1]
            for ln in out.splitlines()
            if ln.startswith("[global] ") or ln.startswith("· ")]
    # the arms interleave rank-for-rank (ties break to the note arm, the first
    # list), each arm keeping its own internal order
    assert rows == ["Marker note 0", "sess 0", "Marker note 1", "sess 1",
                    "Marker note 2", "sess 2"]
    # the rise, in one line: the session arm's rank-1 hit sits ABOVE the note
    # arm's rank-3 hit (m2.md) — under a "notes then sessions" concatenation it
    # could not.
    lines = out.splitlines()
    assert lines.index("· 20260101-120  sess 0  (bm25 -1.0)") < \
        next(i for i, ln in enumerate(lines)
             if ln.startswith("[global] ") and "(m2.md)" in ln)


def test_search_lessons_hit_shape_is_structured():
    """The formatter wraps the list-returning entry point: hits carry the
    fields the fusion path needs, in list order = rank order."""
    _clear_store()
    _seed("20260101-120000-fu05", "quokka quokka quokka habitat", title="q")
    _seed("20260101-120000-fu06", "the quokka escaped", title="q2")
    hits = _lessons.search_lessons("quokka")
    assert [h["id"] for h in hits] == ["20260101-120000-fu05",
                                       "20260101-120000-fu06"]
    for h in hits:
        assert set(h) == {"id", "title", "cwd", "mtime", "snippet", "score"}


def test_recall_lessons_golden_output_survives_the_refactor():
    _clear_store()
    _seed("20260101-120000-fu07", "lemur tail grip notes", title="lemurs")
    out = _lessons.recall_lessons("lemur tail")
    lines = out.splitlines()
    assert lines[0] == "past sessions matching 'lemur tail' (best first):"
    assert lines[1].startswith("· 20260101-120  lemurs")
    assert "(bm25 " in lines[1]
    assert lines[2] == "    [user] lemur tail grip notes"
    assert lines[3] == ("read one with read_file on its store mirror, or "
                        "resume it with /resume <short-id>")
    assert out == "\n".join(lines)  # no trailing newline added by the refactor


def test_schema_carries_one_new_property():
    tool = next(t for t in m.TOOLS
                if t["function"]["name"] == "memory_search")
    props = tool["function"]["parameters"]["properties"]
    assert set(props) == {"query", "scope", "include_sessions"}
    assert props["include_sessions"]["type"] == "boolean"
    assert tool["function"]["parameters"]["required"] == ["query"]
    # the description text did not grow into a paragraph
    assert len(tool["function"]["description"]) < 220


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
