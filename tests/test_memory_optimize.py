#!/usr/bin/env python3
"""/memory optimize: the consolidation pass over the memory store.

Covers the parts that must not regress silently — the index ignoring the
archive dir, the line-protocol plan parser (unknown/ambiguous names dropped
per line, never a whole-pass failure), plan de-duplication across batches,
the archive-instead-of-delete posture, the merged note landing through the
shared writer, and the live transcript's memory note being rebuilt in place.

No live model endpoint: the librarian and merge round-trips are served by a
scripted fake client, and `nx._confirm` is stubbed (save/restored around every
test, same reason as test_memory_tools.py: it is a shared module global).
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-memopt-")
os.environ["NX_HOME"] = _tmp
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _memory_cmd, _memory_optimize as opt  # noqa: E402
from nx._openai_client import APIError as _APIError  # noqa: E402

# Captured stdout lines, for asserting on the printed explanations.
import io  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402


class _Captured(list):
    def __enter__(self):
        self._buf = io.StringIO()
        self._ctx = redirect_stdout(self._buf)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        self.extend(self._buf.getvalue().splitlines())
        return False


_printed = _Captured()

_confirm_calls = []


def _fake_confirm(action, preview=None):
    _confirm_calls.append(action)
    return True


_saved = {}


def _setup_function():
    _confirm_calls.clear()
    _saved["confirm"] = m._confirm
    _saved["home"] = m.st.SESSION_HOME
    _saved["root"] = m.st.PROJECT_ROOT
    m._confirm = _fake_confirm
    m.st.SESSION_HOME = _tmp


def _teardown_function():
    if _saved.get("confirm") is not None:
        m._confirm = _saved["confirm"]
    if _saved.get("home") is not None:
        m.st.SESSION_HOME = _saved["home"]
    if "root" in _saved:
        m.st.PROJECT_ROOT = _saved["root"]
    for k in ("NX_MEMORY_OPTIMIZE_BATCH", "NX_MEMORY_OPTIMIZE_MAX_NOTES",
              "NX_MEMORY_OPTIMIZE_MAX_TOKENS"):
        os.environ.pop(k, None)


setup_function = lambda _function: _setup_function()
teardown_function = lambda _function: _teardown_function()


# --- helpers -----------------------------------------------------------------

def _note(title, body, scope="project", root=None):
    """Write one memory through the shared writer with a controlled root."""
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root if root is not None else _saved["root"]
    try:
        return _memory_cmd._write_memory_file(title, body, scope)[0]
    finally:
        m.st.PROJECT_ROOT = old


def _md_files(d):
    try:
        return sorted(f for f in os.listdir(d) if f.endswith(".md"))
    except OSError:
        return []


def _env(name, value):
    os.environ[name] = str(value)


# --- index -------------------------------------------------------------------

def test_collect_notes_indexes_both_scopes_and_ignores_archive():
    root = tempfile.mkdtemp(prefix="nx-opt-idx-")
    m.st.PROJECT_ROOT = root
    _note("Project note", "project body", scope="project", root=root)
    _note("Global note", "global body", scope="global")
    notes = opt._collect_notes("project")
    titles = {n.title for n in notes}
    assert titles == {"Project note", "Global note"}, titles
    assert {n.label for n in notes} == {"project", "global"}
    assert all(n.digest for n in notes)

    # An archived note is invisible to the scanner (that is the undo story).
    first = notes[0]
    src = os.path.join(first.dir, first.name)
    dst_dir = os.path.join(first.dir, "archive", "20250101-000000")
    os.makedirs(dst_dir, exist_ok=True)
    os.replace(src, os.path.join(dst_dir, first.name))
    remaining = opt._collect_notes("project")
    assert first.name not in {n.name for n in remaining}
    assert os.path.exists(os.path.join(dst_dir, first.name))


def test_digest_drops_title_and_footer():
    content = "# Title Here\n\nfirst fact about `nx/_tools.py`\n\n---\n_scope: project  _saved: 2025-01-01 00:00:00\n"
    d = opt._digest(content)
    assert "Title Here" not in d
    assert "_scope:" not in d
    assert "first fact" in d


# --- plan parsing ------------------------------------------------------------

def _known(*names):
    now = 1_700_000_000.0
    return {n: opt._OptNote(n, n.replace(".md", "").split("_", 1)[-1], "project",
                            "/tmp", now, 100, "gist") for n in names}


def test_parse_plan_accepts_line_protocol_and_ignores_chatter():
    k = _known("20250101-000000_alpha.md", "20250102-000000_beta.md",
               "20250103-000000_gamma.md")
    retire, merges, skipped = opt._parse_plan(
        "Here is my plan:\n"
        "RETIRE 20250101-000000_alpha.md\n"
        "- MERGE 20250102-000000_beta.md + 20250103-000000_gamma.md => Combined topic\n"
        "```\n", k)
    assert retire == ["20250101-000000_alpha.md"], retire
    assert merges == [(["20250102-000000_beta.md", "20250103-000000_gamma.md"],
                       "Combined topic")], merges
    assert skipped == []


def test_parse_plan_names_may_drop_the_extension():
    k = _known("20250101-000000_alpha.md")
    retire, _merges, skipped = opt._parse_plan("RETIRE 20250101-000000_alpha", k)
    assert retire == ["20250101-000000_alpha.md"]
    assert skipped == []


def test_parse_plan_drops_unknown_and_ambiguous_names_per_line():
    k = _known("20250101-000000_alpha.md", "20250101-000001_alpha2.md",
               "20250102-000000_beta.md")
    retire, merges, skipped = opt._parse_plan(
        "RETIRE does-not-exist.md\n"
        "RETIRE 20250101-00000_  # ambiguous prefix over two notes\n"
        "MERGE does-not-exist.md + 20250102-000000_beta.md => Nope\n"
        "RETIRE 20250102-000000_beta.md\n", k)
    assert skipped and len(skipped) == 3, skipped
    assert retire == ["20250102-000000_beta.md"], retire
    assert merges == []


# --- plan merging ------------------------------------------------------------

def test_merge_plan_keeps_each_note_in_one_action_and_respects_budget():
    p = opt._Plan([], [])
    p = opt._merge_plan(p, (["a.md", "b.md"], []), budget=8)
    # A later merge naming a retired note wins; the retire is dropped.
    p = opt._merge_plan(p, (["a.md"], [ (["a.md", "c.md"], "T") ]), budget=8)
    assert p.retire == ["b.md"], p.retire
    assert p.merges == [(["a.md", "c.md"], "T")], p.merges
    # A note already in a merge can't join a second one, and a MERGE left with
    # fewer than two free members is dropped (its notes are simply untouched).
    p2 = opt._merge_plan(p, ([], [(["a.md", "d.md"], "X"), (["e.md"], "Y")]), budget=8)
    assert p2.merges == [(["a.md", "c.md"], "T")], p2.merges
    assert p2.retire == ["b.md"], p2.retire
    # Budget of 0 blocks new merges but keeps retires.
    p3 = opt._merge_plan(p, ([], [(["d.md", "e.md"], "Z")]), budget=0)
    assert p3.merges == [(["a.md", "c.md"], "T")], p3.merges


def test_optimize_batch_and_cap_env():
    os.environ.pop("NX_MEMORY_OPTIMIZE_BATCH", None)
    os.environ.pop("NX_MEMORY_OPTIMIZE_MAX_NOTES", None)
    assert opt._optimize_batch(100) == opt.OPTIMIZE_DEFAULT_BATCH
    assert opt._optimize_batch(3) == 3          # batch larger than the store
    _env("NX_MEMORY_OPTIMIZE_BATCH", 0)
    assert opt._optimize_batch(50) == 50        # 0 = single pass
    assert opt._optimize_max_notes() == opt.OPTIMIZE_DEFAULT_MAX_NOTES
    _env("NX_MEMORY_OPTIMIZE_MAX_NOTES", 0)
    assert opt._optimize_max_notes() == 0       # 0 = everything


def test_group_scope_majority_and_tie():
    def n(label):
        return opt._OptNote("x.md", "t", label, "/tmp", 0.0, 1, "g")
    assert opt._group_scope([n("project"), n("global"), n("global")]) == "global"
    assert opt._group_scope([n("project"), n("global")]) == "project"
    assert opt._group_scope([n("global"), n("project"), n("project")]) == "project"


# --- fake model ---------------------------------------------------------------

class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type(
            "M", (), {"content": content})()})()]
        self.usage = None

    def model_dump(self):
        return {}


class _FakeOptClient:
    """Answers librarian prompts with `plan` and merge prompts with `merged`."""
    base_url = "http://localhost:9999/v1"

    def __init__(self, plan, merged):
        self.plan = plan
        self.merged = merged
        self.prompts = []

    def _chat(self):
        client = self

        class _Completions:
            def create(self, **kwargs):
                prompt = kwargs["messages"][0]["content"]
                client.prompts.append(prompt)
                if prompt.startswith("Merge the memory notes"):
                    return _Resp(client.merged)
                return _Resp(client.plan)

        class _Chat:
            completions = _Completions()

        return _Chat()

    @property
    def chat(self):
        return self._chat()


def _install_client(client):
    m.st.client = client
    m.st.MODEL = "test-model"
    m.st.ACTIVE = None
    m.st._METRICS = None


def _with_store(notes, root):
    """notes: [(title, body)] written newest-last-first (all distinct slugs)."""
    m.st.PROJECT_ROOT = root
    paths = [_note(t, b, scope="project", root=root) for t, b in notes]
    return paths


# --- end-to-end ---------------------------------------------------------------

def test_optimize_merges_retires_archives_and_refreshes_the_note():
    root = tempfile.mkdtemp(prefix="nx-opt-run-")
    pdir = os.path.join(root, ".nx", "memories")
    p1 = _with_store([
        ("Build caching", "body one about caching in `nx/_tool_cache.py`"),
        ("Build cache gotcha", "body two, same topic, older"),
        ("Scratch session log", "we ran pytest and it passed"),
    ], root)
    assert len(_md_files(pdir)) == 3
    names = sorted(os.path.basename(p) for p in p1)
    # Names sort newest-first; the two build notes are the two newest writes.
    plan = (f"RETIRE {names[2]}\n"
            f"MERGE {names[0]} + {names[1]} => Build caching in nx\n")
    client = _FakeOptClient(plan, "# Build caching in nx\n\nthe merged facts\n")
    _install_client(client)
    _env("NX_MEMORY_OPTIMIZE_BATCH", 0)
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "work"},
                {"role": "assistant", "content": "done"}]
    _memory_cmd._add_memory_note_if_absent(messages)
    assert any("[Saved memories" in mm.get("content", "") for mm in messages if isinstance(mm.get("content"), str))

    opt._cmd_memory_optimize(messages, "")

    after = _md_files(pdir)
    assert len(after) == 1, after                      # only the merged note remains
    with open(os.path.join(pdir, after[0]), encoding="utf-8") as f:
        merged_body = f.read()
    assert merged_body.startswith("# Build caching in nx\n\n"), merged_body
    assert "the merged facts" in merged_body
    assert "_scope: project" in merged_body            # shared writer's footer
    # Every source note is recoverable under archive/<stamp>/.
    arch_root = os.path.join(pdir, "archive")
    stamps = os.listdir(arch_root)
    assert len(stamps) == 1, stamps
    archived = _md_files(os.path.join(arch_root, stamps[0]))
    assert sorted(archived) == names, archived
    # The live transcript's memory note now lists only the merged title.
    note_lines = [mm["content"] for mm in messages
                  if isinstance(mm.get("content"), str) and "[Saved memories" in mm["content"]]
    assert len(note_lines) == 1, messages
    assert "Build caching in nx" in note_lines[0]
    assert "Scratch session log" not in note_lines[0]
    # Approval was asked exactly once.
    assert len(_confirm_calls) == 1, _confirm_calls


def test_optimize_dry_run_changes_nothing():
    root = tempfile.mkdtemp(prefix="nx-opt-dry-")
    pdir = os.path.join(root, ".nx", "memories")
    p1 = _with_store([("Alpha note", "alpha body"), ("Beta note", "beta body")], root)
    names = sorted(os.path.basename(p) for p in p1)
    client = _FakeOptClient(f"MERGE {names[0]} + {names[1]} => Combined\n", "unused")
    _install_client(client)
    _env("NX_MEMORY_OPTIMIZE_BATCH", 0)

    opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "dry")

    assert sorted(_md_files(pdir)) == names            # untouched
    assert not os.path.exists(os.path.join(pdir, "archive"))
    assert _confirm_calls == []                        # dry never asks


def test_optimize_denied_approval_writes_nothing():
    root = tempfile.mkdtemp(prefix="nx-opt-deny-")
    pdir = os.path.join(root, ".nx", "memories")
    p1 = _with_store([("Alpha note", "alpha body"), ("Beta note", "beta body")], root)
    names = sorted(os.path.basename(p) for p in p1)
    client = _FakeOptClient(f"RETIRE {names[0]}\n", "unused")
    _install_client(client)
    _env("NX_MEMORY_OPTIMIZE_BATCH", 0)
    m._confirm = lambda action, preview=None: False

    opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")

    assert sorted(_md_files(pdir)) == names
    assert not os.path.exists(os.path.join(pdir, "archive"))


def test_optimize_librarian_ok_leaves_store_alone():
    root = tempfile.mkdtemp(prefix="nx-opt-ok-")
    pdir = os.path.join(root, ".nx", "memories")
    p1 = _with_store([("Alpha note", "alpha body"), ("Beta note", "beta body")], root)
    _install_client(_FakeOptClient("OK", "unused"))
    _env("NX_MEMORY_OPTIMIZE_BATCH", 0)
    opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")
    assert sorted(_md_files(pdir)) == sorted(os.path.basename(p) for p in p1)
    assert _confirm_calls == []


def test_optimize_single_note_store_short_circuits_without_a_call():
    root = tempfile.mkdtemp(prefix="nx-opt-one-")
    _with_store([("Only note", "only body")], root)
    client = _FakeOptClient("OK", "unused")
    _install_client(client)
    opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")
    assert client.prompts == []           # nothing to compare against: no round-trip


def test_optimize_batches_and_reports_unreviewed_notes():
    root = tempfile.mkdtemp(prefix="nx-opt-cap-")
    _with_store([(f"Note {i}", f"body {i} about something distinct") for i in range(6)], root)
    client = _FakeOptClient("OK", "unused")
    _install_client(client)
    _env("NX_MEMORY_OPTIMIZE_BATCH", 2)
    _env("NX_MEMORY_OPTIMIZE_MAX_NOTES", 4)
    opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")
    # 4 reviewed / batch 2 = two librarian passes, and the 2 older notes are
    # counted as unreviewed rather than silently skipped.
    assert len(client.prompts) == 2, client.prompts
    assert all("index" in p for p in client.prompts)


def test_optimize_without_a_source_aborts_before_writing():
    root = tempfile.mkdtemp(prefix="nx-opt-nosrc-")
    pdir = os.path.join(root, ".nx", "memories")
    p1 = _with_store([("Alpha note", "alpha body"), ("Beta note", "beta body")], root)
    old = (m.st.client, m.st.MODEL, m.st.ACTIVE, m.st._METRICS)
    m.st.client, m.st.MODEL, m.st.ACTIVE, m.st._METRICS = None, None, None, None
    try:
        opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")
    finally:
        m.st.client, m.st.MODEL, m.st.ACTIVE, m.st._METRICS = old
    assert sorted(_md_files(pdir)) == sorted(os.path.basename(p) for p in p1)


def test_optimize_drops_cached_memory_search_results():
    root = tempfile.mkdtemp(prefix="nx-opt-cache-")
    p1 = _with_store([("Alpha note", "alpha body"), ("Beta note", "beta body")], root)
    names = sorted(os.path.basename(p) for p in p1)
    _install_client(_FakeOptClient(f"RETIRE {names[0]}\n", "unused"))
    _env("NX_MEMORY_OPTIMIZE_BATCH", 0)
    m.st._TOOL_RESULT_CACHE[("memory_search", "alpha", "project")] = object()
    opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")
    assert not [k for k in m.st._TOOL_RESULT_CACHE if k[0] == "memory_search"]


def test_optimize_degenerate_response_aborts_instead_of_crashing():
    """Empty `choices` (some providers' error payloads) must print an error and
    stop the run, not raise IndexError out of the slash command into the REPL."""
    root = tempfile.mkdtemp(prefix="nx-opt-degenerate-")
    pdir = os.path.join(root, ".nx", "memories")
    p1 = _with_store([("Alpha note", "alpha body"), ("Beta note", "beta body")], root)
    names = sorted(os.path.basename(p) for p in p1)

    class _EmptyResp:
        choices = []           # no message at all
        usage = None

        def model_dump(self):
            return {}

    class _DegenerateClient(_FakeOptClient):
        def _chat(self):
            client = self

            class _Completions:
                def create(self, **kwargs):
                    prompt = kwargs["messages"][0]["content"]
                    client.prompts.append(prompt)
                    return _EmptyResp()

            class _Chat:
                completions = _Completions()

            return _Chat()

    client = _DegenerateClient("unused", "unused")
    _install_client(client)
    _env("NX_MEMORY_OPTIMIZE_BATCH", 0)
    opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")   # must not raise
    assert sorted(_md_files(pdir)) == names        # nothing written, nothing archived
    assert _confirm_calls == []                    # never reached the gate


def test_collect_notes_skips_non_utf8_notes():
    """One hand-edited latin-1 note must not take the whole pass down — the
    scanner skips it like an unreadable file (regression: UnicodeDecodeError
    used to propagate out of _collect_notes into the REPL)."""
    root = tempfile.mkdtemp(prefix="nx-opt-latin1-")
    pdir = os.path.join(root, ".nx", "memories")
    m.st.PROJECT_ROOT = root
    _note("Good note", "good body", scope="project", root=root)
    with open(os.path.join(pdir, "20250101-000000_bad.md"), "wb") as f:
        f.write(b"# Caf\xe9 note\n\ncaf\xe9 body\n")
    notes = opt._collect_notes("project")          # must not raise
    assert {n.title for n in notes} == {"Good note"}, notes


def test_split_collisions_sets_both_scoped_names_aside():
    now = 1_700_000_000.0
    rows = [("a", "project"), ("b", "global"), ("a", "global"),
            ("c", "project"), ("b", "project"), ("b", "global")]
    entries = [opt._OptNote(f"{stem}.md", "t", label, "/tmp", now, 1, "g")
               for stem, label in rows]
    keep, collide = opt._split_collisions(entries)
    assert collide == ["a.md", "b.md"], collide
    assert [k.name for k in keep] == ["c.md"], keep


def test_optimize_leaves_cross_scope_name_collisions_alone():
    """A plan line naming a note that exists in both scopes must not silently
    archive one of them: the run warns and the colliding names sit it out."""
    root = tempfile.mkdtemp(prefix="nx-opt-collide-")
    pdir = os.path.join(root, ".nx", "memories")
    gdir = os.path.join(m.st.SESSION_HOME, "memories")
    paths = _with_store([
        ("Alpha note", "alpha body"), ("Beta note", "beta body"),
    ], root)
    # Hand-copy one project note into the global store under the same name.
    twin = os.path.join(gdir, os.path.basename(paths[0]))
    os.makedirs(gdir, exist_ok=True)
    with open(paths[0], "rb") as f:
        data = f.read()
    with open(twin, "wb") as f:
        f.write(data.replace(b"_scope: project", b"_scope: global"))

    plan = f"RETIRE {os.path.basename(paths[0])}\n"
    client = _FakeOptClient(plan, "unused")
    _install_client(client)
    _env("NX_MEMORY_OPTIMIZE_BATCH", 0)
    try:
        opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")
        assert os.path.exists(paths[0])           # both colliding copies untouched
        assert os.path.exists(twin)
        assert not os.path.exists(os.path.join(pdir, "archive"))
        assert _confirm_calls == []                # never reached the gate
        # The colliding name never reached a librarian prompt.
        assert all(os.path.basename(paths[0]) not in p for p in client.prompts)
    finally:
        # The twin is test pollution in the shared global store — remove it so
        # no later test's store-empty assertions can see it (xdist-safe).
        try:
            os.unlink(twin)
        except OSError:
            pass


# --- carryover / footer / overflow (0.33.0) ------------------------------------

def test_strip_footer_removes_only_the_writers_tail():
    body = "# T\n\nfact line\n\n---\n_scope: project  _saved: 2025-01-01 00:00:00\n"
    out = opt._strip_footer(body)
    assert out == "# T\n\nfact line\n", out
    # No footer shape -> untouched (plus trailing-newline normalization).
    plain = "# T\n\nfinal line\n"
    assert opt._strip_footer(plain) == "# T\n\nfinal line\n"
    # A mid-body horizontal rule is not a footer and survives.
    mid = "# T\n\n---\n\nnot a footer\n"
    assert "not a footer" in opt._strip_footer(mid)


def test_merge_prompt_strips_source_footers():
    root = tempfile.mkdtemp(prefix="nx-opt-merge-")
    pdir = os.path.join(root, ".nx", "memories")
    _with_store([("Alpha note", "alpha body"), ("Beta note", "beta body")], root)
    notes = opt._collect_notes("project")
    prompt = opt._merge_prompt(notes)
    assert "_scope:" not in prompt and "_saved:" not in prompt, prompt
    assert "alpha body" in prompt and "beta body" in prompt


def test_plan_prompt_carries_previous_batch_titles():
    root = tempfile.mkdtemp(prefix="nx-opt-carry-")
    _with_store([("A one", "body a"), ("B two", "body b"), ("C three", "body c")], root)
    notes = opt._collect_notes("project")
    later = notes[1:]
    carry = notes[:1]
    with_carry = opt._plan_prompt(later, "", carry)
    without = opt._plan_prompt(later, "")
    assert "carryover from the previous batch" in with_carry
    assert "carryover" not in without
    assert carry[0].name in with_carry          # titles only, still no digest
    assert carry[0].digest not in with_carry


def test_optimize_overflow_reports_groups_past_the_cap(monkeypatch):
    root = tempfile.mkdtemp(prefix="nx-opt-overflow-")
    _with_store([(f"Note {i}", f"body {i} distinct") for i in range(8)], root)
    names = sorted(os.path.basename(p) for p in
                   [os.path.join(root, ".nx", "memories", f) for f in
                    os.listdir(os.path.join(root, ".nx", "memories"))])
    # Three viable groups, cap 2: two applied, one reported.
    plan = (f"MERGE {names[0]} + {names[1]} => First merged\n"
            f"MERGE {names[2]} + {names[3]} => Second merged\n"
            f"MERGE {names[4]} + {names[5]} => Third merged\n")
    client = _FakeOptClient(plan, "# Merged\n\nbody\n")
    _install_client(client)
    _env("NX_MEMORY_OPTIMIZE_BATCH", 0)
    monkeypatch.setattr(opt, "_MAX_MERGES_PER_RUN", 2)
    opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "dry")
    assert len(client.prompts) == 1


def test_merge_plan_counts_overflow_past_budget():
    p = opt._Plan([], [])
    p = opt._merge_plan(p, ([], [(["a.md", "b.md"], "T")]), budget=1)
    assert p.merges == [(["a.md", "b.md"], "T")] and p.overflow == 0
    p = opt._merge_plan(p, ([], [(["c.md", "d.md"], "U"), (["e.md", "f.md"], "V")]), budget=1)
    assert [t for _n, t in p.merges] == ["T"]
    assert p.overflow == 2


# --- partial run / escape / focus (0.33.1) --------------------------------------

class _FailingPlanClient(_FakeOptClient):
    """Answers pass 1 with `plan`, then fails every later librarian call."""
    base_url = "http://localhost:9999/v1"

    def __init__(self, plan, merged):
        super().__init__(plan, merged)
        self.calls = 0

    def _chat(self):
        client = self

        class _Completions:
            def create(self, **kwargs):
                client.calls += 1
                prompt = kwargs["messages"][0]["content"]
                client.prompts.append(prompt)
                if prompt.startswith("Merge the memory notes"):
                    return _Resp(client.merged)
                if client.calls > 1:
                    raise RuntimeError("endpoint lost")
                return _Resp(client.plan)

        class _Chat:
            completions = _Completions()

        return _Chat()


def test_optimize_partial_run_applies_what_earlier_passes_found():
    """A later librarian pass dying mid-run must not lose the earlier passes'
    findings: the run warns and applies the partial plan."""
    root = tempfile.mkdtemp(prefix="nx-opt-partial-")
    pdir = os.path.join(root, ".nx", "memories")
    paths = _with_store([(f"Note {i}", f"body {i}") for i in range(4)], root)
    names = sorted(os.path.basename(p) for p in paths)
    plan = f"RETIRE {names[0]}\n"
    client = _FailingPlanClient(plan, "unused")
    _install_client(client)
    _env("NX_MEMORY_OPTIMIZE_BATCH", 2)
    opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")
    assert client.calls == 2                       # pass 2 raised, run stopped there
    after = _md_files(pdir)
    assert names[0] not in after, after            # the partial plan was applied
    arch = os.path.join(pdir, "archive")
    assert os.path.isdir(arch) and _md_files(os.path.join(arch, os.listdir(arch)[0])) == [names[0]]


def test_optimize_empty_answer_reports_reason_and_keeps_notes():
    """An HTTP-200 answer with no visible text (a reasoning model cut off by
    the token cap: content=None, finish_reason="length") must not raise, must
    print *why*, and must leave the merge sources untouched."""
    root = tempfile.mkdtemp(prefix="nx-opt-empty-")
    pdir = os.path.join(root, ".nx", "memories")
    paths = _with_store([("Alpha note", "alpha body"), ("Beta note", "beta body")], root)
    names = sorted(os.path.basename(p) for p in paths)

    class _Msg:
        content = None
        reasoning_content = "thinking and thinking, but never answering"

    class _Choice:
        message = _Msg()
        finish_reason = "length"

    class _Resp:
        choices = [_Choice()]
        usage = None

        def model_dump(self):
            return {}

    class _Completions:
        @staticmethod
        def create(**kwargs):
            assert kwargs["max_tokens"] == 36 * 1024    # the tunable rides along
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _CutOffClient:
        base_url = "http://localhost:9999/v1"
        chat = _Chat()

    _install_client(_CutOffClient())
    _env("NX_MEMORY_OPTIMIZE_BATCH", 0)
    with _printed as lines:
        opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")
    # Nothing written, nothing archived; the empty answer was explained.
    assert sorted(_md_files(pdir)) == names
    assert not os.path.exists(os.path.join(pdir, "archive"))
    assert any("finish_reason=length" in ln for ln in lines), lines


def test_optimize_max_tokens_env_is_sent_and_zero_disables_it():
    root = tempfile.mkdtemp(prefix="nx-opt-mt-")
    _with_store([("Alpha note", "alpha body"), ("Beta note", "beta body")], root)
    seen = []

    class _Completions:
        @staticmethod
        def create(**kwargs):
            seen.append(kwargs.get("max_tokens"))
            return _Resp("OK")

    class _Chat:
        completions = _Completions()

    class _MTClient:
        base_url = "http://localhost:9999/v1"
        chat = _Chat()

    _install_client(_MTClient())
    _env("NX_MEMORY_OPTIMIZE_BATCH", 0)
    _env("NX_MEMORY_OPTIMIZE_MAX_TOKENS", 1234)
    opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")
    assert seen and all(v == 1234 for v in seen), seen

    seen.clear()
    _env("NX_MEMORY_OPTIMIZE_MAX_TOKENS", 0)
    opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")
    assert seen and all(v is None for v in seen), seen


def test_optimize_4xx_on_max_tokens_is_retried_without_it():
    root = tempfile.mkdtemp(prefix="nx-opt-4xx-")
    pdir = os.path.join(root, ".nx", "memories")
    paths = _with_store([("Alpha note", "alpha body"), ("Beta note", "beta body")], root)
    names = sorted(os.path.basename(p) for p in paths)
    seen = []

    class _Completions:
        @staticmethod
        def create(**kwargs):
            if "max_tokens" in kwargs:
                raise _APIError("max_tokens not supported", status_code=400)
            seen.append("plain")
            return _Resp("OK")

    class _Chat:
        completions = _Completions()

    class _PickyClient:
        base_url = "http://localhost:9999/v1"
        chat = _Chat()

    _install_client(_PickyClient())
    _env("NX_MEMORY_OPTIMIZE_BATCH", 0)
    with _printed:
        opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")
    assert seen == ["plain"]                    # retried without the kwarg
    assert sorted(_md_files(pdir)) == names


def test_optimize_first_pass_failure_with_empty_plan_returns_quietly():
    root = tempfile.mkdtemp(prefix="nx-opt-firstfail-")
    pdir = os.path.join(root, ".nx", "memories")
    paths = _with_store([(f"Note {i}", f"body {i}") for i in range(4)], root)
    names = sorted(os.path.basename(p) for p in paths)
    client = _FailingPlanClient("OK", "unused")    # pass 1 OK, pass 2 raises
    client.plan = "OK"
    _install_client(client)
    _env("NX_MEMORY_OPTIMIZE_BATCH", 2)
    opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")
    # Pass 1 found nothing and pass 2 failed -> quiet return, nothing touched.
    assert sorted(_md_files(pdir)) == names
    assert _confirm_calls == []


def test_confirm_optimize_escape_returns_false():
    """Esc at the approval prompt must cancel the run, not propagate into the
    REPL."""
    root = tempfile.mkdtemp(prefix="nx-opt-esc-")
    pdir = os.path.join(root, ".nx", "memories")
    paths = _with_store([("Alpha note", "alpha body"), ("Beta note", "beta body")], root)
    names = sorted(os.path.basename(p) for p in paths)
    _install_client(_FakeOptClient(f"RETIRE {names[0]}\n", "unused"))
    _env("NX_MEMORY_OPTIMIZE_BATCH", 0)

    class _Esc(BaseException):
        pass

    def _esc(action, preview=None):
        raise _Esc()

    saved = m._EscToChat
    m._EscToChat = _Esc
    m._confirm = _esc
    try:
        opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")  # no raise
    finally:
        m._EscToChat = saved
        m._confirm = _fake_confirm
    assert sorted(_md_files(pdir)) == names
    assert not os.path.exists(os.path.join(pdir, "archive"))


def test_plan_prompt_focus_guidance_and_unreviewed_print():
    root = tempfile.mkdtemp(prefix="nx-opt-focus-")
    _with_store([(f"Note {i}", f"body {i}") for i in range(5)], root)
    notes = opt._collect_notes("project")
    prompt = opt._plan_prompt(notes[:2], "focus on the build system notes", None)
    assert "The owner adds this guidance: focus on the build system notes" in prompt


# --- the in-context memory note ----------------------------------------------

def test_refresh_memory_note_replaces_in_place_and_drops_when_empty():
    root = tempfile.mkdtemp(prefix="nx-opt-note-")
    m.st.PROJECT_ROOT = root
    _note("Kept note", "kept body", scope="project", root=root)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    _memory_cmd._add_memory_note_if_absent(msgs)
    at = _memory_cmd._find_memory_note_index(msgs)
    assert at == 1, msgs
    assert len(msgs) == 3

    assert _memory_cmd._refresh_memory_note(msgs) is False   # unchanged store
    _note("Second note", "another body", scope="project", root=root)
    assert _memory_cmd._refresh_memory_note(msgs) is True
    assert _memory_cmd._find_memory_note_index(msgs) == at   # same index, in place
    assert "Second note" in msgs[at]["content"]
    assert len(msgs) == 3

    for f in _md_files(os.path.join(root, ".nx", "memories")):
        os.unlink(os.path.join(root, ".nx", "memories", f))
    assert _memory_cmd._refresh_memory_note(msgs) is True
    assert _memory_cmd._find_memory_note_index(msgs) is None
    assert len(msgs) == 2


# --- registration ------------------------------------------------------------

def test_optimize_registered_everywhere():
    from nx._repl import _REPL_COMMANDS
    from nx._chatbox import SLASH_ARG_OPTIONS
    from nx._repl_settings_cmds import _HELP_COMMANDS
    assert _REPL_COMMANDS.get("/memory") is not None
    assert "optimize" in SLASH_ARG_OPTIONS["/memory"], SLASH_ARG_OPTIONS["/memory"]
    assert any(c[0].startswith("/memory") and "optimize" in c[0] for c in _HELP_COMMANDS)
    # dispatch reaches the new branch (sub-name parsed inside the handler)
    import inspect
    src = inspect.getsource(_memory_cmd._cmd_memory)
    assert '"optimize"' in src


def test_usage_text_matches_documented_subcommands():
    import inspect
    src = inspect.getsource(_memory_cmd._cmd_memory)
    for sub in ("save", "remember", "delete", "list", "optimize"):
        assert f"/memory {sub}" in src, sub


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
