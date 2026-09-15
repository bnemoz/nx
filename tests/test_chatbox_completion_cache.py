#!/usr/bin/env python3
"""Regression coverage for the @-mention completion caches (PERF.md PF-01).

Before this cache `_match_file_paths` ran from inside `render()`, re-walking
the whole project tree (up to `_TREE_SCAN_CAP` entries) and re-running
`git rev-parse` on EVERY keystroke -- measured 38-56 ms/keystroke on a
19.7k-entry repo: the worst interactive number in the codebase. These tests
pin the three fixes: a cached tree, a memoized project root, and an
incremental filter; plus the structural invariant (I-1) that `render()` does
no filesystem/subprocess I/O.
"""
import ast
import os
import sys
import tempfile
import time

_tmp = tempfile.mkdtemp(prefix="nx-chatbox-cache-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nx import _chatbox, _risk  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_tree(root, n_dirs=20, n_files=50):
    """A synthetic repo wide enough that an uncached walk is measurable."""
    for d in range(n_dirs):
        sub = os.path.join(root, f"pkg{d:02d}")
        os.makedirs(sub, exist_ok=True)
        for f in range(n_files):
            with open(os.path.join(sub, f"mod{f:03d}.py"), "w") as fh:
                fh.write("# x\n")
    return root


def _with_root(root):
    """Context-manager-ish helper: monkeypatch the project root, return restore."""
    original = _risk._detect_project_root
    _risk._detect_project_root = lambda cwd=None: root
    _chatbox._bust_tree_cache()
    return original


def test_tree_walked_once_across_many_keystrokes():
    """The whole point: 10 keystrokes cost ONE tree walk, not ten."""
    root = _make_tree(os.path.join(_tmp, "once"))
    original = _with_root(root)
    calls = []
    real_walk = _chatbox._walk_tree

    def counting_walk(r):
        calls.append(r)
        return real_walk(r)

    _chatbox._walk_tree = counting_walk
    try:
        for prefix in ("", "p", "pk", "pkg", "pkg0", "pkg01", "m", "mo", "mod", "mod0"):
            _chatbox._match_file_paths(prefix)
    finally:
        _chatbox._walk_tree = real_walk
        _risk._detect_project_root = original
    assert len(calls) == 1, (
        f"_walk_tree ran {len(calls)}x across 10 keystrokes -- the tree cache "
        "is not holding")


def test_warm_keystrokes_are_fast():
    """A warm (cached) keystroke must not scale with repo size."""
    root = _make_tree(os.path.join(_tmp, "fast"), n_dirs=30, n_files=60)
    original = _with_root(root)
    try:
        _chatbox._match_file_paths("")          # cold: one walk
        t0 = time.perf_counter()
        for _ in range(20):
            _chatbox._match_file_paths("pkg")
        warm = (time.perf_counter() - t0) / 20
    finally:
        _risk._detect_project_root = original
    assert warm < 0.005, f"warm keystroke took {warm * 1000:.2f} ms (want < 5 ms)"


def test_incremental_filter_only_narrows_candidates():
    """A longer prefix filters the previous candidate set; the result must be
    identical to a cold, from-scratch scan (subsequence matching is monotone)."""
    root = _make_tree(os.path.join(_tmp, "incr"), n_dirs=8, n_files=20)
    original = _with_root(root)
    try:
        assert _chatbox._match_file_paths("m") == _chatbox._match_file_paths("mo")
        # warm the incremental cache with a short prefix, then extend it
        _chatbox._match_file_paths("m")
        incremental = _chatbox._match_file_paths("mod1")
        _chatbox._bust_tree_cache()             # forget everything
        cold = _chatbox._match_file_paths("mod1")
    finally:
        _risk._detect_project_root = original
    assert incremental == cold, "incremental filter disagreed with a cold scan"


def test_project_root_is_memoized():
    """`git rev-parse` (~2 ms) must not run per keystroke."""
    root = _make_tree(os.path.join(_tmp, "root"), n_dirs=2, n_files=2)
    original = _with_root(root)
    calls = []
    real = _risk._detect_project_root

    def counting(cwd=None):
        calls.append(cwd)
        return real(cwd)

    _risk._detect_project_root = counting
    try:
        for _ in range(5):
            _chatbox._match_file_paths("pkg")
    finally:
        _risk._detect_project_root = original
    assert len(calls) == 1, f"_detect_project_root ran {len(calls)}x -- not memoized"


def test_render_does_no_io():
    """I-1: `render()` must not walk the tree or shell out. Checked statically
    (driving the real key loop needs a TTY) so a future edit that moves the
    completion call back inside render() fails here."""
    src = open(os.path.join(_REPO, "nx", "_chatbox.py")).read()
    tree = ast.parse(src)
    render_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "render":
            render_node = node
            break
    assert render_node is not None, "_chatbox.py no longer defines render()"
    banned = {"_match_file_paths", "_walk_tree", "_cached_tree", "_project_root",
              "scandir", "run", "Popen", "check_output", "getcwd"}
    used = {
        n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
        for n in ast.walk(render_node) if isinstance(n, ast.Call)
    }
    hit = banned & used
    assert not hit, f"render() performs I/O via {sorted(hit)} -- move it to compute_matches()"


def test_repeat_prefix_replays_cached_result():
    """Redrawing with an unchanged buffer (arrow-key navigation) must not
    re-score the tree, and must still return the same candidates."""
    root = _make_tree(os.path.join(_tmp, "repeat"), n_dirs=3, n_files=3)
    original = _with_root(root)
    calls = []
    real_score = _chatbox._fuzzy_score

    def counting_score(query, text):
        calls.append(query)
        return real_score(query, text)

    _chatbox._fuzzy_score = counting_score
    try:
        first = _chatbox._match_file_paths("pk")
        n_after_first = len(calls)
        assert n_after_first > 0
        second = _chatbox._match_file_paths("pk")
    finally:
        _chatbox._fuzzy_score = real_score
        _risk._detect_project_root = original
    assert second == first, "repeat prefix returned different candidates"
    assert len(calls) == n_after_first, (
        "a repeated prefix re-scored the tree -- the result cache is not holding")


def test_bust_tree_cache_clears_everything():
    root = _make_tree(os.path.join(_tmp, "bust"), n_dirs=1, n_files=1)
    original = _with_root(root)
    try:
        _chatbox._match_file_paths("pkg")
        assert _chatbox._TREE_CACHE is not None
        assert _chatbox._TREE_RESULT_CACHE is not None
        _chatbox._bust_tree_cache()
    finally:
        _risk._detect_project_root = original
    assert _chatbox._TREE_CACHE is None
    assert _chatbox._TREE_FILTER_CACHE is None
    assert _chatbox._TREE_RESULT_CACHE is None
    assert _chatbox._PROJECT_ROOT_CACHE is None
