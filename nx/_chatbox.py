"""Multi-line chatbox input, replacing the bare `input()` prompt with a
framed, multi-line editor:
  - Enter submits, Shift+Enter (or Ctrl+J) inserts a newline
  - Paste (bracketed-paste mode) inserts its text verbatim -- newlines stay,
    a trailing newline at the end of paste is stripped so pasting never
    accidentally submits
  - Up/Down navigate history, Left/Right move within the current line,
    Home/End jump to line start/end, Ctrl+U clears the line, Ctrl+C cancels,
    Ctrl+L is shorthand for /clear (reset the conversation), Ctrl+Q for /exit,
    F1 for /help, Ctrl+S is shorthand for /compress
  - Shift+Tab toggles YOLO mode (mirrors /yolo), shown live in the box border
  - Long lines word-wrap visually inside the box; the buffer stays one
    logical string (newlines preserved) so the model sees the real text
  - Typing `@` starts a fuzzy file-path suggestion dropdown (Tab to accept);
    the actual mention expansion into file contents happens in
    nx/_mentions.py when the message is submitted
Falls back to plain `input()` when stdin/stdout is not a TTY.
"""
from __future__ import annotations

import heapq
import os
import shutil
import signal
import sys
import threading
import time
import unicodedata
from typing import Callable, cast

from . import _state as st
from ._ansi import DIM, RESET, C, THEMES, raw_mode, strip_ansi_for_display
from ._tty import read_byte, wait_ready

# (command, description) — kept in sync with the README "Commands" table.
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "list all slash commands"),
    ("/init", "generate/update AGENTS.md for this project"),
    ("/source", "list sources, switch to one, or override its model"),
    ("/theme", "list color themes, or switch to one"),
    ("/provider", "show or set OpenRouter provider-routing order"),
    ("/budget", "show or set the reasoning token budget"),
    ("/yolo", "toggle auto-approve for writes and bash"),
    ("/resources", "show this nx process's CPU% and RSS memory"),
    ("/context", "context breakdown by message; pin/unpin turns against /compress"),
    ("/cost", "session usage + per-source cost scoreboard"),
    ("/status", "snapshot: source, model, caps, budgets, fence, approval"),
    ("/bg", "list this session's background run_bash jobs and sub-agents, or /bg kill <pid>; /bg all lists every session's"),
    ("/tabs", "live dashboard of running surfaces (main turn, bg jobs, agents); Ctrl-T"),
    ("/approval", "show or set risk threshold (all/low/medium/high/yolo)"),
    ("/sessions", "list recent sessions, or show one in full"),
    ("/export", "write the current session to a markdown file"),
    ("/resume", "resume a past session (n/id/prefix/title)"),
    ("/save", "save the current session (optional custom title)"),
    ("/fork", "branch the conversation into a new session (original kept)"),
    ("/memory", "manage persistent memory"),
    ("/delete", "delete a saved session"),
    ("/todo", "show the model's current todo_write list"),
    ("/wip", "snapshot work state (branch, dirty files, todos, note) into the session"),
    ("/plan", "start/view/exit plan mode: /plan <goal>, /plan, /plan done|off|discard"),
    ("/verify", "set / clear / force-run the post-edit verify command (NX_VERIFY_CMD)"),
    ("/compress", "summarize older turns into one, keep last 2 verbatim"),
    ("/compact", "alias for /compress"),
    ("/autocompress", "show or set the auto-compress threshold"),
    ("/recover", "force a low-temp visible checkpoint after a bad stream"),
    ("/reasoning", "on|off|last — toggle or show reasoning trace"),
    ("/tooloutput", "on|off|last — toggle or show tool-call result"),
    ("/tools", "legacy alias for /tooloutput"),
    ("/toolcalls", "tool-calling mode + breaker: text|native|auto, reset [tool], or bare to show"),
    ("/undo", "revert the last write_file/edit_file/multi_edit"),
    ("/reset", "clear conversation, start a fresh session"),
    ("/clear", "alias for /reset"),
    ("/new", "alias for /reset"),
    ("/quit", "exit"),
    ("/exit", "alias for /quit"),
]


# Valid keyword arguments for slash commands that take an enum-like first
# argument -- surfaced as a dropdown once the user types "/cmd " (mirrors the
# command autocomplete above, kept in sync with the handlers in _repl.py).
# A value may be a static list, or a zero-arg callable for options that can
# only be known at runtime (e.g. "/source"'s configured source names).
SLASH_ARG_OPTIONS: dict[str, list[str] | "Callable[[], list[str]]"] = {
    "/theme": sorted(THEMES),
    "/reasoning": ["on", "off", "last"],
    "/tooloutput": ["on", "off", "last"],
    "/tools": ["on", "off", "last"],  # legacy alias arg dropdown
    "/toolcalls": ["reset", "text", "native", "auto"],
    "/approval": ["all", "low", "medium", "high", "yolo"],
    "/autocompress": ["off", "on"],
    "/budget": ["off"],
    "/bg": ["kill", "all"],
    "/verify": ["off", "run"],
    "/context": ["pin", "unpin", "clear"],
    "/plan": ["done", "off", "discard"],
    "/memory": ["save", "remember", "delete", "list", "archived", "optimize", "reflect"],
    "/export": ["md"],
    "/source": lambda: list(st.SOURCE_ORDER),
}


def _fuzzy_score(query: str, text: str) -> float | None:
    """Subsequence match of `query` in `text`; None if not a subsequence,
    else a score rewarding contiguous runs and an early start."""
    qi = 0
    score = 0
    last = -1
    first = None
    for i, ch in enumerate(text):
        if qi < len(query) and ch == query[qi]:
            if first is None:
                first = i
            score += 2 if last == i - 1 else 1
            last = i
            qi += 1
    if qi < len(query):
        return None
    return score - (first or 0) * 0.1


def _match_slash_commands(prefix: str) -> list[tuple[str, str]]:
    """`prefix` includes the leading '/'. Returns (cmd, desc) pairs, best match first."""
    q = prefix[1:].lower()
    if not q:
        return list(SLASH_COMMANDS)
    scored = []
    for cmd, desc in SLASH_COMMANDS:
        name = cmd[1:].lower()
        if name.startswith(q):
            scored.append((1000 - len(name), cmd, desc))
            continue
        s = _fuzzy_score(q, name)
        if s is not None:
            scored.append((s, cmd, desc))
    scored.sort(key=lambda t: -t[0])
    return [(c, d) for _, c, d in scored]


def _fuzzy_filter(opts: list[str], prefix: str) -> list[str]:
    """Fuzzy-match `prefix` against `opts`, best match first."""
    q = prefix.lower()
    if not q:
        return list(opts)
    scored = []
    for o in opts:
        name = o.lower()
        if name.startswith(q):
            scored.append((1000 - len(name), o))
            continue
        s = _fuzzy_score(q, name)
        if s is not None:
            scored.append((s, o))
    scored.sort(key=lambda t: -t[0])
    return [o for _, o in scored]


def _match_slash_args(cmd: str, prefix: str) -> list[tuple[str, str]]:
    """Fuzzy-match `prefix` (no leading '/') against `cmd`'s keyword options."""
    opts = SLASH_ARG_OPTIONS.get(cmd)
    if not opts:
        return []
    names = cast("list[str]", opts if isinstance(opts, list) else opts())
    return [(o, "") for o in _fuzzy_filter(names, prefix)]


def _match_source_model(src_name: str, prefix: str) -> list[tuple[str, str]]:
    """Fuzzy-match `prefix` against the model ids `src_name` advertises via
    GET /models (cached on the Source after the first fetch -- see
    Source.list_model_ids; a NX_SOURCE_<NAME>_MODELS pin replaces the fetch
    when the server's list is stale). Always leads with a "" sentinel
    (rendered as "(default)") so a bare Enter on this dropdown submits the
    plain `/source <name>` switch instead of forcing a model pick. Not
    capped: multi-model hosts can advertise dozens of ids, and the dropdown
    window (_dropdown_window) scrolls instead of hiding them."""
    src = st.SOURCES.get(src_name)
    if src is None:
        return [("", "")]
    ids = src.list_model_ids()
    matches = [("", "")] if not prefix else []
    matches += [(m, "") for m in _fuzzy_filter(ids, prefix)]
    return matches


# Dropdown rows rendered before the list scrolls. Uniform across every
# dropdown (slash, arg, arg2/model, @file): nothing is hard-truncated, long
# lists (multi-model hosts, aggregators, big repos) scroll instead, and the
# window is generous because these lists are often browsed rather than typed.
_DROPDOWN_MAX_ROWS = 8


def _dropdown_window(n: int, sel: int, top: int) -> tuple[int, bool, bool]:
    """Clamp a scrolling dropdown window over `n` rows so it contains row
    `sel`. Returns (window_top, more_above, more_below). `top` is the
    previous window top (0 after a reset) so the window only moves when the
    selection walks off an edge — it never re-centers per keystroke."""
    if n <= _DROPDOWN_MAX_ROWS:
        return 0, False, False
    top = max(0, min(top, n - _DROPDOWN_MAX_ROWS))
    if sel < top:
        top = sel
    elif sel >= top + _DROPDOWN_MAX_ROWS:
        top = sel - _DROPDOWN_MAX_ROWS + 1
    return top, top > 0, top + _DROPDOWN_MAX_ROWS < n


_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
              ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def _file_entries(root: str, directory: str) -> list[str]:
    """Return visible immediate children of *directory*, relative to *root*."""
    try:
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
    except OSError:
        return []
    result = []
    for entry in entries:
        if entry.name.startswith(".") or entry.name in _SKIP_DIRS:
            continue
        rel = os.path.relpath(entry.path, root)
        result.append(rel + "/" if entry.is_dir(follow_symlinks=False) else rel)
    return result


_TREE_SCAN_CAP = 20000  # total entries visited per tree-wide fuzzy search

# --- @-mention completion caches (PERF.md PF-01) --------------------------
#
# `_match_file_paths` runs once per keystroke while the user types an
# `@mention`. Before this cache it re-walked the whole project tree (up to
# _TREE_SCAN_CAP entries) and re-ran `git rev-parse` on EVERY keystroke --
# measured 38-56 ms/keystroke on a 19.7k-entry repo, the worst interactive
# number in the codebase. Two module-level caches fix it:
#   * the tree listing, invalidated by the root dir's mtime and a short TTL;
#   * the previous prefix's surviving candidate set, so a longer prefix filters
#     the previous matches (subsequence matching is monotone under prefix
#     extension) instead of rescanning the tree.
# Both are process-global and re-warmed at each prompt (see `_bust_tree_cache`).
_TREE_CACHE_TTL = 0.5
# entries are (rel_path, base_lower, path_lower, char_mask) -- the lowercased
# forms and the presence mask are precomputed once per walk, so a keystroke's
# fuzzy scan does no per-entry string work it can avoid (PERF.md PF-01).
_TREE_CACHE: tuple[str, float, float, list[tuple[str, str, str, int]]] | None = None
_TREE_FILTER_CACHE: tuple[str, str, list[tuple[str, str, str, int]]] | None = None
_TREE_RESULT_CACHE: tuple[str, str, list[str]] | None = None  # (root, prefix, top-8)
_PROJECT_ROOT_CACHE: tuple[str, str] | None = None  # (cwd, root)

# Stale-while-revalidate (PERF): on TTL expiry a keystroke must NOT pay the
# full `_walk_tree()` (up to _TREE_SCAN_CAP entries) before redrawing -- that is
# the 38-56 ms cold hit PF-01 first fixed. Instead the stale tree is returned
# immediately and a single daemon thread rebuilds it; the lock + flag keep
# concurrent keystrokes from spawning duplicate rebuilds, and the generation
# counter drops a rebuild whose result was invalidated mid-walk.
_TREE_REVALIDATE_LOCK = threading.Lock()
_TREE_REVALIDATING = False
_TREE_CACHE_GEN = 0

# Bitset of which characters an entry's path contains, for a cheap
# necessary-condition rejection before the (Python-loop) fuzzy scorer. A
# subsequence match implies every query char is present, so dropping entries
# whose mask lacks a query char can never hide a real match.
_MASK_CHARS = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz0123456789")}


def _char_mask(s: str) -> int:
    mask = 0
    for ch in s:
        i = _MASK_CHARS.get(ch)
        if i is None:
            mask |= 1  # bit 0 doubles as "some other char"; always allowed
        else:
            mask |= 1 << (i + 1)
    return mask


def _query_mask(q: str) -> int:
    """Bits a candidate must all have to be a possible fuzzy match."""
    mask = 0
    for ch in q:
        i = _MASK_CHARS.get(ch)
        mask |= 1 if i is None else 1 << (i + 1)
    return mask


def _bust_tree_cache() -> None:
    """Drop the completion caches (called on each prompt entry so files created
    mid-session show up, and available to any caller that knows the tree moved)."""
    global _TREE_CACHE, _TREE_FILTER_CACHE, _TREE_RESULT_CACHE, _PROJECT_ROOT_CACHE
    global _TREE_CACHE_GEN
    _TREE_CACHE = None
    _TREE_FILTER_CACHE = None
    _TREE_RESULT_CACHE = None
    _PROJECT_ROOT_CACHE = None
    # Invalidate any in-flight background revalidation: its result was built
    # against a tree snapshot this bust just discarded.
    _TREE_CACHE_GEN += 1


def _project_root() -> str:
    """`_detect_project_root()` memoized per cwd. The cwd does not change
    mid-REPL, so the underlying `git rev-parse` subprocess (~2 ms) must not run
    per keystroke (PERF.md PF-01)."""
    global _PROJECT_ROOT_CACHE
    from ._risk import _detect_project_root

    cwd = os.path.realpath(os.getcwd())
    if _PROJECT_ROOT_CACHE is not None and _PROJECT_ROOT_CACHE[0] == cwd:
        return _PROJECT_ROOT_CACHE[1]
    root = os.path.realpath(_detect_project_root())
    _PROJECT_ROOT_CACHE = (cwd, root)
    return root


def _walk_tree_entries(root: str) -> list[tuple[str, str, str, int]]:
    """`_walk_tree()` plus the per-entry derived data (lowercased base/path,
    char mask) the fuzzy scan consumes -- computed once per walk."""
    entries = []
    for rel in _walk_tree(root):
        path_l = rel.lower()
        entries.append((rel, path_l.rstrip("/").rsplit("/", 1)[-1], path_l,
                        _char_mask(path_l)))
    return entries


def _revalidate_tree(root: str, mtime: float) -> None:
    """Stale-while-revalidate refresh: rebuild the tree cache off the keystroke
    path in a daemon thread. Single-flight (lock + flag) so a burst of
    keystrokes past the TTL spawns one rebuild, not one per key."""
    global _TREE_REVALIDATING
    with _TREE_REVALIDATE_LOCK:
        if _TREE_REVALIDATING:
            return
        _TREE_REVALIDATING = True
    gen = _TREE_CACHE_GEN

    def _worker() -> None:
        global _TREE_CACHE, _TREE_REVALIDATING
        try:
            entries = _walk_tree_entries(root)
            # Skip the swap if the cache was busted while we walked: the new
            # snapshot is already stale by the caller's reckoning.
            if _TREE_CACHE_GEN == gen:
                _TREE_CACHE = (root, mtime, time.monotonic() + _TREE_CACHE_TTL, entries)
        except Exception:
            pass  # a failed refresh just leaves the stale tree in place
        finally:
            with _TREE_REVALIDATE_LOCK:
                _TREE_REVALIDATING = False

    threading.Thread(target=_worker, name="nx-tree-revalidate", daemon=True).start()


def _cached_tree(root: str) -> list[tuple[str, str, str, int]]:
    """`_walk_tree(root)` behind a short TTL + root-mtime guard so a keystroke
    never pays the BFS scan (PERF.md PF-01), with the per-entry derived data
    (lowercased base/path, char mask) computed once per walk.

    Stale-while-revalidate: once the TTL has expired the stale entries are
    returned immediately and a background thread refreshes them (a cold or
    mtime-changed tree still walks synchronously, so the first completion and
    post-write completions stay correct)."""
    global _TREE_CACHE
    now = time.monotonic()
    try:
        mtime = os.stat(root).st_mtime
    except OSError:
        mtime = 0.0
    if _TREE_CACHE is not None:
        c_root, c_mtime, deadline, entries = _TREE_CACHE
        if c_root == root and c_mtime == mtime:
            if now < deadline:
                return entries
            _revalidate_tree(root, mtime)
            return entries
    entries = _walk_tree_entries(root)
    _TREE_CACHE = (root, mtime, now + _TREE_CACHE_TTL, entries)
    return entries


def _walk_tree(root: str) -> list[str]:
    """List every visible file/dir under root (relative paths, dirs suffixed
    with '/'), skipping _SKIP_DIRS and dotfiles. Bounded by _TREE_SCAN_CAP
    total entries. Traversal is breadth-first across directories (queue of
    pending dirs, all of one depth before descending further) rather than
    os.walk's depth-first order -- a single huge nested subtree (e.g. a big
    data/predictions folder) would otherwise exhaust the cap before sibling
    directories are visited at all, hiding real matches elsewhere in the tree.
    """
    result: list[str] = []
    pending = [root]
    while pending and len(result) < _TREE_SCAN_CAP:
        next_pending = []
        for directory in pending:
            try:
                entries = sorted(os.scandir(directory), key=lambda e: e.name)
            except OSError:
                continue
            for entry in entries:
                if entry.name.startswith(".") or entry.name in _SKIP_DIRS:
                    continue
                rel = os.path.relpath(entry.path, root)
                if entry.is_dir(follow_symlinks=False):
                    result.append(rel + "/")
                    next_pending.append(entry.path)
                else:
                    result.append(rel)
                if len(result) >= _TREE_SCAN_CAP:
                    break
            if len(result) >= _TREE_SCAN_CAP:
                break
        pending = next_pending
    return result


def _match_file_paths(prefix: str) -> list[str]:
    """Complete a typed path, or fuzzy-find by name across the whole tree.

    If the prefix has no '/' yet, search recursively (basename-first, so
    "campaign_report" finds src/analysis/campaign_report.py) instead of only
    listing the project root -- this is what lets the user type a filename
    without first spelling out its directory. Once a '/' is present, fall
    back to listing just that one directory's immediate children, which
    stays cheap and reflects files created mid-session.
    """
    global _TREE_FILTER_CACHE, _TREE_RESULT_CACHE

    root = _project_root()
    q = prefix.lower()

    if "/" not in prefix:
        # Unchanged prefix (e.g. arrow-key dropdown navigation redraws with the
        # same buffer) -> replay the cached top-8 instead of re-scoring.
        if _TREE_RESULT_CACHE is not None and _TREE_RESULT_CACHE[0] == root \
                and _TREE_RESULT_CACHE[1] == prefix:
            return list(_TREE_RESULT_CACHE[2])
        candidates = _cached_tree(root)
        # Incremental filter: subsequence matching is monotone under prefix
        # extension, so everything matching a longer prefix also matched the
        # shorter one -- reuse the previous candidate set instead of rescanning.
        prev = _TREE_FILTER_CACHE
        if prev is not None and prev[0] == root and prefix != prev[1] \
                and prefix.startswith(prev[1]):
            candidates = prev[2]
        qmask = _query_mask(q) if q else 0
        scored = []
        matched = []
        for entry in candidates:
            rel, base, path_l, mask = entry
            if qmask & ~mask:
                continue  # missing a query char -> cannot be a subsequence
            s = _fuzzy_score(q, base) if q else 0
            if q and s is None:
                s = _fuzzy_score(q, path_l)
            if s is None:
                continue
            matched.append(entry)
            scored.append((s, rel))
        _TREE_FILTER_CACHE = (root, prefix, matched)
        result = [rel for _, rel in heapq.nlargest(8, scored, key=lambda t: t[0])]
        _TREE_RESULT_CACHE = (root, prefix, result)
        return result

    parent_part, _ = os.path.split(prefix)
    directory = os.path.realpath(os.path.join(root, parent_part))
    try:
        if os.path.commonpath((root, directory)) != root:
            return []
    except ValueError:
        return []

    files = _file_entries(root, directory)
    if not q:
        return files
    scored = []
    for f in files:
        s = _fuzzy_score(q, f.lower())
        if s is not None:
            scored.append((s, f))
    scored.sort(key=lambda t: -t[0])
    return [f for _, f in scored]


def _accept_file_completion(line: str, col: int, start: int, chosen: str) -> tuple[str, int]:
    """Insert a file completion, keeping a selected directory open to browse."""
    suffix = "" if chosen.endswith("/") else " "
    completed = line[:start] + "@" + chosen + suffix + line[col:]
    return completed, start + 1 + len(chosen) + len(suffix)


def _at_token_at_cursor(line: str, col: int) -> tuple[int, str] | None:
    """If the cursor sits inside/at the end of an '@token' word, return
    (start_col, token_without_at); else None. Only triggers when the '@'
    starts the word (so "user@example.com" mid-word doesn't match)."""
    i = col
    while i > 0 and not line[i - 1].isspace():
        i -= 1
    if i < col and line[i] == "@":
        return i, line[i + 1:col]
    return None

def _chatbox_fallback(prompt: str) -> str:
    """Plain `input()` fallback used when raw terminal mode isn't usable. Single-line only;
    newlines must be typed as the literal `\\n` (rare in practice)."""
    return input(prompt)


def _char_width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1


def _str_width(s: str) -> int:
    return sum(_char_width(c) for c in s)


def _str_slice_by_width(s: str, max_width: int) -> str:
    """Return the longest prefix of s that fits within max_width display columns."""
    w = 0
    for i, ch in enumerate(s):
        cw = _char_width(ch)
        if w + cw > max_width:
            return s[:i]
        w += cw
    return s


def _raw_read_key(fd: int) -> str:
    """Read one decoded keypress, raising EOFError when stdin is closed."""
    b = read_byte(fd)
    if not b:
        raise EOFError
    first = b[0]
    if first < 0x80:
        return b.decode("ascii")
    elif first < 0xC0:
        return b.decode("utf-8", "replace")
    elif first < 0xE0:
        need = 1
    elif first < 0xF0:
        need = 2
    else:
        need = 3
    for _ in range(need):
        extra = read_byte(fd)
        if not extra:
            raise EOFError
        b += extra
    return b.decode("utf-8", "replace")


def _raw_read_available(fd: int, timeout: float = 0.02) -> str:
    raw = bytearray()
    while True:
        if not wait_ready(fd, timeout):
            break
        chunk = read_byte(fd)
        if not chunk:
            raise EOFError
        raw += chunk
        timeout = 0
    return raw.decode("utf-8", "replace")


def _chatbox_raw(initial: str = "", history: list[str] | None = None,
                 wake: "object | None" = None) -> str:
    """Normal-scrollback multi-line editor.

    This does not enter the alternate screen. The prompt, streamed model output,
    tool confirmations, and the next prompt all stay in one terminal mode, which
    avoids garbling the REPL after submit.

    wake=None is the plain blocking editor. Passing the WAKE_INPUT sentinel
    makes the key loop poll stdin in short slices and return the sentinel as
    soon as _state._WAKE_DUE is set (a background job finished while the user
    idled — nx/_autowake.py). Whatever the user had typed is stashed in
    _state._WAKE_PREFILL so the REPL can restore it after the wake turn."""
    history = history or []
    _bust_tree_cache()  # fresh tree each prompt: mid-session files appear now
    fd = sys.stdin.fileno()

    buf = initial.split("\n") if initial else [""]
    row, col = len(buf) - 1, len(buf[-1])
    h_idx = len(history)
    rendered_lines = 0
    cursor_line = 0
    last_width: int | None = None  # column count of the last painted frame
    # Physical row count of the last painted frame, and each painted line's
    # visible width. Repaint/erase arithmetic runs in PHYSICAL terminal rows,
    # but a logical line wraps to ceil(width / columns) rows — so after a
    # resize the old frame's true height must be recomputed from these
    # widths at the new column count, or the erase leaves surplus wrapped
    # rows behind (orphaned frames stacking in scrollback, one per resize).
    rendered_rows = 0
    line_widths: list[int] = []
    current_matches: list[tuple[str, str]] = []
    current_kind: str | None = None
    current_at_start: int | None = None
    sel_idx = 0
    win_top = 0  # first rendered row of a scrolling dropdown (see render)

    def _frame_rows(widths: list[int], cols: int) -> int:
        """Physical terminal rows a frame with these per-line widths wraps to."""
        return sum(max(1, -(-w // max(1, cols))) for w in widths)

    def move_up(n: int) -> str:
        return f"\x1b[{n}A" if n > 0 else ""

    def move_down(n: int) -> str:
        return f"\x1b[{n}B" if n > 0 else ""

    def move_right(n: int) -> str:
        return f"\x1b[{n}C" if n > 0 else ""

    def build_visual(inner_w: int):
        visual = []
        for bi, line in enumerate(buf):
            if not line:
                visual.append((bi, 0, ""))
                continue
            start = 0
            while start < len(line):
                seg = _str_slice_by_width(line[start:], inner_w)
                visual.append((bi, start, seg))
                start += len(seg)
            if col == len(line) and row == bi and _str_width(line[visual[-1][1]:]) == inner_w:
                visual.append((bi, len(line), ""))
        return visual

    def compute_matches() -> None:
        """Resolve the dropdown candidates for the current buffer/cursor.

        Called once per key event, after the buffer mutation and before
        `render()`. This is the ONLY place the completion path touches the
        filesystem (PERF.md PF-01 / invariant I-1: `render()` itself must be
        I/O-free and do no work proportional to repo size)."""
        nonlocal current_matches, current_kind, current_at_start, sel_idx, win_top
        matches = []
        kind = None
        at_start = None
        if len(buf) == 1 and buf[0].startswith("/") and " " not in buf[0]:
            matches = _match_slash_commands(buf[0])
            kind = "slash"
        elif len(buf) == 1 and buf[0].startswith("/") and buf[0].count(" ") == 1 \
                and buf[0].split(" ", 1)[0] in SLASH_ARG_OPTIONS:
            cmd, rest = buf[0].split(" ", 1)
            matches = _match_slash_args(cmd, rest)
            kind = "arg"
        elif len(buf) == 1 and buf[0].startswith("/source ") and buf[0].count(" ") == 2:
            _cmd, src_name, rest = buf[0].split(" ", 2)
            matches = _match_source_model(src_name, rest)
            kind = "arg2"
        else:
            at_tok = _at_token_at_cursor(buf[row], col)
            if at_tok is not None:
                at_start, at_prefix = at_tok
                matches = [(p, "") for p in _match_file_paths(at_prefix)]
                kind = "file"
        if matches != current_matches:
            sel_idx = 0
            win_top = 0
        elif sel_idx >= len(matches):
            sel_idx = max(0, len(matches) - 1)
        current_matches = matches
        current_kind = kind
        current_at_start = at_start

    def render() -> None:
        nonlocal rendered_lines, rendered_rows, line_widths, cursor_line, \
            sel_idx, win_top, last_width
        width = shutil.get_terminal_size((80, 24)).columns
        if last_width is not None and width != last_width:
            # The terminal was resized (maximized/windowed): the previously
            # painted frame re-wrapped to the new width, so the cursor is no
            # longer where `cursor_line` claims. Re-anchor to a fresh line and
            # repaint from there (same recovery as the ^C path), or the new
            # frame draws over the wrong rows and doubles the box.
            last_width = width
            erase_stale_frame()
            resync_cursor()
        last_width = width
        box_w = max(20, width - 2)
        inner_w = max(1, box_w - 2)
        visual = build_visual(inner_w)

        cur_vrow = 0
        cur_vcol = 0
        for i, (bi, start, seg) in enumerate(visual):
            if bi != row:
                continue
            if start <= col <= start + len(seg):
                cur_vrow = i
                cur_vcol = _str_width(seg[:col - start])
                break

        mode = "yolo" if st.YOLO else "manual"
        hints = f"Enter send · Shift+Enter / ^J newline · ^C cancel · ^L clear · ^Q exit · ^S compress · shift+tab {mode}"
        max_hints = max(0, box_w - 11)
        truncated = _str_width(hints) > max_hints
        if truncated:
            hints = _str_slice_by_width(hints, max_hints - 1) + "…" if max_hints > 1 else "…"
        top_fill = max(0, box_w - 11 - _str_width(hints))
        if not truncated:
            mode_color = C.SUCCESS if st.YOLO else C.WARNING
            hints = hints[:-len(mode)] + f"{mode_color}{mode}{RESET}{DIM}"
        stats = f"{len(buf)} line{'s' if len(buf) != 1 else ''} · {sum(len(line) for line in buf)} chars"
        max_stats = max(0, box_w - 6)
        if _str_width(stats) > max_stats:
            stats = _str_slice_by_width(stats, max_stats - 1) + "…" if max_stats > 1 else "…"
        bot_fill = max(0, box_w - 4 - _str_width(stats))

        lines = [
            f"{DIM}╭─ {RESET}{C.ACCENT}you{RESET}{DIM} · {hints}{DIM} {'─' * top_fill}╮{RESET}",
            *[f"{DIM}│{RESET}{seg}{' ' * (inner_w - _str_width(seg))}{DIM}│{RESET}" for _, _, seg in visual],
            f"{DIM}╰─ {stats}{' ' * bot_fill}╯{RESET}",
        ]

        matches = current_matches
        kind = current_kind
        # Scroll window: rows beyond _DROPDOWN_MAX_ROWS stay reachable via
        # ↑/↓ (and PageUp/PageDown) instead of being silently cut off.
        win_top, more_above, more_below = _dropdown_window(
            len(matches), sel_idx, win_top)
        if more_above:
            lines.append(f"{DIM}  ···{RESET}")
        for i, (cmd, desc) in enumerate(
                matches[win_top:win_top + _DROPDOWN_MAX_ROWS], start=win_top):
            if kind == "file":
                label = f"@{cmd}"
            elif kind == "arg2" and cmd == "":
                label = "(default)"
            else:
                label = cmd
            text = f"{C.ACCENT}{label}{RESET}{DIM}  {desc}{RESET}" if desc else f"{C.ACCENT}{label}{RESET}"
            pad_w = box_w - 2 - _str_width(f"{label}  {desc}" if desc else label)
            marker = "› " if i == sel_idx else "  "
            lines.append(f"{DIM}{marker}{RESET}{text}{' ' * max(0, pad_w)}")
        if more_below:
            lines.append(f"{DIM}  ···{RESET}")

        # Physical rows the CURRENT frame wraps to at this width, used as the
        # repaint extent (so a frame that grew rows from a narrower width is
        # fully cleared) and, after the write, as the next erase's truth.
        phys_rows = _frame_rows(
            [_str_width(strip_ansi_for_display(ln)) for ln in lines], width)
        redraw_rows = max(rendered_rows, phys_rows)
        out = "\r" + move_up(cursor_line)
        for i in range(redraw_rows):
            out += "\x1b[2K"
            if i < len(lines):
                out += lines[i]
            if i != redraw_rows - 1:
                out += "\n"
        out += "\r" + move_up((redraw_rows - 1) - (cur_vrow + 1)) + move_right(cur_vcol + 1)
        # Emit the whole frame in ONE write with SIGINT masked. The repaint
        # only lands correctly if the cursor is where `cursor_line` says; a
        # SIGINT delivered between the write and the bookkeeping update (or
        # inside the previous paint) desyncs the two, and every later render
        # then paints N rows too low -- stacking orphaned frames in scrollback
        # (one per repeated ^C). Blocking SIGINT here keeps cursor position
        # and bookkeeping atomic; pending ^C still fires right after the
        # unmask, where the key loop's KeyboardInterrupt handling resyncs.
        mask = {signal.SIGINT}
        if hasattr(signal, "pthread_sigmask") and threading.current_thread() \
                is threading.main_thread():
            signal.pthread_sigmask(signal.SIG_BLOCK, mask)
        try:
            sys.stdout.write(out)
            sys.stdout.flush()
            rendered_lines = len(lines)
            rendered_rows = phys_rows
            line_widths = [_str_width(strip_ansi_for_display(ln)) for ln in lines]
            cursor_line = cur_vrow + 1
        finally:
            if hasattr(signal, "pthread_sigmask") and threading.current_thread() \
                    is threading.main_thread():
                signal.pthread_sigmask(signal.SIG_UNBLOCK, mask)

    def erase_stale_frame() -> None:
        """Erase the previously painted chatbox rows. After a resize the old
        frame has re-wrapped to the new width, so simply re-anchoring below it
        would leave the stale frame stuck in view (one orphaned box stacked
        per resize). The frame's true physical extent is recomputed at the
        current column count from the recorded per-line widths — narrowing
        the terminal wraps every line FURTHER, so the frame can occupy more
        rows than when it was painted. Clear down past that recomputed
        bottom, then clear upward through the whole recomputed height."""
        cols = shutil.get_terminal_size((80, 24)).columns
        total_rows = _frame_rows(line_widths, cols) \
            if len(line_widths) == rendered_lines else rendered_lines
        sys.stdout.write("\r")
        for _ in range(max(0, total_rows - cursor_line)):
            sys.stdout.write("\x1b[1B\x1b[2K")
        sys.stdout.write("\x1b[2K\x1b[J")
        for _ in range(max(0, total_rows - 1)):
            sys.stdout.write("\x1b[1A\x1b[2K")
        sys.stdout.flush()

    def resync_cursor() -> None:
        """Re-anchor the repaint origin after an interrupt. A SIGINT that
        lands mid-render (or mid-stream output) leaves the terminal cursor at
        an arbitrary row while `rendered_lines`/`cursor_line` still hold stale
        offsets, so the next paint draws N rows too low and pushes the old
        frame into scrollback. Resetting the bookkeeping to 'nothing rendered,
        cursor at column 0 of a fresh line' makes the next render a plain
        first-frame paint from wherever the cursor physically is."""
        nonlocal rendered_lines, rendered_rows, line_widths, cursor_line
        sys.stdout.write("\n")
        sys.stdout.flush()
        rendered_lines = 0
        rendered_rows = 0
        line_widths = []
        cursor_line = 0

    def finish() -> None:
        sys.stdout.write("\r" + move_down((rendered_rows - 1) - cursor_line) + "\n")
        sys.stdout.write("\x1b[?25h\x1b[?2004l")
        sys.stdout.flush()

    def insert_text(s: str) -> None:
        nonlocal row, col
        if not s:
            return
        parts = s.split("\n")
        cur = buf[row]
        tail = cur[col:]
        if len(parts) == 1:
            buf[row] = cur[:col] + parts[0] + tail
            col += len(parts[0])
            return
        buf[row] = cur[:col] + parts[0]
        new_lines = list(parts[1:-1]) + [parts[-1] + tail]
        buf[row + 1:row + 1] = new_lines
        row += len(new_lines)
        col = len(parts[-1])

    def backspace() -> None:
        nonlocal row, col
        if col > 0:
            buf[row] = buf[row][:col - 1] + buf[row][col:]
            col -= 1
        elif row > 0:
            prev = buf[row - 1]
            col = len(prev)
            buf[row - 1] = prev + buf[row]
            del buf[row]
            row -= 1

    def delete_forward() -> None:
        line = buf[row]
        if col < len(line):
            buf[row] = line[:col] + line[col + 1:]
        elif row < len(buf) - 1:
            buf[row] = line + buf[row + 1]
            del buf[row + 1]

    def load_from_history(hist_text: str) -> None:
        nonlocal row, col
        buf[:] = hist_text.split("\n") if hist_text else [""]
        row = len(buf) - 1
        col = len(buf[-1])

    def word_right() -> None:
        nonlocal row, col
        line = buf[row]
        if col >= len(line):
            if row < len(buf) - 1:
                row += 1
                col = 0
            return
        while col < len(line) and line[col].isspace():
            col += 1
        while col < len(line) and not line[col].isspace():
            col += 1

    def word_left() -> None:
        nonlocal row, col
        if col == 0:
            if row > 0:
                row -= 1
                col = len(buf[row])
            return
        line = buf[row]
        while col > 0 and line[col - 1].isspace():
            col -= 1
        while col > 0 and not line[col - 1].isspace():
            col -= 1

    def handle_escape(seq: str) -> str | None:
        nonlocal row, col, h_idx, sel_idx
        if seq.startswith("[200~"):
            paste = seq[5:]
            while "\x1b[201~" not in paste:
                paste += _raw_read_key(fd)
            paste = paste.split("\x1b[201~", 1)[0]
            if paste.endswith("\n") or paste.endswith("\r"):
                paste = paste[:-1]
            insert_text(paste)
            return
        if seq in ("[A", "OA"):
            if current_matches:
                sel_idx = (sel_idx - 1) % len(current_matches)
            elif history and h_idx > 0:
                h_idx -= 1
                load_from_history(history[h_idx])
            return
        if seq in ("[B", "OB"):
            if current_matches:
                sel_idx = (sel_idx + 1) % len(current_matches)
            elif history and h_idx < len(history):
                h_idx += 1
                load_from_history("" if h_idx == len(history) else history[h_idx])
            return
        # PageUp/PageDown: jump a whole dropdown window at a time (clamped,
        # not wrapping) so long model lists stay browsable.
        if seq in ("[5~", "[6~") and current_matches:
            step = _DROPDOWN_MAX_ROWS if seq == "[6~" else -_DROPDOWN_MAX_ROWS
            sel_idx = max(0, min(len(current_matches) - 1, sel_idx + step))
            return
        if seq in ("[1;5C", "[1;3C", "[5C"):
            word_right()
            return
        if seq in ("[1;5D", "[1;3D", "[5D"):
            word_left()
            return
        if seq in ("[C", "OC"):
            if col < len(buf[row]):
                col += 1
            elif row < len(buf) - 1:
                row += 1
                col = 0
            return
        if seq in ("[D", "OD"):
            if col > 0:
                col -= 1
            elif row > 0:
                row -= 1
                col = len(buf[row])
            return
        if seq in ("[H", "OH"):
            col = 0
            return
        if seq in ("[F", "OF"):
            col = len(buf[row])
            return
        if seq == "[3~":
            delete_forward()
            return
        if seq in ("[13;2u", "[27;2;13~"):
            insert_text("\n")
            return
        if seq in ("[Z", "[1;2Z"):
            st.toggle_yolo()
            return
        if seq in ("OP", "[11~"):
            return "/help"

    with raw_mode(fd, icrnl=True):
        sys.stdout.write("\x1b[?25h\x1b[?2004h\n")
        sys.stdout.flush()
        try:
            compute_matches()
            render()
            while True:
                # Wake-aware wait: with a sentinel set, poll stdin readiness
                # in short slices instead of blocking forever on one key. A
                # keypress still arrives instantly (fd ready → fall through),
                # but a finished background job now ends the wait too. The
                # typed buffer is stashed and restored after the wake turn,
                # so the wake never eats half a message.
                if wake is not None and st._WAKE_DUE:
                    st._WAKE_PREFILL = "\n".join(buf)
                    finish()
                    return wake
                # Poll instead of blocking so a terminal resize repaints the
                # frame within ~0.2s: each idle tick compares the current
                # column count against the last painted frame (cheap and
                # fully portable -- no SIGWINCH, which Windows lacks), and
                # render() re-anchors when it sees the width changed.
                if not wait_ready(fd, 0.2):
                    if last_width is not None and shutil.get_terminal_size(
                            (80, 24)).columns != last_width:
                        render()
                    continue
                c = _raw_read_key(fd)
                if c == "\x1b":
                    # Escape sequences can arrive byte-split: anything after
                    # the CSI/SS3 prefix is drained from a short poll window,
                    # and under load (SIGINT storm, busy tool turn) the tail
                    # bytes simply are not there yet. Hold the prefix and
                    # prepend it to the next read instead of letting the tail
                    # be typed into the buffer as literal text ("o;" residue).
                    tail = _raw_read_available(fd)
                    if not tail:
                        tail = _raw_read_available(fd, timeout=0.05)
                    if not tail:
                        # A bare Esc with nothing following: treat as Esc key.
                        continue
                    esc = handle_escape(tail)
                    if esc is not None:
                        finish()
                        return esc
                elif c == "\r":
                    if current_kind == "file" and current_matches:
                        chosen = current_matches[sel_idx][0]
                        line = buf[row]
                        buf[row], col = _accept_file_completion(
                            line, col, current_at_start or 0, chosen)
                        compute_matches()
                        render()
                        continue
                    if current_kind == "slash" and current_matches and len(buf) == 1 \
                            and buf[0].startswith("/") and " " not in buf[0]:
                        buf[0] = current_matches[sel_idx][0]
                    if current_kind == "arg" and current_matches:
                        cmd = buf[row].split(" ", 1)[0]
                        new_line = cmd + " " + current_matches[sel_idx][0]
                        if new_line != buf[row]:
                            buf[row] = new_line
                            col = len(buf[row])
                            compute_matches()
                            render()
                            continue
                        if cmd == "/source":
                            # /source has a second (model) level -- don't
                            # submit on an exact source-name match, advance
                            # to the model dropdown instead.
                            buf[row] = new_line + " "
                            col = len(buf[row])
                            compute_matches()
                            render()
                            continue
                    if current_kind == "arg2" and current_matches:
                        cmd, src_name = buf[row].split(" ", 2)[:2]
                        chosen = current_matches[sel_idx][0]
                        if chosen == "":
                            buf[row] = f"{cmd} {src_name}"
                        else:
                            new_line = f"{cmd} {src_name} {chosen}"
                            if new_line != buf[row]:
                                buf[row] = new_line
                                col = len(buf[row])
                                compute_matches()
                                render()
                                continue
                    finish()
                    return "\n".join(buf)
                elif c == "\n":
                    insert_text("\n")
                elif c == "\x03":
                    raise KeyboardInterrupt
                elif c == "\x04":
                    if not any(buf):
                        raise EOFError
                elif c == "\x0c":
                    finish()
                    return "/clear"
                elif c == "\x14":
                    # Ctrl+T -> /tabs live dashboard. Same single-byte
                    # shorthand pattern as Ctrl+S (raw_mode clears IXON, so
                    # the byte reaches the editor instead of the terminal's
                    # flow control).
                    finish()
                    return "/tabs"
                elif c == "\x13":
                    # Ctrl+S -> /compress. raw_mode clears IXON, so the
                    # terminal hands the byte over instead of using it as
                    # the output-flow stop; single byte, unambiguous (the
                    # old Ctrl+Alt+C was byte-split into Esc + Ctrl-C and
                    # quit the REPL).
                    finish()
                    return "/compress"
                elif c == "\x11":
                    finish()
                    return "/exit"
                elif c in ("\x7f", "\x08"):
                    backspace()
                elif c == "\x15":
                    buf[row] = ""
                    col = 0
                elif c == "\t":
                    if current_matches and current_kind == "file":
                        chosen = current_matches[sel_idx][0]
                        line = buf[row]
                        buf[row], col = _accept_file_completion(
                            line, col, current_at_start or 0, chosen)
                    elif current_kind == "arg" and current_matches:
                        cmd = buf[row].split(" ", 1)[0]
                        new_line = cmd + " " + current_matches[sel_idx][0]
                        if new_line != buf[row]:
                            buf[row] = new_line
                            col = len(buf[row])
                        elif cmd == "/source":
                            # /source has a second (model) level -- a second
                            # Tab (exact source name already typed) advances
                            # to the model dropdown instead of doing nothing.
                            buf[row] = new_line + " "
                            col = len(buf[row])
                    elif current_matches:
                        buf[row] = current_matches[sel_idx][0] + " "
                        col = len(buf[row])
                elif c >= " ":
                    insert_text(c)
                compute_matches()
                render()
        except (EOFError, KeyboardInterrupt):
            # ^C inside the editor (either \x03 or a real SIGINT racing the
            # key loop) must tear the frame down and re-anchor the cursor
            # before propagating; the finally below only restores mode flags,
            # it does not reset the render bookkeeping.
            try:
                resync_cursor()
            except Exception:
                rendered_lines = 0
                cursor_line = 0
            raise
        finally:
            sys.stdout.write("\x1b[?25h\x1b[?2004l")
            sys.stdout.flush()


def read_multiline(initial: str = "", history: list[str] | None = None,
                   wake: "object | bool" = False) -> str:
    """Public entry point. Returns the entered text, or '' on empty submit.
    Raises EOFError / KeyboardInterrupt to match `input()`'s contract.

    wake=True makes the prompt return the WAKE_INPUT sentinel instead of
    blocking forever when a background job finishes while the user idles at
    an empty prompt (nx/_autowake.py): the raw editor's key loop checks the
    flag between keypresses, so the wake lands within ~0.2s of the next key
    — or immediately if the user hasn't started typing. Half-typed text is
    stashed in _WAKE_PREFILL and restored by the REPL after the wake turn."""
    from . import _autowake
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        # Fallback path — single-line input. Same UX as before this feature.
        return _chatbox_fallback(f"{C.ACCENT}you ›{RESET} ")
    try:
        return _chatbox_raw(initial, history, wake=_autowake.WAKE_INPUT if wake else None)
    except (EOFError, KeyboardInterrupt):
        raise
    except OSError:
        return _chatbox_fallback(f"{C.ACCENT}you ›{RESET} ")
