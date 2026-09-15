"""Global cross-session input recall: what Up/Down at the prompt remembers.

Distinct from session persistence (_sessions_fs.py), which stores what the
*model* sees (the full messages array, resumable via /resume). This module
stores what the *user* typed — a small ring buffer of past submissions in
~/.nx/input_history.json, so Up on a fresh `nx` run reaches earlier runs'
prompts too. Plain JSON list of strings, newest last — the exact shape
_chatbox.py navigates, so the chatbox itself needs no changes.

Policy:
- blank submissions and slash commands are skipped (ephemeral toggles, not
  things to re-edit);
- only repeats of the very last entry are deduped (full dedup would reorder
  entries, which reads as surprising);
- the raw typed text is stored — `@file` mentions are expanded into `messages`
  when submitted, and the recall copy should be what the user actually typed;
- entries longer than INPUT_HISTORY_ENTRY_CHARS are stored truncated — the
  session JSON keeps the full text, and one huge paste shouldn't eat the
  recall budget.

Everything is best-effort: a missing/corrupt/unwritable file degrades to no
cross-run recall (the current session still recalls in memory) rather than
ever breaking the prompt loop.

HISTORY_LINES (NX_HISTORY_LINES via _env_int, resolved in __init__.py) is
looked up lazily via `import nx as _m` rather than bound at import time, so
a /reload or test monkeypatch of nx.HISTORY_LINES is picked up — same pattern
as SESSION_DESC_REFRESH in _sessions_fs.py.
"""
from __future__ import annotations

import json
import os

from . import _state as st
from ._private import open_private, private_dir
from ._sessions_fs import _exclusive_file_lock

# Default for NX_HISTORY_LINES: how many past submissions Up/Down recalls.
INPUT_HISTORY_LINES_DEFAULT = 100
# Per-entry size guard (chars). 0 disables the truncation.
INPUT_HISTORY_ENTRY_CHARS = 8000


def _history_path() -> str:
    """Where the recall file lives: NX_INPUT_HISTORY_FILE, then NX_HOME,
    else SESSION_HOME/input_history.json. NX_HOME is read from the live
    environment (not just the st.SESSION_HOME mirror) so a test module that
    sets NX_HOME at import time — before its module-scope m.main() calls —
    writes into its tempdir rather than the user's real recall file."""
    direct = os.environ.get("NX_HOME")
    if direct:
        return os.environ.get(
            "NX_INPUT_HISTORY_FILE",
            os.path.join(os.path.expanduser(direct), "input_history.json"))
    return os.environ.get(
        "NX_INPUT_HISTORY_FILE",
        os.path.join(st.SESSION_HOME, "input_history.json"),
    )


def _history_lines() -> int:
    import nx as _m
    return _m.HISTORY_LINES


def _read_entries(path: str) -> list[str]:
    """Load the stored entries; any read/parse failure or wrong shape is an
    empty history (never worth failing the prompt loop over)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, IOError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, str)]


def _load_input_history() -> list[str]:
    """Return the recall list (newest last) the REPL should start with.
    NX_HISTORY_LINES=0 disables the feature: no recall, nothing written."""
    cap = _history_lines()
    if cap <= 0:
        return []
    entries = _read_entries(_history_path())
    return entries[-cap:]


def _append_to_file(entry: str) -> None:
    """Best-effort append + cap on disk, under the same interprocess lock
    pattern _write_session uses so two concurrent nx REPLs can't lose each
    other's entries."""
    cap = _history_lines()
    if cap <= 0:
        return
    path = _history_path()
    d = os.path.dirname(path)
    if d:
        try:
            private_dir(d)
        except OSError:
            return  # not our dir (e.g. an NX_INPUT_HISTORY_FILE in a shared
            # /tmp) — makedirs already applied 0700 if we created it
    with _exclusive_file_lock(path):
        entries = _read_entries(path)
        entries.append(entry)
        del entries[:-cap]
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with open_private(tmp) as f:
                json.dump(entries, f, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _record_input(user: str, history: list[str]) -> bool:
    """Record one submission: skip blanks/slash commands and repeats of the
    very last entry, then append to `history` in memory and to disk. Returns
    whether anything was recorded."""
    entry = user.strip()
    if not entry or entry.startswith("/"):
        return False
    entry = entry[:INPUT_HISTORY_ENTRY_CHARS]
    if history and history[-1] == entry:
        return False
    history.append(entry)
    _append_to_file(entry)
    return True
