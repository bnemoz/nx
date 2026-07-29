"""Multi-line chatbox input, replacing the bare `input()` prompt with a
framed, multi-line editor:
  - Enter submits, Shift+Enter (or Ctrl+J) inserts a newline
  - Paste (bracketed-paste mode) inserts its text verbatim -- newlines stay,
    a trailing newline at the end of paste is stripped so pasting never
    accidentally submits
  - Up/Down navigate history, Left/Right move within the current line,
    Home/End jump to line start/end, Ctrl+U clears the line, Ctrl+C cancels
  - Shift+Tab toggles YOLO mode (mirrors /yolo), shown live in the box border
  - Long lines word-wrap visually inside the box; the buffer stays one
    logical string (newlines preserved) so the model sees the real text
  - Typing `@` starts a fuzzy file-path suggestion dropdown (Tab to accept);
    the actual mention expansion into file contents happens in
    nx/_mentions.py when the message is submitted
Falls back to plain `input()` when stdin/stdout is not a TTY.
"""
from __future__ import annotations

import os
import select
import shutil
import sys
import termios
import unicodedata

from . import _state as st
from ._ansi import DIM, RESET, C, THEMES, raw_mode

# (command, description) — kept in sync with the README "Commands" table.
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "list all slash commands"),
    ("/init", "generate/update AGENTS.md for this project"),
    ("/source", "list sources, switch to one, or override its model"),
    ("/theme", "list color themes, or switch to one"),
    ("/provider", "show or set OpenRouter provider-routing order"),
    ("/yolo", "toggle auto-approve for writes and bash"),
    ("/resources", "show this nx process's CPU% and RSS memory"),
    ("/bg", "list background run_bash jobs, or /bg kill <pid>"),
    ("/approval", "show or set risk threshold (all/low/medium/high/yolo)"),
    ("/sessions", "list recent sessions, or show one in full"),
    ("/resume", "resume a past session (n/id/prefix/title)"),
    ("/save", "save the current session (optional custom title)"),
    ("/memory", "manage persistent memory"),
    ("/delete", "delete a saved session"),
    ("/compress", "summarize older turns into one, keep last 2 verbatim"),
    ("/compact", "alias for /compress"),
    ("/autocompress", "show or set the auto-compress threshold"),
    ("/recover", "force a low-temp visible checkpoint after a bad stream"),
    ("/reasoning", "on|off|last — toggle or show reasoning trace"),
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
SLASH_ARG_OPTIONS: dict[str, list[str]] = {
    "/theme": sorted(THEMES),
    "/reasoning": ["on", "off", "last"],
    "/approval": ["all", "low", "medium", "high", "yolo"],
    "/autocompress": ["off", "on"],
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


def _match_slash_args(cmd: str, prefix: str) -> list[tuple[str, str]]:
    """Fuzzy-match `prefix` (no leading '/') against `cmd`'s keyword options."""
    opts = SLASH_ARG_OPTIONS.get(cmd)
    if not opts:
        return []
    q = prefix.lower()
    if not q:
        return [(o, "") for o in opts]
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
    return [(o, "") for _, o in scored]


# Lazily-built, process-lifetime cache of project-relative file paths, used
# for '@file' completion. Not refreshed mid-session (files added after the
# chatbox first renders won't show up) -- a cheap tradeoff for a suggestion
# list, not a source of truth for what gets read.
_FILE_INDEX: list[str] | None = None
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
              ".mypy_cache", ".pytest_cache", ".ruff_cache"}
_FILE_INDEX_CAP = 5000


def _build_file_index() -> list[str]:
    global _FILE_INDEX
    from ._risk import _detect_project_root
    root = _detect_project_root()
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        names = [(dn, True) for dn in dirnames] + [(fn, False) for fn in filenames]
        for name, is_dir in names:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            files.append(rel + "/" if is_dir else rel)
            if len(files) >= _FILE_INDEX_CAP:
                _FILE_INDEX = files
                return files
    _FILE_INDEX = files
    return files


def _match_file_paths(prefix: str) -> list[str]:
    """`prefix` excludes the leading '@'. Returns matching relative paths, best first."""
    files = _FILE_INDEX if _FILE_INDEX is not None else _build_file_index()
    q = prefix.lower()
    if not q:
        return files[:8]
    scored = []
    for f in files:
        s = _fuzzy_score(q, f.lower())
        if s is not None:
            scored.append((s, f))
    scored.sort(key=lambda t: -t[0])
    return [f for _, f in scored[:8]]


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
    b = os.read(fd, 1)
    if not b:
        return ""
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
        extra = os.read(fd, 1)
        if not extra:
            break
        b += extra
    return b.decode("utf-8", "replace")


def _raw_read_available(fd: int, timeout: float = 0.02) -> str:
    raw = bytearray()
    while True:
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            break
        raw += os.read(fd, 1)
        timeout = 0
    return raw.decode("utf-8", "replace")


def _chatbox_raw(initial: str = "", history: list[str] | None = None) -> str:
    """Normal-scrollback multi-line editor.

    This does not enter the alternate screen. The prompt, streamed model output,
    tool confirmations, and the next prompt all stay in one terminal mode, which
    avoids garbling the REPL after submit.
    """
    history = history or []
    fd = sys.stdin.fileno()

    buf = initial.split("\n") if initial else [""]
    row, col = len(buf) - 1, len(buf[-1])
    h_idx = len(history)
    rendered_lines = 0
    cursor_line = 0
    current_matches: list[tuple[str, str]] = []
    current_kind: str | None = None
    current_at_start: int | None = None
    sel_idx = 0

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

    def render() -> None:
        nonlocal rendered_lines, cursor_line, current_matches, current_kind, current_at_start, sel_idx
        width = shutil.get_terminal_size((80, 24)).columns
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
        hints = f"Enter send · Shift+Enter / ^J newline · ^C cancel · shift+tab {mode}"
        max_hints = max(0, box_w - 11)
        if _str_width(hints) > max_hints:
            hints = _str_slice_by_width(hints, max_hints - 1) + "…" if max_hints > 1 else "…"
        top_fill = max(0, box_w - 11 - _str_width(hints))
        stats = f"{len(buf)} line{'s' if len(buf) != 1 else ''} · {sum(len(line) for line in buf)} chars"
        max_stats = max(0, box_w - 6)
        if _str_width(stats) > max_stats:
            stats = _str_slice_by_width(stats, max_stats - 1) + "…" if max_stats > 1 else "…"
        bot_fill = max(0, box_w - 4 - _str_width(stats))

        lines = [
            f"{DIM}╭─ {RESET}{C.ACCENT}you{RESET}{DIM} · {hints} {'─' * top_fill}╮{RESET}",
            *[f"{DIM}│{RESET}{seg}{' ' * (inner_w - _str_width(seg))}{DIM}│{RESET}" for _, _, seg in visual],
            f"{DIM}╰─ {stats}{' ' * bot_fill}╯{RESET}",
        ]

        matches = []
        kind = None
        at_start = None
        if len(buf) == 1 and buf[0].startswith("/") and " " not in buf[0]:
            matches = _match_slash_commands(buf[0])[:6]
            kind = "slash"
        elif len(buf) == 1 and buf[0].startswith("/") and buf[0].count(" ") == 1 \
                and buf[0].split(" ", 1)[0] in SLASH_ARG_OPTIONS:
            cmd, rest = buf[0].split(" ", 1)
            matches = _match_slash_args(cmd, rest)[:6]
            kind = "arg"
        else:
            at_tok = _at_token_at_cursor(buf[row], col)
            if at_tok is not None:
                at_start, at_prefix = at_tok
                matches = [(p, "") for p in _match_file_paths(at_prefix)]
                kind = "file"
        if matches != current_matches:
            sel_idx = 0
        elif sel_idx >= len(matches):
            sel_idx = max(0, len(matches) - 1)
        current_matches = matches
        current_kind = kind
        current_at_start = at_start
        for i, (cmd, desc) in enumerate(matches):
            label = f"@{cmd}" if kind == "file" else cmd
            text = f"{C.ACCENT}{label}{RESET}{DIM}  {desc}{RESET}" if desc else f"{C.ACCENT}{label}{RESET}"
            pad_w = box_w - 2 - _str_width(f"{label}  {desc}" if desc else label)
            marker = "› " if i == sel_idx else "  "
            lines.append(f"{DIM}{marker}{RESET}{text}{' ' * max(0, pad_w)}")

        redraw_rows = max(rendered_lines, len(lines))
        out = "\r" + move_up(cursor_line)
        for i in range(redraw_rows):
            out += "\x1b[2K"
            if i < len(lines):
                out += lines[i]
            if i != redraw_rows - 1:
                out += "\n"
        out += "\r" + move_up((redraw_rows - 1) - (cur_vrow + 1)) + move_right(cur_vcol + 1)
        sys.stdout.write(out)
        sys.stdout.flush()
        rendered_lines = len(lines)
        cursor_line = cur_vrow + 1

    def finish() -> None:
        sys.stdout.write("\r" + move_down((rendered_lines - 1) - cursor_line) + "\n")
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

    def handle_escape(seq: str) -> None:
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

    with raw_mode(fd, icrnl=True):
        sys.stdout.write("\x1b[?25h\x1b[?2004h\n")
        sys.stdout.flush()
        try:
            render()
            while True:
                c = _raw_read_key(fd)
                if c == "\x1b":
                    handle_escape(_raw_read_available(fd))
                elif c == "\r":
                    if current_kind == "file" and current_matches:
                        chosen = current_matches[sel_idx][0]
                        line = buf[row]
                        buf[row] = line[:current_at_start] + "@" + chosen + " " + line[col:]
                        col = current_at_start + 1 + len(chosen) + 1
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
                elif c in ("\x7f", "\x08"):
                    backspace()
                elif c == "\x15":
                    buf[row] = ""
                    col = 0
                elif c == "\t":
                    if current_matches and current_kind == "file":
                        chosen = current_matches[sel_idx][0]
                        line = buf[row]
                        buf[row] = line[:current_at_start] + "@" + chosen + " " + line[col:]
                        col = current_at_start + 1 + len(chosen) + 1
                    elif current_kind == "arg" and current_matches:
                        cmd = buf[row].split(" ", 1)[0]
                        buf[row] = cmd + " " + current_matches[sel_idx][0]
                        col = len(buf[row])
                    elif current_matches:
                        buf[row] = current_matches[sel_idx][0] + " "
                        col = len(buf[row])
                elif c >= " ":
                    insert_text(c)
                render()
        finally:
            sys.stdout.write("\x1b[?25h\x1b[?2004l")
            sys.stdout.flush()


def read_multiline(initial: str = "", history: list[str] | None = None) -> str:
    """Public entry point. Returns the entered text, or '' on empty submit.
    Raises EOFError / KeyboardInterrupt to match `input()`'s contract."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        # Fallback path — single-line input. Same UX as before this feature.
        return _chatbox_fallback(f"{C.ACCENT}you ›{RESET} ")
    try:
        return _chatbox_raw(initial, history)
    except (EOFError, KeyboardInterrupt):
        raise
    except (OSError, termios.error):
        return _chatbox_fallback(f"{C.ACCENT}you ›{RESET} ")


