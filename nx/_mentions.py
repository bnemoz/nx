"""Expand @file mentions typed in chat input into inline file-content blocks,
so `@path/to/file` in a message reaches the model as read_file-style numbered
context without a separate tool round-trip. An image mention (@shot.png) goes
further: when NX_VISION is on, the image itself is attached as content parts
(see _images.read_image), so the model SEES the picture on the turn that asks
about it. A `.pdf` mention is extracted to markdown text (see _pdf.extract_pdf;
optional pymupdf4llm dependency). Pure function, no globals/state.
"""
from __future__ import annotations

import os
import re

from ._risk import _detect_project_root, _is_under_path
from ._env import _env_int
from ._tools import list_dir, read_file
from ._images import _resolve_mime, read_image

# Files above this size (bytes) are not inlined by a mention; a stub names the
# size and points at read_file. CSV/TSV always take the preview path past 1 KB
# (data files tokenize poorly and their header is almost all that matters).
def _mention_inline_limit() -> int:
    return _env_int("NX_MENTION_INLINE_LIMIT", 4096)


def _csv_preview(resolved: str, token: str) -> str | None:
    """Large CSV/TSV: inline only the header + first rows instead of the whole
    file. A CSV mention nearly always means 'look at the columns', and a big
    data file inlined whole once cost 45K tokens of context. Returns None when
    the file turns out not to be tabular text (binary, no header, single row)."""
    ext = os.path.splitext(resolved)[1].lower()
    sep = "\t" if ext in (".tsv", ".tab") else ","
    import csv as _csv
    try:
        with open(resolved, encoding="utf-8", newline="") as f:
            reader = _csv.reader(f, delimiter=sep)
            header = next(reader, None)
            if not header:
                return None
            rows = []
            for row, _ in zip(reader, range(10)):
                rows.append(row)
            # row count only if the file is small enough to scan (counting a
            # multi-MB CSV just to print a number is a waste of the read);
            # past NX_MENTION_CSV_SCAN_BYTES the preview says "10+ rows" instead
            size = os.path.getsize(resolved)
            n_rows: int | None = len(rows)
            if size <= _env_int("NX_MENTION_CSV_SCAN_BYTES", 1 << 20):
                for n_rows, _ in enumerate(reader, len(rows) + 1):
                    pass
            else:
                n_rows = None
    except (OSError, UnicodeDecodeError, _csv.Error):
        return None
    width = len(header)
    lines = [sep.join(header)]
    lines += [sep.join((c if len(c) <= 80 else c[:77] + "...") for c in r[:width])
              for r in rows]
    return (f"--- @{token} (large CSV, {n_rows if n_rows is not None else str(len(rows)) + '+'} data rows, NOT inlined) ---\n"
            "header + first rows only; call read_file with offset/limit to page:\n"
            + "\n".join(lines))

# A mention is '@' immediately preceded by start-of-string or whitespace (so
# "user@example.com" doesn't trigger), followed by non-space characters.
_MENTION_RE = re.compile(r"(?<!\S)@(\S+)")


def _resolve_mention(token: str, cwd: str, root: str | None) -> str | None:
    path = os.path.expanduser(token)
    if not os.path.isabs(path):
        path = os.path.join(cwd, path)
    resolved = os.path.realpath(path)
    if not (os.path.isfile(resolved) or os.path.isdir(resolved)):
        return None
    home = os.path.realpath(os.path.expanduser("~"))
    if not ((root is not None and _is_under_path(resolved, root)) or _is_under_path(resolved, home)):
        return None
    return resolved


def expand_at_mentions(text: str, cwd: str | None = None) -> str | list[dict]:
    """Read files referenced by @path tokens in `text` and append their
    contents as numbered-line blocks after the original text. Tokens that
    don't resolve to a readable file inside the project or home directory
    (typos, stray '@' in prose, paths outside scope) are left alone.

    An @image mention attaches the picture instead of text: with NX_VISION on,
    the return becomes a content-parts list (text + image parts) so the model
    sees the picture this turn. Off → a one-line note names the gate; a failed
    image read falls back to the ERROR text like any other mention.
    """
    cwd = cwd or os.getcwd()
    root = _detect_project_root(cwd)
    seen: set[str] = set()
    blocks: list[str] = []
    image_parts: list[dict] = []
    for m in _MENTION_RE.finditer(text):
        token = m.group(1).rstrip(".,;:!?)")
        if not token or token in seen:
            continue
        seen.add(token)
        resolved = _resolve_mention(token, cwd, root)
        if not resolved:
            continue
        if os.path.isdir(resolved):
            try:
                content = list_dir(resolved)
            except OSError:
                continue
            blocks.append(f"\n\n--- @{token} ---\n{content}")
            continue
        if _resolve_mime(resolved) is not None:
            from . import NX_VISION  # module global; live value, not import-time
            if not NX_VISION:
                blocks.append(f"\n\n--- @{token} ---\n"
                              f"(an image you asked about — not attached: NX_VISION is off, "
                              f"so this nx session cannot receive pictures; set NX_VISION=1 "
                              f"on a native-tool-call backend, or describe the image in text)")
                continue
            out = read_image(resolved)
            if isinstance(out, list):
                # marker part first ([image: path, N bytes]) then the payload
                image_parts.extend(out)
            else:
                blocks.append(f"\n\n--- @{token} ---\n{out}")
            continue
        if os.path.splitext(resolved)[1].lower() == ".pdf":
            from ._pdf import extract_pdf
            blocks.append(extract_pdf(resolved, token))
            continue
        ext = os.path.splitext(resolved)[1].lower()
        try:
            size = os.path.getsize(resolved)
        except OSError:
            size = 0
        if ext in (".csv", ".tsv", ".tab") and size > 1024:
            preview = _csv_preview(resolved, token)
            if preview is not None:
                blocks.append("\n\n" + preview)
                continue
        if size > _mention_inline_limit():
            blocks.append(
                f"\n\n--- @{token} (large, {size} bytes, NOT inlined) ---\n"
                "This file is too big to attach with the mention; it is NOT "
                "in this message. Call read_file (with offset/limit to page) "
                "or search_files to pull in only what you need.")
            continue
        try:
            content = read_file(resolved)
        except OSError:
            continue
        blocks.append(f"\n\n--- @{token} ---\n{content}")
    if not blocks and not image_parts:
        return text
    if not image_parts:
        return text + "".join(blocks)
    parts: list[dict] = [{"type": "text", "text": text + "".join(blocks)}]
    parts.extend(image_parts)
    return parts
