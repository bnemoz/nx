"""Expand @file mentions typed in chat input into inline file-content blocks,
so `@path/to/file` in a message reaches the model as read_file-style numbered
context without a separate tool round-trip. An image mention (@shot.png) goes
further: when NX_VISION is on, the image itself is attached as content parts
(see _images.read_image), so the model SEES the picture on the turn that asks
about it. Pure function, no globals/state.
"""
from __future__ import annotations

import os
import re

from ._risk import _detect_project_root, _is_under_path
from ._tools import list_dir, read_file
from ._images import _resolve_mime, read_image

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
