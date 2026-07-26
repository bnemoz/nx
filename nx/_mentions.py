"""Expand @file mentions typed in chat input into inline file-content blocks,
so `@path/to/file` in a message reaches the model as read_file-style numbered
context without a separate tool round-trip. Pure function, no globals/state.
"""
import os
import re

from ._risk import _detect_project_root, _is_under_path
from ._tools import list_dir, read_file

# A mention is '@' immediately preceded by start-of-string or whitespace (so
# "user@example.com" doesn't trigger), followed by non-space characters.
_MENTION_RE = re.compile(r"(?<!\S)@(\S+)")


def _resolve_mention(token, cwd, root):
    path = os.path.expanduser(token)
    if not os.path.isabs(path):
        path = os.path.join(cwd, path)
    resolved = os.path.realpath(path)
    if not (os.path.isfile(resolved) or os.path.isdir(resolved)):
        return None
    home = os.path.realpath(os.path.expanduser("~"))
    if not (_is_under_path(resolved, root) or _is_under_path(resolved, home)):
        return None
    return resolved


def expand_at_mentions(text, cwd=None):
    """Read files referenced by @path tokens in `text` and append their
    contents as numbered-line blocks after the original text. Tokens that
    don't resolve to a readable file inside the project or home directory
    (typos, stray '@' in prose, paths outside scope) are left alone."""
    cwd = cwd or os.getcwd()
    root = _detect_project_root(cwd)
    seen = set()
    blocks = []
    for m in _MENTION_RE.finditer(text):
        token = m.group(1).rstrip(".,;:!?)")
        if not token or token in seen:
            continue
        seen.add(token)
        resolved = _resolve_mention(token, cwd, root)
        if not resolved:
            continue
        try:
            if os.path.isdir(resolved):
                content = list_dir(resolved)
            else:
                content = read_file(resolved)
        except OSError:
            continue
        blocks.append(f"\n\n--- @{token} ---\n{content}")
    if not blocks:
        return text
    return text + "".join(blocks)
