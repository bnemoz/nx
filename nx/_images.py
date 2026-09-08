"""read_image tool: hand a local image to a vision-capable model.

Returns an OpenAI content-parts array ({"type": "image_url", "image_url":
{"url": "data:image/<fmt>;base64,..."}}) as the tool result, which
_handle_native_tool_calls() stores in the tool message. Servers/models without
vision will error on the next request — that's the user's signal to not enable
NX_VISION on that backend (opt-in for exactly that reason, same posture as
NX_NATIVE_TOOLS).

Read-only like read_file; never goes through approval. Size-capped by
NX_IMAGE_MAX_BYTES (default 8 MB — beyond that, base64 would blow the
context anyway). Only meaningful on native-tool-call sources; on the text
protocol a parts-array result can't be carried, so the schema is withheld
by _active_tools() until native tool_calls have been confirmed this session.

Format resolution is extension-first (like read_file trusting the name), with
a magic-byte sniff as fallback so a missing/unknown extension on a real image
still reads. No decoding is attempted here — a file whose name says .png is
handed through even if the bytes disagree; a backend that can't decode it
errors once and the model moves on. No image library (Pillow &c.) is pulled
in: oversized images are refused, not downscaled, and svg/tiff/heic stay
refused because most OpenAI-compatible decoders reject them anyway.
"""
from __future__ import annotations

import base64
import os
from typing import Any

from ._env import _env_int

# Formats browsers/servers accept in data: URLs; anything else is refused
# with a message naming the supported set (prevents "it read a .exe").
# Aliases normalize to the same MIME (read_image(".jpeg") == ".jpg").
_EXT_TO_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".jpe": "image/jpeg", ".jfif": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}

# Magic numbers for the same set, used when the extension is missing or
# unknown ("screenshot" or "frame.bin" that is really a PNG still reads).
_MAGIC_HEADS = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"), (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)

# cap on the raw image that gets base64'd into context (default 8 MB).
# NX_IMAGE_MAX_BYTES overrides at CALL time — resolving NX_* at import would
# freeze whatever the shell had during test collection (CONTRIBUTING rule).
# The module attribute stays the test-patch seam: with the env var unset,
# _env_int falls back to the current module value.
IMAGE_MAX_BYTES = 8 * 1024 * 1024

# Schema for the read_image tool. Kept terse — it's on every turn's floor
# when NX_VISION=1.
IMAGE_VISION_TOOL = {"type": "function", "function": {
    "name": "read_image",
    "description": "Show a local image to the model as a picture so you can read what's in it: an error screenshot, UI state, mockup, diagram. Returns the image itself, not a text description. png/jpeg/gif/webp/bmp.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Path to the image file (~ works)"},
    }, "required": ["path"]},
}}


def _sniff_mime(path: str) -> str | None:
    """Read the first bytes of `path` and return its image MIME, or None if
    unreadable or unrecognized. 16 bytes is enough for every signature here
    (WEBP needs 12: RIFF + 4 len bytes + WEBP)."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return None
    for magic, mime in _MAGIC_HEADS:
        if head.startswith(magic):
            return mime
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def _resolve_mime(path: str) -> str | None:
    """MIME for an image path: the extension map first (trusted, like
    read_file), then a magic-byte sniff for files whose extension is missing
    or not in the map. Returns None when neither resolves."""
    ext = os.path.splitext(path)[1].lower()
    mime = _EXT_TO_MIME.get(ext)
    if mime is not None:
        return mime
    return _sniff_mime(path)


def read_image(path: str, **_: Any) -> Any:
    """Return a content-parts array carrying the image, or an ERROR string."""
    if not isinstance(path, str) or not path.strip():
        return "ERROR: path is empty"
    path = os.path.expanduser(path.strip())
    if os.path.isdir(path):
        return f"ERROR: {path} is a directory, not an image file"
    cap = _env_int("NX_IMAGE_MAX_BYTES", IMAGE_MAX_BYTES)
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"
    if size == 0:
        return f"ERROR: {path} is empty (0 bytes)"
    if size > cap:
        return (f"ERROR: {path} is {size // 1024} KB; over the "
                f"{cap // (1024 * 1024)} MB context cap (NX_IMAGE_MAX_BYTES)")
    mime = _resolve_mime(path)
    if mime is None:
        ext = os.path.splitext(path)[1].lower()
        return (f"ERROR: {ext or '(no extension)'} is not a supported image format "
                f"(want one of: {', '.join(sorted(k.lstrip('.') for k in _EXT_TO_MIME))} "
                "— or an image file with no/wrong extension whose format can be "
                "sniffed from its first bytes)")
    # Re-check after reading: the size check above is advisory (TOCTOU), and
    # the +1 byte makes an oversize read detectable without a second stat.
    try:
        with open(path, "rb") as f:
            data = f.read(cap + 1)
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"
    if len(data) > cap:
        return (f"ERROR: {path} is over the "
                f"{cap // (1024 * 1024)} MB context cap (NX_IMAGE_MAX_BYTES)")
    b64 = base64.b64encode(data).decode("ascii")
    return [
        {"type": "text", "text": f"[image: {path}, {len(data)} bytes]"},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]


def _parts_preview(result: list) -> str:
    """One-line, payload-free rendering of a content-parts tool result.

    Everywhere a parts result meets a TEXT channel — the terminal preview,
    /tooloutput, the post-hook JSON, a text-protocol Observation, --trace,
    batch stdout — the base64 payload must never follow: it is megabytes of
    unrenderable noise that the receiving channel can't trim (it looks like
    one giant token). The text parts (the '[image: path, N bytes]' marker the
    tool itself emits) are kept so the line stays identifiable."""
    texts = [p.get("text", "") for p in result if isinstance(p, dict)]
    n_img = sum(1 for p in result if isinstance(p, dict) and p.get("type") == "image_url")
    head = " ".join(t for t in texts if t)
    if n_img:
        head += (" " if head else "") + f"[image attached — {n_img} part(s), base64 payload not shown]"
    return head or f"[image attached — {len(result)} part(s), base64 payload not shown]"


def _strip_image_parts(msgs: list[dict[str, Any]]) -> int:
    """Swap every image_url part in these messages' tool content for a text
    note, in place. Returns how many messages were changed.

    Used by the auto-compress path when a context-length 400 may have been
    CAUSED by image parts: keeping them verbatim in the retry tail would 400
    again and kill the turn. The model keeps the '[image: path, N bytes]'
    marker (it saw the picture once and summarized it), the bytes are gone.
    Mutating the dicts in place matters — compress()'s tail holds references
    into the caller's live history."""
    changed = 0
    for m in msgs:
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list):
            continue
        hit = False
        for p in c:
            if isinstance(p, dict) and p.get("type") == "image_url":
                p.clear()
                p["type"] = "text"
                p["text"] = "[image evicted from context to fit the window — re-read the file if needed]"
                hit = True
        if hit:
            changed += 1
    return changed
