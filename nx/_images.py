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
"""
from __future__ import annotations

import base64
import os
from typing import Any

from ._env import _env_int

# Formats browsers/servers accept in data: URLs; anything else is refused
# with a message naming the supported set (prevents "it read a .exe").
_EXT_TO_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}

# cap on the raw image that gets base64'd into context (default 8 MB)
IMAGE_MAX_BYTES = _env_int("NX_IMAGE_MAX_BYTES", 8 * 1024 * 1024)

# Schema for the read_image tool. Kept terse — it's on every turn's floor
# when NX_VISION=1.
IMAGE_VISION_TOOL = {"type": "function", "function": {
    "name": "read_image",
    "description": "Show a local image (png/jpeg/gif/webp/bmp) to the model. Read-only. Use to inspect a screenshot/mockup/diagram the user references.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Path to the image file"},
    }, "required": ["path"]},
}}


def read_image(path: str, **_: Any) -> Any:
    """Return a content-parts array carrying the image, or an ERROR string."""
    if not isinstance(path, str) or not path.strip():
        return "ERROR: path is empty"
    ext = os.path.splitext(path)[1].lower()
    mime = _EXT_TO_MIME.get(ext)
    if mime is None:
        return (f"ERROR: {ext or '(no extension)'} is not a supported image format "
                f"(want one of: {', '.join(sorted(k.lstrip('.') for k in _EXT_TO_MIME))})")
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"
    if size > IMAGE_MAX_BYTES:
        return (f"ERROR: {path} is {size // 1024} KB; over the "
                f"{IMAGE_MAX_BYTES // (1024 * 1024)} MB context cap (NX_IMAGE_MAX_BYTES)")
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"
    b64 = base64.b64encode(data).decode("ascii")
    return [
        {"type": "text", "text": f"[image: {path}, {size} bytes]"},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]
