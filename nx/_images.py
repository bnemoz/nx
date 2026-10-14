"""read_image tool: hand a local image to a vision-capable model.

Returns an OpenAI content-parts array ({"type": "image_url", "image_url":
{"url": "data:image/<fmt>;base64,..."}}) as the tool result, which
_handle_native_tool_calls() stores in the tool message. Servers/models without
vision will error on the next request — that's the user's signal to not enable
NX_VISION on that backend (opt-in for exactly that reason; the read_image
schema is additionally gated on native tool_calls being CONFIRMED this
session, see _dispatch._active_tools — with 0.54's tool_choice="auto"
default that confirmation arrives on the first native call).

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
import binascii
import hashlib
import os
import time
from typing import Any

from ._env import _env_int
from ._private import private_dir

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
    "description": "Show a local image to the model as a picture: error screenshot, UI state, mockup, diagram. Returns the image itself, not a text description. png/jpeg/gif/webp/bmp. read_file handles image files too (same formats).",
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
    n_img = sum(1 for p in result if isinstance(p, dict)
                and p.get("type") in ("image_url", IMAGE_REF_TYPE))
    head = " ".join(t for t in texts if t)
    if n_img:
        head += (" " if head else "") + f"[image attached — {n_img} part(s), base64 payload not shown]"
    return head or f"[image attached — {len(result)} part(s), base64 payload not shown]"


def _strip_image_parts(msgs: list[dict[str, Any]], note: str | None = None) -> int:
    """Swap every image_url part in these messages' tool content for a text
    note, in place. Returns how many messages were changed.

    Used by the auto-compress path when a context-length 400 may have been
    CAUSED by image parts: keeping them verbatim in the retry tail would 400
    again and kill the turn. And by the turn loop once an image has actually
    been consumed (PERF.md PF-13) — a payload costs its megabytes on EVERY
    later request until something evicts it, so it is dropped the turn after it
    was first sent rather than at the next overflow. The model keeps the
    '[image: path, N bytes]' marker (it saw the picture once and summarized
    it), and the session's sidecar image store still holds the bytes for
    resume; only the wire copy goes. Mutating the dicts in place matters —
    compress()'s tail and the session writer hold references into the caller's
    live history."""
    if note is None:
        note = IMAGE_EVICTED_NOTE
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
                p["text"] = note
                hit = True
        if hit:
            changed += 1
    return changed


# The model-visible replacement for an image part, per eviction reason.
IMAGE_EVICTED_NOTE = ("[image evicted from context to fit the window — "
                      "re-read the file if needed]")
IMAGE_CONSUMED_NOTE = ("[image payload dropped after use to keep the context "
                       "small — re-read the file if you need to look again]")


def _strip_consumed_images(msgs: list[dict[str, Any]], sent_before: int) -> int:
    """Evict the base64 payload of every image the model has now SEEN (PF-13).

    `sent_before` is the message count when the request was sent, so only
    msgs[:sent_before] were on the wire — an image appended during the turn
    itself (a read_image call this very turn) has not been seen yet and is
    spared until the following turn. Call this right after model_turn() returns.
    """
    if sent_before < 0:
        sent_before = len(msgs)
    return _strip_image_parts(msgs[:sent_before], IMAGE_CONSUMED_NOTE)


# ---- session storage: payloads in a content-addressed sidecar, not in JSON --

# Store form of an image part, swapped in by _write_session (0.39.0): the
# base64 lives at <sessions_dir>/_images/<sha256>.b64 — deduped across every
# session, written once, garbage-collected to the live reference set — and the
# message carries only this reference. OFF via NX_SESSION_IMAGES=0, which
# restores 0.38.x inline-base64 storage.
IMAGE_REF_TYPE = "image_ref"


def _images_dir() -> str:
    """Sidecar dir next to the session files (same 0700 posture)."""
    from ._sessions_fs import _sessions_dir
    d = os.path.join(_sessions_dir(), "_images")
    private_dir(d)
    return d


def _data_url_to_ref(part: dict[str, Any], sibling_text: str | None = None) -> dict[str, Any] | None:
    """One image_url part → an image_ref part (payload into the sidecar).
    Returns None if the payload can't be parsed/stored (caller keeps or
    degrades the original part). Writes are idempotent per hash.
    sibling_text: the tool's own "[image: <path>, N bytes]" text part, when
    the caller has it — reused as best-effort provenance."""
    url = ""
    try:
        url = part["image_url"]["url"]
    except (KeyError, TypeError):
        return None
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    head, _, b64 = url.partition(",")
    mime = head[5:].split(";", 1)[0] or "image/png"
    try:
        raw = base64.b64decode(b64, validate=True)
    except (ValueError, binascii.Error):
        return None
    sha = hashlib.sha256(raw).hexdigest()
    side = os.path.join(_images_dir(), f"{sha}.b64")
    if not os.path.exists(side):
        # private 0600 write regardless of umask (sessions may hold secrets;
        # image payloads can too — a screenshot of someone's terminal)
        tmp = f"{side}.{os.getpid()}.tmp"
        try:
            fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="ascii") as f:
                f.write(b64)
            os.replace(tmp, side)
        except OSError:
            return None
    ref = {"type": IMAGE_REF_TYPE, "sha256": sha, "mime": mime,
           "bytes": len(raw)}
    if isinstance(sibling_text, str) and sibling_text.startswith("[image: ") and "," in sibling_text:
        ref["path"] = sibling_text[len("[image: "):].split(",", 1)[0].strip()
    return ref


def _ref_to_data_url_part(ref: dict[str, Any]) -> dict[str, Any] | None:
    """One image_ref part → a live image_url part, or None when the sidecar
    file is missing/corrupt (caller renders the degrade note)."""
    sha = ref.get("sha256")
    # reject anything that could traverse: separators (either kind) or dots
    if not isinstance(sha, str) or not sha:
        return None
    if "/" in sha or chr(92) in sha or "." in sha:
        return None
    try:
        with open(os.path.join(_images_dir(), f"{sha}.b64"), "rb") as f:
            b64 = f.read().decode("ascii")
        base64.b64decode(b64, validate=True)
    except (OSError, ValueError, binascii.Error):
        return None
    mime = ref.get("mime") or "image/png"
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _collect_message_image_refs(msgs: list[dict[str, Any]]) -> set[str]:
    """Every sha256 currently referenced by image_ref parts in `msgs` — the
    GC live-set for the sidecar dir."""
    out: set[str] = set()
    for m in msgs:
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list):
            continue
        for p in c:
            if isinstance(p, dict) and p.get("type") == IMAGE_REF_TYPE:
                sha = p.get("sha256")
                if isinstance(sha, str) and sha:
                    out.add(sha)
    return out


def _gc_image_sidecars(live: set[str], min_age_s: float = 0.0) -> int:
    """Delete sidecar payloads no message references anymore. Best-effort:
    a crash mid-GC loses nothing that is referenced; returns files removed.

    min_age_s: skip files younger than this many seconds — a GC running
    concurrently with another session's in-progress save must not delete a
    payload that save is about to reference (its ref is not on disk yet)."""
    try:
        names = os.listdir(_images_dir())
    except OSError:
        return 0
    removed = 0
    now = time.time()
    for name in names:
        if not name.endswith(".b64") or name[: -4] in live:
            continue
        side = os.path.join(_images_dir(), name)
        if min_age_s > 0:
            try:
                if now - os.path.getmtime(side) < min_age_s:
                    continue
            except OSError:
                continue
        try:
            os.remove(side)
        except OSError:
            pass
        else:
            removed += 1
    return removed
