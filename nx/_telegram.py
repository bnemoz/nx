"""Telegram gateway (0.79.0): drive nx from a Telegram chat — send it paper
links, uploaded PDFs, and plain instructions; it answers in the chat.

Posture: this is a *personal* single-chat gateway, not a multi-tenant bot.
One bot process = one conversation (module state is process-global, the same
constraint batch mode has), serialized with a lock so a second message while
a turn is running gets queued as "busy", never interleaved.

Security (mirrors the batch-mode contract in nx/_batch.py):
- NX_TELEGRAM_ALLOWED_CHATS is a hard allowlist of chat IDs. A message from
  any other chat is dropped without reply. An EMPTY allowlist disables the
  gateway entirely — the bot must never respond to a chat it wasn't told to.
- 0.95.1: NX_TELEGRAM_ALLOWED_USERS is a second hard allowlist of user IDs
  (default: same set as the chats — a chat's members are its users); a
  message whose from.id is not on it is dropped like a foreign chat. Group
  chats are refused unless NX_TELEGRAM_ALLOW_GROUPS=1: a group in the
  allowlist would otherwise let every member drive the host. Status bubbles
  echo tool arguments (commands, paths, queries) to Telegram's cloud — that
  is an intended disclosure of this personal gateway.
- Approval: the turn loop runs with the process's own approval posture, and
  since there is no terminal to answer a Y/n prompt, gated calls auto-DENY
  exactly like batch mode (the `_subagent_quiet` check in _approval makes
  prompts impossible, and YOLO is honored when the user opted in). Default
  posture for paper ingestion is therefore read+fetch with writes denied;
  run `nx telegram` under --yolo only in a vault you accept writes to.

Transport: raw Bot-API long polling over urllib (web_fetch already depends
on nothing richer). Outbound text is rendered as Telegram HTML (0.156.0:
the model's markdown becomes bold/code/links/fenced blocks instead of
literal asterisks, with a per-chunk plain-text fallback whenever Telegram
rejects the generated entities; a 4096-cut inside a code fence is closed
and reopened across chunks). Messages > 4096 chars are chunked; the phone gets ONE
status bubble per turn, edited in place at each model-turn boundary via
editMessageText through the on_step seam of _run_model_turn_loop. Since
0.95.0 the bubble is a summary instead of a full transcript diff; since
0.158.1 it is progress-only — the ×N-collapsed tool calls (freshest line as
the header), the latest assistant prose, any live sub-agent progress and
elapsed time, with no echo of the user's own message (it is already the
bubble right above) and a closing edit that drops the prose line so the
status never duplicates the answer. The model's own
auto-compression keeps a long session bounded; `/new` from the phone wipes
the conversation in place for a fresh context without a process restart.
"""
from __future__ import annotations

import base64
import html as _html
import json
import os
import re
import tempfile
import urllib.error
import threading
import time
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable

API_BASE = "https://api.telegram.org"
# Telegram hard limit on one message's text.
TG_MAX_MESSAGE = 4096
POLL_TIMEOUT_SECS = 30
# Local cap on attachment downloads (Telegram's own cap is nominally 20 MB;
# we enforce it rather than trust the remote side).
MAX_DOWNLOAD_BYTES = 21 * 1024 * 1024
# Cap on one instruction's text — a multi-MB message should not become one
# model turn.
MAX_INSTRUCTION_CHARS = 8192
# Minimum seconds between turns from one chat (replay/rate guard).
_TURN_COOLDOWN_SECS = 2.0
_LAST_TURN_AT: dict[int, float] = {}
# Group chats are refused unless this is set: a group in the allowlist would
# otherwise let every member drive the host.
_GROUP_FLAG = "NX_TELEGRAM_ALLOW_GROUPS"
# When set, status bubbles show tool ICONS ONLY — no arguments (commands,
# paths, queries). Those currently echo to Telegram's cloud by design; this
# flag is the opt-out for chats where even that is too much (0.153.3).
_REDACT_FLAG = "NX_TELEGRAM_REDACT_ARGS"
# Optional HTTPS/SOCKS proxy for the Bot API (the institute-firewall case in
# docs/TELEGRAM_SETUP.md). Empty = direct, as before. urllib honors http_proxy
# already; this adds an explicit, telegram-scoped knob. A socks5:// scheme
# needs PySocks installed; anything urllib cannot speak surfaces as a poll
# error at startup, not a silent hang.
_PROXY_FLAG = "NX_TELEGRAM_PROXY"
# Send back the images the model read/generated this turn via sendPhoto
# (screenshots, plots). 0/false/off disables — text-only gateway as before.
_PHOTOS_FLAG = "NX_TELEGRAM_SEND_PHOTOS"
# Never more than this many photos per turn (flood safety; each is a
# multipart upload against the per-chat rate limit).
_MAX_PHOTOS_PER_TURN = 2


def _groups_allowed() -> bool:
    return os.environ.get(_GROUP_FLAG, "").strip().lower() in ("1", "true", "yes")


def _redact_args() -> bool:
    return os.environ.get(_REDACT_FLAG, "").strip().lower() in ("1", "true", "yes")


def _proxy() -> str:
    return os.environ.get(_PROXY_FLAG, "").strip()


def _send_photos_enabled() -> bool:
    return os.environ.get(_PHOTOS_FLAG, "1").strip().lower() not in (
        "0", "false", "no", "off")


def _parse_ids(name: str) -> set[int]:
    """One env allowlist parser for every NX_TELEGRAM_*_IDS-style variable:
    comma/semicolon-separated ints, junk tokens ignored."""
    raw = os.environ.get(name, "")
    out: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part:
            try:
                out.add(int(part))
            except ValueError:
                pass
    return out


def _allowed_users(allowed: set[int]) -> set[int]:
    """User allowlist; empty means 'anyone in the allowed chats' (backwards
    compatible for private chats, where the chat IS the user)."""
    return _parse_ids("NX_TELEGRAM_ALLOWED_USERS") or allowed


# --- env (read lazily so tests can set NX_* after import) --------------------

def _token() -> str:
    return os.environ.get("NX_TELEGRAM_BOT_TOKEN", "").strip()


def _allowed_chats() -> set[int]:
    return _parse_ids("NX_TELEGRAM_ALLOWED_CHATS")


def _poll_interval() -> float:
    try:
        return max(0.0, float(os.environ.get("NX_TELEGRAM_POLL_INTERVAL", "1.0")))
    except ValueError:
        return 1.0


# --- Bot API surface (the only HTTP the gateway speaks) ----------------------

def _opener_for() -> urllib.request.OpenerDirector | None:
    """A proxy-routing opener when NX_TELEGRAM_PROXY is set, else None
    (plain urlopen). Shared by the JSON and multipart senders."""
    proxy = _proxy()
    if not proxy:
        return None
    return urllib.request.build_opener(urllib.request.ProxyHandler(
        {"http": proxy, "https": proxy}))


def _api(method: str, payload: dict | None = None, timeout: float = 35.0) -> dict:
    """One Bot-API call. getUpdates long-polls with `timeout` inside the
    payload; everything else is a plain POST. Returns the parsed JSON; the
    `ok` field is the caller's problem (fail-soft everywhere else).
    NX_TELEGRAM_PROXY routes every call through an HTTPS/SOCKS proxy (the
    firewall case); an unparseable/unreachable proxy fails the call like any
    network error — the poll loop already backs off and retries."""
    token = _token()
    if not token:
        raise RuntimeError("NX_TELEGRAM_BOT_TOKEN is not set")
    url = f"{API_BASE}/bot{token}/{method}"
    data = json.dumps(payload or {}).encode()
    opener = _opener_for()

    # The loop body either returns or re-raises, so `r` is always bound by the
    # time the fall-through `return r` below is reached; the empty default just
    # makes that visible to the checker.
    r: dict = {}
    for attempt in range(2):
        # NOTE: the token is part of the URL (Bot API contract) — never log
        # req.full_url or exception reprs that embed it.
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "User-Agent": "nx-telegram-gateway"})
        try:
            if opener is not None:
                cm = opener.open(req, timeout=timeout)
            else:
                cm = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
            with cm as resp:
                r = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            # 429 arrives as an HTTP error whose body is still JSON
            try:
                r = json.loads(e.read().decode("utf-8", errors="replace"))
            except Exception:
                r = None
            if r is None:
                # unparseable body (or none): a 429 still deserves its retry
                if attempt == 0 and getattr(e, "code", None) == 429:
                    time.sleep(3.0)
                    continue
                raise
        if not r.get("ok") and attempt == 0:
            wait = (r.get("parameters") or {}).get("retry_after")
            code = r.get("error_code")
            if wait or code == 429:
                # retry_after when given, else a small default (a 429 whose
                # parameters block is missing still floods if hammered)
                time.sleep(min(float(wait or 3.0), 30.0) + 0.5)
                continue
        return r
    return r


def _download_file(file_id: str) -> bytes:
    """Two-step download: getFile returns a relative file_path, then the file
    itself is a GET under /file/bot<token>/. 20 MB cap is Telegram's, and is
    plenty for preprints."""
    got = _api("getFile", {"file_id": file_id}, timeout=15.0)
    if not got.get("ok"):
        raise RuntimeError(f"getFile failed: {got.get('description')}")
    path = got["result"]["file_path"]
    token = _token()
    url = f"{API_BASE}/file/bot{token}/{path}"
    req = urllib.request.Request(url, method="GET",
                                 headers={"User-Agent": "nx-telegram-gateway"})
    with urllib.request.urlopen(req, timeout=60.0) as resp:  # noqa: S310
        raw = resp.read(MAX_DOWNLOAD_BYTES)
    if len(raw) >= MAX_DOWNLOAD_BYTES:
        raise RuntimeError("attachment exceeds the local download cap")
    return raw


# --- rendering helpers -------------------------------------------------------

def _u16len(s: str) -> int:
    """Telegram counts message length in UTF-16 code units, not code points
    (emoji are 2); len() under-counts, so the API 400s on emoji-heavy text
    that looks short enough locally."""
    return len(s.encode("utf-16-le")) // 2


def _chunk(text: str, limit: int = TG_MAX_MESSAGE) -> list[str]:
    """Split into Telegram-sized messages, preferring paragraph/code-fence
    boundaries over hard cuts so a fenced block is less likely to be split."""
    if _u16len(text) <= limit:
        return [text] if text else []
    chunks = []
    while _u16len(text) > limit:
        window = text[:limit]
        # Latest safe cut: a fence closer, else a paragraph break, else hard.
        cut = window.rfind("\n```")
        if cut > limit // 2:
            cut = window.find("\n", cut + 4)
            if cut == -1 or cut > limit:
                cut = limit
        else:
            cut = window.rfind("\n\n")
            if cut <= limit // 2:
                cut = window.rfind("\n")
                if cut <= limit // 2:
                    cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


# --- markdown -> Telegram HTML (parse_mode fallback rendering) ----------------

_FENCE_RE = re.compile(r"^\s*```(\w*)\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")


def _mono(text: str) -> str:
    """One block inside Telegram <pre>. ASCII tables (|---|) only stay
    aligned in a monospace block; zero-width spaces (from text wrapping)
    would widen the glyphs, so they go."""
    return "<pre>%s</pre>" % _html.escape(text.replace("\u200b", ""), quote=False)


def _table_html(lines: list[str], start: int) -> tuple[str, int]:
    """`lines[start]` is a `| a | b |` row and `lines[start+1]` the `|---|`
    separator: returns (Telegram HTML, next unconsumed line index). The
    separator's column count is authoritative (blank trailing cells vanish
    otherwise); rows are padded/truncated to it inside one <pre> so the
    pipes stay aligned. A non-row line ends the table."""
    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    ncol = len(cells(lines[start + 1]))
    rows = [cells(lines[start])]
    i = start + 2
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            rows.append(cells(lines[i]))
            i += 1
        else:
            break
    out = []
    for row in rows:
        padded = [_inline_md(c) for c in (row + [""] * ncol)[:ncol]]
        out.append("| " + " | ".join(padded) + " |")
    return _mono("\n".join(out)), i


def _inline_md(line: str) -> str:
    """One markdown line -> Telegram HTML. Order matters: inline code first
    (so its payload is protected from further subs), then links, bold,
    italic, headings, bullets. The line is HTML-escaped before any sub, so
    model output can never inject markup."""
    s = _html.escape(line, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
               lambda m: '<a href="%s">%s</a>'
                         % (m.group(2).replace('"', "%22"), m.group(1)), s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*\s][^*]*)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"^#{1,6} (.*)$", r"<b>\1</b>", s)
    s = re.sub(r"^[-*] ", "\u2022 ", s)
    return s


def _md_html(chunk: str, open_pre: bool = False) -> tuple[str, bool]:
    """Convert one chunk of markdown to Telegram HTML. `open_pre` means the
    previous chunk ended inside a fenced code block (a hard cut split it);
    this chunk therefore starts inside <pre>. Returns (html, still_open) —
    the caller closes a still-open fence and reopens it on the next chunk.
    Fence payloads are escaped verbatim (no markdown inside code)."""
    out: list[str] = []
    in_pre = open_pre
    if in_pre:
        out.append("<pre><code>")   # reopened: the fence opened last chunk
    lines = chunk.split("\n")
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        idx += 1
        m = _FENCE_RE.match(line)
        if in_pre:
            if m:
                in_pre = False
                out.append("</code></pre>")
            else:
                out.append(_html.escape(line, quote=False))
            continue
        if m:
            in_pre = True
            lang = m.group(1)
            out.append('<pre><code class="language-%s">' % lang if lang
                       else "<pre><code>")
            continue
        # A markdown table: this `| ... |` row followed by the `|---|` one.
        # Rendered as one block <pre> (the only place pipes stay aligned);
        # its lines are consumed wholesale.
        if idx < len(lines) and _TABLE_SEP_RE.match(lines[idx]) \
                and line.strip().startswith("|"):
            table_html, idx = _table_html(lines, idx - 1)
            out.append(table_html)
            continue
        out.append(_inline_md(line))
    return "\n".join(out), in_pre


# --- outbound media: the images a turn produced, sent back as photos ---------

def _part_image_bytes(part: dict) -> tuple[bytes, str] | None:
    """Image payload (bytes, mime) from one content part — a live image_url
    data URL or an image_ref whose sidecar file still exists — else None."""
    url = None
    if not isinstance(part, dict):
        return None
    if part.get("type") == "image_url":
        try:
            url = part["image_url"]["url"]
        except (KeyError, TypeError):
            return None
    elif part.get("type") == "image_ref":
        from ._images import _ref_to_data_url_part
        live = _ref_to_data_url_part(part)
        if live:
            url = live["image_url"]["url"]
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    head, _, b64 = url.partition(",")
    mime = head[5:].split(";", 1)[0] or "image/png"
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        return None
    if not raw:
        return None
    return raw, mime


def _turn_images(messages: list, turn_start: int) -> list[tuple[bytes, str]]:
    """Distinct images that appeared in this turn's tool results (read_image,
    rendered plots): at most _MAX_PHOTOS_PER_TURN, in order, deduped by data
    URL. Only parts that carry a real payload count."""
    out: list[tuple[bytes, str]] = []
    seen: set[str] = set()
    for m in messages[turn_start:]:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                key = (part.get("image_url") or {}).get("url") or ""
            elif part.get("type") == "image_ref":
                key = f"ref:{part.get('sha256')}"
            else:
                continue
            if key in seen:
                continue
            seen.add(key)
            got = _part_image_bytes(part)
            if got:
                out.append(got)
            if len(out) >= _MAX_PHOTOS_PER_TURN:
                return out
    return out


def _send_photo(api: Callable, chat_id: int, raw: bytes, mime: str,
                caption: str = "") -> bool:
    """One multipart sendPhoto upload (the only Bot-API call that is not
    JSON). Fail-soft: returns False on any failure — a photo is a bonus, a
    failed upload must not kill the turn or duplicate the answer."""
    ext = ".jpg" if "jpeg" in mime or mime == "image/jpg" else (
        ".gif" if "gif" in mime else ".png")
    boundary = f"nxtg{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in (("chat_id", str(chat_id)),
                        ("caption", caption[:1024])):
        if value:
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; "
         f'filename="image{ext}"\r\nContent-Type: {mime}\r\n\r\n')
        .encode() + raw + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    token = _token()
    req = urllib.request.Request(
        f"{API_BASE}/bot{token}/sendPhoto", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "nx-telegram-gateway"})
    try:
        opener = _opener_for()
        if opener is not None:
            cm = opener.open(req, timeout=60.0)
        else:
            cm = urllib.request.urlopen(req, timeout=60.0)  # noqa: S310
        with cm as resp:
            return bool(json.loads(resp.read().decode(
                "utf-8", errors="replace")).get("ok"))
    except Exception:
        return False


def _call_base(line: str) -> str:
    """A call line without any folded-on result suffix — the dedup key."""
    i = line.find("  ok")
    if i != -1:
        line = line[:i]
    j = line.find(" ×")
    if j != -1:
        line = line[:j]
    return line


def _describe_turn(messages: list, turn_start: int) -> tuple[list[str], int]:
    """A summary of the whole turn (0.95.0 contract, replaces
    the per-step transcript diff; 0.158.1 drops the ask echo): one line per
    tool call (consecutive identical ones collapse to ×N), plus the turn's
    tool-call total. Returns (lines, real tool call count)."""
    calls: list[str] = []
    total = 0
    for m in messages[turn_start:]:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments") or ""
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            line = _tool_line(fn.get("name") or "?", args)
            if calls and _call_base(calls[-1]) == _call_base(line):
                prev = calls[-1]
                j = prev.find(" ×")
                n = 2
                if j != -1:
                    n = int(prev[j + 2:].split(" ", 1)[0]) + 1
                calls[-1] = f"{_call_base(prev)} ×{n}"
            else:
                calls.append(line)
            total += 1
    return calls, total


_ELLIPSIS = "\u2026"


def _brief(value: str, limit: int = 64) -> str:
    """One human-readable line from a tool argument: collapse whitespace,
    shorten home paths, keep the first line of a command."""
    s = " ".join(str(value).split())
    s = s.replace(os.path.expanduser("~"), "~")
    if len(s) > limit:
        s = s[:limit - 1] + _ELLIPSIS
    return s


_TOOL_ICONS = {
    "read_file": "\U0001f4d6", "write_file": "\u270d",
    "edit_file": "\u270f", "multi_edit": "\u270f",
    "delete_file": "\U0001f5d1", "list_dir": "\U0001f4c2",
    "search_files": "\U0001f50d", "run_bash": "\u25b6",
    "todo_write": "\u2611", "web_fetch": "\U0001f310",
    "duckdb": "\U0001f5d4", "read_image": "\U0001f5bc",
    "git_commit": "\u2387", "git_status": "\u2387", "git_diff": "\u2387",
    "git_log": "\u2387", "run_tests": "\U0001f9ea",
    "search_tools": "\U0001f527",
}
_ICON_DEFAULT = "\U0001f527"


def _esc(s: str) -> str:
    """HTML-escape one fragment of a status line (0.158.2: the bubble is
    HTML like everything else we send; tool arguments are untrusted)."""
    return _html.escape(s, quote=False)


def _tool_line(name: str, args: dict) -> str:
    """One line for a tool call, e.g. `run_bash: cargo test -p somatiq-dp`.
    With NX_TELEGRAM_REDACT_ARGS=1 the arguments are withheld entirely —
    commands/paths/queries otherwise echo to Telegram's cloud."""
    icon = _TOOL_ICONS.get(name, _ICON_DEFAULT)
    if _redact_args():
        return f"{icon} <b>{_esc(name)}</b>"
    raw = str(args.get("command") or args.get("path") or args.get("url")
              or args.get("query") or "")
    first = raw.splitlines()[0] if raw.splitlines() else ""
    shown = _esc(_brief(first))
    return f"{icon} <b>{_esc(name)}</b>" + (f" <code>{shown}</code>" if shown else "")


# cap on how many recent lines the status message shows
_STATUS_TAIL = 14


def _compose_status(lines: list[str], tool_n: int,
                    note: str | None, elapsed: float | None,
                    progress: str | None = None) -> str:
    """The whole status message (0.158.1): progress only — no echo of the
    user's ask. The freshest call line is the header, earlier lines are the
    body (with an overflow line), then a footer tally."""
    if lines:
        head = lines[-1]
        rest = lines[:-1]      # head is the last line; never duplicated
        shown = rest[-_STATUS_TAIL:]
        overflow = len(lines) - len(shown) - 1
    else:
        head = "⏳ <b>Working…</b>"
        rest = shown = []
        overflow = 0
    body_lines = ([f"\u2026 <i>({overflow} earlier calls)</i>"] if overflow > 0
                  else []) + shown
    body = "\n".join(body_lines)
    footer = f"total: <b>{tool_n}</b> tool calls"
    if progress:
        footer = f"🤖 <i>{_esc(progress)}</i> \u00b7 " + footer
    if elapsed is not None:
        footer += f" \u00b7 <b>{int(elapsed)}s</b>"
    return (head + ("\n" + body if body else "") + "\n\n" + footer
            + (f"\n\U0001f4ac <i>{_esc(note)}</i>" if note else ""))


def _typing(api: Callable, chat_id: int) -> None:
    """sendChatAction: typing — fail-soft; keeps the phone from looking dead
    during a long turn."""
    try:
        api("sendChatAction", {"chat_id": chat_id, "action": "typing"})
    except Exception:
        pass


def _edit_status(api: Callable, chat_id: int, message_id: int,
                 text: str) -> bool:
    """One editMessageText on the status bubble. Returns False on any
    failure (message deleted, too old, network) so callers can fall back to
    sendMessage."""
    try:
        r = api("editMessageText", {"chat_id": chat_id,
                                    "message_id": message_id,
                                    "text": text, "parse_mode": "HTML"})
        return bool(r.get("ok"))
    except Exception:
        return False


def _send(api: Callable, chat_id: int, text: str) -> dict:
    """Send (chunked), rendering markdown as Telegram HTML (bold, italic,
    inline code, fenced blocks, links) with parse_mode=HTML. Any chunk
    Telegram refuses to parse falls back to its plain text — the model's
    markdown is never allowed to break delivery. A code fence split by the
    4096 chunker is closed and reopened so both halves still render as a
    block. Fail-soft: the returned dict records ok/error instead of raising,
    but failures are never silently indistinguishable from success. Chunks
    are throttled to stay under the per-chat flood limit."""
    out: dict[str, Any] = {"message_id": None, "ok": True}
    open_pre = False
    for i, chunk in enumerate(_chunk(text) or ["(empty)"]):
        if i:
            time.sleep(0.6)
        rendered, open_pre = _md_html(chunk, open_pre)
        if open_pre:
            rendered += "</code></pre>"
        payload = {"chat_id": chat_id, "text": rendered, "parse_mode": "HTML"}
        try:
            r = api("sendMessage", payload)
            if not r.get("ok") and "parse" in (r.get("description") or "").lower():
                r = api("sendMessage", {"chat_id": chat_id, "text": chunk})
                if r.get("ok"):
                    rendered = chunk   # fallback: record what actually rendered
            if r.get("ok"):
                out["message_id"] = (r.get("result") or {}).get("message_id")
            else:
                out["ok"] = False
                out["error"] = r.get("description") or "sendMessage failed"
        except Exception as e:
            out["ok"], out["error"] = False, f"{type(e).__name__}: {e}"
    return out


# --- per-message ingestion ----------------------------------------------------

def _message_instruction(msg: dict) -> str | list | None:
    """Translate one Telegram update's message into (instruction, attachments):
    a PDF document becomes an inline extracted-text attachment; a photo (with
    NX_VISION on) becomes content parts carrying the image itself — the model
    sees the picture, same as read_image/@mentions (0.156.1). With NX_VISION
    off the photo is skipped with a one-line note. Text passes through
    verbatim (URLs included — the model fetches them via web_fetch).
    Returns None for messages with nothing actionable."""
    text = (msg.get("text") or msg.get("caption") or "").strip()
    if len(text) > MAX_INSTRUCTION_CHARS:
        raise ValueError(
            f"instruction is {len(text)} chars; the cap is "
            f"{MAX_INSTRUCTION_CHARS} — split it or send a shorter ask")
    doc = msg.get("document") or {}
    name = doc.get("file_name") or ""
    mime = doc.get("mime_type") or ""
    photo = msg.get("photo") or []
    if photo and not isinstance(photo, list):
        photo = []
    atts: list[str] = []
    if doc and ("pdf" in mime or name.lower().endswith(".pdf")):
        raw = _download_file(doc["file_id"])
        fd, tmp = tempfile.mkstemp(suffix=".pdf", prefix="nx-tg-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            from ._pdf import extract_pdf
            atts.append(extract_pdf(tmp))
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    image_parts: list[dict] = []
    if photo:
        from . import NX_VISION  # module global; live value, not import-time
        if not NX_VISION:
            atts.append("(a photo was sent — not attached: NX_VISION is off, so "
                        "this nx session cannot receive pictures; set "
                        "NX_VISION=1 on a native-tool-call backend, or describe "
                        "the image in text)")
        else:
            # `photo` is an array of sizes ascending; the last is the largest.
            # Telegram photos are always JPEG.
            raw = _download_file(photo[-1]["file_id"])
            fd, tmp = tempfile.mkstemp(suffix=".jpg", prefix="nx-tg-photo-")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(raw)
                from ._images import read_image
                out = read_image(tmp)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            if isinstance(out, list):
                image_parts.extend(out)
            else:
                atts.append(f"(photo download failed: {out})")
    if not atts and not image_parts and not text:
        return None
    body = ""
    if atts:
        body += "\n\n".join(atts) + "\n\n"
    if text:
        body += f"Instruction from the user: {text}"
    body = body.strip()
    if image_parts:
        # Content parts: the instruction text first, then the picture(s) —
        # the same shape read_image returns and the turn loop already sends.
        return [{"type": "text", "text": body or "(photo)"}] + image_parts
    return body or None


# --- the turn (one instruction -> one Telegram reply) -------------------------

_TURN_LOCK = threading.Lock()


def _run_turn(api: Callable, chat_id: int, instruction: str,
              messages: list) -> None:
    """Run one model turn for `instruction`, editing a status message with
    step progress, then sending the final answer. Mutates `messages` (the
    persistent conversation)."""
    from . import _state as st
    from ._dispatch import _initial_messages
    from ._turn import _ensure_ctx_probe, _run_model_turn_loop
    from ._repl_state import _fresh_metrics

    if st.ACTIVE is None:
        _send(api, chat_id, "nx: no source configured — run `nx setup` on the host.")
        return
    _ensure_ctx_probe(st.ACTIVE)

    fresh = not messages
    if fresh:
        messages.extend(_initial_messages())
    messages.append({"role": "user", "content": instruction})

    # No empty "total: 0 tool calls" bubble before any work: the status
    # bubble is created lazily at the first progress edit (or by the final
    # answer when the turn was a pure one-shot reply).
    status_id = None
    turn_start = len(messages)   # activity re-renders the whole turn
    # 0.158.1: the status bubble is progress-only; no echo of the ask.
    started = time.monotonic()
    # live sub-agent feedback (0.95.0): spawn_agent's per-step note_progress
    # lands here so the phone sees `🤖 agent turn 3/100` while it waits.
    progress_note: list[str | None] = [None]

    def _sink(text: str) -> None:
        if text:
            progress_note[0] = text
            on_step()

    def _render(quiet: bool = False) -> tuple[str, int, str | None]:
        """Status body (0.158.1, progress-only): tool calls (×N-collapsed),
        the latest assistant prose, elapsed time. `quiet` drops the live
        notes: the closing edit has nothing left to report live."""
        calls, tool_n = _describe_turn(messages, turn_start)
        note = None
        for m in reversed(messages):
            if m.get("role") == "assistant" and not m.get("tool_calls"):
                text = (m.get("content") or "").strip()
                if text:
                    note = _brief(text, 160)
                    break
        return _compose_status(calls, tool_n, None if quiet else note,
                               time.monotonic() - started,
                               None if quiet else progress_note[0]), tool_n, note

    def on_step() -> None:
        # progress is best-effort; never kill the turn over it
        nonlocal status_id
        try:
            text, tool_n, note = _render()
            if not (tool_n or note):
                return
            if status_id is None:
                sent = _send(api, chat_id, text)
                status_id = sent.get("message_id")
                _typing(api, chat_id)
            else:
                _edit_status(api, chat_id, status_id, text)
        except Exception:
            pass

    # Approval auto-deny, as documented: without the quiet slot, _confirm
    # falls through to a terminal prompt there is nobody to answer (and on a
    # TTY, a stray Enter approves). Batch mode sets the same slot.
    from ._subagent import _ASYNC_QUIET_SLOTS
    from . import _state as _st
    _ASYNC_QUIET_SLOTS.quiet = True
    prev_sink = _st._PROGRESS_SINK
    _st._PROGRESS_SINK = _sink
    summary = {}
    try:
        try:
            _typing(api, chat_id)
            summary = _run_model_turn_loop(messages, on_step=on_step)
        except Exception as e:
            _send(api, chat_id, f"nx: turn failed — {type(e).__name__}: "
                                f"{str(e)[:200]}")
            return
        finally:
            st._METRICS = _fresh_metrics(session_id="telegram")
    finally:
        _st._PROGRESS_SINK = prev_sink
        _ASYNC_QUIET_SLOTS.quiet = False
        _repair_conversation(messages)
        # Phase C mirror (0.161.0, AUTOWAKE_SPEC §3: Phases C/D are mirrored
        # here for consistency and get NO Telegram-only behavior): the same
        # shared post-turn seam the REPL's finally calls. `summary` is {} until
        # the loop returns, so a failed turn reads as "not done" and clears the
        # stall clock rather than arming it on a broken turn.
        try:
            from . import _autowake
            _autowake.note_turn_end(
                isinstance(summary, dict) and summary.get("status") == "done")
            # Phase D mirror (0.162.0): the chain cost seam, beside the stall
            # one (AUTOWAKE_SPEC §3 keeps the mirror behaviorally identical).
            _autowake.note_chain_spend()
        except Exception:
            pass

    # The final answer is the last non-empty assistant text, skipping any
    # trailing tool-call message (an empty assistant tail must not hide a
    # real answer further up).
    answer = None
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if isinstance(content, str) and content.strip():
            answer = content.strip()
            break
        if m.get("tool_calls"):
            continue   # look further up for a real answer
    # 0.95.0 delivery contract: the final answer is ALWAYS its own new
    # message (chunked, push notification, guaranteed delivery), and the
    # status bubble is left as the summary. The old edit-on-status contract
    # (0.91.4) was the reported bug: an edit of an already-edited bubble can
    # silently not render (rate-limit / "message is not modified" /
    # client-side staleness), so the answer sat invisible until the next
    # turn's activity re-edited the bubble.
    if not answer:
        answer = "(no final answer — status: %s)" % (summary or {}).get("status", "?")
    # The status bubble's last live edit still carries the 💬 prose line, which
    # by now is a truncation of the answer about to arrive in full, plus a
    # stale sub-agent note. Re-render it without either so the two bubbles
    # never read as the same message twice: bubble 1 ends as a clean activity
    # log, bubble 2 is the report. Fail-soft — the answer is already settled
    # and an un-editable status (deleted, rate-limited) is cosmetic.
    if status_id is not None:
        try:
            closing, _n, _note = _render(quiet=True)
            _edit_status(api, chat_id, status_id, closing)
        except Exception:
            pass
    _send(api, chat_id, answer)
    # 0.153.4: the pictures the turn produced (a read_image the user asked
    # about, a rendered plot) go back to the phone after the answer text —
    # the guaranteed-delivery contract above still comes first. Opt out with
    # NX_TELEGRAM_SEND_PHOTOS=0.
    if _send_photos_enabled():
        for raw, mime in _turn_images(messages, turn_start):
            _send_photo(api, chat_id, raw, mime)


# --- main loop ----------------------------------------------------------------

def _offset_path() -> str:
    """Where the getUpdates offset persists across restarts. Keyed by a hash
    of the bot token: two bots on one host keep independent offsets."""
    import hashlib
    from . import _state
    h = hashlib.sha256(_token().encode()).hexdigest()[:16]
    return os.path.join(_state.SESSION_HOME, "telegram-offset-%s" % h)


def _load_offset() -> int:
    try:
        with open(_offset_path(), encoding="ascii") as f:
            return max(0, int(f.read().strip()))
    except (OSError, ValueError):
        return 0


def _save_offset(offset: int) -> None:
    try:
        from . import _state
        os.makedirs(_state.SESSION_HOME, exist_ok=True)
        tmp = f"{_offset_path()}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="ascii") as f:
            f.write(str(offset))
        os.replace(tmp, _offset_path())
    except OSError:
        pass   # persistence is best-effort; the gateway runs without it


def run_gateway() -> int:
    """`nx telegram`: long-poll getUpdates forever; each allowed-chat message
    becomes one serialized turn. Ctrl-C exits cleanly. The update offset is
    persisted under NX_HOME (0.153.5): a restart resumes where the last
    process left off instead of replaying (or missing) the downtime window —
    Telegram buffers updates for 24h."""
    from . import _state as st
    if not _token():
        print("nx telegram: NX_TELEGRAM_BOT_TOKEN is not set.")
        return 2
    allowed = _allowed_chats()
    if not allowed:
        print("nx telegram: NX_TELEGRAM_ALLOWED_CHATS is empty — refusing to "
              "serve an open bot. Add your chat ID (message @userinfobot or "
              "check getUpdates once with the allowlist temporarily open).")
        return 2

    # `nx telegram` runs before the REPL's switch_source; do it here.
    if st.ACTIVE is None:
        import nx as _m
        try:
            _m._discover_sources()
        except Exception:
            pass
        for name in st.SOURCE_ORDER or []:
            if _m.switch_source(name):
                break
    if st.ACTIVE is None:
        print("nx telegram: no source configured — run `nx setup`.")
        return 2
    if not _groups_allowed():
        print("nx telegram: only private chats are served "
              f"(set {_GROUP_FLAG}=1 to serve a group you control).")
    _LAST_TURN_AT.clear()

    print(f"nx telegram: polling for chats {sorted(allowed)} …")
    offset = _load_offset()
    while True:
        try:
            r = _api("getUpdates",
                     {"offset": offset, "timeout": POLL_TIMEOUT_SECS,
                      "allowed_updates": ["message"]})
        except KeyboardInterrupt:
            print("\nnx telegram: stopped.")
            return 0
        except Exception:
            # Network hiccup / timeout: back off briefly and keep polling.
            time.sleep(_poll_interval())
            continue
        _maybe_wake_turn(_api, allowed)
        if not r.get("ok"):
            code = (r.get("parameters") or {}).get("error_code") \
                or r.get("error_code")
            if code in (401, 404):
                print(f"\nnx telegram: token rejected ({code}) — "
                      f"{r.get('description')}: stopping.")
                return 2
            time.sleep(_poll_interval())
            continue
        for upd in r.get("result") or []:
            offset = max(offset, upd["update_id"] + 1)
            _save_offset(offset)
            try:
                _handle_update(upd, allowed)
            except Exception as e:
                # one bad update (flaky upload, malformed message) must not
                # kill the gateway
                print(f"nx telegram: update dropped: {type(e).__name__}: "
                      f"{str(e)[:200]}")


def _maybe_wake_turn(api: Callable, allowed: set) -> None:
    """Autowake (0.153.0): mirror the REPL's auto-continuation turn. The wake
    poller (_autowake) raises st._WAKE_DUE when a background sub-agent / bg
    job finishes while the gateway idles, but nothing in this module consumed
    that flag — the completion note sat staged until the user typed something
    (the first tool call of their turn flushed it). Between polls the loop is
    idle and _TURN_LOCK is free, so synthesize the REPL's wake turn here:
    sweep, flush the staged notice into the conversation, run a turn on the
    wake instruction. A real Telegram message arriving concurrently wins the
    lock and this no-ops."""
    from . import _autowake
    from . import _state as st
    from ._dispatch import _run_bg_completion_scan
    from ._runtime_notes import _flush_pending_background_notice
    try:
        if not st._WAKE_DUE or not _autowake.wake_due():
            return
        if not _TURN_LOCK.acquire(blocking=False):
            return  # a real user turn is running; it drains the note itself
        try:
            st._WAKE_DUE = False
            st._TURN_ACTIVE = True
            try:
                st._WAKE_STREAK = getattr(st, "_WAKE_STREAK", 0) + 1
                # Phase D mirror (0.162.0): chain cost baseline at the same
                # seam the REPL snapshots it (see nx/_repl.py).
                st._WAKE_CHAIN_BASE = int(getattr(st, "_LAST_PROMPT_TOKENS", 0) or 0)
                # Phase C mirror: a stall wake stamps the plan revision it woke
                # for, exactly as the REPL does (_repl.py, next to the streak
                # increment) — that int is the trigger's terminator and the
                # baseline note_turn_end() compares against. The branch
                # decision comes from this call's out-param, not the shared
                # marker (poller race, review fix 0.161.0). The outer try's
                # finally releases _TURN_LOCK on every path, including this
                # early return.
                _stall_decision: dict = {}
                if not _autowake.wake_due(_stall_decision):
                    st._TURN_ACTIVE = False
                    return
                _stall_wake = bool(_stall_decision.get("stall"))
                if _stall_wake:
                    from ._todo import _plan_revision
                    st._WAKE_STALL_REVISION = _plan_revision()
                _run_bg_completion_scan()
                _flush_pending_background_notice(_CONVERSATION)
                wake = _autowake.wake_instruction()
            finally:
                st._TURN_ACTIVE = False
            print("nx telegram: background work finished — auto-continuing")
            _run_turn(api, next(iter(sorted(allowed))), wake, _CONVERSATION)
            if _autowake._any_pending_notice():
                st._WAKE_DUE = True
        finally:
            _TURN_LOCK.release()
    except Exception:
        pass  # a wake turn must never take the gateway down


# One conversation for the process (the single-chat constraint, stated in the
# module docstring). Survives across messages; a bot restart starts fresh.
_CONVERSATION: list = []


def _handle_update(upd: dict, allowed: set, allowed_users: set | None = None) -> None:
    """Allowlist checks, instruction translation, /new handling, one turn.
    Raises on bad updates; run_gateway catches per-update."""
    msg = upd.get("message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id not in allowed:
        return  # not even an error: an open reply is the vulnerability
    if not isinstance(chat_id, int):
        # The allowlist holds ints only (see _allowed_chats), so a non-int id
        # can never have reached here — the guard exists for the checker.
        return
    if chat.get("type") not in ("private", None) and not _groups_allowed():
        return
    if allowed_users is None:
        allowed_users = _allowed_users(allowed)
    user_id = (msg.get("from") or {}).get("id")
    if user_id is None:
        user_id = chat_id   # no sender info: a private chat IS the user
    if user_id not in allowed_users:
        return
    text = (msg.get("text") or "").strip()
    if text == "/new":
        with _TURN_LOCK:
            _CONVERSATION.clear()
        _send(_api, chat_id, "✂ conversation cleared — fresh context.")
        return
    # Minimal replay/rate guard: a chat that just ran a turn must wait a beat
    # before another (a crash-looping client or a flooded chat otherwise
    # queues unbounded model-token spend). /new is exempt. A message inside
    # the window is NOT dropped silently: it gets a busy notice (the
    # docstring always claimed queuing; 0.153.2 makes it true).
    now = time.monotonic()
    last = _LAST_TURN_AT.get(chat_id)
    if last is not None and now - last < _TURN_COOLDOWN_SECS:
        _send(_api, chat_id, "⏳ still working on the previous message — "
                             "resend it in a moment.")
        return
    try:
        instruction = _message_instruction(msg)
    except ValueError as e:
        _send(_api, chat_id, f"nx: {e}")
        return
    if not instruction:
        return
    # Queue behind a running turn instead of interleaving or dropping: the
    # lock serializes turns; waiting here (bounded) keeps a message sent
    # mid-turn from vanishing. The cooldown above already bounds how deep
    # the queue can get from one chat.
    with _TURN_LOCK:
        _LAST_TURN_AT[chat_id] = now
        # Phase C mirror: a real inbound message is the gateway's typed-input
        # path, so it clears the stall clock / deliberate-stop hold the same way
        # the REPL's does (AUTOWAKE_SPEC §3 keeps the mirror behaviorally
        # identical; the streak reset itself predates Phase C and is left
        # exactly as it was).
        try:
            from . import _autowake
            _autowake.clear_stall_state()
            # Phase D mirror (0.162.0): an inbound message is the gateway's
            # typed-input path — the chain cost accumulators reset with it,
            # exactly as the REPL's typed path does.
            _autowake.reset_chain_cost()
        except Exception:
            pass
        _run_turn(_api, chat_id, instruction, _CONVERSATION)


def _repair_conversation(messages: list) -> None:
    """Drop assistant tool_calls whose results never arrived — a failed turn
    can leave them dangling, and most OpenAI-compatible endpoints then reject
    EVERY subsequent turn until the prefix is cleaned (or /new)."""
    answered = {m.get("tool_call_id") for m in messages
                if m.get("role") == "tool"}
    messages[:] = [
        m for m in messages
        if not (m.get("role") == "assistant" and m.get("tool_calls")
                and any((tc.get("id") not in answered)
                        for tc in m["tool_calls"]))]


def cli_telegram(argv: list[str]) -> int:
    """Entry point from nx/_cli.py main()."""
    try:
        return run_gateway()
    except KeyboardInterrupt:
        print("\nnx telegram: stopped.")
        return 0
