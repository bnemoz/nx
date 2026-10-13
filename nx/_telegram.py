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
0.95.0 the bubble is a first-and-last summary (the user's ask, the ×N-
collapsed tool calls, the latest assistant prose, any live sub-agent
progress, elapsed time) instead of a full transcript diff — the
first and the final message are what matter on a phone. The model's own
auto-compression keeps a long session bounded; `/new` from the phone wipes
the conversation in place for a fresh context without a process restart.
"""
from __future__ import annotations

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


def _groups_allowed() -> bool:
    return os.environ.get(_GROUP_FLAG, "").strip().lower() in ("1", "true", "yes")


def _allowed_users(allowed: set[int]) -> set[int]:
    """User allowlist; empty means 'anyone in the allowed chats' (backwards
    compatible for private chats, where the chat IS the user)."""
    return _allowed_chats_named("NX_TELEGRAM_ALLOWED_USERS") or allowed


def _allowed_chats_named(name: str) -> set[int]:
    raw = os.environ.get(name, "")
    out = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part:
            try:
                out.add(int(part))
            except ValueError:
                pass
    return out


# --- env (read lazily so tests can set NX_* after import) --------------------

def _token() -> str:
    return os.environ.get("NX_TELEGRAM_BOT_TOKEN", "").strip()


def _allowed_chats() -> set[int]:
    raw = os.environ.get("NX_TELEGRAM_ALLOWED_CHATS", "")
    out = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part:
            try:
                out.add(int(part))
            except ValueError:
                pass
    return out


def _poll_interval() -> float:
    try:
        return max(0.0, float(os.environ.get("NX_TELEGRAM_POLL_INTERVAL", "1.0")))
    except ValueError:
        return 1.0


# --- Bot API surface (the only HTTP the gateway speaks) ----------------------

def _api(method: str, payload: dict | None = None, timeout: float = 35.0) -> dict:
    """One Bot-API call. getUpdates long-polls with `timeout` inside the
    payload; everything else is a plain POST. Returns the parsed JSON; the
    `ok` field is the caller's problem (fail-soft everywhere else)."""
    token = _token()
    if not token:
        raise RuntimeError("NX_TELEGRAM_BOT_TOKEN is not set")
    url = f"{API_BASE}/bot{token}/{method}"
    data = json.dumps(payload or {}).encode()
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
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                r = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            # 429 arrives as an HTTP error whose body is still JSON
            try:
                r = json.loads(e.read().decode("utf-8", errors="replace"))
            except Exception:
                if attempt == 0 and getattr(e, "code", None) == 429:
                    time.sleep(3.0)
                    continue
                raise
        if not r.get("ok") and attempt == 0:
            wait = (r.get("parameters") or {}).get("retry_after")
            if wait:
                time.sleep(min(float(wait), 30.0) + 0.5)
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
    for line in chunk.split("\n"):
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
        out.append(_inline_md(line))
    return "\n".join(out), in_pre


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
    """A first-and-last summary of the whole turn (0.95.0 contract, replaces
    the per-step transcript diff): the user's ask (truncated), one line per
    tool call (consecutive identical ones collapse to ×N), and the turn's
    most recent assistant prose. Returns (lines, real tool call count)."""
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


def _tool_line(name: str, args: dict) -> str:
    """One line for a tool call, e.g. `run_bash: cargo test -p somatiq-dp`."""
    icon = _TOOL_ICONS.get(name, _ICON_DEFAULT)
    raw = str(args.get("command") or args.get("path") or args.get("url")
              or args.get("query") or "")
    first = raw.splitlines()[0] if raw.splitlines() else ""
    shown = _brief(first)
    return f"{icon} {name}" + (f": {shown}" if shown else "")


# cap on how many recent lines the status message shows
_STATUS_TAIL = 14


def _compose_status(ask: str, lines: list[str], tool_n: int,
                    note: str | None, elapsed: float | None,
                    progress: str | None = None) -> str:
    """The whole status message (0.95.0): the user's ask, the latest tool
    activity, the most recent assistant prose, a live sub-agent progress
    note, and a footer tally."""
    shown = lines[-_STATUS_TAIL:]
    if len(lines) > _STATUS_TAIL:
        shown.insert(0, f"\u2026 ({len(lines) - _STATUS_TAIL} earlier calls)")
    body = "\n".join(shown)
    footer = f"total: {tool_n} tool calls"
    if progress:
        footer = f"🤖 {progress} · " + footer
    if elapsed is not None:
        footer += f" \u00b7 {int(elapsed)}s"
    return (f"\u2753 {ask}\n" + body + "\n\n" + footer
            + (f"\n\U0001f4ac {note}" if note else ""))


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
                                    "text": text})
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
        except Exception as e:
            out["ok"], out["error"] = False, f"{type(e).__name__}: {e}"
            continue
        if r.get("ok"):
            out["message_id"] = (r.get("result") or {}).get("message_id")
        else:
            out["ok"] = False
            out["error"] = r.get("description") or "sendMessage failed"
    return out


# --- per-message ingestion ----------------------------------------------------

def _message_instruction(msg: dict) -> str | None:
    """Translate one Telegram update's message into (instruction, attachments):
    a PDF document becomes an inline extracted-text attachment; text passes
    through verbatim (URLs included — the model fetches them via web_fetch).
    Returns None for messages with nothing actionable."""
    text = (msg.get("text") or msg.get("caption") or "").strip()
    if len(text) > MAX_INSTRUCTION_CHARS:
        raise ValueError(
            f"instruction is {len(text)} chars; the cap is "
            f"{MAX_INSTRUCTION_CHARS} — split it or send a shorter ask")
    doc = msg.get("document") or {}
    name = doc.get("file_name") or ""
    mime = doc.get("mime_type") or ""
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
    if not atts and not text:
        return None
    body = ""
    if atts:
        body += "\n\n".join(atts) + "\n\n"
    if text:
        body += f"Instruction from the user: {text}"
    return body.strip()


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
    ask = _brief(instruction, 100)
    started = time.monotonic()
    # live sub-agent feedback (0.95.0): spawn_agent's per-step note_progress
    # lands here so the phone sees `🤖 agent turn 3/100` while it waits.
    progress_note: list[str | None] = [None]

    def _sink(text: str) -> None:
        if text:
            progress_note[0] = text
            on_step()

    def _render() -> tuple[str, int, str | None]:
        """First-and-last status body: the ask, tool calls (×N-collapsed),
        the latest assistant prose, elapsed time."""
        calls, tool_n = _describe_turn(messages, turn_start)
        note = None
        for m in reversed(messages):
            if m.get("role") == "assistant" and not m.get("tool_calls"):
                text = (m.get("content") or "").strip()
                if text:
                    note = _brief(text, 160)
                    break
        return _compose_status(ask, calls, tool_n, note,
                               time.monotonic() - started,
                               progress_note[0]), tool_n, note

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
    _send(api, chat_id, answer)


# --- main loop ----------------------------------------------------------------

def run_gateway() -> int:
    """`nx telegram`: long-poll getUpdates forever; each allowed-chat message
    becomes one serialized turn. Ctrl-C exits cleanly."""
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
    offset = 0
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
    # queues unbounded model-token spend). /new is exempt.
    now = time.monotonic()
    last = _LAST_TURN_AT.get(chat_id, 0.0)
    if now - last < _TURN_COOLDOWN_SECS:
        return
    try:
        instruction = _message_instruction(msg)
    except ValueError as e:
        _send(_api, chat_id, f"nx: {e}")
        return
    if not instruction:
        return
    with _TURN_LOCK:
        _LAST_TURN_AT[chat_id] = now
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
