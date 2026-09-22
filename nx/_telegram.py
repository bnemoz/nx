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
- Approval: the turn loop runs with the process's own approval posture, and
  since there is no terminal to answer a Y/n prompt, gated calls auto-DENY
  exactly like batch mode (the `_subagent_quiet` check in _approval makes
  prompts impossible, and YOLO is honored when the user opted in). Default
  posture for paper ingestion is therefore read+fetch with writes denied;
  run `nx telegram` under --yolo only in a vault you accept writes to.

Transport: raw Bot-API long polling over urllib (web_fetch already depends
on nothing richer). Messages > 4096 chars are chunked; progress edits use
editMessageText on a status message updated once per model-turn boundary via
the on_step seam of _run_model_turn_loop. The model's own auto-compression
keeps a long session bounded; `/new` from the phone wipes the conversation
in place for a fresh context without a process restart.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.parse
import urllib.request

API_BASE = "https://api.telegram.org"
# Telegram hard limit on one message's text.
TG_MAX_MESSAGE = 4096
POLL_TIMEOUT_SECS = 30


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
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": "nx-telegram-gateway"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8", errors="replace"))


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
        return resp.read()


# --- rendering helpers -------------------------------------------------------

def _chunk(text: str, limit: int = TG_MAX_MESSAGE) -> list[str]:
    """Split into Telegram-sized messages, preferring paragraph/code-fence
    boundaries over hard cuts so a fenced block is less likely to be split."""
    if len(text) <= limit:
        return [text] if text else []
    chunks = []
    while len(text) > limit:
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


def _call_base(line: str) -> str:
    """A call line without any folded-on result suffix — the dedup key."""
    i = line.find("  ok")
    if i != -1:
        line = line[:i]
    j = line.find(" ×")
    if j != -1:
        line = line[:j]
    return line


def _describe_step(messages: list, seen_len: int) -> list[str]:
    """Human-readable lines for everything appended to `messages` since the
    last model-turn boundary (the on_step seam gives no payload; the
    transcript diff is the payload). Assistant prose gets its own line; tool
    calls and results pair up. Returns [] when the step added nothing.

    Clutter rules (0.91.3): a plain-`ok` result folds into the preceding
    call line (sized results attach their count) instead of adding its own
    line; consecutive identical call lines collapse to `name: args ×N`."""
    lines: list[str] = []
    pending_tool: list[int] = []   # indices of call lines awaiting a result
    for m in messages[seen_len:]:
        role = m.get("role")
        if role == "assistant":
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
                base = _call_base(line)
                if lines and _call_base(lines[-1]) == base:
                    prev = lines[-1]
                    n = 2
                    j = base.find(" ×")
                    if j != -1:
                        n = int(prev[j + 2:].split(" ", 1)[0]) + 1
                    # keep any folded result suffix after the new ×N
                    suffix = ""
                    k = prev.find("  ok")
                    if k != -1:
                        suffix = prev[k:]
                    lines[-1] = f"{base} ×{n}{suffix}"
                    pending_tool.append(len(lines) - 1)
                else:
                    lines.append(line)
                    pending_tool.append(len(lines) - 1)
            text = (m.get("content") or "")
            if isinstance(text, str) and text.strip():
                lines.append("💬 " + _brief(text.strip(), 120))
        elif role == "tool":
            line = _tool_result_line(m)
            # A plain-`ok` result: fold it into the call line instead of
            # emitting a separate `ok` line (sized results attach their count).
            if pending_tool and (line == "ok" or
                                 line.startswith("ok (")) and lines:
                idx = pending_tool.pop()
                if line != "ok":
                    p = lines[idx].find("  ok")
                    stripped = lines[idx] if p == -1 else lines[idx][:p]
                    lines[idx] = stripped + "  " + line
                continue
            if pending_tool:
                pending_tool.pop()
            if line:
                lines.append(line)
    return lines


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


def _tool_result_line(role_msg: dict) -> str | None:
    """One line for a tool result: a short OK, or the meaningful tail of
    output (errors and test summaries live in the tail)."""
    body = role_msg.get("content")
    if isinstance(body, list):
        return "ok (image)"
    body = str(body or "")
    n = len(body)
    if not body.strip() or n <= 200:
        return "ok"
    tail = next((ln.strip() for ln in reversed(body.splitlines())
                 if ln.strip()), "")
    tail = _brief(tail, 60)
    return f"ok ({n} chars)" + (f" \u00b7 {tail}" if tail else "")


# cap on how many recent lines the status message shows
_STATUS_TAIL = 14


def _compose_status(lines: list[str], tool_n: int, note_n: int) -> str:
    """The whole status message: recent activity lines + a footer tally."""
    shown = lines[-_STATUS_TAIL:]
    if len(lines) > _STATUS_TAIL:
        shown.insert(0, f"\u2026 ({len(lines) - _STATUS_TAIL} earlier lines)")
    body = "\n".join(shown)
    tally = f"total: {tool_n} tool calls"
    if note_n:
        tally += f", {note_n} notes"
    return body + "\n\n" + tally


def _send(api: callable, chat_id: int, text: str) -> dict:
    out = {"message_id": None}
    for chunk in _chunk(text) or ["(empty)"]:
        r = api("sendMessage", {"chat_id": chat_id, "text": chunk})
        if r.get("ok"):
            out["message_id"] = r["result"]["message_id"]
    return out


# --- per-message ingestion ----------------------------------------------------

def _message_instruction(msg: dict) -> str | None:
    """Translate one Telegram update's message into (instruction, attachments):
    a PDF document becomes an inline extracted-text attachment; text passes
    through verbatim (URLs included — the model fetches them via web_fetch).
    Returns None for messages with nothing actionable."""
    text = (msg.get("text") or msg.get("caption") or "").strip()
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


def _run_turn(api: callable, chat_id: int, instruction: str,
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

    status = _send(api, chat_id, _compose_status([], 0, 0))
    status_id = status.get("message_id")
    turn_start = len(messages)   # activity re-renders the whole turn
    activity: list[str] = []   # every line so far this turn
    tool_n = 0                 # tool calls so far
    note_n = 0                 # assistant prose notes so far

    def on_step() -> None:
        nonlocal tool_n, note_n, activity
        activity = _describe_step(messages, turn_start)
        if not activity:
            return
        tool_n = sum(1 for m in messages[turn_start:]
                     if m.get("role") == "assistant"
                     for _ in (m.get("tool_calls") or []))
        note_n = sum(1 for ln in activity if ln.startswith("💬"))
        if status_id:
            try:
                api("editMessageText", {"chat_id": chat_id,
                                        "message_id": status_id,
                                        "text": _compose_status(
                                            activity, tool_n, note_n)})
            except Exception:
                pass  # progress is best-effort; never kill the turn over it

    summary = {}
    try:
        summary = _run_model_turn_loop(messages, on_step=on_step)
    except Exception as e:
        _send(api, chat_id, f"nx: turn failed — {type(e).__name__}: {e}")
        return
    finally:
        st._METRICS = _fresh_metrics(session_id="telegram")

    # The final answer is the tail of assistant text after the last tool call.
    answer = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            if m.get("tool_calls"):
                break
            content = m.get("content") or ""
            if isinstance(content, str) and content.strip():
                answer = content.strip()
            break
    if answer:
        _send(api, chat_id, answer)
    else:
        _send(api, chat_id,
              "(no final answer — status: %s)" % (summary or {}).get("status", "?"))


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
        except Exception as e:
            # Network hiccup / timeout: back off briefly and keep polling.
            time.sleep(_poll_interval())
            continue
        if not r.get("ok"):
            time.sleep(_poll_interval())
            continue
        for upd in r.get("result") or []:
            offset = max(offset, upd["update_id"] + 1)
            msg = upd.get("message") or {}
            chat_id = (msg.get("chat") or {}).get("id")
            if chat_id not in allowed:
                continue  # not even an error: an open reply is the vulnerability
            instruction = _message_instruction(msg)
            if not instruction:
                continue
            if instruction.rstrip().endswith("/new"):
                # matched after the "Instruction from the user: " wrapper;
                # a PDF+"/new" is honored too (the intent is unambiguous).
                with _TURN_LOCK:
                    _CONVERSATION.clear()
                _send(_api, chat_id, "✂ conversation cleared — fresh context.")
                continue
            with _TURN_LOCK:
                _run_turn(_api, chat_id, instruction, _CONVERSATION)


# One conversation for the process (the single-chat constraint, stated in the
# module docstring). Survives across messages; a bot restart starts fresh.
_CONVERSATION: list = []


def cli_telegram(argv: list[str]) -> int:
    """Entry point from nx/_cli.py main()."""
    try:
        return run_gateway()
    except KeyboardInterrupt:
        print("\nnx telegram: stopped.")
        return 0
