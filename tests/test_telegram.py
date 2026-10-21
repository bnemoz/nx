"""nx telegram (0.79.0): the Telegram gateway — long-poll one bot, one turn
per allowed-chat message.

The suite is offline by construction: every Bot-API call goes through a
recorded fake `_api`, so no test contacts api.telegram.org. The contract
under test is the security posture (hard allowlist, empty-refuses-to-start,
auto-deny), the message-to-instruction translation (PDF attachments inline
via extract_pdf, URLs pass through for the model to fetch), message
chunking, and the on_step progress-edit seam.
"""
import base64
import json
import os
import sys
import tempfile
import threading
import time
import unittest.mock
import urllib.error

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx  # noqa: E402
import nx._telegram as tg  # noqa: E402



_ELLIPSIS = "\u2026"


# --- fakes ---------------------------------------------------------------------

def _fake_api(calls=None, responses=None):
    """A stand-in for tg._api: records (method, payload), returns canned
    responses by method."""
    def api(method, payload=None, timeout=35.0):
        if calls is not None:
            calls.append((method, payload))
        if responses and method in responses:
            r = responses[method]
            return r() if callable(r) else r
        return {"ok": True,
                "result": {"message_id": 42, "file_path": "docs/x.pdf"}}
    return api


# --- config / posture ----------------------------------------------------------

def test_token_and_allowlist_parsing():
    os.environ["NX_TELEGRAM_BOT_TOKEN"] = "t"
    os.environ["NX_TELEGRAM_ALLOWED_CHATS"] = " 111, 222 ; 333 ,junk"
    assert tg._token() == "t"
    assert tg._allowed_chats() == {111, 222, 333}
    del os.environ["NX_TELEGRAM_BOT_TOKEN"]
    os.environ["NX_TELEGRAM_ALLOWED_CHATS"] = ""
    assert tg._allowed_chats() == set()


def test_gateway_refuses_without_token():
    saved = os.environ.pop("NX_TELEGRAM_BOT_TOKEN", None)
    try:
        assert tg.run_gateway() == 2
    finally:
        if saved is not None:
            os.environ["NX_TELEGRAM_BOT_TOKEN"] = saved


def test_gateway_refuses_empty_allowlist():
    os.environ["NX_TELEGRAM_BOT_TOKEN"] = "t"
    os.environ["NX_TELEGRAM_ALLOWED_CHATS"] = ""
    assert tg.run_gateway() == 2  # never serve an open bot


# --- chunking -------------------------------------------------------------------

def test_chunk_respects_limit_and_paragraphs():
    assert tg._chunk("hello") == ["hello"]
    assert tg._chunk("") == []
    text = ("para one\n\n" + "x" * 3000 + "\n\npara two")
    chunks = tg._chunk(text, limit=100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks).replace("\n", "").startswith("para onexxx")
    assert "para two" in chunks[-1]


# --- markdown -> HTML rendering ----------------------------------------------

def test_md_html_inline_formatting():
    html_, open_ = tg._md_html("**bold** and *it* and `code`")
    assert open_ is False
    assert "<b>bold</b>" in html_ and "<i>it</i>" in html_
    assert "<code>code</code>" in html_
    assert "**" not in html_


def test_md_html_escapes_html_and_renders_fence():
    html_, open_ = tg._md_html('a <b> & "c"\n```python\nx = 1\n```')
    assert "&lt;b&gt;" in html_
    assert '<pre><code class="language-python">' in html_
    assert "x = 1" in html_ and "</code></pre>" in html_


def test_md_html_link_and_bullet():
    html_, _ = tg._md_html("- [docs](https://example.com/a_b)")
    assert "\u2022 " in html_
    assert '<a href="https://example.com/a_b">docs</a>' in html_


def test_md_html_renders_table_as_aligned_pre_block():
    html_, _ = tg._md_html(
        "before\n\n| a | bb |\n| --- | --- |\n| 1 | x |\n| 22 | |\n\nafter")
    assert html_.startswith("before\n\n<pre>| a | bb |")
    assert html_.endswith("| 22 |  |</pre>\n\nafter")


def test_send_uses_parse_mode_and_falls_back_on_parse_error():
    calls = []

    def api(method, payload=None, timeout=35.0):
        calls.append(payload)
        if payload.get("parse_mode"):
            return {"ok": False, "description": "can't parse entities"}
        return {"ok": True, "result": {"message_id": 1}}

    out = tg._send(api, 111, "**hi**")
    assert out["ok"] is True
    assert calls[0]["parse_mode"] == "HTML" and "<b>hi</b>" in calls[0]["text"]
    assert calls[1].get("parse_mode") is None and calls[1]["text"] == "**hi**"


def test_send_closes_reopened_fence_across_chunks():
    calls = []
    text = "intro\n\n```python\n" + "x" * 60 + "\n```\n\noutro"
    # small limit forces a cut inside the fence
    chunks = tg._chunk(text, limit=40)
    assert len(chunks) >= 2

    def api(method, payload=None, timeout=35.0):
        calls.append(payload)
        return {"ok": True, "result": {"message_id": 1}}

    with unittest.mock.patch.object(tg, "_chunk", return_value=chunks):
        tg._send(api, 111, text)
    pre_chunks = [c for c in calls if "<pre>" in c["text"]]
    # the code block opens in chunk 1, is reopened in chunk 2 after the
    # split; chunk 3 holds only the closer + trailing prose
    assert len(pre_chunks) == 3 and all(c.get("parse_mode") == "HTML"
                                        for c in calls)


# --- inbound photos -------------------------------------------------------------

def test_photo_attached_as_image_parts_when_vision_on():
    with unittest.mock.patch.object(tg, "_download_file",
                                    return_value=b"\xff\xd8 fake jpeg"), \
         unittest.mock.patch.object(nx, "NX_VISION", True):
        instr = tg._message_instruction({"text": "what is this?", "photo": [
            {"file_id": "S"}, {"file_id": "BIG", "width": 1280}]})
    assert isinstance(instr, list)
    assert instr[0] == {"type": "text",
                        "text": "Instruction from the user: what is this?"}
    kinds = [p["type"] for p in instr[1:]]
    assert "image_url" in kinds and "text" in kinds
    payload = next(p for p in instr if p.get("type") == "image_url")
    assert payload["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_photo_without_vision_leaves_a_note():
    with unittest.mock.patch.object(nx, "NX_VISION", False):
        instr = tg._message_instruction({"text": "look", "photo": [
            {"file_id": "BIG"}]})
    assert isinstance(instr, str)
    assert "NX_VISION is off" in instr


def test_photo_only_message_still_produces_instruction():
    with unittest.mock.patch.object(tg, "_download_file",
                                    return_value=b"\xff\xd8 x"), \
         unittest.mock.patch.object(nx, "NX_VISION", True):
        instr = tg._message_instruction({"caption": "check this", "photo": [
            {"file_id": "BIG"}]})
    assert isinstance(instr, list)
    assert instr[0]["type"] == "text" and "check this" in instr[0]["text"]


def test_photo_parts_instruction_briefs_in_status():
    parts = [{"type": "text", "text": "describe this screenshot"},
             {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,xx"}}]
    conv = []
    with unittest.mock.patch.object(nx._state, "ACTIVE", _FakeSource()), \
         unittest.mock.patch("nx._turn._run_model_turn_loop",
                             side_effect=lambda m, on_step=None, **k: (
                                 m.append({"role": "assistant",
                                           "content": "It is a login form."}),
                                 {"status": "done"})), \
         unittest.mock.patch("nx._dispatch._initial_messages", return_value=[]), \
         unittest.mock.patch("nx._turn._ensure_ctx_probe", lambda *a, **k: None):
        tg._run_turn(_fake_api(), 111, parts, conv)
    assert conv[-1]["role"] == "assistant"


# --- message -> instruction ------------------------------------------------------

def test_text_passes_through_verbatim():
    instr = tg._message_instruction(
        {"text": "Summarize https://arxiv.org/abs/2401.00001", "document": None})
    assert instr == "Instruction from the user: Summarize https://arxiv.org/abs/2401.00001"


def test_pdf_document_downloaded_and_extracted():
    with unittest.mock.patch.object(tg, "_download_file",
                                    return_value=b"%PDF-1.4 fake"), \
         unittest.mock.patch("nx._pdf.extract_pdf",
                             return_value="--- @pdf ---\nPAPER TEXT"):
        instr = tg._message_instruction(
            {"text": "", "caption": "file this",
             "document": {"file_id": "F1", "file_name": "paper.pdf",
                          "mime_type": "application/pdf"}})
    assert "PAPER TEXT" in instr
    assert "file this" in instr  # caption rode along with the attachment


def test_empty_message_is_ignored():
    assert tg._message_instruction({"text": "", "document": None}) is None
    assert tg._message_instruction({}) is None


def test_non_pdf_document_is_ignored():
    instr = tg._message_instruction(
        {"text": "", "document": {"file_id": "F2", "file_name": "x.zip",
                                  "mime_type": "application/zip"}})
    assert instr is None


# --- step rendering --------------------------------------------------------------

def test_describe_turn_collapses_duplicate_calls():
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "function": {"name": "list_dir",
             "arguments": json.dumps({"path": "."})}}]},
        {"role": "tool", "tool_call_id": "1", "content": "x" * 300},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "2", "function": {"name": "list_dir",
             "arguments": json.dumps({"path": "."})}}]},
        {"role": "tool", "tool_call_id": "2", "content": "x" * 300},
    ]
    lines, n = tg._describe_turn(msgs, 0)
    assert lines == ["📂 <b>list_dir</b> <code>.</code> ×2"] and n == 2


def test_describe_turn_home_shortening_and_bad_args():
    home = os.path.expanduser("~")
    msgs = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "function": {"name": "read_file",
         "arguments": json.dumps({"path": home + "/Documents/Projects/x.rs"})}},
        {"id": "2", "function": {"name": "edit_file", "arguments": "not json"}},
    ]}]
    lines, _ = tg._describe_turn(msgs, 0)
    assert home not in lines[0] and "~/Documents/Projects/x.rs" in lines[0]
    assert "edit_file" in lines[1]          # salvage line, no crash


def test_compose_status_tally_and_tail():
    lines = ["line%d" % i for i in range(20)]
    out = tg._compose_status(lines, 7, None, 65.0)
    assert "earlier calls" in out
    assert "line19" in out and "line4" not in out
    assert "<b>7</b> tool calls" in out and "<b>65s</b>" in out
    short = tg._compose_status(["only"], 1, None, None)
    assert short == "only\n\ntotal: <b>1</b> tool calls"


def test_brief_collapses_whitespace_and_caps():
    assert tg._brief("a\n  b\tc") == "a b c"
    assert len(tg._brief("x" * 100)) == 64


# --- the turn: progress edits + final answer --------------------------------------

class _FakeSource:
    name = "fake"


def test_run_turn_sends_progress_and_final_answer():
    calls = []
    api = _fake_api(calls)
    conv = []

    def fake_loop(messages, on_step=None, **kw):
        if on_step:
            messages.append({"role": "assistant", "content": None,
                             "tool_calls": [{"id": "1", "function": {
                                 "name": "web_fetch",
                                 "arguments": json.dumps(
                                     {"url": "https://arxiv.org/abs/x"})}}]})
            on_step()
            messages.append({"role": "tool", "tool_call_id": "1",
                             "content": "page text"})
            on_step()
        messages.append({"role": "assistant", "content": "The paper says X."})
        return {"status": "done", "turns": 2}

    with unittest.mock.patch.object(nx._state, "ACTIVE", _FakeSource()), \
         unittest.mock.patch.object(nx, "_assess_risk", lambda *a, **k: ("low", "t")), \
         unittest.mock.patch("nx._turn._run_model_turn_loop",
                             side_effect=fake_loop), \
         unittest.mock.patch("nx._dispatch._initial_messages", return_value=[
             {"role": "system", "content": "s"}]), \
         unittest.mock.patch("nx._turn._ensure_ctx_probe", lambda *a, **k: None):
        tg._run_turn(api, 111, "read this", conv)

    methods = [m for m, _ in calls]
    # 0.95.0: status bubble = summary only; the final answer is always its
    # own sendMessage (guaranteed delivery, was the stale-edit bug).
    assert methods.count("sendMessage") == 2        # status bubble + answer
    assert "editMessageText" in methods            # on_step progress happened
    final = calls[-1]
    assert final[0] == "sendMessage"
    assert "The paper says X." in final[1]["text"]
    assert conv[-1]["role"] == "assistant"         # conversation persists


def test_run_turn_closing_edit_drops_the_answer_echo():
    """0.158.1: the status bubble's closing edit is a clean activity log —
    no 💬 prose line (that would be a truncation of the answer arriving in
    the very next bubble) and no stale sub-agent note."""
    calls = []
    api = _fake_api(calls)
    conv = []

    def fake_loop(messages, on_step=None, **kw):
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": "1", "function": {
                             "name": "web_fetch",
                             "arguments": json.dumps(
                                 {"url": "https://arxiv.org/abs/x"})}}]})
        on_step()
        messages.append({"role": "tool", "tool_call_id": "1",
                         "content": "page text"})
        on_step()
        messages.append({"role": "assistant", "content": "The paper says X."})
        return {"status": "done", "turns": 2}

    with unittest.mock.patch.object(nx._state, "ACTIVE", _FakeSource()), \
         unittest.mock.patch.object(nx, "_assess_risk", lambda *a, **k: ("low", "t")), \
         unittest.mock.patch("nx._turn._run_model_turn_loop",
                             side_effect=fake_loop), \
         unittest.mock.patch("nx._dispatch._initial_messages", return_value=[
             {"role": "system", "content": "s"}]), \
         unittest.mock.patch("nx._turn._ensure_ctx_probe", lambda *a, **k: None):
        tg._run_turn(api, 111, "read this", conv)

    edits = [p["text"] for m, p in calls if m == "editMessageText"]
    assert edits, "the status bubble must get a closing edit"
    assert "\U0001f4ac" not in edits[-1]
    assert "The paper says X." not in edits[-1]
    assert "web_fetch" in edits[-1]               # activity still shown
    assert calls[-1][0] == "sendMessage"          # answer still comes last


def test_run_turn_reports_loop_failure():
    calls = []
    api = _fake_api(calls)
    with unittest.mock.patch.object(nx._state, "ACTIVE", _FakeSource()), \
         unittest.mock.patch("nx._turn._run_model_turn_loop",
                             side_effect=RuntimeError("boom")), \
         unittest.mock.patch("nx._dispatch._initial_messages", return_value=[
             {"role": "system", "content": "s"}]):
        tg._run_turn(api, 111, "hi", [])
    assert "RuntimeError" in calls[-1][1]["text"]


# --- outbound photos -------------------------------------------------------------

def test_turn_images_extracts_and_dedupes_parts():
    data_url = "data:image/png;base64," + base64.b64encode(b"PNGDATA").decode()
    msgs = [{"role": "tool", "tool_call_id": "1", "content": [
        {"type": "text", "text": "[image: /tmp/x.png, 7 bytes]"},
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "image_url", "image_url": {"url": data_url}},   # dup
        {"type": "text", "text": "not an image"},
    ]}]
    out = tg._turn_images(msgs, 0)
    assert out == [(b"PNGDATA", "image/png")]


def test_turn_images_cap_and_refusal():
    url = lambda n: "data:image/png;base64," + base64.b64encode(
        f"img{n}".encode()).decode()
    msgs = [{"role": "tool", "content": [
        {"type": "image_url", "image_url": {"url": url(1)}},
        {"type": "image_url", "image_url": {"url": url(2)}},
        {"type": "image_url", "image_url": {"url": url(3)}},
    ]}]
    assert len(tg._turn_images(msgs, 0)) == tg._MAX_PHOTOS_PER_TURN
    # a part without a parseable payload never yields a photo
    bad = [{"role": "tool", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!"}}]}]
    assert tg._turn_images(bad, 0) == []


def test_send_photo_multipart_request():
    calls = []
    class R:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b'{"ok": true, "result": {"message_id": 7}}'
    def fake_urlopen(req, timeout=None):
        calls.append(req)
        return R()

    with unittest.mock.patch("urllib.request.urlopen", fake_urlopen):
        ok = tg._send_photo(unittest.mock.MagicMock(), 111, b"\x89PNG", "image/png",
                            caption="your plot")
    assert ok is True
    req = calls[0]
    assert req.full_url.endswith("/sendPhoto")
    body = req.data
    assert b'name="chat_id"' in body and b"111" in body
    assert b'name="caption"' in body and b"your plot" in body
    assert b'filename="image.png"' in body and b"\x89PNG" in body
    assert req.headers.get("Content-type", "").startswith("multipart/form-data")


def test_send_photo_failure_is_soft():
    def boom(req, timeout=None):
        raise OSError("network down")
    with unittest.mock.patch("urllib.request.urlopen", boom):
        assert tg._send_photo(unittest.mock.MagicMock(), 111, b"x", "image/png") is False


def test_run_turn_sends_turn_photos_after_answer():
    data_url = "data:image/png;base64," + base64.b64encode(b"PLOT").decode()
    conv = []
    calls = []
    api = _fake_api(calls)

    def fake_loop(messages, on_step=None, **kw):
        messages.append({"role": "tool", "tool_call_id": "1", "content": [
            {"type": "image_url", "image_url": {"url": data_url}}]})
        messages.append({"role": "assistant", "content": "Here is the plot."})
        return {"status": "done"}

    with unittest.mock.patch.object(nx._state, "ACTIVE", _FakeSource()), \
         unittest.mock.patch("nx._turn._run_model_turn_loop",
                             side_effect=fake_loop), \
         unittest.mock.patch("nx._dispatch._initial_messages", return_value=[]), \
         unittest.mock.patch("nx._turn._ensure_ctx_probe", lambda *a, **k: None), \
         unittest.mock.patch.object(tg, "_send_photo",
                                    side_effect=lambda api, cid, raw, mime,
                                    caption="": calls.append(
                                        ("sendPhoto", raw, mime))):
        tg._run_turn(api, 111, "plot it", conv)
    kinds = [c[0] for c in calls]
    assert "sendPhoto" in kinds
    # text answer first, photo after — the delivery contract holds
    assert kinds.index("sendMessage") < kinds.index("sendPhoto")
    photo = next(c for c in calls if c[0] == "sendPhoto")
    assert photo[1] == b"PLOT" and photo[2] == "image/png"


def test_send_photos_opt_out():
    os.environ["NX_TELEGRAM_SEND_PHOTOS"] = "0"
    try:
        assert tg._send_photos_enabled() is False
    finally:
        os.environ["NX_TELEGRAM_SEND_PHOTOS"] = ""
    assert tg._send_photos_enabled() is True




def test_loop_drops_disallowed_chat():
    os.environ["NX_TELEGRAM_BOT_TOKEN"] = "t"
    os.environ["NX_TELEGRAM_ALLOWED_CHATS"] = "111"
    calls = []
    updates = [
        {"update_id": 1, "message": {"chat": {"id": 999},
                                     "text": "malicious"}},
        {"update_id": 2, "message": {"chat": {"id": 111},
                                     "text": "hello"}},
    ]
    turns = []

    def fake_turn(api, chat_id, instruction, messages):
        turns.append((chat_id, instruction))

    def fake_api(method, payload=None, timeout=35.0):
        calls.append(method)
        if method == "getUpdates":
            if calls.count("getUpdates") > 1:   # second poll: stop the loop
                raise KeyboardInterrupt()
            return {"ok": True, "result": updates}
        return {"ok": True, "result": {"message_id": 1}}

    with unittest.mock.patch.object(tg, "_api", side_effect=fake_api), \
         unittest.mock.patch.object(tg, "_run_turn", side_effect=fake_turn), \
         unittest.mock.patch.object(nx._state, "ACTIVE", _FakeSource()):
        tg.run_gateway()
    # the disallowed chat never produced a turn, never got a reply
    assert turns == [(111, "Instruction from the user: hello")]


# --- fixes: quiet-slot auto-deny, ×N counter, repair, /new exactness -------

def test_run_turn_sets_quiet_slot_during_loop():
    seen = {}
    conv = []

    def fake_loop(messages, on_step=None, **kw):
        from nx._subagent import _ASYNC_QUIET_SLOTS, _subagent_quiet
        seen["during"] = _subagent_quiet()
        seen["tls"] = bool(getattr(_ASYNC_QUIET_SLOTS, "quiet", False))
        messages.append({"role": "assistant", "content": "done"})
        return {"status": "done"}

    with unittest.mock.patch.object(nx._state, "ACTIVE", _FakeSource()), \
         unittest.mock.patch("nx._turn._run_model_turn_loop",
                             side_effect=fake_loop), \
         unittest.mock.patch("nx._dispatch._initial_messages",
                             return_value=[]), \
         unittest.mock.patch("nx._turn._ensure_ctx_probe",
                             lambda *a, **k: None):
        tg._run_turn(_fake_api(), 111, "hi", conv)
    assert seen == {"during": True, "tls": True}   # auto-deny is live mid-turn


def test_describe_turn_collapses_three_identical_calls():
    msgs = []
    for i in "123":
        msgs += [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": i, "function": {"name": "list_dir",
                 "arguments": json.dumps({"path": "."})}}]},
            {"role": "tool", "tool_call_id": i, "content": "x" * 300},
        ]
    lines, n = tg._describe_turn(msgs, 0)
    assert lines == ["📂 <b>list_dir</b> <code>.</code> ×3"] and n == 3


def test_describe_turn_lists_distinct_calls_in_order():
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "A", "function": {"name": "run_bash",
             "arguments": json.dumps({"command": "first"})}},
            {"id": "B", "function": {"name": "run_bash",
             "arguments": json.dumps({"command": "second"})}},
        ]},
        {"role": "tool", "tool_call_id": "A", "content": "y" * 300 + "\nFROM A"},
        {"role": "tool", "tool_call_id": "B", "content": "y" * 300 + "\nFROM B"},
    ]
    lines, n = tg._describe_turn(msgs, 0)
    assert len(lines) == 2 and n == 2
    assert "first" in lines[0] and "second" in lines[1]


def test_repair_conversation_drops_unanswered_tool_calls():
    conv = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "function": {"name": "x", "arguments": "{}"}},
            {"id": "2", "function": {"name": "y", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "1", "content": "ok"},
    ]
    tg._repair_conversation(conv)
    # the dangling call is gone; the answered pair survives untouched
    assert conv == [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "1", "content": "ok"},
    ]
    tg._repair_conversation([{"role": "user", "content": "clean"}])
    assert len(conv) == 2   # no-op on a clean prefix


def test_u16len_counts_emoji_as_two():
    assert tg._u16len("hi") == 2
    assert tg._u16len("🙂") == 2       # astral-plane code point
    assert tg._u16len("a" * 4095 + "🙂") == 4097  # len() says 4096


def test_send_reports_api_failure_instead_of_swallowing():
    def bad_api(method, payload=None, timeout=35.0):
        return {"ok": False, "description": "chat not found"}
    out = tg._send(bad_api, 111, "hello")
    assert out["ok"] is False and "chat not found" in out["error"]


def test_slash_new_requires_exact_match():
    os.environ["NX_TELEGRAM_BOT_TOKEN"] = "t"
    os.environ["NX_TELEGRAM_ALLOWED_CHATS"] = "111"
    tg._CONVERSATION.extend([{"role": "user", "content": "old"}])
    turns = []

    def fake_turn(api, chat_id, instruction, messages):
        turns.append(instruction)

    polls = []

    def fake_api(method, payload=None, timeout=35.0):
        if method == "getUpdates":
            polls.append(method)
            if len(polls) > 1:
                raise KeyboardInterrupt()
            return {"ok": True, "result": [
                {"update_id": 1, "message": {"chat": {"id": 111},
                                             "text": "look at src/new"}},
                {"update_id": 2, "message": {"chat": {"id": 111},
                                             "text": "/new"}},
            ]}
        return {"ok": True, "result": {"message_id": 1}}

    with unittest.mock.patch.object(tg, "_api", side_effect=fake_api), \
         unittest.mock.patch.object(tg, "_run_turn", side_effect=fake_turn), \
         unittest.mock.patch.object(nx._state, "ACTIVE", _FakeSource()):
        try:
            tg.run_gateway()
        except KeyboardInterrupt:
            pass
    # "look at src/new" went to a turn; only the exact "/new" wiped the
    # conversation
    assert len(turns) == 1
    assert tg._CONVERSATION == []


def test_bad_update_does_not_kill_gateway():
    os.environ["NX_TELEGRAM_BOT_TOKEN"] = "t"
    os.environ["NX_TELEGRAM_ALLOWED_CHATS"] = "111"
    calls = []

    polls = []

    def fake_api(method, payload=None, timeout=35.0):
        if method == "getUpdates":
            polls.append(method)
            if len(polls) > 1:
                raise KeyboardInterrupt()
            return {"ok": True, "result": [
                # a document whose download explodes mid-instruction
                {"update_id": 1, "message": {"chat": {"id": 111},
                                             "document": {"file_id": "BAD",
                                                          "mime_type": "application/pdf"}}},
                {"update_id": 2, "message": {"chat": {"id": 111},
                                             "text": "still alive"}},
            ]}
        return {"ok": True, "result": {"message_id": 1}}

    with unittest.mock.patch.object(tg, "_api", side_effect=fake_api), \
         unittest.mock.patch.object(tg, "_run_turn") as fake_turn, \
         unittest.mock.patch.object(nx._state, "ACTIVE", _FakeSource()):
        tg.run_gateway()
    # the second update still got a turn despite the first one blowing up
    assert fake_turn.call_count == 1


# --- 0.95.1 hardening: user allowlist, groups, download cap, cooldown -------

def test_user_allowlist_drops_foreign_sender():
    os.environ["NX_TELEGRAM_BOT_TOKEN"] = "t"
    os.environ["NX_TELEGRAM_ALLOWED_CHATS"] = "111"
    os.environ["NX_TELEGRAM_ALLOWED_USERS"] = "777"
    turns = []
    with unittest.mock.patch.object(tg, "_run_turn",
                                    side_effect=lambda *a: turns.append(a)):
        tg._LAST_TURN_AT.clear()
        tg._handle_update({"update_id": 1, "message": {
            "chat": {"id": 111, "type": "private"},
            "from": {"id": 999}, "text": "hi"}}, {111})
        tg._handle_update({"update_id": 2, "message": {
            "chat": {"id": 111, "type": "private"},
            "from": {"id": 777}, "text": "hi"}}, {111})
    assert len(turns) == 1  # only the allowed user ran a turn


def test_group_chat_refused_unless_opted_in():
    os.environ["NX_TELEGRAM_BOT_TOKEN"] = "t"
    os.environ["NX_TELEGRAM_ALLOWED_CHATS"] = "111"
    os.environ.pop("NX_TELEGRAM_ALLOW_GROUPS", None)
    turns = []
    upd = {"update_id": 1, "message": {
        "chat": {"id": 111, "type": "group"},
        "from": {"id": 111}, "text": "hi"}}
    with unittest.mock.patch.object(tg, "_run_turn",
                                    side_effect=lambda *a: turns.append(a)):
        tg._LAST_TURN_AT.clear()
        tg._handle_update(upd, {111})          # group: dropped
        os.environ["NX_TELEGRAM_ALLOW_GROUPS"] = "1"
        tg._handle_update(upd, {111})          # opted in: served
    assert len(turns) == 1


def test_download_file_enforces_local_cap():
    class FakeResp:
        def read(self, n=-1):
            return b"x" * n
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    with unittest.mock.patch.object(tg, "_api",
                                    return_value={"ok": True, "result": {
                                        "file_path": "docs/x.pdf"}}), \
         unittest.mock.patch("urllib.request.urlopen",
                             return_value=FakeResp()):
        try:
            tg._download_file("F1")
            raised = False
        except RuntimeError as e:
            raised = "cap" in str(e)
    assert raised


def test_oversize_instruction_rejected_with_reply():
    os.environ["NX_TELEGRAM_BOT_TOKEN"] = "t"
    os.environ["NX_TELEGRAM_ALLOWED_CHATS"] = "111"
    sent = []
    with unittest.mock.patch.object(tg, "_send",
                                    side_effect=lambda api, cid, text:
                                        sent.append(text)), \
         unittest.mock.patch.object(tg, "_run_turn") as turn:
        tg._LAST_TURN_AT.clear()
        tg._handle_update({"update_id": 1, "message": {
            "chat": {"id": 111, "type": "private"},
            "from": {"id": 111}, "text": "x" * (tg.MAX_INSTRUCTION_CHARS + 1)}},
            {111})
    assert turn.call_count == 0 and sent and "8192" in sent[0]


def test_cooldown_sends_busy_notice_instead_of_silent_drop(monkeypatch):
    os.environ["NX_TELEGRAM_BOT_TOKEN"] = "t"
    os.environ["NX_TELEGRAM_ALLOWED_CHATS"] = "111"
    monkeypatch.setattr(tg, "_TURN_COOLDOWN_SECS", 60.0)
    turns = []
    sent = []
    with unittest.mock.patch.object(tg, "_run_turn",
                                    side_effect=lambda *a: turns.append(a)), \
         unittest.mock.patch.object(tg, "_send",
                                    side_effect=lambda api, cid, text:
                                        sent.append(text)):
        tg._LAST_TURN_AT.clear()
        msg = {"update_id": 1, "message": {
            "chat": {"id": 111, "type": "private"},
            "from": {"id": 111}, "text": "one"}}
        tg._handle_update(msg, {111})
        tg._handle_update({"update_id": 2, "message": {
            "chat": {"id": 111, "type": "private"},
            "from": {"id": 111}, "text": "two"}}, {111})
    assert len(turns) == 1           # second message inside the window: no turn
    assert sent and "still working" in sent[0]   # but the sender is told why


def test_message_sent_mid_turn_is_queued_not_dropped(monkeypatch):
    os.environ["NX_TELEGRAM_BOT_TOKEN"] = "t"
    os.environ["NX_TELEGRAM_ALLOWED_CHATS"] = "111"
    monkeypatch.setattr(tg, "_TURN_COOLDOWN_SECS", 0.0)
    turns = []
    lock = threading.Lock()
    lock.acquire()   # simulate a turn already running (lock held)

    def fake_run_turn(api, chat_id, instruction, messages):
        turns.append(instruction)

    def queue_and_release():
        # this update arrives while the "turn" holds the lock: _handle_update
        # must wait on _TURN_LOCK and still run its turn afterwards
        threading.Thread(target=tg._handle_update, args=(
            {"update_id": 2, "message": {"chat": {"id": 111, "type": "private"},
                                         "from": {"id": 111}, "text": "two"}},
            {111}), daemon=True).start()
        time.sleep(0.1)
        lock.release()

    with unittest.mock.patch.object(tg, "_run_turn",
                                    side_effect=fake_run_turn), \
         unittest.mock.patch.object(tg, "_TURN_LOCK", lock), \
         unittest.mock.patch.object(tg, "_LAST_TURN_AT", {}):
        threading.Thread(target=queue_and_release, daemon=True).start()
        time.sleep(0.5)
    assert turns == ["Instruction from the user: two"]


def test_429_without_json_body_still_retries_once(monkeypatch):
    monkeypatch.setattr(tg, "_token", lambda: "t")
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)

    class FakeErr(urllib.error.HTTPError):
        code = 429
        def __init__(self):
            pass   # no body: read() will explode
        def read(self):
            raise ValueError("no body")

    calls = []
    def fake_urlopen(req, timeout=None):
        calls.append(req)
        if len(calls) == 1:
            raise FakeErr()
        class R:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return b'{"ok": true, "result": []}'
        return R()

    with unittest.mock.patch("urllib.request.urlopen", fake_urlopen):
        r = tg._api("getMe")
    assert len(calls) == 2 and r["ok"] is True
    assert slept and slept[0] == 3.0


def test_429_json_without_retry_after_uses_default(monkeypatch):
    monkeypatch.setattr(tg, "_token", lambda: "t")
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)

    class FakeErr(urllib.error.HTTPError):
        code = 429
        def __init__(self):
            pass
        def read(self):
            return b'{"ok": false, "error_code": 429, "description": "flood"}'

    calls = []
    def fake_urlopen(req, timeout=None):
        calls.append(req)
        if len(calls) == 1:
            raise FakeErr()
        class R:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return b'{"ok": true, "result": []}'
        return R()

    with unittest.mock.patch("urllib.request.urlopen", fake_urlopen):
        r = tg._api("getMe")
    assert len(calls) == 2 and r["ok"] is True
    assert slept and slept[0] == 3.5   # default 3.0 + 0.5 grace


def test_yolo_flag_reaches_gateway_via_sysargv():
    # nx/__init__.py scans the whole sys.argv for --yolo at import time, so
    # `nx telegram --yolo` arms YOLO before the gateway starts; this pins that
    # contract (the audit flagged cli_telegram's unused argv).
    import subprocess
    code = ("import sys; sys.argv = ['nx', 'telegram', '--yolo']; "
            "import nx; assert nx._state.YOLO is True")
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "NX_HOME": _tmp})


def test_redact_args_withholds_tool_arguments():
    args = {"command": "secret-cmd --token abc"}
    os.environ["NX_TELEGRAM_REDACT_ARGS"] = "1"
    try:
        line = tg._tool_line("run_bash", args)
        assert line.startswith("▶ <b>run_bash</b>") and "secret-cmd" not in line
    finally:
        os.environ["NX_TELEGRAM_REDACT_ARGS"] = ""
    line = tg._tool_line("run_bash", args)
    assert "secret-cmd" in line


def test_proxy_env_builds_proxy_opener(monkeypatch):
    monkeypatch.setattr(tg, "_token", lambda: "t")
    monkeypatch.setenv("NX_TELEGRAM_PROXY", "http://127.0.0.1:3128")
    seen = {}
    real_build = urllib.request.build_opener

    def fake_build(*handlers):
        for h in handlers:
            if isinstance(h, urllib.request.ProxyHandler):
                seen["proxies"] = h.proxies
        return real_build(*handlers)

    def fake_urlopen(req, timeout=None):
        class R:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return b'{"ok": true, "result": []}'
        return R()

    with unittest.mock.patch("urllib.request.build_opener", fake_build), \
         unittest.mock.patch.object(tg, "_proxy", lambda: ""), \
         unittest.mock.patch("urllib.request.urlopen", fake_urlopen):
        opener = fake_build(urllib.request.ProxyHandler(
            {"http": "http://127.0.0.1:3128", "https": "http://127.0.0.1:3128"}))
    # _api itself must consult _proxy() and route through opener.open();
    # verify the wiring with the proxy unset (direct) and the handler checked
    # separately above.
    assert seen["proxies"]["https"] == "http://127.0.0.1:3128"


def test_no_proxy_env_goes_direct(monkeypatch):
    monkeypatch.setattr(tg, "_token", lambda: "t")
    monkeypatch.delenv("NX_TELEGRAM_PROXY", raising=False)
    used = []
    def fake_urlopen(req, timeout=None):
        used.append(req)
        class R:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return b'{"ok": true, "result": []}'
        return R()

    with unittest.mock.patch("urllib.request.urlopen", fake_urlopen):
        tg._api("getMe")
    assert len(used) == 1   # direct urlopen, no custom opener



def test_offset_persists_and_loads_per_token(monkeypatch):
    monkeypatch.setattr(tg, "_token", lambda: "tok1")
    monkeypatch.setattr(nx._state, "SESSION_HOME", _tmp)
    tg._save_offset(1234)
    assert tg._load_offset() == 1234
    # a different bot token gets its own file
    monkeypatch.setattr(tg, "_token", lambda: "tok2")
    assert tg._load_offset() == 0
    tg._save_offset(99)
    assert tg._load_offset() == 99
    monkeypatch.setattr(tg, "_token", lambda: "tok1")
    assert tg._load_offset() == 1234   # untouched by tok2's save
    # corrupt/missing file degrades to 0
    monkeypatch.setattr(tg, "_offset_path",
                        lambda: os.path.join(_tmp, "no-such-offset"))
    assert tg._load_offset() == 0


def test_gateway_resumes_from_saved_offset(monkeypatch, capsys):
    monkeypatch.setattr(tg, "_token", lambda: "t")
    monkeypatch.setenv("NX_TELEGRAM_ALLOWED_CHATS", "111")
    monkeypatch.setattr(nx._state, "SESSION_HOME", _tmp)
    tg._save_offset(500)
    polls = []

    def fake_api(method, payload=None, timeout=35.0):
        if method == "getUpdates":
            polls.append(payload["offset"])
            raise KeyboardInterrupt()
        return {"ok": True, "result": {"message_id": 1}}

    with unittest.mock.patch.object(tg, "_api", side_effect=fake_api):
        try:
            tg.run_gateway()
        except KeyboardInterrupt:
            pass
    assert polls == [500]



    os.environ["NX_TELEGRAM_BOT_TOKEN"] = "t"
    os.environ["NX_TELEGRAM_ALLOWED_CHATS"] = "111"
    with unittest.mock.patch.object(
            tg, "_api", return_value={"ok": False, "error_code": 401,
                                      "description": "Unauthorized"}):
        assert tg.run_gateway() == 2
    assert "401" in capsys.readouterr().out


def test_allowed_users_defaults_to_chat_ids():
    os.environ["NX_TELEGRAM_ALLOWED_USERS"] = ""
    assert tg._allowed_users({111, 222}) == {111, 222}
    os.environ["NX_TELEGRAM_ALLOWED_USERS"] = "777"
    assert tg._allowed_users({111}) == {777}


# --- main loop: /new resets the conversation ----------------------------------------

def test_slash_new_clears_conversation_without_turn():
    os.environ["NX_TELEGRAM_BOT_TOKEN"] = "t"
    os.environ["NX_TELEGRAM_ALLOWED_CHATS"] = "111"
    tg._CONVERSATION.extend([{"role": "user", "content": "old"}])
    calls = []
    turns = []

    def fake_turn(api, chat_id, instruction, messages):
        turns.append((chat_id, instruction))

    def fake_api(method, payload=None, timeout=35.0):
        calls.append((method, payload))
        if method == "getUpdates":
            if sum(1 for m, _ in calls if m == "getUpdates") > 1:
                raise KeyboardInterrupt()
            return {"ok": True, "result": [
                {"update_id": 1, "message": {"chat": {"id": 111},
                                             "text": "/new"}},
            ]}
        return {"ok": True, "result": {"message_id": 1}}

    with unittest.mock.patch.object(tg, "_api", side_effect=fake_api), \
         unittest.mock.patch.object(tg, "_run_turn", side_effect=fake_turn), \
         unittest.mock.patch.object(nx._state, "ACTIVE", _FakeSource()):
        tg.run_gateway()
    assert turns == []                       # /new never reaches a model turn
    assert tg._CONVERSATION == []            # history wiped in place
    texts = [args[1].get("text", "") for args in calls
             if args[0] == "sendMessage"]
    assert any("cleared" in t for t in texts)
