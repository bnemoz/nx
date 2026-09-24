"""nx telegram (0.79.0): the Telegram gateway — long-poll one bot, one turn
per allowed-chat message.

The suite is offline by construction: every Bot-API call goes through a
recorded fake `_api`, so no test contacts api.telegram.org. The contract
under test is the security posture (hard allowlist, empty-refuses-to-start,
auto-deny), the message-to-instruction translation (PDF attachments inline
via extract_pdf, URLs pass through for the model to fetch), message
chunking, and the on_step progress-edit seam.
"""
import json
import os
import sys
import tempfile
import unittest.mock

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx  # noqa: E402
import nx._telegram as tg  # noqa: E402


import pytest

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
    assert lines == ["📂 list_dir: . ×2"] and n == 2


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
    out = tg._compose_status("fix the bug", lines, 7, None, 65.0)
    assert "earlier calls" in out
    assert "line19" in out and "line5" not in out
    assert "7 tool calls" in out and "65s" in out
    short = tg._compose_status("q", ["only"], 1, None, None)
    assert short == "❓ q\nonly\n\ntotal: 1 tool calls"


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


# --- main loop: allowlist enforcement -----------------------------------------------

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
    import threading
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
    assert lines == ["📂 list_dir: . ×3"] and n == 3


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
        import pytest
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
