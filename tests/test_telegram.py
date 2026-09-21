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
    assert methods.count("sendMessage") >= 2       # status + final answer
    assert "editMessageText" in methods            # on_step progress happened
    final = calls[-1]
    assert final[0] == "sendMessage"
    assert final[1]["text"] == "The paper says X."
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
            if len(calls) > 1:   # second poll: stop the loop
                raise KeyboardInterrupt()
            return {"ok": True, "result": updates}
        return {"ok": True, "result": {"message_id": 1}}

    with unittest.mock.patch.object(tg, "_api", side_effect=fake_api), \
         unittest.mock.patch.object(tg, "_run_turn", side_effect=fake_turn), \
         unittest.mock.patch.object(nx._state, "ACTIVE", _FakeSource()):
        tg.run_gateway()
    # the disallowed chat never produced a turn, never got a reply
    assert turns == [(111, "Instruction from the user: hello")]
