#!/usr/bin/env python3
"""_ChatStream's SSE framing: split on the protocol's newlines, not Python's.

A JSON string may legally carry raw U+0085 / U+2028 / U+2029 (plus VT, FF, FS,
GS, RS), and httpx's Response.iter_lines() splits with str.splitlines()
semantics, whose NEWLINE_CHARS include all of them. One tool-args delta
carrying such a character therefore shredded one SSE frame into two lines: the
first half failed json.loads (counted as a dropped frame) and the second failed
the data-prefix test, so the delta vanished while the stream itself finished
cleanly. That is the exact shape of the "malformed tool call ...
finish_reason=stop" turns: truncated arguments with no transport error to
explain them. The stream now splits raw bytes on newline (trimming a trailing
carriage return), which is what SSE actually specifies.

Also covers the proxy that lets _turn.py's dropped-frame warning see the
counter through _LoggingStream, which used to swallow it.

The separator characters are built with chr() and never written literally (not
even in this docstring): they are line breaks to a source reader, so a file
that names them inline gets shredded exactly the way a frame was.
"""
import json
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-sseframing-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_HOME"] = _tmp
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SOURCES", "NX_ACTIVE",
          "NX_HTTP_RETRIES"):
    os.environ.pop(k, None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from nx._openai_client import OpenAI  # noqa: E402
from nx._turn import _LoggingStream  # noqa: E402

SEP = [chr(0x0b), chr(0x0c), chr(0x1c), chr(0x1d), chr(0x1e), chr(0x85),
       chr(0x2028), chr(0x2029)]
# The one from the reported session: plausible inside file content an edit
# tool echoes back as its own argument value.
U2028 = chr(0x2028)
NL = "\n"
DONE = ("data: [DONE]" + NL + NL).encode("utf-8")
# One streaming chunk whose delta carries a single tool-call's arguments. The
# arguments value is spliced in raw (no escaping) by frame() below.
TPL = ('data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
       '"function":{"arguments":@ARGS@}}]}}]}')


def frame(args_raw):
    """One SSE frame whose single tool-call delta carries `args_raw` verbatim as
    its arguments value. Hand-built because json.dumps escapes a separator
    character away, and the raw byte is the whole point."""
    return (TPL.replace("@ARGS@", args_raw) + NL + NL).encode("utf-8")


def stream_of(body):
    def handler(request):
        return httpx.Response(200, content=body,
                              headers={"content-type": "text/event-stream"})
    client = OpenAI(base_url="http://test/v1", api_key="sk-test",
                    transport=httpx.MockTransport(handler))
    return client.chat.completions.create(model="m", messages=[], stream=True)


def args_of(stream):
    out = []
    for chunk in stream:
        for tc in (chunk.choices[0].delta.tool_calls or []):
            out.append(tc.function.arguments or "")
    return "".join(out)


def test_separator_char_does_not_split_a_frame():
    """Three deltas assemble one arguments object; the middle frame carries a
    raw U+2028 inside its own JSON string (spliced as a quoted literal, so the
    frame is valid JSON with the character unescaped)."""
    mid = frame('"b' + U2028 + 'r"')
    body = (frame(json.dumps('{"path": "f", "new": "')) + mid
            + frame(json.dumps('x"}')) + DONE)
    st = stream_of(body)
    assert json.loads(args_of(st)) == {"path": "f", "new": "b" + U2028 + "rx"}
    assert st.dropped_lines == 0


def test_every_splitlines_only_separator_survives():
    """For each character httpx breaks on that SSE does not: the delta arrives
    whole and nothing counts as a dropped frame."""
    for ch in SEP:
        # the arguments value has to be a JSON string, so splice the encoded
        # scalar rather than a bare character
        raw = json.dumps(ch, ensure_ascii=False)
        body = frame(raw) + DONE
        st = stream_of(body)
        assert args_of(st) == ch, repr(ch)
        assert st.dropped_lines == 0, repr(ch)
        if ord(ch) >= 0x20:
            # JSON mandates escaping everything below U+0020, so only the three
            # above it can reach the wire unescaped. Assert the frame really did
            # carry the raw byte, otherwise this test proves nothing.
            assert ch.encode("utf-8") in body, hex(ord(ch))


def test_crlf_lone_cr_comment_and_malformed_frame():
    """Every terminator SSE really does allow still frames, the comment and
    blank-line rules hold, and a genuinely malformed frame is still counted."""
    def line(args_raw, eol):
        # a valid frame whose arguments string is exactly args_raw
        return (TPL.replace("@ARGS@", json.dumps(args_raw)) + eol).encode("utf-8")

    body = line("a", "\r\n\r\n")            # CRLF pair
    body += (": an SSE comment" + NL + NL).encode("utf-8")
    body += line("b", "\r\r\n")             # lone CR terminated frame
    body += ("data: {not json at all}" + NL + NL).encode("utf-8")
    body += line("c", "")                   # unterminated tail frame
    st = stream_of(body)
    assert args_of(st) == "abc"
    assert st.dropped_lines == 1


def test_done_stops_iteration():
    body = frame('"x"') + DONE + frame('"NEVER"')
    assert args_of(stream_of(body)) == "x"


def test_dropped_lines_visible_through_logging_stream(tmp_path):
    """_turn.py reads the counter off whatever open_stream returns, which is the
    _LoggingStream wrapper, so it has to be proxied -- otherwise the
    dropped-frame warning reports 0 forever and the truncation stays invisible,
    which is precisely how this went undiagnosed."""
    body = ("data: {\"arguments\": data: nope}\n\n").encode("utf-8") + DONE
    inner = stream_of(body)
    with open(tmp_path / "log.jsonl", "w") as log:
        wrapped = _LoggingStream(inner, log)
        list(wrapped)  # consume, so the inner counter has settled
        assert wrapped.dropped_lines == inner.dropped_lines == 1
