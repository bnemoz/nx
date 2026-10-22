#!/usr/bin/env python3
"""Golden SSE corpus: provider wire fidelity for the streaming parser.

DURABILITY_AUDIT P2 ("Provider wire fidelity is untested"): every fake in the
suite is built from the parser's own assumptions — hand-made chunk objects,
attribute-stub clients, and one byte-framing test whose template is a
single-choice delta with no finish_reason, no usage frame and no role delta. A
provider that ships tool-call arguments under a different shape therefore
passes the whole suite while truncating args in production.

This file closes that gap with the independent half of the loop: raw SSE bodies
written from the *providers'* documented wire format (never from the parser),
replayed through the real transport path —
`OpenAI(transport=httpx.MockTransport(...)).chat.completions.create(stream=True)`
→ `nx._openai_client._ChatStream` → `nx._stream_parse.StreamParser`, i.e. the
same two objects `model_turn()` drives. Nothing is stubbed between the bytes
and the parser.

Adding a provider shape = one entry in CORPUS at the bottom. No fakes are
touched and no helper needs editing: the parametrized test builds the stream,
feeds it, and asserts the parsed turn against that entry's expectations.

Invariants asserted for EVERY trace:
  * every tool call's id / name / arguments equal the bytes the trace put on
    the wire (a delta-key assumption that drops a fragment shows up here
    instead of truncating args in production);
  * `dropped_lines` equals the trace's declared count — 0 unless the trace
    deliberately sends something malformed. That counter is what `_turn.py`'s
    dropped-frame warning reports;
  * no collapse detector fired (`loop_cut` is False), so a legitimate provider
    trace is never mistaken for a degenerated stream.

Two traces are CHARACTERIZATION traces: they pin what the parser does TODAY
with a shape it does not handle (legacy `function_call`, and an SSE event split
over two `data:` lines). Both are labelled in place and reported to the audit;
a future fix has to update them deliberately.

Tier A: no threads (speculative reads forced off — they would hand work to a
ThreadPoolExecutor), no network (httpx.MockTransport answers in-process), no
terminal rendering (StreamParser in quiet mode).
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["NX_ENV_FILE"] = "/dev/null"
for _k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SOURCES", "NX_ACTIVE",
           "NX_HTTP_RETRIES", "NX_SPECULATIVE_READS"):
    os.environ.pop(_k, None)

import httpx  # noqa: E402
import pytest  # noqa: E402

import nx  # noqa: E402,F401  (package import initializes the st.* StreamParser reads)
import nx._compress  # noqa: E402
from nx._openai_client import OpenAI  # noqa: E402
from nx._stream_parse import StreamParser  # noqa: E402

from dataclasses import dataclass, field  # noqa: E402
from typing import Any, Callable  # noqa: E402

DONE = "data: [DONE]\n\n"


# --------------------------------------------------------------------------
# trace authoring helpers (the ONE place a new provider shape needs)
# --------------------------------------------------------------------------
def ev(payload: Any, eol: str = "\n\n") -> str:
    """One SSE `data:` event. `payload` is a dict (serialized) or a str
    (spliced verbatim — that is how a deliberately malformed frame is authored).
    `eol` exposes the framing a real server picks (SSE allows \\n\\n, \\r\\n\\r\\n
    and a missing terminator on the last event)."""
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {body}{eol}"


def delta_chunk(choice_deltas: list[dict], finish: str | None = None) -> dict:
    """An OpenAI chunk with one entry per n-choice: `choice_deltas[i]` is
    choice i's delta, `finish` the (same) finish_reason for each.

    NOT a way to send several content deltas in one frame — a real chunk
    carries exactly ONE delta per choice, and `StreamParser` reads
    `choices[0]` only, so a second element here is a choice the parser drops
    (see check_choice1_not_parsed). Several prose deltas = several events."""
    out: list[dict] = []
    for i, d in enumerate(choice_deltas):
        c: dict[str, Any] = {"index": i, "delta": d}
        if finish is not None:
            c["finish_reason"] = finish
        out.append(c)
    return {"choices": out}


def usage_chunk(usage: dict, **extra: Any) -> dict:
    """vLLM / hosted-API `stream_options.include_usage` shape: a final chunk
    whose choices list is EMPTY and which carries only usage."""
    return {"choices": [], "usage": usage, **extra}


def role_only(role: str = "assistant") -> dict:
    """The role delta every OpenAI-compatible server sends first, with no
    token payload. A parser that treats "no content" as "nothing arrived" still
    has to let it through without counting it as damage."""
    return delta_chunk([{"role": role}])


def tc_delta(index: int, id_: str | None, name: str | None, args: str) -> dict:
    """One tool-call delta fragment. id/name arrive with the first fragment
    only; arguments arrive pre-tokenized — which is exactly what makes a
    dropped fragment invisible at the transport layer."""
    call: dict[str, Any] = {"index": index, "type": "function"}
    if id_ is not None:
        call["id"] = id_
    call["function"] = {k: v for k, v in (("name", name), ("arguments", args))
                        if v is not None}
    return delta_chunk([{"tool_calls": [call]}])


# --------------------------------------------------------------------------
# expectations
# --------------------------------------------------------------------------
@dataclass
class Call:
    """What one tool call must look like after the whole trace was parsed."""
    index: int
    id: str
    name: str
    args: str


@dataclass
class Turn:
    """Everything the real parser recorded from one replay."""
    tcs: dict
    content: str
    reasoning: str
    finish_reasons: list
    usage: Any
    timings: Any
    dropped_lines: int
    loop_cut: bool

    def args(self, index: int) -> str:
        return self.tcs[index]["args"]


@dataclass
class Trace:
    """One provider trace + the turn it must parse into.

    `check` is for the shapes a dataclass field cannot express (malformed JSON,
    a silently-unsupported key); it receives the parsed turn."""
    provider: str
    shape: str
    body: str
    calls: list = field(default_factory=list)
    content: str | None = None
    finish_reasons: list = field(default_factory=list)
    has_usage: bool = False
    has_timings: bool = False
    dropped_lines: int = 0
    check: Callable[[Turn], None] | None = None


# --------------------------------------------------------------------------
# checks for shapes a field cannot express
# --------------------------------------------------------------------------
def check_choice1_not_parsed(t: Turn) -> None:
    """n>1 multi-choice streaming: `_ChatStream` yields the whole chunk (both
    choices) but `StreamParser.process` reads `chunk.choices[0]` only, so
    choice 1's tool call is silently absent from the turn — with
    `dropped_lines == 0`, so nothing reports it. Pinned as a known limitation:
    it is a capability gap, not arg truncation (choice 0's args stay exact)."""
    assert "call_99" not in {c["id"] for c in t.tcs.values()}, t.tcs
    assert len(t.tcs) == 1, t.tcs


def check_reasoning(t: Turn) -> None:
    assert t.reasoning == ("The user wants a listing. "
                           "run_bash is the right tool."), repr(t.reasoning)


def check_incomplete_json(t: Turn) -> None:
    """The args are byte-exact (the Call expectation proved that) AND still
    unparseable: so the truncation the turn layer reports came from the model,
    not from the framing path — and nothing was counted as dropped, so the
    malformed-args breaker fires on a true diagnosis."""
    with pytest.raises(json.JSONDecodeError):
        json.loads(t.args(0))
    assert t.dropped_lines == 0


def check_legacy_function_call_dropped(t: Turn) -> None:
    assert t.tcs == {}, ("legacy function_call support appeared — update this "
                         "characterization trace to assert the parsed call")
    assert t.content == ""
    # the silent part: nothing on the wire was malformed, so nothing was counted
    assert t.dropped_lines == 0


def check_multiline_data_not_reassembled(t: Turn) -> None:
    assert t.tcs == {}, t.tcs
    assert t.dropped_lines == 2, ("a split data: line stopped being counted — "
                                  "_turn.py's dropped-frame warning would go "
                                  "blind on exactly this provider shape")


# --------------------------------------------------------------------------
# THE CORPUS — one entry per provider shape, each body hand-written from the
# provider's own documented wire format.
# --------------------------------------------------------------------------

# vLLM, native tool choice, n=2 (multi-choice) + include_usage.
VLLM_TOOL_MULTI_CHOICE = Trace(
    provider="vLLM",
    shape="native tool_calls, n=2 multi-choice, split argument deltas, "
          "finish_reason=tool_calls, separate usage-only final frame",
    body=(
        ev({"id": "cmpl-a", "object": "chat.completion.chunk",
            "model": "DeepSeek-V4",
            "choices": [
                {"index": 0, "logprobs": None,
                 "delta": {"role": "assistant", "content": None,
                           "tool_calls": None}},
                {"index": 1, "logprobs": None,
                 "delta": {"role": "assistant", "content": None,
                           "tool_calls": None}}]})
        # id + name + opening brace land in the first fragment...
        + ev({"choices": [
            {"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_00", "type": "function",
                 "function": {"name": "read_file",
                              "arguments": "{\"path\""}}]}},
            {"index": 1, "delta": {"tool_calls": [
                {"index": 0, "id": "call_99", "type": "function",
                 "function": {"name": "run_bash",
                              "arguments": "{\"command\""}}]}}]})
        + ev({"choices": [
            {"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ": \"src/app"}}]}},
            {"index": 1, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ": \"ls\""}}]}}]})
        + ev({"choices": [
            {"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function":
                            {"arguments": ".py\", \"limit\": 40}"}}]}},
            {"index": 1, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "\""}}]}}]})
        + ev({"choices": [
            {"index": 0, "delta": {}, "finish_reason": "tool_calls"},
            {"index": 1, "delta": {}, "finish_reason": "tool_calls"}]})
        + ev(usage_chunk({"prompt_tokens": 37, "completion_tokens": 22,
                          "total_tokens": 59}))
        # some vLLM builds end the stream with an UNTERMINATED final event (no
        # blank line after [DONE]); SSE allows it and the framing layer must
        # still see the counter at 0
        + ev("[DONE]", eol="")
    ),
    calls=[Call(0, "call_00", "read_file",
                '{"path": "src/app.py", "limit": 40}')],
    finish_reasons=["tool_calls"],
    has_usage=True,
    check=check_choice1_not_parsed,
)

# vLLM plain-text answer: role delta, prose, finish_reason=stop, and a
# usage-only final frame with an empty choices list.
VLLM_TEXT_USAGE_AT_END = Trace(
    provider="vLLM",
    shape="text answer, role-only first delta, finish_reason=stop, "
          "usage-at-end frame whose choices list is empty",
    body=(
        ev(role_only())
        + ev(delta_chunk([{"content": "Two separators "}]))
        + ev(delta_chunk([{"content": "matter: \n and \r\n."}]))
        + ev(delta_chunk([{}], finish="stop"))
        + ev({"id": "cmpl-b", "choices": [],
              "usage": {"prompt_tokens": 11, "completion_tokens": 9,
                        "total_tokens": 20}})
        + DONE
    ),
    content="Two separators matter: \n and \r\n.",
    finish_reasons=["stop"],
    has_usage=True,
)

# llama.cpp's OpenAI-compatible server: role delta, content deltas, and its own
# top-level `timings` extension riding on the last chunk (NOT `usage`).
LLAMA_CPP_ROLE_ONLY_TIMINGS = Trace(
    provider="llama.cpp / OSS llama-server",
    shape="role-only delta, prose deltas, top-level `timings` extension, "
          "finish_reason=stop, no usage frame",
    body=(
        ev({"choices": [{"index": 0, "delta": {"role": "assistant"},
                         "finish_reason": None}]})
        + ev(delta_chunk([{"content": "ping "}]))
        + ev(delta_chunk([{"content": "pong"}]))
        + ev({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
              "timings": {"predicted_per_second": 41.2, "predicted_n": 4,
                          "prompt_ms": 88.0}})
        + DONE
    ),
    content="ping pong",
    finish_reasons=["stop"],
    has_timings=True,
)

# Qwen behind an OSS OpenAI-compatible gateway: `reasoning_content` (a
# non-standard delta key), prose, a native tool call, then Qwen's literal "\n"
# CONTENT separator, finish_reason and usage.
OSS_QWEN_REASONING_THEN_TOOLS = Trace(
    provider="OSS gateway (Qwen chat template)",
    shape="reasoning_content deltas, native tool call with Qwen \"\\n\" "
          "content separator, finish_reason=tool_calls, usage-at-end",
    body=(
        ev(role_only())
        + ev(delta_chunk([{"reasoning_content": "The user wants a listing. "}]))
        + ev(delta_chunk([{"reasoning_content": "run_bash is the right tool."}]))
        + ev(delta_chunk([{"content": "Listing the tree now."}]))
        + ev(tc_delta(0, "call_7", "run_bash", "{\"command\": \"ls -R\"}"))
        + ev(delta_chunk([{"content": "\n"}]))    # chat-template separator
        + ev(delta_chunk([{}], finish="tool_calls"))
        + ev(usage_chunk({"prompt_tokens": 512, "completion_tokens": 33,
                          "total_tokens": 545}))
        + DONE
    ),
    calls=[Call(0, "call_7", "run_bash", '{"command": "ls -R"}')],
    content="Listing the tree now.",
    finish_reasons=["tool_calls"],
    has_usage=True,
    check=check_reasoning,
)

# Hosted API (OpenAI-shaped gateway): parallel tool calls INTERLEAVED by index
# across chunks — the shape where a parser keyed on "last seen id" instead of
# `index` cross-contaminates two argument buffers.
HOSTED_PARALLEL_TOOL_CALLS = Trace(
    provider="hosted API (OpenAI-compatible gateway)",
    shape="role delta with refusal, two parallel tool calls interleaved by "
          "index across chunks, finish_reason=tool_calls, usage frame with "
          "prompt_tokens_details",
    body=(
        ev({"choices": [{"index": 0,
                         "delta": {"role": "assistant", "content": None,
                                   "refusal": None}}],
            "system_fingerprint": "fp_abc123"})
        + ev({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_A", "type": "function",
             "function": {"name": "search_files",
                          "arguments": "{\"pattern\":"}},
            {"index": 1, "id": "call_B", "type": "function",
             "function": {"name": "list_dir",
                          "arguments": "{\"path\":"}}]}}]})
        # from here the two argument buffers alternate
        + ev(tc_delta(1, None, None, " \"src\"}"))
        + ev(tc_delta(0, None, None, " \"TODO\""))
        + ev(tc_delta(0, None, None, ", \"glob\": \"*.py\"}"))
        + ev({"choices": [{"index": 0, "delta": {},
                           "finish_reason": "tool_calls", "logprobs": None}]})
        + ev(usage_chunk({"prompt_tokens": 800, "completion_tokens": 41,
                          "total_tokens": 841,
                          "prompt_tokens_details": {"cached_tokens": 512}}))
        + DONE
    ),
    calls=[Call(0, "call_A", "search_files",
                '{"pattern": "TODO", "glob": "*.py"}'),
           Call(1, "call_B", "list_dir", '{"path": "src"}')],
    finish_reasons=["tool_calls"],
    has_usage=True,
)

# Malformed tool-call arguments ON THE WIRE: every frame is valid SSE and valid
# chunk JSON, but the accumulated `arguments` string is not valid JSON. This is
# the audit's failure scenario minus any transport bug.
MALFORMED_JSON_TOOL_ARGS = Trace(
    provider="vLLM (destabilized decode)",
    shape="valid frames, valid chunk JSON, arguments string that is NOT valid "
          "json (dangling key + unbalanced brace)",
    body=(
        ev(role_only())
        + ev(tc_delta(0, "call_x", "read_file", "{\"path\": \"n.md\","))
        + ev(tc_delta(0, None, None, " \"offset\": 7,"))
        + ev(delta_chunk([{}], finish="stop"))
        + DONE
    ),
    calls=[Call(0, "call_x", "read_file", '{"path": "n.md", "offset": 7,')],
    finish_reasons=["stop"],
    check=check_incomplete_json,
)

# CHARACTERIZATION — legacy `function_call`. Pre-`tool_calls` OpenAI, still
# emitted by older llama.cpp builds and by proxies that translate
# Anthropic/Gemini tool_use for old clients. `StreamParser` has no branch for
# `delta.function_call`, so this turn yields NO tool calls with
# `dropped_lines == 0`: a completely silent capability loss, which is exactly
# the audit's "is nx wrong or is the gateway wrong?" scenario. Pinned so the
# blind spot is on the record.
LEGACY_FUNCTION_CALL = Trace(
    provider="legacy hosted API (pre-tool_calls `function_call`)",
    shape="delta.function_call {name, arguments} — a shape the parser does not "
          "support today (characterization: args are lost SILENTLY)",
    body=(
        ev({"choices": [{"index": 0,
                         "delta": {"role": "assistant", "content": None}}]})
        + ev(delta_chunk([{"function_call": {"name": "get_weather"}}]))
        + ev(delta_chunk([{"function_call": {"arguments": "{\"city\":"}}]))
        + ev(delta_chunk([{"function_call": {"arguments": " \"Oslo\"}"}}]))
        + ev(delta_chunk([{}], finish="function_call"))
        + DONE
    ),
    finish_reasons=["function_call"],
    check=check_legacy_function_call_dropped,
)

# CHARACTERIZATION — one SSE event split over two `data:` lines, which the
# WHATWG event-stream spec explicitly allows (the client joins them with "\n"
# before parsing). `_ChatStream` json.loads each `data:` line on its own, so
# both halves fail. Unlike the trace above this failure IS visible: the counter
# goes up, which is the property `_turn.py`'s warning depends on.
# One complete, valid chunk JSON document — then sent as two `data:` lines by
# cutting it at MLINE_SPLIT, so each half alone is unparseable. (Editing either
# string below keeps the trace honest as long as both halves stay broken.)
_MLINE_CHUNK = ('{"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"id":"call_m","function":{"name":"run_bash",'
                '"arguments":"{\\"command\\": \\"true\\"}"}}]}}]}')
_MLINE_SPLIT = 52  # mid-"arguments": both halves fail json.loads

assert json.loads(_MLINE_CHUNK)  # the WHOLE document is valid — only the
# framing splits it (the trace's premise; fails loudly if either string drifts)

SSE_MULTILINE_DATA_EVENT = Trace(
    provider="spec-conformant proxy (multi-line data event)",
    shape="one SSE event spread over two `data:` lines — characterization: not "
          "reassembled, but both halves are COUNTED as dropped",
    body=("data: " + _MLINE_CHUNK[:_MLINE_SPLIT] + "\n"
          "data: " + _MLINE_CHUNK[_MLINE_SPLIT:] + "\n\n" + DONE),
    dropped_lines=2,
    check=check_multiline_data_not_reassembled,
)


CORPUS: dict[str, Trace] = {
    "vllm_tool_multi_choice": VLLM_TOOL_MULTI_CHOICE,
    "vllm_text_usage_at_end": VLLM_TEXT_USAGE_AT_END,
    "llama_cpp_role_only_timings": LLAMA_CPP_ROLE_ONLY_TIMINGS,
    "oss_qwen_reasoning_then_tools": OSS_QWEN_REASONING_THEN_TOOLS,
    "hosted_parallel_tool_calls": HOSTED_PARALLEL_TOOL_CALLS,
    "malformed_json_tool_args": MALFORMED_JSON_TOOL_ARGS,
    "legacy_function_call": LEGACY_FUNCTION_CALL,
    "sse_multiline_data_event": SSE_MULTILINE_DATA_EVENT,
}


# --------------------------------------------------------------------------
# replay harness
# --------------------------------------------------------------------------
class _Console:
    """Console stand-in: quiet mode never renders, but StreamParser holds one.
    io.StringIO keeps any stray write out of the test log."""

    def __init__(self) -> None:
        self.file = io.StringIO()

    @property
    def is_terminal(self) -> bool:
        return False

    @property
    def width(self) -> int:
        return 80


@pytest.fixture(autouse=True)
def _no_speculative_reads(monkeypatch):
    """`maybe_speculate` dispatches on a ThreadPoolExecutor, and three traces
    carry balanced args for cacheable read tools. Tier A forbids threads, and a
    warm tool cache is irrelevant to wire fidelity. The flag is read lazily off
    `nx._compress` (see `nx/_spec_reads._enabled`), so patching the module
    attribute is the supported switch."""
    monkeypatch.setattr(nx._compress, "NX_SPECULATIVE_READS", False,
                        raising=False)


def replay(body: str) -> Turn:
    """The production path end to end: MockTransport → the OpenAI-compatible
    client's SSE iterator → `StreamParser.process()`, driven exactly like
    `_turn.py`'s chunk loop (quiet=True: parse and account, write nothing)."""
    def handler(request):
        assert request.url.path.endswith("/chat/completions"), request.url
        return httpx.Response(200, content=body.encode("utf-8"),
                              headers={"content-type": "text/event-stream"})

    client = OpenAI(base_url="http://test/v1", api_key="sk-test",
                    transport=httpx.MockTransport(handler))
    stream = client.chat.completions.create(model="m", messages=[],
                                           stream=True)
    parser = StreamParser(_Console(), 0.0, quiet=True)
    try:
        for chunk in stream:
            if parser.process(chunk):
                break
    finally:
        parser.finish()
        stream.close()
    return Turn(
        tcs={i: dict(c) for i, c in parser.tcs.items()},
        content=parser._content_str,
        reasoning="".join(parser.reasoning_parts),
        finish_reasons=list(parser.finish_reasons),
        usage=parser.usage,
        timings=parser.timings,
        dropped_lines=stream.dropped_lines,
        loop_cut=parser.loop_cut,
    )


# --------------------------------------------------------------------------
# the tests
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(CORPUS))
def test_provider_trace_parses_complete(name: str):
    trace = CORPUS[name]
    turn = replay(trace.body)

    assert turn.loop_cut is False, (
        f"{name}: a collapse detector fired on a legitimate provider trace — "
        "the whole turn would be discarded and re-run")

    # every tool call the trace put on the wire, byte for byte
    assert sorted(turn.tcs) == sorted(c.index for c in trace.calls), (
        f"{name}: tool-call set mismatch, got {turn.tcs}")
    for want in trace.calls:
        got = turn.tcs[want.index]
        assert got["id"] == want.id, f"{name}: call id, got {got!r}"
        assert got["name"] == want.name, f"{name}: call name, got {got!r}"
        assert got["args"] == want.args, (
            f"{name}: tool args truncated/reordered on the wire path\n"
            f"  wire:   {want.args!r}\n  parsed: {got['args']!r}")
        try:
            json.loads(want.args)
        except json.JSONDecodeError:
            continue  # the trace deliberately put broken JSON on the wire
        json.loads(got["args"])  # and the turn layer must be able to parse it

    if trace.content is not None:
        assert turn.content == trace.content, (
            f"{name}: content drift: {turn.content!r}")

    assert turn.finish_reasons == trace.finish_reasons, (
        f"{name}: finish_reason frames {turn.finish_reasons} != "
        f"{trace.finish_reasons}")
    assert (turn.usage is not None) == trace.has_usage, (
        f"{name}: usage-at-end frame not accounted (usage={turn.usage!r})")
    assert (turn.timings is not None) == trace.has_timings, (
        f"{name}: llama.cpp `timings` extension not accounted")

    # the counter _turn.py's dropped-frame warning reports
    assert turn.dropped_lines == trace.dropped_lines, (
        f"{name}: dropped_lines {turn.dropped_lines} != "
        f"{trace.dropped_lines} — "
        + ("a provider frame was silently discarded"
           if trace.dropped_lines == 0 else
           "a malformed frame the trace declares stopped being counted"))

    if trace.check is not None:
        trace.check(turn)


def test_corpus_covers_the_required_shapes():
    """Guard the acceptance criteria: the audit named specific shapes and
    provider tiers. A trace deleted or renamed fails here rather than quietly
    shrinking coverage."""
    bodies = {n: t.body.lower() for n, t in CORPUS.items()}
    shapes = {n: t.shape.lower() for n, t in CORPUS.items()}
    providers = " ".join(t.provider.lower() for t in CORPUS.values())
    for tier in ("vllm", "llama.cpp", "hosted"):
        assert tier in providers, f"{tier} no longer represented in the corpus"
    for shape, marker in (("multi-choice", "multi-choice"),
                          ("finish_reason frames", "finish_reason"),
                          ("usage-at-end", "usage"),
                          ("role-only deltas", "role"),
                          ("malformed-JSON tool args", "not valid json")):
        assert any(marker in shapes[n] or marker in bodies[n] for n in CORPUS), (
            f"corpus lost coverage of {shape}")
    assert any("reasoning_content" in bodies[n] for n in CORPUS), (
        "no trace carries a non-standard reasoning delta key anymore")


def test_replay_stays_tier_a_no_speculative_pool(monkeypatch):
    """Tier A claim, asserted rather than assumed: `StreamParser
    ._handle_tool_call_delta` calls `maybe_speculate` on every args delta, and
    a balanced-JSON read call would hand work to a ThreadPoolExecutor (three of
    these traces carry balanced args for cacheable read tools). With the flag
    off the pool must never be created — otherwise this file is a threaded test
    pretending not to be one."""
    from nx import _spec_reads
    monkeypatch.setattr(_spec_reads, "_pool_holder", {}, raising=False)
    for name in ("hosted_parallel_tool_calls", "vllm_tool_multi_choice",
                 "oss_qwen_reasoning_then_tools"):
        replay(CORPUS[name].body)
    assert _spec_reads._pool_holder == {}, (
        "the speculative-read pool was created from a golden-trace replay")


def test_traces_are_wire_bytes_not_fake_chunks():
    """The corpus is only verification-INDEPENDENT if each trace is plain wire
    text. Assert the invariant so nobody later swaps a body for a fake-chunk
    object — the tautology this file exists to break."""
    for name, trace in CORPUS.items():
        assert isinstance(trace.body, str), f"{name}: body must be raw SSE text"
        assert trace.body.startswith("data: "), f"{name}: not an SSE stream"
        assert "[DONE]" in trace.body, f"{name}: no terminal SSE event"
        # and it has to be legal SSE framing on its own terms: every non-blank
        # line is a field line, and every `data:` line's payload is either
        # [DONE] or JSON — except the one trace whose whole subject is a
        # framing defect.
        defective = {"sse_multiline_data_event"}
        for line in trace.body.splitlines():
            if not line.strip():
                continue
            assert line.startswith(("data:", ":", "event:", "id:")), \
                f"{name}: not an SSE field line: {line!r}"
            if name in defective or line == "data: [DONE]":
                continue
            json.loads(line[len("data:"):].strip())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
