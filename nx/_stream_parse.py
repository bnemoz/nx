"""Per-chunk streaming parser for model_turn()'s SSE loop.

Extracted out of _turn.py because model_turn() had accreted the chunk-by-chunk
state machine (reasoning vs. content mode, leaked-special-token stripping,
repeat-chunk collapse detection, TTFT/timings capture, tool-call-status
spinner) directly inline, on top of everything else it does (stream open,
tool dispatch, retry bookkeeping). That's exactly the shape of tangled mutable
state the changelog's "terminal breaks on resize" entry warns about — pulling
the chunk parser into its own object with named attributes (instead of 15+
interacting locals + nonlocals) makes each piece independently readable and
testable without touching the rest of the turn loop.
"""
from __future__ import annotations

import sys
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from ._ansi import DIM, CLEAR_LINE, C, RESET
from ._special_tokens import (
    _LEAKED_SPECIAL_TOKEN_RE, LEAKED_TOKEN_RUN_LIMIT, REPEAT_CHUNK_LIMIT,
    _strip_leaked_tokens,
)
from ._metrics import _abbr, _precise_abbr
from . import _state as st
from . import REASONING_ONLY_TIME_LIMIT


def _fmt_think_duration(elapsed: float) -> str:
    """ms below 1s, seconds below 1min, else 'Xmin Ys'."""
    if elapsed < 1:
        return f"{elapsed * 1000:.0f}ms"
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    mins, secs = divmod(int(round(elapsed)), 60)
    return f"{mins}min {secs}s"


# Text-protocol tool-call tag openers (see _text_fallback.py's TOOL_TAG /
# NX_TOOL_TAG). Native tool-calling servers (Qwen chat templates etc.) never
# put these in `content` — the call arrives as structured `delta.tool_calls`
# instead, hidden behind the "generating tool call" status line below. A
# server with no native tool-calling convention (Gemma et al.) has nowhere
# else to put the tagged call, so it comes back as plain `content` and would
# otherwise stream the raw `<tool_call>{"name": ...}` JSON straight into the
# chat like ordinary prose — this detects the tag opening mid-stream so it
# gets hidden the same way native tool_calls already are.
_TEXT_TOOL_TAG_OPENERS = ("<tool_call>", "[nx_tool_call]")


def _first_tool_tag_index(text: str) -> int | None:
    idx = None
    for opener in _TEXT_TOOL_TAG_OPENERS:
        i = text.find(opener)
        if i != -1 and (idx is None or i < idx):
            idx = i
    return idx


def _tool_call_progress(tcs: dict[int, dict[str, Any]]) -> str:
    parts = []
    for i in sorted(tcs):
        c = tcs[i]
        name = c["name"] or "tool"
        args_n = len(c["args"])
        parts.append(f"{name} args {_precise_abbr(args_n)} chars" if args_n else f"{name} ...")
    return ", ".join(parts) if parts else "tool call ..."


class StreamParser:
    """Consumes one SSE chunk's delta at a time and accumulates the turn's
    streamed state: reasoning vs. content mode, tool-call arg buffers, TTFT,
    llama.cpp `timings`/OpenAI `usage`, and the two collapse detectors
    (leaked special tokens, repeated identical chunks).

    `process(chunk)` returns True when the stream should be cut short (a
    collapse was detected) — the caller is responsible for closing/breaking
    out of the actual SSE iterator, since only it holds that handle.
    """

    def __init__(self, say_console: Console, t0: float) -> None:
        self._say_console = say_console
        self._t0 = t0
        self.content: list[str] = []
        self.tcs: dict[int, dict[str, Any]] = {}
        self.mode: str | None = None  # None | "think" | "say"
        self.say_live: Live | None = None
        self.reasoning_parts: list[str] = []
        self.timings = None
        self.usage = None
        self.streamed_chars = 0
        self.t_first: float | None = None
        self.loop_cut = False
        self.reasoning_only_chars = 0
        self.reasoning_only_start: float | None = None
        self.finish_reasons: list[str] = []
        self.leaked_token_run = 0
        self.repeat_chunk_run = 0
        self.last_chunk: str | None = None
        self.tool_status_active = False
        self.last_tool_status = 0.0
        self.in_text_tool_tag = False  # a <tool_call>/[nx_tool_call] tag opened in content
        self.think_start: float | None = None

    # --- tool-call status line (throttled) ---------------------------------
    def show_tool_status(self, force: bool = False) -> None:
        now = time.time()
        if not force and self.tool_status_active and now - self.last_tool_status < 0.2:
            return
        sys.stdout.write(CLEAR_LINE + f"{C.ACCENT}  ↳ generating tool call{RESET} "
                         f"{DIM}{_tool_call_progress(self.tcs)}{RESET}")
        sys.stdout.flush()
        self.tool_status_active = True
        self.last_tool_status = now

    def clear_tool_status(self) -> None:
        if self.tool_status_active:
            sys.stdout.write(CLEAR_LINE)
            sys.stdout.flush()
            self.tool_status_active = False

    def close_think_block(self) -> None:
        """Print the reasoning-block closer + elapsed duration. Call once
        when leaving "think" mode, however that happens (content, tool
        call, stream end, or error) — no-op if not currently in think mode."""
        if self.mode != "think" or self.think_start is None:
            return
        elapsed = time.time() - self.think_start
        dur = _fmt_think_duration(elapsed)
        if st.REASONING_VISIBLE:
            print()
            print(f"{DIM}  ────────────── (thought for {dur}){RESET}")
        else:
            print(f"{DIM}  (thought for {dur}){RESET}")
        self.think_start = None

    # --- per-chunk entry point ----------------------------------------------
    def process(self, chunk: Any) -> bool:
        """Consume one SSE chunk. Returns True if the stream should be cut
        (self.loop_cut is also set in that case)."""
        if chunk.usage:
            self.usage = chunk.usage
        if not chunk.choices:
            return False
        choice = chunk.choices[0]
        if choice.finish_reason:
            self.finish_reasons.append(str(choice.finish_reason))
        d = choice.delta
        d_extra = getattr(d, "model_extra", None) or {}
        rc = (getattr(d, "reasoning_content", None) or
              d_extra.get("reasoning_content") or d_extra.get("reasoning"))
        if self.t_first is None and (d.content or d.tool_calls or rc):
            self.t_first = time.time() - self._t0
        extra = getattr(chunk, "model_extra", None) or {}
        if "timings" in extra:
            self.timings = extra["timings"]
        if rc and self._handle_reasoning(rc):
            return True
        if d.content and self._handle_content(d.content):
            return True
        for tc in (d.tool_calls or []):
            self._handle_tool_call_delta(tc)
        return False

    def _handle_reasoning(self, rc: str) -> bool:
        """Returns True if the stream should be cut (collapse detected)."""
        if self.mode != "think":
            self.think_start = time.time()
            if st.REASONING_VISIBLE:
                print(f"{C.REASONING_TEXT}  ── reasoning ──{RESET}")
            else:
                print(f"{DIM}  ▸ reasoning (collapsed — /reasoning to expand){RESET}")
            self.mode = "think"
        if _LEAKED_SPECIAL_TOKEN_RE.search(rc):
            rc = _strip_leaked_tokens(rc)
            self.leaked_token_run += 1
        else:
            self.leaked_token_run = 0
        if rc.strip():
            if st.REASONING_VISIBLE:
                print(f"{C.REASONING_TEXT}{rc}{RESET}", end="", flush=True)
            self.reasoning_parts.append(rc)
        if not self.content and not self.tcs:
            self.reasoning_only_chars += len(rc)
            if self.reasoning_only_start is None:
                self.reasoning_only_start = time.time()
        if LEAKED_TOKEN_RUN_LIMIT > 0 and self.leaked_token_run >= LEAKED_TOKEN_RUN_LIMIT:
            print()
            print(f"{C.ERROR}  ⚠ LEAKED-TOKEN COLLAPSE — {self.leaked_token_run} consecutive "
                  f"special-token chunks; cutting stream now{RESET}")
            self.loop_cut = True
            return True
        if (st.REASONING_ONLY_CHAR_LIMIT > 0 and not self.content and not self.tcs
                and self.reasoning_only_chars >= st.REASONING_ONLY_CHAR_LIMIT):
            print()
            print(f"{C.ERROR}  ⚠ REASONING-ONLY LIMIT HIT — "
                  f"{_abbr(self.reasoning_only_chars)} reasoning chars with no "
                  f"answer/tool call — cutting stream now{RESET}")
            self.loop_cut = True
            return True
        if (REASONING_ONLY_TIME_LIMIT > 0 and not self.content and not self.tcs
                and self.reasoning_only_start is not None
                and time.time() - self.reasoning_only_start >= REASONING_ONLY_TIME_LIMIT):
            elapsed_reason = time.time() - self.reasoning_only_start
            print()
            print(f"{C.ERROR}  ⚠ REASONING-ONLY TIME LIMIT — "
                  f"{elapsed_reason:.0f}s of reasoning with no "
                  f"answer/tool call — cutting stream now{RESET}")
            self.loop_cut = True
            return True
        return False

    def _handle_content(self, raw_chunk: str) -> bool:
        """Returns True if the stream should be cut (collapse detected)."""
        self.clear_tool_status()
        self.close_think_block()
        self.mode = "say"
        chunk = raw_chunk
        if _LEAKED_SPECIAL_TOKEN_RE.search(chunk):
            chunk = _strip_leaked_tokens(chunk)
            self.leaked_token_run += 1
        else:
            self.leaked_token_run = 0
        if REPEAT_CHUNK_LIMIT > 0 and chunk and len(chunk) <= 40:
            if chunk == self.last_chunk:
                self.repeat_chunk_run += 1
            else:
                self.repeat_chunk_run = 1
                self.last_chunk = chunk
        else:
            self.repeat_chunk_run = 0
        if chunk:
            self.content.append(chunk)
            self.streamed_chars += len(chunk)
            if not self.in_text_tool_tag:
                full = "".join(self.content)
                tag_idx = _first_tool_tag_index(full)
                if tag_idx is None:
                    if self.say_live is None:
                        self.say_live = Live(console=self._say_console,
                                             refresh_per_second=12, transient=False)
                        self.say_live.start()
                    self.say_live.update(Markdown(full))
                else:
                    # Tag opened mid-stream: commit whatever prose came
                    # before it, close the live panel, and fall through to
                    # the throttled tool-status line below — same treatment
                    # native tool_calls deltas already get, instead of
                    # streaming raw <tool_call>{"name": ...} JSON as if it
                    # were ordinary prose.
                    #
                    # The opener itself can arrive split across SSE chunks
                    # (e.g. "<tool_call" in one delta, ">..." in the next) —
                    # until the closing ">" lands, _first_tool_tag_index()
                    # can't see the full opener yet, so that dangling partial
                    # fragment gets live-rendered as if it were prose one
                    # chunk earlier. Live.stop() with transient=False freezes
                    # whatever was last rendered, so without re-rendering to
                    # the (tag-free) prefix right before stopping, that stray
                    # partial-opener frame would be the one left on screen.
                    prefix = full[:tag_idx]
                    if self.say_live is None and prefix.strip():
                        self.say_live = Live(console=self._say_console,
                                             refresh_per_second=12, transient=False)
                        self.say_live.start()
                    if self.say_live is not None:
                        self.say_live.update(Markdown(prefix))
                        self.say_live.stop()
                        self.say_live = None
                    self.in_text_tool_tag = True
            if self.in_text_tool_tag:
                self.show_tool_status()
        if ((LEAKED_TOKEN_RUN_LIMIT > 0 and self.leaked_token_run >= LEAKED_TOKEN_RUN_LIMIT)
                or (REPEAT_CHUNK_LIMIT > 0 and self.repeat_chunk_run >= REPEAT_CHUNK_LIMIT)):
            kind = ("special-token collapse" if self.leaked_token_run >= LEAKED_TOKEN_RUN_LIMIT
                    else "repetition collapse")
            if self.say_live is not None:
                self.say_live.stop()
                self.say_live = None
            print(f"{C.ERROR}  ⚠ {kind.upper()} — cutting stream now; "
                  f"discarding degenerated output{RESET}")
            self.loop_cut = True
            return True
        return False

    def _handle_tool_call_delta(self, tc: Any) -> None:
        if self.mode == "think":
            self.close_think_block()
            self.mode = None
        elif self.mode == "say":
            if self.say_live is not None:
                self.say_live.stop()
                self.say_live = None
            self.mode = None
        s = self.tcs.setdefault(tc.index, {"id": "", "name": "", "args": ""})
        if tc.id:
            s["id"] = tc.id
        if tc.function and tc.function.name:
            s["name"] = tc.function.name
        if tc.function and tc.function.arguments:
            s["args"] += tc.function.arguments
            self.streamed_chars += len(tc.function.arguments)
        self.show_tool_status()

    def finish(self) -> None:
        """Close out any open reasoning/say block. Call once the loop ends
        for any reason (normal end, interrupt, error) before reading back
        mode/say_live elsewhere."""
        self.close_think_block()
        print(RESET)
