"""Per-chunk streaming parser for model_turn()'s SSE loop.

Extracted out of _turn.py because model_turn() had accreted the chunk-by-chunk
state machine (reasoning vs. content mode, leaked-special-token stripping,
repeat-chunk collapse detection, TTFT/timings capture, tool-call-status
spinner) directly inline, on top of everything else it does (stream open,
tool dispatch, retry bookkeeping). Pulling the chunk parser into its own
object with named attributes (instead of 15+ interacting locals + nonlocals)
makes each piece independently readable and testable without touching the
rest of the turn loop.
"""
from __future__ import annotations

import sys
import time
from typing import Any

from rich.console import Console
from rich.live import Live

from ._rich import themed_markdown
# EVENTSURFACE_SPEC §1(a): the stream tap lives on the parser's own thread, so
# it resolves the job id from that thread's sub-agent context (see tap_event).
from ._subagent_tls import tap_event

from ._ansi import DIM, CLEAR_LINE, C, RESET, strip_ansi_for_display
from ._special_tokens import (
    _LEAKED_SPECIAL_TOKEN_RE, LEAKED_TOKEN_RUN_LIMIT, REPEAT_CHUNK_LIMIT,
    _strip_leaked_tokens,
)
from ._metrics import _abbr, _precise_abbr
# EVENTSURFACE_SPEC §2 Phase 4: every raw console write below takes the shared
# write lock so a 4 fps /tabs repaint cannot tear a streamed line. _surfaces is
# already in sys.modules by the time this module loads (nx/__init__ imports it
# first) and imports no turn-layer module, so a module-level import adds zero
# modules to the T-9 conftest snapshot.
from ._surfaces import write_lock
from . import _state as st
from . import REASONING_ONLY_TIME_LIMIT

# Below this many seconds of first→last-token span, a client-side tok/s figure
# is pacing-free guessing (the whole generation can arrive in one or two SSE
# reads) — the footer marks it approximate instead of asserting five decimals
# of precision it doesn't have. Not an NX_* tunable: nothing here tunes it
# usefully, it's just the resolution boundary of what the client can observe.
_TPS_TRUST_WINDOW = 0.5

# Below this, the span isn't pacing at all — it's SSE arrival quantization.
# A bursty server (vLLM on a short tool-call turn) can deliver an entire
# generation in one or two reads a fraction of a millisecond apart, and
# dividing completion_tokens by that "span" printed ~51008.7 tok/s. Below the
# threshold the client never observed a pace, so gen_window() falls back to
# the full elapsed time: the end-to-end average, a true lower bound on the
# decode rate, still marked `~`. Above it, real multi-read pacing — kept.
_TPS_MIN_SPAN = 0.02

# PF-04: minimum Live re-render interval. 50 ms for the first ~2 KB of an
# answer, then scaled with its length — no human reads faster than ~20 Hz and a
# 20 KB answer does not need 20 fps of full Markdown re-parses.
_SAY_MIN_INTERVAL = 0.05
_SAY_THROTTLE_SCALE_BYTES = 2048


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

# Longest opener; a tag can only START within len(opener)-1 bytes before the
# previous chunk boundary, so the per-chunk scan resumes that far back and the
# whole-answer scan stays O(chunk) instead of O(buffer) (PERF.md PF-02).
_MAX_OPENER_LEN = max(len(o) for o in _TEXT_TOOL_TAG_OPENERS)


def _first_tool_tag_index(text: str, start: int = 0) -> int | None:
    idx = None
    for opener in _TEXT_TOOL_TAG_OPENERS:
        i = text.find(opener, start)
        if i != -1 and (idx is None or i < idx):
            idx = i
    return idx


# Speculative read-only dispatch (0.103.3, nx/_spec_reads): a text-protocol tag
# body is only worth re-parsing once a CLOSING tag has landed, so that check
# gates the scan. A close tag can straddle chunk boundaries, hence the small
# carried tail rather than a per-chunk find.
_SPEC_CLOSE_TAIL = 64
# A real call body is a name plus a few arguments; past this many bytes since
# the opener the "tag" is prose or a runaway repeat, and re-parsing per close
# would stop being cheap.
_SPEC_TEXT_CAP = 8192


def _close_tag_in(text: str) -> bool:
    from ._text_fallback import TOOL_CLOSE_TAGS
    return any(tag.search(text) for tag in TOOL_CLOSE_TAGS)


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

    Token-arrival timestamps (t_first/t_last/t_usage, all relative to the
    same t0 as TTFT) back gen_window(): a tok/s figure computed from the
    first→last token span instead of wall-clock, so the head (connection +
    prefill) and the tail (final renders, usage computation, teardown)
    don't masquerade as slow decoding.
    """

    def __init__(self, say_console: Console, t0: float, quiet: bool = False) -> None:
        self._say_console = say_console
        self._t0 = t0
        # quiet=True (async sub-agent loop): parse and account everything, but
        # write NOTHING to the shared terminal — no reasoning text, no Live
        # say-panel, no tool-status line. The agent's answer reaches the user
        # only as the staged [Runtime note].
        self.quiet = quiet
        self.content: list[str] = []
        # PF-03/PF-04: content is kept as a list of raw chunks plus an
        # incrementally-stripped mirror (`_clean`). No whole-buffer string is
        # copied or re-joined per chunk; `_content_str`/`_clean_str` below are
        # derived read-only properties, not accumulators.
        self._clean: list[str] = []
        self._content_len = 0
        self._clean_len = 0
        self.tcs: dict[int, dict[str, Any]] = {}
        self.mode: str | None = None  # None | "think" | "say"
        self.say_live: Live | None = None
        self.reasoning_parts: list[str] = []
        self.timings = None
        self.usage = None
        self.streamed_chars = 0
        self.t_first: float | None = None  # first token-bearing delta, rel. to _t0
        self.t_last: float | None = None   # last token-bearing delta, rel. to _t0
        self.t_usage: float | None = None  # the usage-bearing chunk, rel. to _t0
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
        self._tag_carry = ""        # raw tail carried across chunks for the tag scan (PF-02/03)
        self._tag_carry_start = 0   # absolute offset of _tag_carry in raw content
        # Speculative read-only dispatch (0.103.3): raw body text accumulated
        # SINCE a text-protocol tag opened, so a completed tag can be executed
        # while the stream is still running (nx/_spec_reads). Bounded: past
        # _SPEC_TEXT_CAP the tag is too big to be a real call body and the
        # per-chunk re-parse would stop being cheap.
        self._spec_text = ""
        self._spec_text_full = False
        self.had_tool_call = False  # a native tool_calls delta already arrived this turn
        self._live_committed = False  # say_live already frozen once; never reopen it
        self.think_start: float | None = None
        self.last_say_render = 0.0
        self._last_render_len = -1  # clean length at the last Live render (PF-04)

    # --- derived content views (PF-03/04) ----------------------------------
    @property
    def _content_str(self) -> str:
        """The raw accumulated content, joined on demand. Kept as a property
        (not an accumulator) so no whole-buffer copy happens per chunk; only
        tests and the rare tag-prefix path read it."""
        return "".join(self.content)

    @property
    def _clean_str(self) -> str:
        """The ANSI-stripped accumulated content — what every render reads."""
        return "".join(self._clean)

    # --- tool-call status line (throttled) ---------------------------------
    def show_tool_status(self, force: bool = False) -> None:
        if self.quiet:
            return
        now = time.time()
        if not force and self.tool_status_active and now - self.last_tool_status < 0.2:
            return
        line = (CLEAR_LINE + f"{C.ACCENT}  ↳ generating tool call{RESET} "
                f"{DIM}{_tool_call_progress(self.tcs)}{RESET}")
        with write_lock():
            sys.stdout.write(line)
            sys.stdout.flush()
        self.tool_status_active = True
        self.last_tool_status = now

    def _spec_text_poll(self, chunk: str) -> None:
        """Speculatively dispatch a text-protocol call whose tag just CLOSED.

        Called per chunk while a &lt;tool_call&gt;/&#91;nx_tool_call&#93; tag is open. Cheap by
        construction: the body buffer only gets re-parsed once a closing tag
        landed in the carried tail, and it is capped, so a long tag costs
        nothing per chunk. Dispatch and every safety rule live in
        nx/_spec_reads.maybe_speculate; the post-stream text path
        (_handle_text_tool_calls) re-parses the whole answer itself and is
        unaffected — it simply finds warm cache entries."""
        if self._spec_text_full:
            return
        self._spec_text += chunk
        if len(self._spec_text) > _SPEC_TEXT_CAP:
            # Not a call body we can reason about cheaply — give up on this
            # turn's text tags rather than re-parse a growing buffer.
            self._spec_text_full = True
            self._spec_text = ""
            return
        if not _close_tag_in(self._spec_text[-_SPEC_CLOSE_TAIL:]):
            return
        from ._spec_reads import maybe_speculate
        from ._text_fallback import _find_close_tag, parse_text_calls
        for name, args in parse_text_calls(self._spec_text):
            maybe_speculate(name, args)
        # Consume through the LAST close tag: anything before it has already
        # been dispatched, and re-parsing it on the next chunk would queue the
        # same speculative run again.
        pos = 0
        while True:
            m = _find_close_tag(self._spec_text, pos)
            if m is None:
                break
            pos = m.end()
        self._spec_text = self._spec_text[pos:]

    def stop_say_live(self) -> None:
        """Force a final render of any pending content, then stop/clear the
        live panel. Must be used instead of a bare `say_live.stop()` whenever
        content may have arrived since the last throttled render."""
        if self.say_live is not None:
            # Live frames must not interleave with a /tabs dashboard repaint:
            # lock the whole update+stop frame (reentrant RLock — safe when a
            # caller already holds write_lock).
            with write_lock():
                # PF-04: Live(transient=False) already froze the last frame; only
                # re-parse the whole answer if content arrived since that render.
                if self._clean_len != self._last_render_len:
                    self.say_live.update(themed_markdown(self._clean_str))
                    self._last_render_len = self._clean_len
                self.say_live.stop()
                self.say_live = None

    def clear_tool_status(self) -> None:
        # EVENTSURFACE_SPEC latent hazard: this is the one CLEAR_LINE write in
        # the parser with NO quiet check, so a quiet (async-agent) loop would
        # erase a line it does not own. Dead-safe only because
        # tool_status_active cannot become true while quiet — that coupling is
        # an accident of the call sites, not a guarantee, so guard explicitly.
        if self.quiet:
            return
        if self.tool_status_active:
            with write_lock():
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
        # Each closer is one visual unit: locked as a block so a repaint can't
        # land between the two lines of the visible variant.
        if self.quiet:
            pass  # parse only — the closer line is terminal noise here
        elif st.REASONING_VISIBLE:
            with write_lock():
                print()
                print(f"{DIM}  ────────────── (thought for {dur}){RESET}")
        else:
            with write_lock():
                print(f"{DIM}  (thought for {dur}){RESET}")
        self.think_start = None

    def gen_window(self, elapsed_total: float) -> tuple[float, bool]:
        """Best client-side estimate of the generation-only window, in seconds,
        as (window, trusted) for a tok/s figure.

        The window runs first→last token-bearing delta — it excludes the TTFT
        head (connection, queueing, prompt prefill) and the tail (final Rich
        render, server usage computation, HTTP teardown) that `elapsed_total`
        is stuck with, because `elapsed` is taken after all of those. On a
        turn whose prompt just grew by a freshly-read file, the head alone can
        dwarf the decode, and on a short tool-call turn the tail can dwarf it
        from the other side — either one turns tok/s into noise. When the
        usage chunk (some servers send it as a separate final event) lands
        after the last delta, it extends the window so deferred usage isn't
        silently dropped either.

        trusted=False means the window is below the display floor
        (_TPS_TRUST_WINDOW above): a whole short generation can arrive in
        one or two SSE reads on a bursty client or a batching/speculating
        server, and no client-side math can recover pacing it never saw.
        Callers should still render a number but mark it approximate.

        Sub-millisecond spans are a stronger kind of noise than "untrusted":
        they're arrival quantization, not pacing — a bursty server (vLLM on
        a short tool-call turn) delivers the whole generation in one or two
        SSE reads a fraction of a millisecond apart, and dividing 22 tok by
        a 0.0004s span printed ~51008 tok/s. Below _TPS_MIN_SPAN the window
        gives way to `elapsed_total`: the end-to-end average over the whole
        turn, a true lower bound on the decode rate, still `~`-marked.
        """
        if self.t_first is None or self.t_last is None:
            # No token ever arrived (empty turn) — nothing was paced.
            return max(elapsed_total, 0.0), False
        window = self.t_last - self.t_first
        if self.t_usage is not None and self.t_usage - self.t_first > window:
            window = self.t_usage - self.t_first
        if window <= 0:
            # Entire generation in a single event: zero pacing information.
            return 0.0, False
        if window < _TPS_MIN_SPAN:
            # Arrival quantization: no observed pace, so no paced denominator.
            # The wall-time average is the only honest (lower-bound) number.
            return max(elapsed_total, window), False
        return window, window >= _TPS_TRUST_WINDOW

    # --- per-chunk entry point ----------------------------------------------
    def process(self, chunk: Any) -> bool:
        """Consume one SSE chunk. Returns True if the stream should be cut
        (self.loop_cut is also set in that case)."""
        now = time.time() - self._t0  # one clock read per chunk
        if chunk.usage:
            self.usage = chunk.usage
            self.t_usage = now
        if not chunk.choices:
            return False
        choice = chunk.choices[0]
        if choice.finish_reason:
            self.finish_reasons.append(str(choice.finish_reason))
        d = choice.delta
        if d is None:
            return False
        d_extra = getattr(d, "model_extra", None) or {}
        rc = (getattr(d, "reasoning_content", None) or
              d_extra.get("reasoning_content") or d_extra.get("reasoning"))
        if d.content or d.tool_calls or rc:
            # Token-arrival bookkeeping for gen_window(): first sets TTFT,
            # every one paces the decode window.
            if self.t_first is None:
                self.t_first = now
            self.t_last = now
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
            if self.quiet:
                pass
            elif st.REASONING_VISIBLE:
                with write_lock():
                    print(f"{C.REASONING_TEXT}  ── reasoning ──{RESET}")
            else:
                with write_lock():
                    print(f"{DIM}  ▸ reasoning (collapsed — /reasoning to expand){RESET}")
            self.mode = "think"
        if _LEAKED_SPECIAL_TOKEN_RE.search(rc):
            rc = _strip_leaked_tokens(rc)
            self.leaked_token_run += 1
        else:
            self.leaked_token_run = 0
        if rc.strip():
            if self.quiet:
                pass
            elif st.REASONING_VISIBLE:
                # Per-chunk write: the lock wraps only the print, never the
                # chunk loop — an uncontended RLock acquire is ~100 ns against
                # chunks arriving every few ms, and holding across chunks
                # would starve the repaint.
                with write_lock():
                    print(f"{C.REASONING_TEXT}{rc}{RESET}", end="", flush=True)
            self.reasoning_parts.append(rc)
        # Accumulate while there is no prose, even once tool-call deltas have
        # arrived (glm-style interleaved think blocks land after the first
        # tool-call delta). Real prose disqualifies: the footer's divider is
        # for steps with nothing visible above it.
        if not self.content:
            self.reasoning_only_chars += len(rc)
            if self.reasoning_only_start is None:
                self.reasoning_only_start = time.time()
        if LEAKED_TOKEN_RUN_LIMIT > 0 and self.leaked_token_run >= LEAKED_TOKEN_RUN_LIMIT:
            if not self.quiet:
                with write_lock():
                    print()
                    print(f"{C.ERROR}  ⚠ LEAKED-TOKEN COLLAPSE — {self.leaked_token_run} consecutive "
                          f"special-token chunks; cutting stream now{RESET}")
            self.loop_cut = True
            return True
        if (st.REASONING_ONLY_CHAR_LIMIT > 0 and not self.content and not self.tcs
                and self.reasoning_only_chars >= st.REASONING_ONLY_CHAR_LIMIT):
            if not self.quiet:
                with write_lock():
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
            if not self.quiet:
                with write_lock():
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
        if self._live_committed:
            # say_live was already frozen once (text-tool-tag path): never
            # reopen — a second Live would re-render the whole accumulated
            # text again on screen.
            return False
        self.mode = "say"
        chunk = raw_chunk
        if _LEAKED_SPECIAL_TOKEN_RE.search(chunk):
            chunk = _strip_leaked_tokens(chunk)
            self.leaked_token_run += 1
        else:
            self.leaked_token_run = 0
        if self.had_tool_call and not chunk.strip():
            # A whitespace-only content delta AFTER native tool-calling
            # started: Qwen-style chat templates emit a literal "\n" between
            # each tool call (and often trailing ones after the last). It is a
            # separator, not prose — treating it as prose re-opens a Live
            # panel re-rendering the whole accumulated preamble (printed once
            # per separator on screen: 2 calls → doubled, 3 → tripled), and
            # its byte-identical <=40-char shape falsely trips the
            # repetition-collapse detector once a turn hits ~6 calls. Drop it.
            return False
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
            self._content_len += len(chunk)
            self.streamed_chars += len(chunk)
            # EVENTSURFACE_SPEC §1(a) stream tap, chokepoint 1 of 2: this is the
            # one place raw prose is appended unconditionally, so it is where a
            # quiet agent's delta becomes visible on /tabs. The job id comes
            # from THIS thread's sub-agent context (tap_event); on the parent
            # thread there is none and this is one empty-registry check.
            # Display-only: the sink never enters `messages`.
            tap_event("stream", chunk)
            # Real prose landed: it disqualifies the step from reasoning-only
            # accounting (footer divider, stall cutter).
            self.reasoning_only_chars = 0
            # PF-04: strip ANSI once per chunk, not once per rendered frame —
            # the clean mirror is what every render reads.
            clean = strip_ansi_for_display(chunk)
            if clean:
                self._clean.append(clean)
                self._clean_len += len(clean)
            if not self.in_text_tool_tag:
                # PF-02/03: scan only the carried tail plus this chunk — an
                # opener starts <= _MAX_OPENER_LEN-1 bytes before the previous
                # boundary, so the whole-answer scan stays O(chunk) and the raw
                # buffer is never re-joined per chunk.
                window = self._tag_carry + chunk
                rel = _first_tool_tag_index(window)
                if rel is None:
                    carry_n = _MAX_OPENER_LEN - 1
                    self._tag_carry = window[-carry_n:]
                    self._tag_carry_start = self._content_len - len(self._tag_carry)
                    if not self.quiet and self.say_live is None:
                        # One lock per complete frame: creation + start together.
                        with write_lock():
                            self.say_live = Live(console=self._say_console,
                                                 refresh_per_second=12, transient=False)
                            self.say_live.start()
                    now = time.time()
                    # PF-04: raise the throttle with size — 50 ms for the first
                    # ~2 KB, then scaled, so a long answer is not re-parsed at 20 fps.
                    interval = max(
                        _SAY_MIN_INTERVAL,
                        _SAY_MIN_INTERVAL * self._clean_len / _SAY_THROTTLE_SCALE_BYTES)
                    if not self.quiet and self.say_live is not None \
                            and now - self.last_say_render >= interval:
                        with write_lock():
                            self.say_live.update(themed_markdown(self._clean_str))
                        self.last_say_render = now
                        self._last_render_len = self._clean_len
                        # EVENTSURFACE_SPEC §1(a) names this render site as the
                        # second chokepoint, and the stream tap above already
                        # covers every delta that reaches it. Deliberately NOT
                        # tapping here: this branch re-renders the WHOLE
                        # accumulated answer (self._clean_str), not a delta, so
                        # a tap would append the entire answer to the ring once
                        # per frame — the ring would show the answer N times
                        # instead of its tail. The delta above is the delta; a
                        # re-render is not. (Nothing is lost: a quiet agent —
                        # the only kind with a sink — never reaches this branch,
                        # since say_live stays None while quiet.)
                else:
                    tag_idx = self._tag_carry_start + rel
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
                    prefix = strip_ansi_for_display("".join(self.content)[:tag_idx])
                    # One lock per complete frame: creation, final render and
                    # freeze land as a single locked unit.
                    with write_lock():
                        if not self.quiet and self.say_live is None and prefix.strip():
                            self.say_live = Live(console=self._say_console,
                                                 refresh_per_second=12, transient=False)
                            self.say_live.start()
                        if self.say_live is not None:
                            self.say_live.update(themed_markdown(prefix))
                            self.say_live.stop()
                            self.say_live = None
                    self._live_committed = True
                    self.in_text_tool_tag = True
            if self.in_text_tool_tag:
                self._spec_text_poll(chunk)
                self.show_tool_status()
        if ((LEAKED_TOKEN_RUN_LIMIT > 0 and self.leaked_token_run >= LEAKED_TOKEN_RUN_LIMIT)
                or (REPEAT_CHUNK_LIMIT > 0 and self.repeat_chunk_run >= REPEAT_CHUNK_LIMIT)):
            kind = ("special-token collapse" if self.leaked_token_run >= LEAKED_TOKEN_RUN_LIMIT
                    else "repetition collapse")
            if self.say_live is not None:
                with write_lock():
                    self.say_live.stop()
                    self.say_live = None
            if not self.quiet:
                with write_lock():
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
            self.stop_say_live()
            self.mode = None
        self.had_tool_call = True
        s = self.tcs.setdefault(tc.index, {"id": "", "name": "", "args": ""})
        if tc.id:
            s["id"] = tc.id
        if tc.function and tc.function.name:
            s["name"] = tc.function.name
        if tc.function and tc.function.arguments:
            s["args"] += tc.function.arguments
            self.streamed_chars += len(tc.function.arguments)
            # Speculative read-only dispatch (0.103.3, nx/_spec_reads): the
            # args of a cacheable read that just became BALANCED JSON are
            # already a complete call, so start it now on a background pool
            # and let the post-stream dispatch find a warm cache entry.
            # maybe_speculate() re-checks name/balance/in-flight (and is a
            # no-op when disabled), so the per-delta cost outside that set is
            # one membership test — big write_file args never even reach the
            # json.loads.
            from ._spec_reads import maybe_speculate
            maybe_speculate(s["name"], s["args"])
        self.show_tool_status()

    def finish(self) -> None:
        """Close out any open reasoning/say block. Call once the loop ends
        for any reason (normal end, interrupt, error) before reading back
        mode/say_live elsewhere."""
        self.close_think_block()
        if not self.quiet:
            with write_lock():
                print(RESET)
