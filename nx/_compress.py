"""Streaming completion requests, recovery sampling, and context compression.

open_stream() is the one place every model call funnels through (normal
turns and the loop-breaking "recovery" retries alike). compress() (manual
/compress) and _maybe_autocompress() (automatic, triggered by
NX_AUTOCOMPRESS_PERCENT) fold older turns of `messages` into a single
summary turn, keeping the system prompt and the most recent turns verbatim
— freeing context without losing the model's grip on what it was just
doing. Unlike manual /compress (which keeps only COMPRESS_KEEP=2 turns),
auto-compress is conservative: it keeps roughly the last third of the
conversation so in-progress work isn't disturbed.
"""
from __future__ import annotations

import os
import re
from typing import Any

import httpx
from ._openai_client import APIConnectionError

from ._ansi import DIM, RESET, C
from ._env import _env_int, _env_float
from ._logging import _log_event, _llog
from ._sessions_io import _prune_empty_assistant_messages, _drop_leading_dangling_tool_turns
from ._metrics import _abbr, _record_usage
from ._grammar import build_tool_call_grammar
from . import _state as st
from . import TOOLS

RUNTIME_NOTE_RE = re.compile(r"\n\n\[Runtime note: .*?\]\s*$", re.DOTALL)


def _nudge_current_user_turn(messages: list[dict[str, Any]], nudge: str) -> None:
    note = f"[Runtime note: {nudge}]"
    for msg in reversed(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), str):
            continue
        content = RUNTIME_NOTE_RE.sub("", msg["content"]).rstrip()
        msg["content"] = f"{content}\n\n{note}" if content else note
        return
    messages.append({"role": "user", "content": note})


def _flush_pending_background_notice(messages: list[dict[str, Any]]) -> None:
    """Append BACKGROUND_GUIDANCE once, if run_tool() just saw the first
    backgrounded run_bash result this session. Appends a brand-new tail
    message (never rewrites an earlier one) so it can't invalidate a
    server's prompt-prefix cache."""
    if st._PENDING_BACKGROUND_NOTICE:
        messages.append({"role": "user", "content": st._PENDING_BACKGROUND_NOTICE})
        st._PENDING_BACKGROUND_NOTICE = None
    if st._PENDING_MULTI_EDIT_NOTICE:
        messages.append({"role": "user", "content": st._PENDING_MULTI_EDIT_NOTICE})
        st._PENDING_MULTI_EDIT_NOTICE = None


def _last_is_dangling_tool(messages: list[dict[str, Any]]) -> bool:
    """True if the last message is a `tool` result with no assistant turn after it.

    This is the layout right after a tool ran and before the model follows up: an
    empty assistant turn here is most suspicious — the model produced nothing in
    response to a fresh tool result, so the user sees the agent "finish" with no
    answer. Auto-continue is most warranted in exactly this case.
    """
    return bool(messages) and messages[-1].get("role") == "tool"


MAX_COMPLETION_TOKENS = _env_int("NX_MAX_TOKENS", 16000)

# Backend hint. Set NX_BACKEND=vllm to disable llama.cpp-only recovery
# knobs (min_p, repeat_penalty, DRY) that vLLM's speculative decoder rejects.
RECOVERY_BACKEND = os.environ.get("NX_BACKEND", "").strip().lower()

# Recovery sampler. A gibberish/repetition collapse is a LOW-ENTROPY attractor:
# the model is copying long n-grams already in context (paths, flags). Escaping
# it needs MORE entropy (raise temperature, lower min_p) plus anti-repetition
# penalties that lower the looping token's logit — the OPPOSITE of sharpening
# toward greedy. So recovery now RAISES temperature (was 0.2, which made the
# loop more deterministic) and adds DRY + repeat_penalty. The non-OpenAI knobs
# ride in extra_body (llama.cpp reads them; other backends ignore unknown keys).
st.RECOVERY_TEMPERATURE = _env_float("NX_RECOVERY_TEMPERATURE", 1.0)
st.RECOVERY_TOP_P = _env_float("NX_RECOVERY_TOP_P", 0.95)
RECOVERY_MIN_P = None if RECOVERY_BACKEND == "vllm" else _env_float("NX_RECOVERY_MIN_P", 0.02)
RECOVERY_REPEAT_PENALTY = None if RECOVERY_BACKEND == "vllm" else _env_float("NX_RECOVERY_REPEAT_PENALTY", 1.2)
RECOVERY_REPEAT_LAST_N = None if RECOVERY_BACKEND == "vllm" else _env_int("NX_RECOVERY_REPEAT_LAST_N", 512)
RECOVERY_DRY_MULTIPLIER = None if RECOVERY_BACKEND == "vllm" else _env_float("NX_RECOVERY_DRY_MULTIPLIER", 0.8)
RECOVERY_DRY_BASE = None if RECOVERY_BACKEND == "vllm" else _env_float("NX_RECOVERY_DRY_BASE", 1.75)
RECOVERY_DRY_ALLOWED_LENGTH = None if RECOVERY_BACKEND == "vllm" else _env_int("NX_RECOVERY_DRY_ALLOWED_LENGTH", 2)
# Treat path/code punctuation as DRY sequence breakers so a long file path the
# agent must emit verbatim is never penalized as "repetition".
RECOVERY_DRY_SEQUENCE_BREAKERS = None if RECOVERY_BACKEND == "vllm" else ["\n", ":", "\"", "*", "/", "\\", "`", "'"]

# Grammar-constrained tool calls: opt-in (NX_GRAMMAR=1) since the request
# fields involved (llama.cpp's `grammar` + `grammar_triggers`) are a
# llama.cpp server extension whose exact shape has shifted across versions,
# unlike the long-stable min_p/DRY knobs above. It's skipped for vLLM (which
# has its own, differently-shaped guided-decoding fields) — an unsupported
# server just gets the field ignored or the request rejected, in which case
# open_stream()'s existing no-tools/no-extra fallback ladder drops it.
#
# Meaningful only for the text-protocol fallback, NOT once native tool-calling
# is confirmed (st._NATIVE_TOOL_CALLS_CONFIRMED, set by
# _handle_native_tool_calls() in _turn.py the first time a turn actually comes
# back with structured tool_calls). llama.cpp already builds its own
# argument-constraining grammar internally from `tools=` for chat templates it
# recognizes as native-tool-call-capable, with no separate client-observable
# trigger to hook a second grammar off of — sending our own `grammar` field
# on top of that would either be redundant or override/fight llama.cpp's own
# native grammar (some server versions reject `tools` + explicit `grammar`
# together outright). So once a session is confirmed native, we stop
# attaching ours and just let llama.cpp's built-in constraint do the job;
# right up until that first confirmation (and for servers/templates that
# never go native, i.e. permanently text-protocol), our `<tool_call>`-tag
# grammar is the only thing constraining malformed JSON, so it stays on.
# NX_GRAMMAR is now only a fallback gate: once the startup capability probe
# (nx/_capability.py, run at each switch_source()) lands on a source, its
# verdict (CAPABILITY_NATIVE/GRAMMAR/TEXT) decides whether the grammar gets
# attached, regardless of this env var — see _grammar_extra_body() below.
NX_GRAMMAR = RECOVERY_BACKEND != "vllm" and os.environ.get("NX_GRAMMAR", "") not in ("", "0")
_grammar_cache = {}  # tool-name tuple -> extra_body dict, built lazily/once per tool set


def _grammar_extra_body(tools: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not tools or st._NATIVE_TOOL_CALLS_CONFIRMED:
        return None
    # The startup capability probe (nx/_capability.py) takes precedence over
    # the NX_GRAMMAR opt-in once it lands: CAPABILITY_NATIVE means llama.cpp
    # is already constraining tool_calls itself (skip, same as the reactive
    # _NATIVE_TOOL_CALLS_CONFIRMED check above); CAPABILITY_GRAMMAR means the
    # grammar demonstrably helps on this source, so attach it even without
    # NX_GRAMMAR set; CAPABILITY_TEXT means it didn't help in the probe, so
    # skip it (plain text-protocol parsing only). While the probe is still
    # pending (None), fall back to the old NX_GRAMMAR env-var gate.
    capability = getattr(st.ACTIVE, "_capability", None) if st.ACTIVE else None
    if capability is not None:
        from ._capability import CAPABILITY_GRAMMAR
        if capability != CAPABILITY_GRAMMAR:
            return None
    elif not NX_GRAMMAR:
        return None
    key = tuple(t.get("function", {}).get("name") for t in tools)
    if key not in _grammar_cache:
        grammar, trigger = build_tool_call_grammar(tools)
        _grammar_cache[key] = None if grammar is None else {
            "grammar": grammar,
            "grammar_triggers": [{"type": "word", "value": trigger}],
        }
    return _grammar_cache[key]


def _recovery_sampling_opts() -> dict[str, Any]:
    """Loop-breaking sampler for recovery retries (see RECOVERY_* constants).

    temperature/top_p are standard OpenAI params and go top-level. min_p,
    repeat_penalty, repeat_last_n and the DRY family are llama.cpp extensions the
    OpenAI SDK won't accept as kwargs, so they ride in extra_body — llama.cpp's
    server merges that into the request body; non-llama.cpp endpoints ignore
    unknown keys. Set any RECOVERY_* value negative to omit it.  When
    NX_BACKEND=vllm the llama.cpp-only knobs are set to None instead of
    their defaults, so the checks below miss them automatically."""
    opts = {}
    if st.RECOVERY_TEMPERATURE is not None and st.RECOVERY_TEMPERATURE >= 0:
        opts["temperature"] = st.RECOVERY_TEMPERATURE
    if st.RECOVERY_TOP_P is not None and st.RECOVERY_TOP_P >= 0:
        opts["top_p"] = st.RECOVERY_TOP_P
    extra = {}
    if RECOVERY_MIN_P is not None and RECOVERY_MIN_P >= 0:
        extra["min_p"] = RECOVERY_MIN_P
    if RECOVERY_REPEAT_PENALTY is not None and RECOVERY_REPEAT_PENALTY >= 0:
        extra["repeat_penalty"] = RECOVERY_REPEAT_PENALTY
        if RECOVERY_REPEAT_LAST_N is not None:
            extra["repeat_last_n"] = RECOVERY_REPEAT_LAST_N
    if RECOVERY_DRY_MULTIPLIER is not None and RECOVERY_DRY_MULTIPLIER > 0:
        extra["dry_multiplier"] = RECOVERY_DRY_MULTIPLIER
        if RECOVERY_DRY_BASE is not None:
            extra["dry_base"] = RECOVERY_DRY_BASE
        if RECOVERY_DRY_ALLOWED_LENGTH is not None:
            extra["dry_allowed_length"] = RECOVERY_DRY_ALLOWED_LENGTH
        if RECOVERY_DRY_SEQUENCE_BREAKERS is not None:
            extra["dry_sequence_breakers"] = RECOVERY_DRY_SEQUENCE_BREAKERS
    if extra:
        opts["extra_body"] = extra
    return opts


def open_stream(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] = TOOLS,
    tool_choice: str | dict[str, Any] | None = None,
    max_tokens: int | None = None,
    recovery_sampling: bool = False,
):
    """Open a streaming completion. Retries without tools= if the server rejects
    that param; returns None (after a friendly message) on connection/API failure."""
    # _LoggingStream lives in _turn.py, which itself imports open_stream from
    # this module — a lazy import here (instead of at module scope) avoids a
    # circular import between the two.
    from ._turn import _LoggingStream
    assert st.client is not None and st.MODEL is not None
    try:
        _prune_empty_assistant_messages(messages)
        token_limit = MAX_COMPLETION_TOKENS if max_tokens is None else max_tokens
        request_opts: dict[str, Any] = {}
        if token_limit > 0:
            request_opts["max_tokens"] = token_limit
        if recovery_sampling:
            request_opts.update(_recovery_sampling_opts())
        if tool_choice is not None:
            request_opts["tool_choice"] = tool_choice
        # Per-request body fields (e.g. OpenRouter provider routing) ride in
        # extra_body. Merged here once so both the tools and no-tools fallback
        # branches below carry it through unchanged.
        if st.ACTIVE:
            request_opts.update(st.ACTIVE.extra_request_kwargs())
        grammar_extra = _grammar_extra_body(tools)
        if grammar_extra:
            # Merge (not overwrite) — a source's own extra_body (e.g.
            # OpenRouter provider routing) must survive alongside grammar.
            request_opts.setdefault("extra_body", {}).update(grammar_extra)
        try:
            _log_event("req", {"model": st.MODEL, "messages": messages, "tools": tools, "stream": True, **request_opts})
            stream = st.client.chat.completions.create(
                model=st.MODEL, messages=messages, tools=tools, stream=True,
                stream_options={"include_usage": True}, **request_opts)
        except APIConnectionError:
            raise  # server unreachable — don't bother retrying without tools
        except httpx.HTTPError:
            raise  # transport failure — retrying without tools won't help
        except Exception:  # reachable but rejected tools= → text-protocol fallback
            # If extra_body (e.g. min_p / repeat_penalty / dry_*) is also
            # rejected — vLLM's speculative decoder doesn't support min_p etc. —
            # strip it too.  Try no-tools first, then no-tools + no-extra.
            for fallback_keys in ({"tool_choice"}, {"tool_choice", "extra_body"}):
                fallback_opts = {k: v for k, v in request_opts.items()
                                 if k not in fallback_keys}
                try:
                    _tag = "no-tools" if fallback_keys == {"tool_choice"} else "no-tools-no-extra"
                    _log_event("req", {"model": st.MODEL, "messages": messages, "stream": True, "_fallback": _tag, **fallback_opts})
                    stream = st.client.chat.completions.create(
                        model=st.MODEL, messages=messages, stream=True,
                        stream_options={"include_usage": True}, **fallback_opts)
                    break  # succeeded
                except (APIConnectionError, httpx.HTTPError):
                    raise  # transport failure — don't retry further
                except Exception:
                    continue  # try next fallback level
            else:
                raise  # all fallbacks exhausted
        # Wrap the stream so every chunk is captured to the log on its way out.
        return _LoggingStream(stream, _llog)
    except (APIConnectionError, httpx.HTTPError):
        print(f"{C.ERROR}  ✗ can't reach {st.client.base_url} — is the server up? "
              f"Set NX_BASE_URL (and NX_MODEL) to point at it.{RESET}")
    except Exception as e:
        print(f"{C.ERROR}  ✗ API error: {type(e).__name__}: {e}{RESET}")
    return None


# --- context compression ----------------------------------------------------
# Summarize the older turns of `messages` into a single user-role turn, keeping
# the system prompt and the last K turns verbatim. Frees context without losing
# the model's grip on what it was just doing.
st.COMPRESS_KEEP = 2  # how many recent turns to leave untouched


def compress(
    messages: list[dict[str, Any]], keep: int | None = None, auto: bool = False
) -> tuple[int, int, int] | None:
    """Ask the model to summarize everything except system + last `keep` turns.

    Mutates `messages` in place on success: replaces the middle slice with a
    single user-role summary turn. Returns (kept_n, summarized_n, summary_chars)
    or None on failure (in which case `messages` is untouched).

    Non-streaming on purpose — we want the whole summary before splicing it in,
    and a spinner for a one-shot summary would be visual noise.

    When auto=True the keep count is raised above COMPRESS_KEEP so
    auto-compression (triggered when the context window fills past
    AUTOCOMPRESS_PERCENT) is more conservative than a manual /compress —
    it keeps roughly the last third of the conversation verbatim so
    in-progress work and recent tool results survive the fold.
    """
    assert st.client is not None and st.MODEL is not None
    if keep is None:
        keep = st.COMPRESS_KEEP
    # Layout: [system?, ..., user, assistant, tool, ..., user, assistant(tool_calls)?, ...]
    # We assume messages[0] is the system prompt (matches how main() builds it).
    # Anything before the "tail" we want to summarize; the tail stays verbatim.
    sys_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    if auto:
        body_len = len(messages) - (1 if sys_msg else 0)
        keep = max(keep, body_len // 3)
    if len(messages) <= 1 + keep:
        return None  # nothing to compress

    body = messages[1:] if sys_msg else messages
    if len(body) <= keep:
        return None

    head, tail = body[:-keep], body[-keep:]
    summarized_n = len(head)

    # The tail must start on a turn the chat template can render — see
    # _drop_leading_dangling_tool_turns for why. Do not simply discard the
    # leading turns it removes: they are real context, and calling them
    # "summarized" without putting them in the summary prompt loses data.
    # Move them into `head` so they are actually folded into the summary.
    safe_tail, n_dropped = _drop_leading_dangling_tool_turns(tail)
    if n_dropped:
        head.extend(tail[:n_dropped])
        summarized_n += n_dropped
    tail = safe_tail

    # Render the head as plain text the model can summarize. Tool outputs are
    # the bulkiest part of a real session — include them but truncate each one
    # so a single huge read_file doesn't blow up the summary prompt itself.
    def _render(msgs: list[dict[str, Any]]) -> str:
        out = []
        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content")
            if content is None and m.get("tool_calls"):
                # assistant tool-call turn — show the calls so the summary knows what ran
                calls = ", ".join(
                    f"{c['function']['name']}({c['function']['arguments']})"
                    for c in m["tool_calls"]
                )
                out.append(f"[{role}] → {calls}")
            elif isinstance(content, list):
                # some servers return content as a list of parts; flatten it
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                out.append(f"[{role}] {content[:2000]}")
            else:
                out.append(f"[{role}] {(content or '')[:2000]}")
        return "\n\n".join(out)

    summary_prompt = (
        "Summarize the following conversation history for context retention. "
        "Preserve: the original user goal/task, key decisions made, file paths "
        "and identifiers touched, current state of any in-progress work, and "
        "any unresolved questions. Drop: raw tool outputs, full file contents, "
        "and verbose back-and-forth — keep it dense and information-rich. "
        "Write in the same language as the conversation. Output ONLY the "
        "summary, no preamble.\n\n"
        f"---\n{_render(head)}\n---"
    )

    payload = [{"role": "user", "content": summary_prompt}]
    try:
        _log_event("req", {"model": st.MODEL, "messages": payload, "stream": False, "_purpose": "compress"})
        resp = st.client.chat.completions.create(
            model=st.MODEL, messages=payload, stream=False,
            **(st.ACTIVE.extra_request_kwargs() if st.ACTIVE else {}))
        try:
            _log_event("resp", {"_purpose": "compress", "data": resp.model_dump()})
        except Exception:
            pass  # never let logging break the summary call
    except APIConnectionError:
        print(f"{C.ERROR}  ✗ can't reach {st.client.base_url} — context unchanged{RESET}")
        return None
    except Exception as e:
        print(f"{C.ERROR}  ✗ compress failed: {type(e).__name__}: {e}{RESET}")
        return None

    _record_usage(
        getattr(resp, "usage", None),
        fallback_chars=len(resp.choices[0].message.content or ""),
        fallback_prompt_chars=sum(len(str(m.get("content") or "")) for m in payload),
    )

    summary = (resp.choices[0].message.content or "").strip()
    if not summary:
        print(f"{C.ERROR}  ✗ compress returned empty summary — context unchanged{RESET}")
        return None

    header = f"[Compressed context — {summarized_n} earlier turns summarized; last {len(tail)} turns kept verbatim]"
    new_mid = [{"role": "user", "content": f"{header}\n\n{summary}"}]
    messages[:] = ([sys_msg] if sys_msg else []) + new_mid + tail
    return len(tail), summarized_n, len(summary)


def _maybe_autocompress(messages: list[dict[str, Any]], prompt_tokens: int) -> bool:
    """Silently compress when context usage crosses the AUTOCOMPRESS_PERCENT
    threshold. Called after each settled model turn with the last turn's
    `prompt_tokens` (the size of the context the model just saw).

    Returns True if a compression happened, False otherwise. Guards:
      - AUTOCOMPRESS_PERCENT == 0 → disabled
      - ACTIVE context window unknown / ≤ 0 → can't compute a ratio
      - ratio < threshold → nothing to do
      - compress() itself bails when there's too little to fold
    """
    if st.AUTOCOMPRESS_PERCENT <= 0:
        return False
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        return False
    if st.ACTIVE is None:
        return False
    mx = st.ACTIVE.resolve_context_window()
    if not isinstance(mx, int) or mx <= 0:
        return False
    ratio = prompt_tokens / mx
    if ratio * 100 < st.AUTOCOMPRESS_PERCENT:
        return False
    # Via the top-level package (not a bare same-module reference) so a test
    # that monkeypatches `nx.compress` (to stub out the network call, or
    # to assert it's never invoked) takes effect.
    import nx as _m
    result = _m.compress(messages, auto=True)
    if result is None:
        return False
    kept_n, summarized_n, summary_chars = result
    pct = int(ratio * 100)
    print(f"{DIM}  ↻ auto-compressed {summarized_n} turns → 1 summary "
          f"({summary_chars} chars), kept last {kept_n} verbatim "
          f"(context was at {pct}% of {_abbr(mx)}){RESET}")
    return True
