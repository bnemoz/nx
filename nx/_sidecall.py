"""One shared one-shot JSON side call (0.156.3).

Three sites used to hand-roll the same pipeline — the risk classifier
(_risk._assess_risk), the advisor consult (_subagent_tiers._consult_advisor)
and /build's inline fix (_build._ask_main_model): a client/MODEL guard, a
two-message payload, a non-streaming chat.completions.create with
extra_request_kwargs spread, metrics recording with chars//4 fallbacks, and
a json.loads retry ladder. This module owns that machinery; callers own
everything that is theirs: failure mapping, budgets, deadlines, prompts.

Contract: `side_call_json` returns `(parsed_or_None, raw_text)`. The dict is
the successfully parsed reply (None on any failure — no source, connection
ladder exhausted, every parse attempt unparseable, or a non-dict JSON body);
raw_text is always a string, the stripped content of the LAST response (""
when there was none). It never raises.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from ._openai_client import APIConnectionError
from ._logging import _log_event
from . import _state as st


def _parse_json(text: str, loose: bool = False) -> dict | None:
    """Parse `text` as a JSON object, or None. loose=True first brace-slices
    to the outermost {...} so a prose-wrapped reply still parses (the shape
    /build's model tolerates)."""
    if loose:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        text = text[start:end + 1]
    try:
        obj = json.loads(text)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def side_call_json(
    system: str,
    user: str,
    *,
    timeout: float = 30.0,
    parse_attempts: int = 1,
    retry_prompt: str | None = None,
    purpose: str = "side_call",
    conn_retries: int = 0,
    conn_retry_seconds: float = 0.0,
    loose_json: bool = False,
    log_event: "Callable[..., None] | None" = None,
) -> "tuple[dict | None, str]":
    """One-shot, no-tools call to the session's model. Returns
    (parsed_or_None, raw_text). None on any failure; raw_text always set
    (possibly "").

    parse_attempts: json.loads re-calls with the SAME payload on unparseable
    replies. retry_prompt (optional) is appended to the payload between
    attempts as a re-ask instruction (risk's sharp follow-up). conn_retries:
    extra attempts against APIConnectionError per request — the ladder is
    shared across all parse attempts, so a dead server costs
    conn_retries+1 calls total, not per parse attempt;
    conn_retry_seconds is slept between them.

    log_event: overrides the req/resp logger — callers that tests patch a
    module-level `_log_event` binding on (risk) pass theirs so the existing
    patch seam keeps covering the call."""
    _log = log_event or _log_event
    if st.client is None or st.MODEL is None:
        return (None, "")
    payload: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    max_conn = max(0, conn_retries) + 1
    conn_attempts_used = 0
    text = ""
    try:
        for parse_attempt in range(max(1, parse_attempts)):
            resp = None
            while conn_attempts_used < max_conn:
                try:
                    _log("req", {"model": st.MODEL, "messages": payload,
                                       "stream": False, "_purpose": purpose,
                                       "_attempt": conn_attempts_used + 1,
                                       "_parse_attempt": parse_attempt + 1})
                    resp = st.client.chat.completions.create(
                        model=st.MODEL, messages=payload, stream=False,
                        timeout=timeout,
                        **(st.ACTIVE.extra_request_kwargs() if st.ACTIVE else {}))
                    break
                except APIConnectionError:
                    conn_attempts_used += 1
                    if conn_attempts_used >= max_conn:
                        return (None, text)
                    time.sleep(max(0.0, conn_retry_seconds))
            if resp is None:
                return (None, text)
            try:
                _log("resp", {"_purpose": purpose, "data": resp.model_dump()})
            except Exception:
                pass
            text = (resp.choices[0].message.content or "").strip()
            # Side-calls are real usage: every attempt is on the user's bill,
            # so record it into the same totals /cost prints — with BOTH
            # chars//4 fallbacks for servers that send no usage object (this
            # is the prompt-char fallback /build's own call used to miss).
            from ._metrics import _record_api_call
            _record_api_call(
                resp,
                fallback_chars=len(text),
                fallback_prompt_chars=lambda: sum(
                    len(str(m.get("content") or "")) for m in payload),
            )
            obj = _parse_json(text, loose=loose_json)
            if obj is not None:
                return (obj, text)
            # Unparseable — optionally append a re-ask instruction and retry.
            if retry_prompt is not None and parse_attempt < max(1, parse_attempts) - 1:
                payload.append({"role": "assistant", "content": text or "(no output)"})
                payload.append({"role": "user", "content": retry_prompt})
        return (None, text)
    except Exception:
        # Any unexpected failure (bad response shape, metrics blowup) is a
        # None reply, never a crash — the caller owns the fallback.
        return (None, text)


__all__ = ["side_call_json"]
