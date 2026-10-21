"""The one token estimator nx shares (chars/3.5, stdlib only).

Before this module, five call sites carried five different char→token
heuristics (chars//4, utf-8-bytes//4, round(chars/3.5), ...) so /ctx, /cost,
the footer and the compaction gate quietly disagreed about how big the same
prompt was. This module is the single source of truth for that conversion:

  estimate_tokens(text)        ceil(chars / 3.5)
  estimate_message_tokens(msg) one OpenAI-shaped message: string or
                               list-of-parts `content` plus native
                               `tool_calls` (function name + arguments).

An earlier version reached for tiktoken/cl100k_base when installed. It was
removed (0.164.1): nx targets any OpenAI-compatible endpoint, most of them
serve non-OpenAI models, so cl100k counts were exact only for a model family
nx doesn't default to — an optional dependency that changed the guarded
floor's number depending on what happened to be in the venv, downloaded a
vocab file on first use, and still disagreed with the server's own /tokenize
endpoint (which remains the exact path, used when a source has one). The
chars/3.5 estimate over-counts on code/JSON by ~7% and is the safe direction
for every consumer of this module: compaction triggers early, the floor guard
stays conservative, /ctx slightly over-reports.

The divisor: natural-language code and prose tokenize at roughly 3.3–3.5
characters per token for cl100k-family BPEs (the older chars//4 rule of thumb
was a rounder, looser version of the same fact). Counted on CHARACTERS, never
UTF-8 bytes — a multibyte payload would otherwise be overcounted by up to 4x
against a tokenizer that splits on text, not bytes.
"""
from __future__ import annotations

import math
from typing import Any

# The one rate (see module docstring). Kept as a constant so tests and call
# sites reference one number instead of re-typing a magic divisor.
CHARS_PER_TOKEN = 3.5


def estimate_tokens_from_chars(n_chars: int) -> int:
    """The chars/3.5 heuristic on a character COUNT.

    Exposed for call sites that only ever kept a running character count (the
    per-turn footer and _metrics' usage fallback) and must not allocate a
    string of that size to convert it.
    """
    if n_chars <= 0:
        return 0
    return max(1, math.ceil(n_chars / CHARS_PER_TOKEN))


def estimate_tokens(text: str) -> int:
    """Tokens in `text`: ceil(chars / 3.5). Empty/None-ish text is 0."""
    if not text:
        return 0
    if isinstance(text, list):  # content-parts form (the shape /ctx feeds us)
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        if not text:
            return 0
    elif not isinstance(text, str):
        text = str(text)
    return estimate_tokens_from_chars(len(text))


def message_content_text(msg: dict[str, Any]) -> str:
    """The plain text of a message's `content`: str as-is, list-of-parts
    joined over its text parts, anything else as ''."""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def estimate_message_tokens(msg: dict[str, Any]) -> int:
    """Estimated tokens for one message, including native tool_calls args."""
    n = estimate_tokens(message_content_text(msg))
    tcs = msg.get("tool_calls") or []
    for tc in tcs:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        n += estimate_tokens(str(fn.get("name") or "")) + estimate_tokens(str(fn.get("arguments") or ""))
    return n
