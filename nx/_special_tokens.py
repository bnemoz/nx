"""Detect and strip leaked tokenizer-internal special tokens from model output.

Some models (notably DeepSeek-V4 served via vLLM with dspark speculative
decoding) will, when destabilized (e.g. by a harness error like a missing
`kill` binary), collapse into spamming their tokenizer-internal special
tokens.  The clearest signal is the DeepSeek begin/end-of-sentence token
with a FULL-WIDTH pipe (｜, U+FF5C) and SentencePiece marker (▁):
  <｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>...
These NEVER appear in legitimate model output — they are the tokenizer's
internal representation, not a string the model would legitimately emit or
discuss.  (Contrast with half-width "<|endoftext|>" which GPT-2 tasks may
legitimately reference.)  Once leaked, the spam enters the assistant message
history and on the next turn the model sees its own gibberish in context,
reinforcing the collapse and often drifting into regurgitated training data
(Chinese-language passages are a common artifact).
"""
import re

from ._env import _env_int

# Tokenizer-internal markers that are ALWAYS leaks — never legitimate output.
# Full-width pipe (｜ = U+FF5C) + SentencePiece ▁ = U+2581.
_LEAKED_SPECIAL_TOKEN_RE = re.compile(
    r"<\｜(?:begin▁of▁sentence|end▁of▁sentence|tool▁calls▁begin|tool▁calls▁end"
    r"|tool▁call▁begin|tool▁call▁end|reasoning▁begin|reasoning▁end)\｜>"
)
# When a leak run is detected in-stream, cut after this many leaked chunks.
LEAKED_TOKEN_RUN_LIMIT = _env_int("NX_LEAKED_TOKEN_RUN_LIMIT", 3)
# Also detect the degeneration via repeated identical content chunks.  If the
# last N streamed chunks are byte-identical AND short (<=40 chars), the model
# is in a repetition collapse — cut the stream and recover.
REPEAT_CHUNK_LIMIT = _env_int("NX_REPEAT_CHUNK_LIMIT", 6)


def _strip_leaked_tokens(text):
    """Remove tokenizer-internal special tokens that leaked into model output.

    DeepSeek's ``<｜begin▁of▁sentence｜>`` (full-width pipe) is a tokenizer
    artifact that should never appear in content; it signals the model has
    degenerated into a special-token repetition collapse.  Stripping it
    (a) keeps the user-visible output clean, and crucially (b) prevents the
    spam from entering the assistant message history where the model would
    see it on the next turn and have the collapse reinforced.
    """
    if not text:
        return text
    return _LEAKED_SPECIAL_TOKEN_RE.sub("", text)


def _is_leak_chunk(chunk):
    """True if a streamed content/reasoning chunk is (mostly) leaked special
    tokens — i.e. the model has entered the special-token collapse."""
    if not chunk:
        return False
    stripped = _LEAKED_SPECIAL_TOKEN_RE.sub("", chunk)
    # If removing the leaked tokens leaves almost nothing, this chunk was spam.
    return len(stripped.strip()) <= 2 and bool(_LEAKED_SPECIAL_TOKEN_RE.search(chunk))
