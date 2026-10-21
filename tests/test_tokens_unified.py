"""nx/_tokens.py — the shared estimator's contract (0.159.2).

The estimator is chars/3.5, full stop (tiktoken was removed at 0.164.1 —
cl100k counts were exact only for a model family nx doesn't default to, and
the guarded floor's number must not depend on what is in the venv).
"""
import math

from nx import _tokens as tk


def test_empty_is_zero():
    assert tk.estimate_tokens("") == 0
    assert tk.estimate_tokens(None) == 0


def test_estimate_is_ceil_over_3_5_chars():
    assert tk.estimate_tokens("a" * 3) == 1        # max(1, ceil(3/3.5))
    assert tk.estimate_tokens("a" * 7) == 2
    assert tk.estimate_tokens("a" * 8) == 3        # ceil(8/3.5) == 3
    assert tk.estimate_tokens("a" * 400) == math.ceil(400 / 3.5)


def test_counts_chars_not_bytes():
    """The old _occompact bytes-based estimator overcounted multibyte
    payloads up to 4x; the shared one counts characters."""
    text = "é" * 100  # 2 utf-8 bytes each
    assert tk.estimate_tokens(text) == math.ceil(100 / 3.5)


def test_from_chars_matches_estimate_tokens():
    for n in (1, 3, 4, 7, 8, 400, 4000):
        assert tk.estimate_tokens_from_chars(n) == tk.estimate_tokens("x" * n)
    assert tk.estimate_tokens_from_chars(0) == 0
    assert tk.estimate_tokens_from_chars(-5) == 0


def test_estimate_tokens_accepts_content_parts():
    parts = [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]
    assert tk.estimate_tokens(parts) == tk.estimate_tokens("hello world")
    assert tk.estimate_tokens([{"nope": 1}]) == 0


def test_message_tokens_str_parts_and_tool_calls():
    msg = {"role": "user", "content": "hello there"}
    assert tk.estimate_message_tokens(msg) == tk.estimate_tokens("hello there")
    parts = {"role": "user", "content": [{"type": "text", "text": "a" * 20}]}
    assert tk.estimate_message_tokens(parts) == tk.estimate_tokens("a" * 20)
    tc = {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "run_bash", "arguments": "{\"command\": \"ls\"}"}}]}
    assert tk.estimate_message_tokens(tc) == (
        tk.estimate_tokens("run_bash") + tk.estimate_tokens("{\"command\": \"ls\"}"))


def test_obs_and_ctxreport_delegate():
    from nx import _ctxreport, _obs
    text = "x" * 77
    assert _ctxreport._estimate_tokens(text) == tk.estimate_tokens(text)
    assert _obs.estimate_tokens(text) == tk.estimate_tokens(text)


def test_occompact_is_chars_based_now():
    from nx import _occompact
    text = "é" * 100
    assert _occompact._estimate_tokens(text) == tk.estimate_tokens(text)


def test_no_tokenizer_dependency_at_import_or_runtime():
    """nx must stay snappy/offline: the estimator is pure stdlib math. A
    blocked-import probe proves nothing tries to pull a tokenizer module."""
    import subprocess
    import sys
    probe = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name in ('tiktoken', 'transformers'):\n"
        "            raise ImportError('blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import nx._tokens\n"
        "print('import-ok')\n"
        "assert nx._tokens.estimate_tokens('hello world') > 0\n"
        "print('estimate-ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", probe],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "import-ok" in r.stdout and "estimate-ok" in r.stdout
