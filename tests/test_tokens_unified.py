"""nx/_tokens.py — the shared estimator's contract (0.159.2).

tiktoken may or may not be installed in the test env, and its first real use
may need the vocab file, so exact-vs-heuristic assertions route through the
module's own state (the _encoding singleton) instead of assuming either.
"""
import math


def test_empty_is_zero():
    from nx._tokens import estimate_tokens
    assert estimate_tokens("") == 0


def test_fallback_is_ceil_over_3_5_chars(monkeypatch):
    import nx._tokens as tk
    monkeypatch.setattr(tk, "_encoding", False)  # latched-unavailable
    assert tk.estimate_tokens("a" * 3) == 1        # max(1, ceil(3/3.5))
    assert tk.estimate_tokens("a" * 7) == 2
    assert tk.estimate_tokens("a" * 8) == 3        # ceil(8/3.5) == 3
    assert tk.estimate_tokens("a" * 400) == math.ceil(400 / 3.5)


def test_fallback_counts_chars_not_bytes(monkeypatch):
    """The old _occompact bytes-based estimator overcounted multibyte
    payloads up to 4x; the shared one counts characters."""
    import nx._tokens as tk
    monkeypatch.setattr(tk, "_encoding", False)
    text = "\u00e9" * 100  # 2 utf-8 bytes each
    assert tk.estimate_tokens(text) == math.ceil(100 / 3.5)


def test_from_chars_matches_estimate_tokens(monkeypatch):
    import nx._tokens as tk
    monkeypatch.setattr(tk, "_encoding", False)
    for n in (1, 3, 4, 7, 8, 400, 4000):
        assert tk.estimate_tokens_from_chars(n) == tk.estimate_tokens("x" * n)
    assert tk.estimate_tokens_from_chars(0) == 0
    assert tk.estimate_tokens_from_chars(-5) == 0


def test_estimate_tokens_accepts_content_parts():
    from nx._tokens import estimate_tokens
    parts = [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]
    assert estimate_tokens(parts) == estimate_tokens("hello world")
    assert estimate_tokens([{"nope": 1}]) == 0


def test_message_tokens_str_parts_and_tool_calls():
    from nx._tokens import estimate_message_tokens, estimate_tokens
    msg = {"role": "user", "content": "hello there"}
    assert estimate_message_tokens(msg) == estimate_tokens("hello there")
    parts = {"role": "user", "content": [{"type": "text", "text": "a" * 20}]}
    assert estimate_message_tokens(parts) == estimate_tokens("a" * 20)
    tc = {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "run_bash", "arguments": "{\"command\": \"ls\"}"}}]}
    assert estimate_message_tokens(tc) == (
        estimate_tokens("run_bash") + estimate_tokens("{\"command\": \"ls\"}"))


def test_obs_and_ctxreport_delegate():
    from nx import _ctxreport, _obs, _tokens
    text = "x" * 77
    assert _ctxreport._estimate_tokens(text) == _tokens.estimate_tokens(text)
    assert _obs.estimate_tokens(text) == _tokens.estimate_tokens(text)


def test_occompact_is_chars_based_now():
    from nx import _occompact, _tokens
    text = "\u00e9" * 100
    assert _occompact._estimate_tokens(text) == _tokens.estimate_tokens(text)


def test_no_module_level_tiktoken_import():
    """nx must stay snappy/offline at import: tiktoken is imported inside the
    estimator's first use, never at module import."""
    import subprocess
    import sys
    probe = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'tiktoken':\n"
        "            raise ImportError('blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import nx._tokens\n"
        "print('import-ok')\n"
        "assert nx._tokens.estimate_tokens('hello world') > 0\n"
        "print('fallback-ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", probe],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "import-ok" in r.stdout and "fallback-ok" in r.stdout
