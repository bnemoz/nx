"""Phase 1 (transcript detail + as-sent) + Phase 2 (the last-compression
ledger read side) of the explainable harness: the ContextView detail pane, the
live/as-sent toggle over st._LAST_PROMPT, the inline `/context t <n>` output,
and `/compress --show` / CompressionView over st._LAST_COMPRESSION — same
fixtures as test_fullscreen.py's ContextView block."""

from nx import _ctxreport

# A ledger in the write side's shape: ranges are HALF-OPEN 0-based indexes into
# the PRE-compression list, so [0,2) must render as the 1-based inclusive [1..2].
_LEDGER = {
    "kept": 3, "dropped": 0, "summarized": 4, "summary_tokens_est": 187,
    "pinned": 1, "before_messages": 9, "after_messages": 5,
    "ts": 1_750_000_000.0,
    "ranges": [
        {"start": 0, "end": 2, "action": "kept"},
        {"start": 2, "end": 6, "action": "summarized"},
        {"start": 6, "end": 7, "action": "pinned-kept"},
        {"start": 7, "end": 9, "action": "kept"},
    ],
    "a_key_the_renderer_has_never_seen": "ignore me",
}


class _CtxState:
    """The bit of `_ReplState` the view reads: the live message list."""

    def __init__(self, n: int = 3) -> None:
        self.messages = [{"role": "system", "content": "sys prompt"}]
        for i in range(n):
            self.messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"msg {i} " + "x" * 80})


def _digest(msg):
    return _ctxreport._message_digest(msg)


def test_detail_pane_renders_the_full_message():
    from nx import _render
    state = _CtxState(2)
    state.messages.append({
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": "search_files",
                                     "arguments": "{\"pattern\": \"x\"}"}}]})
    view = _ctxreport.context_view(state)
    view.on_key("j", None)          # messages[1] = the padded message
    view.on_key("enter", None)
    assert view.detail
    body = "\n".join(view.render(240, 40))
    assert _render.BOX_TL in body
    # FULL text, not the 64-char preview: wrapped at 72, none elided (the
    # strip accounts for the pane's ANSI; the pane may split the run)
    import re as _re
    plain = _re.sub(r"\x1b\[[0-9;]*m", "", body)
    assert "x" * 66 in plain
    # esc returns to the list (and does NOT quit the view)
    assert view.on_key("esc", None) == "repaint"
    assert not view.detail
    assert view.on_key("esc", None) == "quit"
    # the tool-call row renders its name + args in the pane too
    view.on_key("j", None)
    view.on_key("j", None)
    view.on_key("enter", None)
    body = "\n".join(view.render(240, 40))
    assert "tool call: search_files" in body
    assert '{"pattern": "x"}' in body


def test_detail_pane_shows_tool_result_message():
    state = _CtxState(1)
    state.messages.append({"role": "tool", "content": "42 matches found"})
    view = _ctxreport.context_view(state)
    view.on_key("j", None)
    view.on_key("j", None)          # messages[2] = the tool result
    view.on_key("enter", None)
    assert "42 matches found" in "\n".join(view.render(240, 40))


def test_as_sent_toggle_marks_identical_and_differing(monkeypatch):
    from nx import _state as st
    state = _CtxState(3)
    sent = [{"role": m["role"], "index": i,
             "digest": _digest(m) if i != 1 else "0" * 64}
            for i, m in enumerate(state.messages)]
    monkeypatch.setattr(st, "_LAST_PROMPT", sent, raising=False)
    view = _ctxreport.context_view(state)
    view.on_key("a", None)          # toggle before opening = list-wide hint
    view.on_key("enter", None)      # row 0 — digest matches
    body = "\n".join(view.render(240, 40))
    assert "as-sent index=0" in body
    assert "identical to live" in body
    view.on_key("esc", None)
    view.on_key("j", None)          # row 1 — digest differs
    view.on_key("enter", None)
    body = "\n".join(view.render(240, 40))
    import re as _re
    assert "differs from live (elided/summarized)" in _re.sub(
        r"\x1b\[[0-9;]*m", "", body)
    assert "as-sent index=1" in _re.sub(r"\x1b\[[0-9;]*m", "", body)


def test_as_sent_toggle_without_a_settled_call_shows_placeholder(monkeypatch):
    from nx import _state as st
    monkeypatch.setattr(st, "_LAST_PROMPT", None, raising=False)
    view = _ctxreport.context_view(_CtxState())
    view.on_key("enter", None)
    view.on_key("a", None)
    body = "\n".join(view.render(240, 40))
    assert "— no settled call recorded —" in body
    # and the list view below still renders (stays on live once esc'd)
    view.on_key("esc", None)
    assert not view.as_sent or True
    assert "context" in "\n".join(view.render(120, 20))


def test_as_sent_missing_attribute_falls_back_like_none(monkeypatch):
    from nx import _state as st
    monkeypatch.delattr(st, "_LAST_PROMPT", raising=False)
    view = _ctxreport.context_view(_CtxState())
    view.on_key("a", None)
    view.on_key("enter", None)
    assert "— no settled call recorded —" in "\n".join(view.render(240, 40))


def test_context_t_inline_prints_one_message_transcript(monkeypatch, capsys):
    from nx import _repl_settings_cmds as cmds
    state = _CtxState(3)
    cmds._cmd_ctx("/context t 2", state)
    out = capsys.readouterr().out
    assert "role=user" in out
    assert "msg 0" in out and "x" * 66 in out   # full text, not the preview
    # strip ANSI the view's pane adds; the bytes must match the helper
    import re
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
    assert plain.strip() == _ctxreport._transcript_detail(
        state.messages[1], 1).strip()


def test_context_t_inline_error_handling(monkeypatch, capsys):
    from nx import _repl_settings_cmds as cmds
    cmds._cmd_ctx("/context t", _CtxState(3))
    assert "usage" in capsys.readouterr().out
    cmds._cmd_ctx("/context t 99", _CtxState(3))
    assert "outside 1..4" in capsys.readouterr().out


def test_message_digest_is_stable_sha256_of_role_and_text():
    msg = {"role": "user", "content": "hello"}
    d1 = _digest(msg)
    import hashlib
    assert d1 == hashlib.sha256(
        b"user\x00hello").hexdigest()
    assert d1 == _digest({"role": "user", "content": "hello"})


# --- Phase 2: the last-compression ledger, read side -------------------------

def test_compression_report_renders_counts_and_ranges():
    import time
    lines = _ctxreport._compression_report_lines(_LEDGER)
    body = "\n".join(lines)
    # header: local time + before → after
    assert time.strftime("%Y-%m-%d %H:%M:%S",
                         time.localtime(_LEDGER["ts"])) in lines[0]
    assert "9 → 5 messages" in lines[0]
    # counts line
    counts = lines[1]
    for want in ("kept 3", "dropped 0", "summarized 4", "pinned 1",
                 "summary ~187 tok est"):
        assert want in counts, want
    # one line per range, in order
    range_lines = [ln for ln in lines if ln.strip().startswith("[")]
    assert len(range_lines) == len(_LEDGER["ranges"])
    assert "ignore me" not in body          # unknown keys ignored


def test_compression_ranges_render_1_based_inclusive():
    # The write side's [start, end) is HALF-OPEN and 0-based: [0,2) covers
    # messages 1 and 2, [2,6) covers 3..6.
    lines = _ctxreport._compression_report_lines(_LEDGER)
    range_lines = [ln for ln in lines if ln.strip().startswith("[")]
    assert [ln.strip().split("]")[0] + "]" for ln in range_lines] == [
        "[1..2]", "[3..6]", "[7..7]", "[8..9]"]


def test_kept_action_renders_as_verbatim_untouched():
    # "kept" includes the system + prefix block, so it must say the block was
    # never touched — not merely "kept".
    lines = _ctxreport._compression_report_lines(_LEDGER)
    kept = [ln for ln in lines if ln.strip().startswith("[1..2]")]
    assert len(kept) == 1
    assert "verbatim, untouched" in kept[0]
    # the other actions get their own wording, an unknown one passes through
    assert any("pinned" in ln and "[7..7]" in ln for ln in lines)
    lines = _ctxreport._compression_report_lines(
        {"ranges": [{"start": 0, "end": 1, "action": "spliced"}]})
    assert "[1..1] spliced" in "\n".join(lines)


def test_compression_report_none_placeholder(monkeypatch):
    from nx import _state as st
    monkeypatch.setattr(st, "_LAST_COMPRESSION", None, raising=False)
    assert _ctxreport._compression_report_lines() == [
        "  no compression recorded this session"]
    # an attribute that does not exist at all behaves like None
    monkeypatch.delattr(st, "_LAST_COMPRESSION", raising=False)
    assert "no compression recorded this session" in \
        _ctxreport._compression_report_lines()[0]


def test_compression_report_survives_a_garbage_ledger():
    # Defensive reads: missing counts render 0, a broken range is skipped.
    lines = _ctxreport._compression_report_lines({"ranges": [
        {"start": None, "end": 3, "action": "kept"},
        {"start": 5, "end": 2, "action": "kept"},
        "not-a-dict",
    ]})
    assert "kept 0" in lines[1] and "summary ~0 tok est" in lines[1]
    assert not [ln for ln in lines if ln.strip().startswith("[")]


def test_compress_show_prints_the_ledger_inline(monkeypatch, capsys):
    from nx import _repl_settings_cmds as cmds
    from nx import _state as st
    monkeypatch.setattr(st, "_LAST_COMPRESSION", _LEDGER, raising=False)
    monkeypatch.setenv("NX_FULLSCREEN", "0")   # inline, no alternate screen
    cmds._cmd_compress("/compress --show", _CtxState(3))
    out = capsys.readouterr().out
    import re
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
    assert _ctxreport._compression_report_lines(_LEDGER) == [
        ln for ln in plain.splitlines()]
    assert "9 → 5 messages" in plain and "[3..6]" in plain


def test_compress_show_alias_and_none_case(monkeypatch, capsys):
    from nx import _repl_settings_cmds as cmds
    from nx import _state as st
    monkeypatch.setenv("NX_FULLSCREEN", "0")
    # the hidden /compact alias reaches the same branch
    monkeypatch.setattr(st, "_LAST_COMPRESSION", _LEDGER, raising=False)
    cmds._cmd_compress("/compact show", _CtxState(3))
    assert "[1..2]" in capsys.readouterr().out
    monkeypatch.setattr(st, "_LAST_COMPRESSION", None, raising=False)
    cmds._cmd_compress("/compress --show", _CtxState(3))
    assert "no compression recorded this session" in capsys.readouterr().out


def test_compression_view_paints_the_same_lines(monkeypatch):
    from nx import _render, _state as st
    monkeypatch.setattr(st, "_LAST_COMPRESSION", _LEDGER, raising=False)
    view = _ctxreport.compression_view(_CtxState(3))
    assert isinstance(view, _ctxreport.CompressionView)
    body = "\n".join(view.render(240, 40))
    assert _render.BOX_TL in body
    plain = __import__("re").sub(r"\x1b\[[0-9;]*m", "", body)
    for ln in _ctxreport._compression_report_lines():
        assert ln.strip() in plain
    # esc/q quit (read-only view: no third action to run)
    assert view.on_key("r", None) == "repaint"
    assert view.on_key("esc", None) == "quit"


def test_compression_view_is_registered_and_jumpable_from_context(monkeypatch):
    from nx import _fullscreen, _keys, _state as st
    monkeypatch.setattr(st, "_LAST_COMPRESSION", _LEDGER, raising=False)
    assert "compress" in _fullscreen.view_names()
    assert isinstance(_fullscreen.make_view("compress", _CtxState()),
                      _ctxreport.CompressionView)
    # every advertised key of the new surface is handled
    view = _ctxreport.compression_view()
    for key, _meaning in _keys.keys_for("compress"):
        assert view.on_key(key, None) in ("repaint", "quit", "run"), key
    # `c` in the /context surface jumps to it via the shell's view stack
    assert ("c", "last compression report") in _keys.keys_for("context")

    class _Shell:
        def __init__(self):
            self.pushed = []

        def push(self, v):
            self.pushed.append(v)

    ctx = _ctxreport.context_view(_CtxState(2))
    shell = _Shell()
    assert ctx.on_key("c", shell) == "repaint"
    assert isinstance(shell.pushed[0], _ctxreport.CompressionView)
