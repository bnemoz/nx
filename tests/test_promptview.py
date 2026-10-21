"""Phase 5 (annotated system-prompt floor, /prompt) of the explainable harness:
the _promptview report over the prompt constants + the active tool schemas,
the PromptView surface, and the /prompt command wiring — same fixture patterns
as test_turnlog.py."""

import nx
from nx import _promptview


def _find(lines, needle):
    return [ln for ln in lines if needle in ln]


def test_report_lists_the_main_system_prompt_with_an_estimate():
    lines = _promptview._prompt_report_lines()
    rows = _find(lines, "SYSTEM  nx/__init__.py")
    assert rows, lines
    assert "every turn" in rows[0]
    # the real SYSTEM prompt is a few thousand chars → a real (non-zero) estimate
    est = rows[0].split("~")[-1].split()[0]
    assert est.isdigit() and int(est) > 0


def test_report_lists_satellites_with_estimates_and_cadence():
    lines = _promptview._prompt_report_lines()
    for attr in ("RISK_SYSTEM", "REDUCER_SYSTEM_PROMPT", "DESC_SYSTEM",
                 "SUBAGENT_SYSTEM", "SUBAGENT_WRITE_SYSTEM",
                 "SUBAGENT_IMPLEMENT_SYSTEM"):
        rows = _find(lines, attr)
        assert rows, f"{attr} missing from the prompt report"
        assert "tok" in rows[0]
    # every satellite is marked as NOT per-turn (the report's teaching point)
    assert _find(lines, "NOT on the per-turn floor")
    assert _find(lines, "satellite prompt(s)")


def test_report_shows_the_tool_schema_total():
    from nx._ctxreport import _turn_overhead_est
    lines = _promptview._prompt_report_lines()
    rows = _find(lines, "TOOLS")
    assert rows, lines
    assert f"~{_turn_overhead_est()} tok" in rows[0]
    # the footer total names both halves of the floor
    foot = _find(lines, "total per-turn floor")
    assert foot and "SYSTEM" in foot[0] and "tool schemas" in foot[0]


def test_est_never_raises_on_garbage():
    assert _promptview._est(None) == 0
    assert _promptview._est(123) == 0
    assert _promptview._est("") == 0
    assert _promptview._est("a" * 3500) > 0


def test_missing_system_prompt_renders_a_placeholder(monkeypatch):
    monkeypatch.setattr(nx, "SYSTEM", None, raising=False)
    lines = _promptview._prompt_report_lines()
    assert _find(lines, "SYSTEM  nx/__init__.py"), lines
    assert _find(lines, "(prompt unavailable)")
    # still renders the rest of the floor
    assert _find(lines, "total per-turn floor")


def test_missing_satellite_constants_are_skipped_not_raised(monkeypatch):
    monkeypatch.delattr("nx._prompts.RISK_SYSTEM", raising=False)
    monkeypatch.setattr("nx._prompts.DESC_SYSTEM", 42, raising=False)
    lines = _promptview._prompt_report_lines()
    assert not _find(lines, "RISK_SYSTEM")
    assert not _find(lines, "DESC_SYSTEM")
    assert _find(lines, "SUBAGENT_SYSTEM")   # the rest survive


def test_no_satellites_at_all_renders_a_placeholder(monkeypatch):
    monkeypatch.setattr(_promptview, "_SATELLITES", [])
    lines = _promptview._prompt_report_lines()
    assert _find(lines, "— no satellite prompts importable")


def test_no_active_tools_still_renders(monkeypatch):
    def _boom():
        raise RuntimeError("no active source")
    monkeypatch.setattr("nx._dispatch._active_tools", _boom, raising=False)
    lines = _promptview._prompt_report_lines()
    rows = _find(lines, "TOOLS")
    assert rows and "no active tool registry" in rows[0]
    # schema total drops out of the footer; the SYSTEM half survives
    foot = _find(lines, "total per-turn floor")
    assert foot and "tool schemas ~0" in foot[0]
    assert foot[0] != lines[0]


def test_prompt_view_paints_the_same_lines_and_is_registered(monkeypatch):
    from nx import _fullscreen as fs
    # The active tool registry is live-gated state other tests mutate, so pin
    # it: then the view's body and the inline report must be identical lists.
    tools = [{"type": "function",
              "function": {"name": "read_file", "description": "read"}}]
    monkeypatch.setattr("nx._dispatch._active_tools", lambda: tools)
    view = _promptview.prompt_view()
    inline = _promptview._prompt_report_lines()
    assert view.lines == inline
    painted = "\n".join(view.render(240, 60))
    for ln in inline:
        if not ln.strip():
            continue
        # panel_row clips to the real terminal width (the TurnsView rule), so
        # compare the head of each line against the painted frame.
        head = ln[:40]
        assert head in painted, ln
    assert "total per-turn floor" in painted
    assert fs.make_view("prompt") is not None
    assert "prompt" in fs.view_names()
    assert view.on_key("r", None) == "repaint"
    assert view.on_key("esc", None) == "quit"
    assert view.on_key("q", None) == "quit"


def test_context_view_S_jumps_to_prompt_view():
    from nx import _ctxreport
    pushed = []

    class _Shell:
        def push(self, v):
            pushed.append(v)

    class _State:
        messages = [{"role": "user", "content": "hi"}]

    view = _ctxreport.context_view(_State())
    assert view.on_key("S", _Shell()) == "repaint"
    assert len(pushed) == 1 and type(pushed[0]).__name__ == "PromptView"
    assert view.on_key("S", None) == "repaint"     # shell=None degrades


def test_cmd_prompt_inline_prints_the_report(monkeypatch, capsys):
    monkeypatch.setattr("nx._fullscreen.wants_fullscreen", lambda: False)
    _promptview._cmd_prompt("")
    out = capsys.readouterr().out
    assert "SYSTEM  nx/__init__.py" in out
    assert "total per-turn floor" in out
