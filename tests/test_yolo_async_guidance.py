"""0.110.2: yolo-aware write-tier guidance. The session-start delegation note
told every session that background write agents "work only in yolo mode" —
pessimistic for a session that IS in yolo, so models serialized write work
with explicit async:false. The note's background-mode sentence is now chosen
by the live yolo posture at note build time, and a one-shot runtime nudge
re-teaches the contract when a yolo session passes explicit async:false on a
write-tier spawn anyway (covers the mid-session /yolo toggle, whose note is
stale)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nx  # noqa: E402
import nx._autowake as aw  # noqa: E402
import nx._delegation_note as d  # noqa: E402
import nx._notices  # noqa: E402
import nx._runtime_notes as rn  # noqa: E402
import nx._state as st  # noqa: E402
import nx._subagent as sa  # noqa: E402


def _fresh_notice_state(monkeypatch):
    """Fire-once flag + slot reset between cases (conftest rewinds module
    state, but a same-file sequence needs explicit resets)."""
    monkeypatch.setattr(st, "_YOLO_ASYNC_NUDGED", False, raising=False)
    nx._notices.set_("_PENDING_YOLO_ASYNC_NOTICE", None)


# --- 1. yolo note -------------------------------------------------------------

def test_yolo_note_says_background_permitted(monkeypatch):
    monkeypatch.setattr(nx, "NX_SUBAGENTS", 1)
    monkeypatch.setattr(nx, "NX_SUBAGENT_WRITE", 1)
    monkeypatch.setattr(st, "YOLO", True)
    content = d._load_delegation_note()["content"]
    assert "runs in yolo mode" in content
    assert "ARE permitted" in content
    assert "work only in yolo mode" not in content
    # scope + worktree bullets survive, tail byte-identical
    assert "SHOULD carry scope" in content and "worktree" in content
    assert content.endswith("review/merge of what agents produced.]")
    assert d.DELEGATION_NOTE_MARKER in content


# --- 2. non-yolo note unchanged -----------------------------------------------

def test_non_yolo_note_unchanged(monkeypatch):
    monkeypatch.setattr(nx, "NX_SUBAGENTS", 1)
    monkeypatch.setattr(nx, "NX_SUBAGENT_WRITE", 1)
    monkeypatch.setattr(st, "YOLO", False)
    content = d._load_delegation_note()["content"]
    assert "work only in yolo mode" in content
    assert "async=false so approval prompts can show" in content
    assert content.endswith("review/merge of what agents produced.]")
    assert d.DELEGATION_NOTE_MARKER in content


# --- 3. nudge fires once --------------------------------------------------------

def test_nudge_fires_once_on_explicit_foreground(monkeypatch):
    _fresh_notice_state(monkeypatch)
    monkeypatch.setattr(st, "YOLO", True)
    sa._maybe_nudge_yolo_async("write", False)
    assert st._YOLO_ASYNC_NUDGED is True
    text = nx._notices.get("_PENDING_YOLO_ASYNC_NOTICE")
    assert text and "yolo mode" in text and "async:false" in text
    # second call stages nothing new (fire-once)
    nx._notices.set_("_PENDING_YOLO_ASYNC_NOTICE", None)
    sa._maybe_nudge_yolo_async("write", "false")
    assert nx._notices.get("_PENDING_YOLO_ASYNC_NOTICE") is None


# --- 4. nudge does not fire ------------------------------------------------------

def test_nudge_gated_on_posture_tier_and_omission(monkeypatch):
    # not yolo: silent downgrade is correct, no nudge
    _fresh_notice_state(monkeypatch)
    monkeypatch.setattr(st, "YOLO", False)
    sa._maybe_nudge_yolo_async("write", False)
    assert nx._notices.get("_PENDING_YOLO_ASYNC_NOTICE") is None
    assert getattr(st, "_YOLO_ASYNC_NUDGED", False) is False
    # read tier: never
    _fresh_notice_state(monkeypatch)
    monkeypatch.setattr(st, "YOLO", True)
    sa._maybe_nudge_yolo_async("read", False)
    assert nx._notices.get("_PENDING_YOLO_ASYNC_NOTICE") is None
    # async omitted (None): the model didn't choose foreground — no nag
    _fresh_notice_state(monkeypatch)
    sa._maybe_nudge_yolo_async("write", None)
    assert nx._notices.get("_PENDING_YOLO_ASYNC_NOTICE") is None
    # explicit async:true: background already, no nudge
    sa._maybe_nudge_yolo_async("write", True)
    assert nx._notices.get("_PENDING_YOLO_ASYNC_NOTICE") is None


# --- 5. call-site wiring (regression: omitted async + NX_SUBAGENT_ASYNC=0) ------

def test_spawn_agent_omitted_async_never_nudges_even_with_fg_default(monkeypatch):
    """The nudge must fire only on the model's EXPLICIT async:false. When the
    model omits async and the user runs NX_SUBAGENT_ASYNC=0 (their own
    foreground default), the first buggy wiring passed the defaulted False to
    the helper and nagged on every spawn. spawn_agent must reach the nudge
    with None and stay quiet; it then fails fast on no model source."""
    _fresh_notice_state(monkeypatch)
    monkeypatch.setattr(nx, "NX_SUBAGENTS", 1)
    monkeypatch.setattr(nx, "NX_SUBAGENT_WRITE", 1)
    monkeypatch.setattr(nx, "NX_SUBAGENT_ASYNC", 0)
    monkeypatch.setattr(st, "YOLO", True)
    monkeypatch.setattr(st, "client", None, raising=False)
    out = sa.spawn_agent("test", tier="write")
    assert "no active model source" in out  # reached the sync path, failed fast
    assert nx._notices.get("_PENDING_YOLO_ASYNC_NOTICE") is None
    assert getattr(st, "_YOLO_ASYNC_NUDGED", False) is False


def test_spawn_agent_explicit_false_nudges_via_call_site(monkeypatch):
    _fresh_notice_state(monkeypatch)
    monkeypatch.setattr(nx, "NX_SUBAGENTS", 1)
    monkeypatch.setattr(nx, "NX_SUBAGENT_WRITE", 1)
    monkeypatch.setattr(st, "YOLO", True)
    monkeypatch.setattr(st, "client", None, raising=False)
    out = sa.spawn_agent("test", tier="write", **{"async": False})
    assert "no active model source" in out
    assert nx._notices.get("_PENDING_YOLO_ASYNC_NOTICE") is not None


# --- 6. flush + pending plumbing --------------------------------------------------

def test_flush_and_pending_notice_plumbing(monkeypatch):
    _fresh_notice_state(monkeypatch)
    monkeypatch.setattr(st, "YOLO", True)
    sa._maybe_nudge_yolo_async("implement", "0")
    staged = nx._notices.get("_PENDING_YOLO_ASYNC_NOTICE")
    assert staged
    assert aw._any_pending_notice() is True
    messages: list = []
    rn._flush_pending_background_notice(messages)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == staged
    assert nx._notices.get("_PENDING_YOLO_ASYNC_NOTICE") is None
    assert aw._any_pending_notice() is False
