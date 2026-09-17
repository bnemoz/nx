#!/usr/bin/env python3
"""Prefix-cache integrity (PERF.md 0.58.0, PF-12/13/14/15; invariants I-3/I-4).

The prompt prefix is what a vLLM/SGLang/OpenAI-style server keys its KV cache
on. nx displays `cached:` and documents prefix discipline, but four paths used
to mutate bytes *before* the turn boundary — which busts the cache for the
entire remaining prompt, every turn, and shows up only as a mysterious
`cached: 0%`:

- **PF-12**: the memory note was rewritten in place at index 1 on every
  /memory optimize and every resume over a changed store.
- **PF-14**: a context overrun spilled tool results by rewriting their
  (already-sent) mid-transcript bytes.
- **PF-15**: a fresh session and a resumed one assembled the head (SYSTEM +
  AGENTS.md / memory / orientation / delegation notes) their own way, so the
  two disagreed on the leading bytes.

The fixes make the head one producer (`_prefix.prefix_notes`) with one pinned
copy per session (`pin_prefix`/`check_prefix`), and move every refresh to the
**tail**. These tests are the T-6/T-7 bench harness entries, static enough to
run in the normal suite:

- T-6: no message older than the turn boundary is mutated.
- T-7: SYSTEM + the prefix block are byte-identical over many turns.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for k in list(os.environ):
    if k.startswith("NX_"):
        del os.environ[k]
os.environ["NX_ENV_FILE"] = "/dev/null"

import nx  # noqa: E402,F401
import nx._state as st  # noqa: E402
from nx import _memory_cmd, _prefix  # noqa: E402
from nx._memory_cmd import _refresh_memory_note  # noqa: E402
from nx._memory_optimize import _drop_memory_caches  # noqa: E402


def _write_memory(title, body, scope="project", project_root=None):
    """memory_write with a controlled PROJECT_ROOT (it reads st at call time)
    and the approval gate stubbed (the real one prompts on stdin)."""
    old_root, old_confirm = st.PROJECT_ROOT, nx._confirm
    st.PROJECT_ROOT = project_root
    nx._confirm = lambda action, preview=None: True
    try:
        out = _memory_cmd.memory_write(title=title, body=body, scope=scope)
        _drop_memory_caches()
        return out
    finally:
        st.PROJECT_ROOT = old_root
        nx._confirm = old_confirm


# --- T-7 / I-4: the head is one stable block ---------------------------------
def test_prefix_block_is_the_single_producer_for_fresh_and_resume():
    """Both entry points use _prefix.prefix_notes(), so they cannot disagree on
    the leading bytes. This is the structural guard PF-15 asks for: re-inlining
    the block in either place fails even while the happy path still works."""
    import inspect
    from nx import _dispatch, _resume
    fresh = inspect.getsource(_dispatch._initial_messages)
    resume = inspect.getsource(_resume.resumed_messages)
    assert "prefix_notes()" in fresh, "fresh session no longer uses the shared block"
    assert "prefix_notes()" in resume, "resume no longer uses the shared block"
    for src, label in ((fresh, "fresh"), (resume, "resume")):
        assert "_load_agents_md_note(" not in src, f"{label} re-inlines the AGENTS.md note"
        assert "_load_orientation_note(" not in src, f"{label} re-inlines orientation"


def test_prefix_notes_are_ordered_and_canonical(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "AGENTS.md").write_text("Project rule: be terse.")
    monkeypatch.setattr(st, "PROJECT_ROOT", str(root))
    monkeypatch.setattr(st, "SESSION_HOME", str(tmp_path))
    monkeypatch.setattr(nx, "NX_SUBAGENTS", 1)
    _write_memory("A note", "body", scope="project", project_root=str(root))

    notes = _prefix.prefix_notes()
    heads = [n["content"].split("\n", 1)[0] for n in notes]
    assert heads[0].startswith("[Project instructions from AGENTS.md")
    assert heads[1].startswith("[Saved memories")
    assert heads[2].startswith("[Orientation note:")
    assert heads[3].startswith("[Delegation note:")
    # every one is recognised by the marker scan the pin relies on
    assert all(_prefix.is_prefix_note(n) for n in notes)


def test_system_and_prefix_block_byte_identical_over_20_turns(tmp_path, monkeypatch):
    """T-7: the head never moves across a session, even as memory drifts and
    turns are appended."""
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setattr(st, "PROJECT_ROOT", str(root))
    monkeypatch.setattr(st, "SESSION_HOME", str(tmp_path))
    _write_memory("Seed", "b", scope="project", project_root=str(root))

    msgs = nx._initial_messages()
    head = _prefix.pinned_prefix()
    assert head, "a fresh session must pin its head"
    assert len(msgs) == len(head)
    for turn in range(20):
        # a real turn appends at the tail…
        msgs.append({"role": "user", "content": f"turn {turn}"})
        msgs.append({"role": "assistant", "content": f"answer {turn}"})
        # …and a store change refreshes the note append-only
        if turn == 7:
            _write_memory(f"Late {turn}", "b", scope="project", project_root=str(root))
            assert _refresh_memory_note(msgs) is True
        assert _prefix.check_prefix(msgs, warn=False) is True, f"head moved at turn {turn}"
        assert _prefix.pinned_prefix() == head


# --- T-6 / I-3: nothing older than the turn boundary mutates -----------------
def test_memory_refresh_never_touches_older_messages(tmp_path, monkeypatch):
    """PF-12 regression: the refresh used to write `messages[1] = note`."""
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setattr(st, "PROJECT_ROOT", str(root))
    monkeypatch.setattr(st, "SESSION_HOME", str(tmp_path))
    _write_memory("One", "b", scope="project", project_root=str(root))

    msgs = nx._initial_messages()
    msgs += [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    _refresh_memory_note(msgs)  # no-op (already fresh)
    frozen = [dict(m) for m in msgs]

    # the store moves on
    _write_memory("Two", "b", scope="project", project_root=str(root))
    assert _refresh_memory_note(msgs) is True
    # every pre-existing message is byte-identical; the change is an append
    assert len(msgs) == len(frozen) + 1
    for before, after in zip(frozen, msgs):
        assert before == after, "refresh rewrote an existing message"
    assert "[project] Two" in msgs[-1]["content"]


def test_memory_note_is_never_written_at_index_one(tmp_path, monkeypatch):
    """The exact PF-12 shape: index 1 is the worst place to rewrite, since it
    busts the prefix for the whole prompt. The note must land at the tail."""
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setattr(st, "PROJECT_ROOT", str(root))
    monkeypatch.setattr(st, "SESSION_HOME", str(tmp_path))
    _write_memory("Solo", "b", scope="project", project_root=str(root))
    msgs = [{"role": "system", "content": "S"},
            {"role": "user", "content": "the real first prompt"},
            {"role": "assistant", "content": "a reply"}]
    _refresh_memory_note(msgs)
    assert msgs[1] == {"role": "user", "content": "the real first prompt"}
    assert "[Saved memories" in msgs[-1]["content"]



# --- F-1: a fold never swallows the prefix block -----------------------------
class _Msg:
    content = "digest text"


class _Choice:
    message = _Msg()


class _Resp:
    choices = [_Choice()]
    usage = None


def _stub_summarizer(monkeypatch):
    """compress() hits the network; replace the chat call so the fold runs."""
    calls = []
    def fake_create(**kwargs):
        calls.append(kwargs)
        return _Resp()
    monkeypatch.setattr(st.client.chat.completions, "create", fake_create)
    return calls


def test_compress_preserves_the_prefix_block(tmp_path, monkeypatch):
    """F-1 regression: compress() used to compute body = messages[1:], so the
    foldable head swallowed the pinned prefix notes and the rebuild dropped
    them — busting the prefix cache AND silently erasing the project rules,
    cwd, and memory index from context. The block must survive verbatim."""
    from nx import _compress
    import copy
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setattr(st, "PROJECT_ROOT", str(root))
    monkeypatch.setattr(st, "SESSION_HOME", str(tmp_path))
    monkeypatch.setattr(nx, "NX_SUBAGENTS", 1)
    _write_memory("Seed", "b", scope="project", project_root=str(root))
    _stub_summarizer(monkeypatch)

    msgs = nx._initial_messages()
    head_len = _prefix._head_len(msgs)
    assert head_len > 1, "test needs a multi-message pinned head"
    for i in range(10):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    head_before = copy.deepcopy(msgs[:head_len])

    result = _compress.compress(msgs, keep=4)
    assert result is not None
    # the block survives, in place, byte-identical
    assert msgs[:head_len] == head_before, "fold mutated or dropped the prefix block"
    assert _prefix.check_prefix(msgs, warn=False) is True, "head moved across the fold"
    # rebuild is well-formed: system + block + summary + tail
    assert _compress._is_summary_msg(msgs[head_len])


def test_compress_pins_the_block_even_when_unpinned(monkeypatch):
    """A transcript assembled without pin_prefix (e.g. an old session) still
    keeps its leading prefix notes across a fold — is_prefix_note is the
    authority, not the pin."""
    from nx import _compress
    _stub_summarizer(monkeypatch)
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "[Project instructions from AGENTS.md:]\nrule"},
        {"role": "user", "content": "[Orientation note:]\nWorking directory: /tmp"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "q4"},
        {"role": "assistant", "content": "a4"},
    ]
    result = _compress.compress(msgs, keep=2)
    assert result is not None
    assert _prefix.is_prefix_note(msgs[1]) and _prefix.is_prefix_note(msgs[2])
    assert "rule" in msgs[1]["content"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
