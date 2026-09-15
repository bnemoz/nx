"""2.2 + 2.4 (0.47.0): undo correctness/durability and compression v2.

2.2 — undo:
  - one approved tool call pushes exactly ONE checkpoint (the dispatcher's
    push_if_absent dedups against body-pushed delete_file checkpoints);
  - delete_file's "changed during approval" guard now compares against a
    PRE-approval mtime, so it can actually fire;
  - sed_edit pushes no checkpoint when its staleness abort fires;
  - the stack persists to <NX_HOME>/undo.json (0600) and survives a reload;
  - a restore that fails on every file keeps the entry pending.

2.4 — compression v2:
  - a summary turn is reused append-only: byte-identical prefix, one digest
    message, never re-summarized;
  - NX_COMPRESS_KEEP_TOKENS sizes the kept tail by tokens, not message count;
  - the fold's tool results are spilled to disk before summarizing;
  - NX_COMPRESS_SOURCE/MODEL route the fold call; the summary answer is
    bounded by max_tokens.
"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR"):
    os.environ.pop(k, None)

import pytest


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMsg(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = None

    def model_dump(self):
        return {"choices": [{"message": {"content": self.choices[0].message.content}}]}


class _FakeCompletions:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return _FakeResp("fake summary of the conversation")


@pytest.fixture()
def nx_env(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ.pop("NX_COMPRESS_KEEP_TOKENS", None)
    os.environ.pop("NX_COMPRESS_SOURCE", None)
    os.environ.pop("NX_COMPRESS_MODEL", None)
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    nx.YOLO = True
    nx._state._UNDO_STACK.clear()
    try:
        os.unlink(os.path.join(str(nx._state.SESSION_HOME), "undo.json"))
    except OSError:
        pass
    # Fake the active model client: compress()/undo tests must not touch network.
    class _C:
        base_url = "http://fake"

    nx._state.client = _C()
    nx._state.MODEL = "test-model"
    nx._state.ACTIVE = None
    yield nx, tmp_path
    nx.YOLO = False
    nx._state._UNDO_STACK.clear()
    nx._state._TOOL_RESULT_CACHE.clear()


# --- 2.2: one call, one checkpoint ---------------------------------------------

def test_delete_file_via_run_tool_pushes_one_checkpoint(nx_env):
    nx, tmp = nx_env
    p = tmp / "victim.txt"
    p.write_text("precious")
    out = nx.run_tool("delete_file", {"path": str(p)})
    assert out.startswith("deleted")
    from nx._undo import undo_last, list_checkpoints
    assert len(list_checkpoints()) == 1, list_checkpoints()
    _, msg = undo_last()
    assert "delete_file" in msg and p.exists()
    # the duplicate is gone: the next press has nothing to undo
    _, msg2 = undo_last()
    assert msg2 == "nothing to undo"


def test_repeated_writes_to_one_file_collapse_into_one_press(nx_env):
    nx, tmp = nx_env
    p = tmp / "w.txt"
    p.write_text("v1")
    nx.run_tool("write_file", {"path": str(p), "content": "v2"})
    nx.run_tool("write_file", {"path": str(p), "content": "v3"})
    from nx._undo import undo_last
    entry, msg = undo_last()
    # One press = one write undone: the checkpoint holds the file's content at
    # the START of the most recent write (v2), i.e. the state the user last saw
    # before the latest change. The older duplicate never stacked.
    assert p.read_text() == "v2", p.read_text()
    _, msg2 = undo_last()
    assert msg2 == "nothing to undo", msg2


def test_delete_file_guard_fires_when_changed_during_approval(nx_env, monkeypatch):
    nx, tmp = nx_env
    p = tmp / "g.txt"
    p.write_text("original")
    real_confirm = nx._confirm
    seen = {}

    def fake_confirm(action, preview=None):
        seen["called"] = True
        p.write_text("mutated during approval")
        return True

    monkeypatch.setattr(nx, "_confirm", fake_confirm)
    out = nx.delete_file(str(p))
    assert out.startswith("ERROR"), out
    assert "changed on disk" in out
    assert p.read_text() == "mutated during approval"  # nothing deleted


def test_sed_edit_stale_abort_pushes_no_checkpoint(nx_env, monkeypatch):
    nx, tmp = nx_env
    p = tmp / "s.txt"
    p.write_text("alpha\nbeta\n")
    real_stat = os.stat

    def fake_confirm(action, preview=None):
        # bump mtime past the pre-approval reading
        st = real_stat(p)
        os.utime(p, (st.st_atime, st.st_mtime + 10))
        return True

    monkeypatch.setattr(nx, "_confirm", fake_confirm)
    out = nx.sed_edit("beta", "BETA", str(p))
    assert out.startswith("ERROR") and "changed on disk" in out
    assert p.read_text() == "alpha\nbeta\n"  # not modified
    from nx._undo import list_checkpoints
    assert list_checkpoints() == [], list_checkpoints()


def test_sed_edit_success_still_undoable(nx_env, monkeypatch):
    nx, tmp = nx_env
    p = tmp / "s2.txt"
    p.write_text("alpha\nbeta\n")
    out = nx.sed_edit("beta", "BETA", str(p))
    assert out.startswith("edited")
    from nx._undo import undo_last
    _, msg = undo_last()
    assert "sed_edit" in msg and p.read_text() == "alpha\nbeta\n"


# --- 2.2: durability ------------------------------------------------------------

def test_undo_stack_persists_and_reloads(nx_env):
    nx, tmp = nx_env
    from nx import _undo
    p = tmp / "persist.txt"
    p.write_text("keep me")
    _undo.push_checkpoint("write_file", [_undo.snapshot(str(p))])
    # PF-05 (0.61.0): pushes are amortized, so the turn-boundary flush is what
    # writes undo.json — flush explicitly here, as the REPL does per turn.
    _undo.flush_undo_stack()
    path = _undo._undo_path()
    assert os.path.exists(path)
    assert (os.stat(path).st_mode & 0o777) == 0o600, oct(os.stat(path).st_mode)
    data = json.loads(open(path).read())
    assert data["version"] == 1 and data["entries"][0]["tool"] == "write_file"
    # a fresh process state reloads it
    nx._state._UNDO_STACK.clear()
    _undo._load_persisted()
    assert len(nx._state._UNDO_STACK) == 1
    entry, msg = _undo.undo_last()
    assert "write_file" in msg and p.read_text() == "keep me"
    # the pop persisted too
    nx._state._UNDO_STACK.clear()
    _undo._load_persisted()
    assert nx._state._UNDO_STACK == []


def test_clear_persisted_wipes_disk_state(nx_env):
    nx, tmp = nx_env
    from nx import _undo
    _undo.push_checkpoint("write_file", [{"path": "/tmp/x", "existed": False,
                                          "content": None}])
    _undo.flush_undo_stack()
    assert os.path.exists(_undo._undo_path())
    _undo.clear_persisted()
    assert not os.path.exists(_undo._undo_path())
    assert nx._state._UNDO_STACK == []


def test_corrupt_undo_file_starts_empty(nx_env):
    nx, tmp = nx_env
    from nx import _undo
    os.makedirs(nx._state.SESSION_HOME, exist_ok=True)
    with open(_undo._undo_path(), "w") as f:
        f.write("{not json")
    nx._state._UNDO_STACK.clear()
    _undo._load_persisted()  # must not raise
    assert nx._state._UNDO_STACK == []


def test_failed_restore_keeps_entry_pending(nx_env, monkeypatch):
    nx, tmp = nx_env
    from nx import _undo
    # A restore "failure" without real filesystem games: make the private
    # atomic write raise by pointing it at a directory path.
    doomed_dir = tmp / "not-a-file"
    doomed_dir.mkdir()
    _undo.push_checkpoint("write_file", [{"path": str(doomed_dir / "f.txt"),
                                          "existed": True, "content": "x"}])
    # Replace the parent dir with a FILE so open() inside it fails.
    import shutil
    shutil.rmtree(doomed_dir)
    doomed_dir.write_text("i am a file now")
    entry, msg = _undo.undo_last()
    assert msg.startswith("ERROR"), msg
    # the entry stays pending so the next press can retry
    assert len(nx._state._UNDO_STACK) == 1
    _, msg2 = _undo.undo_last()
    assert msg2.startswith("ERROR")


# --- 2.4: compression v2 --------------------------------------------------------

def _mk(nx):
    class _Chat:
        pass

    client = type("C", (), {})()
    client.base_url = "http://fake"
    client.chat = _Chat()
    client.chat.completions = _FakeCompletions()
    nx._state.client = client
    return client


def test_digest_is_reused_append_only(nx_env):
    nx, tmp = nx_env
    client = _mk(nx)
    from nx import _compress as cp
    msgs = [{"role": "system", "content": nx.SYSTEM}] + [
        {"role": "user" if j % 2 == 0 else "assistant", "content": f"turn {j // 2}"}
        for j in range(20)]
    r1 = cp.compress(msgs, auto=False)
    assert r1 is not None
    digest1 = msgs[1]["content"]
    n_summary_msgs = sum(1 for m in msgs if cp._is_summary_msg(m))
    assert n_summary_msgs == 1
    # second fold: everything between the digest and the new tail
    msgs += [{"role": "user" if j % 2 == 0 else "assistant", "content": f"more {j // 2}"}
             for j in range(20)]
    old_prefix = digest1[:200]
    r2 = cp.compress(msgs, auto=False)
    assert r2 is not None
    assert sum(1 for m in msgs if cp._is_summary_msg(m)) == 1, "still one digest"
    assert msgs[1]["content"].startswith(old_prefix), "prefix bytes must never change"
    assert len(msgs[1]["content"]) > len(digest1), "digest grew append-only"
    # the fresh digest was NOT built from the old digest's text
    prompt = client.chat.completions.requests[-1]["messages"][0]["content"]
    assert "turn 0" in prompt or "more 0" in prompt  # new material summarized
    assert "fake summary of the conversation" not in prompt, "old digest not re-summarized"


def test_token_target_grows_the_kept_tail(nx_env, monkeypatch):
    nx, tmp = nx_env
    _mk(nx)
    from nx import _compress as cp
    monkeypatch.setattr(nx._state, "COMPRESS_KEEP_TOKENS", 400)  # ~5 turns of 400-char bodies
    msgs = [{"role": "system", "content": nx.SYSTEM}] + [
        {"role": "user" if j % 2 == 0 else "assistant",
         "content": f"turn {j // 2} " + "w" * 380}
        for j in range(30)]
    r = cp.compress(msgs, auto=False)
    assert r is not None and r[0] >= 5, r
    # a body entirely inside the target → nothing to fold
    small = [{"role": "system", "content": nx.SYSTEM}] + [
        {"role": "user", "content": f"t{j}"} for j in range(6)]
    assert cp.compress(small, auto=False) is None


def test_window_derived_token_target_auto_only(nx_env, monkeypatch):
    nx, tmp = nx_env
    _mk(nx)
    from nx import _compress as cp

    class _Src:
        name = "fake"

        def resolve_context_window(self, model=None, force=False):
            return 12000

        def extra_request_kwargs(self):
            return {}

    # auto with a known window: target = 3000 tok = 12000 chars → the tiny
    # turns all fit → nothing to fold
    tiny = [{"role": "system", "content": nx.SYSTEM}] + [
        {"role": "user", "content": f"t{j}"} for j in range(10)]
    nx._state.ACTIVE = _Src()
    assert cp.compress(tiny, auto=True) is None
    # manual: count semantics unchanged (COMPRESS_KEEP=2)
    r = cp.compress(tiny, auto=False)
    assert r is not None and r[0] == nx.COMPRESS_KEEP, r
    nx._state.ACTIVE = None


def test_fold_spills_big_tool_results_before_summarizing(nx_env):
    nx, tmp = nx_env
    client = _mk(nx)
    from nx import _compress as cp
    big = "z" * 20000 + "\nEND-MARKER-should-not-appear"
    msgs = [{"role": "system", "content": nx.SYSTEM},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {
                    "name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": big},
            {"role": "user", "content": "go on"}]
    r = cp.compress(msgs, keep=1, auto=False)
    assert r is not None
    prompt = client.chat.completions.requests[-1]["messages"][0]["content"]
    # The full blob never reaches the fold prompt: the stub (echo + saved-to
    # note) replaced it, and _render's per-message 2000-char cap bounds the
    # echo. The END-MARKER lived at the far end of the blob — it must be gone.
    assert "END-MARKER" not in prompt
    assert prompt.count("z") < 6000, prompt.count("z")
    # the spill file exists and holds the full text
    odir = os.path.join(str(nx._state.SESSION_HOME), "overflow")
    assert os.path.isdir(odir) and os.listdir(odir)


def test_compress_call_is_bounded_and_routable(nx_env):
    nx, tmp = nx_env
    from nx import _compress as cp
    from nx._sources import Source
    client = _mk(nx)

    def _msgs():
        return [{"role": "system", "content": nx.SYSTEM}] + [
            {"role": "user" if j % 2 == 0 else "assistant", "content": f"turn {j // 2}"}
            for j in range(10)]

    # default: active client, bounded summary
    r = cp.compress(_msgs(), auto=False)
    assert r is not None
    kw = client.chat.completions.requests[-1]
    assert kw["max_tokens"] == nx.COMPRESS_SUMMARY_TOKENS, kw
    assert kw["model"] == "test-model"
    # routing: a configured compress source wins for the fold call
    cheap = type("Cheap", (), {})()
    cheap.chat = type("H", (), {})()
    cheap.chat.completions = _FakeCompletions()
    src = Source("cheap", "http://localhost:9/v1", "k", model="default-cheap")
    src.client = cheap
    nx._state.SOURCES["cheap"] = src
    nx._state.COMPRESS_SOURCE = "cheap"
    nx._state.COMPRESS_MODEL = "small-model"
    r2 = cp.compress(_msgs(), auto=False)
    assert r2 is not None
    kw2 = cheap.chat.completions.requests[-1]
    assert kw2["model"] == "small-model" and kw2["max_tokens"] == nx.COMPRESS_SUMMARY_TOKENS
    # the active client was not used for the routed fold
    assert len(client.chat.completions.requests) == 1
    # unknown source → fall back to the active client, warn once
    nx._state.SOURCES.pop("cheap")
    nx._state.COMPRESS_SOURCE = "nope"
    nx._state._COMPRESS_SOURCE_WARNED = False
    r3 = cp.compress(_msgs(), auto=False)
    assert r3 is not None
    assert len(client.chat.completions.requests) == 2


def test_compress_for_overrun_still_honors_explicit_keep(nx_env):
    nx, tmp = nx_env
    calls = []

    def fake_compress(msgs, keep=None, auto=False):
        calls.append((keep, auto))
        return (keep or 2, 5, 10)

    nx.compress = fake_compress
    msgs = [{"role": "system", "content": "S"}] + [
        {"role": "user", "content": f"u{i}"} for i in range(20)]
    from nx import _compress as cp
    assert cp._compress_for_overrun(msgs, tier=1) is True
    assert calls[-1][0] is not None and calls[-1][1] is True
