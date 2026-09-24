#!/usr/bin/env python3
"""Regression tests for the four SoL-Pi mechanism ports (nx 0.67.0):
  - action fusion (_fusion.py): then_run on edit tools, markers, skip paths
  - observation pack (_obs.py): pack after FULL_SENDS, obs_recall paging
  - evidence-preserving reducer (_reducer.py): receipt validation, fail-open
  - online context compact (_occompact.py): boundary detection + economics
All offline: no model calls (the reducer's model call is stubbed).
"""
import json
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-sol-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _fusion, _obs, _reducer, _occompact  # noqa: E402


# --- action fusion -----------------------------------------------------------

def test_fusion_default_on_and_zero_flag_disables():
    # 0.67.1: fusion is ON by default; NX_FUSION=0 restores the separate
    # round-trip. Simulate the env-off path via the module flag.
    saved = _fusion.FUSION_ENABLED
    _fusion.FUSION_ENABLED = 0
    try:
        assert _fusion.fusion_enabled() is False
        out = m.run_tool("write_file", {"path": os.path.join(_tmp, "a.txt"),
                                        "content": "hi", "then_run": {"command": "echo x"}})
        assert "[then_run" not in out
    finally:
        _fusion.FUSION_ENABLED = saved
    # the import-time default reads NX_FUSION (default 1)
    from nx._env import _env_int
    assert _env_int("NX_FUSION", 1) == 1


def test_fusion_success_marker(monkeypatch):
    _fusion.FUSION_ENABLED = 1
    try:
        import nx._bash_jobs as bj
        monkeypatch.setattr(bj, "run_bash",
                            lambda command, timeout=None: "fused-ok\n")
        path = os.path.join(_tmp, "b.txt")
        out = _fusion.run_fused(
            {"path": path, "then_run": {"command": "echo fused-ok"}},
            mutate=lambda: "wrote " + path)
        assert "[then_run:succeeded]" in out
        assert "fused-ok" in out
        assert "wrote " + path in out
    finally:
        _fusion.FUSION_ENABLED = 0


def test_fusion_skipped_when_mutation_fails():
    _fusion.FUSION_ENABLED = 1
    try:
        out = _fusion.run_fused(
            {"path": os.path.join(_tmp, "c.txt"), "then_run": {"command": "echo x"}},
            mutate=lambda: "ERROR: bad edit")
        assert out.startswith("ERROR")
        assert "[then_run:skipped]" in out
    finally:
        _fusion.FUSION_ENABLED = 0


def test_fusion_failed_marker(monkeypatch):
    _fusion.FUSION_ENABLED = 1
    try:
        import nx._bash_jobs as bj
        monkeypatch.setattr(bj, "run_bash", lambda command, timeout=None: "ERROR: boom")
        out = _fusion.run_fused(
            {"path": os.path.join(_tmp, "d.txt"), "then_run": {"command": "false"}},
            mutate=lambda: "ok")
        assert "[then_run:failed]" in out
    finally:
        _fusion.FUSION_ENABLED = 0


def test_fusion_hash_guard_skips_on_external_write():
    _fusion.FUSION_ENABLED = 1
    try:
        path = os.path.join(_tmp, "e.txt")
        calls = {"n": 0}
        real_hash = _fusion._hash_file

        def racing_hash(p):
            calls["n"] += 1
            h = real_hash(p)
            if calls["n"] == 1:
                with open(p, "a") as f:
                    f.write("external write")
            return h

        _fusion._hash_file = racing_hash
        try:
            with open(path, "w") as f:
                f.write("base")
            out = _fusion.run_fused(
                {"path": path, "then_run": {"command": "echo x"}},
                mutate=lambda: "edited")
            assert "[then_run:skipped]" in out
        finally:
            _fusion._hash_file = real_hash
    finally:
        _fusion.FUSION_ENABLED = 0


def test_fusion_per_file_queue_serializes():
    import threading
    lk1 = _fusion.with_fused_file_queue(os.path.join(_tmp, "f.txt"))
    lk2 = _fusion.with_fused_file_queue(os.path.join(_tmp, "f.txt"))
    assert lk1 is lk2
    got = threading.Lock()
    assert got.acquire(blocking=False)
    lk1.acquire()
    acquired = []
    t = threading.Thread(target=lambda: (lk2.acquire(), acquired.append(1), lk2.release()))
    t.start()
    import time as _t
    _t.sleep(0.05)
    assert acquired == [], "second fused call on the same file must wait"
    lk1.release()
    t.join(timeout=2)
    assert acquired == [1]


# --- observation pack --------------------------------------------------------

def _big(n_bytes=20000):
    return ("x" * 79 + "\n") * (n_bytes // 80)


import contextlib


@contextlib.contextmanager
def _enabled():
    """NX_OBS=1 for the block (env is read at call time since P10)."""
    old = os.environ.get("NX_OBS")
    os.environ["NX_OBS"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("NX_OBS", None)
        else:
            os.environ["NX_OBS"] = old


def test_obs_disabled_by_default(monkeypatch):
    _obs._SEND_COUNTS.clear()
    text = _big()
    assert _obs.pack_result("run_bash", text, "k1") == text


def test_obs_full_sends_then_placeholder(monkeypatch):
    monkeypatch.setenv("NX_OBS", "1")
    _obs._SEND_COUNTS.clear()
    try:
        text = _big()
        first = _obs.pack_result("run_bash", text, "k2")
        second = _obs.pack_result("run_bash", text, "k2")
        third = _obs.pack_result("run_bash", text, "k2")
        assert first is text or first == text
        assert second == text
        assert third.startswith("[large tool result replaced")
        assert "id: obs_" in third
        assert "retrieve: call obs_recall" in third
        obs_id = third.split("id: obs_")[1].split()[0]
        obs_id = "obs_" + obs_id
        # archived text round-trips verbatim via obs_recall
        page = m.obs_recall(obs_id, 0)
        lines = page.split("\n", 2)
        header = lines[0] + "\n" + lines[1]
        body = lines[2]
        assert "next_offset=" in header
        assert body.startswith(text[:10])
        assert len(body.encode()) <= _obs.RECALL_MAX_BYTES
    finally:
        _obs._SEND_COUNTS.clear()


def test_obs_receipt_line_never_packed():
    try:
        text = _big() + "\nsol_pi_evidence_receipt_v1\n"
        with _enabled():
            out = _obs.pack_result("run_bash", text, "k3")
        assert out is text or out == text
    finally:
        _obs._SEND_COUNTS.clear()


def test_obs_small_result_not_packed():
    with _enabled():
        assert _obs.pack_result("run_bash", "tiny", "k4") == "tiny"


def test_obs_recall_unknown_id():
    out = m.obs_recall("obs_" + "0" * 24)
    assert out.startswith("ERROR: unknown observation id")


# --- evidence-preserving reducer ---------------------------------------------

CMD = "cargo test"
BODY = ("Compiling project v0.1.0\n"
        "error[E0308]: mismatched types\n"
        "  --> src/main.rs:12:5\n"
        "test result: FAILED. 1 passed; 2 failed\n"
        + ("note: some compilation context line\n" * 300))


def _fake_model(receipt):
    def fake_call(body, sha, nbytes, nlines, is_error):
        return json.dumps(receipt), "test-src", "test-model", 42
    return fake_call


def test_reducer_disabled_by_default():
    assert _reducer.reduce_tool_result("run_bash", CMD, BODY) is None


def test_reducer_accepts_verifiable_receipt(monkeypatch):
    _reducer.REDUCER_ENABLED = 1
    try:
        receipt = {"schema": _reducer.RECEIPT_SCHEMA, "source_sha256": None,
                   "status": "success", "uncertain": False,
                   "evidence": [{"kind": "warning",
                                 "quote": "test result: FAILED. 1 passed; 2 failed"}]}
        monkeypatch.setattr(_reducer, "_call_reducer_model",
                            lambda *a, **k: (json.dumps({**receipt, "source_sha256": _reducer._sha256(BODY)}),
                                             "s", "mm", 42))
        out = _reducer.reduce_tool_result("run_bash", CMD, BODY)
        assert out is not None
        assert out.startswith("sol_pi_evidence_receipt_v1")
        assert "status=success" in out
        assert "quote=\"test result: FAILED" in out
    finally:
        _reducer.REDUCER_ENABLED = 0


def test_reducer_rejects_fabricated_quote(monkeypatch):
    _reducer.REDUCER_ENABLED = 1
    try:
        monkeypatch.setattr(_reducer, "_call_reducer_model",
                            lambda *a, **k: (json.dumps({
                                "schema": _reducer.RECEIPT_SCHEMA,
                                "source_sha256": _reducer._sha256(BODY),
                                "status": "failure", "uncertain": False,
                                "evidence": [{"kind": "failure",
                                              "quote": "this line is nowhere in the log"}]}),
                                             "s", "mm", 42))
        assert _reducer.reduce_tool_result("run_bash", CMD, BODY) is None
    finally:
        _reducer.REDUCER_ENABLED = 0


def test_reducer_requires_failure_evidence(monkeypatch):
    _reducer.REDUCER_ENABLED = 1
    try:
        monkeypatch.setattr(_reducer, "_call_reducer_model",
                            lambda *a, **k: (json.dumps({
                                "schema": _reducer.RECEIPT_SCHEMA,
                                "source_sha256": _reducer._sha256(BODY),
                                "status": "failure", "uncertain": False,
                                "evidence": [{"kind": "summary", "quote": "Compiling project v0.1.0"}]}),
                                             "s", "mm", 42))
        assert _reducer.reduce_tool_result("run_bash", CMD, BODY) is None
    finally:
        _reducer.REDUCER_ENABLED = 0


def test_reducer_skips_small_and_non_diagnostic():
    _reducer.REDUCER_ENABLED = 1
    try:
        assert _reducer.reduce_tool_result("run_bash", "ls -la", "x" * 9000) is None
        assert _reducer.reduce_tool_result("run_bash", CMD, "tiny") is None
    finally:
        _reducer.REDUCER_ENABLED = 0


def test_reducer_secret_guard():
    # secret-shaped fixture built at runtime so no literal token lands in
    # this file (the commit guard refuses secret-shaped strings).
    shaped = "api_key = " + "s" + "k-" + "x" * 24
    assert _reducer.LIKELY_SECRET.search(shaped)
    body = (shaped + "\n") * 400
    _reducer.REDUCER_ENABLED = 1
    try:
        assert _reducer.reduce_tool_result("run_bash", CMD, body) is None
    finally:
        _reducer.REDUCER_ENABLED = 0


# --- online context compact --------------------------------------------------

def test_occ_disabled_by_default(monkeypatch):
    m.compress = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fold"))
    try:
        st_reset()
        assert _occompact.maybe_compact_after_boundary([]) is False
    finally:
        del m.compress


def st_reset():
    for k in _occompact._DEFAULTS:
        setattr(__import__("nx")._state, k, _occompact._DEFAULTS[k])


def test_occ_boundary_detection():
    st_reset()
    assert _occompact.inspect_todo_write(
        [{"text": "a", "status": "done"}, {"text": "b", "status": "pending"}]) == (1, 1)
    # second report of the same completed set is not a new boundary
    assert _occompact.inspect_todo_write(
        [{"text": "a", "status": "done"}, {"text": "b", "status": "pending"}]) is None
    assert _occompact.inspect_todo_write(
        [{"text": "a", "status": "done"}, {"text": "b", "status": "done"}]) == (1, 0)


def test_occ_non_positive_saving_defers():
    st_reset()
    # tiny context: archiveTokens <= memoTokens -> non_positive_saving
    d = _occompact.decide_compact(write_tokens=500, prior_compactions=0,
                                  carried_debt_tokens=0)
    assert d["compact"] is False and d["reason"] == "non_positive_saving"


def test_occ_window_protection_compacts():
    st_reset()
    import nx._state as st
    class _Active:
        def resolve_context_window(self):
            return 100000
    st.ACTIVE = _Active()
    _occompact._set("_OCC_LAST_CONTEXT_TOKENS", 100000)  # at window
    _occompact._set("_OCC_REMAINING_BOUNDARIES", 2)
    _occompact._set("_OCC_COMPLETED_BOUNDARY_INTERVALS", [3, 3, 3])
    d = _occompact.decide_compact(write_tokens=90000, prior_compactions=0,
                                  carried_debt_tokens=0)
    assert d["compact"] is True
    assert d["reason"] == "window_protection" or d["reason"] == "economic"


def test_occ_full_fold_sets_reminder(monkeypatch):
    st_reset()
    _occompact.OCC_ENABLED = 1
    try:
        import nx._state as st
        _occompact._set("_OCC_PENDING_BOUNDARY", True)
        _occompact._set("_OCC_LAST_CONTEXT_TOKENS", 100000)
        class _Active:
            def resolve_context_window(self):
                return 100000
        st.ACTIVE = _Active()
        monkeypatch.setattr(m, "compress",
                            lambda messages, auto=False: (2, 5, 300))
        assert _occompact.maybe_compact_after_boundary([]) is True
        note = st._PENDING_OCC_REMINDER
        assert note and note.startswith("[Runtime note: ")
        assert "todo_write" in note
        assert st._OCC_NATIVE_COMPACTIONS == 1
    finally:
        _occompact.OCC_ENABLED = 0
        st_reset()


def test_occ_never_folds_without_boundary(monkeypatch):
    st_reset()
    _occompact.OCC_ENABLED = 1
    try:
        monkeypatch.setattr(m, "compress",
                            lambda messages, auto=False: (_ for _ in ()).throw(
                                AssertionError("no boundary, no fold")))
        assert _occompact.maybe_compact_after_boundary([]) is False
    finally:
        _occompact.OCC_ENABLED = 0
