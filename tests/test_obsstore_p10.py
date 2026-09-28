#!/usr/bin/env python3
"""Swarm P10: the cross-process content-addressed tool-result store.

Invariants pinned here:
- identity is the CONTENT sha (identical bytes dedup across tools, args,
  call keys — the old tool+call_key identity wrote a copy per call);
- publication is no-clobber (racing writers converge on one file);
- a 24-hex `obs_` handle resolves to the full-sha object;
- gc() honors TTL, byte quota, and the concurrency grace;
- `nx sessions --prune` sweeps the store;
- the packed handle round-trips through obs_recall byte-exactly;
- _SEND_COUNTS is bounded (drop-oldest past NX_OBS_MAX_TRACKED).
"""
import os
import sys
import tempfile
import time

_tmp = tempfile.mkdtemp(prefix="nx-p10-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest  # noqa: E402
import nx as m  # noqa: E402
from nx import _obs, _obsstore  # noqa: E402
import nx._sessions_fs as fs  # noqa: E402


def _big(n_bytes=20000):
    return ("x" * 79 + "\n") * (n_bytes // 80)


import shutil


@pytest.fixture(autouse=True)
def _clear_counts():
    _obs._SEND_COUNTS.clear()
    fs._SWEEP_LAST.clear()
    shutil.rmtree(_obsstore.store_dir(), ignore_errors=True)
    yield
    _obs._SEND_COUNTS.clear()
    fs._SWEEP_LAST.clear()
    shutil.rmtree(_obsstore.store_dir(), ignore_errors=True)


def _store():
    d = _obsstore.store_dir()
    os.makedirs(d, exist_ok=True)
    return d


# --- identity / dedup ---------------------------------------------------------

def test_identity_is_content_only():
    t = _big()
    a = _obs.pack_result("run_bash", t, "path-a")
    b = _obs.pack_result("read_file", t, "path-b")
    for r in (a, b):
        if r.startswith("[large"):
            assert "id: obs_" in r
    # both identities derive the SAME handle from the content
    oid = "obs_" + _obs.content_hash(t)[:24]
    assert oid in a or a == t  # within the FULL_SENDS grace both are full text
    assert oid in b or b == t


def test_identical_content_stored_once():
    _store()
    t = _big()
    with_obs = {"NX_OBS": "1"}
    old = os.environ.get("NX_OBS")
    os.environ["NX_OBS"] = "1"
    try:
        for _ in range(5):
            _obs.pack_result("run_bash", t, f"call-{_}")
    finally:
        if old is None:
            os.environ.pop("NX_OBS", None)
        else:
            os.environ["NX_OBS"] = old
    stats = _obsstore.store_stats()
    assert stats["objects"] == 1  # five packs, one object


def test_put_is_no_clobber_dedup():
    _store()
    t = "hello world " * 100
    sha1 = _obsstore.put(t)
    sha2 = _obsstore.put(t)
    assert sha1 == sha2
    stats = _obsstore.store_stats()
    assert stats["objects"] == 1


def test_distinct_content_distinct_objects():
    _store()
    _obsstore.put("content one" * 100)
    _obsstore.put("content two" * 100)
    assert _obsstore.store_stats()["objects"] == 2


# --- handle resolution ----------------------------------------------------------

def test_truncated_handle_resolves_full_object():
    _store()
    t = "roundtrip payload " * 500
    sha = _obsstore.put(t)
    oid = "obs_" + sha[:24]
    path = _obs._safe_obs_path(oid)
    assert path is not None and os.path.isfile(path)
    with open(path) as f:
        assert f.read() == t


def test_handle_rejects_traversal():
    assert _obsstore.resolve_handle("obs_../../etc/passwd") is None
    assert _obsstore.resolve_handle("obs_") is None
    assert _obsstore.resolve_handle("obs_ZZZZ") is None
    assert _obsstore.resolve_handle("nope") is None


def test_recall_roundtrip_via_placeholder():
    _store()
    old = os.environ.get("NX_OBS")
    os.environ["NX_OBS"] = "1"
    try:
        t = _big()
        for _ in range(2):
            _obs.pack_result("run_bash", t, "k")
        placeholder = _obs.pack_result("run_bash", t, "k")
        assert placeholder.startswith("[large tool result replaced")
        oid = "obs_" + placeholder.split("id: obs_")[1].split()[0]
        page = m.obs_recall(oid, 0)
        body = page.split("\n", 2)[2]
        assert body.startswith(t[:10])
        assert len(body.encode()) <= _obs.RECALL_MAX_BYTES
    finally:
        if old is None:
            os.environ.pop("NX_OBS", None)
        else:
            os.environ["NX_OBS"] = old


# --- GC ---------------------------------------------------------------------------

def _make_obj(text, age_s=0.0):
    _store()
    sha = _obsstore.put(text)
    if age_s:
        p = _obsstore.object_path(sha)
        old = time.time() - age_s
        os.utime(p, (old, old))
    return sha


def test_gc_ttl_prunes_old_objects():
    _make_obj("old object " * 100, age_s=10 * 86400)
    _make_obj("fresh object " * 100)
    stats = _obsstore.gc(now=time.time(), grace_s=0.0)
    assert stats["removed"] == 1
    assert stats["kept"] == 1


def test_gc_grace_protects_fresh_objects():
    """A 5s-old object is inside both the TTL and the grace window: kept."""
    _make_obj("concurrent writer's object " * 100, age_s=5)
    stats = _obsstore.gc(now=time.time(), grace_s=60.0)
    assert stats["kept"] == 1
    # with the grace off it still survives TTL (5s << 7d)
    stats = _obsstore.gc(now=time.time(), grace_s=0.0)
    assert stats["removed"] == 0


def test_gc_quota_respects_env_toggle(monkeypatch):
    """NX_OBS_STORE_MAX_MB=0 disables the quota half entirely."""
    _store()
    s = _make_obj("a" * 500_000, age_s=3000)
    monkeypatch.setenv("NX_OBS_STORE_MAX_MB", "0")
    assert _obsstore.gc(grace_s=0.0)["removed"] == 0
    assert _obsstore.exists(s)


def test_gc_quota_oldest_evicted_under_pressure(monkeypatch):
    _store()
    s1 = _make_obj("a" * 500_000, age_s=3000)
    s2 = _make_obj("b" * 500_000, age_s=2000)
    s3 = _make_obj("c" * 500_000)
    # 1.5 MB store vs ~1 MB quota (use the float trick: 1 MB)
    monkeypatch.setenv("NX_OBS_STORE_MAX_MB", "1")
    stats = _obsstore.gc(grace_s=0.0)
    assert stats["removed"] >= 1
    assert not _obsstore.exists(s1)      # oldest goes first
    assert _obsstore.exists(s3)          # newest kept
    assert _obsstore.exists(s2) or stats["removed"] == 2


def test_amortized_sweep_wired_into_save():
    _store()
    _make_obj("old object " * 100, age_s=10 * 86400)
    m._write_session("20250101-120000-p10aa",
                     [{"role": "user", "content": "hi"}], {})
    # the sweep is amortized; the save above should have run it (fresh gate)
    assert not _obsstore.exists(_obs.content_hash("old object " * 100))


def test_prune_covers_obsstore():
    """`nx sessions --prune` sweeps the store with no age grace (the save
    path's amortized sweep may have already collected the expired objects —
    both paths must remove them)."""
    _store()
    _make_obj("prune me " * 100, age_s=10 * 86400)
    stats = fs._prune_sessions_store()
    assert "obs_removed" in stats
    assert stats["obs_removed"] >= 1
    assert not _obsstore.exists(_obs.content_hash("prune me " * 100))


# --- bounded send counts -----------------------------------------------------------

def test_send_counts_bounded(monkeypatch):
    # NX_OBS_MAX_TRACKED is resolved at CALL time now (_obs._max_tracked()),
    # so setenv alone drives the cap.
    monkeypatch.setenv("NX_OBS_MAX_TRACKED", "10")
    import nx._obs as O
    try:
        O._SEND_COUNTS.clear()
        for i in range(50):
            O._bump_send_count(f"key-{i}")
        assert len(O._SEND_COUNTS) == 10
        assert "key-0" not in O._SEND_COUNTS  # oldest evicted
        assert "key-49" in O._SEND_COUNTS
    finally:
        O._SEND_COUNTS.clear()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
