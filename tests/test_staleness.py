"""nx/_staleness.py (0.103.0, session-isolation phase 2): the write tools
refuse to clobber a file that changed on disk after THIS session last read
it — the cross-session counterpart of the per-tool approval-window guard."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest

import nx
import nx._bash_jobs


def _fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("NX_HOME", str(tmp_path / "nxhome"))
    monkeypatch.setenv("NX_ENV_FILE", str(tmp_path / "env"))
    import nx._staleness as stal
    stal._LAST_READ_MTIME.clear()
    stal._ORDER.clear()
    return stal


@pytest.fixture(autouse=True)
def _yolo(monkeypatch):
    monkeypatch.setattr(nx, "YOLO", True)
    monkeypatch.setattr(nx._bash_jobs, "_confirm", lambda *a, **k: True)


def test_unread_file_writes_freely(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    f = tmp_path / "a.txt"
    f.write_text("hello\n")
    assert "wrote" in nx._edit_tools.write_file(str(f), "new\n")


def test_read_then_external_change_refused(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    f = tmp_path / "b.txt"
    f.write_text("v1\n")
    nx._tools.read_file(str(f))
    # Simulate another session / editor touching the file.
    os.utime(f, ns=(1_000_000_000_000, 1_000_000_000_000))
    f.write_text("v2 — someone else's work\n")
    res = nx._edit_tools.write_file(str(f), "clobbered\n")
    assert res.startswith("ERROR:") and "changed on disk" in res
    assert f.read_text() == "v2 — someone else's work\n"  # untouched


def test_stale_file_recovers_after_reread(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    f = tmp_path / "c.txt"
    f.write_text("v1\n")
    nx._tools.read_file(str(f))
    os.utime(f, ns=(1_000_000_000_000, 1_000_000_000_000))
    f.write_text("v2\n")
    assert nx._edit_tools.write_file(str(f), "x\n").startswith("ERROR:")
    nx._tools.read_file(str(f))  # re-read refreshes the anchor
    assert "wrote" in nx._edit_tools.write_file(str(f), "x\n")


def test_edit_file_stale_refused(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    f = tmp_path / "d.txt"
    f.write_text("alpha beta\n")
    nx._tools.read_file(str(f))
    os.utime(f, ns=(1_000_000_000_000, 1_000_000_000_000))
    f.write_text("alpha gamma\n")
    res = nx._edit_tools.edit_file(str(f), old="beta", new="BETA")
    assert res.startswith("ERROR:") and "changed on disk" in res


def test_bounded_registry_evicts_oldest(tmp_path, monkeypatch):
    stal = _fresh(tmp_path, monkeypatch)
    monkeypatch.setattr(stal, "_max_paths", lambda: 3)
    for i in range(5):
        f = tmp_path / f"e{i}.txt"
        f.write_text("x\n")
        nx._tools.read_file(str(f))
    assert len(stal._LAST_READ_MTIME) == 3
    # oldest two evicted: writing them again must NOT hit the staleness gate
    # even after touching (no record => free write).
    f0 = tmp_path / "e0.txt"
    os.utime(f0, ns=(1_000_000_000_000, 1_000_000_000_000))
    f0.write_text("changed\n")
    assert "wrote" in nx._edit_tools.write_file(str(f0), "free\n")
