#!/usr/bin/env python3
"""0.168.7 (audit P3-1): the PTC sandbox layers are silent on failure —
pin the parent-side status probe contract and the /toolgates line."""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-ptc-sandbox-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nx import _ptc_exec as pe  # noqa: E402


def test_sandbox_status_contract():
    st = pe.sandbox_status()
    assert set(st) == {"seccomp", "landlock", "nnp"}
    assert all(isinstance(v, bool) for v in st.values())
    # cached: the second call returns the same dict without re-probing
    assert pe.sandbox_status() == st


def test_sandbox_summary_shapes(monkeypatch):
    monkeypatch.setattr(pe, "_SANDBOX_PROBE",
                        {"seccomp": True, "landlock": True, "nnp": True})
    assert pe.sandbox_summary() == "seccomp + landlock + nnp"
    monkeypatch.setattr(pe, "_SANDBOX_PROBE",
                        {"seccomp": True, "landlock": False, "nnp": True})
    assert pe.sandbox_summary() == "seccomp + nnp"
    monkeypatch.setattr(pe, "_SANDBOX_PROBE",
                        {"seccomp": False, "landlock": False, "nnp": False})
    assert pe.sandbox_summary() == ("no OS sandbox (seccomp unavailable, "
                                    "landlock unavailable, nnp unavailable)")
    monkeypatch.setattr(pe, "_SANDBOX_PROBE", None)  # restore the real probe


def test_toolgates_reports_sandbox_line(capsys):
    from nx import _toolview
    lines = _toolview._tools_report_lines()
    sandbox_lines = [ln for ln in lines if "PTC sandbox" in ln]
    assert len(sandbox_lines) == 1, lines
    # the line carries the summary shape, not just a label
    assert "seccomp" in sandbox_lines[0] or "no OS sandbox" in sandbox_lines[0]
