#!/usr/bin/env python3
"""Risk-classifier context for project-root path decisions."""
import json
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402


def test_downloads_project_file_is_in_project(monkeypatch, tmp_path):
    home = tmp_path / "home"
    project = home / "Downloads" / "didenstuff"
    project.mkdir(parents=True)
    target = project / "pose_editor.py"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    monkeypatch.setattr(m, "PROJECT_ROOT", os.path.realpath(project))

    payload = json.loads(m._risk_user_message(f"edit {target}"))

    assert payload["project_root"] == os.path.realpath(project)
    assert payload["primary_path"] == os.path.realpath(target)
    assert payload["path_scope"] == "in_project"
    assert payload["path_in_downloads"] is True
    assert "Downloads" in payload["scope_guidance"]


def test_downloads_file_outside_project_is_outside(monkeypatch, tmp_path):
    home = tmp_path / "home"
    project = home / "code" / "app"
    downloads_project = home / "Downloads" / "didenstuff"
    project.mkdir(parents=True)
    downloads_project.mkdir(parents=True)
    target = downloads_project / "pose_editor.py"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    monkeypatch.setattr(m, "PROJECT_ROOT", os.path.realpath(project))

    payload = json.loads(m._risk_user_message(f"edit {target}"))

    assert payload["project_root"] == os.path.realpath(project)
    assert payload["primary_path"] == os.path.realpath(target)
    assert payload["path_scope"] == "outside_project"
    assert payload["path_in_downloads"] is True


def test_write_memory_action_path_is_extracted(monkeypatch, tmp_path):
    # memory_write's approval action is "write memory <path> (N bytes)" — the
    # risk classifier must extract the path (not swallow it as "unknown") so
    # the path_scope is computed and a project memory isn't mis-rated high.
    project = tmp_path / "repo"
    project.mkdir()
    target = project / ".nx" / "memories" / "20260807-120000_build.md"
    monkeypatch.setattr(m, "PROJECT_ROOT", os.path.realpath(project))

    payload = json.loads(
        m._risk_user_message(f"write memory {target} (73 bytes)")
    )

    assert payload["primary_path"] == os.path.realpath(target)
    assert payload["path_scope"] == "in_project"


def test_write_memory_global_action_path_is_outside_project(monkeypatch, tmp_path):
    project = tmp_path / "repo"
    project.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    global_mem = home / ".nx" / "memories" / "20260807-120000_tip.md"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(m, "PROJECT_ROOT", os.path.realpath(project))

    payload = json.loads(
        m._risk_user_message(f"write memory {global_mem} (42 bytes)")
    )

    assert payload["path_scope"] == "outside_project"
