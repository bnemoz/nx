"""0.96.0: the session-start delegation note names the write tiers ONLY when
this session authorizes write-capable sub-agents (NX_SUBAGENT_WRITE=1). A
read-only session must not invite the model to try tier='write'."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nx  # noqa: E402
import nx._delegation_note as d  # noqa: E402


def test_write_guidance_absent_without_authorization(monkeypatch):
    monkeypatch.setattr(nx, "NX_SUBAGENTS", 1)
    monkeypatch.setattr(nx, "NX_SUBAGENT_WRITE", 0)
    content = d._load_delegation_note()["content"]
    assert "tier=" not in content
    assert content.endswith("instead.]")


def test_write_guidance_present_when_authorized(monkeypatch):
    monkeypatch.setattr(nx, "NX_SUBAGENTS", 1)
    monkeypatch.setattr(nx, "NX_SUBAGENT_WRITE", 1)
    content = d._load_delegation_note()["content"]
    assert "tier='write'" in content and "'implement'" in content
    assert "async=false" in content and "scope" in content
    assert "worktree" in content
    assert content.endswith("merge or cherry-pick from it.]")


def test_marker_contract_kept(monkeypatch):
    # _add_delegation_note_if_absent scans for the marker before inserting;
    # both note shapes (with and without the write block) must carry it.
    monkeypatch.setattr(nx, "NX_SUBAGENTS", 1)
    monkeypatch.setattr(nx, "NX_SUBAGENT_WRITE", 1)
    assert d.DELEGATION_NOTE_MARKER in d._load_delegation_note()["content"]


def test_env_fallback_when_attr_missing(monkeypatch):
    # A fresh import surface without the live attributes still honors the env.
    monkeypatch.delattr(nx, "NX_SUBAGENTS", raising=False)
    monkeypatch.delattr(nx, "NX_SUBAGENT_WRITE", raising=False)
    monkeypatch.setenv("NX_SUBAGENTS", "1")
    monkeypatch.setenv("NX_SUBAGENT_WRITE", "1")
    assert "'implement'" in d._load_delegation_note()["content"]
