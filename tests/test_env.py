"""Tests for .env-file parsing."""
from __future__ import annotations

import os

from nx._env import _load_env_file


def test_load_env_file_strips_unquoted_inline_comment(tmp_path, monkeypatch):
    env_file = tmp_path / "nx.env"
    env_file.write_text("NX_API_KEY=sk-xxx # my key\n", encoding="utf-8")
    monkeypatch.setenv("NX_ENV_FILE", str(env_file))
    monkeypatch.delenv("NX_API_KEY", raising=False)

    _load_env_file()

    assert os.environ["NX_API_KEY"] == "sk-xxx"


def test_load_env_file_preserves_hashes_in_values(tmp_path, monkeypatch):
    env_file = tmp_path / "nx.env"
    env_file.write_text(
        "NX_UNQUOTED=abc#123\nNX_QUOTED='abc # 123'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NX_ENV_FILE", str(env_file))
    monkeypatch.delenv("NX_UNQUOTED", raising=False)
    monkeypatch.delenv("NX_QUOTED", raising=False)

    _load_env_file()

    assert os.environ["NX_UNQUOTED"] == "abc#123"
    assert os.environ["NX_QUOTED"] == "abc # 123"
