"""First-run onboarding (nx/_onboard.py): detection, URL normalization, the
wizard's ~/.env writes, and the SearXNG probe. All offline — the probe is
monkeypatched, and no test talks to the network."""
import os
import sys
import urllib.request
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("NX_SKIP_STARTUP_PROBES", "1")
os.environ["NX_ENV_FILE"] = "/dev/null"

from nx import _onboard  # noqa: E402


@pytest.fixture()
def env_file(tmp_path, monkeypatch):
    """A fresh, empty env file; _load_env_file/_set_env_var honor NX_ENV_FILE
    at call time, so pointing it at a tmp file keeps tests hermetic."""
    path = tmp_path / "env"
    path.write_text("")
    monkeypatch.setenv("NX_ENV_FILE", str(path))
    return path


def _scrub(monkeypatch):
    for k in ("NX_SOURCES", "NX_BASE_URL", "TOGETHER_API_KEY",
              "OPENROUTER_API_KEY", "NX_SEARX_URL"):
        monkeypatch.delenv(k, raising=False)
    for k in list(os.environ):
        if k.startswith("NX_SOURCE_"):
            monkeypatch.delenv(k, raising=False)


# --- detection -------------------------------------------------------------

def test_needs_onboarding_when_nothing_configured(monkeypatch):
    _scrub(monkeypatch)
    monkeypatch.setenv("NX_ENV_FILE", "/dev/null")
    assert _onboard._needs_onboarding() is True


def test_no_onboarding_when_nx_sources_set(monkeypatch):
    _scrub(monkeypatch)
    monkeypatch.setenv("NX_SOURCES", "local")
    assert _onboard._needs_onboarding() is False


def test_no_onboarding_when_builtin_key_set(monkeypatch):
    _scrub(monkeypatch)
    monkeypatch.setenv("NX_ENV_FILE", "/dev/null")
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    assert _onboard._needs_onboarding() is False


def test_no_onboarding_when_env_file_has_base_url(env_file, monkeypatch):
    _scrub(monkeypatch)
    env_file.write_text("NX_SOURCE_MINE_BASE_URL=http://x/v1\n")
    monkeypatch.setenv("NX_ENV_FILE", str(env_file))
    # Load the file so the direct-read fallback ALSO sees config (it must
    # agree with _load_env_file, or detection double-books).
    from nx._env import _load_env_file
    _load_env_file()
    assert _onboard._needs_onboarding() is False


# --- URL handling ------------------------------------------------------------

def test_normalize_base_url_adds_scheme_and_v1():
    assert _onboard._normalize_base_url("localhost:8080") == "http://localhost:8080/v1"
    assert _onboard._normalize_base_url("myhost.dev") == "http://myhost.dev/v1"


def test_normalize_base_url_keeps_existing():
    assert _onboard._normalize_base_url("https://api.together.xyz/v1") \
        == "https://api.together.xyz/v1"
    assert _onboard._normalize_base_url("http://h:1/v1/") == "http://h:1/v1"


# --- wizard ------------------------------------------------------------------

def _answers(monkeypatch, *seq):
    it = iter(seq)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


def _fake_tty(monkeypatch):
    """run_onboarding refuses to prompt without a tty; xdist runs captured."""
    monkeypatch.setattr(_onboard.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(_onboard.sys.stdout, "isatty", lambda: True)


def test_wizard_writes_env_file(env_file, monkeypatch):
    _scrub(monkeypatch)
    _fake_tty(monkeypatch)
    _answers(monkeypatch, "localhost:8080", "", "n", "")
    assert _onboard.run_onboarding()
    text = env_file.read_text()
    assert "NX_SOURCES=local\n" in text
    assert "NX_SOURCE_LOCAL_BASE_URL=http://localhost:8080/v1\n" in text
    assert "NX_SOURCE_LOCAL_API_KEY=sk-noop\n" in text
    assert "NX_SEARX_URL" not in text
    # And the process env got the same values (import nx picks them up).
    assert os.environ["NX_SOURCE_LOCAL_BASE_URL"] == "http://localhost:8080/v1"


def test_wizard_persists_key_and_locks(env_file, monkeypatch):
    _scrub(monkeypatch)
    _fake_tty(monkeypatch)
    _answers(monkeypatch, "http://h:9999/v1", "secret-key", "n", "")
    assert _onboard.run_onboarding()
    assert "NX_SOURCE_LOCAL_API_KEY=secret-key\n" in env_file.read_text()
    if os.name == "posix":
        assert (env_file.stat().st_mode & 0o777) == 0o600


def test_wizard_abort_writes_nothing(env_file, monkeypatch):
    _scrub(monkeypatch)
    _fake_tty(monkeypatch)
    _answers(monkeypatch, "", "n")  # blank endpoint → abort
    assert _onboard.run_onboarding() is False
    assert env_file.read_text() == ""
    assert "NX_SOURCES" not in os.environ


def test_wizard_saves_working_searx(env_file, monkeypatch):
    _scrub(monkeypatch)
    _fake_tty(monkeypatch)
    _answers(monkeypatch, "localhost:8080", "", "n", "http://localhost:8888")
    with mock.patch.object(_onboard, "_probe_searx", return_value=None):
        assert _onboard.run_onboarding()
    assert "NX_SEARX_URL=http://localhost:8888\n" in env_file.read_text()
    assert os.environ["NX_SEARX_URL"] == "http://localhost:8888"


def test_wizard_retries_bad_searx_then_skips(env_file, monkeypatch):
    _scrub(monkeypatch)
    _fake_tty(monkeypatch)
    _answers(monkeypatch, "localhost:8080", "", "n",
             "http://bad:1", "n")  # probe fails, user declines retry
    with mock.patch.object(_onboard, "_probe_searx",
                           return_value="URLError: refused"):
        assert _onboard.run_onboarding()
    assert "NX_SEARX_URL" not in env_file.read_text()


# --- non-interactive guard ---------------------------------------------------

def test_wizard_refuses_non_tty(env_file, monkeypatch, capsys):
    _scrub(monkeypatch)
    monkeypatch.setattr(_onboard.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(_onboard.sys.stdout, "isatty", lambda: True)
    assert _onboard.run_onboarding() is False
    assert env_file.read_text() == ""
    assert "nx setup" in capsys.readouterr().err


# --- local SearXNG installer (yes-branch; _run mocked, no network) ----------

def test_install_local_searx_rejects_non_linux(monkeypatch, capsys):
    """Only Linux and macOS are automated (systemd / launchd); anything else
    must bail BEFORE running any command."""
    monkeypatch.setattr(_onboard.sys, "platform", "win32")
    monkeypatch.setattr(_onboard, "_run",
                        lambda cmd: (_ for _ in ()).throw(
                            AssertionError("must not run any command")))
    assert _onboard.install_local_searx() is None
    assert "Linux (systemd) and macOS" in capsys.readouterr().out


def test_install_local_searx_macos_launchd(tmp_path, monkeypatch):
    """On darwin the installer writes a LaunchAgent plist (RunAtLoad/KeepAlive,
    same settings path as Linux) and bootstraps it via launchctl — no systemd
    anywhere."""
    home = tmp_path
    src = home / ".local/share/searxng-src"
    (src / "searx").mkdir(parents=True)  # post-clone tree
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(_onboard.sys, "platform", "darwin")
    monkeypatch.setattr(_onboard, "_probe_searx", lambda url: None)
    calls = []
    monkeypatch.setattr(_onboard, "_run", lambda cmd: calls.append(cmd) or True)
    monkeypatch.setattr(_onboard, "_have", lambda c: c in ("uv", "git"))
    assert _onboard.install_local_searx().startswith("http://127.0.0.1:")
    plist = home / "Library/LaunchAgents/io.github.searxng.nx.plist"
    text = plist.read_text()
    assert "<string>-m</string>" in text and "<string>searx.webapp</string>" in text
    assert "RunAtLoad" in text and "KeepAlive" in text
    assert ".local/share/searxng/settings.yml" in text
    assert not any("systemctl" in c for c in calls)
    assert any(c[0] == "launchctl" and c[1] == "bootstrap" for c in calls)
    assert any(c[0] == "launchctl" and c[1] == "kickstart" for c in calls)
    assert ["uv", "pip", "install", "--python",
            str(src / ".venv/bin/python"), "-r",
            str(src / "requirements.txt"), "setuptools"] in calls


def test_install_local_searx_uses_uv_pip_for_uv_venvs(tmp_path, monkeypatch):
    """uv-created venvs ship no pip, so installs must go through
    `uv pip install --python <venv python>` (regression: `python -m pip`
    failed with 'No module named pip' right after a successful `uv venv`)."""
    src = tmp_path / ".local/share/searxng-src"
    (src / "searx").mkdir(parents=True)  # post-clone tree; no venv yet
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(_onboard.sys, "platform", "linux")
    monkeypatch.setattr(_onboard, "_probe_searx", lambda url: None)
    calls = []
    monkeypatch.setattr(_onboard, "_run", lambda cmd: calls.append(cmd) or True)
    monkeypatch.setattr(_onboard, "_have", lambda c: c in ("uv", "git"))
    assert _onboard.install_local_searx().startswith("http://127.0.0.1:")
    assert ["uv", "pip", "install", "--python",
            str(src / ".venv/bin/python"), "-r",
            str(src / "requirements.txt"), "setuptools"] in calls
    assert not any("-m" in c and "pip" in c for c in calls)

def test_install_local_searx_happy_path(tmp_path, monkeypatch):
    src = tmp_path / "searxng-src"
    (src / "searx").mkdir(parents=True)
    py = src / ".venv/bin/python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(_onboard, "_probe_searx", lambda url: None)
    calls = []
    monkeypatch.setattr(_onboard, "_run", lambda cmd: calls.append(cmd) or True)
    # Pretend systemctl exists; the unit file gets written by the installer.
    monkeypatch.setattr("shutil.which", lambda c: "/usr/bin/" + c)
    url = _onboard.install_local_searx()
    assert url and url.startswith("http://127.0.0.1:")
    settings = (tmp_path / ".local/share/searxng/settings.yml").read_text()
    assert f"port: {url.rsplit(':', 1)[1]}" in settings
    assert 'bind_address: "127.0.0.1"' in settings
    assert "- json" in settings
    assert "secret_key" in settings and "PASTE_RANDOM" not in settings
    unit = (tmp_path / ".config/systemd/user/searxng.service").read_text()
    assert "-m searx.webapp" in unit
    assert "SEARXNG_SETTINGS_PATH" in unit
    assert any("daemon-reload" in " ".join(c) for c in calls)
    assert any("enable" in c and "searxng" in c for c in calls)


def test_install_local_searx_idempotent(tmp_path, monkeypatch):
    """Existing settings.yml / unit / venv are left alone — a re-run must not
    regenerate the secret_key or clobber manual edits."""
    home = tmp_path
    src = home / ".local/share/searxng-src"
    (src / "searx").mkdir(parents=True)
    (src / ".venv/bin").mkdir(parents=True)
    (src / ".venv/bin/python").write_text("")
    (home / ".local/share/searxng").mkdir(parents=True)
    (home / ".local/share/searxng/settings.yml").write_text("secret_key: keep\n")
    (home / ".config/systemd/user").mkdir(parents=True)
    (home / ".config/systemd/user/searxng.service").write_text("# custom\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(_onboard, "_probe_searx", lambda url: None)
    monkeypatch.setattr(_onboard, "_run", lambda cmd: True)
    assert _onboard.install_local_searx().startswith("http://127.0.0.1:")
    assert "keep" in (home / ".local/share/searxng/settings.yml").read_text()
    assert "# custom" in (home / ".config/systemd/user/searxng.service").read_text()


def test_install_local_searx_git_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(_onboard, "_run", lambda cmd: False)
    assert _onboard.install_local_searx() is None


def test_pick_searx_port_skips_taken(tmp_path, monkeypatch):
    """Port selection must never land on a taken loopback port — 8888 is
    Jupyter's default and notebook containers run one there (regression: the
    installer pointed NX_SEARX_URL at the Jupyter server, whose /search 404s
    fall back to DDG)."""
    import socket
    # Occupy two REAL ports (not the fixed candidates, which another test
    # run / the dev box's live searxng may already hold) and point the
    # candidate list at them: first taken, second free.
    s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s1.bind(("127.0.0.1", 0))
    s1.listen(1)
    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s2.bind(("127.0.0.1", 0))
    s2.listen(1)
    free_port = s2.getsockname()[1]
    s2.close()  # the free candidate must be genuinely free when picked
    try:
        taken_port = s1.getsockname()[1]
        real_pick = _onboard._pick_impl
        monkeypatch.setattr(_onboard, "_pick_searx_port",
                            lambda c=None: real_pick((taken_port, free_port)))
        assert _onboard._pick_searx_port() == free_port
    finally:
        s1.close()
        s2.close()

def test_install_local_searx_reuses_existing_settings_port(tmp_path, monkeypatch):
    """An existing settings.yml decides the port even when 8888 is free — a
    prior install's unit/plist is already pointed at it."""
    home = tmp_path
    src = home / ".local/share/searxng-src"
    (src / "searx").mkdir(parents=True)
    (src / ".venv/bin").mkdir(parents=True)
    (src / ".venv/bin/python").write_text("")
    cfg = home / ".local/share/searxng"
    cfg.mkdir(parents=True)
    (cfg / "settings.yml").write_text(
        "server:\n  port: 9444\nsearch:\n  formats: [html, json]\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(_onboard.sys, "platform", "linux")
    monkeypatch.setattr(_onboard, "_probe_searx", lambda url: None)
    monkeypatch.setattr(_onboard, "_run", lambda cmd: True)
    assert _onboard.install_local_searx() == "http://127.0.0.1:9444"
    assert "port: 9444" in (cfg / "settings.yml").read_text()


def test_install_local_searx_container_fallback(tmp_path, monkeypatch):
    """No systemd user session (containers): the installer must still leave a
    working instance — detached webapp process (start_new_session, never an
    external setsid) with SEARXNG_SETTINGS_PATH in its env — instead of
    failing the whole install."""
    home = tmp_path
    src = home / ".local/share/searxng-src"
    (src / "searx").mkdir(parents=True)
    (src / ".venv/bin").mkdir(parents=True)
    (src / ".venv/bin/python").write_text("")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(_onboard.sys, "platform", "linux")
    monkeypatch.setattr(_onboard, "_probe_searx", lambda url: None)
    monkeypatch.setattr(_onboard, "_run",
                        lambda cmd: cmd[0] != "systemctl")  # systemd unusable
    spawned = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            spawned.append((cmd, kw))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

        def communicate(self, *a, **kw):
            return b"", b""

        @property
        def args(self):
            return []

    monkeypatch.setattr("subprocess.Popen", FakePopen)
    url = _onboard.install_local_searx()
    assert url and url.startswith("http://127.0.0.1:")
    assert len(spawned) == 1
    cmd, kw = spawned[0]
    assert cmd[1:3] == ["-m", "searx.webapp"]
    assert kw["start_new_session"] is True
    assert kw["env"]["SEARXNG_SETTINGS_PATH"].endswith("settings.yml")
    out = (home / ".local/share/searxng/searxng.log").read_text()  # created
    assert isinstance(out, str)



def test_wizard_yes_installs_searx(env_file, monkeypatch):
    _scrub(monkeypatch)
    _fake_tty(monkeypatch)
    _answers(monkeypatch, "localhost:8080", "", "y")
    with mock.patch.object(_onboard, "install_local_searx",
                           return_value="http://127.0.0.1:8888") as inst:
        assert _onboard.run_onboarding()
    inst.assert_called_once()
    assert "NX_SEARX_URL=http://127.0.0.1:8888\n" in env_file.read_text()


# --- SearXNG probe (offline: mocked transport) -------------------------------

def test_probe_searx_parses_json(monkeypatch):
    class Resp:
        def read(self, n=-1):
            return b'{"results": [{"url": "http://x"}]}'
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    def fake_urlopen(req, timeout):
        assert "format=json" in req.full_url
        return Resp()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert _onboard._probe_searx("http://localhost:8888") is None


def test_probe_searx_reports_errors(monkeypatch):
    def boom(req, timeout):
        raise OSError("refused")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert "URLError" in _onboard._probe_searx("http://localhost:8888") \
        or "OSError" in _onboard._probe_searx("http://localhost:8888")
