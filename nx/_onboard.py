"""First-run onboarding (0.77.0): when nx starts with no usable API endpoint
configured, walk the user through pasting an endpoint (base URL), an API key,
and optionally a SearXNG instance URL, then persist everything to ~/.env so
every future launch just works.

Design notes:

- Runs from main() BEFORE `import nx`, because importing nx triggers
  switch_source()'s startup network probes — useless (and slow) against an
  endpoint that does not exist yet. The wizard instead imports only the tiny
  leaf modules it needs (_env, _webfetch for the SearXNG check) after it has
  written ~/.env, then re-dispatches into normal startup.
- Detection is deliberately conservative: onboarding triggers only when there
  is genuinely nothing to talk to — no NX_SOURCES / NX_SOURCE_*_BASE_URL /
  NX_BASE_URL / TOGETHER_API_KEY / OPENROUTER_API_KEY anywhere in the
  environment or ~/.env. Anything configured means the user knows what they
  are doing; never nag them.
- The SearXNG question is folded in (SEARXNG_SETUP.md documents the server
  side): if NX_SEARX_URL is unset we offer three paths — paste an existing
  instance's URL (probed live with ?format=json so a typo never silently
  degrades search to the DDG fallback), install a fresh loopback-only local
  instance from source (the SEARXNG_SETUP.md steps, automated: clone, venv,
  settings.yml, systemd unit), or skip.
- Everything lands in ~/.env (NX_ENV_FILE override honored, same file
  _load_env_file reads at import) via _set_env_var, and the file is chmod
  600 afterwards — it now holds a real API key.
"""
from __future__ import annotations

import os
import sys

# Re-exported for tests / the `nx setup` verb.
SEARX_DEFAULT_URL = "http://localhost:8888"
SEARX_SRC_DIR = "~/.local/share/searxng-src"
SEARX_SETTINGS_DIR = "~/.local/share/searxng"


def _run(cmd: list[str]) -> bool:
    """Run a setup command, streaming its output so the user sees progress.
    Returns True on exit 0. Kept on subprocess (not shell) so paths with
    spaces survive and there is no injection surface from prompts."""
    import subprocess
    print("  $ " + " ".join(cmd), flush=True)
    try:
        return subprocess.call(cmd) == 0
    except OSError as e:
        print(f"  ✗ {e}", file=sys.stderr)
        return False


def _have(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None


def _pick_impl(candidates):
    """The real port-picker logic, shared with _pick_searx_port so tests can
    call it directly with custom candidates (avoiding recursion when the
    module attribute is patched)."""
    import socket
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    return 0


def _pick_searx_port(candidates: tuple[int, ...] = (8888, 8090, 8080, 9090)) -> int:
    """First loopback port free among the candidates, verified by
    bind-and-release. 8888 is the documented default but is ALSO Jupyter's
    default — in notebook containers a hardcoded 8888 either fails to start
    or, worse, gets nx pointed at the notebook server. Returns 0 when every
    candidate is taken."""
    return _pick_impl(candidates)


def install_local_searx() -> str | None:
    """Automate the SEARXNG_SETUP.md steps for a fresh local instance: clone
    the source into ~/.local/share/searxng-src, build a uv venv (with the
    --no-build-isolation workaround for upstream's undeclared build deps;
    setuptools is installed explicitly since uv venvs don't seed it),
    write a loopback-only settings.yml with a fresh random secret_key, and
    keep it running: a systemd **user** service on Linux, a LaunchAgent on
    macOS, or — where neither exists (containers, chroots) — a detached
    background process with a persistence hint.

    The port is chosen to avoid collisions: an existing settings.yml keeps
    the port it configures; otherwise the first free candidate among
    8888/8090/8080/9090 is picked (8888 alone is not safe — it is Jupyter's
    default port).

    Returns the URL to store in NX_SEARX_URL on success, or None when
    unsupported/unavailable/failed — the wizard then falls back to the
    paste-a-URL path so nothing is lost.
    """
    # sys.platform, not os.name: macOS is also "posix", and each platform
    # gets its own service manager branch below (systemd user unit on Linux,
    # a LaunchAgent plist on macOS). Anything else is unsupported.
    if sys.platform not in ("linux", "darwin"):
        print("  ✗ automated install supports Linux (systemd) and macOS "
              "(launchd); elsewhere, install SearXNG manually and paste "
              "its URL.")
        return None
    if not _have("git"):
        print("  ✗ git not found — install git first, or paste an existing "
              "instance's URL instead.")
        return None
    home = os.path.expanduser("~")
    src = os.path.join(home, ".local/share/searxng-src")
    cfg_dir = os.path.join(home, ".local/share/searxng")

    # 1. source + venv --------------------------------------------------------
    if not os.path.isdir(os.path.join(src, "searx")):
        if not _run(["git", "clone", "--depth", "1",
                     "https://github.com/searxng/searxng.git", src]):
            return None
    py = os.path.join(src, ".venv", "bin", "python")
    venv_exists = os.path.exists(py)
    if not venv_exists:
        use_uv = _have("uv")
        if use_uv:
            if not _run(["uv", "venv", os.path.join(src, ".venv"),
                         "--python", "3.12"]):
                return None
        elif _have("python3"):
            if not _run(["python3", "-m", "venv", os.path.join(src, ".venv")]):
                return None
        else:
            print("  ✗ neither uv nor python3 found — cannot build a venv.")
            return None
        # venv-module venvs bundle pip; uv-created ones do NOT (uv never needs
        # it), so `python -m pip` there fails with "No module named pip". Route
        # installs through uv's own resolver against the venv instead.
        # The editable install runs with --no-build-isolation (SearXNG's
        # build backend is undeclared upstream), so setuptools must already
        # be IN the venv. Neither uv venv (never seeds it) nor python -m
        # venv on 3.12+ (dropped setuptools from seed packages) provides it
        # — install it explicitly alongside the runtime requirements.
        if use_uv:
            if not _run(["uv", "pip", "install", "--python", py, "-r",
                         os.path.join(src, "requirements.txt"), "setuptools"]):
                return None
            if not _run(["uv", "pip", "install", "--python", py,
                         "--no-build-isolation", "-e", src]):
                return None
        else:
            if not _run([py, "-m", "pip", "install", "-r",
                         os.path.join(src, "requirements.txt"), "setuptools"]):
                return None
            if not _run([py, "-m", "pip", "install", "--no-build-isolation",
                         "-e", src]):
                return None

    # 2. settings.yml — fresh secret per machine, loopback bind, json format.
    # Port: reuse what an existing settings.yml configures (a prior install
    # may already have a unit/plist pointed at it); for a fresh file, pick a
    # free port instead of trusting 8888 (Jupyter's default — inside notebook
    # containers it is taken, and nx must never point at Jupyter by accident).
    import re as _re
    import secrets
    os.makedirs(cfg_dir, exist_ok=True)
    settings = os.path.join(cfg_dir, "settings.yml")
    port = 0
    if os.path.exists(settings):
        try:
            with open(settings, encoding="utf-8") as f:
                m = _re.search(r"^\s*port:\s*(\d+)", f.read(), _re.M)
            if m:
                port = int(m.group(1))
        except OSError:
            pass
    if not port:
        port = _pick_searx_port()
        if not port:
            print("  ✗ no free loopback port among 8888/8090/8080/9090 — "
                  "free one up and re-run `nx setup`.")
            return None
    if not os.path.exists(settings):
        with open(settings, "w", encoding="utf-8") as f:
            f.write(
                "use_default_settings: true\n\n"
                "server:\n"
                f"  secret_key: \"{secrets.token_hex(32)}\"\n"
                "  bind_address: \"127.0.0.1\"   # loopback ONLY\n"
                f"  port: {port}\n"
                "  limiter: false              # private instance\n\n"
                "search:\n"
                "  formats:\n"
                "    - html\n"
                "    - json                    # required: nx queries ?format=json\n")

    # 3. service manager: LaunchAgent on macOS, systemd user unit on Linux ----
    import subprocess
    if sys.platform == "darwin":
        # launchd has no per-user "enable" state: a plist in
        # ~/Library/LaunchAgents is loaded and RunAtLoad keeps it alive
        # across reboots. `launchctl setenv` is not needed since the env
        # vars travel inside the plist itself.
        launch_dir = os.path.join(home, "Library/LaunchAgents")
        os.makedirs(launch_dir, exist_ok=True)
        plist = os.path.join(launch_dir, "io.github.searxng.nx.plist")
        log = os.path.join(cfg_dir, "searxng.log")
        if not os.path.exists(plist):
            with open(plist, "w", encoding="utf-8") as f:
                f.write(
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                    "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
                    "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
                    "<plist version=\"1.0\">\n<dict>\n"
                    "  <key>Label</key><string>io.github.searxng.nx</string>\n"
                    "  <key>ProgramArguments</key>\n  <array>\n"
                    f"    <string>{py}</string>\n"
                    "    <string>-m</string>\n"
                    "    <string>searx.webapp</string>\n"
                    "  </array>\n"
                    "  <key>EnvironmentVariables</key>\n  <dict>\n"
                    "    <key>SEARXNG_SETTINGS_PATH</key>\n"
                    f"    <string>{cfg_dir}/settings.yml</string>\n"
                    "  </dict>\n"
                    f"  <key>WorkingDirectory</key><string>{src}</string>\n"
                    "  <key>RunAtLoad</key><true/>\n"
                    "  <key>KeepAlive</key><true/>\n"
                    "  <key>StandardOutPath</key>\n"
                    f"    <string>{log}</string>\n"
                    "  <key>StandardErrorPath</key>\n"
                    f"    <string>{log}</string>\n"
                    "</dict>\n</plist>\n")
        # kickstart -k (re)starts now; bootout first so an edited plist is
        # re-read (idempotent: bootout of an unloaded label just fails, fine).
        _run(["launchctl", "bootout", "gui", plist])
        if not _run(["launchctl", "bootstrap", "gui", plist]) or not _run(
                ["launchctl", "kickstart", "-k",
                 f"gui/{os.getuid()}/io.github.searxng.nx"]):
            print("  ✗ launchd agent failed to start — check "
                  f"`launchctl list | grep searxng` and {log}. You can paste "
                  "a SearXNG URL instead.")
            return None
    else:
        unit_dir = os.path.join(home, ".config/systemd/user")
        os.makedirs(unit_dir, exist_ok=True)
        unit = os.path.join(unit_dir, "searxng.service")
        if not os.path.exists(unit):
            with open(unit, "w", encoding="utf-8") as f:
                f.write(
                    "[Unit]\n"
                    "Description=SearXNG metasearch (local, for nx web_fetch)\n\n"
                    "[Service]\n"
                    f"ExecStart={py} -m searx.webapp\n"
                    f"Environment=SEARXNG_SETTINGS_PATH={cfg_dir}/settings.yml\n"
                    f"WorkingDirectory={src}\n"
                    "Restart=on-failure\n\n"
                    "[Install]\n"
                    "WantedBy=default.target\n")
        started = False
        try:
            _run(["systemctl", "--user", "daemon-reload"])
            started = _run(["systemctl", "--user", "enable", "--now", "searxng"])
        except OSError:
            started = False  # no systemd user session at all (containers)
        if not started:
            # Container / chroot / systemd-less box: run the webapp directly
            # as a detached child (own session via start_new_session — never
            # an external setsid binary, see the 0.91.1 macOS fix), logging
            # to cfg_dir/searxng.log. It does not survive a container
            # restart; the printed hint says where to add the launch.
            log_path = os.path.join(cfg_dir, "searxng.log")
            with open(log_path, "ab") as log:
                subprocess.Popen(
                    [py, "-m", "searx.webapp"],
                    cwd=src,
                    env={**os.environ,
                         "SEARXNG_SETTINGS_PATH": os.path.join(cfg_dir, "settings.yml")},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=True)
            print(f"  ! no systemd user session here (container?) — SearXNG "
                  f"started as a plain background process (log: {log_path}). "
                  "It will NOT restart on reboot: add the launch to the "
                  "container entrypoint or ~/.bashrc.")
        else:
            # Linger check (boot-time autostart); advisory only.
            try:
                out = subprocess.check_output(
                    ["loginctl", "show-user",
                     os.environ.get("USER", ""), "-p", "Linger"])
                if b"Linger=no" in out:
                    print("  ! Linger=no — the service won't autostart at boot. "
                          "Enable with: sudo loginctl enable-linger $USER")
            except (OSError, subprocess.CalledProcessError):
                pass  # no loginctl — advisory only

    # 4. wait briefly for the port, then probe --------------------------------
    import time
    url = f"http://127.0.0.1:{port}"
    for _ in range(20):
        if _probe_searx(url) is None:
            print(f"  ✓ SearXNG installed and answering at {url}")
            return url
        time.sleep(1.0)
    print("  ✗ installed but not answering yet (first start can be slow) — "
          "re-run `nx setup` later to point nx at it.")
    return None


def _needs_onboarding() -> bool:
    """True only when no API endpoint is configured anywhere.

    Mirrors _discover_sources()'s triggers: NX_SOURCES names, any
    NX_SOURCE_<NAME>_BASE_URL, the legacy NX_BASE_URL, or a key for one of
    the built-in convenience sources (together/openrouter). Any hit means
    config exists → skip the wizard.
    """
    if os.environ.get("NX_SOURCES", "").strip():
        return False
    for k in os.environ:
        if k.startswith("NX_SOURCE_") and k.endswith("_BASE_URL"):
            return False
    if os.environ.get("NX_BASE_URL", "").strip():
        return False
    if os.environ.get("TOGETHER_API_KEY", "").strip():
        return False
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return False
    # ~/.env was loaded into os.environ by _load_env_file() before this runs
    # (nx/__init__ does it at import; main() does it explicitly pre-import),
    # so the checks above already cover the file. Belt-and-braces: read the
    # file directly in case a future refactor drops the eager load.
    from ._env import _env_file_path
    try:
        with open(_env_file_path(), encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith("#") or "=" not in s:
                    continue
                key = s.split("=", 1)[0].strip()
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                if key in ("NX_SOURCES", "NX_BASE_URL",
                           "TOGETHER_API_KEY", "OPENROUTER_API_KEY") \
                        or (key.startswith("NX_SOURCE_")
                            and key.endswith("_BASE_URL")):
                    return False
    except OSError:
        pass
    return True


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{label}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""
    return val or default


def _normalize_base_url(raw: str) -> str:
    """Accept bare hosts and missing /v1 — OpenAI-compatible servers want
    http(s)://host[:port]/v1."""
    url = raw.strip().rstrip("/")
    if url and "://" not in url:
        url = "http://" + url
    if url and not url.endswith("/v1") and "/v1/" not in url:
        url += "/v1"
    return url


def _probe_searx(url: str) -> str | None:
    """GET {url}/search?q=test&format=json; return an error string or None
    when the instance answers with JSON. Kept dependency-free (urllib) so it
    can run before `import nx`."""
    import json as _json
    import urllib.request
    try:
        req = urllib.request.Request(
            url.rstrip("/") + "/search?q=nx+setup&format=json",
            headers={"User-Agent": "nx-onboarding/0.77"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            _json.loads(resp.read(65536).decode("utf-8", "replace"))
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _maybe_lock_env_file() -> None:
    """chmod 600 on the env file once it holds a key (POSIX only; a no-op on
    Windows, where ntfs ACLs govern and os.chmod maps only the read bit)."""
    from ._env import _env_file_path
    path = _env_file_path()
    if os.name == "posix":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def run_onboarding(interactive: bool = True) -> bool:
    """Interactive first-run wizard. Returns True when config was written.

    Asks for (1) an OpenAI-compatible base URL, (2) its API key, (3) a
    SearXNG URL for web search, and persists NX_SOURCES=local,
    NX_SOURCE_LOCAL_BASE_URL/API_KEY and NX_SEARX_URL into ~/.env. A blank
    answer to the endpoint prompt (or Ctrl-D/Ctrl-C) aborts with instructions
    instead of writing a half configuration.
    """
    from ._env import _set_env_var
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        # Piped/captured stdin (CI, harnesses, `nx --prompt-file -`): there is
        # no human to paste a key, so prompting would hang or crash. Print a
        # pointer instead and let startup carry on / fail with its usual
        # "no source configured" message.
        print("nx: no endpoint configured — run `nx setup` (interactive) to "
              "configure one, or set NX_SOURCE_* vars in ~/.env.",
              file=sys.stderr)
        return False
    print(
        "\nWelcome to nx! No model endpoint is configured yet.\n"
        "Let's set one up — paste the base URL of an OpenAI-compatible\n"
        "API endpoint (llama.cpp, vLLM, LM Studio, OpenRouter, Z.ai, ...).\n"
        "Examples: http://localhost:8080  ·  https://api.together.xyz/v1\n"
        "Press Ctrl-C any time to quit (nothing is saved until you finish).\n")
    raw = _prompt("API base URL")
    if not raw:
        print(
            "nx: no endpoint given — nothing saved. Set it up later with\n"
            "    `nx setup`, or by hand in ~/.env (see sources.example.env).")
        return False
    base_url = _normalize_base_url(raw)
    api_key = _prompt("API key (blank for local servers)", "sk-noop")

    print("\nOptional: set up SearXNG for web search (the preferred backend;\n"
          "without it nx falls back to DuckDuckGo, which rate-limits).\n"
          "Install a fresh local instance? (Linux or macOS; ~5 min: clones\n"
          "SearXNG, builds a venv, loopback-only service) [Y/n]")
    searx = ""
    if _prompt("Install SearXNG locally? (Y/n)", "y").strip().lower() != "n":
        searx = install_local_searx() or ""
    else:
        while True:
            cand = _prompt("SearXNG URL (blank to skip — DuckDuckGo fallback)")
            if not cand:
                break
            cand = cand.rstrip("/")
            if "://" not in cand:
                cand = "http://" + cand
            err = _probe_searx(cand)
            if err is None:
                searx = cand
                print(f"  ✓ SearXNG reachable at {cand}")
                break
            retry = _prompt(f"  ✗ not reachable ({err}) — try again? (y/N)", "n")
            if retry.lower() != "y":
                break

    _set_env_var("NX_SOURCES", "local")
    _set_env_var("NX_SOURCE_LOCAL_BASE_URL", base_url)
    _set_env_var("NX_SOURCE_LOCAL_API_KEY", api_key)
    if searx:
        _set_env_var("NX_SEARX_URL", searx)
    _maybe_lock_env_file()

    # Push into os.environ so the subsequent `import nx` sees the config
    # without re-reading the file (mirrors _load_env_file's no-clobber rule —
    # these keys were just confirmed absent by _needs_onboarding()).
    os.environ["NX_SOURCES"] = "local"
    os.environ["NX_SOURCE_LOCAL_BASE_URL"] = base_url
    os.environ["NX_SOURCE_LOCAL_API_KEY"] = api_key
    if searx:
        os.environ["NX_SEARX_URL"] = searx
    # Rebind the webfetch module's copy so this process (which imported
    # _webfetch for the probe) searches the new instance immediately.
    import nx._webfetch as _wf
    _wf.SEARX_URL = searx

    print(f"\nSaved to the nx env file ({os.environ.get('NX_ENV_FILE', '~/.env')})."
          + ("\nWeb search will use your SearXNG instance." if searx else "")
          + "\nStarting nx…\n")
    return True


def cli_onboard(argv: list[str]) -> int:
    """`nx setup` entry: re-run the wizard on demand. The wizard overwrites
    the `local` source's URL/key (and NX_SEARX_URL) in place via
    _set_env_var, so re-running it is always safe. Exit 2 on abort."""
    from ._env import _load_env_file
    _load_env_file()
    return 0 if run_onboarding() else 2
