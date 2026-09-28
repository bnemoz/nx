"""Sub-agent turn budget (0.103.10): spawn-time floor/ceiling clamping.

The default is a FLOOR, not just the omitted-value fallback, and it is
tier-aware: read-tier spawns floor at NX_SUBAGENT_DEFAULT_TURNS (100) while
write/implement-tier spawns floor at the higher NX_SUBAGENT_WRITE_TURNS (250),
because delegated multi-file work (read the area, edit, verify, iterate) needs
more turns than a read sweep. The ceiling (NX_SUBAGENT_MAX_TURNS, 500) matches
the parent loop's MAX_MODEL_TURNS for the write-capable era.

These tests drive spawn_agent's clamp arithmetic without a live model by
monkeypatching the two entry points it calls after the clamp
(_spawn_async / _run_sync_agent) to capture the max_turns they are handed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("NX_ENV_FILE", "/dev/null")
# Clear ambient NX_* config so a developer's ~/.env / shell can't affect results
# — per repo test guidelines.
_AMBIENT_NX = [k for k in os.environ if k == "NX_ENV_FILE" or k.startswith("NX_")]
for k in _AMBIENT_NX:
    os.environ.pop(k, None)
os.environ["NX_ENV_FILE"] = "/dev/null"


def _capture(monkeypatch):
    """Monkeypatch spawn_agent's two terminal calls to record max_turns.

    Returns a dict with a 'turns' key set to the captured budget (as a plain
    int) and a 'jobs' list of the kwargs each call received.
    """
    import nx
    from nx import _subagent

    captured = {"turns": None, "jobs": []}

    def _fake_spawn(instruction, max_turns, tier="read", scope=None):
        captured["turns"] = int(max_turns)
        captured["jobs"].append(("async", tier, scope))
        return "job-1"

    def _fake_sync(instruction, max_turns, tier="read", scope=None):
        captured["turns"] = int(max_turns)
        captured["jobs"].append(("sync", tier, scope))
        return "answer"

    monkeypatch.setattr(_subagent, "_spawn_async", _fake_spawn)
    monkeypatch.setattr(_subagent, "_run_sync_agent", _fake_sync)
    # Reliable baseline for every test: read=100, write=250, ceiling=500.
    monkeypatch.setattr(nx, "NX_SUBAGENT_DEFAULT_TURNS", 100, raising=False)
    monkeypatch.setattr(nx, "NX_SUBAGENT_WRITE_TURNS", 250, raising=False)
    monkeypatch.setattr(nx, "NX_SUBAGENT_MAX_TURNS", 500, raising=False)
    return captured


def _spawn(tier, max_turns_marker, monkeypatch):
    """Call spawn_agent in the foreground (async_=False) with the given tier.

    `max_turns_marker` is a sentinel: _OMIT means the argument is not passed at
    all; any other value is forwarded verbatim.
    """
    from nx import _subagent

    kwargs = {"async_": False}
    if max_turns_marker is not _OMIT:
        kwargs["max_turns"] = max_turns_marker
    if tier is not None:
        kwargs["tier"] = tier
    if tier in ("write", "implement"):
        monkeypatch.setattr(_subagent, "_write_tier_enabled", lambda: True)
    return _subagent.spawn_agent("do a thing", **kwargs)


_OMIT = object()


def test_read_floor_raises_small_request(monkeypatch):
    captured = _capture(monkeypatch)
    _spawn("read", 5, monkeypatch)
    assert captured["turns"] == 100, captured


def test_read_omitted_defaults_to_floor(monkeypatch):
    captured = _capture(monkeypatch)
    _spawn("read", _OMIT, monkeypatch)
    assert captured["turns"] == 100, captured


def test_write_omitted_floors_at_write_turns(monkeypatch):
    captured = _capture(monkeypatch)
    _spawn("write", _OMIT, monkeypatch)
    assert captured["turns"] == 250, captured


def test_write_small_request_raised_to_write_floor(monkeypatch):
    captured = _capture(monkeypatch)
    _spawn("write", 5, monkeypatch)
    assert captured["turns"] == 250, captured


def test_write_above_floor_passes_through(monkeypatch):
    captured = _capture(monkeypatch)
    _spawn("write", 400, monkeypatch)
    assert captured["turns"] == 400, captured


def test_ceiling_clamps_oversized_request(monkeypatch):
    captured = _capture(monkeypatch)
    _spawn("write", 1000, monkeypatch)
    assert captured["turns"] == 500, captured


def test_ceiling_clamps_read_request_too(monkeypatch):
    captured = _capture(monkeypatch)
    _spawn("read", 1000, monkeypatch)
    assert captured["turns"] == 500, captured


def test_write_turns_override_raises_write_floor(monkeypatch):
    captured = _capture(monkeypatch)
    import nx
    monkeypatch.setattr(nx, "NX_SUBAGENT_WRITE_TURNS", 400, raising=False)
    _spawn("write", _OMIT, monkeypatch)
    assert captured["turns"] == 400, captured


def test_write_floor_is_ceiling_bounded(monkeypatch):
    # An inverted config (write floor above ceiling) can never exceed the
    # ceiling: the floor is itself ceiling-bounded.
    captured = _capture(monkeypatch)
    import nx
    monkeypatch.setattr(nx, "NX_SUBAGENT_WRITE_TURNS", 900, raising=False)
    _spawn("write", _OMIT, monkeypatch)
    assert captured["turns"] == 500, captured


def test_write_turns_falls_back_to_250_when_attr_missing(monkeypatch):
    import nx
    from nx import _subagent
    monkeypatch.delattr(nx, "NX_SUBAGENT_WRITE_TURNS", raising=False)
    monkeypatch.delenv("NX_SUBAGENT_WRITE_TURNS", raising=False)
    assert _subagent._subagent_write_turns() == 250


def test_write_turns_env_fallback(monkeypatch):
    import nx
    from nx import _subagent
    monkeypatch.delattr(nx, "NX_SUBAGENT_WRITE_TURNS", raising=False)
    monkeypatch.setenv("NX_SUBAGENT_WRITE_TURNS", "320")
    assert _subagent._subagent_write_turns() == 320


def test_subagent_max_turns_falls_back_to_500(monkeypatch):
    import nx
    from nx import _subagent
    monkeypatch.delattr(nx, "NX_SUBAGENT_MAX_TURNS", raising=False)
    monkeypatch.delenv("NX_SUBAGENT_MAX_TURNS", raising=False)
    assert _subagent._subagent_max_turns() == 500
