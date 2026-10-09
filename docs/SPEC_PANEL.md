# SPEC: right status panel (`/panel`, 0.127.0)

Goal: a persistent right-hand status rail on the existing inline live region
(no alternate screen), as phase 1 of a staged path toward an interactive
sub-agent dashboard (docs/ROADMAP.md defers full-TUI; this stays inline).

## Design

- New module `nx/_panel.py` owns panel *content*: what cells exist, from which
  snapshot data, and how they render. It does NOT own painting — painting
  stays in `nx/_surfaces.py` (one region, one erase arithmetic, one
  `write_lock()` acquisition per tick).
- The live-region frame (`_surfaces._live_frame`) composes each line as
  `left[:LW] + pad + │ + cell[:PW]` where `PW = panel width` (NX_PANEL_WIDTH,
  default 24, clamped 12..40) and `LW = cols - PW - 2`. When the panel is off,
  `_live_frame` output is byte-identical to today.
- Panel cells (phase 1, all data already collected):
  1. `ctx <cur>/<max>` — utilization-colored, from `st._LAST_PROMPT_TOKENS` /
     `st.ACTIVE._context_window` (same rule as `_turn_footer._ctx_field`,
     without kicking a probe).
  2. `jobs N` — alive bg-job count (from `_surfaces_snapshot()` rows).
  3. `agents N` — alive async sub-agent count.
  4. `steer N` — pending steering lines (0 hidden).
  5. claims row — `claims N` active cross-process file claims
     (`_fileclaim.active_claims()` count, phase-3 placeholder OK if the
     getter is awkward: degrade to nothing).
- Fewer left lines than cells: leftover cells are dropped (right rail is a
  status summary, not a table). More left lines than cells: remaining left
  lines get blank cells.
- Gates: `NX_PANEL=1` enables; default off this release. Panel silently off
  when `_vt_supported()` is false or `cols - PW < 60`.

## Touch points

- `nx/_panel.py` (new): `panel_cells(width, rows) -> list[str]` (pure, styled),
  `panel_enabled() -> bool` (env + width gate).
- `nx/_surfaces.py`: `_live_frame` composes the rail via `_panel.panel_cells`
  when enabled; `_tabs_height()` unchanged.
- `nx/_repl.py`: `/panel` command → toggles the `NX_PANEL`-equivalent in-process
  flag only (0.127.1: no prompt-time `open_dashboard()` — the region is a
  mid-turn construct owned by whoever holds stdin, spec §2/§3); the rail opens
  at turn start via `_panel.ensure_region()` and closes in the turn's `finally`.
- `nx/_repl_settings_cmds.py`: `_HELP_COMMANDS` entry.
- `nx/_chatbox.py`: completions entry next to `/tabs` (line ~49).
- `nx/__init__.py`: export `_panel_toggle` if the command needs it (keep
  surface minimal).
- README.md: short `/panel` row in the dashboard section.
- Version bump in `pyproject.toml` (semver: minor — new feature).

## Tests (tests/test_panel.py)

- `panel_cells` pure: styled cells, blank fill, width truncation, disabled →
  empty list.
- `_live_frame` composition: panel on → every line ends with `│` + cell;
  panel off → identical to the pre-panel frame (regression).
- Gates: NX_PANEL unset → off; NX_PANEL_WIDTH clamping.
- No tty required (headless-safe, like the rest of the suite).

## Out of scope (later phases)

Width-clamped chat streaming (phase 2), claims/agent-tail depth (3),
interactive focus + input routing (4).
