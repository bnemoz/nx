# SPEC: right status panel (`/panel`, 0.127.0; reworked 0.140.0)

Goal: a status surface on the existing inline live region (no alternate
screen), as a staged path toward an interactive sub-agent dashboard
(docs/ROADMAP.md defers full-TUI; this stays inline).

0.140.0 reworked the phase-1 rail: its bugs were geometry bugs (codepoint
slices on ANSI lines, full-frame repaint churn), and its content was too thin
to be worth the column it occupied. The rework keeps the original ownership
rule — `_panel.py` owns *content*, `_surfaces.py` owns *painting*.

## Design

- `nx/_ansi.py` provides the one display-width trio (`display_width`,
  `clip_display`, `pad_display`; widths from `rich.cells`). Every frame slice
  routes through it: a wide char can neither shift the `│` nor wrap a row, and
  no slice cuts an escape sequence mid-way.
- Rail geometry is in display cells with the terminal's LAST column reserved
  (`lw = cols - pw - 3`): writing the last column auto-wraps on most
  terminals, which adds a row the Live's erase count never accounted for.
- Content: 7 fixed slots, hide-when-zero, all passive (never fork/probe per
  tick; polled cells are TTL-cached, 2 s, in `_panel.py`):
  1. `ctx` — utilization-colored (green <30%, yellow <60%, red ≥); bar form
     (`▰▱` + %) when the rail is under `BAR_MODE_WIDTH` (20); a `/compact`
     hint at the red threshold when it fits.
  2. `mode` — permission mode (yolo / prompt-all / auto:<level>); hidden when
     unknown.
  3. `branch` — git branch + dirty count, TTL-cached `git status --porcelain
     --branch --untracked-files=no`; detached HEAD → short hash; no repo →
     cell hidden; name left-elided.
  4. `jobs` / 5. `agents` — alive counts from `_surfaces_snapshot()` rows.
  6. `steer` — pending steering lines.
  7. `cost` (session total, priced models only; blank ≠ 0) or `claims`
     (`_fileclaim.active_claims()`, in-TTL rows).
- Rail = live aggregating state; the committed turn footer = per-turn facts.
  Both render through `_render.kv_line` (UIUX_SPEC B1) and share the MUTED
  role (B3).
- Repaint change suppression: `repaint_live` composes the frame once and skips
  `live.update + refresh` when the plain (ANSI-stripped) lines match the last
  painted frame for the same Live instance. Rich rewrites whole frames, so an
  unchanged frame is pure traffic + flicker.
- Height clamp: `_frame_height() = max(1, min(NX_TABS_HEIGHT, rows - 3))` — a
  tall frame on a short terminal pushes the prompt off-screen. `_tabs_height()`
  keeps its own 1..20 env contract.
- Layout constants live in `nx/_layout.py`: `PANEL_MIN_WIDTH/PANEL_MAX_WIDTH`
  (NX_PANEL_WIDTH clamps, 12..40), `PANEL_MIN_CHAT_COLS` (60), and
  `PANEL_DEGRADED_MIN_COLS` (46).

## Degradation ladder

- Rail on: `cols - PW >= PANEL_MIN_CHAT_COLS` (60).
- Degraded: the user wants the rail (`panel_wants_on()`, the intent flag that
  `/panel` toggles) but it doesn't fit, and `cols >= PANEL_DEGRADED_MIN_COLS`
  (46): the same cells render as ONE bottom frame line joined by a muted `·`
  (rail-off, compact-on; `_panel.degraded_line`). The line is a frame row
  inside the height clamp.
- Below 46: one line of status is itself noise — the frame stays byte-identical
  to panel-off.

## Gates and invariants

- `NX_PANEL` env default; `/panel` sets the in-process override
  (`set_panel_enabled`); intent vs actual is split (`panel_wants_on` vs
  `panel_enabled`) so `/panel` toggles sanely on narrow terminals.
- Panel off ⇒ `_live_frame` byte-identical to the pre-panel frame.
- One region, one erase arithmetic, one `write_lock()` acquisition per tick;
  repaints only from the stdin owner's tick; a repaint crash disables the
  region for the session (SPEC_LIVEENGINE).
- The rail is read-only/passive: no probes, no network, no per-tick forks.

## Touch points

- `nx/_panel.py` — content: `panel_cells`, `degraded_line`, gates, TTL caches.
- `nx/_ansi.py` — the display-width trio.
- `nx/_surfaces.py` — rail composite, height clamp, repaint suppression.
- `nx/_layout.py` — panel width/chat-col/degraded-floor constants.
- `nx/_fileclaim.py` — `active_claims()`.
- `nx/_repl.py` — `/panel` toggle; `nx/_chatbox.py` completions entry.
- `tests/test_panel.py` — the contract above (display-width clips, cell
  contents, TTL caches, row-diff, height clamp, degraded mode, the
  byte-identical regression).

## Out of scope (later phases)

Width-clamped chat streaming, interactive focus + input routing, a
Claude-Code-style user-overridable status command fed JSON (one subprocess per
turn, rendered through the same clip discipline).
