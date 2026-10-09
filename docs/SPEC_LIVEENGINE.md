# SPEC: Live-engine repaint for the live region (0.128.0)

## Problem

The /tabs live region paints with hand-rolled cursor arithmetic
(`_DASH_DRAWN`, `\x1b[{n}A\r\x1b[J`). The arithmetic assumes the region is
the last thing written; every mid-turn print below it (stats footer, tool
blocks, spinner) invalidates the anchor, so frames stack and tear (observed
0.127.x). rich's `redirect_stdout` is NOT the fix — its AnsiDecoder drops
non-SGR control bytes, destroying nx's raw writes.

## Design

One shared rich console, one Live, refresh under the existing lock:

1. **Console singleton moves to `nx/_rich.py`**: `shared_console()` returns
   the lazily-created themed `Console` (what `_turn_footer._say_console` is
   today, built there at import). `_turn_footer` re-exports it (keep the
   `_say_console` name working; moved-names rule). All Lives must share THIS
   console: same-console Lives stack into one block via rich's
   `_live_stack`; different consoles on stdout fight.

2. **Region becomes a rich Live** in `nx/_surfaces.py`:
   `Live(console=shared, auto_refresh=False, transient=True,
   redirect_stdout=False, redirect_stderr=False)`; `get_renderable` returns
   a `Group` from `_live_frame()` lines joined. `open_dashboard()` starts it
   (lazy, idempotent, inside `write_lock()`); `close_dashboard()` stops it
   (transient=True erases the frame — replaces `_erase_live`). Delete
   `_DASH_DRAWN`, `_erase_live`, and the manual erase prefix entirely.
   Keep `_DASH_OPEN/_DASH_DISABLED/_DASH_LOCK`, `_vt_supported()` gate, and
   the crash-disables-session behavior (except → `close_dashboard()`).
   `repaint_live()` becomes: build frame, `live.update(Group)`,
   `live.refresh()` — all inside `write_lock()`. No auto-refresh thread
   (guardrail: never repaint from an external thread; refresh stays on the
   owner hooks `_interrupt` poll tick + chatbox idle tick).

3. **Unlocked writers funnel through the shared console** so rich composes
   them above the live frame (Console.print during an active Live emits the
   line above the frame; SGR survives; no manual control bytes needed):
   - `_turn_footer._print_stats_footer`: `print(...)` →
     `shared.print(...)` (keep the exact same strings).
   - `_toolblocks.render_group` prints → same.
   - `_stream_parse` raw `sys.stdout.write(CLEAR_LINE + ...)` sites →
     `console.print` of the plain text (the Live owns line discipline).
   - `LifeSpinner`: its raw daemon-thread writes tear under Live. Convert
     its frame emission to `shared_console.print` of the spinner line (plain
     text + SGR only, no CLEAR_LINE) — one line per tick above the frame;
     and its erase/stop path likewise. Keep its daemon thread + tick (it
     already exists); only the write channel changes. Guard with
     `_ansi`-level try/except as today.

4. **Unchanged**: `run_tabs_dashboard` (interactive full-screen raw loop,
   its own terminal ownership), `_chatbox` (owns the prompt; region never
   open at the prompt — 0.127.1 contract), `_turn.py`/`_dispatch.py` locked
   `print()` sites (they may stay raw: they run under `write_lock()`, and a
   print while Live is up but the cursor mid-frame is still resolved by the
   next `live.refresh()` repositioning... NO — follow-up: migrate these to
   `console.print` too, mechanically, in the same modules, so ALL mid-turn
   text goes through the console. The AST lock test is replaced by an AST
   "no bare print in _turn/_turn_footer/_toolblocks/_stream_parse turn paths"
   guard asserting console.print instead).

## Migration mechanics

- Writer migration is mechanical: within the four writer modules, replace
  `print(` with `console.print(` where console = the shared console (import
  from `._rich`), preserving f-strings/args. `sys.stdout.write` sites in
  `_stream_parse` become console.print of the text payload.
- `tests/test_perf_render_path.py` patches `sp.Live` — Live construction
  moves but `StreamParser` keeps its own Live; verify the fake still fits
  (`auto_refresh` unchanged there).
- Tests to rework: `tests/test_tabs_live.py` byte-erase assertions
  (122-144) become Live-behavior assertions (update called, refresh called,
  closed → nothing); crash-disables and tick-driven tests keep semantics.
  `tests/test_panel.py` `_live_frame` tests unchanged (pure function).
  AST guards re-written for console.print discipline.
- Version 0.128.0 (semver minor: behavior change, no API break).

## Non-goals (later)

Absolute-cursor anchoring, phase-4 interactive dashboard (Textual, separate
`nx dash` surface), width-clamped streaming (SPEC_PANEL phase 2).
