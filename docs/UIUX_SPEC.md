# Spec: UI/UX overhaul (Phases A–D)

Status: draft, not started. Semver: A/B minor, C/D minor (D is feature-scale, not breaking — scroll-chat remains default).

## Problem

nx's human-facing UI has accreted independently: 12 themes, a live dashboard,
a status rail, a steering queue — each good, none sharing a layout contract,
and several registries/widths hand-synced. The overhaul is architectural, not
cosmetic: one render vocabulary, one command registry, one disclosure policy.

## UX principles (binding for all phases)

1. **Spatial consistency** — rail, footer, chatbox, tool-box never move or
   change shape mid-session.
2. **Progressive disclosure, 4 tiers** — T0 always visible (chatbox border,
   turn footer); T1 on demand (/help, /tooloutput); T2 dashboard (/tabs,
   full-screen views); T3 logs/transcripts. Every emission site gets a tier.
3. **Semantic color only** — no raw palette reads (C.CYAN…) outside
   `_ansi.py`/`_rich.py`; diff +/- must use SUCCESS/ERROR roles, DIM/BOLD
   must become roles (MUTED/EMPHASIS).
4. **Never freeze, always escapable** — Esc has a defined escalation ladder.
5. **Keyboard layers** — L0 arrows/Enter/Esc, L1 vim-ish + ctrl bindings
   (existing), L2 mnemonic single keys per surface, L3 power/chords.
   Each surface documents its own keys in its `?` overlay.

## Phase A — Registry consolidation (first)

### A1. Single slash-command registry
Today three hand-synced registries: `SLASH_COMMANDS` display list
(`nx/_chatbox.py:36-80`, "kept in sync with the README"), `_REPL_COMMANDS`
handler map (`nx/_repl.py:118-163`, 36 entries), `SLASH_ARG_OPTIONS`
(`nx/_chatbox.py:88-104`, "mirrors the handlers").

**Change**: new `nx/_commands.py` with one decorator:
```python
@command("panel", aliases=[], args=None, help="toggle the status rail", tier=0)
def _cmd_panel(...) -> None: ...
```
`SLASH_COMMANDS`, `_REPL_COMMANDS`, `SLASH_ARG_OPTIONS`, and `/help`
rendering all derive from the registry. Handlers stay in `_repl.py` (or move
by domain later — out of scope); the registry module only holds metadata.
Keep module-level names (`SLASH_COMMANDS` etc.) re-exported so tests that
patch them keep working (moved-names rule).

**Acceptance**: `/help`, chatbox completion, and arg dropdowns all derive from
one table; a test asserts registry commands == README Commands table rows
(fixes README drift mechanically); no behavior change in completion order.

### A2. Layout constants module
Magic numbers scattered: spinner `LABEL_CELLS=34`/`TAIL_BUDGET=48`
(`_progress.py:47-49`), reasoning divider 46 (`_turn_footer.py:216`),
chatbox fallback 80 (`_chatbox.py:674,796`), surfaces 80/40
(`_surfaces.py:368`), panel min-width 60 + clamp 12–40 (`_panel.py:40-56`),
diff context 2 (`_approval.py:34`), tool result preview 8 lines
(`_dispatch.py:645-647`), dropdown 8 rows.

**Change**: `nx/_layout.py` (or extend `_progress.py`) defines named
constants with the current values — a pure move, no retuning. Env overrides
only where one already exists; do not add new env vars without need.

**Acceptance**: grep finds no bare width literals at the old sites; suite
passes unchanged.

### A3. Shared-writer discipline
Ad-hoc `print()` bypassing the shared console / `write_lock()`:
`_ansi.py:315-316,393-394` (theme warnings, stderr — allowed to stay stderr
but via one helper), `_repl.py:111,114-115` (/panel feedback),
`_approval.py:208,249,267` (fallback messages), `_approval.py:243-244`
(yolo toggle line).

**Change**: route through a `notice()` helper (see B1) or, for stderr
warnings, a single `_warn()` in `_ansi.py`. Off-tty behavior preserved.

**Acceptance**: no raw `print(` in the touched files except `_chatbox.py`
fallback and off-tty input paths.

## Phase B — Rendering vocabulary

### B1. Component set in `nx/_render.py` (new)
One implementation each, consuming theme roles:
- `panel(title, lines, accent)` — box drawing (unified glyph set, see B2);
  consumed by tool-box (`_dispatch.py:492-522`), chatbox frame
  (`_chatbox.py:720-724`), rail cells (`_panel.py`), diff preview.
- `notice(level, text)` — one-line WARNING/ERROR/SUCCESS/INFO emission with
  the right role color; replaces A3's print sites.
- `kv_line(pairs)` — footer-style `key value ·` rows (`_turn_footer.py`).
- `diff(old, new, context)` — approval diff (`_approval.py:40-69`), now via
  SUCCESS/ERROR roles, context from `_layout`.
- `stream_tail(text, width)` — spinner/log tails (`_progress.py`,
  `_bash_jobs.py`).

Dual raw-ANSI/rich-Live paths: each component decides once which mode it
renders in; callers never branch on it. Where a surface has two code paths
today (spinner `_draw_frame` `_ansi.py:475-500`, tool-status
`_stream_parse.py:250-276`), fold the branch inside the component.

**Acceptance**: tool-box, chatbox frame, footer, rail, diff all render via
`_render.py` components; visual smoke via `/theme` across 2+ themes; no
regression in the tearing fixes documented in `docs/SPEC_LIVEENGINE.md`
(single console, single Live, write_lock per tick — must hold).

### B2. Unified box glyphs
`┌─│└` (tool box) vs `╭─│╰╮╯` (chatbox) vs `└` (footer). One glyph set from
`_render.py` (rounded `╭` family everywhere, or square everywhere — decide at
implementation; rounded matches the chatbox and is the recommendation).

### B3. Role completion
Add roles `MUTED` (replaces raw DIM on text), `EMPHASIS` (replaces raw BOLD),
`DIFF_ADD`/`DIFF_REMOVE` (alias SUCCESS/ERROR by default). Deprecate raw
`DIM`/`BOLD` usage on text outside `_ansi.py`/`_rich.py`. Keep `DIM`/`BOLD`
constants (mode control, not color) — they stay for spinners/hints.

**Acceptance**: `grep -n "C\.\(CYAN\|GREEN\|YELLOW\|RED\|MAGENTA\|BLUE\|ORANGE\|PURPLE\|TEAL\|PINK\|BROWN\)" nx/*.py`
returns hits only in `_ansi.py`/`_rich.py`/`_render.py`; per-role env overrides
still work; budget check (`tests/test_initial_context.py`) unaffected (no
schema changes).

## Phase C — Interaction upgrades

### C1. Esc escalation ladder + hard tool cancellation
Today: bare Esc stops the stream wait but "Tools are never killed … hard
cancellation remains a separate follow-up" (`_interrupt.py:4-6`).

**Spec**: Esc (1×) = interrupt the current wait (existing). Esc (2× within
800 ms) = abort the turn (existing REDIRECT path). Ctrl+C = escalate to hard
cancellation: cooperative kill of the current tool group —
- run_bash: SIGTERM → SIGKILL after 2 s to the process group; job marked
  killed in `/bg`.
- Sub-agents: cooperative abort flag checked at step boundaries (threads
  cannot be killed; document this honestly).
- Other tools (read/write/etc.): single cancel-on-wait check; they are
  sub-second anyway.
Implementation lands in `_interrupt.py` (ladder) + `_bash_jobs.py` (pgid
kill) + `_turn_toolcalls.py` (abort flag plumb). Cancelled tools return a
`cancelled` result to the model (never silently vanish — transcript stays
honest).

**Acceptance**: test kills a `sleep 300` bg job via the ladder; model sees a
cancelled-tool result; no thread leak (the `_interrupt.py:17-35` history is
the regression to guard against).

### C2. Contextual `?` overlay
Each surface (chatbox, approval prompt, /tabs, full-screen views) defines its
key list; `?` renders a full-screen-free overlay (one frame above the
surface, any key dismisses). Data from the same registry as the border hints —
hints and overlay can't drift.

### C3. Approval prompt polish
- `+/`/`-` widens/narrows diff context (default 2 from `_layout`).
- Shift+Tab YOLO toggle: after confirming, proceed with the pending action
  instead of re-prompting it (`_approval.py:231-246`).
- Show the resolved approval reason (allowlist hit, risk level) on one line.

### C4. Headless parity
Approval on headless surfaces currently auto-DENYs (`_telegram.py:20-24`,
`_subagent_bash.py:20-21`).
**Spec**: a policy pre-check hook: `NX_APPROVAL_POLICY=headless-allow:<glob>`
lets an operator pre-approve tool patterns for a headless surface; default
stays deny. Telegram: approval requests are posted as a message with
`y`/`n` reply, gated behind an explicit opt-in flag (default unchanged).
`nx run` batch (`_batch.py`): honor the same policy env. Everything else
(spinner, footer, dashboard) stays degrade-to-print as today.

## Phase D — Full-screen layout modes

**Feasibility (Python)**: yes — standard library only. The pattern is
alternate-screen buffer (`\033[?1049h/l`) + the existing raw-mode key reader
(`_chatbox.py:_raw_read_key`) + the existing shared-console discipline.
No dependency needed; Textual/urwid considered and rejected (dependency
budget, and nx's render needs are modest).

**Views** (entered from the prompt, Esc returns to scroll-chat; inline
fallback off-tty and when `NX_FULLSCREEN=0`):
1. `/tabs` — live job/agent dashboard already exists; promote to
   alternate-screen with focus navigation + j/k + steering per row (steering
   queue plumbing already in `_surfaces.py`).
2. `/review` — full-screen diff review of working tree: file list, per-file
   diff via `diff()`, a=approve-style commit flow reusing `_gitcommit.py`.
3. `/sessions` — session browser (`_sessions_cmd.py`): search (FTS5 exists),
   preview, resume. Drill-down pattern: list → preview → confirm.

**Architecture**: `nx/_fullscreen.py` owns buffer enter/exit, the view
dispatch, and the "stop the world" contract (turn loop idle or paused;
dashboard/tails paused, not lost). Views render via `_render.py`
components only. State is read-only from views except through existing
commands (resume/commit) — no new mutation paths.

**Acceptance**: enter/exit leaves cursor position and scrollback exactly as
before-entry (alternate screen guarantees this; test with a pty); resize
handled (SIGWINCH repaint); every view has inline fallback output identical
to today's command output.

## Ordering & estimates

- A1+A2+A3: mechanical, low risk. One implementer pass + tests.
- B1–B3: touches many render sites; do after A so `_render.py` can assume the
  registry/constants exist. Highest visual payoff per effort.
- C1 is the largest functional item; C2–C4 small.
- D depends on B (components) and is four smallish views sharing one
  fullscreen shell; do the shell + `/tabs` first, other views follow.

## Explicitly out of scope

- Theming engine (already ahead of peers), streaming markdown pipeline,
  security stripping, steering queue, chatbox editor internals.
- Raising the prompt/schema budget — this spec is display-layer only, zero
  model-facing token cost.
