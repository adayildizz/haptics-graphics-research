# Tactile Bar Graph Perception Experiment

Psychophysics experiment measuring the JND (just noticeable difference) for tactile bar height on an electroadhesion haptic display, using the method of constant stimuli. The goal is to characterize tactile height discrimination without visual information, enabling data access for non-visual users.

## Hardware

| Component | Specification |
|-----------|---------------|
| Tactile display | Electroadhesion, 4V peak, 125 Hz carrier frequency |
| Position sensor | Nexio NIB170BP infrared frame, ~100 Hz |
| Interface | IR frame presents as virtual mouse via OS driver |

## Experimental Design: Method of Constant Stimuli

One base (reference) height and one bar width are fixed per session (a "configuration"). Comparison heights are placed symmetrically around the base at fixed percentage offsets (`±delta_max_pct` by default), with a fixed number of trials per level -- no staircase, no reversals. Every parameter of this design is a field on `ExperimentConfig` (see `config.py`), loaded from a YAML/TOML file; nothing about the design is hard-coded.

- `levels = linspace(-delta_max_pct, +delta_max_pct, n_levels)`, dropping the 0% level unless `include_zero_level` is set.
- `comparison_height_mm = base_height_mm * (1 + level)`.
- `trials_per_level` reps per level, plus `catch_trial_pct` extra easy (±`delta_max_pct`) trials as a lapse/attention check.
- Reference/comparison side (left/right) is counterbalanced within each level, across the whole block rather than within each sub-block.
- Trial order is randomized **within blocks**, not by one shuffle of the whole set. Each block holds `sweeps_per_block` presentations of every level (default 1, so blocks of `n_levels`), and order is shuffled inside the block. The seeded RNG makes this reproducible; the seed actually used is always logged, even when `rng_seed` is left unset (a fresh one is drawn and recorded).

### Why blocked, not one global shuffle

Both schemes give each level its configured number of trials and both are unbiased *on average*. The difference is what happens within a single session, which is what a threshold is estimated from. Measured over 2000 seeds of the default design (6 levels x 10 trials + 6 catch):

| | one global shuffle | blocked (default) |
| --- | --- | --- |
| A level's trials split between first and second half | 4.6 apart on average, worst 10-0 | exactly even, always |
| Sessions with a 4-6 split or worse | 84% | 0% |
| Longest run of the same level | up to 6 | 2 |
| Same level within a 10-trial window | up to 7 | 3 |
| Four or more catch trials in one third of the session | 28% | 0% |
| A third of the session with no catch trial at all | 23% | 5% |

A session runs ~20 minutes, over which fatigue, learning, and the skin's coupling to the surface all drift. A global shuffle leaves level confounded with time-on-task by exactly that much, and the drift lands on the threshold estimate. Blocking removes the confound by construction and caps same-level runs at two (one at each side of a block boundary).

Catch trials are placed one per block, at a random position inside it, for the same reason: sampling attention across the session works poorly if a quarter of sessions leave a third of the session unsampled.

`sweeps_per_block: 2` gives blocks of 12 instead of 6. Balance is still exact and runs cap at 4; the trade is that with blocks of exactly one sweep the last trial of a block is in principle determined by the preceding five. That is not a realistic concern for a blindfolded 2AFC height judgment, but the wider block is there if a design wants it. `trials_per_level` must be divisible by `sweeps_per_block`.

Both bars are rendered simultaneously on a split screen. Bar interior = signal ON, exterior = OFF.

## Rendering Method

The signal is on inside a bar and off outside it. `render_mode` selects how the
moment of switching is decided. The selected mode is shown on the start screen,
printed at session start and stored in the session's config snapshot.

| `render_mode` | Rule |
| --- | --- |
| `predictive` (default) | The finger's velocity vector is estimated by a linear fit over the last 3 position samples. Along that straight line, the instants at which the finger enters and leaves each bar are computed for the next 1.5 sample intervals (at most 25 ms) and the signal is switched at those instants. Entry and exit can both fall inside one interval (a clipped corner). Each new sample cancels what was planned and plans again. Below 5 mm/s nothing is predicted: the sample itself decides. |
| `position` | On exactly while the latest sample is inside a bar. No prediction; the baseline for comparison. |
| `timed_legacy` | The original rule, kept for comparison and documentation only: on while the sample is inside the bar, and after it leaves until `bar_width / entry speed` has passed since entry (speed floor 1 mm/s, at most 2 s). |

**A lifted finger switches the signal off, in every mode.** The cursor keeps
reporting its last position after the finger leaves the glass, so without this
a lift inside a bar would leave the signal on. The rule sits in one place
(`trial_render.py`) for all three modes. On touching down again every mode starts
afresh, because the cursor has jumped; a switch-on at that moment is logged with
edge `touch`, a switch-off at a lift with edge `lift`. The rule applies only
once the input device has reported a touch in the session (see *Contact* under
the movement trace). A driver that never reports touch keeps the earlier
behaviour, so the stimulus is never silenced by a device that cannot say a
finger is down.

All four edges of a bar are treated alike (left, right, top, bottom), from any
direction. `predictive` and `position` test the finger against the bars at
their **configured millimetre size**; `timed_legacy` keeps the pixel-rounded
rectangles it always used (up to one pixel different). What is drawn on the
operator's screen is unchanged in every mode.

**Who switches the output.** One scheduler thread (`signal_scheduler.py`) is the
only code that writes `OUTP ON` / `OUTP OFF` to the generator during a session.
The trial loop sends it a plan every frame; a newer plan replaces the pending
one. Decisions are still made once per frame (60 per second), from the position
read in that frame. What no longer depends on the frame is the *instant* of a
predicted switch. An earlier version of this section said a timer controlled
the signal independently of the frame rate; that was not so. In every mode
before this one, and in `position` and `timed_legacy` today, a switch can only
be decided at a frame. In `timed_legacy` the decisions and the sequence of
commands are identical to the original code (pinned by
`tests/test_render_legacy_golden.py`), but the write now happens on the
scheduler thread, so its timing is not identical.

The rule `duration = bar_width / finger_speed` came from Sun et al. (2023),
where the program sets the finger's speed. Here the speed is estimated from
two position samples of a free-moving finger, the rule uses bar *width* whatever
the direction of movement, and bar height never enters it. After a slow entry
it keeps the signal on above the bar's top edge. `analysis/render_replay`
measures this (see below).

Settings (all in the YAML config and the supervisor form):

```yaml
render_mode: predictive       # predictive | position | timed_legacy
predict_window_samples: 3     # samples in the velocity fit; unchanged positions are not counted
predict_horizon_factor: 1.5   # horizon = this x the measured median sample interval ...
predict_horizon_max_ms: 25.0  # ... capped here; nothing is ever predicted beyond the horizon
predict_min_speed_mm_s: 5.0   # slower: no prediction
latency_compensation_ms: 0.0  # treat each position as this old and start the prediction further along
signal_watchdog_ms: 100.0     # no plan from the trial loop for this long -> signal off
```

Each bar crossing ("pass") is still logged per trial with its entry speed and
time in the bar. `commanded_duration_s` is filled only in `timed_legacy`; it is
`null` in the other modes.

### Before the first real session: timing check

The scheduler aims to start each planned write within 1 ms of its planned
instant (95th percentile). That depends on the computer. On the lab PC, run:

```
python -m experiment.signal_scheduler
```

It uses a fake device and contacts no generator. It reports the planned-to-write
delay with the main loop idle and under the trial's own drawing loop, and ends
with `PASS` or `FAIL`. If it fails, do not run a session; the fallback is to
move the scheduler into its own process. The delay inside the generator
(output relay, carrier phase) is not part of this figure and has to be measured
with an oscilloscope.

### Before the first real session: input check

```
python -m experiment.input_probe
```

Opens a fullscreen window for 15 s (no generator, no trial). Slide one finger
around, lift it and touch again a few times. It reports how many position
reports per second reach the program and answers three questions: are position
reports kept apart rather than merged into one per frame (judged against the
loop rate it measured; it cannot tell whether every report the frame produced
arrived), is contact reported as a mouse-button press, and is it reported as
touch events. The summary is also
saved to `experiment/data/input_probe_<time>.json`. The answers decide whether
recording a contact flag and sampling per report are worth adding.

### Safety

The output is switched off when a trial ends, when the session ends, when the
scheduler hits an error (the session then ends as `aborted`, with everything
recorded so far on disk), when the trial loop sends no plan for
`signal_watchdog_ms`, and when the program exits normally, through an exception,
Ctrl+C or a closed window. **If the process is killed or crashes, the output can
stay on.** That was already true before this change. The operator must then
switch the generator's output off by hand.

### Known limits

- **Not every IR update is seen.** The position is read once per frame, so about
  40% of the frame's ~100 Hz updates are never read, and a position is 0-10 ms
  old when read. In simulation this leaves a one-sided lag of about 0.6 mm at
  100 mm/s in `predictive`. `latency_compensation_ms` can remove the average
  lag once it has been measured on the rig; it does not narrow the spread.
- **Practice speed guidance uses the two-point speed**, which swings between
  about 55 and 127 mm/s for a true 100 mm/s for the same reason. It was left as
  it is.
- **A prediction can be wrong for one interval.** If the finger slows or turns
  just before an edge, `predictive` may switch on for up to one sample interval
  with the finger still outside.
- **The scheduler is occasionally late.** On the development Mac about one
  switch in a few hundred started 2-11 ms late (the operating system, or a
  pygame call in the main loop that holds the interpreter lock), although the
  95th percentile stayed far below 1 ms. The timing check prints the count.
- **The carrier limits edge precision.** One period of the 125 Hz carrier is
  8 ms, which is 0.8 mm at 100 mm/s.

## Procedure

Each session runs as a single participant-facing fullscreen flow. The
participant is blindfolded and wears headphones; the screen is for the
operator. **Space** or **Enter** (main keyboard) advances the flow and
**Escape** safely exits at any point (as does closing the window). Both are
operator keys, deliberately out of reach of the participant's answering hand.
The task is two-alternative forced choice (2AFC): two bars are rendered
side-by-side and the participant decides which is **taller**.

The participant explores with the **index finger of the dominant hand** and
answers with the **other hand** on the arrow keys, which the operator places on
the side of the non-exploring hand. Record which hand explores in
`dominant_hand` (`left` / `right`); it is written to every trial row and the
config snapshot so exploration strategies can be compared by hand.

From the moment the program starts until it exits, **continuous brown noise**
(`masking_noise`, level `masking_noise_volume`) plays in the headphones from a
reserved mixer channel, masking the rig; the cues and the practice voice play
over it (`experiment/masking_noise.py`).

1. **Start screen.** The session opens on a "Ready to begin" prompt showing the
   mode and the base height/width for this configuration. Press **Space** or
   **Enter** to begin.
2. **Practice block** (skipped if `n_practice_trials == 0`). `n_practice_trials`
   easy trials drawn from the `practice_easiest_levels` largest level magnitudes
   (with the default design: ±20% and ±30%), **feedback forced on**, under the
   same response limit as the main block (unanswered practice trials are re-shown
   at the end of the round, subject to the same `max_trial_attempts` cap).
   Because the participant is blindfolded, feedback is **spoken**
   (`practice_spoken_feedback`: “Correct” / “Incorrect”) as well as shown on
   screen for the operator. Optional spoken speed coaching says “Faster,”
   “Good speed,” or “Slower” after a stable speed reading; it is disabled
   throughout the main block. Voice output uses native Windows SAPI through
   PowerShell on Windows, `say` on macOS, and `espeak` or `spd-say` on Linux.

   **Pass mark.** A round passes when at least
   `ceil(practice_pass_fraction × n_practice_trials)` trials were answered
   correctly (default 0.75 × 16 = **12**); timed-out trials count as wrong. Below
   that, the operator sees the score, the participant hears “The practice block
   will be repeated,” and a freshly shuffled round runs. There is no cap on rounds
   (the operator can always end the session with Escape). Every practice row in
   the CSV and every practice attempt in the trace carries its `practice_round`.
3. **Main block.** The full shuffled constant-stimuli sequence (levels + catch
   trials, interleaved). **Feedback is off** here unless `feedback: true` is set.
   With the default `break_every_n_trials: 0` the block runs **continuously**:
   no break screens, nothing for the participant to talk about, matching the
   briefing. Set it to a positive number to pause on a break prompt every that
   many answered trials (resumed with **Space** / **Enter**).
4. **End screen.** A "Session complete" message confirms data was saved.

Within a single trial the participant slides a finger across the touch surface to
explore the two bars. The electroadhesion signal is delivered over the **bar
interior only** (interior = ON, exterior = OFF), so each bar is felt as a raised
region whose width is governed by the timing method above. When ready, the
participant responds with either keyboard layout:

- **Left arrow** (or **Numpad 4**, which carries the ◀ legend on a numpad) — the
  left bar felt taller.
- **Right arrow** (or **Numpad 6**, ▶) — the right bar felt taller.
- **Escape** (main keyboard, operator only) — safely exit the experiment. Numpad
  0 and Numpad Enter do nothing, so a stray press by the answering hand cannot
  end the session or advance a screen.

Every trial, **practice included**, runs under the same response limit
(`response_timeout_s`, 30 seconds by default), with a visible countdown that turns
red for the final five seconds — practice is where the participant should first
meet the clock, not the main block. Two short synthesized cues mark the two ways a
trial can end (`experiment/audio_cues.py`):

- **Answer registered** — a brief rising blip. Identical for correct and incorrect
  answers, so it never leaks feedback in the no-feedback main block; it only means
  "recorded, moving on".
- **Time expired** — a soft descending chime: slow attack, falling interval, long
  release. Deliberately an invitation back to the task rather than an alarm.

An unanswered trial is **not** dropped and **not** replaced on the spot. It is
deferred to a retry pool that is replayed, reshuffled, once the whole scheduled
sequence has been shown, so the session still collects the configured number of
responses. The handover into that retry round is silent — no announcement screen,
no break prompt — so from the participant's side a repeated trial is just the next
trial.

`max_trial_attempts` (default 3, first showing included) is what keeps the retry
loop finite: it caps how many times any one trial is presented, so total
presentations are bounded by `n_trials * max_trial_attempts` no matter how many go
unanswered. A trial that uses up its attempts is recorded as `exhausted` — logged
to the console and left in the trace database as a `timeout` attempt, costing one
observation at its level rather than stalling the run. The curve fit only ever
consumes `answered` rows, so none of this reaches the psychometric estimate.

Response time is logged per completed trial for reference. Which side holds the
reference vs. the comparison is counterbalanced within each level, and a response
is scored correct when it matches the objectively taller side.

### Blind test variant

Setting `blind_test_mode: true` hides the bars visually so they can be located
**by touch alone** — no on-screen rectangles, only the haptic signal and a touch
cursor. This enforces the study's premise of height discrimination *without*
visual information, and is the condition of interest for non-visual data access;
leave it `false` for sighted-guided piloting and rig checks.

## Repository Structure

```
experiment/
├── main.py              # Entry point: the constant-stimuli session, or --dry-run
├── config.py            # Rig constants + ExperimentConfig (the single design-parameter source)
├── constant_stimuli.py  # Level generation + trial scheduling (no pygame dependency)
├── stimulus.py          # Generator connection and the VISA write helpers
├── render_logic.py      # Pure render decisions: bars, velocity, predictive plans, top-edge cue (no pygame, no VISA)
├── signal_scheduler.py  # The one thread that switches the output; watchdog; timing benchmark
├── trial_render.py      # Per-trial glue: sample -> plan -> scheduler -> trace events
├── input_probe.py       # Rig check: position reports per second, and whether contact is reported
├── trial.py             # Single trial logic, per-pass logging
├── display.py           # Pygame UI and screen layout
├── data_logger.py       # CSV/JSON logging
├── configs/             # Example YAML configs (default.yaml)
└── data/                # Output directory

analysis/
├── fit_psychometric.py  # Psychometric-function fitting (psignifit preferred, scipy MLE fallback),
│                         # plotting, ideal-observer simulation. No pygame dependency; runs standalone
│                         # on saved CSVs.
├── trace_movement.py    # Per-attempt finger animations (GIF/MP4)
├── exploration_maps/    # Visit-count and dwell-time maps per trial and per participant
└── render_replay/       # Replays a recording through each render mode; edge errors in mm; top-edge cue sweep

tests/                   # pytest: level generation, scheduling, psychometric-fit recovery
```

## Configuration (`ExperimentConfig`, loaded from YAML)

```yaml
base_height_mm: 10.0
bar_width_mm: 10.0
inter_bar_gap_mm: 40.0   # >= 3.0 mm (BANA 3.4.3.13; Tang & Beebe 1998)

delta_max_pct: 0.30
n_levels: 7               # odd -> the 0% midpoint is dropped: ±10/±20/±30, six real levels
include_zero_level: false
trials_per_level: 10
sweeps_per_block: 1       # each block holds this many of every level
catch_trial_pct: 0.10

feedback: false
n_practice_trials: 16
practice_easiest_levels: 2       # practice draws from the 2 largest magnitudes: ±20%, ±30%
practice_pass_fraction: 0.75     # 16 trials -> at least 12 correct or the block repeats
practice_spoken_feedback: true   # spoken Correct/Incorrect (the participant is blindfolded)
break_every_n_trials: 0          # 0 = no break screens; the main block runs continuously
response_timeout_s: 30.0  # response time limit, practice trials included
max_trial_attempts: 3     # first showing + retries; bounds the retry loop
practice_voice_feedback: true
ideal_finger_speed_mm_s: 100.0
ideal_speed_tolerance_pct: 0.30
record_main_trace: true   # practice attempts are traced too, tagged is_practice / practice_round

render_mode: predictive   # predictive | position | timed_legacy; see "Rendering Method" for the rest

masking_noise: true       # continuous brown noise in the headphones for the whole session
masking_noise_volume: 0.30

blind_test_mode: false   # true -> hide bars, locate them by touch only (no visual)

rng_seed: null           # null -> random, but the resolved seed is always logged
participant_id: ""
dominant_hand: ""        # "left" | "right": the exploring hand; arrow keys go under the other
mode: constant_stimuli   # the only mode
```

Run with:

```
python -m experiment.main --config experiment/configs/default.yaml --participant P01
python -m experiment.main --config experiment/configs/default.yaml --dry-run         # ideal-observer sanity check, no pygame/hardware
```

Fit a saved session:

```
python -m analysis.fit_psychometric experiment/data/P01_*_trials.csv --out fit.png
```

Animate the finger path of every main-block trial from the session's trace (see
*Movement trace* below; heat maps come from `analysis/exploration_maps`):

```
python -m analysis.trace_movement animate experiment/data/P01_<ts>_trace.sqlite3 --out out/P01/anim
```

### Trial log (`<session_id>_trials.csv`)

Every trial the participant was actually shown gets a row — practice included,
unanswered included — told apart by the **`outcome`** column:

| `outcome` | meaning | in the trace DB |
| --- | --- | --- |
| `answered` | a response was given; the only rows the curve fit consumes | `answered` |
| `timeout` | the response window closed and the trial was deferred for another attempt | `timeout` |
| `exhausted` | the last attempt allowed by `max_trial_attempts` also expired | `timeout` |
| `aborted` | the exit key, a closed window or a signal failure ended the trial, and with it the session | `aborted` |

The vocabulary lines up with the trace database's per-attempt `outcome`, so the
CSV and the replay recordings describe the same set of events. The one refinement
is `exhausted`: whether a timeout was the *last* allowed attempt is a scheduling
fact rather than a property of the attempt itself, so the trace still stores it as
`timeout`.

On every non-`answered` row `response` and `correct` are left **blank** rather than
`0`: no judgment was made, and a `0` would read as a wrong answer. `response_time_s`
holds how long the trial was actually open, and `passes_json` still describes what
the finger was doing — so "was the participant exploring or idle?" stays answerable
from the CSV alone, without opening the trace database.

Practice rows carry `is_practice = 1`. `analysis/fit_psychometric.py` skips both
practice and non-`answered` rows, so the psychometric fit is unaffected by any of
this. CSVs written before the column existed load as `answered` (they only ever
contained answered trials).

### Movement trace

With `record_main_trace: true`, each constant-stimuli session also creates
`experiment/data/<session_id>_trace.sqlite3`. It records every attempt of
every **practice and main-block** trial, including attempts that end in timeout
or abort; practice attempts are tagged `is_practice = 1` with their
`practice_round`, main-block attempts have `is_practice = 0`. Cursor samples (screen px and surface mm, speed,
side, signal state, ~60 Hz) are accumulated in small in-memory batches and
SQLite writes run on a dedicated worker thread, so the real-time trial loop
never performs disk I/O. The trace is intended for the offline replay,
animation and heat-map tools; none of those run during an experiment.

`cursor_samples.signal_on` is the state **the frame asked for**, sampled together
with the position. It does not show when the output actually switched. The real
switches are in `trace_events` (no schema change; times in microseconds from the
start of the trial, taken from `time.perf_counter()`):

| `event_type` | Written | Payload |
| --- | --- | --- |
| `signal_write` | once per switch of the output | `kind` (`plan`: decided at a frame; `transition`: timed between frames; or the reason the trial ended), `on`, `planned_us`, `write_start_us`, `write_end_us`, `bar`, `edge` (`left`, `right`, `top`, `bottom`), `x_mm`, `y_mm`, `vx_mm_s`, `vy_mm_s`, `mode` |
| `signal_watchdog` | each time the watchdog switches the signal off | the same |
| `sample_stats` | once per trial | sample count, interval min / median / p95 / max in ms, number and share of samples whose position had not changed, effective horizon, the scheduler's busy-wait margin, `mode` |
| `signal_on`, `signal_off` | when the frame-level state changes | as before |
| `contact_down`, `contact_up` | when the finger touches or leaves the glass | `active_side` |

A trial is not closed until every write returned by the scheduler is in the trace.

**Contact (trace schema 3).** The touch frame reports a finger on the glass as the
left mouse button held down. `cursor_samples.contact` records it per sample, and
`contact_down` / `contact_up` events mark the changes. A device that never
reports touch would otherwise look like a finger that is never down, so contact
is only trusted once the device has reported a touch at least once in the
session. Until then `contact` is written as 1 (unknown), the signal is not
switched off at a lift, and the program prints a warning at the end of the
first trial. `sample_stats` says which case applied
(`device_reports_touch`). Recordings from schemas 1 and 2 have no `contact`
column and still load; opening one for writing adds the newer columns in place.
`analysis/exploration_maps` uses the column instead of its lift-off estimate
whenever it is present.

#### Per-trial animation (`analysis/trace_movement.py`)

```
python -m analysis.trace_movement animate TRACE [--trial N] [--include-practice] [--speed 2] [--fps 30] [--format gif|mp4] --out DIR
```

`animate` writes one GIF (or MP4, if `ffmpeg` is on the PATH) per attempt: the
two bars to scale in surface millimetres (reference blue, comparison green,
taller side named in the title), the full path so far in grey, the last second
in red, and the fingertip as a marker that is green while the electroadhesion
signal is on. The HUD shows time, speed and the side being touched.

#### Exploration maps (`analysis/exploration_maps/`)

```
python -m analysis.exploration_maps DATA [--participant ID|all] [--cell-mm 2] [--margin-mm 10] [--out DIR]
       [--normalize per-trial|none] [--freeze-mm 0.5] [--freeze-ms 200] [--jump-mm 10]
       [--keep-end-freeze] [--include-practice]
```

`DATA` is one `*_trace.sqlite3` file or a folder of them. The script only reads
the trace (opened read-only) and does not import the experiment package. For
every trial it writes one figure with two panels, **visit count | dwell time**,
in screen coordinates with both bars outlined, plus the raw grid as CSV. For
every participant it writes one aggregated figure in a **reference/comparison
frame**: trials whose reference bar was on the right are mirrored, so the
reference bar is always on the left and every comparison height is dashed on
the right. Output goes to `<data folder>/exploration_maps/<participant>/<session>/`;
`trials.csv` there lists what was excluded from each trial.

- **Visits** count entries into a cell. The straight segment between two
  samples is rasterised, so a fast movement does not skip cells and consecutive
  samples in one cell add nothing.
- **Dwell** sums real time from the sample timestamps, split across the cells a
  segment crosses in proportion to the length inside each. The dwell map plus
  the time spent outside the grid equals the in-contact time of the trial.
- **Contact.** The trace has no touch flag, so lift-off is estimated: a *freeze*
  is the finger staying within `--freeze-mm` of one point for `--freeze-ms`. A
  freeze before the first movement is excluded, as is one followed by a
  single-step jump of `--jump-mm` (the finger was in the air). A freeze that
  continues smoothly is a resting finger and counts as dwell. The freeze that
  lasts until the response is excluded unless `--keep-end-freeze` is given, and
  is reported either way. Jump segments are always excluded. If a `contact`
  column is ever added to `cursor_samples`, it is used instead. Every figure
  title names the method and the milliseconds excluded.
- **Aggregation** defaults to `--normalize per-trial`: each trial's map is
  divided by its own total before averaging, so long trials do not dominate.
  `none` gives plain sums. Only finished main-block trials are pooled.

The supervisor GUI runs this for the session that just ended (see below).

#### Render replay (`analysis/render_replay/`)

```
python -m analysis.render_replay TRACE [--out DIR] [--window 3] [--horizon-factor 1.5] [--horizon-max-ms 25]
       [--min-speed 5] [--latency-ms 0]
python -m analysis.render_replay --synthetic [--out DIR] ...
```

Runs the decision code of all three render modes (`experiment/render_logic.py`,
the same module the trial loop uses, and nothing else from the experiment
package) over recorded finger positions. For every bar edge the finger crossed
it reports how far past the edge the finger was when that mode switched the
signal, in mm (+ = after the crossing). The top edge is reported as its effect
on felt height: **upward exit, + = the bar feels taller; downward entry, - = the
bar feels shorter**. If the signal was still on when the finger came back in,
the figure is the furthest the finger got outside and the case is counted under
"never off". The table gives median, 5%, 95% and largest value per edge type,
the number of visits a mode missed altogether, and the time it had the signal on
away from any bar.

**Lifts.** A recording with a `contact` column (schema 3) is replayed with the
trial loop's own rule for a lifted finger (`TrialRenderer.frame`): the signal
goes off at the first lifted sample, and every mode starts afresh at the
touchdown, so nothing is drawn or extrapolated across the jump from the lift
point. The column already holds the trial loop's decision (contact trusted only
once the device has reported a touch), so it is used as recorded. As in the
trial loop, `timed_legacy` takes its speed at a touchdown from that jump. The
true path is split at each lift as well: an end of a visit that a lift or a
touchdown cut short gets no edge error, and time off the glass never counts as
signal on away from a bar. Above the table the tool prints the number of lifts,
the time lifted, and how many visits began with a touchdown or ended with a lift
inside a bar. Recordings without the column (schemas 1 and 2) are replayed as if
the finger never lifted.

`--synthetic` uses a built-in session with a known finger path and what a 60 Hz
loop polling a 100 Hz frame would record of it, and prints two tables: judged
as a recording would be, and judged against the finger path itself.

What the figures do not include:

- The recording's `signal_on` column is sampled in the same frame as the
  position, so it cannot show a delay. The figures come from replaying the
  decisions, with the true path taken as straight lines between samples.
- How old a position already was when it was read is invisible in a recording.
  Only the second synthetic table shows it.
- A lift the touch frame does not report is invisible too: the cursor's jump is
  then taken for a movement.
- Switches are assumed to happen at their planned instant. Scheduler lateness,
  the VISA write and the generator's own delay are not included; the last two
  need an oscilloscope.

Open the interactive replay browser with:

```bash
python -m experiment.replay
```

The left panel lists trace databases and their attempts in chronological order.
Select an attempt, then use **Play/Pause**, **Restart**, or the clickable timeline.
The left/right arrow keys seek by one second and Space toggles playback. A
deterministic `demo_replay_trace.sqlite3` recording is created automatically so
the browser can be evaluated before real participant traces are available. Pass
`--no-demo` to hide that recording.

#### Top-edge cue (offline only)

A short vibration pulse when the finger crosses a bar's top edge, for a
voice-coil actuator worn on the finger. So far only the decision exists
(`TopEdgeCue` in `render_logic.py`). It is not wired into the trial loop, and
the pulse pattern and the driver are not chosen yet. The rules:

- **Detected, not predicted.** The crossing is found on the straight step
  between two samples, and the pulse fires at the sample that shows it. A pulse
  can come late, but never for a crossing that did not happen. The interpolated
  crossing time and point are kept, so the lateness can be measured.
- **Both directions.** A downward entry and an upward exit both fire, with the
  same pattern, tagged `entering` or `leaving`.
- **Corner tolerance `c`.** The top edge counts `c` mm beyond each side of the
  bar, so a diagonal pass over a top corner fires. Leaving through a side
  without crossing the top line does not.
- **Re-arm distance `d`.** A bar that fired is disarmed until the finger is at
  least `d` mm from its extended top edge. At the first sample and at every
  touchdown, a bar is armed if and only if the finger is that far away. Each bar
  has its own state.
- **Lifts.** Nothing fires while the finger is lifted, and the step to a
  touchdown is never a crossing.
- **Jump guard `J`.** A finger-down step longer than `J` mm is taken as a lift
  and a touchdown. The touch frame can miss a short lift, and an actuator on the
  finger is felt in the air too.
- **One actuator.** Pulses are at least `min_interval` apart, whichever bar
  they are for, and one crossing gives one pulse at most. Where two extended
  edges overlap (equal heights, gap under `2c`), the crossing is the nearer
  bar's.
- The bars are the configured ones, whatever the render mode. Practice and
  catch trials get pulses as well.

Defaults: `d` = 3 mm, `c` = 1 mm, `J` = 20 mm, and `min_interval` = 100 ms,
which is a placeholder until the pulse pattern is chosen.

```
python -m analysis.render_replay --pulses TRACE [TRACE ...] [--out DIR] [--rearm 2,3,5] [--corner 0,1,2,3]
       [--jump 10,20,30] [--min-interval-ms 100]
```

Replays the cue over recorded traces for every combination of the listed
values. For each setting it prints: pulses per trial (main, catch and practice
apart); true top-edge crossings that got no pulse and why, with the path
travelled since the pulse that disarmed the bar; pulses that exist only because
of the tolerance; jump-guard hits and how many would have fired; how far past
the top line the finger was at the firing sample (median and 95%, entering and
leaving apart); pulses within 300 ms of one at the same bar; and the
entering/leaving split. `--out` writes the table and every crossing as CSV.
Recordings with and without a contact column are reported apart. Without one,
the finger is taken to be on the glass throughout, so only the jump guard
separates lifts. Not included: how old a position was when it was read, the
loop's own delay, and the driver's and actuator's delays.

Why these defaults (recordings of 14 October with contact, 158 trials):

- `J` = 20: every top crossing on a 10–20 mm step (26) went on in the same
  direction at the next step, with no lift within 10 frames, so these were fast
  strokes, not glitches. `J` = 10 would drop them all. The three longer steps
  that cross a top edge are two relocations of 190–250 mm and a 29 mm jump
  against the direction of travel. The price: such fast entries are seen late
  (95th percentile 9.9 mm below the top, against 4.0 mm with `J` = 10).
- `c` = 1: it fires for 19 passes through a corner (the finger inside the bar
  within 3 frames of the crossing), `c` = 2 for 20. Pulses with the finger not
  in the bar within those frames grow faster: 19 at `c` = 1, 32 at `c` = 2, 42
  at `c` = 3. At the smallest gap the configuration allows (3 mm), extended
  edges of 1 mm never overlap.
- `d` = 3: half the pulses fire with the finger 1.5 mm or more past the top
  line. With `d` = 2, 38% fire at least 2 mm past it, so the bar re-arms at
  once; with `d` = 3, 17% fire at least 3 mm past it. 21 of 336 true crossings
  get no pulse because that bar's last pulse disarmed it, after a median 3.5 mm
  of travel since that pulse (`d` = 2: 10, `d` = 5: 34).

### Supervisor launcher (GUI)

`experiment/gui.py` is a small Tkinter control panel for the supervisor, so
repeat sessions don't require re-typing config values or CLI flags by hand:

```
python -m experiment.gui
```

It opens pre-filled with the previously used configuration
(`experiment/configs/.last_used.yaml`, not versioned), so most sessions only
need the participant ID changed before pressing **Apply**. Apply writes the
form values out, hides the panel, and launches the participant-facing pygame
screen (`experiment.main`) as a subprocess; the panel reappears automatically
once that session ends.

After the session process has exited, the panel starts
`analysis.exploration_maps` on that session's trace in the background and
reports the result in the status line. Maps and a `<session>_maps.log` go to
`experiment/data/exploration_maps/`; a failure only leaves that log.

The **Past Recordings** tab lists saved trace databases and their attempts in
chronological order. Selecting an attempt opens Play/Pause, Restart, and
timeline controls and renders the replay directly inside the launcher window;
it does not create a separate Pygame window. Replay modules are loaded only
when that tab is opened and playback is stopped before an experiment starts.
Answered attempts show the selected side (Left or Right), prefixed by a check
mark for a correct response or a cross for an incorrect response. With
Auto-next enabled, the player waits briefly at the end of an attempt and then
continues with the next attempt in chronological order.

## Key References

- Sun et al. (2023). Investigating the minimum perceived linewidth of electroadhesion devices. *Displays*, 76, 102342.
- Holmes & Tamè (2023). Two-point discrimination. *(methodology rationale for separated bar design)*
- Tang & Beebe (1998). *(inter-bar spacing baseline)*
