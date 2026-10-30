"""The top-edge cue on recorded finger paths: where its pulses would have fired, and a sweep of its parameters."""

# Uses experiment.render_logic only, like the rest of the replay: the same cue a trial loop would run.

from __future__ import annotations

import csv
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiment import render_logic as rl
from experiment.render_logic import Rect, Sample

BLOCKS = ("main", "catch", "practice")
REARM_MM = (2.0, 3.0, 5.0)
CORNER_MM = (0.0, 1.0, 2.0, 3.0)
JUMP_MM = (10.0, 20.0, 30.0)
MIN_INTERVAL_MS = 100.0
CHATTER_S = 0.3
CAVEATS = (
    "A pulse is placed at the sample that showed the crossing. How old the position already was (0-10 ms for a "
    "100 Hz frame) and the loop's own delay are not in the recording, so the finger was further on than shown.",
    "The driver's and the actuator's delays are not included; they are not known yet.",
)


@dataclass(frozen=True)
class CueTrial:
    """One recorded attempt as the cue sees it."""

    label: str
    block: str  # "main", "catch" or "practice"
    samples: Sequence[Sample]
    down: Sequence[bool] | None  # None: no contact column, taken as on the glass throughout
    bars: Sequence[Rect]  # nominal: the cue does not depend on the render mode


@dataclass(frozen=True)
class Setting:
    rearm_mm: float
    corner_mm: float
    jump_mm: float
    min_interval_ms: float = MIN_INTERVAL_MS

    def cue(self, bars: Sequence[Rect]) -> rl.TopEdgeCue:
        return rl.TopEdgeCue(
            bars,
            rearm_mm=self.rearm_mm,
            corner_mm=self.corner_mm,
            jump_mm=self.jump_mm,
            min_interval_s=self.min_interval_ms / 1000.0,
        )


@dataclass(frozen=True)
class Event:
    """One top crossing the cue saw, and what the sweep measures about it."""

    trial: str
    block: str
    crossing: rl.TopCrossing
    own_edge: bool  # on the bar's own top edge, not only within the corner tolerance
    after_pulse_mm: float | None  # disarmed by a pulse at this bar: the path from that pulse's sample to here
    after_pulse_ms: float | None
    chatter: bool  # a pulse less than CHATTER_S after the previous one at the same bar


@dataclass(frozen=True)
class TrialCues:
    events: list[Event]
    jumps: int  # finger-down steps longer than the jump guard
    jumps_crossing: int  # of which crossed a top edge, tolerance included
    jumps_firing: int  # of which would have fired without the guard


def cue_trial(trial: CueTrial, setting: Setting) -> TrialCues:
    """Steps the cue through one trial with its recorded contact and notes every top crossing it saw."""
    cue = setting.cue(trial.bars)
    bars = {bar.name: bar for bar in trial.bars}
    events: list[Event] = []
    jumps = jumps_crossing = jumps_firing = 0
    travelled = 0.0  # path length over the steps the cue took as movement
    disarmed_by: dict[str, tuple[float, float]] = {}  # bar -> (t_s, travelled) of its pulse since the last touchdown
    last_pulse_s: dict[str, float] = {}  # bar -> its latest pulse in the trial, lifts or not
    previous: Sample | None = None
    for index, sample in enumerate(trial.samples):
        down = True if trial.down is None else trial.down[index]
        pulse = cue.step(sample, down)
        if not down:
            previous = None
            continue
        if cue.jumped:
            jumps += 1
            jumps_crossing += bool(cue.crossings)
            jumps_firing += any(crossing.outcome == "jump" for crossing in cue.crossings)
        for crossing in cue.crossings:
            bar = bars[crossing.bar]
            after_mm = after_ms = None
            if crossing.outcome == "disarmed" and not crossing.on_jump and crossing.bar in disarmed_by:
                pulse_s, pulse_travelled = disarmed_by[crossing.bar]
                partial = math.hypot(crossing.x_cross_mm - previous.x_mm, crossing.y_cross_mm - previous.y_mm)
                after_mm = travelled + partial - pulse_travelled
                after_ms = 1000.0 * (crossing.t_cross_s - pulse_s)
            since_s = crossing.t_s - last_pulse_s.get(crossing.bar, -math.inf)
            chatter = crossing.outcome == "pulse" and since_s < CHATTER_S
            own_edge = bar.x0 <= crossing.x_cross_mm < bar.x1
            events.append(Event(trial.label, trial.block, crossing, own_edge, after_mm, after_ms, chatter))
        if previous is None or cue.jumped:
            disarmed_by.clear()  # a touchdown decides the arm state afresh
        else:
            travelled += math.hypot(sample.x_mm - previous.x_mm, sample.y_mm - previous.y_mm)
        if pulse is not None:
            disarmed_by[pulse.bar] = (pulse.t_s, travelled)
            last_pulse_s[pulse.bar] = pulse.t_s
        previous = sample
    return TrialCues(events, jumps, jumps_crossing, jumps_firing)


def grid(
    rearm_mm: Sequence[float] = REARM_MM,
    corner_mm: Sequence[float] = CORNER_MM,
    jump_mm: Sequence[float] = JUMP_MM,
    min_interval_ms: float = MIN_INTERVAL_MS,
) -> list[Setting]:
    return [Setting(d, c, j, min_interval_ms) for d, c, j in itertools.product(rearm_mm, corner_mm, jump_mm)]


def sweep(trials: Sequence[CueTrial], settings: Sequence[Setting]) -> list[tuple[Setting, list[TrialCues]]]:
    return [(setting, [cue_trial(trial, setting) for trial in trials]) for setting in settings]


def _median_p95(values: Sequence[float]) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    return round(float(np.median(values)), 3), round(float(np.percentile(values, 95)), 3)


def summarize(setting: Setting, trials: Sequence[CueTrial], results: Sequence[TrialCues]) -> dict[str, object]:
    """One setting on a group of trials: pulse rates by block, misses, jumps, lateness, chatter."""
    events = [event for result in results for event in result.events]
    pulses = [event for event in events if event.crossing.outcome == "pulse"]
    # True top-edge crossings: on a bar's own top edge, on a step the cue took as movement.
    true = [event for event in events if event.own_edge and not event.crossing.on_jump]
    disarmed = [event for event in true if event.crossing.outcome == "disarmed"]
    after = [event for event in disarmed if event.after_pulse_mm is not None]
    row: dict[str, object] = {
        "rearm_mm": setting.rearm_mm,
        "corner_mm": setting.corner_mm,
        "jump_mm": setting.jump_mm,
        "min_interval_ms": setting.min_interval_ms,
    }
    for block in BLOCKS:
        chosen = [result for trial, result in zip(trials, results) if trial.block == block]
        fired = sum(event.crossing.outcome == "pulse" for result in chosen for event in result.events)
        row[f"trials_{block}"] = len(chosen)
        row[f"pulses_per_trial_{block}"] = round(fired / len(chosen), 2) if chosen else ""
    row["top_crossings"] = len(true)
    row["top_crossings_pulsed"] = sum(event.crossing.outcome == "pulse" for event in true)
    row["missed_disarmed_after_pulse"] = len(after)
    path = _median_p95([event.after_pulse_mm for event in after])
    row["after_pulse_path_median_mm"], row["after_pulse_path_p95_mm"] = path
    row["after_pulse_median_ms"] = _median_p95([event.after_pulse_ms for event in after])[0]
    row["missed_disarmed_at_touchdown"] = len(disarmed) - len(after)
    row["missed_interval"] = sum(event.crossing.outcome == "interval" for event in true)
    row["pulses_corner_only"] = sum(not event.own_edge for event in pulses)
    row["jumps"] = sum(result.jumps for result in results)
    row["jumps_crossing"] = sum(result.jumps_crossing for result in results)
    row["jumps_would_fire"] = sum(result.jumps_firing for result in results)
    for direction in rl.CUE_DIRECTIONS:
        late = [event.crossing.late_mm for event in pulses if event.crossing.direction == direction]
        row[f"pulses_{direction}"] = len(late)
        row[f"late_{direction}_median_mm"], row[f"late_{direction}_p95_mm"] = _median_p95(late)
    row["chatter"] = sum(event.chatter for event in pulses)
    return row


def _cell(value: object, spec: str, width: int) -> str:
    return f"{value:{spec}}".rjust(width) if value != "" else "-".rjust(width)


def format_sweep(title: str, rows: Sequence[dict[str, object]]) -> str:
    lines = [
        title,
        f"  d re-arm, c corner tolerance, J jump guard, in mm; minimum interval {rows[0]['min_interval_ms']:g} ms",
        "  main, catch, pract: pulses per trial in that block. Every other column counts all blocks together.",
        "  top: crossings of a bar's own top edge, jumps excluded; fired: the share that got a pulse",
        "  missed: disarm, the bar's last pulse had disarmed it (path: median mm travelled since that pulse);",
        "    touch, it was disarmed at a touchdown; intvl, inside the minimum interval",
        "  c only: pulses on the tolerance alone, beyond the bar's own top edge",
        "  jumps: steps longer than J; cross: those crossing a top edge; fire: those that would have fired",
        "  entering, leaving: mm past the top line at the firing sample (below it, above it), median/95%",
        f"  chatter: pulses less than {1000 * CHATTER_S:.0f} ms after one at the same bar; in:out, entering:leaving",
        "      d    c    J   main catch pract    top fired  disarm  path touch intvl c only  jumps cross fire"
        "   entering    leaving chatter    in:out",
    ]
    for row in rows:
        true = row["top_crossings"]
        share = f"{row['top_crossings_pulsed'] / true:.0%}" if true else "-"
        late = [
            f"{_cell(row[f'late_{d}_median_mm'], '.1f', 4)}/{_cell(row[f'late_{d}_p95_mm'], '.1f', 4).strip():<4s}"
            for d in rl.CUE_DIRECTIONS
        ]
        lines.append(
            f"  {row['rearm_mm']:5g} {row['corner_mm']:4g} {row['jump_mm']:4g}"
            + "".join(_cell(row[f"pulses_per_trial_{block}"], ".1f", 6) for block in BLOCKS)
            + f" {true:6d} {share:>5s} {row['missed_disarmed_after_pulse']:7d}"
            + _cell(row["after_pulse_path_median_mm"], ".1f", 6)
            + f" {row['missed_disarmed_at_touchdown']:5d} {row['missed_interval']:5d} {row['pulses_corner_only']:6d}"
            + f" {row['jumps']:6d} {row['jumps_crossing']:5d} {row['jumps_would_fire']:4d}"
            + f"  {late[0]:>9s}  {late[1]:>9s} {row['chatter']:7d} {row['pulses_entering']:5d}:{row['pulses_leaving']}"
        )
    return "\n".join(lines)


def write_csvs(
    out_dir: Path,
    name: str,
    swept: Sequence[tuple[Setting, list[TrialCues]]],
    rows: Sequence[dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}_sweep.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / f"{name}_events.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["rearm_mm", "corner_mm", "jump_mm", "min_interval_ms", "trial", "block", "t_s", "bar", "direction",
             "t_cross_s", "x_cross_mm", "y_cross_mm", "late_mm", "outcome", "on_jump", "own_edge", "after_pulse_mm",
             "after_pulse_ms", "chatter"]
        )  # fmt: skip
        for setting, results in swept:
            for result in results:
                for e in result.events:
                    c = e.crossing
                    writer.writerow(
                        [setting.rearm_mm, setting.corner_mm, setting.jump_mm, setting.min_interval_ms, e.trial,
                         e.block, f"{c.t_s:.6f}", c.bar, c.direction, f"{c.t_cross_s:.6f}", f"{c.x_cross_mm:.3f}",
                         f"{c.y_cross_mm:.3f}", f"{c.late_mm:.3f}", c.outcome, int(c.on_jump), int(e.own_edge),
                         "" if e.after_pulse_mm is None else f"{e.after_pulse_mm:.3f}",
                         "" if e.after_pulse_ms is None else f"{e.after_pulse_ms:.1f}", int(e.chatter)]
                    )  # fmt: skip
