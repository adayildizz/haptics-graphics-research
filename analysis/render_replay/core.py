"""Simulate the render modes on a sampled path and compare their switches with the true edge crossings."""

# Uses experiment.render_logic only: the same decision code the trial loop runs, nothing else from the package.

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Sequence
from dataclasses import astuple, dataclass
from functools import partial

from experiment import render_logic as rl
from experiment.render_logic import Rect, Sample

MATCH_WINDOW_S = 0.1


@dataclass(frozen=True)
class Params:
    window: int = 3
    horizon_factor: float = 1.5
    horizon_max_ms: float = 25.0
    min_speed_mm_s: float = 5.0
    latency_ms: float = 0.0
    nominal_interval_s: float = 1.0 / 60.0
    position_quantum_mm: float = 0.2
    kalman_accel_psd: float = 30000.0
    kalman_meas_sd_mm: float = 0.2
    kalman_confidence: float = 0.9


@dataclass(frozen=True)
class Inside:
    """One visit of the true path to a bar; a missing end means the recording, or the contact, starts or stops there."""

    bar: str
    enter: rl.EdgeCrossing | None
    exit: rl.EdgeCrossing | None


@dataclass(frozen=True)
class EdgeError:
    """How far past an edge the finger was when the mode switched (+ late, - early), in mm."""

    mode: str
    bar: str
    edge: str
    entering: bool
    t_true_s: float
    late_mm: float
    late_ms: float
    bridged: bool = False  # the signal never switched off before the finger came back in

    @property
    def category(self) -> str:
        if self.edge == "top":
            return "top edge, downward entry" if self.entering else "top edge, upward exit"
        side = "side" if self.edge in ("left", "right") else "bottom"
        return f"{side} edge, {'entry' if self.entering else 'exit'}"

    @property
    def height_effect_mm(self) -> float | None:
        """Top edge only: + means the bar feels taller than it is, - shorter."""
        if self.edge != "top":
            return None
        return -self.late_mm if self.entering else self.late_mm


@dataclass(frozen=True)
class Outcome:
    errors: list[EdgeError]
    missed: int  # visits to a bar during which the mode never had the signal on
    spurious_ms: float  # signal on with the finger nowhere near the inside of a bar
    breaks_ms: tuple[float, ...] = ()  # each time the signal went off and on again with the finger inside one bar
    inside_s: float = 0.0  # time the finger spent inside a bar, the denominator for the break rate


@dataclass(frozen=True)
class Stretch:
    """Samples ``first`` to ``last`` with the finger on the glass; whether a lift, not the recording, bounds them."""

    first: int
    last: int
    after_lift: bool
    before_lift: bool


@dataclass(frozen=True)
class Contact:
    """The finger's lifts in one or more trials, and the visits to a bar they cut short."""

    trials: int = 0
    unrecorded: int = 0  # trials whose trace has no contact column: taken as on the glass throughout
    lifts: int = 0
    lifted_s: float = 0.0  # from each first lifted sample to the next one on the glass, or to the end
    recorded_s: float = 0.0
    touch_starts: int = 0  # visits that began with a touchdown inside the bar: no entry error
    lift_ends: int = 0  # visits that ended with a lift inside the bar: no exit error

    def __add__(self, other: Contact) -> Contact:
        return Contact(*(a + b for a, b in zip(astuple(self), astuple(other))))


def stretches(count: int, down: Sequence[bool] | None) -> list[Stretch]:
    """The runs of samples with the finger on the glass; all of them when contact is unknown."""
    if down is None:
        return [Stretch(0, count - 1, False, False)]
    found: list[Stretch] = []
    first: int | None = None
    for index, finger_down in enumerate([*down, False]):
        if finger_down and first is None:
            first = index
        elif not finger_down and first is not None:
            found.append(Stretch(first, index - 1, first > 0, index < count))
            first = None
    return found


def planner_for(
    mode: str,
    nominal: Sequence[Rect],
    pixel: Sequence[Rect] | None,
    bar_width_mm: float,
    params: Params,
    previous: Sample | None = None,
):
    """A fresh planner; ``previous`` is the sample just before it starts, i.e. the last one lifted."""
    if mode == "predictive":
        return rl.PredictivePlanner(
            nominal,
            window=params.window,
            horizon_factor=params.horizon_factor,
            horizon_max_s=params.horizon_max_ms / 1000.0,
            min_speed_mm_s=params.min_speed_mm_s,
            latency_s=params.latency_ms / 1000.0,
            nominal_interval_s=params.nominal_interval_s,
            position_quantum_mm=params.position_quantum_mm,
        )
    if mode == "position":
        return rl.PositionPlanner(nominal)
    if mode == "kalman":
        return rl.KalmanPlanner(
            nominal,
            accel_psd=params.kalman_accel_psd,
            meas_sd_mm=params.kalman_meas_sd_mm,
            confidence=params.kalman_confidence,
            horizon_factor=params.horizon_factor,
            horizon_max_s=params.horizon_max_ms / 1000.0,
            min_speed_mm_s=params.min_speed_mm_s,
            latency_s=params.latency_ms / 1000.0,
            nominal_interval_s=params.nominal_interval_s,
            position_quantum_mm=params.position_quantum_mm,
        )
    if mode == "timed_legacy":
        # The legacy rule tests the pixel-rounded rectangles; fall back to nominal when they are unknown.
        planner = rl.LegacyPlanner(pixel or nominal, bar_width_mm)
        # Live, its speed always comes from the frame before (speed_mm_s in experiment/trial.py), so at a touchdown
        # it spans the jump from where the finger lifted.
        planner._previous = previous
        return planner
    raise ValueError(f"unknown render mode {mode!r}")


def submitted_plans(
    new_planner: Callable[[Sample | None], object], samples: Sequence[Sample], down: Sequence[bool] | None = None
) -> list[rl.Plan]:
    """The plan the trial loop submits at each sample: the mode's own, inside the live rule for a lifted finger."""
    # The rule of TrialRenderer.frame (experiment/trial_render.py), which this tool may not import. A lifted finger
    # gets the signal off; touching down starts the mode afresh, as the cursor sat still and has now jumped.
    planner = new_planner(None)
    submitted: list[rl.Plan] = []
    for index, sample in enumerate(samples):
        if down is not None and not down[index]:
            submitted.append(rl.Plan(sample.t_s, False, None, "lift", (), sample.x_mm, sample.y_mm))
            continue
        if down is not None and index > 0 and not down[index - 1]:
            planner = new_planner(samples[index - 1])
        submitted.append(planner.plan(sample))
    return submitted


def simulate(plans: Sequence[rl.Plan]) -> list[tuple[float, bool]]:
    """(time, on) switches an ideal scheduler would execute: each plan's state, then its transitions until the next plan."""
    on = False
    switches: list[tuple[float, bool]] = []
    for index, plan in enumerate(plans):
        if plan.on != on:
            on = plan.on
            switches.append((plan.t_s, on))
        next_t = plans[index + 1].t_s if index + 1 < len(plans) else float("inf")
        for transition in plan.transitions:
            if transition.t_s >= next_t:
                break
            if transition.on != on:
                on = transition.on
                switches.append((transition.t_s, on))
    return switches


def on_intervals(switches: Sequence[tuple[float, bool]], end_s: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    for t_s, on in switches:
        if on and start is None:
            start = t_s
        elif not on and start is not None:
            intervals.append((start, t_s))
            start = None
    if start is not None:
        intervals.append((start, end_s))
    return intervals


def true_insides(truth: Sequence[Sample], bars: Sequence[Rect]) -> list[Inside]:
    """Every visit of one unbroken true path to a bar, from the edge it came in by to the edge it left by."""
    open_visits: dict[str, rl.EdgeCrossing | None] = {}
    start_bar = rl.side_at(bars, truth[0].x_mm, truth[0].y_mm)
    if start_bar is not None:
        open_visits[start_bar] = None
    visits: list[Inside] = []
    for previous, current in zip(truth, truth[1:]):
        for crossing in rl.crossings_between(previous, current, bars):
            if crossing.entering:
                open_visits[crossing.bar] = crossing
            elif crossing.bar in open_visits:
                visits.append(Inside(crossing.bar, open_visits.pop(crossing.bar), crossing))
    visits.extend(Inside(bar, enter, None) for bar, enter in open_visits.items())
    return sorted(visits, key=lambda visit: visit.enter.t_s if visit.enter else truth[0].t_s)


class Path:
    """One unbroken true finger path, linearly interpolated between its samples; never across a lift."""

    def __init__(self, samples: Sequence[Sample]) -> None:
        self.samples = samples
        self._times = [sample.t_s for sample in samples]

    def at(self, t_s: float) -> tuple[float, float]:
        index = min(max(bisect.bisect_right(self._times, t_s), 1), len(self.samples) - 1)
        a, b = self.samples[index - 1], self.samples[index]
        fraction = 0.0 if b.t_s == a.t_s else min(max((t_s - a.t_s) / (b.t_s - a.t_s), 0.0), 1.0)
        return a.x_mm + fraction * (b.x_mm - a.x_mm), a.y_mm + fraction * (b.y_mm - a.y_mm)

    def furthest_outside(self, t0_s: float, t1_s: float, bar: Rect, edge: str) -> float:
        """Largest distance beyond ``edge`` reached between two instants."""
        lo, hi = bisect.bisect_left(self._times, t0_s), bisect.bisect_right(self._times, t1_s)
        points = [self.at(t0_s), *((s.x_mm, s.y_mm) for s in self.samples[lo:hi]), self.at(t1_s)]
        return max(_beyond(x, y, bar, edge, entering=False) for x, y in points)


def _beyond(x: float, y: float, bar: Rect, edge: str, *, entering: bool) -> float:
    """Distance past ``edge`` in the direction of travel: into the bar when entering, away from it when leaving."""
    inward = {"left": x - bar.x0, "right": bar.x1 - x, "top": y - bar.y_top, "bottom": bar.y_bottom - y}[edge]
    return inward if entering else -inward


def _span(visit: Inside, start_s: float, end_s: float) -> tuple[float, float]:
    return (visit.enter.t_s if visit.enter else start_s), (visit.exit.t_s if visit.exit else end_s)


def _match(spans: Sequence[tuple[float, float]], intervals: Sequence[tuple[float, float]]) -> list[int | None]:
    """For each true visit, the on-interval that covers it best, or a free one within the match window."""
    overlaps = [[min(b, t_out) - max(a, t_in) for a, b in intervals] for t_in, t_out in spans]
    covering = {i for row in overlaps for i, overlap in enumerate(row) if overlap > 0}
    matches: list[int | None] = []
    for row in overlaps:
        best = max(range(len(row)), key=row.__getitem__, default=None)
        if best is None or row[best] <= 0:
            near = [i for i, overlap in enumerate(row) if overlap > -MATCH_WINDOW_S and i not in covering]
            best = max(near, key=row.__getitem__, default=None)
        matches.append(best)
    return matches


def evaluate(
    mode: str,
    switches: Sequence[tuple[float, bool]],
    truth: Sequence[Sample],
    bars: Sequence[Rect],
    down: Sequence[bool] | None = None,
) -> Outcome:
    """Match each true visit to the mode's on-interval and measure the entry and exit errors, per stretch of contact."""
    errors: list[EdgeError] = []
    missed, spurious_ms, inside_s = 0, 0.0, 0.0
    breaks_ms: list[float] = []
    for stretch in stretches(len(truth), down):
        # Where the finger went between two stretches is unknown: no visit, on-interval or on-time spans a lift, and
        # an end that a lift or touchdown cut short has no edge error, like one the recording cut short.
        t0_s = truth[stretch.first].t_s if stretch.after_lift else -math.inf
        t1_s = truth[stretch.last].t_s if stretch.before_lift else math.inf
        own = [(t_s, on) for t_s, on in switches if t0_s <= t_s <= t1_s]
        outcome = _evaluate_path(mode, own, truth[stretch.first : stretch.last + 1], bars)
        errors += outcome.errors
        missed += outcome.missed
        spurious_ms += outcome.spurious_ms
        breaks_ms += outcome.breaks_ms
        inside_s += outcome.inside_s
    return Outcome(errors, missed, spurious_ms, tuple(breaks_ms), inside_s)


def _evaluate_path(
    mode: str, switches: Sequence[tuple[float, bool]], truth: Sequence[Sample], bars: Sequence[Rect]
) -> Outcome:
    """``evaluate`` on one unbroken path, with the switches made while the finger was on it."""
    path = Path(truth)
    by_name = {bar.name: bar for bar in bars}
    start_s, end_s = truth[0].t_s, truth[-1].t_s
    visits = true_insides(truth, bars)
    spans = [_span(visit, start_s, end_s) for visit in visits]
    intervals, breaks_ms = close_breaks(on_intervals(switches, end_s), spans)
    matches = _match(spans, intervals)
    errors: list[EdgeError] = []
    for index, (visit, matched) in enumerate(zip(visits, matches)):
        if matched is None:
            continue
        on_s, off_s = intervals[matched]
        bar = by_name[visit.bar]
        t_in, t_out = spans[index]
        # An on-interval shared with a neighbouring visit means the signal stayed on in between.
        shared_before = index > 0 and matches[index - 1] == matched
        shared_after = index + 1 < len(visits) and matches[index + 1] == matched
        if visit.enter is not None and not shared_before:
            late = _beyond(*path.at(on_s), bar, visit.enter.edge, entering=True)
            errors.append(EdgeError(mode, visit.bar, visit.enter.edge, True, t_in, late, 1000.0 * (on_s - t_in)))
        if visit.exit is None:
            continue
        if shared_after or off_s >= end_s:
            # Still on when the finger came back in, or when the recording or the contact ended: how far out it got.
            until_s = spans[index + 1][0] if shared_after else end_s
            reach = path.furthest_outside(t_out, until_s, bar, visit.exit.edge)
            outside_ms = 1000.0 * (until_s - t_out)
            errors.append(EdgeError(mode, visit.bar, visit.exit.edge, False, t_out, reach, outside_ms, bridged=True))
        else:
            # Late: the furthest the finger got with the signal still on. Early: how far inside it still was.
            late = (
                path.furthest_outside(t_out, off_s, bar, visit.exit.edge)
                if off_s > t_out
                else _beyond(*path.at(off_s), bar, visit.exit.edge, entering=False)
            )
            errors.append(EdgeError(mode, visit.bar, visit.exit.edge, False, t_out, late, 1000.0 * (off_s - t_out)))
    used = {matched for matched in matches if matched is not None}
    spurious_ms = 1000.0 * sum(b - a for i, (a, b) in enumerate(intervals) if i not in used)
    inside_s = sum(t_out - t_in for t_in, t_out in spans)
    return Outcome(errors, sum(matched is None for matched in matches), spurious_ms, tuple(breaks_ms), inside_s)


def close_breaks(
    intervals: Sequence[tuple[float, float]], spans: Sequence[tuple[float, float]]
) -> tuple[list[tuple[float, float]], list[float]]:
    """Join on-intervals separated by an off-time that lies wholly inside one visit, and list those off-times in ms.

    The finger was inside the bar throughout such an off-time, so it is a break in the bar, not an exit and a new
    entry: joined, the visit keeps one on-interval for its edge errors, and the re-switch-on is not counted as
    signal on away from a bar.
    """
    joined: list[tuple[float, float]] = []
    breaks_ms: list[float] = []
    for on_s, off_s in intervals:
        if joined:
            gap_from, gap_to = joined[-1][1], on_s
            if gap_to > gap_from and any(t_in <= gap_from and gap_to <= t_out for t_in, t_out in spans):
                breaks_ms.append(1000.0 * (gap_to - gap_from))
                joined[-1] = (joined[-1][0], off_s)
                continue
        joined.append((on_s, off_s))
    return joined, breaks_ms


def contact_in(samples: Sequence[Sample], down: Sequence[bool] | None, bars: Sequence[Rect]) -> Contact:
    """One trial's lifts, the time lifted, and the visits a touchdown or lift cut short."""
    recorded_s = samples[-1].t_s - samples[0].t_s
    if down is None:
        return Contact(trials=1, unrecorded=1, recorded_s=recorded_s)
    touch_starts = lift_ends = 0
    runs = stretches(len(samples), down)
    for run in runs:
        for visit in true_insides(samples[run.first : run.last + 1], bars):
            touch_starts += run.after_lift and visit.enter is None
            lift_ends += run.before_lift and visit.exit is None
    return Contact(
        trials=1,
        lifts=sum(run.before_lift for run in runs),
        lifted_s=sum(b.t_s - a.t_s for a, b, finger_down in zip(samples, samples[1:], down) if not finger_down),
        recorded_s=recorded_s,
        touch_starts=touch_starts,
        lift_ends=lift_ends,
    )


def replay(
    samples: Sequence[Sample],
    nominal: Sequence[Rect],
    pixel: Sequence[Rect] | None,
    bar_width_mm: float,
    params: Params,
    truth: Sequence[Sample] | None = None,
    down: Sequence[bool] | None = None,
) -> dict[str, Outcome]:
    """Every render mode on one trial; ``truth`` defaults to the samples themselves, joined by straight lines.

    ``down`` says per sample whether the finger was on the glass; without it, it never lifts. Contact is only known
    at the samples, so it cannot be combined with a separate truth.
    """
    if down is not None and truth is not None:
        raise ValueError("contact is known per recorded sample: judge against the samples themselves")
    return {
        mode: evaluate(
            mode,
            simulate(submitted_plans(partial(planner_for, mode, nominal, pixel, bar_width_mm, params), samples, down)),
            truth or samples,
            nominal,
            down,
        )
        for mode in rl.RENDER_MODES
    }
