"""Pure render decisions: bar geometry, when the signal should switch, and when a cue pulse fires."""

# Standard library only, and nothing from this package: analysis tools replay recordings through it.

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

EDGES = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class Rect:
    """One bar in surface mm, y down; left and top edges are inside, right and bottom outside."""

    name: str
    x0: float
    x1: float
    y_top: float
    y_bottom: float

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y_top <= y < self.y_bottom


@dataclass(frozen=True)
class Sample:
    t_s: float
    x_mm: float
    y_mm: float


@dataclass(frozen=True)
class Transition:
    t_s: float
    on: bool
    bar: str
    edge: str


@dataclass(frozen=True)
class Plan:
    """Signal state at the sample time, plus timed switches after it; the position rule plans none."""

    t_s: float
    on: bool
    bar: str | None
    edge: str
    transitions: tuple[Transition, ...]
    x_mm: float
    y_mm: float
    vx_mm_s: float = 0.0
    vy_mm_s: float = 0.0
    horizon_s: float = 0.0


@dataclass(frozen=True)
class EdgeCrossing:
    """The finger crossing one bar edge on the straight path between two samples."""

    t_s: float
    bar: str
    edge: str
    entering: bool
    x_mm: float
    y_mm: float


@dataclass(frozen=True)
class TopCrossing:
    """The finger's path crossing a bar's top line within the corner tolerance, and what the cue made of it."""

    t_s: float  # the sample that showed the crossing; a pulse fires here
    bar: str
    direction: str  # "entering": downward, into the bar's height; "leaving": upward, out of it
    t_cross_s: float  # interpolated on the straight step from the previous sample
    x_cross_mm: float
    y_cross_mm: float  # the bar's top
    late_mm: float  # beyond the top line at ``t_s`` in the direction of travel: below it entering, above it leaving
    outcome: str  # "pulse", or what stopped one: "disarmed", "interval", or "jump" when only the jump guard did
    on_jump: bool = False  # seen on a step longer than the jump guard, which is taken as a lift and a touchdown


def nominal_bars(
    surface_width_mm: float,
    surface_height_mm: float,
    inter_bar_gap_mm: float,
    bar_width_mm: float,
    left_height_mm: float,
    right_height_mm: float,
) -> tuple[Rect, Rect]:
    """Both bars at their configured size, centred on the surface with bottoms on its lower edge."""
    center = surface_width_mm / 2.0
    left_x1 = center - inter_bar_gap_mm / 2.0
    right_x0 = center + inter_bar_gap_mm / 2.0
    return (
        Rect("left", left_x1 - bar_width_mm, left_x1, surface_height_mm - left_height_mm, surface_height_mm),
        Rect("right", right_x0, right_x0 + bar_width_mm, surface_height_mm - right_height_mm, surface_height_mm),
    )


def side_at(bars: Sequence[Rect], x_mm: float, y_mm: float) -> str | None:
    for bar in bars:
        if bar.contains(x_mm, y_mm):
            return bar.name
    return None


class SampleClock:
    """Sample intervals, totalled for the per-trial log."""

    def __init__(self) -> None:
        self._all: list[float] = []
        self._last_t_s: float | None = None
        self.samples = 0
        self.repeated = 0

    def add(self, t_s: float, repeated: bool) -> None:
        if self._last_t_s is not None:
            self._all.append(t_s - self._last_t_s)
        self._last_t_s = t_s
        self.samples += 1
        self.repeated += int(repeated)

    def summary(self) -> dict[str, float | int]:
        """Interval distribution in ms and the share of samples whose position had not changed."""
        intervals = sorted(1000.0 * dt for dt in self._all)
        if not intervals:
            return {"samples": self.samples, "repeated": self.repeated, "repeated_ratio": 0.0}
        return {
            "samples": self.samples,
            "repeated": self.repeated,
            "repeated_ratio": round(self.repeated / self.samples, 4),
            "interval_min_ms": round(intervals[0], 3),
            "interval_median_ms": round(statistics.median(intervals), 3),
            "interval_p95_ms": round(intervals[min(len(intervals) - 1, int(0.95 * len(intervals)))], 3),
            "interval_max_ms": round(intervals[-1], 3),
        }


def path_inside(x: float, y: float, vx: float, vy: float, bar: Rect) -> tuple[float, str, float, str] | None:
    """(enter time, enter edge, exit time, exit edge) of a straight path through a bar, or None if it misses."""
    enter_s, enter_edge, exit_s, exit_edge = -math.inf, "", math.inf, ""
    for position, velocity, low, high, low_edge, high_edge in (
        (x, vx, bar.x0, bar.x1, "left", "right"),
        (y, vy, bar.y_top, bar.y_bottom, "top", "bottom"),
    ):
        if velocity == 0.0:
            if not low <= position < high:
                return None
            continue
        t_low, t_high = (low - position) / velocity, (high - position) / velocity
        axis_enter, axis_exit = (t_low, t_high) if velocity > 0 else (t_high, t_low)
        if axis_enter > enter_s:
            enter_s, enter_edge = axis_enter, low_edge if velocity > 0 else high_edge
        if axis_exit < exit_s:
            exit_s, exit_edge = axis_exit, high_edge if velocity > 0 else low_edge
    if enter_s >= exit_s:
        return None
    return enter_s, enter_edge, exit_s, exit_edge


def crossings_between(previous: Sample, current: Sample, bars: Sequence[Rect]) -> list[EdgeCrossing]:
    """Every bar edge crossed on the straight line from one sample to the next, in time order."""
    dt = current.t_s - previous.t_s
    dx, dy = current.x_mm - previous.x_mm, current.y_mm - previous.y_mm
    found: list[EdgeCrossing] = []
    for bar in bars:
        inside = path_inside(previous.x_mm, previous.y_mm, dx, dy, bar)
        if inside is None:
            continue
        enter_s, enter_edge, exit_s, exit_edge = inside
        for fraction, edge, entering in ((enter_s, enter_edge, True), (exit_s, exit_edge, False)):
            if 0.0 < fraction <= 1.0:
                found.append(
                    EdgeCrossing(
                        previous.t_s + fraction * dt,
                        bar.name,
                        edge,
                        entering,
                        previous.x_mm + fraction * dx,
                        previous.y_mm + fraction * dy,
                    )
                )
    return sorted(found, key=lambda crossing: crossing.t_s)


def explain_switch(
    previous: Sample | None, current: Sample, bars: Sequence[Rect], on: bool
) -> tuple[str | None, str]:
    """Which bar and edge explain the signal state at ``current``, judged from the step that led there."""
    bar = side_at(bars, current.x_mm, current.y_mm)
    if previous is None:
        return bar, "start"
    # Switching on is explained by the bar now entered; switching off by the bar just left.
    name = bar if on else side_at(bars, previous.x_mm, previous.y_mm)
    for rect in bars:
        if rect.name == name:
            step = (current.x_mm - previous.x_mm, current.y_mm - previous.y_mm)
            inside = path_inside(previous.x_mm, previous.y_mm, *step, rect)
            if inside is not None:
                return name, inside[1] if on else inside[3]
    return name or bar, ""


class ContactGate:
    """Whether the finger is on the glass, trusting contact only once the device has reported a touch."""

    def __init__(self) -> None:
        self.device_reports_touch = False

    def finger_down(self, contact: bool) -> bool:
        # A driver that never reports touch would otherwise read as a finger that is never down.
        self.device_reports_touch = self.device_reports_touch or contact
        return contact or not self.device_reports_touch


class PositionPlanner:
    """Signal on exactly while the latest sample is inside a bar; no prediction."""

    def __init__(self, bars: Sequence[Rect]) -> None:
        self.bars = tuple(bars)
        self._previous: Sample | None = None

    def plan(self, sample: Sample) -> Plan:
        on = side_at(self.bars, sample.x_mm, sample.y_mm) is not None
        bar, edge = explain_switch(self._previous, sample, self.bars, on)
        self._previous = sample
        return Plan(sample.t_s, on, bar, edge, (), sample.x_mm, sample.y_mm)


CUE_DIRECTIONS = ("entering", "leaving")


def distance_to_top(x_mm: float, y_mm: float, bar: Rect, corner_mm: float) -> float:
    """Shortest distance from a point to the bar's top edge, extended by ``corner_mm`` beyond each side."""
    dx = max(bar.x0 - corner_mm - x_mm, 0.0, x_mm - (bar.x1 + corner_mm))
    return math.hypot(dx, y_mm - bar.y_top)


def top_line_crossing(previous: Sample, current: Sample, y_top: float) -> tuple[float, bool] | None:
    """(fraction of the step, entering) where the straight step crosses the line y = ``y_top``, or None."""
    # On the line counts as below it, as a bar's top row is inside the bar.
    above_before, above_now = previous.y_mm < y_top, current.y_mm < y_top
    if above_before == above_now:
        return None
    return (y_top - previous.y_mm) / (current.y_mm - previous.y_mm), above_before


class TopEdgeCue:
    """When a vibration pulse fires: at the sample that shows the finger has crossed a bar's top edge.

    Detected on the straight step between two samples, never predicted, so a pulse can be late but is never for a
    crossing that did not happen. Downward entries and upward exits both fire, tagged with their direction. The top
    edge counts ``corner_mm`` beyond each side of the bar, so a pass over a top corner fires; leaving through a
    side without crossing the top line does not. A bar that fired stays disarmed until the finger is ``rearm_mm``
    from its extended top edge. A step longer than ``jump_mm`` is taken as a lift and a touchdown, not a crossing.
    Pulses are ``min_interval_s`` apart at least, whichever bar they are for: there is one actuator.
    """

    def __init__(
        self,
        bars: Sequence[Rect],
        *,
        rearm_mm: float = 3.0,
        corner_mm: float = 1.0,
        jump_mm: float = 20.0,
        min_interval_s: float = 0.1,
    ) -> None:
        if min_interval_s <= 0.0:
            raise ValueError("min_interval_s must be positive: one actuator gives one pulse at a time")
        self.bars = tuple(bars)
        self.rearm_mm = rearm_mm
        self.corner_mm = corner_mm
        self.jump_mm = jump_mm
        self.min_interval_s = min_interval_s
        self.armed = {bar.name: False for bar in self.bars}
        self.crossings: tuple[TopCrossing, ...] = ()  # every top crossing on the latest step, pulse or not
        self.jumped = False  # the latest step was longer than ``jump_mm``
        self._previous: Sample | None = None  # the latest sample, while the finger is down
        self._last_pulse_s = -math.inf

    def step(self, sample: Sample, down: bool = True) -> TopCrossing | None:
        """The pulse to fire at this sample, if any; ``down`` is whether the finger is on the glass."""
        previous, self._previous = self._previous, sample if down else None
        self.crossings, self.jumped = (), False
        if not down:
            return None
        if previous is None:
            self._touch_down(sample)
            return None
        self.jumped = math.hypot(sample.x_mm - previous.x_mm, sample.y_mm - previous.y_mm) > self.jump_mm
        found, pulse = [], None
        last_s = self._last_pulse_s
        for fraction, bar, entering, x_mm in self._top_crossings(previous, sample):
            if not self.armed[bar.name]:
                outcome = "disarmed"
            elif sample.t_s - last_s < self.min_interval_s:
                outcome = "interval"
            else:
                # On a jump, still note that it would have fired, for the analysis; nothing fires.
                outcome = "jump" if self.jumped else "pulse"
                last_s = sample.t_s
            late = sample.y_mm - bar.y_top if entering else bar.y_top - sample.y_mm
            crossing = TopCrossing(
                sample.t_s,
                bar.name,
                CUE_DIRECTIONS[0] if entering else CUE_DIRECTIONS[1],
                previous.t_s + fraction * (sample.t_s - previous.t_s),
                x_mm,
                bar.y_top,
                late,
                outcome,
                self.jumped,
            )
            found.append(crossing)
            if outcome == "pulse":
                pulse = crossing
                self.armed[bar.name] = False
                self._last_pulse_s = sample.t_s
        self.crossings = tuple(found)
        if self.jumped:
            self._touch_down(sample)
        else:
            for bar in self.bars:
                self.armed[bar.name] = self.armed[bar.name] or self._far(sample, bar)
        return pulse

    def _far(self, sample: Sample, bar: Rect) -> bool:
        return distance_to_top(sample.x_mm, sample.y_mm, bar, self.corner_mm) >= self.rearm_mm

    def _touch_down(self, sample: Sample) -> None:
        """At the first sample and every touchdown, armed if and only if the finger is far enough from the edge."""
        for bar in self.bars:
            self.armed[bar.name] = self._far(sample, bar)

    def _top_crossings(self, previous: Sample, current: Sample) -> list[tuple[float, Rect, bool, float]]:
        """(fraction, bar, entering, x) for every top line the step crosses within a bar's extended top edge."""
        found = []
        for y_top in dict.fromkeys(bar.y_top for bar in self.bars):
            crossed = top_line_crossing(previous, current, y_top)
            if crossed is None:
                continue
            fraction, entering = crossed
            x_mm = previous.x_mm + fraction * (current.x_mm - previous.x_mm)
            near = [b for b in self.bars if b.y_top == y_top and b.x0 - self.corner_mm <= x_mm < b.x1 + self.corner_mm]
            if near:
                # One crossing gives one pulse at most. Two extended edges overlap only on one line (equal heights,
                # gap under twice the tolerance); the crossing is then the nearer bar's.
                bar = min(near, key=lambda b: max(b.x0 - x_mm, 0.0, x_mm - b.x1))
                found.append((fraction, bar, entering, x_mm))
        return sorted(found, key=lambda crossing: crossing[0])
