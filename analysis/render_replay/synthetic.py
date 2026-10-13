"""A synthetic session with a known finger path, and what a 60 Hz loop reading a 100 Hz frame would record of it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiment import render_logic as rl
from experiment.render_logic import Rect, Sample

SURFACE_MM = (194.0, 145.0)
GAP_MM, WIDTH_MM, REFERENCE_MM = 40.0, 10.0, 10.0
LEVELS = (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)
PIXEL_MM = (249.0 / 1920.0, 187.0 / 1080.0)


@dataclass(frozen=True)
class SyntheticTrial:
    label: str
    bars: tuple[Rect, Rect]
    truth: list[Sample]  # the finger itself, every millisecond
    recorded: list[Sample]  # what the trial loop would have stored


def _path(rng: np.random.Generator, bars: tuple[Rect, Rect]) -> list[tuple[float, float, float]]:
    """(x, y, speed to reach it) waypoints: two sweeps, slow probing of each bar top, a clipped corner."""
    left, right = bars
    sweep_y = 141.0 + rng.uniform(-1.5, 1.5)
    points = [(52.0, sweep_y, 0.0), (140.0, sweep_y + rng.uniform(-1, 1), rng.uniform(80, 130))]
    points.append((54.0, sweep_y + rng.uniform(-1, 1), rng.uniform(80, 130)))
    for bar in (left, right) if rng.random() < 0.5 else (right, left):
        x = rng.uniform(bar.x0 + 2.5, bar.x1 - 2.5)
        points.append((x, bar.y_top - rng.uniform(7, 10), rng.uniform(60, 100)))
        for _ in range(2):  # down through the top edge, back up through it
            points.append((x + rng.uniform(-0.5, 0.5), bar.y_top + rng.uniform(2.0, 4.0), rng.uniform(15, 40)))
            points.append((x + rng.uniform(-0.5, 0.5), bar.y_top - rng.uniform(5.0, 8.0), rng.uniform(25, 60)))
    corner = right if rng.random() < 0.5 else left
    points.append((corner.x1 - 7.0, corner.y_top - 4.0, rng.uniform(60, 100)))
    points.append((corner.x1 + 3.0, corner.y_top + 4.0, rng.uniform(80, 120)))  # in by the top, out by the side
    return points


def _truth(rng: np.random.Generator, bars: tuple[Rect, Rect]) -> list[Sample]:
    points = _path(rng, bars)
    t, xs, ys = [0.0], [points[0][0]], [points[0][1]]
    for (x0, y0, _), (x1, y1, speed) in zip(points, points[1:]):
        steps = max(2, int(np.hypot(x1 - x0, y1 - y0) / speed * 1000.0))
        ease = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, steps + 1)[1:])  # speed rises and falls on each leg
        xs.extend(x0 + (x1 - x0) * ease)
        ys.extend(y0 + (y1 - y0) * ease)
        t.extend(t[-1] + np.arange(1, steps + 1) / 1000.0)
    tremor = np.cumsum(rng.normal(0, 0.004, (len(t), 2)), axis=0)
    tremor -= np.linspace(0, 1, len(t))[:, None] * tremor[-1]
    return [Sample(10.0 + ti, xi + dx, yi + dy) for ti, xi, yi, (dx, dy) in zip(t, xs, ys, tremor)]


def _recorded(rng: np.random.Generator, truth: list[Sample], loop_hz: float = 60.0, frame_hz: float = 100.0) -> list[Sample]:
    """The loop polls; each poll returns the IR frame's latest report, rounded to whole pixels."""
    t_true = np.array([s.t_s for s in truth])
    x_true = np.array([s.x_mm for s in truth])
    y_true = np.array([s.y_mm for s in truth])
    polls = np.arange(t_true[0] + 0.02, t_true[-1], 1.0 / loop_hz)
    polls = polls + rng.normal(0, 0.0003, len(polls))
    phase = rng.uniform(0, 1.0 / frame_hz)
    reported_at = np.floor((polls - phase) * frame_hz) / frame_hz + phase
    x = np.round(np.interp(reported_at, t_true, x_true) / PIXEL_MM[0]) * PIXEL_MM[0]
    y = np.round(np.interp(reported_at, t_true, y_true) / PIXEL_MM[1]) * PIXEL_MM[1]
    return [Sample(float(t), float(xi), float(yi)) for t, xi, yi in zip(polls, x, y)]


def session(trials_per_level: int = 4, seed: int = 20261014) -> list[SyntheticTrial]:
    """Four trials at each of the six comparison levels, reference side alternating."""
    rng = np.random.default_rng(seed)
    trials = []
    for repeat in range(trials_per_level):
        for level in LEVELS:
            reference_left = (repeat + int(level > 0)) % 2 == 0
            comparison = REFERENCE_MM * (1.0 + level)
            heights = (REFERENCE_MM, comparison) if reference_left else (comparison, REFERENCE_MM)
            bars = rl.nominal_bars(*SURFACE_MM, GAP_MM, WIDTH_MM, *heights)
            truth = _truth(rng, bars)
            label = f"level {level:+.0%} ref {'left' if reference_left else 'right'} #{repeat + 1}"
            trials.append(SyntheticTrial(label, bars, truth, _recorded(rng, truth)))
    return trials
