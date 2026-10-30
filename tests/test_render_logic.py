from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiment import render_logic as rl
from experiment.render_logic import PositionPlanner, PredictivePlanner, Sample

HZ = 60.0
# Default design on the 194 x 145 mm surface: bars 67-77 and 117-127 mm, baseline 145, reference 10 mm tall.
LEFT, RIGHT = BARS = rl.nominal_bars(194.0, 145.0, 40.0, 10.0, 10.0, 12.0)


def sampled(x0: float, y0: float, vx: float, vy: float, n: int, hz: float = HZ, t0: float = 5.0) -> list[Sample]:
    """A finger moving at constant velocity, seen at ``hz``."""
    return [Sample(t0 + k / hz, x0 + vx * k / hz, y0 + vy * k / hz) for k in range(n)]


def run(planner, samples) -> list[rl.Plan]:
    return [planner.plan(sample) for sample in samples]


def switch_times(plans: list[rl.Plan]) -> list[tuple[float, bool, str, str]]:
    """What an ideal scheduler would execute: each plan's state now, then its transitions until the next plan."""
    on, out = False, []
    for plan, following in zip(plans, plans[1:] + [None]):
        if plan.on != on:
            on = plan.on
            out.append((plan.t_s, on, plan.bar or "", plan.edge))
        for transition in plan.transitions:
            if following is not None and transition.t_s >= following.t_s:
                break
            if transition.on != on:
                on = transition.on
                out.append((transition.t_s, on, transition.bar, transition.edge))
    return out


def test_nominal_bars_use_the_configured_sizes():
    assert (LEFT.x0, LEFT.x1, LEFT.y_top, LEFT.y_bottom) == (67.0, 77.0, 135.0, 145.0)
    assert (RIGHT.x0, RIGHT.x1, RIGHT.y_top, RIGHT.y_bottom) == (117.0, 127.0, 133.0, 145.0)
    assert rl.side_at(BARS, 67.0, 135.0) == "left" and rl.side_at(BARS, 77.0, 140.0) is None
    assert rl.side_at(BARS, 120.0, 133.5) == "right" and rl.side_at(BARS, 120.0, 145.0) is None


@pytest.mark.parametrize("screen", [(1920, 1080), (1512, 982), (1280, 720)])
@pytest.mark.parametrize("heights", [(10.0, 12.0), (7.0, 10.0), (10.0, 13.0)])
def test_pixel_bars_are_the_rectangles_the_live_layout_uses(screen, heights):
    import pygame

    from experiment import display
    from experiment.calibration import make_configured_ir_frame_calibration

    cal = make_configured_ir_frame_calibration(screen)
    layout = display.make_trial_layout(None, cal, 10.0, heights[0], heights[1], False, 40.0)
    bars = rl.pixel_bars(
        cal.active_left_px, cal.active_top_px, cal.active_width_px, cal.active_height_px,
        cal.px_per_mm_x, cal.px_per_mm_y, 40.0, 10.0, heights[0], heights[1],
    )  # fmt: skip
    rng = np.random.default_rng(0)
    rects = {"left": layout.left_bar, "right": layout.right_bar}
    edge_pixels = [(r.left + dx, r.top + dy) for r in rects.values() for dx in (-1, 0, r.width - 1, r.width) for dy in (-1, 0, r.height - 1, r.height)]
    random_pixels = [(int(x), int(y)) for x, y in zip(rng.integers(0, screen[0], 4000), rng.integers(0, screen[1], 4000))]
    for px, py in edge_pixels + random_pixels:
        expected = next((name for name, rect in rects.items() if rect.collidepoint((px, py))), None)
        x_mm = (px - cal.active_left_px) / cal.px_per_mm_x
        y_mm = (py - cal.active_top_px) / cal.px_per_mm_y
        assert rl.side_at(bars, x_mm, y_mm) == expected, (px, py)
    assert isinstance(layout.left_bar, pygame.Rect)


def test_velocity_is_exact_for_straight_motion_and_two_point_speed_keeps_its_old_formula():
    samples = sampled(10.0, 20.0, 80.0, -30.0, 3)
    assert rl.estimate_velocity(samples) == pytest.approx((80.0, -30.0))
    assert rl.estimate_velocity(samples[:1]) == (0.0, 0.0)
    assert rl.two_point_speed(samples[0], samples[1]) == pytest.approx(math.hypot(80.0, 30.0))
    assert rl.two_point_speed(samples[0], samples[0]) == 0.0


def test_repeated_positions_do_not_dilute_the_velocity_estimate():
    estimator = rl.VelocityEstimator(window=3, stale_s=0.040)
    moving = sampled(0.0, 0.0, 60.0, 0.0, 3)
    for sample in moving:
        vx, _, repeated = estimator.add(sample)
        assert not repeated
    assert vx == pytest.approx(60.0)
    # The same position read again one frame later: no new information, the estimate stands.
    vx, vy, repeated = estimator.add(Sample(moving[-1].t_s + 1 / HZ, moving[-1].x_mm, moving[-1].y_mm))
    assert repeated and (vx, vy) == pytest.approx((60.0, 0.0))


def test_velocity_drops_to_zero_when_the_position_stays_put_and_restarts_from_the_last_frame():
    estimator = rl.VelocityEstimator(window=3, stale_s=0.040)
    for sample in sampled(0.0, 0.0, 60.0, 0.0, 3):
        estimator.add(sample)
    t, x = 5.0 + 2 / HZ, 2.0
    for k in range(1, 30):  # half a second without moving
        vx, vy, repeated = estimator.add(Sample(t + k / HZ, x, 0.0))
    assert repeated and (vx, vy) == (0.0, 0.0)
    # First movement after the pause: the speed comes from the last frame, not from before the pause.
    vx, _, repeated = estimator.add(Sample(t + 30 / HZ, x + 1.0, 0.0))
    assert not repeated and vx == pytest.approx(1.0 * HZ)


def test_sample_clock_reports_intervals_and_repeats():
    clock = rl.SampleClock(nominal_interval_s=1 / HZ)
    assert clock.median_s() == pytest.approx(1 / HZ)
    for k, repeated in enumerate([False, False, True, False, True]):
        clock.add(1.0 + 0.020 * k, repeated)
    summary = clock.summary()
    assert clock.median_s() == pytest.approx(0.020)
    assert summary["samples"] == 5 and summary["repeated"] == 2 and summary["repeated_ratio"] == 0.4
    assert summary["interval_median_ms"] == pytest.approx(20.0) and summary["interval_max_ms"] == pytest.approx(20.0)
    assert rl.horizon_s(0.020, 1.5, 0.025) == 0.025 and rl.horizon_s(0.010, 1.5, 0.025) == pytest.approx(0.015)


@pytest.mark.parametrize(
    "start, velocity, expected",
    [
        ((60.0, 140.0), (100.0, 0.0), (0.07, "left", 0.17, "right")),
        ((84.0, 140.0), (-100.0, 0.0), (0.07, "right", 0.17, "left")),
        ((72.0, 125.0), (0.0, 50.0), (0.2, "top", 0.4, "bottom")),
        ((72.0, 150.0), (0.0, -50.0), (0.1, "bottom", 0.3, "top")),
        ((72.0, 140.0), (0.0, -50.0), (-0.1, "bottom", 0.1, "top")),
        ((74.0, 133.0), (40.0, 40.0), (0.05, "top", 0.075, "right")),
    ],
)
def test_path_inside_names_the_edge_from_every_direction(start, velocity, expected):
    enter_s, enter_edge, exit_s, exit_edge = rl.path_inside(*start, *velocity, LEFT)
    assert (enter_edge, exit_edge) == (expected[1], expected[3])
    assert (enter_s, exit_s) == pytest.approx((expected[0], expected[2]))


def test_path_inside_handles_misses_and_a_finger_that_is_not_moving():
    assert rl.path_inside(60.0, 130.0, 100.0, 0.0, LEFT) is None  # passes above the bar
    assert rl.path_inside(75.0, 133.0, 40.0, 40.0, LEFT) is None  # only touches the corner
    assert rl.path_inside(60.0, 140.0, 0.0, 0.0, LEFT) is None  # resting outside
    enter_s, _, exit_s, _ = rl.path_inside(72.0, 140.0, 0.0, 0.0, LEFT)  # resting inside
    assert enter_s == -math.inf and exit_s == math.inf


def test_crossings_between_two_samples_are_ordered_and_located():
    crossings = rl.crossings_between(Sample(1.0, 60.0, 140.0), Sample(1.2, 80.0, 140.0), BARS)
    assert [(c.bar, c.edge, c.entering) for c in crossings] == [("left", "left", True), ("left", "right", False)]
    assert [c.t_s for c in crossings] == pytest.approx([1.07, 1.17])
    assert [c.x_mm for c in crossings] == pytest.approx([67.0, 77.0])
    assert rl.crossings_between(Sample(1.0, 60.0, 140.0), Sample(1.2, 60.0, 140.0), BARS) == []


def test_horizontal_pass_switches_on_the_left_edge_and_off_on_the_right_edge_at_the_true_instants():
    speed = 100.0
    plans = run(PredictivePlanner(BARS), sampled(55.0, 140.0, speed, 0.0, 30))
    switches = switch_times(plans)
    assert [(on, bar, edge) for _, on, bar, edge in switches] == [(True, "left", "left"), (False, "left", "right")]
    assert switches[0][0] == pytest.approx(5.0 + (67.0 - 55.0) / speed, abs=1e-9)
    assert switches[1][0] == pytest.approx(5.0 + (77.0 - 55.0) / speed, abs=1e-9)


def test_upward_exit_switches_off_exactly_at_the_top_edge():
    plans = run(PredictivePlanner(BARS), sampled(72.0, 143.0, 0.0, -40.0, 30))
    switches = switch_times(plans)
    assert [(on, edge) for _, on, _, edge in switches] == [(True, "start"), (False, "top")]
    assert switches[1][0] == pytest.approx(5.0 + (143.0 - 135.0) / 40.0, abs=1e-9)


def test_downward_entry_switches_on_exactly_at_the_top_edge():
    plans = run(PredictivePlanner(BARS), sampled(122.0, 125.0, 0.0, 30.0, 30))
    switches = switch_times(plans)
    assert [(on, bar, edge) for _, on, bar, edge in switches] == [(True, "right", "top")]
    assert switches[0][0] == pytest.approx(5.0 + (133.0 - 125.0) / 30.0, abs=1e-9)


def test_corner_cut_gets_both_switches_inside_one_sample_interval():
    # Clips the top-right corner of the left bar: in through the top, out through the right, 5 ms apart.
    vx = vy = 100.0
    samples = sampled(76.5 - vx * 0.105, 135.0 - vy * 0.105, vx, vy, 12)
    plans = run(PredictivePlanner(BARS), samples)
    both = [plan for plan in plans if len(plan.transitions) == 2]
    assert both and [(t.on, t.edge) for t in both[-1].transitions] == [(True, "top"), (False, "right")]
    assert both[-1].transitions[0].t_s == pytest.approx(5.105, abs=1e-9)
    assert both[-1].transitions[1].t_s == pytest.approx(5.110, abs=1e-9)
    # A position-only rule never sees the finger inside the bar.
    assert not any(plan.on for plan in run(PositionPlanner(BARS), samples))


def test_stopping_inside_a_bar_keeps_the_signal_on_without_predicting():
    moving = sampled(60.0, 140.0, 100.0, 0.0, 8)  # ends at x = 71.67, inside
    resting = [Sample(moving[-1].t_s + k / HZ, moving[-1].x_mm, 140.0) for k in range(1, 40)]
    plans = run(PredictivePlanner(BARS), moving + resting)
    assert all(plan.on for plan in plans[-30:])
    assert all(not plan.transitions and (plan.vx_mm_s, plan.vy_mm_s) == (0.0, 0.0) for plan in plans[-30:])


def test_below_the_speed_threshold_the_sample_decides_and_nothing_is_scheduled():
    plans = run(PredictivePlanner(BARS, min_speed_mm_s=5.0), sampled(66.0, 140.0, 3.0, 0.0, 60))
    assert all(not plan.transitions for plan in plans)
    assert [plan.on for plan in plans] == [rl.side_at(BARS, 66.0 + 3.0 * k / HZ, 140.0) is not None for k in range(60)]


def test_noisy_positions_still_put_the_switches_close_to_the_edges():
    rng = np.random.default_rng(3)
    speed, errors = 100.0, []
    for _ in range(200):
        start = 55.0 - rng.uniform(0, 1.6)
        samples = [
            Sample(s.t_s, s.x_mm + rng.normal(0, 0.05), s.y_mm + rng.normal(0, 0.05))
            for s in sampled(start, 140.0, speed, 0.0, 30)
        ]
        on_times = [t for t, on, _, _ in switch_times(run(PredictivePlanner(BARS), samples)) if on]
        errors.append(start + speed * (on_times[0] - 5.0) - 67.0)
    assert abs(np.median(errors)) < 0.05 and np.percentile(np.abs(errors), 95) < 0.35


def test_nothing_is_ever_scheduled_beyond_the_horizon():
    rng = np.random.default_rng(1)
    planner = PredictivePlanner(BARS, horizon_max_s=0.025)
    t, x, y = 0.0, 60.0, 140.0
    for _ in range(2000):
        t += rng.uniform(0.012, 0.022)
        x, y = x + rng.normal(0, 1.5), y + rng.normal(0, 1.0)
        plan = planner.plan(Sample(t, x, y))
        assert plan.horizon_s <= 0.025
        assert all(plan.t_s < transition.t_s <= plan.t_s + plan.horizon_s + 1e-12 for transition in plan.transitions)


def test_position_planner_follows_the_sample_and_labels_the_edge():
    plans = run(PositionPlanner(BARS), sampled(60.0, 140.0, 100.0, 0.0, 16))
    switches = switch_times(plans)
    assert [(on, edge) for _, on, _, edge in switches] == [(True, "left"), (False, "right")]
    # It can only react at a sample, so it is late by up to one interval.
    assert 0.0 <= switches[0][0] - (5.0 + 0.07) < 1 / HZ
    assert all(not plan.transitions for plan in plans)


def test_legacy_timer_keeps_the_signal_on_after_a_quick_exit():
    timer = rl.LegacyTimer(bar_width_mm=10.0)
    assert timer.step(10.00, None, 50.0) is False
    assert timer.step(10.02, "left", 20.0) is True  # entry at 20 mm/s: 10 mm / 20 mm/s = 0.5 s
    assert timer.commanded_s == pytest.approx(0.5)
    assert timer.step(10.10, None, 20.0) is True  # outside, but the timer is still running
    assert timer.step(10.51, None, 20.0) is True
    assert timer.step(10.53, None, 20.0) is False
    assert rl.LegacyTimer(10.0).step(1.0, "left", 0.0) and rl.LegacyTimer(10.0, max_duration_s=2.0).commanded_s == 0.0


def _polled_edge_errors(latency_ms: float, speed: float = 100.0, n: int = 4000) -> np.ndarray:
    """Edge error (mm, + = late) when a 100 Hz position is read by a 60 Hz loop that cannot know its age."""
    rng = np.random.default_rng(11)
    pixel = 249.0 / 1920.0
    phase = rng.uniform(0, 0.010)
    times = np.cumsum(1 / HZ + rng.normal(0, 0.0003, n))
    held = np.floor((times - phase) / 0.010) * 0.010 + phase  # when the position being read was measured
    xs = np.round(speed * held / pixel) * pixel
    errors = []
    planner = PredictivePlanner(BARS, latency_s=latency_ms / 1000.0)
    plans = [planner.plan(Sample(t, x, 140.0)) for t, x in zip(times, xs)]
    for k in range(5, n - 1):
        plan, edge = plans[k], xs[k] + rng.uniform(0, 1) * speed / HZ
        if abs(plan.vx_mm_s) < 5.0:
            continue
        # Where the planner would time a crossing of ``edge``, against where the finger really is then.
        t_switch = plan.t_s + (edge - (plan.x_mm + plan.vx_mm_s * planner.latency_s)) / plan.vx_mm_s
        if plan.t_s <= t_switch <= plan.t_s + plan.horizon_s:
            errors.append(speed * t_switch - edge)
    return np.array(errors)


def test_latency_compensation_removes_the_one_sided_lag_of_polled_positions():
    uncompensated = _polled_edge_errors(0.0)
    compensated = _polled_edge_errors(5.0)
    # Read 0-10 ms after it was measured, the position lags by about 5 ms of travel: half a millimetre.
    assert 0.4 < np.median(uncompensated) < 0.8
    assert abs(np.median(compensated)) < 0.15
    assert PredictivePlanner(BARS).latency_s == 0.0


def test_latency_compensation_is_exact_for_a_known_constant_delay():
    speed, delay = 100.0, 0.012
    samples = [Sample(s.t_s + delay, s.x_mm, s.y_mm) for s in sampled(55.0, 140.0, speed, 0.0, 30)]
    on_times = [t for t, on, _, _ in switch_times(run(PredictivePlanner(BARS, latency_s=delay), samples)) if on]
    assert on_times[0] == pytest.approx(5.0 + (67.0 - 55.0) / speed, abs=1e-9)


def test_module_loads_nothing_from_the_package_and_no_pygame_or_visa():
    code = (
        "import sys, experiment.render_logic\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('pygame', 'pyvisa', 'numpy')"
        " or (m.startswith('experiment.') and m != 'experiment.render_logic')]\n"
        "print(bad); raise SystemExit(1 if bad else 0)"
    )
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


# Top-edge cue: a pulse at the sample that shows the finger crossed a bar's top edge.
NARROW = rl.nominal_bars(194.0, 145.0, 3.0, 10.0, 10.0, 12.0)  # left 85.5-95.5 top 135, right 98.5-108.5 top 133
LEVEL = rl.nominal_bars(194.0, 145.0, 3.0, 10.0, 10.0, 10.0)  # the same gap, both tops at 135


def cue_run(cue, points, hz=10.0, down=None):
    """Steps the cue through (x, y) points seen at ``hz``: the pulse and the top crossings at each frame."""
    pulses, crossings = {}, {}
    for frame, (x, y) in enumerate(points):
        pulse = cue.step(Sample(frame / hz, x, y), True if down is None else down[frame])
        if pulse is not None:
            pulses[frame] = pulse
        if cue.crossings:
            crossings[frame] = cue.crossings
    return pulses, crossings


def outcomes(crossings) -> dict[int, list[tuple[str, str]]]:
    return {frame: [(c.bar, c.outcome) for c in seen] for frame, seen in crossings.items()}


def test_top_edge_cue_fires_on_the_way_in_and_out_and_tags_the_direction():
    # Down through the left bar's top (135 mm), 5 mm below it, back up and out.
    points = [(72, 130), (72, 133), (72, 136), (72, 140), (72, 138), (72, 134), (72, 128)]
    pulses, crossings = cue_run(rl.TopEdgeCue(BARS, rearm_mm=3.0), points)
    tags = {frame: (p.bar, p.direction) for frame, p in pulses.items()}
    assert tags == {2: ("left", "entering"), 5: ("left", "leaving")}
    assert all(seen == (pulses[frame],) for frame, seen in crossings.items())
    entering, leaving = pulses[2], pulses[5]
    # Fired at the sample that showed the crossing; the crossing itself is interpolated on the step.
    assert entering.t_s == pytest.approx(0.2) and entering.t_cross_s == pytest.approx(0.1 + 0.1 * 2 / 3)
    assert (entering.x_cross_mm, entering.y_cross_mm, entering.late_mm) == (72.0, 135.0, 1.0)  # 1 mm below the top
    assert leaving.t_s == pytest.approx(0.5) and leaving.t_cross_s == pytest.approx(0.475)
    assert leaving.late_mm == 1.0 and leaving.outcome == "pulse" and not leaving.on_jump  # 1 mm above the top
    cue = rl.TopEdgeCue(BARS)  # the defaults the sweep on recorded traces supports
    assert (cue.rearm_mm, cue.corner_mm, cue.jump_mm, cue.min_interval_s) == (3.0, 1.0, 20.0, 0.1)


def test_a_sample_on_the_top_line_counts_as_below_it_so_each_crossing_is_seen_once():
    points = [(72, 131), (72, 135), (72, 139), (72, 135), (72, 131)]
    pulses, crossings = cue_run(rl.TopEdgeCue(BARS, rearm_mm=3.0), points, hz=5.0)
    assert {frame: (p.direction, p.late_mm) for frame, p in pulses.items()} == {
        1: ("entering", 0.0),
        4: ("leaving", 4.0),
    }
    assert set(crossings) == {1, 4}


def test_a_pass_over_a_top_corner_fires_within_the_tolerance_and_a_side_exit_does_not():
    # Diagonally down past the left bar's top-right corner (77, 135): it crosses the top line at x = 78, 1 mm out.
    corner = [(73, 130), (75, 132), (77, 134), (79, 136), (81, 138)]
    pulses, crossings = cue_run(rl.TopEdgeCue(BARS, corner_mm=2.0), corner)
    assert list(pulses) == [3] and pulses[3].bar == "left" and pulses[3].direction == "entering"
    assert pulses[3].x_cross_mm == pytest.approx(78.0)
    for tolerance in (0.0, 0.5):
        assert cue_run(rl.TopEdgeCue(BARS, corner_mm=tolerance), corner) == ({}, {})

    # Out of the bar through its right side and up past the top line at x = 81, 4 mm beyond the side.
    side_exit = [(74, 142), (76, 140), (78, 138), (80, 136), (82, 134), (84, 132)]
    assert cue_run(rl.TopEdgeCue(BARS, corner_mm=2.0), side_exit) == ({}, {})
    pulses, _ = cue_run(rl.TopEdgeCue(BARS, corner_mm=4.5), side_exit)
    assert [(p.bar, p.direction) for p in pulses.values()] == [("left", "leaving")]
    # Straight out through the side: no top line crossed at all.
    assert cue_run(rl.TopEdgeCue(BARS, corner_mm=3.0), [(74, 140), (78, 140), (82, 140), (86, 140)]) == ({}, {})


def test_distance_to_the_extended_top_edge():
    assert rl.distance_to_top(72.0, 131.0, LEFT, 2.0) == 4.0
    assert rl.distance_to_top(82.0, 135.0, LEFT, 2.0) == 3.0  # 3 mm beyond the extended end at 79
    assert rl.distance_to_top(82.0, 139.0, LEFT, 2.0) == 5.0
    assert rl.distance_to_top(66.0, 131.0, LEFT, 2.0) == 4.0  # above the extension on the left


def test_a_bar_rearms_exactly_at_the_rearm_distance():
    points = [(72, 130), (72, 136), (72, 137.99), (72, 134.5), (72, 132.01), (72, 136), (72, 138.0), (72, 134)]
    _, crossings = cue_run(rl.TopEdgeCue(BARS, rearm_mm=3.0), points, hz=5.0)
    # 2.99 mm away is not enough; exactly 3 mm is.
    assert outcomes(crossings) == {
        1: [("left", "pulse")],
        3: [("left", "disarmed")],
        5: [("left", "disarmed")],
        7: [("left", "pulse")],
    }


def test_each_bar_is_armed_on_its_own():
    cue = rl.TopEdgeCue(NARROW, rearm_mm=4.0, corner_mm=0.0)
    # Into the left bar, beside the right bar's left side and up through its top (133 mm) at x = 99.3.
    points = [(90, 130), (90, 136), (98, 137), (100, 131)]
    armed = []
    for frame, (x, y) in enumerate(points):
        cue.step(Sample(frame / 5.0, x, y))
        armed.append(dict(cue.armed))
        if frame == 3:
            assert [(c.bar, c.direction, c.outcome) for c in cue.crossings] == [("right", "leaving", "pulse")]
    assert armed[1] == {"left": False, "right": True}  # the left bar fired; the right one is untouched
    assert armed[2] == {"left": False, "right": True}  # 3.2 mm from the left bar's top: still disarmed
    assert armed[3] == {"left": True, "right": False}  # the right bar fired while the left one was disarmed


def test_a_touchdown_is_not_a_crossing_and_decides_whether_the_bar_is_armed():
    # Lifted above the left bar, the cursor sits still; the touchdown lands below the top, then the finger goes up.
    down = [True, False, False, True, True, True, True]
    for landing_y, way_up in ((137.0, "disarmed"), (139.0, "pulse")):  # 2 and 4 mm below the top
        points = [(72, 130), (72, 130), (72, 130), (72, landing_y), (72, 133), (72, 128), (72, 136)]
        _, crossings = cue_run(rl.TopEdgeCue(BARS, rearm_mm=3.0), points, down=down)
        # The jump from the lift point crosses the top line at frame 3, but is not a crossing.
        assert outcomes(crossings) == {4: [("left", way_up)], 6: [("left", "pulse")]}
    # Nothing while lifted, even if the cursor were to move across the top.
    pulses, crossings = cue_run(rl.TopEdgeCue(BARS), [(72, 130), (72, 137), (72, 130)], down=[False, False, False])
    assert (pulses, crossings) == ({}, {})
    # The first sample of a trial is a touchdown too: 1 mm below the top, so the way up does not fire.
    _, crossings = cue_run(rl.TopEdgeCue(BARS, rearm_mm=3.0), [(72, 136), (72, 133)])
    assert outcomes(crossings) == {1: [("left", "disarmed")]}


def test_a_step_longer_than_the_jump_guard_is_a_lift_and_a_touchdown():
    jump = [(72, 120), (72, 136), (72, 133)]  # 16 mm in one frame, landing 1 mm below the top
    pulses, crossings = cue_run(rl.TopEdgeCue(BARS, rearm_mm=3.0, jump_mm=10.0), jump)
    assert pulses == {} and outcomes(crossings) == {1: [("left", "jump")], 2: [("left", "disarmed")]}
    assert crossings[1][0].on_jump and not crossings[2][0].on_jump
    assert list(cue_run(rl.TopEdgeCue(BARS, rearm_mm=3.0, jump_mm=20.0), jump)[0]) == [1]
    # Landing 5 mm below the top: armed at once, as after a touchdown.
    pulses, _ = cue_run(rl.TopEdgeCue(BARS, rearm_mm=3.0, jump_mm=10.0), [(72, 120), (72, 140), (72, 133)])
    assert {frame: p.direction for frame, p in pulses.items()} == {2: "leaving"}
    # Exactly the guard's length is still a step; a jump the bar would not have fired for keeps that reason.
    assert list(cue_run(rl.TopEdgeCue(BARS, jump_mm=10.0), [(72, 128), (72, 138)])[0]) == [1]
    cue = rl.TopEdgeCue(BARS, rearm_mm=3.0, jump_mm=10.0)
    _, crossings = cue_run(cue, [(72, 134), (72, 150)])
    assert outcomes(crossings) == {1: [("left", "disarmed")]} and crossings[1][0].on_jump and cue.jumped


def test_pulses_keep_the_minimum_interval_across_both_bars():
    hz, interval = 64.0, 0.125  # exact in binary: the 8th frame after a pulse is exactly one interval later
    same_bar = [(72, 133), (72, 135.5), (72, 137), (72, 134), *[(72, 133)] * 5, (72, 136)]
    pulses, crossings = cue_run(rl.TopEdgeCue(BARS, rearm_mm=1.0, min_interval_s=interval), same_bar, hz)
    # Too soon after the first pulse: no pulse, and the bar stays armed for the next crossing, 8 frames on.
    assert outcomes(crossings) == {1: [("left", "pulse")], 3: [("left", "interval")], 9: [("left", "pulse")]}
    assert pulses[9].t_s - pulses[1].t_s == interval

    other_bar = [(90, 130), (90, 136), (97, 136), (100, 136), (100, 131)]  # into the left bar, out of the right one
    _, crossings = cue_run(rl.TopEdgeCue(NARROW, rearm_mm=1.0, min_interval_s=interval), other_bar, hz)
    assert outcomes(crossings) == {1: [("left", "pulse")], 4: [("right", "interval")]}
    assert outcomes(cue_run(rl.TopEdgeCue(NARROW, rearm_mm=1.0, min_interval_s=0.04), other_bar, hz)[1])[4] == [
        ("right", "pulse")
    ]
    with pytest.raises(ValueError):
        rl.TopEdgeCue(BARS, min_interval_s=0.0)


def test_a_crossing_where_two_extended_edges_overlap_is_the_nearer_bars_only():
    # 3 mm gap, both tops at 135, 2 mm tolerance: the extended edges overlap from 96.5 to 97.5 mm.
    for x, nearer in ((96.8, "left"), (97.2, "right")):
        cue = rl.TopEdgeCue(LEVEL, corner_mm=2.0, rearm_mm=3.0, min_interval_s=0.001)
        pulses, crossings = cue_run(cue, [(x, 131), (x, 136)])
        assert [(c.bar, c.outcome) for c in crossings[1]] == [(nearer, "pulse")]
    # The nearer bar is disarmed: the crossing is still its own, and the armed neighbour does not take it.
    cue = rl.TopEdgeCue(LEVEL, corner_mm=2.0, rearm_mm=3.0, min_interval_s=0.001)
    _, crossings = cue_run(cue, [(96.8, 131), (96.8, 136), (96.8, 133)])
    assert outcomes(crossings) == {1: [("left", "pulse")], 2: [("left", "disarmed")]} and cue.armed["right"]
