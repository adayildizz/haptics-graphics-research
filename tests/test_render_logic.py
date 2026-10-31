from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from experiment import render_logic as rl
from experiment.render_logic import PositionPlanner, Sample

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


def test_sample_clock_reports_intervals_and_repeats():
    clock = rl.SampleClock()
    for k, repeated in enumerate([False, False, True, False, True]):
        clock.add(1.0 + 0.020 * k, repeated)
    summary = clock.summary()
    assert summary["samples"] == 5 and summary["repeated"] == 2 and summary["repeated_ratio"] == 0.4
    assert summary["interval_median_ms"] == pytest.approx(20.0) and summary["interval_max_ms"] == pytest.approx(20.0)


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


def test_position_planner_follows_the_sample_and_labels_the_edge():
    plans = run(PositionPlanner(BARS), sampled(60.0, 140.0, 100.0, 0.0, 16))
    switches = switch_times(plans)
    assert [(on, edge) for _, on, _, edge in switches] == [(True, "left"), (False, "right")]
    # It can only react at a sample, so it is late by up to one interval.
    assert 0.0 <= switches[0][0] - (5.0 + 0.07) < 1 / HZ
    assert all(not plan.transitions for plan in plans)


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
