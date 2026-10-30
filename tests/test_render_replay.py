from __future__ import annotations

import csv
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from render_harness import polyline, run_scripted_trial

from analysis.render_replay import pulses, synthetic
from analysis.render_replay.core import Params, contact_in, evaluate, planner_for, replay, simulate, submitted_plans
from analysis.render_replay.run import main, replay_synthetic, summarize
from experiment import render_logic as rl
from experiment.calibration import make_configured_ir_frame_calibration
from experiment.config import load_experiment_config
from experiment.constant_stimuli import TrialSpec
from experiment.demo_trace import ensure_demo_trace
from experiment.render_logic import Sample

HZ = 60.0
CONFIG_PATH = Path(__file__).resolve().parent.parent / "experiment" / "configs" / "default.yaml"
BARS = rl.nominal_bars(194.0, 145.0, 40.0, 10.0, 10.0, 12.0)  # left 67-77 top 135, right 117-127 top 133
LEFT = BARS[0]


def finger(legs: list[tuple[float, float, float, float]], start: tuple[float, float], hold_s: float = 0.0) -> list[Sample]:
    """The true path at 1 kHz: straight legs given as (dx, dy, speed, pause afterwards)."""
    t, x, y = 20.0, *start
    truth = [Sample(t, x, y)]
    for dx, dy, speed, pause in legs:
        steps = max(1, round(1000.0 * np.hypot(dx, dy) / speed))
        for k in range(1, steps + 1):
            truth.append(Sample(t + k / 1000.0, x + dx * k / steps, y + dy * k / steps))
        t, x, y = t + steps / 1000.0, x + dx, y + dy
        for k in range(1, round(1000.0 * pause) + 1):
            truth.append(Sample(t + k / 1000.0, x, y))
        t += round(1000.0 * pause) / 1000.0
    return truth


def seen_at_60hz(truth: list[Sample], phase_ms: int = 0, noise_mm: float = 0.0, seed: int = 0) -> list[Sample]:
    rng = np.random.default_rng(seed)
    picked = truth[phase_ms :: round(1000.0 / HZ)]
    return [Sample(s.t_s, s.x_mm + rng.normal(0, noise_mm), s.y_mm + rng.normal(0, noise_mm)) for s in picked]


def errors_by_mode(truth, phase_ms: int = 0, noise_mm: float = 0.0, seed: int = 0) -> dict[str, dict[tuple[str, bool], list[float]]]:
    """late_mm per (edge, entering) for each mode, the finger being seen at 60 Hz."""
    outcomes = replay(seen_at_60hz(truth, phase_ms, noise_mm, seed), BARS, None, 10.0, Params(), truth=truth)
    table: dict[str, dict[tuple[str, bool], list[float]]] = {}
    for mode, outcome in outcomes.items():
        table[mode] = {}
        for error in outcome.errors:
            table[mode].setdefault((error.edge, error.entering), []).append(error.late_mm)
    return table


@pytest.mark.parametrize("phase_ms", [0, 5, 11])
def test_horizontal_pass_edge_errors_of_the_three_modes(phase_ms):
    truth = finger([(35.0, 0.0, 100.0, 0.0)], (55.0, 140.0))
    table = errors_by_mode(truth, phase_ms)
    frame_mm = 100.0 / HZ
    for edge in (("left", True), ("right", False)):
        assert abs(table["predictive"][edge][0]) < 0.02
        assert 0.0 <= table["position"][edge][0] <= frame_mm + 0.01
    # The timed rule starts its clock at the first frame inside, so it can only be as late or later.
    assert table["timed_legacy"][("left", True)] == table["position"][("left", True)]
    assert table["position"][("right", False)][0] <= table["timed_legacy"][("right", False)][0] <= 2 * frame_mm + 0.01


def test_upward_exit_through_the_top_edge():
    # Starts inside the left bar and leaves upward at 40 mm/s, then keeps going.
    truth = finger([(0.0, -30.0, 40.0, 0.0)], (72.0, 143.0))
    outcomes = replay(seen_at_60hz(truth, 3), BARS, None, 10.0, Params(), truth=truth)
    predictive, position, legacy = (outcomes[mode].errors[0] for mode in rl.RENDER_MODES)
    assert (predictive.edge, predictive.entering) == ("top", False)
    assert abs(predictive.late_mm) < 0.02
    assert 0.0 <= position.late_mm <= 40.0 / HZ + 0.01 and position.height_effect_mm == position.late_mm
    # The first frame is inside at speed 0, so the timed rule commands its 2 s maximum: on far above the bar.
    assert legacy.bridged and legacy.late_mm > 15.0 and legacy.height_effect_mm > 15.0


def test_downward_entry_through_the_top_edge():
    truth = finger([(0.0, 14.0, 30.0, 0.0)], (122.0, 125.0))  # right bar, top at 133 mm
    outcomes = replay(seen_at_60hz(truth, 7), BARS, None, 10.0, Params(), truth=truth)
    predictive, position, legacy = (outcomes[mode].errors[0] for mode in rl.RENDER_MODES)
    assert (predictive.edge, predictive.entering) == ("top", True)
    assert abs(predictive.late_mm) < 0.02
    assert 0.0 <= position.late_mm <= 30.0 / HZ + 0.01
    assert position.height_effect_mm == -position.late_mm  # switched on late: the bar feels shorter
    assert legacy.late_mm == position.late_mm


def test_corner_cut_is_rendered_only_by_the_predictive_mode():
    # In through the top of the left bar and out through its right side 5 ms later, between two frames.
    truth = finger([(21.0, 21.0, 141.42, 0.0)], (66.0, 124.5))
    recorded = truth[10 :: round(1000.0 / HZ)]
    inside = [s for s in recorded if LEFT.contains(s.x_mm, s.y_mm)]
    assert not inside, "pick a phase where no frame lands inside the bar"
    outcomes = replay(recorded, BARS, None, 10.0, Params(), truth=truth)
    assert outcomes["position"].missed == 1 and outcomes["timed_legacy"].missed == 1
    assert outcomes["predictive"].missed == 0
    cut = {(e.edge, e.entering): e.late_mm for e in outcomes["predictive"].errors}
    assert set(cut) == {("top", True), ("right", False)}
    assert all(abs(late) < 0.05 for late in cut.values())


def test_stopping_inside_a_bar_then_leaving():
    truth = finger([(17.0, 0.0, 100.0, 0.5), (15.0, 0.0, 100.0, 0.0)], (55.0, 140.0))  # stops at x = 72
    table = errors_by_mode(truth, 2)
    assert abs(table["predictive"][("left", True)][0]) < 0.02
    # Leaving from rest: one frame to see the movement, then the exit is timed.
    assert abs(table["predictive"][("right", False)][0]) < 0.02
    assert 0.0 <= table["position"][("right", False)][0] <= 100.0 / HZ + 0.01


def test_below_the_speed_threshold_all_modes_follow_the_sample():
    truth = finger([(0.0, 4.0, 3.0, 0.0)], (72.0, 133.0))  # 3 mm/s down through the top at 135 mm
    table = errors_by_mode(truth, 9)
    late = {mode: table[mode][("top", True)][0] for mode in rl.RENDER_MODES}
    assert late["predictive"] == late["position"] == late["timed_legacy"]
    assert 0.0 <= late["position"] <= 3.0 / HZ + 0.01


def test_noisy_positions_keep_the_predictive_mode_ahead_of_the_position_mode():
    truth = finger([(35.0, 0.0, 100.0, 0.0)], (55.0, 140.0))
    predictive, position = [], []
    for seed in range(120):
        table = errors_by_mode(truth, phase_ms=seed % 16, noise_mm=0.05, seed=seed)
        predictive += table["predictive"].get(("left", True), []) + table["predictive"].get(("right", False), [])
        position += table["position"].get(("left", True), []) + table["position"].get(("right", False), [])
    assert abs(np.median(predictive)) < 0.1 and np.percentile(np.abs(predictive), 95) < 0.5
    assert np.median(position) > 0.6


def test_top_edge_sign_convention():
    up = finger([(0.0, -20.0, 50.0, 0.0)], (72.0, 143.0))  # leaves the top (135 mm) at t = 20.16 s
    late_off = evaluate("x", [(up[0].t_s, True), (20.16 + 0.020, False)], up, BARS).errors[0]
    assert late_off.late_mm == pytest.approx(1.0, abs=0.02) and late_off.height_effect_mm > 0  # feels taller
    early_off = evaluate("x", [(up[0].t_s, True), (20.16 - 0.020, False)], up, BARS).errors[0]
    assert early_off.late_mm == pytest.approx(-1.0, abs=0.02) and early_off.height_effect_mm < 0

    down = finger([(0.0, 20.0, 50.0, 0.0)], (72.0, 125.0))  # reaches the top at t = 20.20 s
    late_on = evaluate("x", [(20.20 + 0.020, True)], down, BARS).errors[0]
    assert late_on.late_mm == pytest.approx(1.0, abs=0.02) and late_on.height_effect_mm < 0  # feels shorter
    assert late_on.category == "top edge, downward entry" and late_off.category == "top edge, upward exit"


def test_signal_left_on_between_two_visits_is_reported_with_how_far_the_finger_got():
    # Down 3 mm into the bar at 20 mm/s (a 0.5 s timer), up 6 mm above its top and back within that time.
    truth = finger([(0.0, 9.0, 20.0, 0.0), (0.0, -9.0, 60.0, 0.0), (0.0, 9.0, 60.0, 0.0)], (72.0, 129.0))
    outcomes = replay(seen_at_60hz(truth, 1), BARS, None, 10.0, Params(), truth=truth)
    held = [e for e in outcomes["timed_legacy"].errors if e.bridged]
    assert len(held) == 1 and held[0].edge == "top" and not held[0].entering
    assert held[0].late_mm == pytest.approx(6.0, abs=0.1) and held[0].height_effect_mm == held[0].late_mm
    assert not any(e.bridged for e in outcomes["predictive"].errors)
    assert max(abs(e.late_mm) for e in outcomes["predictive"].errors) < 0.2


def test_synthetic_session_ranks_the_modes_and_latency_compensation_helps_against_the_finger():
    rows = {(row["mode"], row["edge"]): row for row in summarize(replay_synthetic(Params(), against_finger=False))}
    exits = [rows[(mode, "top edge, upward exit")]["median_mm"] for mode in rl.RENDER_MODES]
    assert 0 <= exits[0] < exits[1] < exits[2]  # predictive < position < timed_legacy: felt height inflation
    assert rows[("timed_legacy", "top edge, upward exit")]["never_switched_off"] > 0
    assert rows[("predictive", "(visits with no signal at all / signal on away from any bar)")]["n"] == 0

    def entry_lag(latency_ms: float) -> float:
        results = replay_synthetic(Params(latency_ms=latency_ms), against_finger=True)
        return {(r["mode"], r["edge"]): r for r in summarize(results)}[("predictive", "side edge, entry")]["median_mm"]

    assert entry_lag(0.0) > 0.4  # the recording cannot show this: the positions were already old when read
    assert abs(entry_lag(6.0)) < 0.2


def test_command_line_on_the_synthetic_session_and_on_a_recording(tmp_path, capsys):
    assert main(["--synthetic", "--out", str(tmp_path / "out")]) == 0
    printed = capsys.readouterr().out
    assert "judged against the finger path" in printed and printed.count("Note:") == 3
    assert "oscilloscope" in printed and "signal_on column" in printed
    with (tmp_path / "out" / "synthetic_vs_recording_crossings.csv").open() as file:
        crossings = list(csv.DictReader(file))
    assert {row["mode"] for row in crossings} == set(rl.RENDER_MODES)
    assert {"top", "left", "right"} <= {row["edge"] for row in crossings}
    assert (tmp_path / "out" / "synthetic_vs_finger_summary.csv").exists()

    trace = ensure_demo_trace(tmp_path / "demo_replay_trace.sqlite3")
    assert main([str(trace)]) == 0
    assert "recorded trial(s)" in capsys.readouterr().out
    assert main([str(tmp_path / "empty_dir_that_has_no_traces")]) == 2


def test_synthetic_recording_looks_like_a_60hz_poll_of_a_100hz_frame():
    trial = synthetic.session(trials_per_level=1)[0]
    intervals = np.diff([s.t_s for s in trial.recorded])
    assert np.median(intervals) == pytest.approx(1 / HZ, abs=1e-3)
    steps_px = np.diff([s.x_mm for s in trial.recorded]) / synthetic.PIXEL_MM[0]
    assert np.allclose(steps_px, np.round(steps_px), atol=1e-6)  # whole pixels
    assert len(trial.truth) > 10 * len(trial.recorded)


def test_tool_uses_only_the_pure_render_module_from_the_experiment_package():
    code = (
        "import sys, analysis.render_replay.run\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('pygame', 'pyvisa')"
        " or (m.startswith('experiment.') and m != 'experiment.render_logic')]\n"
        "print(bad); raise SystemExit(1 if bad else 0)"
    )
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


def recording(path: list[tuple[float, float]], touch: dict[int, bool]) -> tuple[list[Sample], list[bool]]:
    """60 Hz samples along ``path``, and contact per sample from {frame: down} switches; down until told otherwise."""
    down, state = [], True
    for frame in range(len(path)):
        state = touch.get(frame, state)
        down.append(state)
    return [Sample(20.0 + frame / HZ, x, y) for frame, (x, y) in enumerate(path)], down


def kinds(errors) -> list[tuple[str, str, bool]]:
    return [(error.bar, error.edge, error.entering) for error in errors]


def test_a_lift_inside_a_bar_ends_the_visit_without_an_exit_error():
    # In through the left side of the left bar, stops at x = 72, lifts; the cursor stays there and jumps on touchdown.
    path = polyline([(0, 55, 140), (17, 72, 140), (40, 72, 140), (41, 97, 140), (60, 105, 140)], 61)
    samples, down = recording(path, {25: False, 41: True})
    lifted = replay(samples, BARS, None, 10.0, Params(), down=down)
    unknown = replay(samples, BARS, None, 10.0, Params())
    for mode in rl.RENDER_MODES:
        assert kinds(lifted[mode].errors) == [("left", "left", True)]
        assert lifted[mode].missed == 0 and lifted[mode].spurious_ms == 0
        assert ("left", "right", False) in kinds(unknown[mode].errors)  # the jump, taken for a movement out of the bar
    contact = contact_in(samples, down, BARS)
    assert (contact.lifts, contact.touch_starts, contact.lift_ends) == (1, 0, 1)
    assert contact.lifted_s == pytest.approx(16 / HZ) and contact.recorded_s == pytest.approx(60 / HZ)


def test_a_touchdown_elsewhere_draws_no_path_through_the_bar_in_between():
    # Lifted at x = 60, put down again at x = 100: the straight line between them runs through the left bar.
    path = polyline([(0, 50, 140), (10, 60, 140), (30, 60, 140), (31, 100, 140), (45, 105, 140)], 46)
    samples, down = recording(path, {12: False, 31: True})
    lifted = replay(samples, BARS, None, 10.0, Params(), down=down)
    for mode in rl.RENDER_MODES:
        assert lifted[mode].errors == [] and lifted[mode].missed == 0 and lifted[mode].spurious_ms == 0
    assert replay(samples, BARS, None, 10.0, Params())["position"].missed == 1  # an invented visit
    assert contact_in(samples, down, BARS).lift_ends == 0


def test_the_signal_goes_off_at_the_first_lifted_sample_and_on_again_at_the_touchdown():
    # Moving slowly inside the left bar; lifted for frames 10-19, put down again 2 mm further on.
    path = polyline([(0, 70, 140), (10, 72, 140), (19, 72, 140), (20, 74, 140), (30, 75.5, 140)], 31)
    samples, down = recording(path, {10: False, 20: True})
    for mode in rl.RENDER_MODES:
        new_planner = partial(planner_for, mode, BARS, None, 10.0, Params())
        switches = simulate(submitted_plans(new_planner, samples, down))
        assert switches == [(samples[0].t_s, True), (samples[10].t_s, False), (samples[20].t_s, True)]
        assert simulate(submitted_plans(new_planner, samples)) == [(samples[0].t_s, True)]
        outcome = replay(samples, BARS, None, 10.0, Params(), down=down)[mode]
        assert outcome.errors == [] and outcome.missed == 0 and outcome.spurious_ms == 0


def test_touching_down_inside_a_bar_switches_on_at_once_with_no_entry_error():
    # Lifted at x = 60 from the start, put down inside the right bar (117-127), then out through its right side.
    path = polyline([(0, 60, 140), (10, 60, 140), (11, 122, 140), (40, 140, 140)], 41)
    samples, down = recording(path, {0: False, 11: True})
    outcomes = replay(samples, BARS, None, 10.0, Params(), down=down)
    for mode in rl.RENDER_MODES:
        new_planner = partial(planner_for, mode, BARS, None, 10.0, Params())
        assert simulate(submitted_plans(new_planner, samples, down))[0] == (samples[11].t_s, True)
        assert kinds(outcomes[mode].errors) == [("right", "right", False)]
        assert outcomes[mode].missed == 0
    # The live timer's speed at a touchdown spans the jump, so it runs out at once: off like the position mode.
    legacy, position = outcomes["timed_legacy"].errors[0], outcomes["position"].errors[0]
    assert not legacy.bridged and legacy.late_mm == position.late_mm
    contact = contact_in(samples, down, BARS)
    assert (contact.lifts, contact.touch_starts, contact.lift_ends) == (0, 1, 0)


@pytest.mark.parametrize("mode", rl.RENDER_MODES)
def test_the_replay_takes_lifts_as_the_trial_loop_does(mode):
    # The live loop on a scripted trial: lifted inside the left bar, down inside the right one and out of it,
    # lifted again and put down across the left bar. Replaying what it recorded gives the same state at every sample.
    calibration = make_configured_ir_frame_calibration((1920, 1080))
    spec = TrialSpec(level_pct=0.2, comparison_height_mm=12.0, reference_height_mm=10.0, reference_side="left")
    waypoints = [(0, 55, 141), (20, 72, 141), (39, 72, 141), (40, 122, 141), (60, 135, 141), (69, 135, 141)]
    scenario = dict(
        path=polyline([*waypoints, (70, 60, 141), (85, 64, 141)], 86),
        touch={0: True, 25: False, 40: True, 62: False, 70: True},
        end="left",
    )
    # The replay does not run the IR glitch filter yet.
    cfg = replace(load_experiment_config(CONFIG_PATH), render_mode=mode, ir_filter=False)
    _, attempt, _, _ = run_scripted_trial(cfg, calibration, spec, scenario)
    c = calibration
    heights = (10.0, 12.0)
    nominal = rl.nominal_bars(c.active_width_mm, c.active_height_mm, cfg.inter_bar_gap_mm, cfg.bar_width_mm, *heights)
    pixel = rl.pixel_bars(
        c.active_left_px, c.active_top_px, c.active_width_px, c.active_height_px, c.px_per_mm_x, c.px_per_mm_y,
        cfg.inter_bar_gap_mm, cfg.bar_width_mm, *heights,
    )  # fmt: skip
    params = Params(
        cfg.predict_window_samples, cfg.predict_horizon_factor, cfg.predict_horizon_max_ms, cfg.predict_min_speed_mm_s,
        cfg.latency_compensation_ms, 1.0 / 60.0, max(1.0 / c.px_per_mm_x, 1.0 / c.px_per_mm_y),
    )  # fmt: skip
    samples = [Sample(s.t_us / 1e6, s.x_mm, s.y_mm) for s in attempt.samples]
    down = [s.contact for s in attempt.samples]
    assert not all(down)
    new_planner = partial(planner_for, mode, nominal, pixel, cfg.bar_width_mm, params)
    live = [s.signal_on for s in attempt.samples]
    assert [plan.on for plan in submitted_plans(new_planner, samples, down)] == live
    assert [plan.on for plan in submitted_plans(new_planner, samples)] != live


def test_command_line_reports_lifts_and_reads_a_trace_without_contact_as_always_down(tmp_path, capsys):
    trace = ensure_demo_trace(tmp_path / "demo_replay_trace.sqlite3")
    (tmp_path / "old").mkdir()
    (tmp_path / "lifted").mkdir()
    old = Path(shutil.copy(trace, tmp_path / "old" / "demo_trace.sqlite3"))
    lifted = Path(shutil.copy(trace, tmp_path / "lifted" / "demo_trace.sqlite3"))
    connection = sqlite3.connect(old)
    connection.execute("ALTER TABLE cursor_samples DROP COLUMN contact")  # as in a schema 1 or 2 recording
    connection.commit()
    connection.close()
    connection = sqlite3.connect(lifted)
    connection.execute("UPDATE cursor_samples SET contact = 0 WHERE attempt_id = 2 AND sequence BETWEEN 100 AND 159")
    connection.commit()
    connection.close()

    assert main([str(trace), "--out", str(tmp_path / "out_v3")]) == 0
    assert "contact: 0 lift(s), lifted 0.0 s" in capsys.readouterr().out
    assert main([str(old), "--out", str(tmp_path / "out_old")]) == 0
    assert "contact: not recorded" in capsys.readouterr().out
    for name in ("recording_crossings.csv", "recording_summary.csv"):
        assert (tmp_path / "out_old" / name).read_text() == (tmp_path / "out_v3" / name).read_text()
    assert main([str(lifted)]) == 0
    assert "contact: 1 lift(s), lifted 1.0 s" in capsys.readouterr().out


def cue_trial(points, hz=20.0, down=None, block="main", setting=pulses.Setting(3.0, 1.0, 20.0)):
    samples = [Sample(frame / hz, x, y) for frame, (x, y) in enumerate(points)]
    return pulses.cue_trial(pulses.CueTrial("t", block, samples, down, BARS), setting)


def fired(result) -> list[tuple[str, str]]:
    return [(e.crossing.bar, e.crossing.direction) for e in result.events if e.crossing.outcome == "pulse"]


def test_pulses_take_lifts_from_the_recorded_contact():
    # Down into the left bar, lifted while the cursor sits there, put down 7 mm above the top and down again.
    path = polyline([(0, 72, 128), (10, 72, 140), (20, 72, 140), (21, 72, 128), (41, 72, 140)], 42)
    samples, down = recording(path, {11: False, 21: True})
    trial = pulses.CueTrial("t", "main", samples, down, BARS)
    result = pulses.cue_trial(trial, pulses.Setting(3.0, 1.0, 20.0))
    assert fired(result) == [("left", "entering"), ("left", "entering")]
    # Without contact the 12 mm jump back up is a step like any other, and it crosses the top on the way.
    unknown = pulses.cue_trial(replace(trial, down=None), pulses.Setting(3.0, 1.0, 20.0))
    assert fired(unknown) == [("left", "entering"), ("left", "leaving"), ("left", "entering")]
    assert unknown.jumps == 0
    assert pulses.cue_trial(replace(trial, down=None), pulses.Setting(3.0, 1.0, 10.0)).jumps == 1


def test_pulse_events_measure_the_path_since_the_pulse_that_disarmed_the_bar_and_chatter():
    # In, 1 mm further, back up through the top (disarmed: 2 mm below it), up 5 mm, and in again 200 ms later.
    result = cue_trial([(72, 130), (72, 136), (72, 137), (72, 134), (72, 130), (72, 136)])
    first, missed, again = result.events
    assert (first.crossing.outcome, missed.crossing.outcome, again.crossing.outcome) == ("pulse", "disarmed", "pulse")
    assert missed.after_pulse_mm == pytest.approx(3.0)  # 1 mm down to 137, 2 mm back up to the top
    assert missed.after_pulse_ms == pytest.approx(1000.0 * (0.10 + 0.05 * 2 / 3 - 0.05))
    assert not first.chatter and again.chatter and all(e.own_edge for e in result.events)
    # Disarmed by the touchdown rather than by a pulse: nothing to measure from.
    landed = cue_trial([(72, 136), (72, 133)])
    assert [(e.crossing.outcome, e.after_pulse_mm) for e in landed.events] == [("disarmed", None)]


def test_pulse_summary_counts_blocks_tolerance_pulses_and_jumps():
    setting = pulses.Setting(3.0, 1.0, 20.0)
    down_through = [(72, 130), (72, 136)]
    corner = [(72.5, 130), (74.5, 132), (76.5, 134), (78.5, 136)]  # crosses the top line at x = 77.5, beside the bar
    jump = [(72, 110), (72, 136)]  # 26 mm in one frame across the top
    trials = [
        pulses.CueTrial("a", "main", [Sample(k / 20, x, y) for k, (x, y) in enumerate(down_through)], None, BARS),
        pulses.CueTrial("b", "main", [Sample(k / 20, x, y) for k, (x, y) in enumerate(corner)], None, BARS),
        pulses.CueTrial("c", "catch", [Sample(k / 20, x, y) for k, (x, y) in enumerate(jump)], None, BARS),
    ]
    row = pulses.summarize(setting, trials, [pulses.cue_trial(trial, setting) for trial in trials])
    assert (row["trials_main"], row["pulses_per_trial_main"]) == (2, 1.0)
    assert (row["trials_catch"], row["pulses_per_trial_catch"], row["pulses_per_trial_practice"]) == (1, 0.0, "")
    assert (row["top_crossings"], row["top_crossings_pulsed"], row["pulses_corner_only"]) == (1, 1, 1)
    assert (row["jumps"], row["jumps_crossing"], row["jumps_would_fire"]) == (1, 1, 1)
    assert (row["pulses_entering"], row["pulses_leaving"]) == (2, 0)
    assert row["late_entering_median_mm"] == 1.0  # both 1 mm below the top


def test_command_line_sweeps_the_top_edge_cue_with_and_without_contact(tmp_path, capsys):
    trace = ensure_demo_trace(tmp_path / "demo_replay_trace.sqlite3")
    (tmp_path / "old").mkdir()
    old = Path(shutil.copy(trace, tmp_path / "old" / "demo_trace.sqlite3"))
    connection = sqlite3.connect(old)
    connection.execute("ALTER TABLE cursor_samples DROP COLUMN contact")
    connection.commit()
    connection.close()
    grid = ["--rearm", "2,3", "--corner", "1", "--jump", "20"]
    assert main(["--pulses", str(trace), str(old), *grid, "--out", str(tmp_path / "out")]) == 0
    printed = capsys.readouterr().out
    assert "Recordings with contact (schema 3)" in printed and "without a contact column" in printed
    assert printed.count("Note:") == len(pulses.CAVEATS)
    with (tmp_path / "out" / "pulses_contact_sweep.csv").open() as file:
        rows = list(csv.DictReader(file))
    settings = [(row["rearm_mm"], row["corner_mm"], row["jump_mm"]) for row in rows]
    assert settings == [("2.0", "1.0", "20.0"), ("3.0", "1.0", "20.0")]
    assert (tmp_path / "out" / "pulses_no_contact_events.csv").exists()
    with pytest.raises(SystemExit):
        main(["--pulses"])
    with pytest.raises(SystemExit):
        main([str(trace), str(old)])  # the edge-error replay takes one
