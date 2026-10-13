from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from analysis.render_replay import synthetic
from analysis.render_replay.core import Params, evaluate, replay
from analysis.render_replay.run import main, replay_synthetic, summarize
from experiment import render_logic as rl
from experiment.demo_trace import ensure_demo_trace
from experiment.render_logic import Sample

HZ = 60.0
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
