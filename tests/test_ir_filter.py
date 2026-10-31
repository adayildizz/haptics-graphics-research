"""The IR glitch filter: short false lifts and jumps to a second object, ignored while the finger stays on the glass."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from render_harness import polyline, run_scripted_trial

from analysis.render_replay.run import main
from experiment.calibration import make_configured_ir_frame_calibration
from experiment.config import ExperimentConfig, load_experiment_config
from experiment.constant_stimuli import TrialSpec
from experiment.demo_trace import ensure_demo_trace
from experiment.render_logic import IrGlitchFilter

HZ = 60.0
CONFIG_PATH = Path(__file__).resolve().parent.parent / "experiment" / "configs" / "default.yaml"
CALIBRATION = make_configured_ir_frame_calibration((1920, 1080))
SPEC = TrialSpec(level_pct=0.2, comparison_height_mm=12.0, reference_height_mm=10.0, reference_side="left")


def feed(ir: IrGlitchFilter, reports: list[tuple[float, float, bool]]) -> tuple[list[tuple[float, float, bool]], list]:
    """(x, y, down) the renderer uses for each 60 Hz (x, y, down) report, and the glitches the filter ignored."""
    used, glitches = [], []
    for frame, (x, y, down) in enumerate(reports):
        reading = ir.step(frame / HZ, x, y, down)
        glitches += ir.glitches
        used.append((reading.x_mm, reading.y_mm, reading.down))
    return used, glitches


def test_a_short_lift_is_ignored_and_the_finger_taken_as_down_where_it_was():
    ir = IrGlitchFilter()
    # Down at 70, the cursor moves on to 71, the report says "up" for 2 frames (33 ms) at a frozen 72, down at 72.
    used, glitches = feed(ir, [(70, 140, True), (71, 140, True), (72, 140, False), (72, 140, False), (72, 140, True)])
    assert used == [(70, 140, True), (71, 140, True), (71, 140, True), (71, 140, True), (72, 140, True)]
    (glitch,) = glitches
    assert glitch.kind == "ir_lift_ignored" and glitch.duration_s == pytest.approx(2 / HZ)


def test_a_lift_that_lasts_the_debounce_is_real_from_then_on():
    ir = IrGlitchFilter(lift_debounce_s=0.09)  # not a whole number of frames, so no frame sits on the limit
    reports = [(70, 140, True)] + [(70 + k, 140, False) for k in range(1, 10)] + [(100, 140, True)]
    used, glitches = feed(ir, reports)
    # Frames 1-6 have been up for less than 90 ms: still down at 70. From frame 7 (100 ms up) the lift is real.
    assert used[1:7] == [(70, 140, True)] * 6
    assert used[7:10] == [(77, 140, False), (78, 140, False), (79, 140, False)]
    assert used[10] == (100, 140, True)  # the touchdown is accepted wherever it is
    assert glitches == []


def test_a_jump_is_rejected_until_the_report_comes_back():
    ir = IrGlitchFilter(jump_mm=20.0)
    reports = [(70, 140, True), (71, 140, True), (111, 120, True), (112, 121, True), (72, 140, True), (73, 140, True)]
    used, glitches = feed(ir, reports)
    assert used == [(70, 140, True), (71, 140, True), (71, 140, True), (71, 140, True), (72, 140, True), (73, 140, True)]
    (glitch,) = glitches
    assert (glitch.kind, glitch.ended) == ("ir_jump_rejected", "return")
    assert glitch.duration_s == pytest.approx(2 / HZ) and glitch.distance_mm == pytest.approx(500**0.5 * 2)


def test_a_step_of_exactly_the_limit_is_accepted():
    ir = IrGlitchFilter(jump_mm=20.0)
    assert feed(ir, [(70, 140, True), (90, 140, True)]) == ([(70, 140, True), (90, 140, True)], [])


def test_the_first_position_after_an_ignored_lift_is_checked_for_a_jump():
    ir = IrGlitchFilter()
    # The finger stays at 72; the frame drops it for one frame and comes back on the palm, 40 mm away.
    reports = [(72, 140, True), (72, 140, False), (112, 140, True), (112, 140, True), (72, 140, True)]
    used, glitches = feed(ir, reports)
    assert used == [(72, 140, True)] * 5
    assert [glitch.kind for glitch in glitches] == ["ir_lift_ignored", "ir_jump_rejected"]


def test_a_rejection_that_lasts_the_reanchor_time_accepts_the_report():
    ir = IrGlitchFilter(jump_mm=20.0, reanchor_s=0.99)  # not a whole number of frames, as above
    # The finger really moved 40 mm (or the frame keeps seeing the second object): taken as true after ~1 s.
    reports = [(70, 140, True)] + [(110, 140, True)] * 70
    used, glitches = feed(ir, reports)
    assert used[1:61] == [(70, 140, True)] * 60  # frames 1-60: rejected for less than 0.99 s
    assert used[61:] == [(110, 140, True)] * 10
    (glitch,) = glitches
    assert glitch.kind == "ir_jump_reanchored"
    assert glitch.duration_s == pytest.approx(1.0) and glitch.distance_mm == pytest.approx(40.0)
    # From the new position on, it is measured from there: back at the old one is now the jump.
    assert ir.step(71 / HZ, 70, 140, True).held


def test_a_real_lift_ends_the_rejection_and_the_next_touchdown_is_accepted_anywhere():
    ir = IrGlitchFilter(lift_debounce_s=0.09)
    reports = [(70, 140, True), (110, 140, True)] + [(110, 140, False)] * 7 + [(150, 140, True)]
    used, glitches = feed(ir, reports)
    assert used[1] == (70, 140, True)  # the jump, rejected
    assert used[2:8] == [(70, 140, True)] * 6  # the lift, not real yet
    assert used[8] == (110, 140, False)  # real: as reported
    assert used[9] == (150, 140, True)
    (glitch,) = glitches
    assert (glitch.kind, glitch.ended) == ("ir_jump_rejected", "lift")


def test_a_rejection_still_open_at_the_end_of_the_trial_is_reported():
    ir = IrGlitchFilter()
    assert feed(ir, [(70, 140, True), (110, 140, True), (110, 140, True)])[1] == []
    glitch = ir.finish(3 / HZ)
    assert glitch is not None and (glitch.kind, glitch.ended) == ("ir_jump_rejected", "end")
    assert glitch.duration_s == pytest.approx(2 / HZ)
    assert ir.finish(4 / HZ) is None


def test_a_trial_that_starts_lifted_is_lifted_until_the_touchdown():
    ir = IrGlitchFilter()
    used, glitches = feed(ir, [(10, 10, False), (12, 10, False), (72, 140, True), (73, 140, True)])
    assert used == [(10, 10, False), (12, 10, False), (72, 140, True), (73, 140, True)]


@pytest.mark.parametrize(
    "overrides",
    [{"ir_lift_debounce_ms": -1.0}, {"ir_jump_mm": 0.0}, {"ir_jump_reanchor_ms": 0.0}],
)
def test_invalid_ir_filter_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        ExperimentConfig(base_height_mm=10.0, bar_width_mm=10.0, **overrides)


def test_the_filter_is_on_by_default_with_the_lab_settings():
    for cfg in (load_experiment_config(CONFIG_PATH), ExperimentConfig(base_height_mm=10.0, bar_width_mm=10.0)):
        assert cfg.ir_filter is True
        assert (cfg.ir_lift_debounce_ms, cfg.ir_jump_mm, cfg.ir_jump_reanchor_ms) == (100.0, 20.0, 1000.0)


# The live trial loop. Moving right through the left bar (x 67-77) at 1 mm per frame, on the glass throughout.
SLIDE = polyline([(0, 55, 141), (40, 95, 141)], 41)


def live(scenario: dict, **overrides):
    cfg = replace(load_experiment_config(CONFIG_PATH), **{"render_mode": "position", **overrides})
    return run_scripted_trial(cfg, CALIBRATION, SPEC, scenario)[1]


def events(attempt, kind: str) -> list[dict]:
    return [payload for _, event, payload in attempt.events if event == kind]


def test_a_30_ms_lift_inside_a_bar_no_longer_switches_the_signal_off():
    scenario = dict(path=SLIDE, touch={0: True, 17: False, 19: True}, end="left")  # up for frames 17-18 at x 72-73
    filtered, unfiltered = live(scenario), live(scenario, ir_filter=False)
    assert [s.signal_on for s in unfiltered.samples[17:19]] == [False, False]
    on = [s.signal_on for s in filtered.samples]
    assert all(on[13:22]) and not any(on[:12]) and not any(on[23:])
    assert not [w for w in events(filtered, "signal_write") if w["edge"] in ("lift", "touch")]
    # The trace keeps the report as it came, and what the renderer used beside it.
    assert [s.contact for s in filtered.samples[16:20]] == [True, False, False, True]
    assert all(s.used_contact for s in filtered.samples)
    assert [s.used_x_mm for s in filtered.samples[16:19]] == [filtered.samples[16].x_mm] * 3
    (ignored,) = events(filtered, "ir_lift_ignored")
    assert ignored["duration_ms"] == pytest.approx(2000 / HZ, abs=3)
    assert [kind for _, kind, _ in filtered.events if kind.startswith("contact_")] == ["contact_up", "contact_down"]


def test_a_jump_to_a_second_object_keeps_the_signal_and_the_speed_where_the_finger_is():
    path = list(SLIDE)
    path[17:22] = [(x + 45.0, y - 30.0) for x, y in path[17:22]]  # 5 frames on the palm, the button still down
    scenario = dict(path=path, touch={0: True}, end="left")
    filtered, unfiltered = live(scenario), live(scenario, ir_filter=False)
    assert not any(s.signal_on for s in unfiltered.samples[17:22])
    assert all(s.signal_on for s in filtered.samples[13:22])
    assert [s.used_x_mm for s in filtered.samples[17:22]] == [filtered.samples[16].x_mm] * 5
    assert [s.x_mm for s in filtered.samples[17:22]] == [s.x_mm for s in unfiltered.samples[17:22]]  # raw as before
    # The speed coach and the timed rule are given the speed of the position used, not of the jump.
    assert max(s.speed_mm_s for s in unfiltered.samples) > 2000
    assert max(s.speed_mm_s for s in filtered.samples) < 500
    (rejected,) = events(filtered, "ir_jump_rejected")
    assert rejected["ended"] == "return" and rejected["distance_mm"] > 50
    assert rejected["duration_ms"] == pytest.approx(5000 / HZ, abs=3)


def test_a_real_lift_switches_the_signal_off_once_it_has_lasted_the_debounce():
    scenario = dict(path=SLIDE, touch={0: True, 14: False, 30: True}, end="left")  # up from x 69, 267 ms
    attempt = live(scenario)
    on = [s.signal_on for s in attempt.samples]
    first_off = on.index(False, 14)
    assert 19 <= first_off <= 21  # 100 ms after the lift began, at 60 Hz with jitter
    assert not any(on[first_off:30])
    (lift,) = [w for w in events(attempt, "signal_write") if w["edge"] == "lift"]
    assert lift["on"] is False and lift["bar"] == "left"
    assert events(attempt, "ir_lift_ignored") == []


def test_with_the_filter_off_the_renderer_uses_the_report_as_it_came():
    path = list(SLIDE)
    path[17:22] = [(x + 45.0, y) for x, y in path[17:22]]
    attempt = live(dict(path=path, touch={0: True, 8: False, 10: True}, end="left"), ir_filter=False)
    assert all((s.used_x_mm, s.used_y_mm, s.used_contact) == (s.x_mm, s.y_mm, s.contact) for s in attempt.samples)
    assert not [kind for _, kind, _ in attempt.events if kind.startswith("ir_")]


@pytest.mark.parametrize("mode", ["predictive", "position", "timed_legacy"])
def test_on_clean_input_the_filter_changes_nothing(mode):
    scenario = dict(path=polyline([(0, 50, 140), (60, 140, 140), (90, 50, 140)], 100), touch={0: True}, end="left")
    filtered = live(scenario, render_mode=mode)
    unfiltered = live(scenario, render_mode=mode, ir_filter=False)
    assert [s.signal_on for s in filtered.samples] == [s.signal_on for s in unfiltered.samples]
    assert [s.speed_mm_s for s in filtered.samples] == [s.speed_mm_s for s in unfiltered.samples]
    assert not [kind for _, kind, _ in filtered.events if kind.startswith("ir_")]


def test_the_replay_filters_a_recording_made_with_the_filter_and_can_force_it(tmp_path, capsys):
    trace = ensure_demo_trace(tmp_path / "demo_replay_trace.sqlite3")
    connection = sqlite3.connect(trace)
    connection.execute("UPDATE cursor_samples SET contact = 0 WHERE attempt_id = 2 AND sequence BETWEEN 100 AND 101")
    connection.commit()
    assert main([str(trace)]) == 0  # recorded without the filter: replayed as reported
    assert "IR glitch filter: off" in (printed := capsys.readouterr().out) and "contact: 1 lift(s)" in printed
    assert main([str(trace), "--ir-filter", "on"]) == 0
    printed = capsys.readouterr().out
    assert "IR glitch filter: on in 4 of 4 trial(s): 1 ir_lift_ignored" in printed and "contact: 0 lift(s)" in printed
    (config,) = connection.execute("SELECT config_json FROM sessions").fetchone()
    connection.execute("UPDATE sessions SET config_json = ?", (json.dumps({**json.loads(config), "ir_filter": True}),))
    connection.commit()
    connection.close()
    assert main([str(trace)]) == 0  # recorded with it: replayed through it
    assert "1 ir_lift_ignored" in capsys.readouterr().out
