"""The force sensor in a session: start, hands-off steps, a reader that stops midway, and the trace's clock."""

from __future__ import annotations

import json
import sqlite3
import time
from functools import partial
from pathlib import Path

import pygame
import pytest
from render_harness import frame_times, polyline, run_scripted_trial

import experiment.main as main_module
from experiment import display
from experiment import trial as trial_module
from experiment.calibration import make_configured_ir_frame_calibration
from experiment.config import ExperimentConfig
from experiment.constant_stimuli import TrialSpec
from experiment.force_daq import FakeForceSource, load_force
from experiment.force_session import ForceSensorError, ForceSession, force_file_path
from experiment.trace_recorder import AsyncTraceRecorder

CALIBRATION = make_configured_ir_frame_calibration((1920, 1080))


def force_config(**changes) -> ExperimentConfig:
    settings = dict(base_height_mm=10.0, bar_width_mm=10.0, force_sensor=True, force_fake=True, force_zero_s=0.5)
    return ExperimentConfig(**{**settings, **changes})


def new_trace(tmp_path: Path) -> tuple[AsyncTraceRecorder, Path]:
    path = tmp_path / "S_trace.sqlite3"
    recorder = AsyncTraceRecorder(path)
    recorder.create_session(session_id="S", participant_id="P", started_at="t", config={}, calibration={}, app_version="test")
    return recorder, path


def metadata(trace_path: Path) -> dict[str, str]:
    connection = sqlite3.connect(trace_path)
    try:
        return dict(connection.execute("SELECT key, value FROM trace_metadata"))
    finally:
        connection.close()


@pytest.fixture
def quiet_screens(monkeypatch):
    """No window: the screens only remember what they would have shown, and no key is ever pressed."""
    shown: list[tuple[str, str | None]] = []
    monkeypatch.setattr(display, "draw_break", lambda _screen, message, hint=None: shown.append((message, hint)))
    monkeypatch.setattr(pygame.event, "get", lambda: [])
    return shown


def test_a_session_notes_its_force_file_card_hands_off_steps_and_reader_summary(tmp_path, quiet_screens, capsys):
    recorder, trace_path = new_trace(tmp_path)
    force = ForceSession(force_config(), recorder, force_file_path(tmp_path, "S"))
    force.start()
    try:
        assert main_module.hands_off_step(None, force, "practice")
        assert main_module.hands_off_step(None, force, "main")
        time.sleep(0.05)  # in a session the main block follows; the reader reads a few samples at a time
    finally:
        force.stop()
    assert recorder.close() is None

    facts = metadata(trace_path)
    assert facts["force_file"] == "S_force.sqlite3" and (tmp_path / "S_force.sqlite3").exists()
    assert facts["force_cal_file"] == "FT05411.cal"
    assert json.loads(facts["force_card"]) == {"source": "fake", "rate_hz": 1000.0, "calibration_serial": "FT5411"}
    windows = json.loads(facts["force_hands_off"])
    assert [w["block"] for w in windows] == ["practice", "main"]
    assert all(0.5 <= w["end_s"] - w["start_s"] < 0.7 for w in windows)
    summary = json.loads(facts["force_reader_summary"])
    assert summary["error"] is None and summary["samples"] > 900
    assert "force_reader_problem" not in facts

    # Both windows lie inside the force file, on the same clock: their zero can be computed afterwards.
    recording = load_force(tmp_path / "S_force.sqlite3")
    assert len(recording.t_s) == summary["samples"]
    assert recording.t_s[0] < windows[0]["start_s"] and windows[-1]["end_s"] < recording.t_s[-1]
    assert ("Hands off the screen: 1", "Zeroing the force sensor.") in quiet_screens
    assert "Force data saved" in capsys.readouterr().out


def test_the_operator_can_leave_during_a_hands_off_step(monkeypatch, quiet_screens):
    escape = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)
    monkeypatch.setattr(pygame.event, "get", lambda: [escape])
    force = ForceSession(force_config(), None, Path("unused.sqlite3"))  # never started: nothing reaches the trace
    assert not main_module.hands_off_step(None, force, "practice")
    assert force.hands_off == []


def test_a_calibration_file_that_cannot_be_read_stops_the_session_before_it_starts(tmp_path, quiet_screens, monkeypatch, capsys):
    monkeypatch.setattr(pygame.time, "wait", lambda _ms: None)
    recorder, trace_path = new_trace(tmp_path)
    force = ForceSession(force_config(force_cal_file=str(tmp_path / "missing.cal")), recorder, force_file_path(tmp_path, "S"))
    assert not main_module.start_force_sensor(None, force)
    force.stop()  # what the session's finally does: nothing to stop
    assert recorder.close() is None

    out = capsys.readouterr().out
    assert "the force sensor could not start, so the session does not start" in out and "missing.cal" in out
    assert quiet_screens[-1][0] == "Force sensor error: the session does not start."
    assert "missing.cal" in json.loads(metadata(trace_path)["force_reader_problem"])["error"]
    assert not (tmp_path / "S_force.sqlite3").exists()


def test_a_card_that_cannot_be_opened_stops_the_session_before_it_starts(tmp_path):
    recorder, trace_path = new_trace(tmp_path)
    force = ForceSession(force_config(), recorder, force_file_path(tmp_path, "S"), partial(FakeForceSource, fail_at_start=True))
    with pytest.raises(ForceSensorError, match="could not be opened"):
        force.start()
    assert recorder.close() is None
    assert "could not be opened" in json.loads(metadata(trace_path)["force_reader_problem"])["error"]


def test_a_reader_that_stops_mid_session_is_noted_once_and_the_session_goes_on(tmp_path, quiet_screens, monkeypatch, capsys):
    cfg = force_config(
        n_levels=2, trials_per_level=2, sweeps_per_block=1, catch_trial_pct=0.0, n_practice_trials=2,
        practice_easiest_levels=1, practice_pass_fraction=0.0, practice_voice_feedback=False,
        practice_spoken_feedback=False,
    )  # fmt: skip
    recorder, trace_path = new_trace(tmp_path)
    # The fake card fails after 1.5 s: after the practice hands-off step, during the first trial.
    force = ForceSession(cfg, recorder, force_file_path(tmp_path, "S"), partial(FakeForceSource, fail_after=1500))
    force.start()
    calls: list[bool] = []

    def run_trial(_screen, _clock, _calibration, _signal, _cfg, spec, trial_index, **_kwargs):
        if not calls:  # the reader is still running when the first trial starts; it stops during it
            assert force.problem is None
            deadline = time.perf_counter() + 5.0
            while force._reader.running and time.perf_counter() < deadline:
                time.sleep(0.01)
        calls.append(spec.is_practice)
        return trial_module.TrialResult(
            trial_index, spec.level_pct, spec.comparison_height_mm, spec.reference_height_mm, cfg.bar_width_mm,
            spec.reference_side, "left", True, spec.is_catch, spec.is_practice, 1.0,
        )  # fmt: skip

    monkeypatch.setattr(main_module, "wait_for_continue_or_exit", lambda _screen, _message: True)
    monkeypatch.setattr(trial_module, "run_trial", run_trial)
    try:
        main_module.run_constant_stimuli(
            None, None, CALIBRATION, None, cfg, "S", tmp_path / "S_trials.csv", 7, trace_recorder=recorder, force=force
        )
    finally:
        force.stop()
    assert recorder.close() is None

    assert calls == [True, True, False, False, False, False]  # every trial ran, practice and main
    out = capsys.readouterr().out
    assert out.count("WARNING: the force reader stopped during the session") == 1
    facts = metadata(trace_path)
    problem = json.loads(facts["force_reader_problem"])
    assert "fake DAQ card failed while reading" in problem["error"] and 0 < problem["samples"] <= 1500
    assert [w["block"] for w in json.loads(facts["force_hands_off"])] == ["practice", "main"]
    assert json.loads(facts["force_reader_summary"])["samples"] == problem["samples"]


def test_every_cursor_sample_maps_onto_perf_counter_through_the_trial_start():
    """The trace stores the trial start on perf_counter; every sample then lands on that clock to a microsecond."""
    spec = TrialSpec(level_pct=0.2, comparison_height_mm=12.0, reference_height_mm=10.0, reference_side="left")
    scenario = dict(path=polyline([(0, 55, 141), (40, 90, 141)], 41), end="left")
    cfg = ExperimentConfig(base_height_mm=10.0, bar_width_mm=10.0)
    result, attempt, _, _ = run_scripted_trial(cfg, CALIBRATION, spec, scenario)
    assert isinstance(result, trial_module.TrialResult)
    # The scripted clock starts near 1000 s and gives one time per frame; the trial began at the first of them.
    times = frame_times(len(scenario["path"]) + 5, seed=len(scenario["path"]))
    assert attempt.trial_start_s == times[0]
    for k, sample in enumerate(attempt.samples):
        assert abs(attempt.trial_start_s + sample.t_us / 1e6 - times[k]) <= 1e-6  # t_us is rounded to 1 us
