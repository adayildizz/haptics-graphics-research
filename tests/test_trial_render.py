from __future__ import annotations

import json
import re
import sqlite3
import statistics
import time
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pygame
import pytest
from render_harness import polyline, run_scripted_trial

from analysis.exploration_maps.run import run as run_exploration_maps
from experiment import data_logger, display
from experiment import main as main_module
from experiment import render_logic as rl
from experiment import trial as trial_module
from experiment.calibration import make_configured_ir_frame_calibration
from experiment.config import load_experiment_config
from experiment.constant_stimuli import TrialSpec
from experiment.signal_scheduler import RecordingDevice, SignalScheduler
from experiment.trace_recorder import AsyncTraceRecorder
from experiment.trace_store import AttemptDefinition

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "experiment" / "configs" / "default.yaml"
CALIBRATION = make_configured_ir_frame_calibration((1920, 1080))
SPEC = TrialSpec(level_pct=0.2, comparison_height_mm=12.0, reference_height_mm=10.0, reference_side="left")


def config(**overrides):
    return replace(load_experiment_config(CONFIG_PATH), **overrides)


def to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
    return (
        round(x_mm * CALIBRATION.px_per_mm_x) + CALIBRATION.active_left_px,
        round(y_mm * CALIBRATION.px_per_mm_y) + CALIBRATION.active_top_px,
    )


def events_of(trace_path: Path, event_type: str) -> list[dict]:
    connection = sqlite3.connect(trace_path)
    rows = connection.execute(
        "SELECT attempt_id, t_us, payload_json FROM trace_events WHERE event_type = ? ORDER BY event_id", (event_type,)
    ).fetchall()
    connection.close()
    return [{"attempt_id": a, "t_us": t, **json.loads(payload)} for a, t, payload in rows]


def test_only_the_generator_module_can_switch_the_output():
    for path in sorted((REPO_ROOT / "experiment").glob("*.py")):
        if path.name == "stimulus.py":
            continue
        source = path.read_text()
        assert "OUTP " not in source and "OUTP:" not in source, path.name  # the SCPI output command
        assert not re.search(r"\b(stimulus\.)?signal_(on|off)\(", source), path.name
        assert "instrument.write" not in source, path.name
    # Inside stimulus.py the switch is reachable only through the device handed to the scheduler.
    stimulus_source = (REPO_ROOT / "experiment" / "stimulus.py").read_text()
    assert "deliver_timed_signal" not in stimulus_source
    assert stimulus_source.count("signal_on(self._instrument)") == 1


def test_predictive_and_position_modes_use_the_configured_bar_size_not_the_pixel_rectangle():
    # 67.05 mm is inside the nominal bar (67-77 mm) but one pixel left of the rectangle that is drawn.
    pos = to_px(67.05, 140.0)
    layout = display.make_trial_layout(None, CALIBRATION, 10.0, 10.0, 12.0, False, 40.0)
    assert not layout.left_bar.collidepoint(pos)
    scenario = dict(path=[(67.05, 140.0)] * 12, end="left")
    for mode, expected in (("position", 1), ("predictive", 1), ("timed_legacy", 0)):
        _, attempt, _, _ = run_scripted_trial(config(render_mode=mode), CALIBRATION, SPEC, scenario)
        assert {int(sample.signal_on) for sample in attempt.samples} == {expected}, mode
        assert {sample.active_side for sample in attempt.samples} == ({"left"} if expected else {None}), mode


def test_position_mode_writes_one_event_per_switch_with_times_edge_and_position():
    scenario = dict(path=polyline([(0, 55, 140), (40, 90, 140)], 45), end="right")
    result, attempt, _, commands = run_scripted_trial(config(render_mode="position"), CALIBRATION, SPEC, scenario)
    writes = [payload for _, kind, payload in attempt.events if kind == "signal_write"]
    assert commands == ["VOLT 4.00", "OUTP ON", "OUTP OFF"]
    assert [(w["kind"], w["on"], w["bar"], w["edge"]) for w in writes] == [
        ("plan", True, "left", "left"),
        ("plan", False, "left", "right"),
    ]
    for write in writes:
        assert write["mode"] == "position" and write["write_end_us"] >= write["write_start_us"]
        assert {"planned_us", "x_mm", "y_mm", "vx_mm_s", "vy_mm_s"} <= write.keys()
    assert 67.0 <= writes[0]["x_mm"] < 77.0 and writes[1]["x_mm"] >= 77.0
    # New modes command no duration; the field stays in the pass log as null.
    assert [entry["commanded_duration_s"] for entry in result.passes] == [None]
    assert json.loads(json.dumps(result.passes))[0]["actual_duration_s"] > 0


def test_every_trial_logs_its_sample_intervals_and_repeated_positions():
    path = polyline([(0, 55, 140), (30, 80, 140)], 31) + [(80.0, 140.0)] * 9  # 9 of 40 frames do not move
    for mode in rl.RENDER_MODES:
        _, attempt, _, _ = run_scripted_trial(config(render_mode=mode), CALIBRATION, SPEC, dict(path=path, end="left"))
        (stats,) = [payload for _, kind, payload in attempt.events if kind == "sample_stats"]
        assert stats["mode"] == mode and stats["samples"] == len(attempt.samples)
        assert stats["interval_median_ms"] == pytest.approx(1000 / 60, abs=1.0)
        assert stats["interval_min_ms"] <= stats["interval_median_ms"] <= stats["interval_p95_ms"] <= stats["interval_max_ms"]
        assert stats["repeated_ratio"] == pytest.approx(stats["repeated"] / stats["samples"], abs=1e-3)
        assert 8 <= stats["repeated"] <= 11
        assert attempt.events[-1][1] in ("response",)  # the statistics are in before the attempt is closed
        assert attempt.finished is not None


class LiveTrial:
    """Real-time stand-in for pygame: the finger position is a function of the time since the trial began."""

    def __init__(self, path_at, duration_s: float, stall_at_s: float | None = None, stall_s: float = 0.0) -> None:
        self.path_at = path_at
        self.duration_s = duration_s
        self.stall_at_s = stall_at_s
        self.stall_s = stall_s
        self.t0: float | None = None

    def elapsed(self) -> float:
        if self.t0 is None:
            self.t0 = time.perf_counter()
        return time.perf_counter() - self.t0

    def get_pos(self) -> tuple[int, int]:
        return to_px(*self.path_at(self.elapsed()))

    def events(self) -> list:
        return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)] if self.elapsed() >= self.duration_s else []

    def tick(self, _fps: int) -> None:
        if self.stall_at_s is not None and self.elapsed() >= self.stall_at_s:
            self.stall_at_s = None
            time.sleep(self.stall_s)  # the main loop hangs
        time.sleep(1 / 120)


@contextmanager
def live_pygame(live: LiveTrial):
    with ExitStack() as stack:
        stack.enter_context(patch.object(pygame.mouse, "get_pos", live.get_pos))
        stack.enter_context(patch.object(pygame.mouse, "get_pressed", lambda: (False,) * 3))
        stack.enter_context(patch.object(pygame.event, "get", live.events))
        stack.enter_context(patch.object(pygame.display, "flip", lambda: None))
        stack.enter_context(patch.object(pygame.time, "wait", lambda _ms: None))
        stack.enter_context(patch.object(trial_module.display, "draw_trial", lambda *a, **k: None))
        stack.enter_context(patch.object(trial_module.audio_cues, "play_response_cue", lambda: None))
        yield


def record_live_trial(tmp_path: Path, cfg, live: LiveTrial):
    """One real-time trial with a fake device, recorded to a real trace database."""
    trace_path = tmp_path / "P07_live_trace.sqlite3"
    recorder = AsyncTraceRecorder(trace_path)
    recorder.create_session(
        session_id="P07_live",
        participant_id="P07",
        started_at="2026-10-14T12:00:00+03:00",
        config=main_module.config_to_dict(cfg),
        calibration=CALIBRATION.__dict__,
        app_version="test",
    )
    attempt = recorder.start_attempt(
        "1:1",
        AttemptDefinition("P07_live", 1, 1, recorder.elapsed_us(), 0.2, 12.0, 10.0, cfg.bar_width_mm, "left", False),
    )
    device = RecordingDevice()
    signal = SignalScheduler(device, watchdog_s=cfg.signal_watchdog_ms / 1000.0).start()
    try:
        with live_pygame(live):
            result = trial_module.run_trial(None, live, CALIBRATION, signal, cfg, SPEC, 1, fps=60, trace_attempt=attempt)
    finally:
        signal.close()
        assert recorder.close() is None
    return result, device, trace_path


def sweep_then_probe(t: float) -> tuple[float, float]:
    """Across both bars at 100 mm/s, then down onto the right bar's top (133 mm) and back up."""
    if t < 0.9:
        return 55.0 + 100.0 * t, 140.0
    if t < 1.1:
        return 122.0, 124.0 + 60.0 * (t - 0.9) if t >= 0.9 else 140.0
    if t < 1.4:
        return 122.0, 136.0 - 40.0 * (t - 1.1)
    return 122.0, 124.0


def test_a_predictive_trial_records_timed_writes_and_the_heat_map_script_reads_it(tmp_path):
    cfg = config(render_mode="predictive")
    result, device, trace_path = record_live_trial(tmp_path, cfg, LiveTrial(sweep_then_probe, duration_s=1.6))

    assert isinstance(result, trial_module.TrialResult)
    assert device.states and device.states[-1][1] is False
    writes = events_of(trace_path, "signal_write")
    timed = [w for w in writes if w["kind"] == "transition"]
    assert sum(w["on"] for w in writes) >= 2  # both bars were switched on
    assert timed, "no switch was placed between two frames"
    assert {w["edge"] for w in timed} <= set(rl.EDGES)
    assert {"left", "right"} & {w["edge"] for w in timed}
    late_ms = [(w["write_start_us"] - w["planned_us"]) / 1000.0 for w in timed]
    assert min(late_ms) > -0.01 and statistics.median(late_ms) < 2.0
    assert all(w["mode"] == "predictive" and w["vx_mm_s"] is not None for w in timed)
    assert all(abs(w["vx_mm_s"]) > 5.0 or abs(w["vy_mm_s"]) > 5.0 for w in timed)

    (stats,) = events_of(trace_path, "sample_stats")
    assert stats["mode"] == "predictive" and 4.0 < stats["interval_median_ms"] < 20.0
    assert 0 < stats["horizon_ms"] <= cfg.predict_horizon_max_ms and stats["busy_wait_margin_ms"] >= 1.0
    assert [entry["commanded_duration_s"] for entry in result.passes] == [None] * len(result.passes)

    # The exploration maps read positions only, so a predictive recording must work unchanged.
    out = tmp_path / "maps"
    assert run_exploration_maps(trace_path, "all", out) == 0
    session_dir = out / "P07" / "P07_live"
    assert (session_dir / "main_trial001_attempt1_id1.png").exists() and (session_dir / "aggregate.png").exists()
    with (session_dir / "trials.csv").open() as file:
        header, row = (line.strip().split(",") for line in file.readlines())
    summary = dict(zip(header, row))
    assert float(summary["dwell_sum_ms"]) + float(summary["outside_grid_ms"]) == pytest.approx(
        float(summary["included_ms"]), abs=0.2
    )
    assert float(summary["dwell_sum_ms"]) > 500


def test_a_stalled_main_loop_trips_the_watchdog_and_the_trace_says_so(tmp_path):
    cfg = config(render_mode="predictive", signal_watchdog_ms=100.0)
    inside = lambda t: (72.0 + 0.5 * t, 140.0)  # noqa: E731 -- resting inside the left bar
    live = LiveTrial(inside, duration_s=0.9, stall_at_s=0.3, stall_s=0.3)
    result, device, trace_path = record_live_trial(tmp_path, cfg, live)

    assert isinstance(result, trial_module.TrialResult)
    (watchdog,) = events_of(trace_path, "signal_watchdog")
    assert watchdog["kind"] == "watchdog" and watchdog["on"] is False
    # The signal came back with the first plan after the stall, and that is logged as an ordinary write.
    later_on = [w for w in events_of(trace_path, "signal_write") if w["on"] and w["t_us"] > watchdog["t_us"]]
    assert later_on
    assert [on for _, on in device.states] == [True, False, True, False]
    off_after_s = (device.states[1][0] - device.states[0][0]) - 0.3
    assert 0.05 < off_after_s < 0.2  # about 100 ms into the stall


class FailsOnSixthSwitchOn(RecordingDevice):
    """Two switch-ons per trial (one per bar), so this fails in the middle of the third trial."""

    def __init__(self) -> None:
        super().__init__()
        self.on_calls = 0

    def on(self) -> None:
        self.on_calls += 1
        if self.on_calls == 6:
            raise OSError("VI_ERROR_CONN_LOST")
        super().on()


class SessionScript:
    """Scripted pygame for several trials in a row: one sweep across both bars, then an answer."""

    FRAMES = 60

    def __init__(self, signal: SignalScheduler) -> None:
        self.signal = signal
        self.total = 0
        self.frame = 0
        self.path = polyline([(0, 55, 140), (self.FRAMES - 1, 140, 140)], self.FRAMES)

    def perf_counter(self) -> float:
        return 2000.0 + self.total / 60.0

    def strftime(self, _fmt: str) -> str:
        return "2026-10-14T12:00:00"

    def get_pos(self) -> tuple[int, int]:
        return to_px(*self.path[min(self.frame, self.FRAMES - 1)])

    def tick(self, _fps: int) -> None:
        self.signal.flush()
        self.total += 1
        self.frame += 1

    def events(self) -> list:
        if self.frame < self.FRAMES - 1:
            return []
        self.frame = 0  # the answer ends this trial; the next one starts from the left again
        self.total += 1
        return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)]


def test_a_signal_failure_aborts_the_session_and_everything_before_it_is_on_disk(tmp_path, capsys):
    # Scripted time cannot drive timed transitions (the scheduler runs on the real clock), hence position mode.
    cfg = config(render_mode="position", n_practice_trials=0, n_levels=3, trials_per_level=2, catch_trial_pct=0.0, participant_id="P09")
    trial_path = tmp_path / "P09_session_trials.csv"
    trace_path = tmp_path / "P09_session_trace.sqlite3"
    recorder = AsyncTraceRecorder(trace_path)
    recorder.create_session(
        session_id="P09_session", participant_id="P09", started_at="2026-10-14T12:00:00+03:00",
        config=main_module.config_to_dict(cfg), calibration=CALIBRATION.__dict__, app_version="test",
    )  # fmt: skip
    device = FailsOnSixthSwitchOn()
    signal = SignalScheduler(device).start()
    script = SessionScript(signal)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(trial_module, "time", script))
            stack.enter_context(patch.object(pygame.mouse, "get_pos", script.get_pos))
            stack.enter_context(patch.object(pygame.mouse, "get_pressed", lambda: (False,) * 3))
            stack.enter_context(patch.object(pygame.event, "get", script.events))
            stack.enter_context(patch.object(pygame.display, "flip", lambda: None))
            stack.enter_context(patch.object(pygame.time, "wait", lambda _ms: None))
            stack.enter_context(patch.object(trial_module.display, "draw_trial", lambda *a, **k: None))
            stack.enter_context(patch.object(trial_module.audio_cues, "play_response_cue", lambda: None))
            stack.enter_context(patch.object(main_module, "wait_for_continue_or_exit", lambda _screen, _message: True))
            main_module.run_constant_stimuli(
                None, script, CALIBRATION, signal, cfg, "P09_session", trial_path, seed=3, trace_recorder=recorder
            )
    finally:
        signal.close()
        assert recorder.close() is None

    assert isinstance(signal.error, OSError)
    assert device.states[-1][1] is False and not signal.is_on
    assert "session aborted mid-trial (signal_error)" in capsys.readouterr().out

    # The trial CSV has the two finished trials and the one that was cut short; the fourth never ran.
    rows = data_logger.load_trials(trial_path)
    assert [(row["trial_index"], row["outcome"]) for row in rows] == [(1, "answered"), (2, "answered"), (3, "aborted")]
    assert [row["response"] for row in rows] == ["left", "left", ""]
    assert all(len(row["passes"]) >= 1 for row in rows)

    # The trace has every sample up to the failure, and says why the attempt ended.
    connection = sqlite3.connect(trace_path)
    attempts = connection.execute("SELECT attempt_id, outcome FROM attempts ORDER BY attempt_id").fetchall()
    samples = dict(connection.execute("SELECT attempt_id, COUNT(*) FROM cursor_samples GROUP BY attempt_id").fetchall())
    connection.close()
    assert [outcome for _, outcome in attempts] == ["answered", "answered", "aborted"]
    assert samples[1] == samples[2] == SessionScript.FRAMES - 1 and 10 < samples[3] < SessionScript.FRAMES - 1
    (aborted,) = events_of(trace_path, "aborted")
    assert aborted["reason"] == "signal_error" and aborted["attempt_id"] == 3
    assert [w["kind"] for w in events_of(trace_path, "signal_write") if w["attempt_id"] == 3][-1] == "error"
    assert len(events_of(trace_path, "sample_stats")) == 3


def test_main_starts_and_closes_the_scheduler_and_names_the_render_mode(monkeypatch, tmp_path, capsys):
    import threading

    from experiment import signal_scheduler, stimulus

    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setattr("sys.argv", ["experiment.main", "--config", str(CONFIG_PATH), "--participant", "T1", "--windowed"])
    monkeypatch.setattr(stimulus, "connect_hardware", lambda _cfg: None)  # never reach for the generator
    monkeypatch.setattr(data_logger, "ensure_data_dir", lambda *_: tmp_path)
    monkeypatch.setattr(signal_scheduler, "exit_cleanly_on_sigterm", lambda: None)
    shown: list[str] = []
    monkeypatch.setattr(main_module, "wait_for_continue_or_exit", lambda _screen, message: shown.append(message) or False)

    assert main_module.run() == 0

    assert "render: predictive" in shown[0]  # the operator's start screen
    out = capsys.readouterr().out
    assert "Render mode: predictive (window 3 samples" in out and "watchdog 100 ms" in out
    assert not any(t.name == "signal-scheduler" and t.is_alive() for t in threading.enumerate())
    (snapshot,) = tmp_path.glob("T1_*_config.json")
    assert json.loads(snapshot.read_text())["render_mode"] == "predictive"
    (trace,) = tmp_path.glob("T1_*_trace.sqlite3")
    connection = sqlite3.connect(trace)
    (config_json,) = connection.execute("SELECT config_json FROM sessions").fetchone()
    connection.close()
    assert json.loads(config_json)["latency_compensation_ms"] == 0.0


def test_the_felt_gap_key_is_logged_with_the_mode_and_never_answers():
    scenario = dict(
        path=polyline([(0, 55, 140), (40, 90, 140)], 45),
        end="right",
        keys={10: pygame.K_DOWN, 20: pygame.K_KP2, 25: pygame.K_DOWN},
    )
    result, attempt, _, _ = run_scripted_trial(config(render_mode="position", felt_gap_key=True), CALIBRATION, SPEC, scenario)
    felt = [payload for _, kind, payload in attempt.events if kind == "felt_gap"]
    assert result.response == "right" and result.felt_gaps == 3 and len(felt) == 3
    assert {payload["mode"] for payload in felt} == {"position"}
    assert all({"x_mm", "y_mm", "signal_on", "contact", "active_side"} <= payload.keys() for payload in felt)
    assert felt[0]["x_mm"] is not None


def test_the_felt_gap_key_does_nothing_unless_switched_on():
    scenario = dict(path=polyline([(0, 55, 140), (40, 90, 140)], 45), end="right", keys={10: pygame.K_DOWN})
    result, attempt, _, _ = run_scripted_trial(config(render_mode="position"), CALIBRATION, SPEC, scenario)
    assert result.felt_gaps == 0 and not [e for e in attempt.events if e[1] == "felt_gap"]


@pytest.mark.parametrize("assigned", ["kalman", "position", "predictive"])
def test_a_mixed_session_runs_each_trial_in_the_mode_it_drew(assigned):
    scenario = dict(path=polyline([(0, 55, 140), (40, 90, 140)], 45), end="right")
    spec = replace(SPEC, render_mode=assigned)
    _, attempt, _, commands = run_scripted_trial(config(render_mode="mixed"), CALIBRATION, spec, scenario)
    (stats,) = [payload for _, kind, payload in attempt.events if kind == "sample_stats"]
    assert stats["mode"] == assigned
    assert {payload["mode"] for _, kind, payload in attempt.events if kind == "signal_write"} == {assigned}
    assert "OUTP ON" in commands and commands[-1] == "OUTP OFF"
