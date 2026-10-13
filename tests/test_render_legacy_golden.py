"""The timed render rule, pinned against output recorded from the code before the render refactor."""

# data/render_legacy_golden.json was written by running this file with --write on commit 7af43d7.

from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pygame
import pytest

from experiment import stimulus
from experiment import trial as trial_module
from experiment.calibration import make_configured_ir_frame_calibration
from experiment.config import load_experiment_config
from experiment.constant_stimuli import TrialSpec

GOLDEN_PATH = Path(__file__).parent / "data" / "render_legacy_golden.json"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "experiment" / "configs" / "default.yaml"
SCREEN = (1920, 1080)
HZ = 60.0
LEGACY_EVENTS = ("signal_on", "signal_off", "response", "timeout", "aborted")


def _frame_times(n: int, seed: int) -> list[float]:
    """Frame times near 60 Hz with a little jitter, starting at an arbitrary perf_counter value."""
    rng = np.random.default_rng(seed)
    return (1000.0 + np.arange(n) / HZ + rng.uniform(-0.001, 0.001, n)).tolist()


def _polyline(points: list[tuple[float, float, float]], n: int) -> list[tuple[float, float]]:
    """mm positions at each of ``n`` frames along (frame, x, y) waypoints, linearly interpolated."""
    frames = [p[0] for p in points]
    k = np.arange(n)
    return list(zip(np.interp(k, frames, [p[1] for p in points]), np.interp(k, frames, [p[2] for p in points])))


def scenarios() -> dict[str, dict]:
    """Paths in surface mm over the default layout: bars at x 67-77 and 117-127, baseline y 145."""
    jitter = np.random.default_rng(5).uniform(-0.15, 0.15, (150, 2))
    noisy = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(_polyline([(0, 55, 140), (149, 140, 139)], 150), jitter)]
    return {
        "horizontal_sweeps": dict(
            path=_polyline([(0, 50, 140), (60, 140, 140), (90, 50, 140), (140, 140, 141)], 150), end="left"
        ),
        "fast_sweep": dict(path=_polyline([(0, 40, 141), (30, 150, 141), (60, 40, 141)], 70), end="right"),
        "top_edge_probing": dict(
            path=_polyline(
                [(0, 72, 125), (50, 72, 138), (60, 72, 138), (70, 72, 128), (100, 72, 137), (106, 72, 127), (130, 72, 127)],
                140,
            ),
            end="left",
        ),
        "corner_cut": dict(path=_polyline([(0, 60, 130), (20, 72, 137), (22, 79, 139), (40, 100, 139)], 50), end="right"),
        "stop_inside": dict(
            path=_polyline([(0, 60, 141), (20, 72, 141), (50, 72, 141), (120, 82, 141)], 130), end="left"
        ),
        "quick_dip": dict(
            path=_polyline([(0, 60, 141), (8, 68, 141), (10, 70, 133), (30, 70, 120)], 40), end="numpad_right"
        ),
        "bar_to_bar_jump": dict(
            path=_polyline([(0, 60, 141), (10, 72, 141), (11, 122, 141), (30, 124, 141), (40, 140, 141)], 50),
            end="left",
        ),
        "noisy_sweep": dict(path=noisy, end="left"),
        "comparison_on_left": dict(
            path=_polyline([(0, 50, 139), (60, 140, 139), (100, 72, 128)], 110), end="right", reference_side="right"
        ),
        "ends_by_timeout": dict(path=_polyline([(0, 60, 141), (90, 130, 141)], 100), end="timeout"),
        "ends_by_exit_key": dict(path=_polyline([(0, 60, 141), (40, 75, 141)], 45), end="escape"),
        "ends_by_window_close": dict(path=_polyline([(0, 60, 141), (40, 75, 141)], 45), end="quit"),
    }


class FakeInstrument:
    """Stands in for the VISA resource and records every command, in order."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.write_termination = self.read_termination = ""

    def write(self, command: str) -> None:
        self.commands.append(command)

    def query(self, _command: str) -> str:
        return "FAKE,GENERATOR"

    def close(self) -> None:
        self.commands.append("<close>")


class FakeAttempt:
    """Stands in for the trace buffer and keeps what the trial loop hands it."""

    def __init__(self) -> None:
        self.samples: list = []
        self.events: list[tuple[int, str, dict]] = []
        self.finished: dict | None = None

    def add_sample(self, sample) -> None:
        self.samples.append(sample)

    def add_event(self, *, t_us: int, event_type: str, payload: dict | None = None) -> None:
        self.events.append((t_us, event_type, payload or {}))

    def session_elapsed_us(self) -> int:
        return 0

    def finish(self, **kwargs) -> None:
        self.finished = kwargs


class _Script:
    """Feeds the trial loop one scripted frame per ``clock.tick``."""

    def __init__(self, scenario: dict, calibration) -> None:
        self.frame = 0
        self.times = _frame_times(len(scenario["path"]) + 5, seed=len(scenario["path"]))
        self.positions = [
            (
                round(x * calibration.px_per_mm_x) + calibration.active_left_px,
                round(y * calibration.px_per_mm_y) + calibration.active_top_px,
            )
            for x, y in scenario["path"]
        ]
        self.end = scenario["end"]

    def perf_counter(self) -> float:
        return self.times[min(self.frame, len(self.times) - 1)]

    def strftime(self, _fmt: str) -> str:
        return "2026-10-14T00:00:00"

    def get_pos(self) -> tuple[int, int]:
        return self.positions[min(self.frame, len(self.positions) - 1)]

    def tick(self, _fps: int) -> None:
        self.frame += 1

    def events(self) -> list:
        if self.frame < len(self.positions) - 1 or self.end == "timeout":
            return []
        if self.end == "quit":
            return [pygame.event.Event(pygame.QUIT)]
        key = {"left": pygame.K_LEFT, "right": pygame.K_RIGHT, "numpad_right": pygame.K_KP6, "escape": pygame.K_ESCAPE}
        return [pygame.event.Event(pygame.KEYDOWN, key=key[self.end])]


def connect(instrument: FakeInstrument, cfg):
    """Run the real connection sequence against the fake instrument (no network)."""

    class FakeResourceManager:
        def open_resource(self, _address: str) -> FakeInstrument:
            return instrument

    with patch("pyvisa.ResourceManager", FakeResourceManager):
        return stimulus.connect_hardware(cfg)


def run_scenario(name: str, mode: str | None = None) -> dict:
    """One scripted trial; returns everything the render rule produced."""
    scenario = scenarios()[name]
    cfg = load_experiment_config(CONFIG_PATH)
    cfg = replace(cfg, response_timeout_s=1.2 if scenario["end"] == "timeout" else 30.0)
    if mode is not None:
        cfg = replace(cfg, render_mode=mode)
    calibration = make_configured_ir_frame_calibration(SCREEN)
    spec = TrialSpec(
        level_pct=0.2,
        comparison_height_mm=12.0,
        reference_height_mm=10.0,
        reference_side=scenario.get("reference_side", "left"),
    )
    script = _Script(scenario, calibration)
    instrument = FakeInstrument()
    attempt = FakeAttempt()
    with ExitStack() as stack:
        stack.enter_context(patch.object(trial_module, "time", script))
        stack.enter_context(patch.object(pygame.mouse, "get_pos", script.get_pos))
        stack.enter_context(patch.object(pygame.event, "get", script.events))
        stack.enter_context(patch.object(pygame.display, "flip", lambda: None))
        stack.enter_context(patch.object(pygame.time, "wait", lambda _ms: None))
        stack.enter_context(patch.object(trial_module.display, "draw_trial", lambda *a, **k: None))
        stack.enter_context(patch.object(trial_module.audio_cues, "play_response_cue", lambda: None))
        stack.enter_context(patch.object(trial_module.audio_cues, "play_timeout_cue", lambda: None))
        result, trial_commands = _run_trial(cfg, calibration, spec, script, instrument, attempt)
    return {
        "connect_commands": instrument.commands[: len(instrument.commands) - len(trial_commands)],
        "trial_commands": trial_commands,
        "signal_on": [int(sample.signal_on) for sample in attempt.samples],
        "active_side": [sample.active_side for sample in attempt.samples],
        "events": [[t_us, kind, payload] for t_us, kind, payload in attempt.events if kind in LEGACY_EVENTS],
        "finished": attempt.finished,
        "result": type(result).__name__,
        "response": getattr(result, "response", None),
        "passes": [
            {key: round(value, 9) if isinstance(value, float) else value for key, value in entry.items()}
            for entry in result.passes
        ],
    }


def _run_trial(cfg, calibration, spec, script, instrument, attempt):
    connected = connect(instrument, cfg)
    n_connect = len(instrument.commands)
    result = trial_module.run_trial(
        None, script, calibration, connected, cfg, spec, trial_index=1, fps=60, trace_attempt=attempt
    )
    return result, instrument.commands[n_connect:]


@pytest.mark.parametrize("name", sorted(scenarios()))
def test_timed_render_rule_matches_the_recording_from_before_the_refactor(name):
    golden = json.loads(GOLDEN_PATH.read_text())[name]
    assert json.loads(json.dumps(run_scenario(name))) == golden


def test_the_scenarios_exercise_the_timed_tail():
    golden = json.loads(GOLDEN_PATH.read_text())
    # Signal on while the finger is outside both bars is the timed tail; the pin is worthless without it.
    tail_frames = sum(
        on and side is None for run in golden.values() for on, side in zip(run["signal_on"], run["active_side"])
    )
    assert tail_frames > 20
    assert {run["result"] for run in golden.values()} == {"TrialResult", "TrialTimeout", "TrialAborted"}


if __name__ == "__main__" and "--write" in sys.argv:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps({name: run_scenario(name) for name in sorted(scenarios())}, indent=1))
    print(f"wrote {GOLDEN_PATH}")
