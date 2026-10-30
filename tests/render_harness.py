"""Scripted pygame, a fake instrument and a fake trace buffer for driving ``run_trial`` in tests."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import numpy as np
import pygame

from experiment import stimulus
from experiment import trial as trial_module
from experiment.signal_scheduler import SignalScheduler

HZ = 60.0


def frame_times(n: int, seed: int) -> list[float]:
    """Frame times near 60 Hz with a little jitter, starting at an arbitrary perf_counter value."""
    rng = np.random.default_rng(seed)
    return (1000.0 + np.arange(n) / HZ + rng.uniform(-0.001, 0.001, n)).tolist()


def polyline(points: list[tuple[float, float, float]], n: int) -> list[tuple[float, float]]:
    """mm positions at each of ``n`` frames along (frame, x, y) waypoints, linearly interpolated."""
    frames = [p[0] for p in points]
    k = np.arange(n)
    return list(zip(np.interp(k, frames, [p[1] for p in points]), np.interp(k, frames, [p[2] for p in points])))


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


class Script:
    """Feeds the trial loop one scripted frame per ``clock.tick``."""

    def __init__(self, scenario: dict, calibration) -> None:
        self.frame = 0
        self.signal: SignalScheduler | None = None
        self.times = frame_times(len(scenario["path"]) + 5, seed=len(scenario["path"]))
        self.positions = [
            (
                round(x * calibration.px_per_mm_x) + calibration.active_left_px,
                round(y * calibration.px_per_mm_y) + calibration.active_top_px,
            )
            for x, y in scenario["path"]
        ]
        self.end = scenario["end"]
        # Optional touch script: {frame: True/False} switches the left button, as the touch frame reports contact.
        self.touch: dict[int, bool] = scenario.get("touch", {})
        self.contact = self.touch.get(0, False)
        # Optional key presses during the trial: {frame: pygame key}.
        self.keys: dict[int, int] = scenario.get("keys", {})

    def get_pressed(self) -> tuple[bool, bool, bool]:
        return (self.contact, False, False)

    def perf_counter(self) -> float:
        return self.times[min(self.frame, len(self.times) - 1)]

    def strftime(self, _fmt: str) -> str:
        return "2026-10-14T00:00:00"

    def get_pos(self) -> tuple[int, int]:
        return self.positions[min(self.frame, len(self.positions) - 1)]

    def tick(self, _fps: int) -> None:
        # A real frame lasts 16 ms, ample for the scheduler; a scripted one must wait for it explicitly.
        if self.signal is not None:
            self.signal.flush()
        self.frame += 1

    def events(self) -> list:
        events = []
        if self.frame in self.touch and self.frame > 0 and self.touch[self.frame] != self.contact:
            self.contact = self.touch[self.frame]
            kind = pygame.MOUSEBUTTONDOWN if self.contact else pygame.MOUSEBUTTONUP
            events.append(pygame.event.Event(kind, button=1, pos=self.get_pos()))
        if self.touch and 0 < self.frame < len(self.positions) and self.positions[self.frame] != self.positions[self.frame - 1]:
            events.append(pygame.event.Event(pygame.MOUSEMOTION, pos=self.get_pos(), rel=(0, 0), buttons=(int(self.contact), 0, 0)))
        if self.frame in self.keys:
            events.append(pygame.event.Event(pygame.KEYDOWN, key=self.keys[self.frame]))
        if self.frame < len(self.positions) - 1 or self.end == "timeout":
            return events
        if self.end == "quit":
            return events + [pygame.event.Event(pygame.QUIT)]
        key = {"left": pygame.K_LEFT, "right": pygame.K_RIGHT, "numpad_right": pygame.K_KP6, "escape": pygame.K_ESCAPE}
        return events + [pygame.event.Event(pygame.KEYDOWN, key=key[self.end])]


def connect(instrument: FakeInstrument, cfg):
    """Run the real connection sequence against the fake instrument (no network)."""

    class FakeResourceManager:
        def open_resource(self, _address: str) -> FakeInstrument:
            return instrument

    with patch("pyvisa.ResourceManager", FakeResourceManager):
        return stimulus.connect_hardware(cfg)




@contextmanager
def scripted_pygame(script):
    """Route the trial loop's clock, mouse, events, drawing and sound to ``script`` and no-ops."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(trial_module, "time", script))
        stack.enter_context(patch.object(pygame.mouse, "get_pos", script.get_pos))
        stack.enter_context(patch.object(pygame.mouse, "get_pressed", getattr(script, "get_pressed", lambda: (False,) * 3)))
        stack.enter_context(patch.object(pygame.event, "get", script.events))
        stack.enter_context(patch.object(pygame.display, "flip", lambda: None))
        stack.enter_context(patch.object(pygame.time, "wait", lambda _ms: None))
        stack.enter_context(patch.object(trial_module.display, "draw_trial", lambda *a, **k: None))
        stack.enter_context(patch.object(trial_module.display, "draw_feedback", lambda *a, **k: None))
        stack.enter_context(patch.object(trial_module.audio_cues, "play_response_cue", lambda: None))
        stack.enter_context(patch.object(trial_module.audio_cues, "play_timeout_cue", lambda: None))
        yield


def run_scripted_trial(cfg, calibration, spec, scenario: dict):
    """One scripted trial against a fake instrument; returns (result, attempt, connect commands, trial commands)."""
    script = Script(scenario, calibration)
    instrument = FakeInstrument()
    attempt = FakeAttempt()
    with scripted_pygame(script):
        connected = connect(instrument, cfg)
        n_connect = len(instrument.commands)
        script.signal = SignalScheduler(stimulus.device_for(connected)).start()
        try:
            result = trial_module.run_trial(
                None, script, calibration, script.signal, cfg, spec, trial_index=1, fps=60, trace_attempt=attempt
            )
        finally:
            script.signal.close()
    return result, attempt, instrument.commands[:n_connect], instrument.commands[n_connect:]
