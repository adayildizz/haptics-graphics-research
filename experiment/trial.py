"""Single-trial orchestration for the tactile bar-height 2AFC task."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from typing import TYPE_CHECKING

import pygame

from . import audio_cues, display, stimulus
from .calibration import DisplayCalibration
from .config import ExperimentConfig
from .constant_stimuli import TrialSpec
from .input_keys import is_exit_key, response_for_key
from .signal_scheduler import SignalScheduler
from .trace_store import CursorSample
from .trial_render import LEGACY, TrialRenderer

if TYPE_CHECKING:
    from .speed_coach import PracticeSpeedCoach
    from .trace_recorder import AttemptTraceBuffer


@dataclass(frozen=True)
class TrialResult:
    trial_index: int
    level_pct: float
    comparison_height_mm: float
    reference_height_mm: float
    bar_width_mm: float
    reference_side: str
    response: str
    correct: bool
    is_catch: bool
    is_practice: bool
    response_time_s: float
    passes: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""


@dataclass(frozen=True)
class TrialAborted:
    """The trial was cut short by an exit key or a closed window.

    Returned instead of ``None`` so the caller can log the partial trial
    before shutting the session down: an abort is a real thing that
    happened to a participant, and the trial CSV should say so rather than
    the row simply being absent.
    """

    reason: str  # "exit_key" | "window_closed" | "signal_error"
    elapsed_s: float
    passes: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""


@dataclass(frozen=True)
class TrialTimeout:
    """Signals that a trial expired without a response (practice included).

    Carries the same exploration record an answered trial does: the passes
    show whether the participant was working the bars or idle when the
    window closed, which is the interesting question about a timeout.
    """

    elapsed_s: float
    passes: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""


def should_time_out(elapsed_s: float, timeout_s: float) -> bool:
    """Every trial, practice included, uses the configured response limit.

    Practice used to be unlimited, which meant participants first met the
    countdown in the main block -- the one place where running out of time
    actually costs a trial. Practicing under the same clock is the point of
    practice.
    """
    return elapsed_s >= timeout_s


class _PassTracker:
    """Tracks per-bar-crossing ("pass") rendering fidelity within a trial."""

    def __init__(self, max_sample_gap_s: float) -> None:
        self.max_sample_gap_s = max_sample_gap_s
        self.passes: list[dict[str, Any]] = []
        self.side: str | None = None
        self.start_s = 0.0
        self.commanded_s: float | None = 0.0
        self.entry_speed_mm_s = 0.0
        self.leading_edge_detected = True

    def start(
        self, side: str, now_s: float, speed_mm_s: float, commanded_s: float | None, sample_gap_s: float
    ) -> None:
        self.finish(now_s)
        self.side = side
        self.start_s = now_s
        self.commanded_s = commanded_s
        self.entry_speed_mm_s = speed_mm_s
        self.leading_edge_detected = sample_gap_s <= self.max_sample_gap_s

    def finish(self, now_s: float) -> None:
        if self.side is None:
            return
        actual_s = max(0.0, now_s - self.start_s)
        if self.commanded_s is not None:  # only the timed mode commands a duration
            actual_s = min(self.commanded_s, actual_s)
        self.passes.append(
            {
                "side": self.side,
                "commanded_duration_s": self.commanded_s,
                "actual_duration_s": actual_s,
                "finger_speed_mm_s": self.entry_speed_mm_s,
                "leading_edge_detected": self.leading_edge_detected,
            }
        )
        self.side = None


def _taller_side(spec: TrialSpec) -> str:
    """The objectively taller side, given the signed level percentage."""
    if spec.level_pct > 0:
        return "right" if spec.reference_side == "left" else "left"
    return spec.reference_side


def run_trial(
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    calibration: DisplayCalibration,
    signal: SignalScheduler,
    cfg: ExperimentConfig,
    spec: TrialSpec,
    trial_index: int,
    fps: int,
    speed_coach: PracticeSpeedCoach | None = None,
    trace_attempt: AttemptTraceBuffer | None = None,
    feedback_speaker: Callable[[str], object] | None = None,
    total_trials: int | None = None,
) -> TrialResult | TrialTimeout | TrialAborted:
    """Run one 2AFC trial, returning a response, a timeout, or an abort.

    ``feedback_speaker`` is called with "Correct" / "Incorrect" whenever
    feedback is shown. The participant is blindfolded, so the on-screen word
    is for the operator; the spoken one is the feedback the participant
    actually gets. It is expected to block until the word has been said.
    """
    comparison_side = "right" if spec.reference_side == "left" else "left"
    left_is_comparison = comparison_side == "left"
    layout = display.make_trial_layout(
        screen,
        calibration=calibration,
        bar_width_mm=cfg.bar_width_mm,
        reference_height_mm=spec.reference_height_mm,
        comparison_height_mm=spec.comparison_height_mm,
        left_is_comparison=left_is_comparison,
        inter_bar_gap_mm=cfg.inter_bar_gap_mm,
    )

    trial_start = time.perf_counter()
    last_pos = pygame.mouse.get_pos()
    # The touch frame reports a finger on the glass as the left mouse button held down.
    contact = bool(pygame.mouse.get_pressed()[0])
    last_time = trial_start
    previous_side: str | None = None
    previous_signal_on = False
    trace_sequence = 0

    if speed_coach is not None and spec.is_practice:
        speed_coach.reset_trial()

    tracker = _PassTracker(max_sample_gap_s=3.0 / cfg.ir_sample_hz_nominal)
    renderer = TrialRenderer(cfg, calibration, layout, signal, trial_start, fps, trace_attempt)

    def aborted(reason: str, now: float) -> TrialAborted:
        renderer.finish(reason, now)
        tracker.finish(now)
        if trace_attempt is not None:
            t_us = round((now - trial_start) * 1_000_000)
            if previous_signal_on:
                trace_attempt.add_event(t_us=t_us, event_type="signal_off", payload={"reason": reason})
            trace_attempt.add_event(t_us=t_us, event_type="aborted", payload={"reason": reason})
            trace_attempt.finish(ended_us=trace_attempt.session_elapsed_us(), outcome="aborted")
        return TrialAborted(
            reason=reason,
            elapsed_s=now - trial_start,
            passes=tracker.passes,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    while True:
        now = time.perf_counter()
        if renderer.error is not None:
            # Without a working signal the trial is not the experiment any more: stop here, keep what was recorded.
            return aborted("signal_error", now)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                contact = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                contact = False
            elif event.type == pygame.MOUSEMOTION:
                contact = bool(event.buttons[0])
            if event.type == pygame.QUIT:
                return aborted("window_closed", now)
            if event.type == pygame.KEYDOWN:
                if is_exit_key(event.key, pygame):
                    return aborted("exit_key", now)
                response = response_for_key(event.key, pygame)
                if response is not None:
                    renderer.finish("response", now)
                    tracker.finish(now)
                    # Same cue whether the answer was right or wrong: it marks
                    # "recorded, moving on", not correctness.
                    audio_cues.play_response_cue()
                    correct = response == _taller_side(spec)
                    result = TrialResult(
                        trial_index=trial_index,
                        level_pct=spec.level_pct,
                        comparison_height_mm=spec.comparison_height_mm,
                        reference_height_mm=spec.reference_height_mm,
                        bar_width_mm=cfg.bar_width_mm,
                        reference_side=spec.reference_side,
                        response=response,
                        correct=correct,
                        is_catch=spec.is_catch,
                        is_practice=spec.is_practice,
                        response_time_s=now - trial_start,
                        passes=tracker.passes,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    )
                    if trace_attempt is not None:
                        t_us = round((now - trial_start) * 1_000_000)
                        if previous_signal_on:
                            trace_attempt.add_event(
                                t_us=t_us,
                                event_type="signal_off",
                                payload={"reason": "response"},
                            )
                        trace_attempt.add_event(
                            t_us=t_us,
                            event_type="response",
                            payload={"response": response, "correct": correct},
                        )
                        trace_attempt.finish(
                            ended_us=trace_attempt.session_elapsed_us(),
                            outcome="answered",
                            response=response,
                            response_time_us=t_us,
                        )
                    if cfg.feedback or spec.is_practice:
                        display.draw_feedback(screen, correct)
                        pygame.display.flip()
                        if feedback_speaker is not None:
                            feedback_speaker("Correct" if correct else "Incorrect")
                            pygame.time.wait(250)
                        else:
                            pygame.time.wait(500)
                    return result

        elapsed_trial_s = now - trial_start
        if should_time_out(elapsed_trial_s, cfg.response_timeout_s):
            renderer.finish("timeout", now)
            tracker.finish(now)
            audio_cues.play_timeout_cue()
            if trace_attempt is not None:
                t_us = round(elapsed_trial_s * 1_000_000)
                if previous_signal_on:
                    trace_attempt.add_event(
                        t_us=t_us,
                        event_type="signal_off",
                        payload={"reason": "timeout"},
                    )
                trace_attempt.add_event(
                    t_us=t_us,
                    event_type="timeout",
                    payload={"limit_s": cfg.response_timeout_s},
                )
                trace_attempt.finish(
                    ended_us=trace_attempt.session_elapsed_us(),
                    outcome="timeout",
                )
            return TrialTimeout(
                elapsed_s=elapsed_trial_s,
                passes=tracker.passes,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )

        pos = pygame.mouse.get_pos()
        x_mm = (pos[0] - calibration.active_left_px) / calibration.px_per_mm_x
        y_mm = (pos[1] - calibration.active_top_px) / calibration.px_per_mm_y
        # Through the IR glitch filter: from here on the frame works with what the renderer takes from the report.
        finger = renderer.read(now, pos, x_mm, y_mm, contact)
        elapsed = max(now - last_time, 1e-6)
        dx = (finger.pos[0] - last_pos[0]) / calibration.px_per_mm_x
        dy = (finger.pos[1] - last_pos[1]) / calibration.px_per_mm_y
        speed_mm_s = ((dx * dx + dy * dy) ** 0.5) / elapsed

        if speed_coach is not None and spec.is_practice:
            speed_coach.update(
                speed_mm_s,
                elapsed,
                now,
                in_active_area=layout.haptic_area.collidepoint(pos),
            )

        # ``signal_on`` is what this frame asked for; the scheduler logs when the output really switched.
        decision = renderer.frame(now, finger, speed_mm_s)
        signal_on, active_side = decision.signal_on, decision.active_side
        if active_side is not None and active_side != previous_side:
            commanded_s = stimulus.stimulus_duration(cfg.bar_width_mm, speed_mm_s) if cfg.render_mode == LEGACY else None
            tracker.start(active_side, now, speed_mm_s, commanded_s, elapsed)
        elif active_side is None and previous_side is not None:
            tracker.finish(now)

        if trace_attempt is not None:
            t_us = round(elapsed_trial_s * 1_000_000)
            trace_attempt.add_sample(
                CursorSample(
                    sequence=trace_sequence,
                    t_us=t_us,
                    frame_dt_us=round(elapsed * 1_000_000),
                    x_px=pos[0],
                    y_px=pos[1],
                    x_mm=x_mm,
                    y_mm=y_mm,
                    speed_mm_s=speed_mm_s,
                    in_active_area=layout.haptic_area.collidepoint(pos),
                    active_side=active_side,
                    signal_on=signal_on,
                    contact=decision.finger_down,
                    used_x_mm=finger.x_mm,
                    used_y_mm=finger.y_mm,
                    used_contact=finger.down,
                )
            )
            trace_sequence += 1
            if signal_on != previous_signal_on:
                trace_attempt.add_event(
                    t_us=t_us,
                    event_type="signal_on" if signal_on else "signal_off",
                    payload={"active_side": active_side},
                )
        previous_signal_on = signal_on

        display.draw_trial(
            screen,
            layout,
            bar_width_mm=cfg.bar_width_mm,
            trial_index=trial_index,
            is_practice=spec.is_practice,
            active_side=active_side,
            blind_test_mode=cfg.blind_test_mode,
            touch_pos=pos,
            remaining_time_s=max(0.0, cfg.response_timeout_s - elapsed_trial_s),
            total_trials=total_trials,
        )
        pygame.display.flip()
        previous_side = active_side
        last_pos = finger.pos
        last_time = now
        clock.tick(fps)
