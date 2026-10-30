"""Per-trial glue between the trial loop, the render decisions and the signal scheduler."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from . import render_logic as rl
from .calibration import DisplayCalibration
from .config import MAX_SIGNAL_DURATION_S, MIN_SPEED_MM_S, ExperimentConfig
from .signal_scheduler import SignalScheduler, WriteRecord

if TYPE_CHECKING:
    from .display import TrialLayout
    from .trace_recorder import AttemptTraceBuffer

LEGACY = "timed_legacy"
# One per process, i.e. per session: once the device has reported a touch, contact is trusted from then on.
CONTACT = rl.ContactGate()
_warned_no_touch = False


def reset_contact_gate() -> None:
    """Forget what the input device has shown (tests, or a new device)."""
    global CONTACT, _warned_no_touch
    CONTACT = rl.ContactGate()
    _warned_no_touch = False


@dataclass(frozen=True)
class Finger:
    """One frame's IR report as the renderer takes it: through the glitch filter when ``ir_filter`` is on."""

    pos: tuple[int, int]  # the report's own, or the last accepted one's while a glitch is ignored
    x_mm: float
    y_mm: float
    down: bool  # contact as the renderer takes it
    reported_down: bool  # the report's contact through ContactGate: what the trace records as ``contact``
    contact: bool  # the left-button state the device reported


@dataclass(frozen=True)
class FrameDecision:
    signal_on: bool  # the state this frame asks for
    active_side: str | None  # the bar under the finger
    finger_down: bool  # contact as recorded: True also while the device has not shown it reports touch


class TrialRenderer:
    """Turns each frame's finger sample into a plan, submits it, and logs every write that results."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        calibration: DisplayCalibration,
        layout: TrialLayout,
        signal: SignalScheduler,
        trial_start_s: float,
        fps: int,
        trace_attempt: AttemptTraceBuffer | None,
    ) -> None:
        self.mode = cfg.render_mode
        self._layout = layout
        self._signal = signal
        self._trial_start_s = trial_start_s
        self._trace = trace_attempt
        self._clock = rl.SampleClock(1.0 / fps)
        self._previous: rl.Sample | None = None
        self._last_pos: tuple[int, int] | None = None
        self._last_contact: bool | None = None
        self._horizon_s = 0.0
        left_is_comparison = layout.left_is_comparison
        heights = (
            layout.comparison_height_mm if left_is_comparison else layout.reference_height_mm,
            layout.reference_height_mm if left_is_comparison else layout.comparison_height_mm,
        )
        # Decisions use the bars at their configured size; the drawing keeps its pixel-rounded rectangles.
        self._nominal = rl.nominal_bars(
            calibration.active_width_mm, calibration.active_height_mm, cfg.inter_bar_gap_mm, cfg.bar_width_mm, *heights
        )
        self._pixel = rl.pixel_bars(
            calibration.active_left_px,
            calibration.active_top_px,
            calibration.active_width_px,
            calibration.active_height_px,
            calibration.px_per_mm_x,
            calibration.px_per_mm_y,
            cfg.inter_bar_gap_mm,
            cfg.bar_width_mm,
            *heights,
        )
        self._cfg = cfg
        self._fps = fps
        self._quantum_mm = max(1.0 / calibration.px_per_mm_x, 1.0 / calibration.px_per_mm_y)
        self._lifted = False
        self._ir = (
            rl.IrGlitchFilter(cfg.ir_lift_debounce_ms / 1000.0, cfg.ir_jump_mm, cfg.ir_jump_reanchor_ms / 1000.0)
            if cfg.ir_filter
            else None
        )
        self._used_pos: tuple[int, int] | None = None
        self._start_afresh()

    def _start_afresh(self) -> None:
        """New decision state: at the start of a trial and whenever the finger comes back onto the glass."""
        cfg = self._cfg
        self._previous = None
        self._legacy = rl.LegacyTimer(cfg.bar_width_mm, MIN_SPEED_MM_S, MAX_SIGNAL_DURATION_S)
        self._planner: rl.PredictivePlanner | rl.PositionPlanner | None = None
        if self.mode == "predictive":
            self._planner = rl.PredictivePlanner(
                self._nominal,
                window=cfg.predict_window_samples,
                horizon_factor=cfg.predict_horizon_factor,
                horizon_max_s=cfg.predict_horizon_max_ms / 1000.0,
                min_speed_mm_s=cfg.predict_min_speed_mm_s,
                latency_s=cfg.latency_compensation_ms / 1000.0,
                nominal_interval_s=1.0 / self._fps,
                position_quantum_mm=self._quantum_mm,
            )
        elif self.mode == "position":
            self._planner = rl.PositionPlanner(self._nominal)

    @property
    def error(self) -> BaseException | None:
        return self._signal.error

    def read(self, now_s: float, pos: tuple[int, int], x_mm: float, y_mm: float, contact: bool) -> Finger:
        """This frame's IR report as the renderer will take it; ``contact`` is the left-button state reported."""
        self._clock.add(now_s, repeated=pos == self._last_pos)
        self._last_pos = pos
        reported_down = CONTACT.finger_down(contact)
        if self._ir is None:
            return Finger(pos, x_mm, y_mm, reported_down, reported_down, contact)
        # The time the trace records, so that the replay, filtering the recording, decides exactly alike.
        reading = self._ir.step(self._us(now_s) / 1_000_000, x_mm, y_mm, reported_down)
        self._log_glitches(now_s, self._ir.glitches)
        if not reading.held:
            self._used_pos = pos
        assert self._used_pos is not None  # the filter holds only a position it accepted before
        return Finger(self._used_pos, reading.x_mm, reading.y_mm, reading.down, reported_down, contact)

    def frame(self, now_s: float, finger: Finger, speed_mm_s: float) -> FrameDecision:
        """Decide and submit the plan for one frame, from what ``read`` made of its IR report."""
        pos, x_mm, y_mm, finger_down = finger.pos, finger.x_mm, finger.y_mm, finger.down
        sample = rl.Sample(now_s, x_mm, y_mm)
        touched_down = finger_down and self._lifted
        if touched_down:
            # The cursor sat still while the finger was up and has now jumped: no mode may build on that.
            self._start_afresh()
        self._lifted = not finger_down
        if not finger_down:
            # Every mode, one rule: a lifted finger feels nothing, so the signal is off.
            side = None
            plan = rl.Plan(now_s, False, rl.side_at(self._nominal, x_mm, y_mm), "lift", (), x_mm, y_mm)
        elif self._planner is None:
            side = self._pixel_side(pos)
            on = self._legacy.step(now_s, side, speed_mm_s)
            bar, edge = rl.explain_switch(self._previous, sample, self._pixel, on)
            plan = rl.Plan(now_s, on, bar, "timer" if on and side is None else edge, (), x_mm, y_mm)
        else:
            plan = self._planner.plan(sample)
            side = rl.side_at(self._nominal, x_mm, y_mm)
            self._horizon_s = plan.horizon_s
        if touched_down and plan.on:
            plan = replace(plan, edge="touch")
        self._previous = sample if finger_down else None
        self._signal.submit(plan)
        self._log(self._signal.drain())
        self._log_contact(now_s, finger.contact, side)
        return FrameDecision(plan.on, side, finger.reported_down)

    def finish(self, reason: str, now_s: float) -> None:
        """Signal off; every write and the trial's sampling statistics are in the trace before this returns."""
        global _warned_no_touch
        self._log(self._signal.stop_trial(reason))
        if self._ir is not None:
            glitch = self._ir.finish(self._us(now_s) / 1_000_000)
            self._log_glitches(now_s, () if glitch is None else (glitch,))
        if not CONTACT.device_reports_touch and not _warned_no_touch:
            _warned_no_touch = True
            print(
                "WARNING: the input device has not reported a finger touching the surface; contact is recorded "
                "as 1 (unknown) and the signal is not switched off when the finger is lifted."
            )
        if self._trace is not None:
            payload: dict[str, Any] = {
                "mode": self.mode,
                "device_reports_touch": CONTACT.device_reports_touch,
                **self._clock.summary(),
                "horizon_ms": round(1000.0 * self._horizon_s, 3),
                "busy_wait_margin_ms": round(1000.0 * (self._signal.spin_s or 0.0), 3),
            }
            self._trace.add_event(t_us=self._us(now_s), event_type="sample_stats", payload=payload)

    def _log_contact(self, now_s: float, contact: bool, side: str | None) -> None:
        """contact_down / contact_up events, once the device has shown that it reports touch."""
        previous, self._last_contact = self._last_contact, contact
        if self._trace is None or previous is None or contact == previous or not CONTACT.device_reports_touch:
            return
        self._trace.add_event(
            t_us=self._us(now_s),
            event_type="contact_down" if contact else "contact_up",
            payload={"active_side": side},
        )

    def _log_glitches(self, now_s: float, glitches: tuple[rl.IrGlitch, ...]) -> None:
        """One event per IR glitch the filter ignored, when it is over."""
        if self._trace is None:
            return
        for glitch in glitches:
            payload: dict[str, Any] = {"duration_ms": round(1000.0 * glitch.duration_s, 3)}
            if glitch.distance_mm is not None:
                payload["distance_mm"] = round(glitch.distance_mm, 3)
            if glitch.ended:
                payload["ended"] = glitch.ended
            self._trace.add_event(t_us=self._us(now_s), event_type=glitch.kind, payload=payload)

    def _pixel_side(self, pos: tuple[int, int]) -> str | None:
        if self._layout.left_bar.collidepoint(pos):
            return "left"
        if self._layout.right_bar.collidepoint(pos):
            return "right"
        return None

    def _us(self, t_s: float) -> int:
        return round((t_s - self._trial_start_s) * 1_000_000)

    def _log(self, records: list[WriteRecord]) -> None:
        if self._trace is None:
            return
        for record in records:
            self._trace.add_event(
                t_us=self._us(record.write_start_s),
                # A watchdog switch-off is a fault, not a render decision: it gets its own event type.
                event_type="signal_watchdog" if record.kind == "watchdog" else "signal_write",
                payload={
                    "kind": record.kind,
                    "on": record.on,
                    "mode": self.mode,
                    "planned_us": self._us(record.planned_s),
                    "write_start_us": self._us(record.write_start_s),
                    "write_end_us": self._us(record.write_end_s),
                    "bar": record.bar,
                    "edge": record.edge,
                    "x_mm": record.x_mm,
                    "y_mm": record.y_mm,
                    "vx_mm_s": record.vx_mm_s,
                    "vy_mm_s": record.vy_mm_s,
                },
            )
