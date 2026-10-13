"""Per-trial glue between the trial loop, the render decisions and the signal scheduler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import render_logic as rl
from .calibration import DisplayCalibration
from .config import MAX_SIGNAL_DURATION_S, MIN_SPEED_MM_S, ExperimentConfig
from .signal_scheduler import SignalScheduler, WriteRecord

if TYPE_CHECKING:
    from .display import TrialLayout
    from .trace_recorder import AttemptTraceBuffer

LEGACY = "timed_legacy"


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
                nominal_interval_s=1.0 / fps,
                position_quantum_mm=max(1.0 / calibration.px_per_mm_x, 1.0 / calibration.px_per_mm_y),
            )
        elif self.mode == "position":
            self._planner = rl.PositionPlanner(self._nominal)

    @property
    def error(self) -> BaseException | None:
        return self._signal.error

    def frame(
        self, now_s: float, pos: tuple[int, int], x_mm: float, y_mm: float, speed_mm_s: float
    ) -> tuple[bool, str | None]:
        """Decide and submit; returns the signal state wanted at this frame and the bar under the finger."""
        sample = rl.Sample(now_s, x_mm, y_mm)
        self._clock.add(now_s, repeated=pos == self._last_pos)
        self._last_pos = pos
        if self._planner is None:
            side = self._pixel_side(pos)
            on = self._legacy.step(now_s, side, speed_mm_s)
            bar, edge = rl.explain_switch(self._previous, sample, self._pixel, on)
            plan = rl.Plan(now_s, on, bar, "timer" if on and side is None else edge, (), x_mm, y_mm)
        else:
            plan = self._planner.plan(sample)
            side = rl.side_at(self._nominal, x_mm, y_mm)
            self._horizon_s = plan.horizon_s
        self._previous = sample
        self._signal.submit(plan)
        self._log(self._signal.drain())
        return plan.on, side

    def finish(self, reason: str, now_s: float) -> None:
        """Signal off; every write and the trial's sampling statistics are in the trace before this returns."""
        self._log(self._signal.stop_trial(reason))
        if self._trace is not None:
            payload: dict[str, Any] = {
                "mode": self.mode,
                **self._clock.summary(),
                "horizon_ms": round(1000.0 * self._horizon_s, 3),
                "busy_wait_margin_ms": round(1000.0 * (self._signal.spin_s or 0.0), 3),
            }
            self._trace.add_event(t_us=self._us(now_s), event_type="sample_stats", payload=payload)

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
