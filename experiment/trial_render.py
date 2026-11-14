"""Per-trial glue between the trial loop, the render decisions and the signal scheduler."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from . import render_logic as rl
from .calibration import DisplayCalibration
from .config import ExperimentConfig
from .cue_link import CueRecord, message_for
from .signal_scheduler import SignalScheduler, WriteRecord

if TYPE_CHECKING:
    from .cue_link import CueLink
    from .display import TrialLayout
    from .trace_recorder import AttemptTraceBuffer

# The only render rule; the predictive and timed rules are kept on the render-modes branch.
MODE = "position"
# One per process, i.e. per session: once the device has reported a touch, contact is trusted from then on.
CONTACT = rl.ContactGate()
_warned_no_touch = False
# One per session when the Haptuator cue is on: main opens it before the first trial and closes it after the last.
CUE_LINK: CueLink | None = None
CUE_FLUSH_S = 0.05  # at the end of a trial, how long to wait for a cue still being written


def reset_contact_gate() -> None:
    """Forget what the input device has shown (tests, or a new device)."""
    global CONTACT, _warned_no_touch
    CONTACT = rl.ContactGate()
    _warned_no_touch = False


def use_cue_link(link: CueLink | None) -> None:
    """The link every following trial sends its top-edge cues to; None when the session has none."""
    global CUE_LINK
    CUE_LINK = link


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
        trace_attempt: AttemptTraceBuffer | None,
    ) -> None:
        self.mode = MODE
        self._signal = signal
        self._trial_start_s = trial_start_s
        self._trace = trace_attempt
        self._clock = rl.SampleClock()
        self._last_pos: tuple[int, int] | None = None
        self._last_contact: bool | None = None
        left_is_comparison = layout.left_is_comparison
        heights = (
            layout.comparison_height_mm if left_is_comparison else layout.reference_height_mm,
            layout.reference_height_mm if left_is_comparison else layout.comparison_height_mm,
        )
        # Decisions use the bars at their configured size; the drawing keeps its pixel-rounded rectangles.
        self._nominal = rl.nominal_bars(
            calibration.active_width_mm, calibration.active_height_mm, cfg.inter_bar_gap_mm, cfg.bar_width_mm, *heights
        )
        self._lifted = False
        self._start_afresh()
        # The Haptuator cue: a pulse when the finger crosses a bar's top edge, sent to the ESP32 (cue_link.py).
        self._link = CUE_LINK
        self._cue = (
            rl.TopEdgeCue(
                self._nominal,
                rearm_mm=cfg.cue_rearm_mm,
                corner_mm=cfg.cue_corner_mm,
                jump_mm=cfg.cue_jump_mm,
                min_interval_s=cfg.cue_min_interval_ms / 1000.0,
            )
            if cfg.cue_enabled
            else None
        )

    def _start_afresh(self) -> None:
        """New decision state: at the start of a trial and whenever the finger comes back onto the glass."""
        self._planner = rl.PositionPlanner(self._nominal)

    @property
    def error(self) -> BaseException | None:
        return self._signal.error

    def frame(
        self, now_s: float, pos: tuple[int, int], x_mm: float, y_mm: float, contact: bool
    ) -> FrameDecision:
        """Decide and submit the plan for one frame; ``contact`` is the left-button state the device reports."""
        finger_down = CONTACT.finger_down(contact)
        self._clock.add(now_s, repeated=pos == self._last_pos)
        self._last_pos = pos
        touched_down = finger_down and self._lifted
        if touched_down:
            # The cursor sat still while the finger was up and has now jumped: no edge may be read off that jump.
            self._start_afresh()
        self._lifted = not finger_down
        # The signal follows position only. Contact is recorded but not used: the frame reports false lifts while
        # the finger is still on the glass, and a finger that is really lifted cannot feel the signal anyway.
        plan = self._planner.plan(rl.Sample(now_s, x_mm, y_mm))
        side = rl.side_at(self._nominal, x_mm, y_mm)
        if touched_down:
            plan = replace(plan, edge="touch")
        self._signal.submit(plan)
        if self._cue is not None:
            self._step_cue(now_s, x_mm, y_mm, finger_down)
        self._log(self._signal.drain())
        self._log_contact(now_s, contact, side)
        return FrameDecision(plan.on, side, finger_down)

    def _step_cue(self, now_s: float, x_mm: float, y_mm: float, finger_down: bool) -> None:
        """Send this frame's top-edge pulse, if any, then log every top crossing and what the link did."""
        pulse = self._cue.step(rl.Sample(now_s, x_mm, y_mm), finger_down)
        if pulse is not None and self._link is not None:
            self._link.send(message_for(pulse.bar, pulse.direction), now_s)
        if self._trace is not None:
            for crossing in self._cue.crossings:
                self._trace.add_event(
                    t_us=self._us(crossing.t_s),
                    event_type="cue_crossing",
                    payload={
                        "bar": crossing.bar,
                        "direction": crossing.direction,
                        "outcome": crossing.outcome,
                        "t_cross_us": self._us(crossing.t_cross_s),
                        "x_cross_mm": crossing.x_cross_mm,
                        "late_mm": crossing.late_mm,
                        "on_jump": crossing.on_jump,
                    },
                )
        self._log_cues()

    def finish(self, reason: str, now_s: float) -> None:
        """Signal off; every write and the trial's sampling statistics are in the trace before this returns."""
        global _warned_no_touch
        self._log(self._signal.stop_trial(reason))
        if self._cue is not None and self._link is not None:
            self._link.flush(CUE_FLUSH_S)
            self._log_cues()
        if not CONTACT.device_reports_touch and not _warned_no_touch:
            _warned_no_touch = True
            print(
                "WARNING: the input device has not reported a finger touching the surface; contact is recorded "
                "as 1 (unknown). The signal does not depend on it."
            )
        if self._trace is not None:
            payload: dict[str, Any] = {
                "mode": self.mode,
                "device_reports_touch": CONTACT.device_reports_touch,
                **self._clock.summary(),
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

    def _us(self, t_s: float) -> int:
        return round((t_s - self._trial_start_s) * 1_000_000)

    def _log_cues(self) -> None:
        """One cue_sent or cue_failed event per message the link has finished with."""
        if self._link is None:
            return
        records: list[CueRecord] = self._link.drain()
        if self._trace is None:
            return
        for record in records:
            payload: dict[str, Any] = {"message": record.message, "queued_us": self._us(record.queued_s)}
            if record.write_start_s is not None and record.write_end_s is not None:
                payload["write_start_us"] = self._us(record.write_start_s)
                payload["write_end_us"] = self._us(record.write_end_s)
            if record.error is not None:
                payload["error"] = record.error
            self._trace.add_event(
                t_us=self._us(record.queued_s),
                event_type="cue_failed" if record.error is not None else "cue_sent",
                payload=payload,
            )

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
