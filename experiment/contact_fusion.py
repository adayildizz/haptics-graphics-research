"""Whether the finger is on the glass, from the IR frame and the force sensor together. Standard library only.

The IR frame reports contact when something breaks its beams a few millimetres above the glass, and it drops
contact (and moves the cursor) at times when the finger stays on the glass. The force sensor under the screen
measures the load on the glass but not where it is. Each frame falls in one of four states:

``both``        IR and force agree the finger is down: the normal case.
``force_only``  IR says lifted, force says loaded: an IR dropout with the finger on the glass, or something
                else (a palm, the wrist) resting on the glass while the finger is up. The position is stale.
``ir_only``     IR says down, force says unloaded: a finger in the beams but not on the glass (hovering), or a
                touch lighter than the force threshold.
``neither``     a real lift.

``ContactFusion`` turns the states into what the renderer would use. The signal stays switchable whenever
either source says down, because a signal on while the finger is in the air is not felt, but a signal cut while
the finger is on the glass is. In ``force_only`` the position is stale, so the last decision is held for at most
``hold_s`` and then dropped: held longer, a finger that left the bar would feel it as a longer bar. An IR
position that jumps further than ``jump_mm`` in one frame while the force never let go is a teleport, not a
movement, and is not used until the next frame confirms it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from .ati_calibration import AtiCalibration

STATES = ("both", "force_only", "ir_only", "neither")


def load_of(volts: Sequence[float], bias: Sequence[float], calibration: AtiCalibration | None) -> float:
    """How hard the glass is pressed: |Fz| in the calibration's force units, or without one the size of the
    change in gauge voltages (V), which rises with any load but has no physical unit."""
    if calibration is not None:
        return abs(calibration.apply(volts, bias)[2])
    return math.sqrt(sum((v - b) ** 2 for v, b in zip(volts, bias)))


@dataclass(frozen=True)
class Thresholds:
    on: float
    off: float
    noise_mean: float
    noise_sd: float


def thresholds_from_noise(loads: Sequence[float], on_sd: float = 6.0, off_sd: float = 3.0) -> Thresholds:
    """Thresholds above the unloaded noise: loaded at mean + ``on_sd`` SD, unloaded again below mean + ``off_sd`` SD."""
    if len(loads) < 2:
        raise ValueError("need at least two unloaded samples to measure the noise")
    mean, sd = statistics.fmean(loads), statistics.stdev(loads)
    return Thresholds(mean + on_sd * sd, mean + off_sd * sd, mean, sd)


def contact_state(ir_down: bool, force_down: bool) -> str:
    if ir_down:
        return "both" if force_down else "ir_only"
    return "force_only" if force_down else "neither"


class HysteresisContact:
    """Loaded once the value reaches ``on``; unloaded again only below ``off``."""

    def __init__(self, on: float, off: float) -> None:
        if not 0 <= off <= on:
            raise ValueError(f"need 0 <= off <= on, got on={on}, off={off}")
        self.on = on
        self.off = off
        self.down = False

    def update(self, value: float) -> bool:
        self.down = value >= self.on if not self.down else value >= self.off
        return self.down


@dataclass(frozen=True)
class FusedFrame:
    state: str  # one of STATES
    render_down: bool  # whether the renderer should treat the finger as on the glass
    position_valid: bool  # whether this frame's IR position may be used
    held: bool = False  # force_only, still within hold_s: the last decision stands
    teleport: bool = False  # the IR position jumped while the force never let go


class ContactFusion:
    """Frame-by-frame fusion of IR contact, force contact and the IR position (see the module docstring)."""

    def __init__(self, hold_s: float = 0.15, jump_mm: float = 20.0) -> None:
        self.hold_s = hold_s
        self.jump_mm = jump_mm
        self._force_only_since: float | None = None
        self._last_valid: tuple[float, float] | None = None
        self._pending: tuple[float, float] | None = None
        self._lifted_since_valid = True

    def update(self, t_s: float, ir_down: bool, force_down: bool, x_mm: float, y_mm: float) -> FusedFrame:
        state = contact_state(ir_down, force_down)
        if state == "neither":
            self._force_only_since = None
            self._pending = None
            self._lifted_since_valid = True
            return FusedFrame(state, False, False)
        if state == "force_only":
            if self._force_only_since is None:
                self._force_only_since = t_s
            held = t_s - self._force_only_since <= self.hold_s
            return FusedFrame(state, held, False, held=held)
        self._force_only_since = None
        position = (x_mm, y_mm)
        if not force_down:
            # Hovering or a very light touch: nothing pins the finger to its last place, so any position is fine.
            self._accept(position)
            self._lifted_since_valid = True
            return FusedFrame(state, True, True)
        if self._lifted_since_valid or self._last_valid is None or _distance(position, self._last_valid) <= self.jump_mm:
            self._accept(position)
            return FusedFrame(state, True, True)
        # A jump with the finger loaded the whole time: believe it only once the next frame lands near it too.
        if self._pending is not None and _distance(position, self._pending) <= self.jump_mm:
            self._accept(position)
            return FusedFrame(state, True, True)
        self._pending = position
        return FusedFrame(state, True, False, teleport=True)

    def _accept(self, position: tuple[float, float]) -> None:
        self._last_valid = position
        self._pending = None
        self._lifted_since_valid = False


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class Episode:
    state: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def episodes(t_s: Sequence[float], states: Sequence[str]) -> list[Episode]:
    """Runs of one state; a run lasts until the next one starts, and the last until the last frame."""
    runs: list[Episode] = []
    start = 0
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index] != states[start]:
            end_s = t_s[index] if index < len(states) else t_s[-1]
            runs.append(Episode(states[start], t_s[start], end_s))
            start = index
    return runs
