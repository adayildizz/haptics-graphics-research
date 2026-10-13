"""Decide which sample-to-sample segments count as finger-on-surface exploration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ContactParams:
    freeze_mm: float = 0.5
    freeze_ms: float = 200.0
    jump_mm: float = 10.0
    keep_end_freeze: bool = False


@dataclass(frozen=True)
class ContactResult:
    include: NDArray[np.bool_]  # per segment: counted in the maps
    resting: NDArray[np.bool_]  # per segment: finger held still on the surface (dwell only)
    method: str  # "contact flag" or "heuristic"
    total_ms: float
    included_ms: float
    start_freeze_ms: float
    lift_ms: float
    end_freeze_ms: float
    end_freeze_excluded: bool
    no_contact_ms: float
    jump_segments: int
    jump_ms: float

    @property
    def excluded_ms(self) -> float:
        return self.total_ms - self.included_ms

    def describe(self, params: ContactParams) -> str:
        """One line for a figure title: the method used and what it removed."""
        if self.method == "contact flag":
            parts = [f"no contact {self.no_contact_ms:.0f}"]
            rule = f"contact: recorded flag (jump ≥{params.jump_mm:g} mm)"
        else:
            kept = self.end_freeze_ms > 0 and not self.end_freeze_excluded
            end = f"end {self.end_freeze_ms:.0f}" + (" kept" if kept else "")
            parts = [f"start {self.start_freeze_ms:.0f}", f"lift {self.lift_ms:.0f}", end]
            rule = (
                f"contact: heuristic (freeze <{params.freeze_mm:g} mm for ≥{params.freeze_ms:g} ms, "
                f"jump ≥{params.jump_mm:g} mm)"
            )
        parts.append(f"jumps {self.jump_ms:.0f} in {self.jump_segments} seg")
        return f"{rule} · excluded {self.excluded_ms:.0f} ms ({', '.join(parts)})"


def freeze_runs(
    t_s: NDArray[np.float64],
    x_mm: NDArray[np.float64],
    y_mm: NDArray[np.float64],
    freeze_mm: float,
    freeze_ms: float,
) -> list[tuple[int, int]]:
    """Sample ranges (first, last) staying within ``freeze_mm`` of their first sample for ``freeze_ms``."""
    runs: list[tuple[int, int]] = []
    n = len(t_s)
    radius_sq = freeze_mm * freeze_mm
    i = 0
    while i < n - 1:
        j = i
        while j + 1 < n and (x_mm[j + 1] - x_mm[i]) ** 2 + (y_mm[j + 1] - y_mm[i]) ** 2 < radius_sq:
            j += 1
        if j > i and (t_s[j] - t_s[i]) * 1000.0 >= freeze_ms:
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def classify(
    t_s: NDArray[np.float64],
    x_mm: NDArray[np.float64],
    y_mm: NDArray[np.float64],
    contact: NDArray[np.bool_] | None,
    params: ContactParams,
) -> ContactResult:
    """Use the recorded contact flag if there is one, otherwise the freeze/jump heuristic."""
    n = len(t_s)
    dt_ms = np.diff(t_s) * 1000.0
    jump = np.hypot(np.diff(x_mm), np.diff(y_mm)) >= params.jump_mm
    include = ~jump
    resting = np.zeros(max(n - 1, 0), dtype=bool)
    start_ms = lift_ms = end_ms = no_contact_ms = 0.0
    end_excluded = False

    if contact is not None:
        touching = contact[:-1] & contact[1:]
        no_contact_ms = float(dt_ms[~touching].sum())
        include &= touching

    for first, last in freeze_runs(t_s, x_mm, y_mm, params.freeze_mm, params.freeze_ms):
        duration_ms = float((t_s[last] - t_s[first]) * 1000.0)
        if contact is None:
            if first == 0:  # cursor still parked where the previous trial left it
                include[first:last] = False
                start_ms += duration_ms
                continue
            if last == n - 1:  # held (or lifted) until the response
                end_ms = duration_ms
                if not params.keep_end_freeze:
                    include[first:last] = False
                    end_excluded = True
                    continue
            elif jump[last]:  # frozen, then teleported: the finger was in the air
                include[first:last] = False
                lift_ms += duration_ms
                continue
        resting[first:last] = True

    resting &= include
    return ContactResult(
        include=include,
        resting=resting,
        method="heuristic" if contact is None else "contact flag",
        total_ms=float(dt_ms.sum()),
        included_ms=float(dt_ms[include].sum()),
        start_freeze_ms=start_ms,
        lift_ms=lift_ms,
        end_freeze_ms=end_ms,
        end_freeze_excluded=end_excluded,
        no_contact_ms=no_contact_ms,
        jump_segments=int(jump.sum()),
        jump_ms=float(dt_ms[jump].sum()),
    )
