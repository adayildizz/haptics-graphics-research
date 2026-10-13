"""Grid over the bars, and visit-count / dwell-time accumulation along the finger path."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Cell = tuple[int, int]  # (row, column)


@dataclass(frozen=True)
class Grid:
    x_edges: NDArray[np.float64]
    y_edges: NDArray[np.float64]

    @property
    def cell_mm(self) -> float:
        return float(self.x_edges[1] - self.x_edges[0])

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.y_edges) - 1, len(self.x_edges) - 1


@dataclass(frozen=True)
class TrialMaps:
    visits: NDArray[np.int64]
    dwell_s: NDArray[np.float64]
    outside_s: float  # included time spent outside the grid


def make_grid(
    left_x0: float,
    right_x1: float,
    baseline_y: float,
    tallest_mm: float,
    cell_mm: float,
    margin_mm: float,
) -> Grid:
    """Cells covering both bars plus a margin, anchored on the left bar edge and the baseline."""
    if cell_mm <= 0 or margin_mm < 0:
        raise ValueError("cell_mm must be > 0 and margin_mm >= 0")
    eps = 1e-9
    pad = math.ceil(margin_mm / cell_mm - eps)
    nx = pad + math.ceil((right_x1 + margin_mm - left_x0) / cell_mm - eps)
    above = math.ceil((tallest_mm + margin_mm) / cell_mm - eps)
    x_edges = left_x0 + cell_mm * (np.arange(nx + 1) - pad)
    y_edges = baseline_y + cell_mm * (np.arange(above + pad + 1) - above)
    return Grid(x_edges.astype(np.float64), y_edges.astype(np.float64))


def _edge_crossings(a: float, b: float, origin: float, cell_mm: float) -> NDArray[np.float64]:
    """Segment parameters in (0, 1) where a coordinate moving from a to b crosses a grid line."""
    if a == b:
        return np.empty(0)
    low, high = (a, b) if a < b else (b, a)
    first = math.floor((low - origin) / cell_mm) + 1
    last = math.ceil((high - origin) / cell_mm) - 1
    if last < first:
        return np.empty(0)
    return (origin + cell_mm * np.arange(first, last + 1) - a) / (b - a)


def cells_on_segment(
    x0: float, y0: float, x1: float, y1: float, grid: Grid
) -> tuple[list[Cell | None], NDArray[np.float64]]:
    """Cells a straight segment passes through, in order, with the length fraction inside each."""
    cell_mm = grid.cell_mm
    ny, nx = grid.shape
    cuts = np.concatenate(
        (
            [0.0, 1.0],
            _edge_crossings(x0, x1, float(grid.x_edges[0]), cell_mm),
            _edge_crossings(y0, y1, float(grid.y_edges[0]), cell_mm),
        )
    )
    cuts = np.unique(np.clip(cuts, 0.0, 1.0))
    fractions = np.diff(cuts)
    keep = fractions > 1e-9  # a path through a grid corner must not visit the cells beside it
    mids = ((cuts[:-1] + cuts[1:]) / 2.0)[keep]
    fractions = fractions[keep] / fractions[keep].sum()
    columns = np.floor((x0 + mids * (x1 - x0) - grid.x_edges[0]) / cell_mm).astype(int)
    rows = np.floor((y0 + mids * (y1 - y0) - grid.y_edges[0]) / cell_mm).astype(int)
    cells: list[Cell | None] = [
        (int(row), int(column)) if 0 <= row < ny and 0 <= column < nx else None
        for row, column in zip(rows, columns)
    ]
    return cells, fractions


def accumulate(
    t_s: NDArray[np.float64],
    x_mm: NDArray[np.float64],
    y_mm: NDArray[np.float64],
    include: NDArray[np.bool_],
    resting: NDArray[np.bool_],
    grid: Grid,
) -> TrialMaps:
    """Visit counts (cell entries) and dwell seconds over the included segments of one trial."""
    visits = np.zeros(grid.shape, dtype=np.int64)
    dwell_s = np.zeros(grid.shape, dtype=np.float64)
    outside_s = 0.0
    current: Cell | None = None
    entered = False  # whether the current contact run has registered its first cell
    for k in range(len(t_s) - 1):
        if not include[k]:
            current, entered = None, False
            continue
        dt = float(t_s[k + 1] - t_s[k])
        cells, fractions = cells_on_segment(float(x_mm[k]), float(y_mm[k]), float(x_mm[k + 1]), float(y_mm[k + 1]), grid)
        for cell, fraction in zip(cells, fractions):
            if cell is None:
                outside_s += dt * float(fraction)
            else:
                dwell_s[cell] += dt * float(fraction)
            # A resting finger jittering over a grid line is not entering cells.
            if cell != current and (not resting[k] or not entered):
                if cell is not None:
                    visits[cell] += 1
                current = cell
            entered = True
    return TrialMaps(visits, dwell_s, outside_s)


def aggregate(maps: Sequence[NDArray], *, normalize: bool) -> tuple[NDArray[np.float64], int]:
    """Mean of per-trial shares in percent, or the plain sum; trials with an empty map are skipped."""
    used = [m.astype(np.float64) for m in maps if m.sum() > 0]
    if not used:
        return np.zeros_like(maps[0], dtype=np.float64), 0
    if normalize:
        return 100.0 * np.mean([m / m.sum() for m in used], axis=0), len(used)
    return np.sum(used, axis=0), len(used)
