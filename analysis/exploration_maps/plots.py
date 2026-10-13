"""Two-panel figures (visit count | dwell time) with the bar outlines drawn on top."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from .grid import Grid
from .trace_io import Bars

REFERENCE_COLOR = "#4299e1"
COMPARISON_COLOR = "#48bb78"
BASELINE_COLOR = "#f5a623"


def _outline(ax: Axes, x0: float, baseline_y: float, width: float, height: float, color: str, style: str) -> None:
    ax.add_patch(
        Rectangle((x0, baseline_y - height), width, height, fill=False, edgecolor=color, linewidth=1.6, linestyle=style)
    )


def _panel(ax: Axes, values: NDArray, grid: Grid, baseline_y: float, label: str, x_label: str) -> None:
    extent = (grid.x_edges[0], grid.x_edges[-1], grid.y_edges[-1], grid.y_edges[0])
    image = ax.imshow(values, origin="upper", extent=extent, cmap="magma", interpolation="nearest", aspect="equal")
    ax.axhline(baseline_y, color=BASELINE_COLOR, linewidth=1.0, alpha=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("y (mm from surface top)")
    ax.set_title(label, fontsize=10)
    ax.figure.colorbar(image, ax=ax, label=label, shrink=0.85)


def _save(fig: plt.Figure, title: str, out_path: Path) -> None:
    fig.suptitle(title, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_trial(
    visits: NDArray, dwell_s: NDArray[np.float64], grid: Grid, bars: Bars, title: str, out_path: Path
) -> None:
    """One trial in screen coordinates: reference bar blue, comparison bar green."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True)
    panels = ((visits, "visits (entries into cell)"), (dwell_s * 1000.0, "dwell time (ms)"))
    for ax, (values, label) in zip(axes, panels):
        _panel(ax, values, grid, bars.baseline_y, label, "x (mm from surface left)")
        left_color = REFERENCE_COLOR if bars.left_is_reference else COMPARISON_COLOR
        right_color = COMPARISON_COLOR if bars.left_is_reference else REFERENCE_COLOR
        _outline(ax, bars.left_x0, bars.baseline_y, bars.width, bars.left_height, left_color, "-")
        _outline(ax, bars.right_x0, bars.baseline_y, bars.width, bars.right_height, right_color, "-")
    _save(fig, title, out_path)


def plot_aggregate(
    visits: NDArray[np.float64],
    dwell: NDArray[np.float64],
    grid: Grid,
    bars: Bars,
    comparison_heights: Sequence[float],
    labels: tuple[str, str],
    title: str,
    out_path: Path,
) -> None:
    """Pooled trials, mirrored so the reference bar (solid) is on the left; comparison heights dashed."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True)
    for ax, values, label in zip(axes, (visits, dwell), labels):
        _panel(ax, values, grid, bars.baseline_y, label, "x (mm, mirrored so the reference bar is on the left)")
        _outline(ax, bars.left_x0, bars.baseline_y, bars.width, bars.left_height, REFERENCE_COLOR, "-")
        for height in comparison_heights:
            _outline(ax, bars.right_x0, bars.baseline_y, bars.width, height, COMPARISON_COLOR, "--")
    _save(fig, title, out_path)
