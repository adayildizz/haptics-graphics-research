"""Per-trial figure (time-coloured path | region lanes) and the per-session overview of region strips."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
from numpy.typing import NDArray

from analysis.exploration_maps.trace_io import Bars

from .regions import REGIONS, RegionParams, Segment

REFERENCE = "#2a78d6"
COMPARISON = "#1baf7a"
NEUTRAL = "#8a8986"
TEXT_MUTED = "#52514e"
# Start to end of a trial, as the professor's suggestion asked: blue to red.
TIME_CMAP = LinearSegmentedColormap.from_list("trial_time", ["#2a78d6", "#e34948"])


@dataclass(frozen=True)
class Style:
    label: str
    face: str
    hatch: str | None = None
    edge: str | None = None


# Hue says which bar, solid vs hatched says top edge vs the rest of that bar's regions.
STYLES = {
    "ref_top": Style("reference top edge", REFERENCE),
    "ref_other": Style("reference, elsewhere", "#e3eefb", "////", REFERENCE),
    "comp_top": Style("comparison top edge", COMPARISON),
    "comp_other": Style("comparison, elsewhere", "#dcf3ea", "\\\\\\\\", COMPARISON),
    "between": Style("between the bars / outer sides", NEUTRAL),
    "off_surface": Style("off the active surface", "#efeee9", "....", NEUTRAL),
    "no_contact": Style("no contact", "#d3d1c7"),
}


def style_key(code: int) -> str:
    key = REGIONS[code].key
    if key in ("gap", "outer"):
        return "between"
    if key in ("off_surface", "no_contact"):
        return key
    bar, _, place = key.partition("_")
    return f"{bar}_top" if place.startswith("top") else f"{bar}_other"


def _bar_kwargs(style: Style) -> dict:
    kwargs = {"facecolor": style.face, "linewidth": 0}
    if style.hatch:
        kwargs.update(hatch=style.hatch, edgecolor=style.edge)
    return kwargs


def _legend_handles() -> list[Patch]:
    handles = []
    for style in STYLES.values():
        handles.append(
            Patch(facecolor=style.face, hatch=style.hatch, edgecolor=style.edge or style.face, label=style.label)
        )
    return handles


# --------------------------------------------------------------------------- path panel


def _draw_bars_and_bands(ax: Axes, bars: Bars, params: RegionParams) -> None:
    """Bar outlines in their role colour, the edge bands as light grey."""
    b, o = params.band_mm, params.overhang_mm
    left_colour, right_colour = (REFERENCE, COMPARISON) if bars.left_is_reference else (COMPARISON, REFERENCE)
    for x0, height, colour in (
        (bars.left_x0, bars.left_height, left_colour),
        (bars.right_x0, bars.right_height, right_colour),
    ):
        top = bars.baseline_y - height
        band = {"facecolor": "#e6e4dd", "edgecolor": "none", "zorder": 0}
        ax.add_patch(Rectangle((x0 - o, top - b), bars.width + 2 * o, 2 * b, **band))
        ax.add_patch(Rectangle((x0 - b, top + b), 2 * b, height - b, **band))
        ax.add_patch(Rectangle((x0 + bars.width - b, top + b), 2 * b, height - b, **band))
        ax.add_patch(Rectangle((x0, top), bars.width, height, fill=False, edgecolor=colour, linewidth=1.6, zorder=1))
    ax.axhline(bars.baseline_y, color=NEUTRAL, linewidth=0.8, zorder=1)


def _arrows(
    ax: Axes, x: NDArray, y: NDArray, t: NDArray, drawn: NDArray[np.bool_], every_mm: float, length_mm: float
) -> None:
    """An arrowhead each time the drawn path has covered another ``every_mm``."""
    if every_mm <= 0 or not drawn.any():
        return
    step = np.hypot(np.diff(x), np.diff(y)) * drawn
    travelled = np.concatenate(([0.0], np.cumsum(step)))
    marks = np.arange(every_mm, travelled[-1], every_mm)
    if len(marks) == 0:
        return
    ends = np.searchsorted(travelled, marks)  # first sample at or past each mark
    dx, dy = x[ends] - x[ends - 1], y[ends] - y[ends - 1]
    norm = np.hypot(dx, dy)
    ax.quiver(
        x[ends], y[ends], dx / norm * length_mm, dy / norm * length_mm,
        color=TIME_CMAP(Normalize(0.0, t[-1])(t[ends])), angles="xy", scale_units="xy", scale=1.0,
        pivot="mid", width=0.006, headwidth=4.5, headlength=5, headaxislength=4.5, zorder=4,
    )  # fmt: skip


def _path_panel(
    ax: Axes,
    t: NDArray,
    x: NDArray,
    y: NDArray,
    contact: NDArray[np.bool_] | None,
    bars: Bars,
    params: RegionParams,
    arrow_every_mm: float,
    margin_mm: float,
) -> None:
    _draw_bars_and_bands(ax, bars, params)
    down = np.ones(len(t), dtype=bool) if contact is None else contact
    drawn = down[:-1] & down[1:]  # the line breaks while the finger is lifted
    points = np.column_stack((x, y))
    lines = LineCollection(
        np.stack((points[:-1], points[1:]), axis=1)[drawn],
        colors=TIME_CMAP(Normalize(0.0, t[-1])(t[:-1][drawn])),
        linewidths=1.6,
        zorder=3,
    )
    ax.add_collection(lines)

    tallest = max(bars.left_height, bars.right_height)
    x_lo, x_hi = bars.left_x0 - margin_mm, bars.right_x1 + margin_mm
    y_lo, y_hi = bars.baseline_y - tallest - margin_mm, bars.baseline_y + 3.0
    if down.any():
        x_lo, x_hi = min(x_lo, x[down].min() - 3), max(x_hi, x[down].max() + 3)
        y_lo, y_hi = min(y_lo, y[down].min() - 3), max(y_hi, y[down].max() + 3)
        first, last = np.flatnonzero(down)[[0, -1]]
        ax.plot(x[first], y[first], "o", color=TIME_CMAP(0.0), markersize=9, markeredgecolor="white", zorder=5)
        ax.plot(x[last], y[last], "s", color=TIME_CMAP(1.0), markersize=9, markeredgecolor="white", zorder=5)
        ax.annotate("start", (x[first], y[first]), xytext=(6, 6), textcoords="offset points", fontsize=8)
        ax.annotate("end", (x[last], y[last]), xytext=(6, -12), textcoords="offset points", fontsize=8)
    _arrows(ax, x, y, t, drawn, arrow_every_mm, 0.035 * max(x_hi - x_lo, y_hi - y_lo))

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_hi, y_lo)  # y down, like the screen and the trace
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm from surface left)")
    ax.set_ylabel("y (mm from surface top)")
    ax.set_title("finger path, coloured by time (grey: edge bands)", fontsize=9)


# --------------------------------------------------------------------------- lane panel


def _lane_panel(ax: Axes, segments: Sequence[Segment], end_s: float) -> None:
    """One lane per region, every region always listed so trials can be compared."""
    for code in range(len(REGIONS)):
        mine = [(s.start_s, s.duration_s) for s in segments if s.code == code]
        if mine:
            ax.broken_barh(mine, (code - 0.38, 0.76), **_bar_kwargs(STYLES[style_key(code)]))
        total = sum(duration for _, duration in mine)
        shown = f"{total:.1f} s" if total >= 1.0 else f"{total * 1000.0:.0f} ms"
        summary = f"{shown} · {len(mine)}×" if mine else "–"
        ax.text(1.005, code, summary, transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=TEXT_MUTED)
    for boundary in (5.5, 11.5):  # reference | comparison | everything else
        ax.axhline(boundary, color=NEUTRAL, linewidth=0.6)
    ax.set_yticks(range(len(REGIONS)), [region.label for region in REGIONS], fontsize=8)
    ax.set_ylim(len(REGIONS) - 0.5, -0.5)
    ax.set_xlim(0.0, end_s)
    ax.grid(axis="x", color="#e6e4dd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_title("region over time (right: total time · number of entries)", fontsize=9)


def _time_strip(ax: Axes, end_s: float) -> None:
    ax.imshow(np.linspace(0.0, 1.0, 256)[None, :], extent=(0.0, end_s, 0.0, 1.0), aspect="auto", cmap=TIME_CMAP)
    ax.set_yticks([])
    ax.set_xlabel("time from trial start (s); this strip is the path colour")


def plot_attempt(
    t: NDArray,
    x: NDArray,
    y: NDArray,
    contact: NDArray[np.bool_] | None,
    segments: Sequence[Segment],
    bars: Bars,
    params: RegionParams,
    title: str,
    out_path: Path,
    *,
    arrow_every_mm: float = 20.0,
    margin_mm: float = 10.0,
) -> None:
    fig = plt.figure(figsize=(16, 7.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 2.3), height_ratios=(24, 1))
    _path_panel(fig.add_subplot(grid[:, 0]), t, x, y, contact, bars, params, arrow_every_mm, margin_mm)
    lanes = fig.add_subplot(grid[0, 1])
    _lane_panel(lanes, segments, float(t[-1]))
    _time_strip(fig.add_subplot(grid[1, 1], sharex=lanes), float(t[-1]))
    fig.suptitle(title, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- session overview


def plot_overview(rows: Sequence[tuple[str, Sequence[Segment]]], title: str, out_path: Path) -> None:
    """One strip per trial, in recording order, on a shared time axis."""
    end_s = max((s.end_s for _, segments in rows for s in segments), default=1.0)
    fig, ax = plt.subplots(figsize=(14, 1.6 + 0.26 * len(rows)), constrained_layout=True)
    for row, (_, segments) in enumerate(rows):
        by_style: dict[str, list[tuple[float, float]]] = {}
        for segment in segments:
            by_style.setdefault(style_key(segment.code), []).append((segment.start_s, segment.duration_s))
        for key, spans in by_style.items():
            ax.broken_barh(spans, (row - 0.4, 0.8), **_bar_kwargs(STYLES[key]))
    ax.set_yticks(range(len(rows)), [label for label, _ in rows], fontsize=7)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0.0, end_s)
    ax.set_xlabel("time from trial start (s)")
    ax.grid(axis="x", color="#e6e4dd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=_legend_handles(), loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
