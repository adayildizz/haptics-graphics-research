"""Which region of the display each recorded finger sample falls in, and the runs of samples in one region."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analysis.exploration_maps.trace_io import Bars


@dataclass(frozen=True)
class Region:
    key: str
    label: str
    role: str  # "reference", "comparison" or "common"
    signal: bool  # inside a bar, where the signal is on


def _bar_regions(role: str, short: str) -> tuple[Region, ...]:
    return (
        Region(f"{short}_above", f"{role}: above", role, False),
        Region(f"{short}_top_out", f"{role}: top edge, outside", role, False),
        Region(f"{short}_top_in", f"{role}: top edge, inside", role, True),
        Region(f"{short}_side_out", f"{role}: side edge, outside", role, False),
        Region(f"{short}_side_in", f"{role}: side edge, inside", role, True),
        Region(f"{short}_body", f"{role}: body", role, True),
    )


# Figure order, top to bottom. Codes are indices into this tuple.
REGIONS: tuple[Region, ...] = (
    *_bar_regions("reference", "ref"),
    *_bar_regions("comparison", "comp"),
    Region("gap", "between the bars", "common", False),
    Region("outer", "outer sides", "common", False),
    Region("off_surface", "off the active surface", "common", False),
    Region("no_contact", "no contact", "common", False),
)
CODE = {region.key: code for code, region in enumerate(REGIONS)}


@dataclass(frozen=True)
class RegionParams:
    # Depth of an edge band on each side of the edge. A moving finger travels a median 2-2.6 mm between
    # two frame reports (about 5 mm at the 90th percentile), and a band narrower than that step is often
    # jumped over; at 4 mm about 9 in 10 edge crossings leave a sample in the band.
    band_mm: float = 4.0
    overhang_mm: float = 4.0  # how far the top-edge band reaches past the bar's corners


@dataclass(frozen=True)
class Segment:
    """A run of consecutive samples in one region; it lasts until the next run starts."""

    code: int
    start_s: float
    end_s: float

    @property
    def region(self) -> Region:
        return REGIONS[self.code]

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def classify(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    contact: NDArray[np.bool_] | None,
    bars: Bars,
    surface_width_mm: float,
    params: RegionParams = RegionParams(),
) -> NDArray[np.int_]:
    """One region code per sample; the first matching rule wins.

    Order: no contact, off the active surface, top-edge bands, side-edge bands,
    bar bodies, the space above each bar, between the bars, the outer sides.
    A recording without a contact column is taken as finger down throughout.
    """
    b, o = params.band_mm, params.overhang_mm
    by_role = {
        "ref": (bars.left_x0, bars.left_height) if bars.left_is_reference else (bars.right_x0, bars.right_height),
        "comp": (bars.right_x0, bars.right_height) if bars.left_is_reference else (bars.left_x0, bars.left_height),
    }
    rules: list[tuple[NDArray[np.bool_], str]] = []
    if contact is not None:
        rules.append((~contact, "no_contact"))
    # The same test pygame's collidepoint makes on the haptic area: right and bottom edges are outside.
    rules.append(((x < 0) | (x >= surface_width_mm) | (y < 0) | (y >= bars.baseline_y), "off_surface"))

    tops, sides, rest = [], [], []
    for short, (x0, height) in by_role.items():
        x1, top = x0 + bars.width, bars.baseline_y - height
        column = (x >= x0) & (x <= x1)
        inside = column & (y >= top)
        tops += [
            (inside & (y < top + b), f"{short}_top_in"),
            ((x >= x0 - o) & (x <= x1 + o) & (y >= top - b) & (y < top + b), f"{short}_top_out"),
        ]
        near_side = (x < x0 + b) | (x > x1 - b)
        sides += [
            (inside & near_side, f"{short}_side_in"),
            (~column & (x >= x0 - b) & (x <= x1 + b) & (y >= top), f"{short}_side_out"),
        ]
        rest += [(inside, f"{short}_body"), (column & (y < top - b), f"{short}_above")]
    rules += tops + sides + rest
    rules.append(((x > bars.left_x0 + bars.width) & (x < bars.right_x0), "gap"))

    return np.select([mask for mask, _ in rules], [CODE[key] for _, key in rules], default=CODE["outer"])


def segments(t_s: NDArray[np.float64], codes: NDArray[np.int_]) -> list[Segment]:
    """Runs of equal codes; each run ends where the next begins, the last at the final sample."""
    if len(codes) == 0:
        return []
    starts = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], starts))
    ends = np.concatenate((t_s[starts[1:]], [t_s[-1]]))
    return [Segment(int(codes[i]), float(t_s[i]), float(end)) for i, end in zip(starts, ends)]
