from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from analysis.exploration_maps.trace_io import Bars
from analysis.exploration_timeline.regions import CODE, REGIONS, RegionParams, classify, segments
from analysis.exploration_timeline.run import run
from experiment.trace_store import AttemptDefinition, CursorSample, TraceStore

HZ = 60.0
SURFACE_W = 194.0
# Default design: 10 mm bars, 40 mm gap, on a 194 x 145 mm surface.
# Reference on the left, 10 mm tall (top at y 135); comparison on the right, 12 mm tall (top at y 133).
BARS = Bars(
    left_x0=67.0, right_x0=117.0, width=10.0, baseline_y=145.0, left_height=10.0, right_height=12.0,
    left_is_reference=True,
)  # fmt: skip
PARAMS = RegionParams(band_mm=2.0, overhang_mm=2.0)

PROBES = [
    ((72.0, 136.0), "ref_top_in"),  # inside, within 2 mm below the top
    ((72.0, 134.0), "ref_top_out"),  # just above the top
    ((66.0, 134.5), "ref_top_out"),  # past the corner, within the overhang
    ((72.0, 120.0), "ref_above"),  # above the band, over the bar
    ((67.5, 140.0), "ref_side_in"),
    ((65.5, 140.0), "ref_side_out"),
    ((72.0, 141.0), "ref_body"),
    ((97.0, 140.0), "gap"),
    ((97.0, 60.0), "gap"),  # between the bars at any height
    ((40.0, 140.0), "outer"),
    ((122.0, 110.0), "comp_above"),
    ((122.0, 132.0), "comp_top_out"),
    ((122.0, 134.0), "comp_top_in"),
    ((126.5, 140.0), "comp_side_in"),
    ((128.5, 140.0), "comp_side_out"),
    ((122.0, 140.0), "comp_body"),
    ((97.0, 150.0), "off_surface"),  # below the bars' baseline
    ((-1.0, 100.0), "off_surface"),
    ((194.0, 100.0), "off_surface"),  # the right edge is outside, as for pygame's collidepoint
]


def _classify(points, contact=None, bars=BARS, params=PARAMS):
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    return [REGIONS[code].key for code in classify(x, y, contact, bars, SURFACE_W, params)]


def test_every_probe_point_lands_in_its_region():
    points, expected = zip(*PROBES)
    assert _classify(points) == list(expected)
    assert {key for _, key in PROBES} | {"no_contact"} == {region.key for region in REGIONS}


def test_roles_follow_the_reference_side_not_the_screen_side():
    swapped = Bars(**{**BARS.__dict__, "left_is_reference": False, "left_height": 12.0, "right_height": 10.0})
    assert _classify([(72.0, 141.0), (122.0, 141.0)], bars=swapped) == ["comp_body", "ref_body"]


def test_no_contact_wins_and_a_missing_contact_column_means_finger_down():
    points = [(72.0, 141.0), (97.0, 150.0)]
    assert _classify(points, contact=np.array([False, False])) == ["no_contact", "no_contact"]
    assert _classify(points, contact=None) == ["ref_body", "off_surface"]


def test_wider_bands_take_over_the_body_and_the_overhang_reaches_further():
    wide = RegionParams(band_mm=4.0, overhang_mm=6.0)
    assert _classify([(72.0, 138.5), (62.0, 134.0)], params=wide) == ["ref_top_in", "ref_top_out"]
    assert _classify([(72.0, 138.5), (62.0, 134.0)]) == ["ref_body", "outer"]


def test_segments_last_until_the_next_one_starts():
    t = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    found = segments(t, np.array([1, 1, 2, 2, 1]))
    assert [(s.code, s.start_s, s.end_s) for s in found] == [(1, 0.0, 0.2), (2, 0.2, 0.4), (1, 0.4, 0.4)]
    assert sum(s.duration_s for s in found) == pytest.approx(0.4)
    assert segments(t[:0], np.array([], dtype=int)) == []


def _sweep(y: float, seconds: float):
    """Finger sliding left to right across both bars at height ``y``."""
    t = np.arange(round(seconds * HZ)) / HZ
    return t, 40.0 + (150.0 - 40.0) * t / t[-1], np.full_like(t, y)


def _write_trace(path: Path) -> None:
    with TraceStore(path) as store:
        store.create_session(
            session_id="P01_S1",
            participant_id="P01",
            started_at="2026-10-30T10:00:00+03:00",
            config={"inter_bar_gap_mm": 40.0},
            calibration={"active_width_mm": 194.0, "active_height_mm": 145.0},
            app_version="test",
        )
        for index, practice in enumerate((False, True), start=1):
            attempt_id = store.start_attempt(
                AttemptDefinition(
                    session_id="P01_S1",
                    trial_index=index,
                    attempt_index=1,
                    started_us=0,
                    level_pct=0.2,
                    comparison_height_mm=12.0,
                    reference_height_mm=10.0,
                    bar_width_mm=10.0,
                    reference_side="left",
                    is_catch=False,
                    is_practice=practice,
                    practice_round=int(practice),
                )
            )
            t, x, y = _sweep(140.0, 3.0)
            lifted = (t > 2.6) & (t < 2.75)  # past the comparison bar, x about 136-141 mm
            store.append_samples(
                attempt_id,
                [
                    CursorSample(
                        i, round(t[i] * 1e6), 16_667, 0, 0, float(x[i]), float(y[i]), 0.0, True, None, False,
                        contact=not lifted[i],
                    )  # fmt: skip
                    for i in range(len(t))
                ],
            )
            store.finish_attempt(attempt_id, ended_us=0, outcome="answered", response="right", response_time_us=0)


def test_run_writes_a_figure_per_trial_an_overview_and_the_segments(tmp_path):
    trace = tmp_path / "P01_S1_trace.sqlite3"
    _write_trace(trace)
    before = trace.read_bytes()

    assert run(trace, "all", tmp_path / "out", params=PARAMS) == 0
    session_dir = tmp_path / "out" / "P01" / "P01_S1"
    assert sorted(p.name for p in session_dir.iterdir()) == [
        "main_trial001_attempt1_id1.png",
        "overview.png",
        "segments.csv",
    ]
    assert trace.read_bytes() == before  # the recording is opened read-only

    with (session_dir / "segments.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert {row["attempt_id"] for row in rows} == {"1"}  # practice is left out by default
    assert [row["region"] for row in rows] == [
        "outer", "ref_side_out", "ref_side_in", "ref_body", "ref_side_in", "ref_side_out", "gap",
        "comp_side_out", "comp_side_in", "comp_body", "comp_side_in", "comp_side_out", "outer",
        "no_contact", "outer",
    ]  # fmt: skip
    total_ms = sum(float(row["duration_ms"]) for row in rows)
    assert total_ms == pytest.approx(float(rows[-1]["end_s"]) * 1000.0, abs=0.5)

    assert run(trace, "all", tmp_path / "with_practice", include_practice=True) == 0
    assert (tmp_path / "with_practice" / "P01" / "P01_S1" / "practice1_trial002_attempt1_id2.png").exists()
    assert run(trace, "nobody", tmp_path / "none") == 2


def test_region_codes_match_their_keys():
    assert all(REGIONS[code].key == key for key, code in CODE.items())
