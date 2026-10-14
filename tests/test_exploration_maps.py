from __future__ import annotations

import csv
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from analysis.exploration_maps.contact import ContactParams, classify
from analysis.exploration_maps.grid import accumulate, aggregate, cells_on_segment, make_grid
from analysis.exploration_maps.run import run
from analysis.exploration_maps.trace_io import load_trace
from experiment.trace_store import AttemptDefinition, CursorSample, TraceStore

HZ = 60.0
CELL = 2.0
# Default design: 10 mm bars, 40 mm gap, on a 194 x 145 mm surface, tallest comparison 13 mm.
GRID = make_grid(left_x0=67.0, right_x1=127.0, baseline_y=145.0, tallest_mm=13.0, cell_mm=CELL, margin_mm=10.0)
ROW_Y = 140.0  # centre of the cell row 139..141, crossing both bars
PARAMS = ContactParams()


def line(x0: float, y0: float, x1: float, y1: float, speed: float, t0: float = 0.0):
    """Samples at 60 Hz along a straight constant-speed line, ending exactly on its end point."""
    duration = float(np.hypot(x1 - x0, y1 - y0)) / speed
    t = np.append(np.arange(0.0, duration, 1.0 / HZ), duration)
    t = t[np.append(np.diff(t) > 1e-9, True)]
    return t0 + t, x0 + (x1 - x0) * t / duration, y0 + (y1 - y0) * t / duration


def hold(x: float, y: float, seconds: float, t0: float):
    """Samples at 60 Hz with the finger not moving, starting one frame after ``t0``."""
    t = t0 + np.arange(1, round(seconds * HZ) + 1) / HZ
    return t, np.full_like(t, x), np.full_like(t, y)


def path(*pieces):
    return tuple(np.concatenate(parts) for parts in zip(*pieces))


def maps_for(t, x, y, params: ContactParams = PARAMS, contact=None):
    result = classify(t, x, y, contact, params)
    return result, accumulate(t, x, y, result.include, result.resting, GRID)


def cell_of(x: float, y: float) -> tuple[int, int]:
    return int((y - GRID.y_edges[0]) // CELL), int((x - GRID.x_edges[0]) // CELL)


def test_grid_covers_bars_plus_margin_with_bar_edges_on_grid_lines():
    assert GRID.x_edges[0] == pytest.approx(57.0) and GRID.x_edges[-1] == pytest.approx(137.0)
    assert GRID.y_edges[0] <= 145.0 - 13.0 - 10.0 and GRID.y_edges[-1] == pytest.approx(155.0)
    for edge in (67.0, 77.0, 117.0, 127.0):
        assert np.isclose(GRID.x_edges, edge).any()
    assert np.isclose(GRID.y_edges, 145.0).any()


@pytest.mark.parametrize("speed", [50.0, 100.0, 200.0])
def test_constant_speed_sweep_visits_each_crossed_cell_once_with_dwell_cell_over_speed(speed):
    t, x, y = line(GRID.x_edges[0], ROW_Y, GRID.x_edges[-1], ROW_Y, speed)
    contact, maps = maps_for(t, x, y)

    row = cell_of(60.0, ROW_Y)[0]
    crossed = np.zeros(GRID.shape, dtype=bool)
    crossed[row, :] = True
    assert (maps.visits[crossed] == 1).all()
    assert (maps.visits[~crossed] == 0).all()
    # Equal across cells and proportional to 1 / speed, although 60 Hz samples do not line up with cells.
    assert maps.dwell_s[crossed] == pytest.approx(CELL / speed, abs=1e-9)
    assert (maps.dwell_s[~crossed] == 0).all()
    assert contact.excluded_ms == pytest.approx(0.0)


def test_two_second_stop_keeps_one_visit_and_adds_two_seconds_of_dwell():
    speed = 100.0
    first = line(61.0, ROW_Y, 90.0, ROW_Y, speed)
    stop = hold(90.0, ROW_Y, 2.0, first[0][-1])
    t, x, y = path(first, stop, line(90.0, ROW_Y, 120.0, ROW_Y, speed, stop[0][-1]))
    contact, maps = maps_for(t, x, y)

    stopped = cell_of(90.0, ROW_Y)
    assert maps.visits[stopped] == 1
    assert maps.dwell_s[stopped] == pytest.approx(2.0 + CELL / speed, abs=1e-6)
    assert contact.lift_ms == 0 and contact.excluded_ms == pytest.approx(0.0)
    assert contact.resting.sum() > 100  # the stop was recognised as a finger resting on the surface


def test_dwell_map_sums_to_in_contact_time():
    rng = np.random.default_rng(7)
    # A wandering path that stays inside the grid: every included millisecond must land in a cell.
    t = np.arange(600) / HZ
    x = np.clip(97.0 + np.cumsum(rng.normal(0, 1.2, 600)), 58.0, 136.0)
    y = np.clip(138.0 + np.cumsum(rng.normal(0, 0.8, 600)), 122.0, 154.0)
    contact, maps = maps_for(t, x, y)
    assert maps.outside_s == 0.0
    assert maps.dwell_s.sum() == pytest.approx(contact.included_ms / 1000.0, abs=1e-9)
    assert contact.included_ms + contact.excluded_ms == pytest.approx((t[-1] - t[0]) * 1000.0)

    # Leaving the grid moves time to ``outside_s``; a lift removes it from both.
    lead = hold(60.0, ROW_Y, 0.5, -1.0 / HZ)
    sweep = line(60.0, ROW_Y, 80.0, ROW_Y, 100.0, lead[0][-1])
    air = hold(80.0, ROW_Y, 0.4, sweep[0][-1])
    away = line(120.0, 100.0, 120.0, 140.0, 80.0, air[0][-1] + 1.0 / HZ)
    t, x, y = path(lead, sweep, air, away)
    contact, maps = maps_for(t, x, y)
    assert maps.outside_s > 0.1
    assert maps.dwell_s.sum() + maps.outside_s == pytest.approx(contact.included_ms / 1000.0, abs=1e-9)
    assert contact.excluded_ms == pytest.approx(
        contact.start_freeze_ms + contact.lift_ms + contact.jump_ms, abs=1e-6
    )


def test_freeze_before_the_first_movement_is_excluded():
    lead = hold(60.0, ROW_Y, 0.5, -1.0 / HZ)
    t, x, y = path(lead, line(60.0, ROW_Y, 100.0, ROW_Y, 100.0, lead[0][-1]))
    contact, maps = maps_for(t, x, y)

    assert contact.start_freeze_ms == pytest.approx(500.0 - 1000.0 / HZ, abs=1e-6)
    assert maps.dwell_s[cell_of(60.0, ROW_Y)] < CELL / 100.0 + 1e-9
    assert maps.visits[cell_of(60.0, ROW_Y)] == 1


def test_freeze_followed_by_a_jump_is_a_lift_and_is_excluded():
    first = line(60.0, ROW_Y, 80.0, ROW_Y, 100.0)
    air = hold(80.0, ROW_Y, 0.5, first[0][-1])
    t, x, y = path(first, air, line(110.0, ROW_Y, 130.0, ROW_Y, 100.0, air[0][-1] + 1.0 / HZ))
    contact, maps = maps_for(t, x, y)

    assert contact.lift_ms == pytest.approx(500.0, abs=1e-6)
    assert contact.jump_segments == 1
    assert maps.dwell_s[cell_of(79.9, ROW_Y)] < 0.05  # the 0.5 s in the air is not dwell
    assert maps.visits[cell_of(110.0, ROW_Y)] == 1  # touching down again enters a cell
    assert maps.visits[cell_of(95.0, ROW_Y)] == 0  # nothing is drawn along the jump


def test_freeze_until_the_response_is_excluded_by_default_and_reported():
    first = line(60.0, ROW_Y, 100.0, ROW_Y, 100.0)
    t, x, y = path(first, hold(100.0, ROW_Y, 1.0, first[0][-1]))
    stopped = cell_of(99.9, ROW_Y)

    contact, maps = maps_for(t, x, y)
    assert contact.end_freeze_excluded and contact.end_freeze_ms == pytest.approx(1000.0, abs=1e-6)
    assert maps.dwell_s[stopped] < 0.05

    contact, maps = maps_for(t, x, y, ContactParams(keep_end_freeze=True))
    assert not contact.end_freeze_excluded and contact.end_freeze_ms == pytest.approx(1000.0, abs=1e-6)
    assert maps.dwell_s[stopped] > 1.0
    assert "kept" in contact.describe(ContactParams(keep_end_freeze=True))


def test_resting_finger_jittering_over_a_grid_line_adds_dwell_but_no_visits():
    first = line(61.0, ROW_Y, 91.0, ROW_Y, 100.0)  # stops exactly on the 91 mm grid line
    t_hold, x_hold, y_hold = hold(91.0, ROW_Y, 2.0, first[0][-1])
    x_hold = x_hold + 0.2 * np.where(np.arange(len(x_hold)) % 2 == 0, -1.0, 1.0)
    t, x, y = path(first, (t_hold, x_hold, y_hold), line(91.0, ROW_Y, 120.0, ROW_Y, 100.0, t_hold[-1]))
    contact, maps = maps_for(t, x, y)

    before, after = cell_of(90.0, ROW_Y), cell_of(92.0, ROW_Y)
    assert maps.visits[before] == 1 and maps.visits[after] == 1
    assert maps.dwell_s[before] + maps.dwell_s[after] == pytest.approx(2.0 + 2 * CELL / 100.0, abs=0.02)


def test_slow_exploration_is_not_mistaken_for_a_freeze():
    t, x, y = line(70.0, 128.0, 70.0, 138.0, 5.0)  # 5 mm/s down onto the bar top
    contact, _ = maps_for(t, x, y)
    assert contact.excluded_ms == pytest.approx(0.0)
    assert not contact.resting.any()


def test_recorded_contact_flag_replaces_the_heuristic():
    lead = hold(60.0, ROW_Y, 0.5, -1.0 / HZ)
    t, x, y = path(lead, line(60.0, ROW_Y, 100.0, ROW_Y, 100.0, lead[0][-1]))
    touching = np.ones(len(t), dtype=bool)
    touching[40:46] = False
    contact, maps = maps_for(t, x, y, contact=touching)

    assert contact.method == "contact flag"
    assert contact.start_freeze_ms == 0  # in contact and still: kept, unlike the heuristic
    assert contact.no_contact_ms == pytest.approx(7 * 1000.0 / HZ, abs=1e-6)
    assert maps.dwell_s[cell_of(60.0, ROW_Y)] > 0.4
    assert "recorded flag" in contact.describe(PARAMS)


def test_segment_through_a_grid_corner_does_not_visit_the_cells_beside_it():
    cells, fractions = cells_on_segment(58.0, 140.0, 60.0, 142.0, GRID)
    assert cells == [cell_of(58.0, 140.0), cell_of(60.0, 142.0)]
    assert fractions == pytest.approx([0.5, 0.5])


def test_per_trial_normalization_stops_a_long_trial_from_dominating():
    long_trial = np.zeros((2, 2))
    long_trial[0, 0] = 10.0
    short_trial = np.zeros((2, 2))
    short_trial[1, 1] = 1.0
    empty = np.zeros((2, 2))

    shares, n_used = aggregate([long_trial, short_trial, empty], normalize=True)
    assert n_used == 2
    assert shares[0, 0] == pytest.approx(50.0) and shares[1, 1] == pytest.approx(50.0)
    sums, _ = aggregate([long_trial, short_trial, empty], normalize=False)
    assert sums[0, 0] == pytest.approx(10.0) and sums[1, 1] == pytest.approx(1.0)


def _scrub(center_x: float, seconds: float):
    """Finger sliding back and forth across one bar, 4 mm either side of its centre."""
    t = np.arange(round(seconds * HZ)) / HZ
    return t, center_x + 4.0 * np.sin(2 * np.pi * 1.5 * t), np.full_like(t, ROW_Y)


def _write_trace(path: Path) -> None:
    """A session whose finger only ever explores the reference bar, on either side."""
    with TraceStore(path) as store:
        store.create_session(
            session_id="P01_S1",
            participant_id="P01",
            started_at="2026-10-14T10:00:00+03:00",
            config={"inter_bar_gap_mm": 40.0},
            calibration={"active_width_mm": 194.0, "active_height_mm": 145.0},
            app_version="test",
        )
        trials = [("left", 72.0, 8.0, False), ("right", 122.0, 2.0, False), ("right", 122.0, 2.0, True)]
        for index, (side, center_x, seconds, practice) in enumerate(trials, start=1):
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
                    reference_side=side,
                    is_catch=False,
                    is_practice=practice,
                    practice_round=int(practice),
                )
            )
            t, x, y = _scrub(center_x, seconds)
            store.append_samples(
                attempt_id,
                [
                    CursorSample(i, round(t[i] * 1e6), 16_667, 0, 0, float(x[i]), float(y[i]), 0.0, True, None, False)
                    for i in range(len(t))
                ],
            )
            store.finish_attempt(attempt_id, ended_us=0, outcome="answered", response="left", response_time_us=0)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def test_run_writes_trial_maps_and_a_mirrored_normalized_aggregate(tmp_path):
    trace = tmp_path / "P01_S1_trace.sqlite3"
    _write_trace(trace)
    before = trace.read_bytes()

    assert run(tmp_path, "P01", tmp_path / "maps") == 0
    session_dir = tmp_path / "maps" / "P01" / "P01_S1"
    assert sorted(p.name for p in session_dir.glob("*.png")) == [
        "aggregate.png",
        "main_trial001_attempt1_id1.png",
        "main_trial002_attempt1_id2.png",
    ]
    assert trace.read_bytes() == before  # the recording is opened read-only

    # Screen frame: the second trial's finger was on the right bar.
    right_trial = _rows(session_dir / "main_trial002_attempt1_id2.csv")
    assert all(float(r["x0_mm"]) >= 97.0 for r in right_trial if float(r["dwell_ms"]) > 0)

    # Reference frame: both trials fall on the left, and the 8 s trial weighs the same as the 2 s one.
    cells = _rows(session_dir / "aggregate.csv")
    assert all(float(r["x1_mm"]) <= 97.0 for r in cells if float(r["dwell_share_pct"]) > 0)
    assert sum(float(r["dwell_share_pct"]) for r in cells) == pytest.approx(100.0, abs=1e-3)
    assert sum(float(r["visits_share_pct"]) for r in cells) == pytest.approx(100.0, abs=1e-3)

    summary = _rows(session_dir / "trials.csv")
    assert [row["attempt_id"] for row in summary] == ["1", "2"]
    for row in summary:
        accounted = float(row["dwell_sum_ms"]) + float(row["outside_grid_ms"])
        assert accounted == pytest.approx(float(row["included_ms"]), abs=0.2)


def test_run_can_include_practice_and_reports_an_unknown_participant(tmp_path):
    trace = tmp_path / "P01_S1_trace.sqlite3"
    _write_trace(trace)

    assert run(trace, "all", tmp_path / "maps", include_practice=True) == 0
    assert (tmp_path / "maps" / "P01" / "P01_S1" / "practice1_trial003_attempt1_id3.png").exists()
    # Practice gets its own map but stays out of the aggregate.
    summary = _rows(tmp_path / "maps" / "P01" / "P01_S1" / "trials.csv")
    assert [row["in_aggregate"] for row in summary] == ["1", "1", "0"]
    assert run(trace, "nobody", tmp_path / "none") == 2


def test_load_trace_reads_the_contact_column_and_older_schemas_without_it(tmp_path):
    trace = tmp_path / "P01_S1_trace.sqlite3"
    _write_trace(trace)
    # Schema v3 writes contact; samples that never set it are "finger down".
    assert all(attempt.contact is not None and attempt.contact.all() for attempt in load_trace(trace))

    connection = sqlite3.connect(trace)
    connection.execute("UPDATE cursor_samples SET contact = 0 WHERE sequence < 10")
    connection.commit()
    attempts = load_trace(trace)
    assert [bool(a.contact[:10].any()) for a in attempts] == [False, False, False]
    assert all(a.contact[10:].all() for a in attempts)

    # Schema v1 had no contact, is_practice or practice_round columns.
    connection.executescript(
        """
        CREATE TABLE attempts_v1 AS
            SELECT attempt_id, session_id, trial_index, attempt_index, started_us, ended_us, level_pct,
                   comparison_height_mm, reference_height_mm, bar_width_mm, reference_side, is_catch,
                   outcome, response, response_time_us
            FROM attempts;
        DROP TABLE attempts;
        ALTER TABLE attempts_v1 RENAME TO attempts;
        ALTER TABLE cursor_samples DROP COLUMN contact;
        """
    )
    connection.commit()
    connection.close()
    attempts = load_trace(trace)
    assert all(a.contact is None and not a.is_practice for a in attempts)


def test_module_does_not_import_the_experiment_runtime_or_pygame():
    code = (
        "import sys, analysis.exploration_maps.run\n"
        "bad = [m for m in sys.modules if m == 'pygame' or m.split('.')[0] == 'experiment']\n"
        "raise SystemExit(1 if bad else 0)"
    )
    repo_root = Path(__file__).resolve().parent.parent
    assert subprocess.run([sys.executable, "-c", code], cwd=repo_root).returncode == 0
