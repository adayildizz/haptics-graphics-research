"""The force sync check on a synthetic session whose delays, rotation and zero are known."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from analysis import force_sync
from experiment import force_daq
from experiment.ati_calibration import load_ati_calibration
from experiment.force_session import REPO_ROOT
from experiment.trace_store import AttemptDefinition, CursorSample, TraceStore

T0 = 5000.0  # the experiment's perf_counter at the first force sample
RATE_HZ = 1000.0
DURATION_S = 12.0
HANDS_OFF = [("practice", 0.2, 1.2), ("main", 6.2, 7.2)]  # s after T0
STROKES = [(1.6, 3.4), (4.0, 5.6), (7.6, 9.4), (10.0, 11.6)]  # finger down, finger up (s after T0)
PRESS_N = 0.5
RAMP_S = 0.004  # Fn rises from 0 to PRESS_N in this time after touching down, and falls to 0 by lifting
ZERO_V = np.array([0.12, -0.31, 0.05, 0.44, -0.08, 0.27])  # the unloaded gauge volts
DRIFT_V = -0.004  # the zero moves by this much on every gauge between the two hands-off steps: +0.02 N of Fn
ROTATION_DEG = 54.0  # screen = R(54 deg) @ mirror @ sensor + SENSOR_AXIS_MM
SENSOR_AXIS_MM = np.array([97.0, 72.0])
CONTACT_DELAY_S = 0.030  # the frame reports touching down and lifting this much late ...
POSITION_DELAY_S = 0.025  # ... and the finger's position this much late
FRAME_S = 1.0 / 60.0


def finger_on_screen(t: np.ndarray) -> np.ndarray:
    """Where the finger is, in screen mm (y down), at ``t`` s after T0."""
    return np.column_stack([97.0 + 40.0 * np.sin(1.3 * t), 72.0 + 25.0 * np.sin(2.1 * t + 0.5)])


def press_force(t: np.ndarray) -> np.ndarray:
    fn = np.zeros_like(t)
    for down, up in STROKES:
        fn = np.maximum(fn, PRESS_N * np.clip(np.minimum(t - down, up - t) / RAMP_S, 0.0, 1.0))
    return fn


def gauge_volts(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The six gauge volts the card would read: the zero, plus the calibration undone on the finger's load."""
    angle = np.radians(ROTATION_DEG)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    mirror = np.diag([1.0, -1.0])
    sensor = (finger_on_screen(t) - SENSOR_AXIS_MM) @ np.linalg.inv(rotation @ mirror).T
    fn = press_force(t)
    zero = np.zeros_like(fn)
    loads = np.column_stack([zero, zero, -fn, -sensor[:, 1] * fn, sensor[:, 0] * fn, zero])  # Fx Fy Fz Tx Ty Tz
    matrix = np.array(load_ati_calibration(REPO_ROOT / "FT05411.cal").matrix)
    volts = loads @ np.linalg.inv(matrix).T + ZERO_V
    volts += np.where(t >= HANDS_OFF[1][1] - 0.5, DRIFT_V, 0.0)[:, None]
    return volts + rng.normal(0.0, 2e-5, volts.shape)


def write_force_file(path: Path, t: np.ndarray, volts: np.ndarray) -> None:
    """A force file as the reader writes it: raw volts, and reads that time them on perf_counter."""
    connection = sqlite3.connect(path)
    force_daq._create_schema(connection, 6)
    connection.executemany("INSERT INTO force_metadata VALUES (?, ?)", [("rate_hz", "1000.0"), ("channels", "6")])
    rows = [(k, *row) for k, row in enumerate(volts.tolist())]
    connection.executemany("INSERT INTO force_samples VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    connection.executemany("INSERT INTO force_reads VALUES (?, ?)", [(k + 1, T0 + t[k]) for k in range(1, len(t), 2)])
    connection.commit()
    connection.close()


def write_trace(path: Path, force_file: str) -> None:
    """One attempt per stroke: cursor samples at 60 Hz and contact events, as the IR frame reports them (late)."""
    with TraceStore(path) as store:
        store.create_session(
            session_id="SYN", participant_id="P", started_at="t", config={}, calibration={}, app_version="test"
        )
        store.set_metadata("force_file", force_file)
        store.set_metadata("force_cal_file", "FT05411.cal")
        windows = [{"block": block, "start_s": T0 + start, "end_s": T0 + end} for block, start, end in HANDS_OFF]
        store.set_metadata("force_hands_off", json.dumps(windows))
        for index, (down, up) in enumerate(STROKES, start=1):
            start = down - 0.4  # trial start, s after T0
            definition = AttemptDefinition("SYN", index, 1, 0, 0.2, 12.0, 10.0, 10.0, "left", False)
            attempt_id = store.start_attempt(definition)
            store.set_trial_start(attempt_id, T0 + start)
            t = start + np.arange(0.0, up + 0.4 - start, FRAME_S)
            seen = finger_on_screen(t - POSITION_DELAY_S)
            contact = (t >= down + CONTACT_DELAY_S) & (t < up + CONTACT_DELAY_S)
            samples = [
                CursorSample(k, round((t[k] - start) * 1e6), 16_667, 0, 0, x, y, 0.0, True, None, False, c)
                for k, ((x, y), c) in enumerate(zip(seen.tolist(), contact.tolist()))
            ]
            store.append_samples(attempt_id, samples)
            for kind, at in (("contact_down", down), ("contact_up", up)):
                store.append_event(attempt_id, t_us=round((at + CONTACT_DELAY_S - start) * 1e6), event_type=kind)


@pytest.fixture
def session(tmp_path) -> Path:
    t = np.arange(0.0, DURATION_S, 1.0 / RATE_HZ)
    write_force_file(tmp_path / "SYN_force.sqlite3", t, gauge_volts(t, np.random.default_rng(3)))
    write_trace(tmp_path / "SYN_trace.sqlite3", "SYN_force.sqlite3")
    return tmp_path / "SYN_trace.sqlite3"


def test_the_check_recovers_the_known_delays_rotation_and_sensor_position(session, tmp_path, capsys):
    result = force_sync.report(session, tmp_path / "out")

    assert result["lag_ms"] == round(1000 * POSITION_DELAY_S)
    fit = result["fit"]
    assert fit.rotation_deg == pytest.approx(ROTATION_DEG, abs=0.2) and fit.mirrored
    assert fit.scale == pytest.approx(1.0, abs=0.005)
    assert fit.shift == pytest.approx(SENSOR_AXIS_MM, abs=0.2)
    assert fit.rms_mm < 0.2
    # A press is found at the first sample above 0.03 N (up to 1 ms after touching down) and ends when Fn is 0.
    assert result["contact_ms"]["contact_down"] == pytest.approx(1000 * CONTACT_DELAY_S, abs=1.5)
    assert result["contact_ms"]["contact_up"] == pytest.approx(1000 * CONTACT_DELAY_S, abs=1.5)

    out = capsys.readouterr().out
    assert "best lag +25 ms" in out and "mirrored" in out
    # The second zero takes out the drift: without it the main block's presses would never end.
    assert "contact_up   vs press end (Fn < 0.015 N): 4 of 4 paired" in out
    rows = (tmp_path / "out" / "SYN_force_sync.csv").read_text().splitlines()
    assert rows[0] == "event,ir_s,force_s,diff_ms" and len(rows) == 1 + 2 * len(STROKES)
    assert (tmp_path / "out" / "SYN_force_sync.png").stat().st_size > 0


def test_presses_start_and_end_once_despite_noise_around_a_threshold():
    t = np.arange(10) / 1000.0
    fn = np.array([0.0, 0.031, 0.029, 0.031, 0.2, 0.02, 0.2, 0.014, 0.016, 0.0])
    starts, ends = force_sync.press_edges(t, fn)
    assert starts.tolist() == [0.001] and ends.tolist() == [0.007]


def test_the_similarity_fit_undoes_a_mirror_rotation_scale_and_shift():
    rng = np.random.default_rng(1)
    sensor = rng.uniform(-50, 50, (200, 2))
    angle = np.radians(-120.0)
    matrix = np.array([[np.cos(angle), np.sin(angle)], [np.sin(angle), -np.cos(angle)]])  # mirror, then rotate
    fit = force_sync.fit_similarity(sensor, 1.1 * sensor @ matrix.T + [3.0, -4.0])
    assert fit.mirrored and fit.rotation_deg == pytest.approx(-120.0) and fit.scale == pytest.approx(1.1)
    assert fit.shift == pytest.approx([3.0, -4.0]) and fit.rms_mm == pytest.approx(0.0, abs=1e-9)


def test_a_trace_without_the_force_sensor_is_refused_with_a_reason(tmp_path, capsys):
    with TraceStore(tmp_path / "NOFORCE_trace.sqlite3"):
        pass
    assert force_sync.main([str(tmp_path / "NOFORCE_trace.sqlite3")]) == 1
    assert "recorded without the force sensor" in capsys.readouterr().err
