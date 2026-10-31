from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from analysis.trace_movement import animate_attempt, bar_geometry, drawn_path, select_attempts
from experiment.demo_trace import ensure_demo_trace
from experiment.replay_data import load_trace
from experiment.trace_store import AttemptDefinition, CursorSample, TraceStore

FRAME_US = 16_667
# Finger down at the left, lifted for two samples (the trace repeats the last position), down again at the right.
LIFT_SAMPLES = [((60.0, 140.0), True), ((61.0, 140.0), True), ((61.0, 140.0), False), ((61.0, 140.0), False),
                ((130.0, 140.0), True), ((131.0, 140.0), True)]  # fmt: skip


@pytest.fixture
def demo_session(tmp_path):
    return load_trace(ensure_demo_trace(tmp_path / "demo_replay_trace.sqlite3"))


def _write_lift_trace(path):
    with TraceStore(path) as store:
        store.create_session(
            session_id="P01_S1",
            participant_id="P01",
            started_at="2026-10-31T10:00:00+03:00",
            config={"inter_bar_gap_mm": 40.0},
            calibration={"active_width_mm": 194.0, "active_height_mm": 145.0},
            app_version="test",
        )
        attempt_id = store.start_attempt(
            AttemptDefinition(
                session_id="P01_S1", trial_index=1, attempt_index=1, started_us=0, level_pct=0.2,
                comparison_height_mm=12.0, reference_height_mm=10.0, bar_width_mm=10.0, reference_side="left",
                is_catch=False,
            )
        )  # fmt: skip
        store.append_samples(
            attempt_id,
            [
                CursorSample(i, i * FRAME_US, FRAME_US, 0, 0, x, y, 0.0, True, None, False, contact=down)
                for i, ((x, y), down) in enumerate(LIFT_SAMPLES)
            ],
        )
        store.finish_attempt(attempt_id, ended_us=len(LIFT_SAMPLES) * FRAME_US, outcome="answered", response="right",
                             response_time_us=len(LIFT_SAMPLES) * FRAME_US)  # fmt: skip
    return path


def test_bar_geometry_matches_live_layout(demo_session):
    attempt = demo_session.attempts[0]  # level -18%, reference left -> comparison right, shorter
    geometry = bar_geometry(demo_session, attempt)

    assert geometry.surface_width_mm == pytest.approx(194.0)
    assert geometry.baseline_y == pytest.approx(145.0)
    # 40 mm gap between the bars' inner edges, both 10 mm wide, centred.
    assert geometry.right_x0 - geometry.left_x1 == pytest.approx(40.0)
    assert (geometry.left_x0 + geometry.right_x1) / 2 == pytest.approx(97.0)
    assert geometry.left_is_comparison is False
    assert geometry.left_height_mm == pytest.approx(10.0)
    assert geometry.right_height_mm == pytest.approx(8.2)
    assert geometry.taller_side == "left"


def test_selection_defaults_to_main_block(demo_session):
    assert all(not a.is_practice for a in select_attempts(demo_session))
    assert [a.trial_index for a in select_attempts(demo_session, trial=2)] == [2, 2]


def test_animation_renders_gif(demo_session, tmp_path):
    attempt = demo_session.attempts[0]
    out = animate_attempt(demo_session, attempt, tmp_path / "trial.gif", fps=10, speed=4.0, max_frames=4)
    assert out.exists() and out.stat().st_size > 0


def test_a_lift_breaks_the_path_and_the_state_reports_it(tmp_path):
    session = load_trace(_write_lift_trace(tmp_path / "P01_S1_trace.sqlite3"))
    attempt = session.attempts[0]
    x, y = drawn_path(attempt)

    assert np.isnan(x).tolist() == [False, False, True, True, False, False]
    assert np.isnan(y).tolist() == np.isnan(x).tolist()
    assert x[4] == pytest.approx(130.0)  # no stroke from 61 mm to 130 mm
    assert attempt.state_at(round(2.5 * FRAME_US)).contact is False
    assert attempt.state_at(round(4.5 * FRAME_US)).contact is True

    out = animate_attempt(session, attempt, tmp_path / "lift.gif", fps=60)
    assert out.exists() and out.stat().st_size > 0


def test_traces_without_a_contact_column_are_drawn_as_finger_down(tmp_path):
    trace = _write_lift_trace(tmp_path / "P01_S1_trace.sqlite3")
    connection = sqlite3.connect(trace)
    connection.execute("ALTER TABLE cursor_samples DROP COLUMN contact")  # as before schema v3
    connection.commit()
    connection.close()

    attempt = load_trace(trace).attempts[0]
    assert all(sample.contact for sample in attempt.samples)
    assert not np.isnan(drawn_path(attempt)[0]).any()
