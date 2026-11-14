from __future__ import annotations

import json
import sqlite3

from experiment.trace_store import AttemptDefinition, CursorSample, TraceStore


def _attempt(attempt_index: int = 1) -> AttemptDefinition:
    return AttemptDefinition(
        session_id="P01_session",
        trial_index=12,
        attempt_index=attempt_index,
        started_us=1_000_000,
        level_pct=0.18,
        comparison_height_mm=11.8,
        reference_height_mm=10.0,
        bar_width_mm=10.0,
        reference_side="left",
        is_catch=False,
    )


def test_trace_store_round_trip(tmp_path):
    path = tmp_path / "session_trace.sqlite3"
    with TraceStore(path) as store:
        store.create_session(
            session_id="P01_session",
            participant_id="P01",
            started_at="2026-08-14T12:00:00+03:00",
            config={"response_timeout_s": 30.0},
            calibration={"px_per_mm_x": 7.7, "px_per_mm_y": 5.8},
            app_version="test-version",
        )
        attempt_id = store.start_attempt(_attempt())
        store.append_samples(
            attempt_id,
            [
                CursorSample(0, 0, 16_667, 100, 200, 10.0, 20.0, 0.0, True, None, False),
                CursorSample(1, 16_667, 16_667, 108, 200, 11.0, 20.0, 60.0, True, "left", True),
            ],
        )
        store.append_event(attempt_id, t_us=16_667, event_type="signal_on", payload={"side": "left"})
        store.finish_attempt(
            attempt_id,
            ended_us=2_500_000,
            outcome="answered",
            response="right",
            response_time_us=1_500_000,
        )

    connection = sqlite3.connect(path)
    session = connection.execute(
        "SELECT participant_id, config_json, calibration_json FROM sessions"
    ).fetchone()
    assert session is not None
    assert session[0] == "P01"
    assert json.loads(session[1]) == {"response_timeout_s": 30.0}
    assert json.loads(session[2])["px_per_mm_x"] == 7.7

    attempt = connection.execute(
        "SELECT trial_index, attempt_index, outcome, response FROM attempts"
    ).fetchone()
    assert attempt == (12, 1, "answered", "right")

    samples = connection.execute(
        "SELECT sequence, t_us, x_px, active_side, signal_on FROM cursor_samples ORDER BY sequence"
    ).fetchall()
    assert samples == [(0, 0, 100, None, 0), (1, 16_667, 108, "left", 1)]

    event = connection.execute(
        "SELECT event_type, payload_json FROM trace_events"
    ).fetchone()
    assert event is not None
    assert event[0] == "signal_on"
    assert json.loads(event[1]) == {"side": "left"}
    connection.close()


def test_same_trial_can_store_multiple_attempts(tmp_path):
    path = tmp_path / "session_trace.sqlite3"
    with TraceStore(path) as store:
        store.create_session(
            session_id="P01_session",
            participant_id="P01",
            started_at="2026-08-14T12:00:00+03:00",
            config={},
            calibration={},
            app_version="test-version",
        )
        first_id = store.start_attempt(_attempt(attempt_index=1))
        store.finish_attempt(first_id, ended_us=31_000_000, outcome="timeout")
        second_id = store.start_attempt(_attempt(attempt_index=2))
        store.finish_attempt(
            second_id,
            ended_us=42_000_000,
            outcome="answered",
            response="left",
            response_time_us=11_000_000,
        )

    connection = sqlite3.connect(path)
    attempts = connection.execute(
        "SELECT trial_index, attempt_index, outcome FROM attempts ORDER BY attempt_index"
    ).fetchall()
    assert attempts == [(12, 1, "timeout"), (12, 2, "answered")]
    connection.close()


def test_practice_attempts_are_tagged_and_do_not_collide_with_main(tmp_path):
    """Practice trial 1 (round 1 and round 2) and main trial 1 coexist."""
    from dataclasses import replace

    path = tmp_path / "session_trace.sqlite3"
    with TraceStore(path) as store:
        store.create_session(
            session_id="P01_session", participant_id="P01",
            started_at="2026-08-14T12:00:00+03:00", config={}, calibration={}, app_version="t",
        )
        base = replace(_attempt(), trial_index=1)
        store.start_attempt(replace(base, is_practice=True, practice_round=1, started_us=0))
        store.start_attempt(replace(base, is_practice=True, practice_round=2, started_us=10))
        store.start_attempt(replace(base, started_us=20))

    connection = sqlite3.connect(path)
    rows = connection.execute(
        "SELECT is_practice, practice_round, trial_index FROM attempts ORDER BY started_us"
    ).fetchall()
    assert rows == [(1, 1, 1), (1, 2, 1), (0, 0, 1)]
    connection.close()

    from experiment.replay_data import load_trace

    session = load_trace(path)
    assert [(a.is_practice, a.practice_round) for a in session.attempts] == [(True, 1), (True, 2), (False, 0)]
    assert session.attempts[0].label == "Practice 1 \u00b7 Trial 1"
    assert session.attempts[2].label == "Trial 1"


def test_schema_v1_traces_load_as_main_block(tmp_path):
    """Traces recorded before practice was traced have no practice columns."""
    path = tmp_path / "old_trace.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY, participant_id TEXT, started_at TEXT,
            config_json TEXT, calibration_json TEXT, app_version TEXT);
        CREATE TABLE attempts (attempt_id INTEGER PRIMARY KEY, session_id TEXT, trial_index INTEGER,
            attempt_index INTEGER, started_us INTEGER, ended_us INTEGER, level_pct REAL,
            comparison_height_mm REAL, reference_height_mm REAL, bar_width_mm REAL,
            reference_side TEXT, is_catch INTEGER, outcome TEXT, response TEXT, response_time_us INTEGER);
        CREATE TABLE cursor_samples (attempt_id INTEGER, sequence INTEGER, t_us INTEGER, frame_dt_us INTEGER,
            x_px INTEGER, y_px INTEGER, x_mm REAL, y_mm REAL, speed_mm_s REAL, in_active_area INTEGER,
            active_side TEXT, signal_on INTEGER);
        CREATE TABLE trace_events (event_id INTEGER PRIMARY KEY, attempt_id INTEGER, t_us INTEGER,
            event_type TEXT, payload_json TEXT);
        INSERT INTO sessions VALUES ('S', 'P', 't', '{}', '{}', 'v1');
        INSERT INTO attempts VALUES (1, 'S', 3, 1, 0, 5000000, 0.3, 13.0, 10.0, 10.0, 'left', 0,
            'answered', 'right', 5000000);
        """
    )
    connection.commit()
    connection.close()

    from experiment.replay_data import load_trace

    session = load_trace(path)
    assert len(session.attempts) == 1
    assert session.attempts[0].is_practice is False
    assert session.attempts[0].practice_round == 0


def test_an_older_trace_gets_the_trial_start_column_and_its_old_attempts_leave_it_empty(tmp_path):
    path = tmp_path / "old_trace.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE attempts (attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, trial_index INTEGER,
            attempt_index INTEGER, started_us INTEGER, ended_us INTEGER, level_pct REAL,
            comparison_height_mm REAL, reference_height_mm REAL, bar_width_mm REAL,
            reference_side TEXT, is_catch INTEGER, outcome TEXT, response TEXT, response_time_us INTEGER);
        INSERT INTO attempts VALUES (1, 'P01_session', 3, 1, 0, 5000000, 0.3, 13.0, 10.0, 10.0, 'left', 0,
            'answered', 'right', 5000000);
        """
    )
    connection.commit()
    connection.close()

    with TraceStore(path) as store:
        attempt_id = store.start_attempt(_attempt(attempt_index=2))
        store.set_trial_start(attempt_id, 1000.25)

    connection = sqlite3.connect(path)
    starts = connection.execute("SELECT attempt_id, trial_start_s FROM attempts ORDER BY attempt_id").fetchall()
    connection.close()
    assert starts == [(1, None), (attempt_id, 1000.25)]
