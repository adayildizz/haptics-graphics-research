"""SQLite storage primitives for post-session cursor replay.

This module is deliberately independent from pygame, the trial loop, and the
hardware layer. Nothing imports it during an experiment yet; integration will
be added only after the storage format is reviewed and approved.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 3  # v2: attempts carry is_practice / practice_round; v3: cursor_samples.contact
# Columns added after v1, with the value older recordings implicitly carried.
ADDED_COLUMNS = (
    ("attempts", "is_practice", "INTEGER NOT NULL DEFAULT 0"),
    ("attempts", "practice_round", "INTEGER NOT NULL DEFAULT 0"),
    ("cursor_samples", "contact", "INTEGER NOT NULL DEFAULT 1"),
)

AttemptOutcome = Literal["answered", "timeout", "aborted"]
Side = Literal["left", "right"]


@dataclass(frozen=True)
class CursorSample:
    """One processed frame sample from the experiment loop."""

    sequence: int
    t_us: int
    frame_dt_us: int
    x_px: int
    y_px: int
    x_mm: float
    y_mm: float
    speed_mm_s: float
    in_active_area: bool
    active_side: Side | None
    signal_on: bool
    contact: bool = True  # finger on the glass; also True while the device has not shown it reports touch


@dataclass(frozen=True)
class AttemptDefinition:
    """Stimulus and timing metadata for one presentation attempt."""

    session_id: str
    trial_index: int
    attempt_index: int
    started_us: int
    level_pct: float
    comparison_height_mm: float
    reference_height_mm: float
    bar_width_mm: float
    reference_side: Side
    is_catch: bool
    # Practice attempts are recorded too (the finger-posture record is what
    # the practice block is there to clean up), tagged so analysis can keep
    # them apart from the main block. ``practice_round`` is 1-based for
    # practice and 0 for main-block attempts.
    is_practice: bool = False
    practice_round: int = 0


class TraceStore:
    """Transactional writer for one session trace database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS trace_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                config_json TEXT NOT NULL,
                calibration_json TEXT NOT NULL,
                app_version TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                trial_index INTEGER NOT NULL,
                attempt_index INTEGER NOT NULL,
                started_us INTEGER NOT NULL,
                ended_us INTEGER,
                level_pct REAL NOT NULL,
                comparison_height_mm REAL NOT NULL,
                reference_height_mm REAL NOT NULL,
                bar_width_mm REAL NOT NULL,
                reference_side TEXT NOT NULL CHECK (reference_side IN ('left', 'right')),
                is_catch INTEGER NOT NULL CHECK (is_catch IN (0, 1)),
                is_practice INTEGER NOT NULL DEFAULT 0 CHECK (is_practice IN (0, 1)),
                practice_round INTEGER NOT NULL DEFAULT 0,
                outcome TEXT CHECK (outcome IN ('answered', 'timeout', 'aborted')),
                response TEXT CHECK (response IS NULL OR response IN ('left', 'right')),
                response_time_us INTEGER,
                UNIQUE (session_id, is_practice, practice_round, trial_index, attempt_index)
            );

            CREATE TABLE IF NOT EXISTS cursor_samples (
                attempt_id INTEGER NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL,
                t_us INTEGER NOT NULL,
                frame_dt_us INTEGER NOT NULL,
                x_px INTEGER NOT NULL,
                y_px INTEGER NOT NULL,
                x_mm REAL NOT NULL,
                y_mm REAL NOT NULL,
                speed_mm_s REAL NOT NULL,
                in_active_area INTEGER NOT NULL CHECK (in_active_area IN (0, 1)),
                active_side TEXT CHECK (active_side IS NULL OR active_side IN ('left', 'right')),
                signal_on INTEGER NOT NULL CHECK (signal_on IN (0, 1)),
                contact INTEGER NOT NULL DEFAULT 1 CHECK (contact IN (0, 1)),
                PRIMARY KEY (attempt_id, sequence)
            );

            CREATE TABLE IF NOT EXISTS trace_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id INTEGER NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
                t_us INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS cursor_samples_attempt_time
                ON cursor_samples (attempt_id, t_us);
            CREATE INDEX IF NOT EXISTS trace_events_attempt_time
                ON trace_events (attempt_id, t_us);
            """
        )
        # An older recording reopened for writing gets the newer columns in place.
        for table, column, definition in ADDED_COLUMNS:
            existing = {row[1] for row in self.connection.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        self.connection.execute(
            "INSERT OR REPLACE INTO trace_metadata (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.connection.commit()

    def create_session(
        self,
        *,
        session_id: str,
        participant_id: str,
        started_at: str,
        config: dict[str, Any],
        calibration: dict[str, Any],
        app_version: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO sessions (
                session_id, participant_id, started_at,
                config_json, calibration_json, app_version
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                participant_id,
                started_at,
                json.dumps(config, sort_keys=True),
                json.dumps(calibration, sort_keys=True),
                app_version,
            ),
        )
        self.connection.commit()

    def start_attempt(self, attempt: AttemptDefinition) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO attempts (
                session_id, trial_index, attempt_index, started_us,
                level_pct, comparison_height_mm, reference_height_mm,
                bar_width_mm, reference_side, is_catch,
                is_practice, practice_round
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt.session_id,
                attempt.trial_index,
                attempt.attempt_index,
                attempt.started_us,
                attempt.level_pct,
                attempt.comparison_height_mm,
                attempt.reference_height_mm,
                attempt.bar_width_mm,
                attempt.reference_side,
                int(attempt.is_catch),
                int(attempt.is_practice),
                int(attempt.practice_round),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def append_samples(self, attempt_id: int, samples: list[CursorSample]) -> None:
        if not samples:
            return
        self.connection.executemany(
            """
            INSERT INTO cursor_samples (
                attempt_id, sequence, t_us, frame_dt_us,
                x_px, y_px, x_mm, y_mm, speed_mm_s,
                in_active_area, active_side, signal_on, contact
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    attempt_id,
                    sample.sequence,
                    sample.t_us,
                    sample.frame_dt_us,
                    sample.x_px,
                    sample.y_px,
                    sample.x_mm,
                    sample.y_mm,
                    sample.speed_mm_s,
                    int(sample.in_active_area),
                    sample.active_side,
                    int(sample.signal_on),
                    int(sample.contact),
                )
                for sample in samples
            ],
        )
        self.connection.commit()

    def append_event(
        self,
        attempt_id: int,
        *,
        t_us: int,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO trace_events (attempt_id, t_us, event_type, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (attempt_id, t_us, event_type, json.dumps(payload or {}, sort_keys=True)),
        )
        self.connection.commit()

    def finish_attempt(
        self,
        attempt_id: int,
        *,
        ended_us: int,
        outcome: AttemptOutcome,
        response: Side | None = None,
        response_time_us: int | None = None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE attempts
            SET ended_us = ?, outcome = ?, response = ?, response_time_us = ?
            WHERE attempt_id = ?
            """,
            (ended_us, outcome, response, response_time_us, attempt_id),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> TraceStore:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()
