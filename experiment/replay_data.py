"""Read-only models and queries for recorded experiment traces."""

from __future__ import annotations

import bisect
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReplaySample:
    sequence: int
    t_us: int
    frame_dt_us: int
    x_px: int
    y_px: int
    x_mm: float
    y_mm: float
    speed_mm_s: float
    in_active_area: bool
    active_side: str | None
    signal_on: bool
    contact: bool = True  # finger on the glass; traces before schema v3 have no flag


@dataclass(frozen=True)
class ReplayEvent:
    t_us: int
    event_type: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ReplayFrameState:
    x_px: float
    y_px: float
    x_mm: float
    y_mm: float
    speed_mm_s: float
    in_active_area: bool
    active_side: str | None
    signal_on: bool
    contact: bool = True


@dataclass(frozen=True)
class ReplayAttempt:
    attempt_id: int
    trial_index: int
    attempt_index: int
    started_us: int
    ended_us: int | None
    level_pct: float
    comparison_height_mm: float
    reference_height_mm: float
    bar_width_mm: float
    reference_side: str
    is_catch: bool
    outcome: str | None
    response: str | None
    response_time_us: int | None
    samples: tuple[ReplaySample, ...]
    sample_times_us: tuple[int, ...]
    events: tuple[ReplayEvent, ...]
    is_practice: bool = False
    practice_round: int = 0

    @property
    def label(self) -> str:
        """Short human label: 'Practice 1 · Trial 3' or 'Trial 12'."""
        if self.is_practice:
            return f"Practice {self.practice_round} · Trial {self.trial_index}"
        return f"Trial {self.trial_index}"

    @property
    def duration_us(self) -> int:
        candidates = [sample.t_us for sample in self.samples]
        candidates.extend(event.t_us for event in self.events)
        if self.response_time_us is not None:
            candidates.append(self.response_time_us)
        if self.ended_us is not None:
            candidates.append(max(0, self.ended_us - self.started_us))
        return max(candidates, default=0)

    @property
    def correct(self) -> bool | None:
        """Return the recorded result, or reconstruct it for older traces."""
        if self.outcome != "answered" or self.response is None:
            return None
        for event in reversed(self.events):
            if event.event_type != "response" or "correct" not in event.payload:
                continue
            recorded = event.payload["correct"]
            if isinstance(recorded, bool):
                return recorded
            if isinstance(recorded, int) and recorded in (0, 1):
                return bool(recorded)

        comparison_side = "right" if self.reference_side == "left" else "left"
        taller_side = comparison_side if self.level_pct > 0 else self.reference_side
        return self.response == taller_side

    def state_at(self, t_us: int) -> ReplayFrameState | None:
        """Interpolate cursor position; retain the previous discrete signal and contact state."""
        if not self.samples:
            return None
        right_index = bisect.bisect_right(self.sample_times_us, t_us)
        if right_index <= 0:
            left = right = self.samples[0]
        elif right_index >= len(self.samples):
            left = right = self.samples[-1]
        else:
            left = self.samples[right_index - 1]
            right = self.samples[right_index]

        span = right.t_us - left.t_us
        ratio = 0.0 if span <= 0 else (t_us - left.t_us) / span
        return ReplayFrameState(
            x_px=left.x_px + ratio * (right.x_px - left.x_px),
            y_px=left.y_px + ratio * (right.y_px - left.y_px),
            x_mm=left.x_mm + ratio * (right.x_mm - left.x_mm),
            y_mm=left.y_mm + ratio * (right.y_mm - left.y_mm),
            speed_mm_s=left.speed_mm_s + ratio * (right.speed_mm_s - left.speed_mm_s),
            in_active_area=left.in_active_area,
            active_side=left.active_side,
            signal_on=left.signal_on,
            contact=left.contact,
        )


@dataclass(frozen=True)
class ReplaySession:
    path: Path
    session_id: str
    participant_id: str
    started_at: str
    config: dict[str, Any]
    calibration: dict[str, Any]
    app_version: str
    attempts: tuple[ReplayAttempt, ...]


def load_trace(path: str | Path) -> ReplaySession:
    """Load one trace database without modifying it."""
    trace_path = Path(path).resolve()
    connection = sqlite3.connect(f"file:{trace_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        session_row = connection.execute(
            """
            SELECT session_id, participant_id, started_at,
                   config_json, calibration_json, app_version
            FROM sessions
            LIMIT 1
            """
        ).fetchone()
        if session_row is None:
            raise ValueError(f"trace has no session row: {trace_path}")

        # Schema v1 traces (recorded before practice was traced) lack the
        # practice columns; read them as main-block attempts.
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(attempts)")}
        practice_select = (
            "is_practice, practice_round"
            if "is_practice" in columns
            else "0 AS is_practice, 0 AS practice_round"
        )
        # Traces before schema v3 have no contact flag; read them as finger down throughout.
        sample_columns = {row["name"] for row in connection.execute("PRAGMA table_info(cursor_samples)")}
        contact_select = "contact" if "contact" in sample_columns else "1 AS contact"
        # started_us is session-elapsed time, so this is true chronological
        # order: practice rounds first, then the main block, retries where
        # they were actually shown.
        attempt_rows = connection.execute(
            f"""
            SELECT attempt_id, trial_index, attempt_index, started_us, ended_us,
                   level_pct, comparison_height_mm, reference_height_mm,
                   bar_width_mm, reference_side, is_catch,
                   outcome, response, response_time_us,
                   {practice_select}
            FROM attempts
            ORDER BY started_us, attempt_id
            """
        ).fetchall()

        attempts: list[ReplayAttempt] = []
        for row in attempt_rows:
            sample_rows = connection.execute(
                f"""
                SELECT sequence, t_us, frame_dt_us, x_px, y_px, x_mm, y_mm,
                       speed_mm_s, in_active_area, active_side, signal_on, {contact_select}
                FROM cursor_samples
                WHERE attempt_id = ?
                ORDER BY sequence
                """,
                (row["attempt_id"],),
            ).fetchall()
            event_rows = connection.execute(
                """
                SELECT t_us, event_type, payload_json
                FROM trace_events
                WHERE attempt_id = ?
                ORDER BY t_us, event_id
                """,
                (row["attempt_id"],),
            ).fetchall()
            samples = tuple(
                ReplaySample(
                    sequence=sample["sequence"],
                    t_us=sample["t_us"],
                    frame_dt_us=sample["frame_dt_us"],
                    x_px=sample["x_px"],
                    y_px=sample["y_px"],
                    x_mm=sample["x_mm"],
                    y_mm=sample["y_mm"],
                    speed_mm_s=sample["speed_mm_s"],
                    in_active_area=bool(sample["in_active_area"]),
                    active_side=sample["active_side"],
                    signal_on=bool(sample["signal_on"]),
                    contact=bool(sample["contact"]),
                )
                for sample in sample_rows
            )
            attempts.append(
                ReplayAttempt(
                    attempt_id=row["attempt_id"],
                    trial_index=row["trial_index"],
                    attempt_index=row["attempt_index"],
                    started_us=row["started_us"],
                    ended_us=row["ended_us"],
                    level_pct=row["level_pct"],
                    comparison_height_mm=row["comparison_height_mm"],
                    reference_height_mm=row["reference_height_mm"],
                    bar_width_mm=row["bar_width_mm"],
                    reference_side=row["reference_side"],
                    is_catch=bool(row["is_catch"]),
                    outcome=row["outcome"],
                    response=row["response"],
                    response_time_us=row["response_time_us"],
                    is_practice=bool(row["is_practice"]),
                    practice_round=int(row["practice_round"]),
                    samples=samples,
                    sample_times_us=tuple(sample.t_us for sample in samples),
                    events=tuple(
                        ReplayEvent(
                            t_us=event["t_us"],
                            event_type=event["event_type"],
                            payload=json.loads(event["payload_json"]),
                        )
                        for event in event_rows
                    ),
                )
            )

        return ReplaySession(
            path=trace_path,
            session_id=session_row["session_id"],
            participant_id=session_row["participant_id"],
            started_at=session_row["started_at"],
            config=json.loads(session_row["config_json"]),
            calibration=json.loads(session_row["calibration_json"]),
            app_version=session_row["app_version"],
            attempts=tuple(attempts),
        )
    finally:
        connection.close()


def discover_trace_files(data_dir: str | Path) -> list[Path]:
    """Return trace databases newest first, with the demo last."""
    paths = list(Path(data_dir).glob("*_trace.sqlite3"))
    return sorted(
        paths,
        key=lambda path: (path.name == "demo_replay_trace.sqlite3", -path.stat().st_mtime),
    )
