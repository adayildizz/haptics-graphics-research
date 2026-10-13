"""Read-only loading of session trace databases, without importing the experiment package."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

CONTACT_COLUMN = "contact"


@dataclass(frozen=True)
class Session:
    path: Path
    session_id: str
    participant_id: str
    surface_width_mm: float
    surface_height_mm: float
    inter_bar_gap_mm: float


@dataclass(frozen=True)
class Bars:
    """Both bars in surface millimetres, y measured down from the surface's top edge."""

    left_x0: float
    right_x0: float
    width: float
    baseline_y: float
    left_height: float
    right_height: float
    left_is_reference: bool

    @property
    def right_x1(self) -> float:
        return self.right_x0 + self.width


@dataclass(frozen=True)
class Attempt:
    session: Session
    attempt_id: int
    trial_index: int
    attempt_index: int
    is_practice: bool
    practice_round: int
    is_catch: bool
    level_pct: float
    reference_height_mm: float
    comparison_height_mm: float
    bar_width_mm: float
    reference_side: str
    outcome: str | None
    response: str | None
    t_s: NDArray[np.float64]
    x_mm: NDArray[np.float64]
    y_mm: NDArray[np.float64]
    contact: NDArray[np.bool_] | None

    @property
    def block(self) -> str:
        return f"practice{self.practice_round}" if self.is_practice else "main"

    @property
    def stem(self) -> str:
        return f"{self.block}_trial{self.trial_index:03d}_attempt{self.attempt_index}_id{self.attempt_id}"

    @property
    def taller_side(self) -> str:
        other = "right" if self.reference_side == "left" else "left"
        return other if self.level_pct > 0 else self.reference_side

    @property
    def correct(self) -> bool | None:
        return None if self.response is None else self.response == self.taller_side

    def bars(self, *, reference_left: bool = False) -> Bars:
        """Bars as laid out on screen, or mirrored so the reference bar is on the left."""
        session = self.session
        center = session.surface_width_mm / 2.0
        left_is_reference = reference_left or self.reference_side == "left"
        ref, comp = self.reference_height_mm, self.comparison_height_mm
        return Bars(
            left_x0=center - session.inter_bar_gap_mm / 2.0 - self.bar_width_mm,
            right_x0=center + session.inter_bar_gap_mm / 2.0,
            width=self.bar_width_mm,
            baseline_y=session.surface_height_mm,
            left_height=ref if left_is_reference else comp,
            right_height=comp if left_is_reference else ref,
            left_is_reference=left_is_reference,
        )

    def x_reference_left(self) -> NDArray[np.float64]:
        """x mirrored about the surface centre when the reference bar was on the right."""
        if self.reference_side == "left":
            return self.x_mm
        return self.session.surface_width_mm - self.x_mm


def find_traces(data_path: Path) -> list[Path]:
    """A single trace file, or every ``*_trace.sqlite3`` directly inside a folder."""
    if data_path.is_file():
        return [data_path]
    return sorted(data_path.glob("*_trace.sqlite3"))


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _session(path: Path, row: tuple[str, str, str, str]) -> Session:
    session_id, participant_id, config_json, calibration_json = row
    config = json.loads(config_json)
    calibration = json.loads(calibration_json)
    if "inter_bar_gap_mm" not in config:
        raise ValueError(f"{path.name}: config_json has no inter_bar_gap_mm; bar positions cannot be derived")
    try:
        width = calibration.get("active_width_mm") or calibration["active_width_px"] / calibration["px_per_mm_x"]
        height = calibration.get("active_height_mm") or calibration["active_height_px"] / calibration["px_per_mm_y"]
    except KeyError as exc:
        raise ValueError(f"{path.name}: calibration_json lacks the surface size ({exc})") from exc
    return Session(path, session_id, participant_id, float(width), float(height), float(config["inter_bar_gap_mm"]))


def load_trace(path: Path) -> list[Attempt]:
    """Every attempt that has at least two samples, in recording order."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        attempt_columns = _columns(connection, "attempts")
        has_practice = {"is_practice", "practice_round"} <= attempt_columns
        has_contact = CONTACT_COLUMN in _columns(connection, "cursor_samples")
        sessions = {
            row[0]: _session(path, row)
            for row in connection.execute(
                "SELECT session_id, participant_id, config_json, calibration_json FROM sessions"
            )
        }
        practice_sql = "is_practice, practice_round" if has_practice else "0, 0"
        rows = connection.execute(
            f"""
            SELECT attempt_id, session_id, trial_index, attempt_index, {practice_sql}, is_catch,
                   level_pct, reference_height_mm, comparison_height_mm, bar_width_mm,
                   reference_side, outcome, response
            FROM attempts ORDER BY attempt_id
            """
        ).fetchall()
        contact_sql = f", {CONTACT_COLUMN}" if has_contact else ""
        attempts: list[Attempt] = []
        for row in rows:
            samples = connection.execute(
                f"SELECT t_us, x_mm, y_mm{contact_sql} FROM cursor_samples WHERE attempt_id = ? ORDER BY sequence",
                (row[0],),
            ).fetchall()
            if len(samples) < 2:
                continue
            data = np.asarray(samples, dtype=np.float64)
            attempts.append(
                Attempt(
                    session=sessions[row[1]],
                    attempt_id=int(row[0]),
                    trial_index=int(row[2]),
                    attempt_index=int(row[3]),
                    is_practice=bool(row[4]),
                    practice_round=int(row[5]),
                    is_catch=bool(row[6]),
                    level_pct=float(row[7]),
                    reference_height_mm=float(row[8]),
                    comparison_height_mm=float(row[9]),
                    bar_width_mm=float(row[10]),
                    reference_side=str(row[11]),
                    outcome=row[12],
                    response=row[13],
                    t_s=data[:, 0] / 1e6,
                    x_mm=data[:, 1],
                    y_mm=data[:, 2],
                    contact=data[:, 3].astype(bool) if has_contact else None,
                )
            )
        return attempts
    finally:
        connection.close()
