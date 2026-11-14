"""Check that the force sensor and the IR frame of a session line up, in time and in space.

    python -m analysis.force_sync experiment/data/P01_<time>_trace.sqlite3 [--out DIR]

For a session recorded with ``force_sensor: true``. The trace's ``trace_metadata`` names the force file (next to
the trace) and the calibration file (looked for as configured, then in the repository root, then next to the
trace). Both recordings use the experiment's ``time.perf_counter``: a force sample carries that time, and a
cursor sample or trace event is at ``attempts.trial_start_s + t_us / 1e6``.

The forces:

1. Zero: each gauge's mean over the quietest 0.5 s of each hands-off window (``force_measure.measure_zero``).
   Every sample is measured from the zero of the last window that began before it (the first window's zero
   for samples before that).
2. Fx, Fy, Fz (N) and Tx, Ty, Tz (N mm): the calibration matrix times the gauge volts minus the zero. Nothing
   else is corrected.
3. The press force is Fn = -Fz (a finger pressing down gives a negative Fz).

Two checks; every difference is IR minus force, so a positive number means the IR frame is later:

(a) Touch timing. A press starts when Fn rises above 0.03 N and ends when it falls below 0.015 N again (the
    gap between the two stops noise from chattering around one threshold). Each ``contact_down`` /
    ``contact_up`` event of the trace is paired with the nearest press start / end within 0.3 s; the report
    gives the median and the interquartile range of the differences.
(b) Finger position. While Fn > 0.1 N the torques say where the finger presses: x = Ty / Fn, y = -Tx / Fn, in
    mm in the sensor's own axes (this ignores the small torque of the sideways force). Each cursor sample with
    the finger down is compared with the force position ``lag`` earlier, for every lag from -150 to +150 ms in
    1 ms steps. At each lag a similarity transform (rotation, one scale, shift, and a mirror if that fits
    better) maps the force positions onto the screen's; the lag with the smallest residual is the IR frame's
    delay, and its residual RMS in mm says how well the two positions agree.

With ``--out`` it also writes ``<session>_force_sync.csv`` (every contact event and its paired force time)
and ``<session>_force_sync.png`` (the time differences, the residual over the lag, and both paths).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import numpy as np

from experiment.ati_calibration import AtiCalibration, load_ati_calibration
from experiment.force_daq import load_force
from experiment.force_measure import measure_zero
from experiment.force_session import REPO_ROOT, cal_file_path

PRESS_ON_N = 0.03  # a press starts when Fn rises above this ...
PRESS_OFF_N = 0.015  # ... and ends when it falls below this
PAIR_WITHIN_S = 0.3  # an IR contact event is paired with a press start / end at most this far away
POSITION_MIN_FN_N = 0.1  # below this the torque / force ratio is too noisy to give a position
LAGS_MS = range(-150, 151)  # the lags (IR minus force) scanned in check (b)
MIN_PAIRS = 20  # fewer position pairs than this at a lag: no fit there


# --- reading ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class IrRecording:
    """What the trace says about the finger, on ``time.perf_counter``."""

    t_s: np.ndarray  # every cursor sample, in time order
    x_mm: np.ndarray
    y_mm: np.ndarray
    contact: np.ndarray  # True: the frame reports a finger on the glass
    events: list[tuple[str, float]]  # ("contact_down" | "contact_up", time), in time order
    attempts: int  # attempts that have a trial start (schema 4)


@dataclass(frozen=True)
class Forces:
    """The force file as forces, on ``time.perf_counter``."""

    t_s: np.ndarray
    fn: np.ndarray  # press force, N
    x_mm: np.ndarray  # where the finger presses, in the sensor's axes; NaN while Fn <= POSITION_MIN_FN_N
    y_mm: np.ndarray
    noise_fz_n: float  # SD of Fz in the quietest 0.5 s of the first hands-off window used
    zero_blocks: list[str]  # the hands-off windows the zeros came from ("practice", "main")


def read_trace(path: Path) -> tuple[dict[str, str], IrRecording]:
    """The trace's metadata and its cursor samples and contact events, timed on ``time.perf_counter``."""
    if not path.is_file():  # sqlite3 would quietly create an empty file
        raise FileNotFoundError(f"no trace at {path}")
    connection = sqlite3.connect(path)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM trace_metadata"))
        columns = {row[1] for row in connection.execute("PRAGMA table_info(attempts)")}
        if "trial_start_s" not in columns:
            raise ValueError(f"{path.name} was recorded before trace schema 4: it has no trial clock")
        (attempts,) = connection.execute("SELECT COUNT(*) FROM attempts WHERE trial_start_s IS NOT NULL").fetchone()
        samples = connection.execute(
            """
            SELECT a.trial_start_s + s.t_us / 1e6, s.x_mm, s.y_mm, s.contact
            FROM cursor_samples s JOIN attempts a USING (attempt_id)
            WHERE a.trial_start_s IS NOT NULL ORDER BY 1
            """
        ).fetchall()
        events = connection.execute(
            """
            SELECT e.event_type, a.trial_start_s + e.t_us / 1e6
            FROM trace_events e JOIN attempts a USING (attempt_id)
            WHERE a.trial_start_s IS NOT NULL AND e.event_type IN ('contact_down', 'contact_up') ORDER BY 2
            """
        ).fetchall()
    finally:
        connection.close()
    table = np.array(samples, dtype=float).reshape(-1, 4)
    ir = IrRecording(table[:, 0], table[:, 1], table[:, 2], table[:, 3] > 0.5, [(k, t) for k, t in events], attempts)
    return metadata, ir


def find_cal_file(name: str, trace_path: Path) -> Path:
    """The calibration file the trace names: as configured, else by its name in the repository or the trace's folder."""
    bare = PureWindowsPath(name).name  # also strips a Windows folder when the trace comes from the lab PC
    for candidate in (cal_file_path(name), REPO_ROOT / bare, trace_path.parent / bare):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"calibration file {name!r} not found (also looked in {REPO_ROOT} and {trace_path.parent})")


def forces_from(path: Path, windows: list[dict], calibration: AtiCalibration) -> Forces:
    """The force file through the zeros of the hands-off windows and the calibration."""
    recording = load_force(path)
    t_s = np.asarray(recording.t_s, dtype=float)
    volts = np.asarray(recording.volts, dtype=float)
    rate_hz = float(recording.metadata["rate_hz"])
    zeros, used = [], []
    for window in windows:
        try:
            zeros.append(measure_zero(t_s, volts, window["start_s"], window["end_s"], rate_hz, calibration))
            used.append(window)
        except ValueError:  # no samples in it: the reader had stopped by then
            continue
    if not zeros:
        raise ValueError("no hands-off window of the trace has force samples to take the zero from")
    starts = np.array([w["start_s"] for w in used])
    which = np.clip(np.searchsorted(starts, t_s, side="right") - 1, 0, len(used) - 1)
    zero = np.array([z.volts for z in zeros])[which]  # each sample's zero
    fx, fy, fz, tx, ty, tz = ((volts - zero) @ np.array(calibration.matrix).T).T
    fn = -fz
    pressing = fn > POSITION_MIN_FN_N
    with np.errstate(divide="ignore", invalid="ignore"):
        x_mm = np.where(pressing, ty / fn, np.nan)
        y_mm = np.where(pressing, -tx / fn, np.nan)
    return Forces(t_s, fn, x_mm, y_mm, zeros[0].noise_sd["Fz"], [w["block"] for w in used])


# --- (a) touch timing ------------------------------------------------------------------------


def press_edges(t_s: np.ndarray, fn: np.ndarray, on_n: float = PRESS_ON_N, off_n: float = PRESS_OFF_N):
    """Times when a press starts (Fn rises above ``on_n``) and ends (Fn falls below ``off_n`` again)."""
    starts: list[float] = []
    ends: list[float] = []
    pressed = False
    for t, value in zip(t_s.tolist(), fn.tolist()):
        if not pressed and value > on_n:
            pressed = True
            starts.append(t)
        elif pressed and value < off_n:
            pressed = False
            ends.append(t)
    return np.array(starts), np.array(ends)


def nearest(times: np.ndarray, t: float, within_s: float = PAIR_WITHIN_S) -> float | None:
    """The time in ``times`` (sorted) closest to ``t``, if it is at most ``within_s`` away."""
    if len(times) == 0:
        return None
    k = int(np.searchsorted(times, t))
    best = min((times[j] for j in (k - 1, k) if 0 <= j < len(times)), key=lambda c: abs(c - t))
    return float(best) if abs(best - t) <= within_s else None


def pair_contacts(ir: IrRecording, forces: Forces) -> list[dict]:
    """Every IR contact event with its nearest press start / end; ``diff_ms`` is IR minus force."""
    starts, ends = press_edges(forces.t_s, forces.fn)
    rows = []
    for kind, t in ir.events:
        match = nearest(starts if kind == "contact_down" else ends, t)
        diff_ms = None if match is None else 1000.0 * (t - match)
        rows.append({"event": kind, "ir_s": t, "force_s": match, "diff_ms": diff_ms})
    return rows


# --- (b) finger position ---------------------------------------------------------------------


@dataclass(frozen=True)
class Similarity:
    """screen = scale * matrix @ sensor + shift; ``matrix`` is a rotation, or a mirror followed by one."""

    scale: float
    matrix: np.ndarray
    shift: np.ndarray
    rms_mm: float
    pairs: int

    @property
    def mirrored(self) -> bool:
        return bool(np.linalg.det(self.matrix) < 0)

    @property
    def rotation_deg(self) -> float:
        return math.degrees(math.atan2(self.matrix[1, 0], self.matrix[0, 0]))

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * points @ self.matrix.T + self.shift


def fit_similarity(sensor: np.ndarray, screen: np.ndarray) -> Similarity:
    """Least-squares ``screen ~ scale * matrix @ sensor + shift`` for two (n, 2) arrays of matching points.

    The orthogonal Procrustes solution: with both point sets centred, the SVD of their cross-covariance gives
    the best rotation or mirror (whichever fits better), and the scale follows from it.
    """
    sensor_mean, screen_mean = sensor.mean(axis=0), screen.mean(axis=0)
    a, b = sensor - sensor_mean, screen - screen_mean
    u, s, vt = np.linalg.svd(a.T @ b)
    matrix = vt.T @ u.T
    scale = float(s.sum() / (a * a).sum())
    shift = screen_mean - scale * matrix @ sensor_mean
    residual = screen - (scale * sensor @ matrix.T + shift)
    rms = float(np.sqrt((residual * residual).sum(axis=1).mean()))
    return Similarity(scale, matrix, shift, rms, len(sensor))


def position_pairs(ir: IrRecording, forces: Forces, lag_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(sensor, screen) positions: each finger-down cursor sample and the force position ``lag_s`` before it.

    The force position is interpolated between the two force samples around that time; it is unknown (and the
    pair dropped) where either has Fn <= POSITION_MIN_FN_N, or outside the force file.
    """
    t = ir.t_s[ir.contact] - lag_s
    fx = np.interp(t, forces.t_s, forces.x_mm, left=np.nan, right=np.nan)
    fy = np.interp(t, forces.t_s, forces.y_mm, left=np.nan, right=np.nan)
    known = np.isfinite(fx) & np.isfinite(fy)
    sensor = np.column_stack([fx[known], fy[known]])
    screen = np.column_stack([ir.x_mm[ir.contact][known], ir.y_mm[ir.contact][known]])
    return sensor, screen


def scan_lags(ir: IrRecording, forces: Forces, lags_ms: Sequence[int] = LAGS_MS) -> list[tuple[int, Similarity | None]]:
    """The fit at every lag (IR minus force, ms); None where too few pairs are left."""
    fits = []
    for lag_ms in lags_ms:
        sensor, screen = position_pairs(ir, forces, lag_ms / 1000.0)
        fits.append((lag_ms, fit_similarity(sensor, screen) if len(sensor) >= MIN_PAIRS else None))
    return fits


# --- report ----------------------------------------------------------------------------------


def _spread(values: list[float]) -> str:
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return f"median {median:+.1f} ms, IQR {q25:+.1f} .. {q75:+.1f} ms"


def report(trace_path: Path, out_dir: Path | None = None) -> dict:
    """Runs both checks, prints the report, writes the CSV and figure with ``out_dir``; returns the numbers."""
    metadata, ir = read_trace(trace_path)
    if "force_file" not in metadata:
        raise ValueError(f"{trace_path.name} was recorded without the force sensor (no force_file in its metadata)")
    force_path = trace_path.parent / metadata["force_file"]
    if not force_path.is_file():
        raise FileNotFoundError(f"the trace names the force file {force_path.name}, but it is not next to the trace")
    cal_path = find_cal_file(metadata["force_cal_file"], trace_path)
    calibration = load_ati_calibration(cal_path)
    windows = json.loads(metadata.get("force_hands_off", "[]"))
    forces = forces_from(force_path, windows, calibration)
    session = trace_path.name.removesuffix("_trace.sqlite3")

    span_s = forces.t_s[-1] - forces.t_s[0] if len(forces.t_s) > 1 else 0.0
    print(f"Force sync check: {session}")
    print(f"  force file {force_path.name}: {len(forces.t_s)} samples over {span_s:.1f} s")
    print(f"  calibration {cal_path.name} ({calibration.serial}); Fn = -Fz")
    skipped = len(windows) - len(forces.zero_blocks)
    print(
        f"  zero from the hands-off window(s) before {', '.join(forces.zero_blocks)} (quietest 0.5 s of each"
        + (f"; {skipped} window(s) without force samples skipped" if skipped else "")
        + f"); Fz noise there {1000 * forces.noise_fz_n:.1f} mN SD"
    )
    print(
        f"  trace: {ir.attempts} attempts with a trial clock, {len(ir.t_s)} cursor samples, "
        f"{len(ir.events)} contact events"
    )
    if "force_reader_problem" in metadata:
        error = json.loads(metadata["force_reader_problem"])["error"].strip().splitlines()[-1]
        print(f"  NOTE: the force reader stopped during the session ({error}); its data end there")

    print("\n(a) touch timing, IR minus force (positive: the IR frame reports later)")
    pairs = pair_contacts(ir, forces)
    timing = {}
    edges = {
        "contact_down": f"press start (Fn > {PRESS_ON_N:g} N)",
        "contact_up": f"press end (Fn < {PRESS_OFF_N:g} N)",
    }
    for kind, edge in edges.items():
        diffs = [p["diff_ms"] for p in pairs if p["event"] == kind and p["diff_ms"] is not None]
        total = sum(p["event"] == kind for p in pairs)
        timing[kind] = float(np.median(diffs)) if diffs else None
        line = _spread(diffs) if diffs else "no pairs"
        print(f"  {kind:<12} vs {edge}: {len(diffs)} of {total} paired within {1000 * PAIR_WITHIN_S:.0f} ms; {line}")
    if not ir.events:
        print("  (no contact events: the frame did not report touch in this session)")

    print("\n(b) finger position, torques against the IR frame")
    fits = scan_lags(ir, forces)
    fitted = [(lag, fit) for lag, fit in fits if fit is not None]
    best_lag, best = min(fitted, key=lambda item: item[1].rms_mm) if fitted else (None, None)
    if best is None:
        print(f"  too few finger-down samples with Fn > {POSITION_MIN_FN_N:g} N to compare positions")
    else:
        at_zero = dict(fits).get(0)
        print(
            f"  best lag {best_lag:+d} ms (IR later than force when positive); residual RMS {best.rms_mm:.2f} mm "
            f"over {best.pairs} samples" + (f" ({at_zero.rms_mm:.2f} mm at lag 0)" if at_zero is not None else "")
        )
        print(
            f"  sensor -> screen: rotation {best.rotation_deg:+.1f} deg, "
            f"{'mirrored' if best.mirrored else 'not mirrored'}, scale {best.scale:.3f}, "
            f"sensor axis at ({best.shift[0]:.1f}, {best.shift[1]:.1f}) mm on the screen"
        )
        if best_lag in (LAGS_MS[0], LAGS_MS[-1]):
            print("  WARNING: the best lag is at the edge of the scan; the true delay may lie outside it")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{session}_force_sync.csv"
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["event", "ir_s", "force_s", "diff_ms"])
            writer.writeheader()
            writer.writerows(pairs)
        png_path = out_dir / f"{session}_force_sync.png"
        _plot(png_path, session, pairs, fits, best_lag, best, ir, forces)
        print(f"\nSaved {csv_path}\n  and {png_path}")
    return {"contact_ms": timing, "lag_ms": best_lag, "fit": best}


def _plot(path, session, pairs, fits, best_lag, best, ir, forces) -> None:
    """One figure, three panels: the touch-timing differences, the residual over the lag, both paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (ax_t, ax_lag, ax_xy) = plt.subplots(1, 3, figsize=(15, 4.8))
    for kind, color in (("contact_down", "tab:blue"), ("contact_up", "tab:orange")):
        diffs = [p["diff_ms"] for p in pairs if p["event"] == kind and p["diff_ms"] is not None]
        if diffs:
            ax_t.hist(diffs, bins=30, alpha=0.6, color=color, label=f"{kind} ({len(diffs)})")
    ax_t.set(title="(a) touch timing", xlabel="IR minus force (ms)", ylabel="events")
    if pairs:
        ax_t.legend()

    lags = [lag for lag, fit in fits if fit is not None]
    ax_lag.plot(lags, [fit.rms_mm for _, fit in fits if fit is not None], color="tab:green")
    if best is not None:
        ax_lag.axvline(best_lag, color="grey", linestyle="--")
    ax_lag.set(title="(b) position residual over the lag", xlabel="lag, IR minus force (ms)")
    ax_lag.set_ylabel("residual RMS (mm)")

    if best is not None:
        sensor, screen = position_pairs(ir, forces, best_lag / 1000.0)
        step = max(1, len(screen) // 5000)
        mapped = best.apply(sensor[::step])
        ax_xy.plot(screen[::step, 0], screen[::step, 1], ".", ms=2, color="tab:blue", label="IR frame")
        ax_xy.plot(mapped[:, 0], mapped[:, 1], ".", ms=2, color="tab:red", alpha=0.6, label="force (mapped)")
        ax_xy.legend()
    title = "(b) positions" + (f" at lag {best_lag:+d} ms" if best is not None else ": too few to compare")
    ax_xy.set(title=title, xlabel="x on the screen (mm)", ylabel="y on the screen (mm, down)", aspect="equal")
    ax_xy.invert_yaxis()
    figure.suptitle(session)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.force_sync", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("trace", type=Path, help="<session>_trace.sqlite3 of a session recorded with the force sensor")
    parser.add_argument("--out", type=Path, default=None, help="Also write a CSV and a figure into this folder.")
    args = parser.parse_args(argv)
    try:
        report(args.trace, args.out)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
