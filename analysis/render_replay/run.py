"""Command line: edge errors of the three render modes on a recorded or synthetic session."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from analysis.exploration_maps.trace_io import Attempt, find_traces, load_trace
from experiment import render_logic as rl
from experiment.render_logic import Rect, Sample

from . import synthetic
from .core import Outcome, Params, replay

CATEGORIES = (
    "side edge, entry",
    "side edge, exit",
    "top edge, downward entry",
    "top edge, upward exit",
    "bottom edge, entry",
    "bottom edge, exit",
)
CAVEATS = (
    "The recording's signal_on column is sampled in the same frame as the position, so it cannot show a delay; "
    "these figures come from replaying the decision code on the recorded positions.",
    "The true path is taken to be straight lines between recorded samples. How old a sample already was when it "
    "was read (0-10 ms for a 100 Hz frame) is invisible here and affects every mode alike.",
    "Switches are assumed to happen at their planned instant. Scheduler lateness, the VISA write and the "
    "generator's own delay are not included; the last two need an oscilloscope.",
)


def _calibrations(path: Path) -> dict[str, dict]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT session_id, calibration_json FROM sessions").fetchall()
    finally:
        connection.close()
    return {session_id: json.loads(calibration) for session_id, calibration in rows}


def _bars(attempt: Attempt, calibration: dict) -> tuple[tuple[Rect, Rect], tuple[Rect, Rect] | None, float]:
    """Nominal bars, the pixel-rounded ones when the calibration allows, and the size of one pixel in mm."""
    session = attempt.session
    reference_left = attempt.reference_side == "left"
    heights = (
        attempt.reference_height_mm if reference_left else attempt.comparison_height_mm,
        attempt.comparison_height_mm if reference_left else attempt.reference_height_mm,
    )
    nominal = rl.nominal_bars(
        session.surface_width_mm, session.surface_height_mm, session.inter_bar_gap_mm, attempt.bar_width_mm, *heights
    )
    keys = ("active_left_px", "active_top_px", "active_width_px", "active_height_px", "px_per_mm_x", "px_per_mm_y")
    if not all(key in calibration for key in keys):
        return nominal, None, 0.2
    pixel = rl.pixel_bars(*(calibration[key] for key in keys), session.inter_bar_gap_mm, attempt.bar_width_mm, *heights)
    return nominal, pixel, max(1.0 / calibration["px_per_mm_x"], 1.0 / calibration["px_per_mm_y"])


def replay_trace(data_path: Path, params: Params) -> list[tuple[str, dict[str, Outcome]]]:
    """Every recorded attempt under ``data_path`` through the three modes."""
    results = []
    for path in find_traces(data_path):
        calibrations = _calibrations(path)
        for attempt in load_trace(path):
            nominal, pixel, quantum = _bars(attempt, calibrations[attempt.session.session_id])
            samples = [Sample(float(t), float(x), float(y)) for t, x, y in zip(attempt.t_s, attempt.x_mm, attempt.y_mm)]
            trial_params = Params(**{**params.__dict__, "position_quantum_mm": quantum})
            label = f"{attempt.session.session_id} {attempt.stem}"
            results.append((label, replay(samples, nominal, pixel, attempt.bar_width_mm, trial_params)))
    return results


def replay_synthetic(params: Params, *, against_finger: bool) -> list[tuple[str, dict[str, Outcome]]]:
    """The synthetic session, judged against its recording or against the finger path that produced it."""
    quantum = max(synthetic.PIXEL_MM)
    trial_params = Params(**{**params.__dict__, "position_quantum_mm": quantum})
    return [
        (
            trial.label,
            replay(
                trial.recorded, trial.bars, None, synthetic.WIDTH_MM, trial_params,
                truth=trial.truth if against_finger else None,
            ),
        )
        for trial in synthetic.session()
    ]  # fmt: skip


def summarize(results: Sequence[tuple[str, dict[str, Outcome]]]) -> list[dict[str, object]]:
    """One row per mode and edge category; top-edge rows report the effect on felt height."""
    rows: list[dict[str, object]] = []
    for mode in rl.RENDER_MODES:
        outcomes = [by_mode[mode] for _, by_mode in results]
        errors = [error for outcome in outcomes for error in outcome.errors]
        for category in CATEGORIES:
            chosen = [error for error in errors if error.category == category]
            if not chosen:
                continue
            values = np.array([e.late_mm if e.height_effect_mm is None else e.height_effect_mm for e in chosen])
            rows.append(
                {
                    "mode": mode,
                    "edge": category,
                    "n": len(chosen),
                    "median_mm": round(float(np.median(values)), 3),
                    "p5_mm": round(float(np.percentile(values, 5)), 3),
                    "p95_mm": round(float(np.percentile(values, 95)), 3),
                    "max_abs_mm": round(float(np.abs(values).max()), 3),
                    "never_switched_off": sum(error.bridged for error in chosen),
                }
            )
        rows.append(
            {
                "mode": mode,
                "edge": "(visits with no signal at all / signal on away from any bar)",
                "n": sum(outcome.missed for outcome in outcomes),
                "median_mm": "",
                "p5_mm": "",
                "p95_mm": "",
                "max_abs_mm": "",
                "never_switched_off": f"{sum(outcome.spurious_ms for outcome in outcomes):.0f} ms",
            }
        )
    return rows


def format_table(title: str, rows: Sequence[dict[str, object]]) -> str:
    lines = [
        title,
        "  mm: side and bottom edges + = switched after the finger crossed; top edge + = bar feels taller, - = shorter",
        f"  {'mode':13s} {'edge':27s} {'n':>4s} {'median':>8s} {'5%':>8s} {'95%':>8s} {'max |.|':>8s}  never off",
    ]
    for row in rows:
        if row["median_mm"] == "":
            lines.append(f"  {row['mode']:13s} visits missed: {row['n']}, signal on away from any bar: {row['never_switched_off']}")
            continue
        lines.append(
            f"  {row['mode']:13s} {row['edge']:27s} {row['n']:4d} {row['median_mm']:+8.2f} {row['p5_mm']:+8.2f} "
            f"{row['p95_mm']:+8.2f} {row['max_abs_mm']:8.2f}  {row['never_switched_off'] or ''}"
        )
    return "\n".join(lines)


def write_csvs(out_dir: Path, name: str, results: Sequence[tuple[str, dict[str, Outcome]]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}_crossings.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["trial", "mode", "bar", "edge", "direction", "t_true_s", "late_mm", "late_ms", "height_effect_mm", "never_switched_off"]
        )
        for label, by_mode in results:
            for outcome in by_mode.values():
                for e in outcome.errors:
                    effect = "" if e.height_effect_mm is None else f"{e.height_effect_mm:.4f}"
                    direction = "entry" if e.entering else "exit"
                    writer.writerow(
                        [label, e.mode, e.bar, e.edge, direction, f"{e.t_true_s:.6f}", f"{e.late_mm:.4f}", f"{e.late_ms:.3f}", effect, int(e.bridged)]
                    )
    rows = summarize(results)
    with (out_dir / f"{name}_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.render_replay",
        description="Where each render mode would have switched the signal, against the true edge crossings.",
    )
    parser.add_argument("data", type=Path, nargs="?", help="A *_trace.sqlite3 file or a folder of them.")
    parser.add_argument("--synthetic", action="store_true", help="Use the built-in synthetic session instead.")
    parser.add_argument("--out", type=Path, default=None, help="Folder for the CSV files (default: none written).")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--horizon-factor", type=float, default=1.5)
    parser.add_argument("--horizon-max-ms", type=float, default=25.0)
    parser.add_argument("--min-speed", type=float, default=5.0)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    args = parser.parse_args(argv)
    if not args.synthetic and args.data is None:
        parser.error("give a trace file or folder, or --synthetic")
    params = Params(args.window, args.horizon_factor, args.horizon_max_ms, args.min_speed, args.latency_ms)

    reports: list[tuple[str, str, list[tuple[str, dict[str, Outcome]]]]] = []
    if args.synthetic:
        reports.append(("synthetic_vs_recording", "Synthetic session, judged as a recording would be (straight lines between its samples)", replay_synthetic(params, against_finger=False)))
        reports.append(("synthetic_vs_finger", "Synthetic session, judged against the finger path that produced it", replay_synthetic(params, against_finger=True)))
    else:
        results = replay_trace(args.data, params)
        if not results:
            print(f"no recorded trials under {args.data}")
            return 2
        reports.append(("recording", f"{args.data.name}: {len(results)} recorded trial(s)", results))
    for name, title, results in reports:
        print(format_table(title, summarize(results)))
        print()
        if args.out is not None:
            write_csvs(args.out, name, results)
    for caveat in CAVEATS:
        print(f"Note: {caveat}")
    return 0
