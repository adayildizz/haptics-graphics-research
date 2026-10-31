"""Command line: edge errors of the position render rule on a recorded or synthetic session; the top-edge cue sweep."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.exploration_maps.trace_io import Attempt, find_traces, load_trace
from experiment import render_logic as rl
from experiment.render_logic import Rect, Sample

from . import pulses, synthetic
from .core import Contact, Outcome, contact_in, replay

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
    "was read (0-10 ms for a 100 Hz frame) is invisible here.",
    "Switches are assumed to happen at their planned instant. Scheduler lateness, the VISA write and the "
    "generator's own delay are not included; the last two need an oscilloscope.",
)


def _bars(attempt: Attempt) -> tuple[Rect, Rect]:
    """The bars at their configured size, as the trial loop decides with them."""
    session = attempt.session
    reference_left = attempt.reference_side == "left"
    heights = (
        attempt.reference_height_mm if reference_left else attempt.comparison_height_mm,
        attempt.comparison_height_mm if reference_left else attempt.reference_height_mm,
    )
    return rl.nominal_bars(
        session.surface_width_mm, session.surface_height_mm, session.inter_bar_gap_mm, attempt.bar_width_mm, *heights
    )


@dataclass(frozen=True)
class Recorded:
    """One recorded attempt, ready to replay."""

    label: str
    attempt: Attempt
    samples: list[Sample]
    down: list[bool] | None  # None: the trace has no contact column
    nominal: tuple[Rect, Rect]


def recorded_trials(data_path: Path) -> Iterator[Recorded]:
    """Every recorded attempt under ``data_path``, with its contact and bars."""
    for path in find_traces(data_path):
        for attempt in load_trace(path):
            samples = [Sample(float(t), float(x), float(y)) for t, x, y in zip(attempt.t_s, attempt.x_mm, attempt.y_mm)]
            # The column holds the live FrameDecision.finger_down, i.e. ContactGate's answer, so it is used as is.
            # Gating it again would take a trial that starts lifted, after an earlier trial showed touch, as down.
            down = None if attempt.contact is None else attempt.contact.tolist()
            label = f"{attempt.session.session_id} {attempt.stem}"
            yield Recorded(label, attempt, samples, down, _bars(attempt))


def replay_trace(data_path: Path) -> tuple[list[tuple[str, dict[str, Outcome]]], Contact]:
    """Every recorded attempt under ``data_path`` through the position rule, and the finger's lifts in them."""
    results = []
    contact = Contact()
    for trial in recorded_trials(data_path):
        results.append((trial.label, replay(trial.samples, trial.nominal, down=trial.down)))
        contact += contact_in(trial.samples, trial.down, trial.nominal)
    return results, contact


def replay_synthetic(*, against_finger: bool) -> list[tuple[str, dict[str, Outcome]]]:
    """The synthetic session, judged against its recording or against the finger path that produced it."""
    return [
        (trial.label, replay(trial.recorded, trial.bars, truth=trial.truth if against_finger else None))
        for trial in synthetic.session()
    ]


def summarize(results: Sequence[tuple[str, dict[str, Outcome]]]) -> list[dict[str, object]]:
    """One row per mode and edge category; top-edge rows report the effect on felt height."""
    rows: list[dict[str, object]] = []
    for mode in dict.fromkeys(mode for _, by_mode in results for mode in by_mode):
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


def format_contact(contact: Contact) -> str:
    """Lines for a recording's title: how often and how long the finger was off the glass, and what that excluded."""
    if contact.unrecorded == contact.trials:
        return "  contact: not recorded (trace schema 1 or 2), so the finger is taken to be on the glass throughout"
    share = contact.lifted_s / contact.recorded_s if contact.recorded_s > 0 else 0.0
    lines = [
        f"  contact: {contact.lifts} lift(s), lifted {contact.lifted_s:.1f} s of {contact.recorded_s:.1f} s "
        f"({share:.0%}). Signal off while lifted; the rule starts afresh at each touchdown.",
        f"  {contact.touch_starts} visit(s) began with a touchdown inside a bar and {contact.lift_ends} ended with a "
        "lift there; those ends have no edge error.",
    ]
    if contact.unrecorded:
        lines.append(f"  {contact.unrecorded} trial(s) have no contact column: taken as on the glass throughout.")
    return "\n".join(lines)


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


def _block(attempt: Attempt) -> str:
    return "practice" if attempt.is_practice else "catch" if attempt.is_catch else "main"


def pulse_sweep(paths: Sequence[Path], settings: Sequence[pulses.Setting], out_dir: Path | None) -> int:
    """The top-edge cue over every recorded attempt, for each setting; with and without contact apart."""
    trials = [
        pulses.CueTrial(trial.label, _block(trial.attempt), trial.samples, trial.down, trial.nominal)
        for path in paths
        for trial in recorded_trials(path)
    ]
    if not trials:
        print(f"no recorded trials under {', '.join(str(path) for path in paths)}")
        return 2
    groups = (
        ("contact", "Recordings with contact (schema 3)", [t for t in trials if t.down is not None]),
        (
            "no_contact",
            "Recordings without a contact column (schemas 1 and 2): taken as on the glass throughout, so only the "
            "jump guard separates lifts",
            [t for t in trials if t.down is None],
        ),
    )
    for name, title, chosen in groups:
        if not chosen:
            continue
        swept = pulses.sweep(chosen, settings)
        rows = [pulses.summarize(setting, chosen, results) for setting, results in swept]
        sessions = len({trial.label.split()[0] for trial in chosen})
        print(pulses.format_sweep(f"{title}\n  {len(chosen)} trial(s) from {sessions} session(s)", rows))
        print()
        if out_dir is not None:
            pulses.write_csvs(out_dir, f"pulses_{name}", swept, rows)
    for caveat in pulses.CAVEATS:
        print(f"Note: {caveat}")
    return 0


def _mm_list(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.render_replay",
        description="Where the position rule would have switched the signal, against the true edge crossings.",
    )
    parser.add_argument(
        "data", type=Path, nargs="*", help="A *_trace.sqlite3 file or a folder of them; --pulses takes several."
    )
    parser.add_argument("--synthetic", action="store_true", help="Use the built-in synthetic session instead.")
    parser.add_argument("--out", type=Path, default=None, help="Folder for the CSV files (default: none written).")
    cue = parser.add_argument_group("top-edge cue", "--pulses sweeps the cue's parameters instead of the render replay")
    cue.add_argument("--pulses", action="store_true", help="Sweep the top-edge cue on the recorded traces.")
    cue.add_argument("--rearm", type=_mm_list, default=pulses.REARM_MM, help="re-arm distances d, mm, comma-separated")
    cue.add_argument("--corner", type=_mm_list, default=pulses.CORNER_MM, help="corner tolerances c, mm")
    cue.add_argument("--jump", type=_mm_list, default=pulses.JUMP_MM, help="jump guards J, mm")
    cue.add_argument("--min-interval-ms", type=float, default=pulses.MIN_INTERVAL_MS, help="between any two pulses")
    args = parser.parse_args(argv)
    if args.pulses:
        if args.synthetic or not args.data:
            parser.error("--pulses needs recorded traces")
        settings = pulses.grid(args.rearm, args.corner, args.jump, args.min_interval_ms)
        return pulse_sweep(args.data, settings, args.out)
    if not args.synthetic and not args.data:
        parser.error("give a trace file or folder, or --synthetic")
    if len(args.data) > 1:
        parser.error("give one trace file or folder; only --pulses takes several")
    data = args.data[0] if args.data else None

    reports: list[tuple[str, str, list[tuple[str, dict[str, Outcome]]]]] = []
    if args.synthetic:
        reports.append(("synthetic_vs_recording", "Synthetic session, judged as a recording would be (straight lines between its samples)", replay_synthetic(against_finger=False)))
        reports.append(("synthetic_vs_finger", "Synthetic session, judged against the finger path that produced it", replay_synthetic(against_finger=True)))
    else:
        results, contact = replay_trace(data)
        if not results:
            print(f"no recorded trials under {data}")
            return 2
        title = f"{data.name}: {len(results)} recorded trial(s)\n{format_contact(contact)}"
        reports.append(("recording", title, results))
    for name, title, results in reports:
        print(format_table(title, summarize(results)))
        print()
        if args.out is not None:
            write_csvs(args.out, name, results)
    for caveat in CAVEATS:
        print(f"Note: {caveat}")
    return 0
