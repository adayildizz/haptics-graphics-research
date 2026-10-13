"""Command line: per-trial and per-participant exploration maps from trace databases."""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import traceback
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .contact import ContactParams, ContactResult, classify
from .grid import Grid, TrialMaps, accumulate, aggregate, make_grid
from .plots import plot_aggregate, plot_trial
from .trace_io import Attempt, find_traces, load_trace

POOLED_OUTCOMES = ("answered", "timeout")
SUMMARY_FIELDS = [
    "participant_id", "session_id", "attempt_id", "block", "trial_index", "attempt_index",
    "level_pct", "reference_side", "reference_height_mm", "comparison_height_mm",
    "outcome", "response", "correct", "in_aggregate", "contact_method", "n_samples",
    "total_ms", "included_ms", "excluded_ms", "start_freeze_ms", "lift_ms", "end_freeze_ms",
    "end_freeze_excluded", "no_contact_ms", "jump_segments", "jump_ms",
    "outside_grid_ms", "dwell_sum_ms", "visits_sum",
]  # fmt: skip


@dataclass(frozen=True)
class TrialOutput:
    attempt: Attempt
    contact: ContactResult
    screen: TrialMaps
    reference_frame: TrialMaps

    @property
    def pooled(self) -> bool:
        return not self.attempt.is_practice and self.attempt.outcome in POOLED_OUTCOMES


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "unknown"


def _group_key(attempt: Attempt) -> tuple[str, float, float, float, float, float]:
    """Trials can only share a grid and an aggregate when the participant and bar layout match."""
    session = attempt.session
    return (
        session.participant_id,
        session.surface_width_mm,
        session.surface_height_mm,
        session.inter_bar_gap_mm,
        attempt.bar_width_mm,
        attempt.reference_height_mm,
    )


def _write_grid_csv(path: Path, grid: Grid, columns: dict[str, NDArray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = grid.shape
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ix", "iy", "x0_mm", "x1_mm", "y0_mm", "y1_mm", *columns])
        for iy in range(rows):
            for ix in range(cols):
                edges = (grid.x_edges[ix], grid.x_edges[ix + 1], grid.y_edges[iy], grid.y_edges[iy + 1])
                values = [f"{float(column[iy, ix]):.6g}" for column in columns.values()]
                writer.writerow([ix, iy, *(f"{edge:.3f}" for edge in edges), *values])


def _trial_title(attempt: Attempt, contact: ContactResult, maps: TrialMaps, params: ContactParams) -> str:
    session = attempt.session
    answer = attempt.outcome or "unfinished"
    if attempt.response is not None:
        answer += f" {attempt.response} {'✓' if attempt.correct else '✗'}"
    return "\n".join(
        (
            f"{session.participant_id} · {session.session_id} · {attempt.block} trial {attempt.trial_index} "
            f"attempt {attempt.attempt_index} (id {attempt.attempt_id})",
            f"level {attempt.level_pct:+.0%}{' (catch)' if attempt.is_catch else ''} · "
            f"reference {attempt.reference_side} {attempt.reference_height_mm:g} mm · "
            f"comparison {attempt.comparison_height_mm:g} mm · {answer}",
            f"{contact.describe(params)} · outside grid {maps.outside_s * 1000.0:.0f} ms",
        )
    )


def _summary_row(output: TrialOutput) -> dict[str, object]:
    attempt, contact, maps = output.attempt, output.contact, output.screen
    correct = attempt.correct
    return {
        "participant_id": attempt.session.participant_id,
        "session_id": attempt.session.session_id,
        "attempt_id": attempt.attempt_id,
        "block": attempt.block,
        "trial_index": attempt.trial_index,
        "attempt_index": attempt.attempt_index,
        "level_pct": f"{attempt.level_pct:.6f}",
        "reference_side": attempt.reference_side,
        "reference_height_mm": f"{attempt.reference_height_mm:.4f}",
        "comparison_height_mm": f"{attempt.comparison_height_mm:.4f}",
        "outcome": attempt.outcome or "",
        "response": attempt.response or "",
        "correct": "" if correct is None else int(correct),
        "in_aggregate": int(output.pooled),
        "contact_method": contact.method,
        "n_samples": len(attempt.t_s),
        "total_ms": f"{contact.total_ms:.1f}",
        "included_ms": f"{contact.included_ms:.1f}",
        "excluded_ms": f"{contact.excluded_ms:.1f}",
        "start_freeze_ms": f"{contact.start_freeze_ms:.1f}",
        "lift_ms": f"{contact.lift_ms:.1f}",
        "end_freeze_ms": f"{contact.end_freeze_ms:.1f}",
        "end_freeze_excluded": int(contact.end_freeze_excluded),
        "no_contact_ms": f"{contact.no_contact_ms:.1f}",
        "jump_segments": contact.jump_segments,
        "jump_ms": f"{contact.jump_ms:.1f}",
        "outside_grid_ms": f"{maps.outside_s * 1000.0:.1f}",
        "dwell_sum_ms": f"{maps.dwell_s.sum() * 1000.0:.1f}",
        "visits_sum": int(maps.visits.sum()),
    }


def _process_attempt(attempt: Attempt, grid: Grid, params: ContactParams, session_dir: Path) -> TrialOutput:
    contact = classify(attempt.t_s, attempt.x_mm, attempt.y_mm, attempt.contact, params)
    screen = accumulate(attempt.t_s, attempt.x_mm, attempt.y_mm, contact.include, contact.resting, grid)
    mirrored = screen
    if attempt.reference_side != "left":
        mirrored = accumulate(
            attempt.t_s, attempt.x_reference_left(), attempt.y_mm, contact.include, contact.resting, grid
        )
    accounted_ms = (screen.dwell_s.sum() + screen.outside_s) * 1000.0
    if abs(accounted_ms - contact.included_ms) > 1e-3:
        raise AssertionError(f"dwell {accounted_ms:.3f} ms does not add up to in-contact {contact.included_ms:.3f} ms")
    plot_trial(
        screen.visits,
        screen.dwell_s,
        grid,
        attempt.bars(),
        _trial_title(attempt, contact, screen, params),
        session_dir / f"{attempt.stem}.png",
    )
    _write_grid_csv(
        session_dir / f"{attempt.stem}.csv", grid, {"visits": screen.visits, "dwell_ms": screen.dwell_s * 1000.0}
    )
    return TrialOutput(attempt, contact, screen, mirrored)


def _total_contact(results: Sequence[ContactResult]) -> ContactResult:
    """Field-wise sum, for the aggregate figure's title."""
    empty = np.zeros(0, dtype=bool)
    return replace(
        results[0],
        include=empty,
        resting=empty,
        total_ms=sum(r.total_ms for r in results),
        included_ms=sum(r.included_ms for r in results),
        start_freeze_ms=sum(r.start_freeze_ms for r in results),
        lift_ms=sum(r.lift_ms for r in results),
        end_freeze_ms=sum(r.end_freeze_ms for r in results),
        end_freeze_excluded=any(r.end_freeze_excluded for r in results),
        no_contact_ms=sum(r.no_contact_ms for r in results),
        jump_segments=sum(r.jump_segments for r in results),
        jump_ms=sum(r.jump_ms for r in results),
    )


def _write_aggregate(
    outputs: Sequence[TrialOutput], grid: Grid, normalize: bool, params: ContactParams, participant_dir: Path
) -> Path | None:
    pooled = [output for output in outputs if output.pooled]
    if not pooled:
        return None
    visits, n_trials = aggregate([output.reference_frame.visits for output in pooled], normalize=normalize)
    dwell, _ = aggregate([output.reference_frame.dwell_s for output in pooled], normalize=normalize)
    first = pooled[0].attempt
    sessions = sorted({output.attempt.session.session_id for output in pooled})
    heights = sorted({output.attempt.comparison_height_mm for output in pooled})
    if len(sessions) == 1:
        stem = participant_dir / _safe(sessions[0]) / "aggregate"
    else:
        tag = f"ref{first.reference_height_mm:g}mm_w{first.bar_width_mm:g}mm"
        stem = participant_dir / f"aggregate_{len(sessions)}sessions_{tag}"
    if normalize:
        norm = "per-trial share, then mean over trials"
        labels = ("share of a trial's visits (%)", "share of a trial's dwell time (%)")
        columns = {"visits_share_pct": visits, "dwell_share_pct": dwell}
    else:
        norm = "none (sum over trials)"
        labels = ("visits (sum over trials)", "dwell time (s, sum over trials)")
        columns = {"visits_sum": visits, "dwell_sum_s": dwell}
    methods = {output.contact.method for output in pooled}
    total = _total_contact([output.contact for output in pooled])
    contact_line = total.describe(params) if len(methods) == 1 else (
        f"contact: flag in some sessions, heuristic in others · excluded {total.excluded_ms:.0f} ms"
    )
    title = "\n".join(
        (
            f"{first.session.participant_id} · {n_trials} of {len(pooled)} main-block trials with data · "
            f"{len(sessions)} session(s) · reference/comparison frame (reference bar on the left)",
            f"normalization: {norm} · {grid.cell_mm:g} mm cells · reference {first.reference_height_mm:g} mm · "
            f"comparison {heights[0]:g}–{heights[-1]:g} mm",
            contact_line,
        )
    )
    plot_aggregate(
        visits, dwell, grid, first.bars(reference_left=True), heights, labels, title, stem.with_suffix(".png")
    )
    _write_grid_csv(stem.with_suffix(".csv"), grid, columns)
    return stem.with_suffix(".png")


def _process_group(
    group: Sequence[Attempt],
    out_dir: Path,
    cell_mm: float,
    margin_mm: float,
    normalize: bool,
    params: ContactParams,
) -> int:
    """Per-trial maps, the per-session summary CSV and the aggregate; returns the number of failed trials."""
    first = group[0]
    bars = first.bars(reference_left=True)
    tallest = max(max(a.reference_height_mm, a.comparison_height_mm) for a in group)
    grid = make_grid(bars.left_x0, bars.right_x1, bars.baseline_y, tallest, cell_mm, margin_mm)
    participant_dir = out_dir / _safe(first.session.participant_id)
    outputs: list[TrialOutput] = []
    failures = 0
    for attempt in group:
        session_dir = participant_dir / _safe(attempt.session.session_id)
        try:
            outputs.append(_process_attempt(attempt, grid, params, session_dir))
        except Exception:  # noqa: BLE001 -- one bad trial must not stop the rest
            print(f"FAILED {attempt.session.session_id} {attempt.stem}")
            traceback.print_exc()
            failures += 1

    by_session: dict[str, list[TrialOutput]] = defaultdict(list)
    for output in outputs:
        by_session[output.attempt.session.session_id].append(output)
    for session_id, session_outputs in by_session.items():
        summary_path = participant_dir / _safe(session_id) / "trials.csv"
        with summary_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(_summary_row(output) for output in session_outputs)

    aggregate_path = _write_aggregate(outputs, grid, normalize, params, participant_dir)
    jumps = sum(output.contact.jump_segments for output in outputs)
    excluded_s = sum(output.contact.excluded_ms for output in outputs) / 1000.0
    print(
        f"{first.session.participant_id}: {len(outputs)} trial map(s) in {len(by_session)} session(s), "
        f"{jumps} jump segment(s) excluded, {excluded_s:.1f} s excluded in total -> {participant_dir}"
    )
    print(f"  aggregate: {aggregate_path if aggregate_path else 'none (no finished main-block trials)'}")
    return failures


def run(
    data_path: Path,
    participant: str,
    out_dir: Path,
    *,
    cell_mm: float = 2.0,
    margin_mm: float = 10.0,
    normalize: bool = True,
    params: ContactParams = ContactParams(),
    include_practice: bool = False,
) -> int:
    """Returns 0 on success, 1 if anything failed, 2 if no trial matched."""
    failures = 0
    attempts: list[Attempt] = []
    for path in find_traces(data_path):
        try:
            attempts.extend(load_trace(path))
        except (sqlite3.Error, ValueError, KeyError) as exc:
            print(f"FAILED to read {path.name}: {exc}")
            failures += 1
    attempts = [
        attempt
        for attempt in attempts
        if (participant == "all" or attempt.session.participant_id == participant)
        and (include_practice or not attempt.is_practice)
    ]
    if not attempts:
        print(f"no recorded trials for participant {participant!r} under {data_path}")
        return 1 if failures else 2
    groups: dict[tuple[str, float, float, float, float, float], list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        groups[_group_key(attempt)].append(attempt)
    for group in groups.values():
        failures += _process_group(group, out_dir, cell_mm, margin_mm, normalize, params)
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.exploration_maps",
        description="Visit-count and dwell-time heat maps of finger exploration, per trial and per participant.",
    )
    parser.add_argument("data", type=Path, help="A *_trace.sqlite3 file, or a folder containing them.")
    parser.add_argument("--participant", default="all", help="Participant ID, or 'all' (default).")
    parser.add_argument("--cell-mm", type=float, default=2.0, help="Grid cell size in mm (default 2).")
    parser.add_argument("--margin-mm", type=float, default=10.0, help="Margin around the bars in mm (default 10).")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default: <data folder>/exploration_maps).")
    parser.add_argument(
        "--normalize",
        choices=("per-trial", "none"),
        default="per-trial",
        help="Aggregate as the mean of per-trial shares (default) or as plain sums.",
    )
    parser.add_argument("--freeze-mm", type=float, default=0.5, help="Finger counts as frozen within this radius.")
    parser.add_argument("--freeze-ms", type=float, default=200.0, help="Minimum duration of a freeze.")
    parser.add_argument("--jump-mm", type=float, default=10.0, help="Single-step distance treated as a jump.")
    parser.add_argument("--keep-end-freeze", action="store_true", help="Count the freeze before the response as dwell.")
    parser.add_argument("--include-practice", action="store_true", help="Also write per-trial maps for practice.")
    args = parser.parse_args(argv)
    data_dir = args.data if args.data.is_dir() else args.data.parent
    return run(
        args.data,
        args.participant,
        args.out or data_dir / "exploration_maps",
        cell_mm=args.cell_mm,
        margin_mm=args.margin_mm,
        normalize=args.normalize == "per-trial",
        params=ContactParams(args.freeze_mm, args.freeze_ms, args.jump_mm, args.keep_end_freeze),
        include_practice=args.include_practice,
    )
