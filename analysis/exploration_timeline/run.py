"""Command line: per-trial path + region-lane figures, a per-session overview and the region segments as CSV."""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import traceback
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from analysis.exploration_maps.trace_io import Attempt, find_traces, load_trace

from .plots import plot_attempt, plot_overview
from .regions import RegionParams, Segment, classify, segments

SEGMENT_FIELDS = [
    "participant_id", "session_id", "attempt_id", "block", "trial_index", "attempt_index",
    "level_pct", "reference_side", "outcome", "response", "correct",
    "segment", "region", "start_s", "end_s", "duration_ms",
]  # fmt: skip


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "unknown"


def _answer(attempt: Attempt) -> str:
    answer = attempt.outcome or "unfinished"
    if attempt.response is not None:
        answer += f" {attempt.response} {'✓' if attempt.correct else '✗'}"
    return answer


def _title(attempt: Attempt, params: RegionParams) -> str:
    session = attempt.session
    contact = "contact flag" if attempt.contact is not None else "no contact column: finger taken as down throughout"
    return "\n".join(
        (
            f"{session.participant_id} · {session.session_id} · {attempt.block} trial {attempt.trial_index} "
            f"attempt {attempt.attempt_index} (id {attempt.attempt_id})",
            f"level {attempt.level_pct:+.0%}{' (catch)' if attempt.is_catch else ''} · "
            f"reference {attempt.reference_side} {attempt.reference_height_mm:g} mm · "
            f"comparison {attempt.comparison_height_mm:g} mm · {_answer(attempt)}",
            f"edge bands {params.band_mm:g} mm each side of an edge, top band {params.overhang_mm:g} mm past "
            f"the corners · {contact}",
        )
    )


def _row_label(attempt: Attempt) -> str:
    verdict = "" if attempt.response is None else (" ✓" if attempt.correct else " ✗")
    return f"{attempt.block} t{attempt.trial_index} a{attempt.attempt_index} {attempt.level_pct:+.0%}{verdict}"


def _segment_rows(attempt: Attempt, found: Sequence[Segment]) -> list[dict[str, object]]:
    session = attempt.session
    correct = attempt.correct
    common = {
        "participant_id": session.participant_id,
        "session_id": session.session_id,
        "attempt_id": attempt.attempt_id,
        "block": attempt.block,
        "trial_index": attempt.trial_index,
        "attempt_index": attempt.attempt_index,
        "level_pct": f"{attempt.level_pct:.6f}",
        "reference_side": attempt.reference_side,
        "outcome": attempt.outcome or "",
        "response": attempt.response or "",
        "correct": "" if correct is None else int(correct),
    }
    return [
        {
            **common,
            "segment": index,
            "region": segment.region.key,
            "start_s": f"{segment.start_s:.4f}",
            "end_s": f"{segment.end_s:.4f}",
            "duration_ms": f"{segment.duration_s * 1000.0:.1f}",
        }
        for index, segment in enumerate(found)
    ]


def _process_session(
    attempts: Sequence[Attempt], session_dir: Path, params: RegionParams, arrow_every_mm: float, margin_mm: float
) -> int:
    """Figures, overview and segments CSV for one session; returns the number of failed trials."""
    failures = 0
    rows: list[tuple[str, list[Segment]]] = []
    csv_rows: list[dict[str, object]] = []
    for attempt in attempts:
        try:
            bars = attempt.bars()
            codes = classify(
                attempt.x_mm, attempt.y_mm, attempt.contact, bars, attempt.session.surface_width_mm, params
            )
            found = segments(attempt.t_s, codes)
            plot_attempt(
                attempt.t_s, attempt.x_mm, attempt.y_mm, attempt.contact, found, bars, params,
                _title(attempt, params), session_dir / f"{attempt.stem}.png",
                arrow_every_mm=arrow_every_mm, margin_mm=margin_mm,
            )  # fmt: skip
        except Exception:  # noqa: BLE001 -- one bad trial must not stop the rest
            print(f"FAILED {attempt.session.session_id} {attempt.stem}")
            traceback.print_exc()
            failures += 1
            continue
        rows.append((_row_label(attempt), found))
        csv_rows.extend(_segment_rows(attempt, found))
    if not rows:
        return failures

    with (session_dir / "segments.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SEGMENT_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    session = attempts[0].session
    plot_overview(
        rows,
        f"{session.participant_id} · {session.session_id} · {len(rows)} trial(s) in recording order · "
        f"edge bands {params.band_mm:g} mm, top band {params.overhang_mm:g} mm past the corners",
        session_dir / "overview.png",
    )
    print(f"{session.participant_id} {session.session_id}: {len(rows)} trial figure(s) + overview -> {session_dir}")
    return failures


def run(
    data_path: Path,
    participant: str,
    out_dir: Path,
    *,
    params: RegionParams = RegionParams(),
    arrow_every_mm: float = 20.0,
    margin_mm: float = 10.0,
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
    by_session: dict[tuple[Path, str], list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        by_session[(attempt.session.path, attempt.session.session_id)].append(attempt)
    for session_attempts in by_session.values():
        session = session_attempts[0].session
        session_dir = out_dir / _safe(session.participant_id) / _safe(session.session_id)
        failures += _process_session(session_attempts, session_dir, params, arrow_every_mm, margin_mm)
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.exploration_timeline",
        description="Time-coloured finger paths and region-over-time strips, per trial and per session.",
    )
    parser.add_argument("data", type=Path, help="A *_trace.sqlite3 file, or a folder containing them.")
    parser.add_argument("--participant", default="all", help="Participant ID, or 'all' (default).")
    parser.add_argument(
        "--out", type=Path, default=None, help="Output folder (default: <data folder>/exploration_timeline)."
    )
    parser.add_argument(
        "--band-mm", type=float, default=RegionParams.band_mm, help="Edge band depth on each side of an edge."
    )
    parser.add_argument(
        "--overhang-mm", type=float, default=RegionParams.overhang_mm, help="Top band reach past the bar corners."
    )
    parser.add_argument("--arrow-every-mm", type=float, default=20.0, help="Path length between direction arrows.")
    parser.add_argument("--margin-mm", type=float, default=10.0, help="Space shown around the bars.")
    parser.add_argument("--include-practice", action="store_true", help="Also draw practice trials.")
    args = parser.parse_args(argv)
    data_dir = args.data if args.data.is_dir() else args.data.parent
    return run(
        args.data,
        args.participant,
        args.out or data_dir / "exploration_timeline",
        params=RegionParams(args.band_mm, args.overhang_mm),
        arrow_every_mm=args.arrow_every_mm,
        margin_mm=args.margin_mm,
        include_practice=args.include_practice,
    )
