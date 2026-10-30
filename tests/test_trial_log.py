"""The trial CSV now records every trial the participant saw, tagged by outcome."""

from __future__ import annotations

import csv

from analysis.fit_psychometric import load_session_csvs
from experiment.data_logger import OUTCOMES, TRIAL_FIELDS, append_trial, load_trials
from experiment.trace_store import AttemptOutcome


def _row(**overrides):
    row = {
        "session_id": "S1",
        "participant_id": "01",
        "timestamp": "2026-08-19T14:00:00",
        "trial_index": 1,
        "mode": "constant_stimuli",
        "reference_side": "left",
        "outcome": "answered",
        "response": "right",
        "correct": 1,
        "level_pct": "0.300000",
        "comparison_height_mm": "13.0000",
        "reference_height_mm": "10.0000",
        "bar_width_mm": "10.0000",
        "is_catch": 0,
        "is_practice": 0,
        "response_time_s": "1.2500",
        "passes": [],
    }
    row.update(overrides)
    return row


def test_outcome_is_part_of_the_schema():
    assert "outcome" in TRIAL_FIELDS


def test_csv_outcomes_cover_the_trace_database_vocabulary():
    """The two stores must describe the same events, not overlapping subsets."""
    trace_outcomes = set(AttemptOutcome.__args__)

    assert trace_outcomes <= set(OUTCOMES)
    # The one CSV-only value is a refinement of timeout, not a new event kind.
    assert set(OUTCOMES) - trace_outcomes == {"exhausted"}


def test_unanswered_rows_round_trip_with_blank_response(tmp_path):
    path = tmp_path / "trials.csv"
    append_trial(_row(), path)
    append_trial(_row(trial_index=2, outcome="timeout", response="", correct="", response_time_s="30.0000"), path)
    append_trial(_row(trial_index=3, outcome="exhausted", response="", correct="", response_time_s="30.0000"), path)
    append_trial(_row(trial_index=4, outcome="aborted", response="", correct="", response_time_s="4.5000"), path)

    rows = load_trials(path)

    assert [r["outcome"] for r in rows] == ["answered", "timeout", "exhausted", "aborted"]
    assert rows[0]["correct"] is True
    # None, not False: no judgment was made, which is not the same as a wrong one.
    assert rows[1]["correct"] is None
    assert rows[2]["correct"] is None
    assert rows[3]["correct"] is None
    assert rows[1]["response"] == ""
    # The window length is still recorded, so idle vs. exploring stays answerable.
    assert rows[1]["response_time_s"] == 30.0


def test_practice_rows_are_recorded_and_tagged(tmp_path):
    path = tmp_path / "trials.csv"
    append_trial(_row(is_practice=1), path)

    rows = load_trials(path)

    assert rows[0]["is_practice"] is True
    assert rows[0]["outcome"] == "answered"


def test_pre_outcome_csvs_still_load_as_answered(tmp_path):
    """Sessions recorded before the column existed only ever wrote answered trials."""
    path = tmp_path / "legacy.csv"
    legacy_fields = [f for f in TRIAL_FIELDS if f != "outcome"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=legacy_fields)
        writer.writeheader()
        writer.writerow({f: _row(passes_json="[]").get(f, "") for f in legacy_fields})

    rows = load_trials(path)

    assert rows[0]["outcome"] == "answered"
    assert rows[0]["correct"] is True


def test_curve_fit_ignores_practice_and_unanswered_rows(tmp_path):
    path = tmp_path / "trials.csv"
    append_trial(_row(), path)
    append_trial(_row(trial_index=2, outcome="timeout", response="", correct=""), path)
    append_trial(_row(trial_index=3, outcome="exhausted", response="", correct=""), path)
    append_trial(_row(trial_index=4, outcome="aborted", response="", correct=""), path)
    append_trial(_row(trial_index=5, is_practice=1), path)

    levels, n_trials, n_taller = load_session_csvs([path])

    assert levels == [0.30]
    assert n_trials == [1]  # only the one answered main-block trial
    assert n_taller == [1]


def test_practice_round_and_dominant_hand_round_trip(tmp_path):
    path = tmp_path / "trials.csv"
    append_trial(_row(is_practice=1, practice_round=2, dominant_hand="right"), path)
    append_trial(_row(trial_index=2), path)

    rows = load_trials(path)

    assert rows[0]["practice_round"] == 2
    assert rows[0]["dominant_hand"] == "right"
    # Main-block rows and rows written without the column read as round 0.
    assert rows[1]["practice_round"] == 0


def test_each_row_names_the_render_mode_the_trial_ran_in_and_its_felt_gaps():
    from dataclasses import replace

    from experiment.config import ExperimentConfig
    from experiment.constant_stimuli import TrialSpec
    from experiment.main import _trial_row
    from experiment.trial import TrialTimeout

    assert TRIAL_FIELDS[-2:] == ["render_mode", "felt_gaps"]
    spec = TrialSpec(level_pct=0.1, comparison_height_mm=110.0, reference_height_mm=100.0, reference_side="left")
    timeout = TrialTimeout(elapsed_s=30.0, felt_gaps=2)
    mixed = ExperimentConfig(base_height_mm=100.0, bar_width_mm=10.0, render_mode="mixed")
    row = _trial_row("S", mixed, replace(spec, render_mode="kalman"), 1, "timeout", timeout)
    assert row["render_mode"] == "kalman" and row["felt_gaps"] == 2
    single = ExperimentConfig(base_height_mm=100.0, bar_width_mm=10.0, render_mode="position")
    assert _trial_row("S", single, spec, 1, "timeout", timeout)["render_mode"] == "position"
