from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from experiment import main as main_module
from experiment import trial as trial_module
from experiment.config import load_experiment_config
from experiment.display import trial_counter_text

CONFIG_PATH = Path(__file__).resolve().parents[1] / "experiment" / "configs" / "default.yaml"


def test_counter_shows_the_block_size_when_known():
    assert trial_counter_text(False, 12, 66) == "Trial 12/66"
    assert trial_counter_text(True, 3) == "Practice 3"
    assert trial_counter_text(False, 4, None) == "Trial 4"


def test_main_block_counts_answered_trials_out_of_the_scheduled_total(tmp_path):
    cfg = replace(load_experiment_config(CONFIG_PATH), n_practice_trials=0, participant_id="P01")
    shown: list[tuple[int, int | None]] = []
    doomed: list[object] = []

    def fake_run_trial(_screen, _clock, _calibration, _signal, _cfg, spec, trial_index, fps, **kwargs):
        shown.append((trial_index, kwargs.get("total_trials")))
        if not doomed:
            doomed.append(spec)
        if spec is doomed[0]:  # never answered: shown max_trial_attempts times, then given up
            return trial_module.TrialTimeout(elapsed_s=30.0)
        return trial_module.TrialResult(
            trial_index, spec.level_pct, spec.comparison_height_mm, spec.reference_height_mm,
            cfg.bar_width_mm, spec.reference_side, "left", True, spec.is_catch, False, 1.0,
        )  # fmt: skip

    with (
        patch.object(trial_module, "run_trial", fake_run_trial),
        patch.object(main_module, "wait_for_continue_or_exit", lambda _screen, _message: True),
    ):
        main_module.run_constant_stimuli(None, None, None, None, cfg, "P01_s", tmp_path / "P01_s_trials.csv", seed=3)

    assert shown[0] == (1, 66)
    assert len(shown) == 66 + cfg.max_trial_attempts - 1
    # Timeouts do not advance the counter: the unanswered trial's retries at the end all read 66/66.
    assert shown[-2:] == [(66, 66), (66, 66)]
    assert all(index <= total for index, total in shown)
