from __future__ import annotations

import json

import pytest

from experiment.config import ExperimentConfig, load_experiment_config, write_config_snapshot

DEFAULT_CONFIG = "experiment/configs/default.yaml"


def test_load_default_yaml_config():
    cfg = load_experiment_config("experiment/configs/default.yaml")
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.base_height_mm == 10.0
    assert cfg.delta_max_pct == 0.30
    assert cfg.response_timeout_s == 30.0
    assert cfg.practice_voice_feedback is True
    assert cfg.ideal_finger_speed_mm_s == 100.0
    assert cfg.ideal_speed_tolerance_pct == 0.30
    assert cfg.record_main_trace is True
    assert cfg.mode == "constant_stimuli"
    # The briefing: six real levels at +-10/20/30 (odd n drops 0%), no break
    # screens, 16 practice trials passed at 12, spoken feedback, brown noise.
    assert cfg.n_levels == 7
    assert cfg.break_every_n_trials == 0
    assert cfg.n_practice_trials == 16
    assert cfg.practice_pass_fraction == 0.75
    assert cfg.practice_spoken_feedback is True
    assert cfg.masking_noise is True
    assert cfg.dominant_hand == ""


def test_dominant_hand_is_validated():
    ExperimentConfig(base_height_mm=10.0, bar_width_mm=10.0, dominant_hand="left")
    with pytest.raises(ValueError):
        ExperimentConfig(base_height_mm=10.0, bar_width_mm=10.0, dominant_hand="both")
    with pytest.raises(ValueError):
        ExperimentConfig(base_height_mm=10.0, bar_width_mm=10.0, masking_noise_volume=1.5)


def test_non_positive_response_timeout_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(base_height_mm=10.0, bar_width_mm=10.0, response_timeout_s=0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ideal_finger_speed_mm_s": 0},
        {"ideal_speed_tolerance_pct": -0.1},
        {"ideal_speed_tolerance_pct": 1.0},
    ],
)
def test_invalid_practice_speed_config_rejected(overrides):
    values = {"base_height_mm": 10.0, "bar_width_mm": 10.0, **overrides}
    with pytest.raises(ValueError):
        ExperimentConfig(**values)


def test_the_removed_staircase_pilot_is_refused_and_its_old_settings_are_ignored(tmp_path):
    old = tmp_path / "last_used.yaml"
    old.write_text(
        "base_height_mm: 100.0\nbar_width_mm: 20.0\nstaircase_dh_start_pct: 2.0\nstaircase_n_reversals: 12\n"
    )
    cfg = load_experiment_config(old)  # a config the GUI saved before the removal still loads
    assert cfg.mode == "constant_stimuli" and cfg.bar_width_mm == 20.0
    pilot = tmp_path / "pilot.yaml"
    pilot.write_text("base_height_mm: 100.0\nbar_width_mm: 20.0\nmode: staircase_pilot\n")
    with pytest.raises(ValueError, match="staircase_pilot.*removed"):
        load_experiment_config(pilot)


def test_unknown_config_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("base_height_mm: 10.0\nbar_width_mm: 10.0\ntypo_field: 1\n")
    with pytest.raises(ValueError):
        load_experiment_config(bad)


def test_changing_yaml_changes_config_with_zero_code_edits(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "base_height_mm: 20.0\nbar_width_mm: 5.0\ninter_bar_gap_mm: 6.0\n"
        "delta_max_pct: 0.15\nn_levels: 4\ntrials_per_level: 20\n"
    )
    cfg = load_experiment_config(custom)
    assert cfg.base_height_mm == 20.0
    assert cfg.delta_max_pct == 0.15
    assert cfg.n_levels == 4
    assert cfg.trials_per_level == 20


def test_render_settings_default_to_predictive_and_reach_the_config_snapshot(tmp_path):
    cfg = load_experiment_config(DEFAULT_CONFIG)
    assert cfg.render_mode == "predictive"
    assert (cfg.predict_window_samples, cfg.predict_horizon_factor, cfg.predict_horizon_max_ms) == (3, 1.5, 25.0)
    assert (cfg.predict_min_speed_mm_s, cfg.latency_compensation_ms, cfg.signal_watchdog_ms) == (5.0, 0.0, 100.0)
    snapshot = tmp_path / "config.json"
    write_config_snapshot(cfg, snapshot)
    saved = json.loads(snapshot.read_text())
    for key in ("render_mode", "predict_window_samples", "predict_horizon_factor", "predict_horizon_max_ms",
                "predict_min_speed_mm_s", "latency_compensation_ms", "signal_watchdog_ms"):  # fmt: skip
        assert saved[key] == getattr(cfg, key)


@pytest.mark.parametrize(
    "bad",
    [
        {"render_mode": "timed"},
        {"predict_window_samples": 1},
        {"predict_horizon_factor": 0},
        {"predict_horizon_max_ms": 0},
        {"predict_min_speed_mm_s": -1},
        {"latency_compensation_ms": -1},
        {"signal_watchdog_ms": 0},
    ],
)
def test_render_settings_are_validated(bad):
    with pytest.raises(ValueError):
        ExperimentConfig(base_height_mm=10.0, bar_width_mm=10.0, **bad)
    for mode in ("predictive", "position", "timed_legacy"):
        assert ExperimentConfig(base_height_mm=10.0, bar_width_mm=10.0, render_mode=mode).render_mode == mode
