"""The timed render rule, pinned against output recorded from the code before the render refactor."""

# data/render_legacy_golden.json was written by running this file with --write on commit 7af43d7.

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from render_harness import polyline, run_scripted_trial

from experiment.calibration import make_configured_ir_frame_calibration
from experiment.config import load_experiment_config
from experiment.constant_stimuli import TrialSpec

GOLDEN_PATH = Path(__file__).parent / "data" / "render_legacy_golden.json"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "experiment" / "configs" / "default.yaml"
SCREEN = (1920, 1080)
LEGACY_EVENTS = ("signal_on", "signal_off", "response", "timeout", "aborted")


def scenarios() -> dict[str, dict]:
    """Paths in surface mm over the default layout: bars at x 67-77 and 117-127, baseline y 145."""
    jitter = np.random.default_rng(5).uniform(-0.15, 0.15, (150, 2))
    noisy = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(polyline([(0, 55, 140), (149, 140, 139)], 150), jitter)]
    return {
        "horizontal_sweeps": dict(
            path=polyline([(0, 50, 140), (60, 140, 140), (90, 50, 140), (140, 140, 141)], 150), end="left"
        ),
        "fast_sweep": dict(path=polyline([(0, 40, 141), (30, 150, 141), (60, 40, 141)], 70), end="right"),
        "top_edge_probing": dict(
            path=polyline(
                [(0, 72, 125), (50, 72, 138), (60, 72, 138), (70, 72, 128), (100, 72, 137), (106, 72, 127), (130, 72, 127)],
                140,
            ),
            end="left",
        ),
        "corner_cut": dict(path=polyline([(0, 60, 130), (20, 72, 137), (22, 79, 139), (40, 100, 139)], 50), end="right"),
        "stop_inside": dict(
            path=polyline([(0, 60, 141), (20, 72, 141), (50, 72, 141), (120, 82, 141)], 130), end="left"
        ),
        "quick_dip": dict(
            path=polyline([(0, 60, 141), (8, 68, 141), (10, 70, 133), (30, 70, 120)], 40), end="numpad_right"
        ),
        "bar_to_bar_jump": dict(
            path=polyline([(0, 60, 141), (10, 72, 141), (11, 122, 141), (30, 124, 141), (40, 140, 141)], 50),
            end="left",
        ),
        "noisy_sweep": dict(path=noisy, end="left"),
        "comparison_on_left": dict(
            path=polyline([(0, 50, 139), (60, 140, 139), (100, 72, 128)], 110), end="right", reference_side="right"
        ),
        "ends_by_timeout": dict(path=polyline([(0, 60, 141), (90, 130, 141)], 100), end="timeout"),
        "ends_by_exit_key": dict(path=polyline([(0, 60, 141), (40, 75, 141)], 45), end="escape"),
        "ends_by_window_close": dict(path=polyline([(0, 60, 141), (40, 75, 141)], 45), end="quit"),
    }


def run_scenario(name: str, mode: str = "timed_legacy") -> dict:
    """One scripted trial; returns everything the render rule produced."""
    scenario = scenarios()[name]
    cfg = load_experiment_config(CONFIG_PATH)
    # The recording predates the IR glitch filter, which would reject the 50 mm step of bar_to_bar_jump.
    timeout_s = 1.2 if scenario["end"] == "timeout" else 30.0
    cfg = replace(cfg, render_mode=mode, response_timeout_s=timeout_s, ir_filter=False)
    spec = TrialSpec(
        level_pct=0.2,
        comparison_height_mm=12.0,
        reference_height_mm=10.0,
        reference_side=scenario.get("reference_side", "left"),
    )
    result, attempt, connect_commands, trial_commands = run_scripted_trial(
        cfg, make_configured_ir_frame_calibration(SCREEN), spec, scenario
    )
    return {
        "connect_commands": connect_commands,
        "trial_commands": trial_commands,
        "signal_on": [int(sample.signal_on) for sample in attempt.samples],
        "active_side": [sample.active_side for sample in attempt.samples],
        "events": [[t_us, kind, payload] for t_us, kind, payload in attempt.events if kind in LEGACY_EVENTS],
        "finished": attempt.finished,
        "result": type(result).__name__,
        "response": getattr(result, "response", None),
        "passes": [
            {key: round(value, 9) if isinstance(value, float) else value for key, value in entry.items()}
            for entry in result.passes
        ],
    }


@pytest.mark.parametrize("name", sorted(scenarios()))
def test_timed_render_rule_matches_the_recording_from_before_the_refactor(name):
    golden = json.loads(GOLDEN_PATH.read_text())[name]
    assert json.loads(json.dumps(run_scenario(name))) == golden


def test_the_scenarios_exercise_the_timed_tail():
    golden = json.loads(GOLDEN_PATH.read_text())
    # Signal on while the finger is outside both bars is the timed tail; the pin is worthless without it.
    tail_frames = sum(
        on and side is None for run in golden.values() for on, side in zip(run["signal_on"], run["active_side"])
    )
    assert tail_frames > 20
    assert {run["result"] for run in golden.values()} == {"TrialResult", "TrialTimeout", "TrialAborted"}


if __name__ == "__main__" and "--write" in sys.argv:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps({name: run_scenario(name) for name in sorted(scenarios())}, indent=1))
    print(f"wrote {GOLDEN_PATH}")
