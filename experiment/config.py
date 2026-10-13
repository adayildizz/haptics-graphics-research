"""Hardware/rig constants plus the single experiment-design config source.

Rig geometry (signal generator address, IR frame mapping, physical surface
padding) is fixed hardware truth that is measured once and not varied per
session, so it stays as plain module constants here. Everything the
supervisor might reasonably want to change between sessions --- stimulus
geometry, the constant-stimuli design, task behavior --- lives in
``ExperimentConfig`` and is loaded from a YAML file. No experiment parameter
may be hard-coded outside that dataclass.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .render_logic import RENDER_MODES

# --- Hardware (signal generator) ---
CARRIER_FREQUENCY = 125       # Hz
PEAK_VOLTAGE = 4.0            # Vpp, before external amplification
MIN_VOLTAGE = 0.0             # V when the electroadhesion signal is off
DISABLE_OUTPUT_WHEN_OFF = True
OFFSET_V = 0
WAVE_SQUARE = "SQU"
VISA_ADDRESS = "TCPIP0::169.254.2.20::inst0::INSTR"

# --- Display/input calibration ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
USE_FULLSCREEN = True
FPS = 60

# The IR frame maps directly to the full 1920 x 1080 display.
IR_FRAME_WIDTH_MM = 249.0
IR_FRAME_HEIGHT_MM = 187.0
IR_FRAME_SCREEN_WIDTH_PX = 1920
IR_FRAME_SCREEN_HEIGHT_PX = 1080

# Physical touch surface placement inside the IR frame.
HAPTIC_SURFACE_WIDTH_MM = 194.0
HAPTIC_SURFACE_HEIGHT_MM = 145.0
HAPTIC_SURFACE_LEFT_PADDING_MM = 23.0
HAPTIC_SURFACE_TOP_PADDING_MM = 16.0

# Rendering (rig behavior, not experimental design)
MIN_SPEED_MM_S = 1.0
MAX_SIGNAL_DURATION_S = 2.0


@dataclass(frozen=True)
class ExperimentConfig:
    """The single source of experiment-design parameters.

    Loaded from a YAML file via ``load_experiment_config``; never
    hand-edited constants scattered through the code. Every field the
    supervisor is likely to tune lives here.
    """

    # Stimulus geometry (mm)
    base_height_mm: float
    bar_width_mm: float
    inter_bar_gap_mm: float = 40.0  # >= 3.0 (BANA 3.4.3.13; Tang & Beebe 1998)

    # Constant-stimuli design
    delta_max_pct: float = 0.30
    n_levels: int = 6
    include_zero_level: bool = False
    trials_per_level: int = 10
    sweeps_per_block: int = 1
    catch_trial_pct: float = 0.10

    # Task
    feedback: bool = False
    n_practice_trials: int = 16
    practice_easiest_levels: int = 2
    break_every_n_trials: int = 30
    response_timeout_s: float = 30.0
    max_trial_attempts: int = 3
    practice_voice_feedback: bool = True
    practice_spoken_feedback: bool = True
    practice_pass_fraction: float = 0.75
    ideal_finger_speed_mm_s: float = 100.0
    ideal_speed_tolerance_pct: float = 0.30
    record_main_trace: bool = True

    # Audio environment
    masking_noise: bool = True
    masking_noise_volume: float = 0.30

    # Display
    blind_test_mode: bool = False  # hide the bars/columns entirely; find them by touch only

    # Rendering (README: Rendering Method)
    render_mode: str = "predictive"  # "predictive" | "position" | "timed_legacy"
    predict_window_samples: int = 3
    predict_horizon_factor: float = 1.5  # x the measured median sample interval
    predict_horizon_max_ms: float = 25.0
    predict_min_speed_mm_s: float = 5.0
    latency_compensation_ms: float = 0.0
    signal_watchdog_ms: float = 100.0

    # Hardware
    carrier_freq_hz: float = 125.0
    voltage_peak: float = 4.0
    ir_sample_hz_nominal: float = 100.0

    # Staircase pilot mode (only used when mode == "staircase_pilot")
    staircase_dh_start_pct: float = 2.0
    staircase_dh_min_pct: float = 0.5
    staircase_dh_step_pct: float = 0.1
    staircase_n_reversals: int = 12
    staircase_n_reversals_averaged: int = 8

    # Bookkeeping
    rng_seed: int | None = None
    participant_id: str = ""
    dominant_hand: str = ""  # "left" | "right" | "" (not recorded)
    mode: str = "constant_stimuli"  # or "staircase_pilot"

    def __post_init__(self) -> None:
        if self.inter_bar_gap_mm < 3.0:
            raise ValueError("inter_bar_gap_mm must be >= 3.0 mm (BANA 3.4.3.13)")
        if self.delta_max_pct <= 0:
            raise ValueError("delta_max_pct must be > 0")
        if self.n_levels < 2:
            raise ValueError("n_levels must be >= 2")
        if self.trials_per_level < 1:
            raise ValueError("trials_per_level must be >= 1")
        if self.practice_easiest_levels < 1:
            raise ValueError("practice_easiest_levels must be >= 1")
        if self.sweeps_per_block < 1:
            raise ValueError("sweeps_per_block must be >= 1")
        if self.trials_per_level % self.sweeps_per_block:
            raise ValueError(
                f"trials_per_level ({self.trials_per_level}) must be divisible by "
                f"sweeps_per_block ({self.sweeps_per_block}), so every block is balanced"
            )
        if self.response_timeout_s <= 0:
            raise ValueError("response_timeout_s must be > 0")
        if self.max_trial_attempts < 1:
            raise ValueError("max_trial_attempts must be >= 1")
        if self.ideal_finger_speed_mm_s <= 0:
            raise ValueError("ideal_finger_speed_mm_s must be > 0")
        if not 0 <= self.ideal_speed_tolerance_pct < 1:
            raise ValueError("ideal_speed_tolerance_pct must be >= 0 and < 1")
        if not 0 <= self.practice_pass_fraction <= 1:
            raise ValueError("practice_pass_fraction must be between 0 and 1")
        if not 0 <= self.masking_noise_volume <= 1:
            raise ValueError("masking_noise_volume must be between 0 and 1")
        if self.render_mode not in RENDER_MODES:
            raise ValueError(f"render_mode must be one of {RENDER_MODES}, got {self.render_mode!r}")
        if self.predict_window_samples < 2:
            raise ValueError("predict_window_samples must be >= 2")
        if self.predict_horizon_factor <= 0 or self.predict_horizon_max_ms <= 0:
            raise ValueError("predict_horizon_factor and predict_horizon_max_ms must be > 0")
        if self.predict_min_speed_mm_s < 0 or self.latency_compensation_ms < 0:
            raise ValueError("predict_min_speed_mm_s and latency_compensation_ms must be >= 0")
        if self.signal_watchdog_ms <= 0:
            raise ValueError("signal_watchdog_ms must be > 0")
        if self.dominant_hand not in ("", "left", "right"):
            raise ValueError("dominant_hand must be 'left', 'right', or empty")
        if self.mode not in ("constant_stimuli", "staircase_pilot"):
            raise ValueError(f"unknown mode: {self.mode!r}")


def _load_raw_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text) or {}
    if path.suffix.lower() == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        return tomllib.loads(text)
    raise ValueError(f"unsupported config format: {path.suffix}")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an ``ExperimentConfig`` from a YAML or TOML file."""
    raw = _load_raw_config(Path(path))
    known = {f.name for f in fields(ExperimentConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    return ExperimentConfig(**raw)


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return asdict(cfg)


def write_config_snapshot(cfg: ExperimentConfig, path: str | Path, extra: dict[str, Any] | None = None) -> None:
    """Persist the fully-resolved config (with the actual rng_seed used) as JSON."""
    payload = config_to_dict(cfg)
    if extra:
        payload.update(extra)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))
