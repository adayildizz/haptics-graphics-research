"""Entry point for the tactile bar-height JND experiment.

One run mode, ``constant_stimuli``: the method-of-constant-stimuli block,
with its practice block first.

``--dry-run`` simulates an ideal observer end-to-end (no pygame, no
hardware) to sanity-check that a config produces a reasonable psychometric
curve before running a live participant.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from . import calibration as calibration_module
from . import constant_stimuli, data_logger
from .config import ExperimentConfig, config_to_dict, load_experiment_config, write_config_snapshot
from .constant_stimuli import TrialSpec, practice_passed, practice_required_correct, resolve_seed
from .input_keys import is_continue_key, is_exit_key

if TYPE_CHECKING:
    from .trace_recorder import AsyncTraceRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tactile bar-height JND experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML/TOML ExperimentConfig file.")
    parser.add_argument("--participant", default=None)
    parser.add_argument("--windowed", action="store_true", help="Use a debug window instead of fullscreen.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate an ideal observer; no pygame/hardware.")
    return parser.parse_args()


def _session_paths(cfg: ExperimentConfig, participant: str) -> tuple[str, Path, Path]:
    data_dir = data_logger.ensure_data_dir()
    session_id = f"{participant}_{time.strftime('%Y%m%d_%H%M%S')}"
    trial_path = data_dir / f"{session_id}_trials.csv"
    config_snapshot_path = data_dir / f"{session_id}_config.json"
    return session_id, trial_path, config_snapshot_path


def run_dry_run(cfg: ExperimentConfig) -> int:
    """Simulate an ideal observer through the configured design; no pygame."""
    from analysis.fit_psychometric import fit_psychometric, plot_psychometric, simulate_ideal_observer

    seed = resolve_seed(cfg)
    levels = constant_stimuli.compute_levels(cfg)
    print(f"dry-run: mode={cfg.mode} seed={seed} levels={[f'{l:+.1%}' for l in levels]}")

    true_sigma = cfg.delta_max_pct / 3.0
    levels_out, n_trials, n_taller = simulate_ideal_observer(
        levels,
        trials_per_level=cfg.trials_per_level,
        true_pse=0.0,
        true_sigma=true_sigma,
        true_lapse=0.02,
        seed=seed,
    )
    fit = fit_psychometric(levels_out, n_trials, n_taller)
    data_dir = data_logger.ensure_data_dir()
    out_path = data_dir / f"dry_run_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plot_psychometric(fit, out_path)
    print(
        f"dry-run fit ({fit.backend}): pse={fit.pse:+.4f} slope_sigma={fit.slope_sigma:.4f} "
        f"lapse={fit.lapse_rate:.4f} jnd={fit.jnd:.4f} (true_sigma={true_sigma:.4f})"
    )
    print(f"saved {out_path} and {out_path.with_suffix('.json')}")
    return 0


def wait_for_continue_or_exit(screen, message: str) -> bool:
    import pygame

    from . import display

    display.draw_break(screen, message)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if is_exit_key(event.key, pygame):
                    return False
                if is_continue_key(event.key, pygame):
                    return True


def _trial_row(
    session_id: str,
    cfg: ExperimentConfig,
    spec: TrialSpec,
    trial_index: int,
    outcome: str,
    result: Any,
    practice_round: int = 0,
) -> dict[str, Any]:
    """Build one trial CSV row for any outcome.

    Every trial the participant was actually shown gets a row -- answered,
    timed out, exhausted, or aborted -- told apart by ``outcome``, so the CSV
    and the trace database describe the same set of events. Unanswered rows
    leave ``response`` and ``correct`` blank rather than writing 0, because a
    0 there would read as a wrong answer; ``response_time_s`` holds how long
    the trial was actually open, and the passes still describe what the
    finger was doing.
    """
    answered = outcome == "answered"
    return {
        "session_id": session_id,
        "participant_id": cfg.participant_id,
        "dominant_hand": cfg.dominant_hand,
        "timestamp": result.timestamp,
        "trial_index": trial_index,
        "mode": cfg.mode,
        "reference_side": spec.reference_side,
        "outcome": outcome,
        "response": result.response if answered else "",
        "correct": int(result.correct) if answered else "",
        "level_pct": f"{spec.level_pct:.6f}",
        "comparison_height_mm": f"{spec.comparison_height_mm:.4f}",
        "reference_height_mm": f"{spec.reference_height_mm:.4f}",
        "bar_width_mm": f"{cfg.bar_width_mm:.4f}",
        "is_catch": int(spec.is_catch),
        "is_practice": int(spec.is_practice),
        "practice_round": practice_round,
        "response_time_s": f"{(result.response_time_s if answered else result.elapsed_s):.4f}",
        "passes": result.passes,
    }


def _start_trace_attempt(
    trace_recorder: AsyncTraceRecorder,
    session_id: str,
    cfg: ExperimentConfig,
    spec: TrialSpec,
    *,
    trial_index: int,
    attempt_index: int,
    practice_round: int = 0,
):
    """Open a trace attempt for one presentation, practice or main.

    Practice attempts are keyed by round as well, because ``trial_index``
    restarts at 1 in every practice round and again in the main block.
    """
    from .trace_store import AttemptDefinition

    is_practice = practice_round > 0
    prefix = f"p{practice_round}:" if is_practice else ""
    return trace_recorder.start_attempt(
        f"{prefix}{trial_index}:{attempt_index}",
        AttemptDefinition(
            session_id=session_id,
            trial_index=trial_index,
            attempt_index=attempt_index,
            started_us=trace_recorder.elapsed_us(),
            level_pct=spec.level_pct,
            comparison_height_mm=spec.comparison_height_mm,
            reference_height_mm=spec.reference_height_mm,
            bar_width_mm=cfg.bar_width_mm,
            reference_side=spec.reference_side,
            is_catch=spec.is_catch,
            is_practice=is_practice,
            practice_round=practice_round,
        ),
    )


def run_constant_stimuli(
    screen,
    clock,
    calibration,
    signal,
    cfg: ExperimentConfig,
    session_id: str,
    trial_path: Path,
    seed: int,
    trace_recorder: AsyncTraceRecorder | None = None,
) -> None:
    from . import trial as trial_module

    print(f"constant_stimuli rng_seed={seed}")

    import random

    rng = random.Random(seed)

    if cfg.n_practice_trials > 0:
        from .speed_coach import PRACTICE_INSTRUCTION, PracticeSpeedCoach, SystemVoice

        # One system voice serves both jobs in practice: the speed coach
        # ("Faster" / "Slower" / "Good speed") and the spoken correct /
        # incorrect feedback. The participant is blindfolded, so without the
        # spoken word the practice block would give no feedback at all.
        voice = SystemVoice() if (cfg.practice_voice_feedback or cfg.practice_spoken_feedback) else None
        voice_ok = voice is not None and voice.available
        speed_coach = (
            PracticeSpeedCoach(
                cfg.ideal_finger_speed_mm_s,
                cfg.ideal_speed_tolerance_pct,
                voice.speak,
            )
            if voice_ok and cfg.practice_voice_feedback
            else None
        )
        feedback_speaker = voice.speak_now if voice_ok and cfg.practice_spoken_feedback else None
        required = practice_required_correct(cfg.n_practice_trials, cfg.practice_pass_fraction)
        try:
            if not wait_for_continue_or_exit(
                screen,
                f"Practice trials (feedback on). Pass mark {required}/{cfg.n_practice_trials}.",
            ):
                return
            if voice_ok and cfg.practice_voice_feedback:
                voice.speak(PRACTICE_INSTRUCTION, wait=True)

            practice_round = 0
            while True:
                practice_round += 1
                practice_pending = [
                    constant_stimuli.ScheduledTrial(spec)
                    for spec in constant_stimuli.build_practice_sequence(cfg, rng)
                ]
                practice_deferred: list[constant_stimuli.ScheduledTrial] = []
                shown = 0
                n_correct = 0
                while practice_pending or practice_deferred:
                    if not practice_pending:
                        practice_pending = constant_stimuli.take_retry_round(practice_deferred, rng)
                    scheduled = practice_pending.pop(0)
                    scheduled.attempts += 1
                    shown += 1
                    trace_attempt = None
                    if trace_recorder is not None:
                        trace_attempt = _start_trace_attempt(
                            trace_recorder,
                            session_id,
                            cfg,
                            scheduled.spec,
                            trial_index=shown,
                            attempt_index=scheduled.attempts,
                            practice_round=practice_round,
                        )
                    result = trial_module.run_trial(
                        screen,
                        clock,
                        calibration,
                        signal,
                        cfg,
                        scheduled.spec,
                        shown,
                        fps=60,
                        speed_coach=speed_coach,
                        trace_attempt=trace_attempt,
                        feedback_speaker=feedback_speaker,
                    )
                    if isinstance(result, trial_module.TrialAborted):
                        outcome = "aborted"
                    elif isinstance(result, trial_module.TrialTimeout):
                        # Practice is timed too, so it needs the same bounded
                        # retry: re-shown at the end of the round, then let go.
                        requeued = constant_stimuli.defer_timed_out_trial(
                            scheduled, practice_deferred, cfg.max_trial_attempts
                        )
                        outcome = "timeout" if requeued else "exhausted"
                    else:
                        outcome = "answered"
                        n_correct += int(result.correct)
                    data_logger.append_trial(
                        _trial_row(
                            session_id, cfg, scheduled.spec, shown, outcome, result,
                            practice_round=practice_round,
                        ),
                        trial_path,
                    )
                    if outcome == "aborted":
                        return

                passed = practice_passed(n_correct, cfg.n_practice_trials, cfg.practice_pass_fraction)
                print(
                    f"practice round {practice_round}: {n_correct}/{cfg.n_practice_trials} correct "
                    f"(pass mark {required}) -> {'passed' if passed else 'repeat'}"
                )
                if passed:
                    break
                # Below the pass mark: the whole block runs again, freshly
                # shuffled. There is no cap on rounds by design (the
                # supervisor can always end the session with Escape).
                if voice_ok:
                    voice.speak_now("The practice block will be repeated.")
                if not wait_for_continue_or_exit(
                    screen,
                    f"Practice round {practice_round}: {n_correct}/{cfg.n_practice_trials} correct, "
                    f"below the pass mark of {required}. Repeating practice.",
                ):
                    return
        finally:
            if voice is not None:
                voice.close()

    if not wait_for_continue_or_exit(screen, "Main block. No feedback unless configured."):
        return

    pending_trials = constant_stimuli.build_schedule(cfg, seed)
    total_trials = len(pending_trials)
    deferred_trials: list[constant_stimuli.ScheduledTrial] = []
    completed_trials = 0
    exhausted_trials = 0
    attempt_counts: dict[int, int] = {}
    while pending_trials or deferred_trials:
        if not pending_trials:
            # The scheduled sequence is done; everything that expired is
            # replayed here, at the very end of the block. No announcement
            # screen: from the participant's side this is just the next trial,
            # so retries are not flagged as retries.
            pending_trials = constant_stimuli.take_retry_round(deferred_trials, rng)
        scheduled = pending_trials.pop(0)
        scheduled.attempts += 1
        spec = scheduled.spec
        trial_index = completed_trials + 1
        attempt_index = attempt_counts.get(trial_index, 0) + 1
        attempt_counts[trial_index] = attempt_index
        trace_attempt = None
        if trace_recorder is not None:
            trace_attempt = _start_trace_attempt(
                trace_recorder,
                session_id,
                cfg,
                spec,
                trial_index=trial_index,
                attempt_index=attempt_index,
            )
        result = trial_module.run_trial(
            screen,
            clock,
            calibration,
            signal,
            cfg,
            spec,
            trial_index,
            fps=60,
            trace_attempt=trace_attempt,
            total_trials=total_trials,
        )
        if isinstance(result, trial_module.TrialAborted):
            data_logger.append_trial(
                _trial_row(session_id, cfg, spec, trial_index, "aborted", result),
                trial_path,
            )
            print(f"session aborted mid-trial ({result.reason})")
            return
        if isinstance(result, trial_module.TrialTimeout):
            requeued = constant_stimuli.defer_timed_out_trial(
                scheduled, deferred_trials, cfg.max_trial_attempts
            )
            if not requeued:
                exhausted_trials += 1
                total_trials -= 1
                print(
                    f"trial exhausted after {scheduled.attempts} unanswered attempts "
                    f"(level {spec.level_pct:+.1%})"
                )
            data_logger.append_trial(
                _trial_row(
                    session_id,
                    cfg,
                    spec,
                    trial_index,
                    "timeout" if requeued else "exhausted",
                    result,
                ),
                trial_path,
            )
            continue

        data_logger.append_trial(
            _trial_row(session_id, cfg, spec, trial_index, "answered", result),
            trial_path,
        )
        completed_trials += 1

        if (
            cfg.break_every_n_trials
            and completed_trials % cfg.break_every_n_trials == 0
            and completed_trials < total_trials
        ):
            if not wait_for_continue_or_exit(
                screen,
                f"Break. {completed_trials}/{total_trials} trials done.",
            ):
                return

    print(f"constant_stimuli complete: {completed_trials} answered, {exhausted_trials} exhausted")


def run() -> int:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    participant = args.participant or cfg.participant_id or time.strftime("%Y%m%d_%H%M%S")

    if args.dry_run:
        return run_dry_run(cfg)

    import pygame

    from . import audio_cues, display, masking_noise, stimulus
    from .signal_scheduler import SignalScheduler, exit_cleanly_on_sigterm
    from .trial_render import MODE as RENDER_MODE

    pygame.init()
    # Synthesize the response/timeout cues now: building one costs tens of
    # thousands of Python-level sine evaluations, which would drop frames if
    # it happened lazily inside the first trial.
    audio_cues.preload()
    # Brown masking noise runs from here until the session ends, so the
    # participant never hears the rig (relay clicks, fans, the operator's
    # keyboard). Cues and speech play over it on other mixer channels.
    if cfg.masking_noise:
        masking_noise.start(cfg.masking_noise_volume)
    clock = pygame.time.Clock()
    session_id, trial_path, config_snapshot_path = _session_paths(cfg, participant)

    screen = display.init_window(fullscreen=not args.windowed)
    current_calibration = calibration_module.make_configured_ir_frame_calibration(screen.get_size())
    print(
        "Using display calibration "
        f"({current_calibration.source}): "
        f"{current_calibration.active_width_mm:.2f} mm x "
        f"{current_calibration.active_height_mm:.2f} mm, "
        f"{current_calibration.px_per_mm_x:.4f} px/mm X, "
        f"{current_calibration.px_per_mm_y:.4f} px/mm Y"
    )

    resolved_seed = resolve_seed(cfg)
    write_config_snapshot(
        cfg,
        config_snapshot_path,
        # render_mode is no longer a setting; it stays in the snapshot so sessions remain comparable with older ones.
        extra={"session_id": session_id, "resolved_rng_seed": resolved_seed, "render_mode": RENDER_MODE},
    )

    trace_recorder = None
    trace_path = trial_path.parent / f"{session_id}_trace.sqlite3"
    if cfg.mode == "constant_stimuli" and cfg.record_main_trace:
        from .trace_recorder import AsyncTraceRecorder

        trace_recorder = AsyncTraceRecorder(trace_path)
        trace_config = config_to_dict(cfg)
        trace_config["resolved_rng_seed"] = resolved_seed
        trace_config["render_mode"] = RENDER_MODE
        trace_recorder.create_session(
            session_id=session_id,
            participant_id=participant,
            started_at=datetime.now().astimezone().isoformat(),
            config=trace_config,
            calibration=asdict(current_calibration),
            app_version="ea-demo-trace-v1",
        )
        print(f"Main-trial trace enabled: {trace_path}")

    instrument = stimulus.connect_hardware(cfg)
    # From here until close() below, the scheduler thread is the only thing that switches the output.
    signal = SignalScheduler(stimulus.device_for(instrument), watchdog_s=cfg.signal_watchdog_ms / 1000.0).start()
    exit_cleanly_on_sigterm()
    print(
        f"Render mode: {RENDER_MODE} (watchdog {cfg.signal_watchdog_ms:g} ms, "
        f"busy-wait margin {1000.0 * (signal.spin_s or 0.0):.1f} ms)"
    )

    try:
        if not wait_for_continue_or_exit(
            screen,
            f"Ready to begin ({cfg.mode}, render: {RENDER_MODE}). "
            f"Base height {cfg.base_height_mm:g} mm, width {cfg.bar_width_mm:g} mm.",
        ):
            return 0

        run_constant_stimuli(
            screen,
            clock,
            current_calibration,
            signal,
            cfg,
            session_id,
            trial_path,
            resolved_seed,
            trace_recorder=trace_recorder,
        )

        if signal.error is not None:
            print(f"ERROR: the signal scheduler failed ({signal.error!r}); the session was aborted.")
            display.draw_break(screen, f"Signal error: session aborted. Data saved in {trial_path.parent.name}/")
            pygame.display.flip()
            pygame.time.wait(1500)
            return 1
        display.draw_break(screen, f"Session complete. Data saved in {trial_path.parent.name}/")
        pygame.display.flip()
        pygame.time.wait(1500)
        return 0
    finally:
        signal.close()
        stimulus.close_hardware(instrument)
        masking_noise.stop()
        pygame.quit()
        if trace_recorder is not None:
            trace_error = trace_recorder.close()
            if trace_error is None:
                print(f"Trace saved: {trace_path}")
            else:
                print(f"WARNING: Trace recording failed ({trace_error})")


if __name__ == "__main__":
    sys.exit(run())
