"""Rig check: what the touch frame actually delivers to pygame. No generator, no experiment."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

JUMP_MM = 10.0
MIN_MOVING_FRAMES = 30  # half a second of movement; less and the answers would be guesses
DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Frame:
    """What one pass of the 60 Hz loop saw."""

    t_s: float
    pos: tuple[int, int]
    pressed: bool  # left mouse button held, as polled
    motions: int = 0  # MOUSEMOTION events since the previous frame
    button_down: int = 0
    button_up: int = 0
    finger_down: int = 0  # SDL touch events
    finger_up: int = 0
    finger_motion: int = 0


def summarize(frames: Sequence[Frame], px_per_mm: tuple[float, float]) -> dict[str, object]:
    """Answers two questions: does every position report arrive, and is touch contact reported?"""
    duration = frames[-1].t_s - frames[0].t_s
    intervals = [b.t_s - a.t_s for a, b in zip(frames, frames[1:])]
    moving = [b for a, b in zip(frames, frames[1:]) if b.pos != a.pos]
    moving_s = sum(dt for (a, b), dt in zip(zip(frames, frames[1:]), intervals) if b.pos != a.pos)
    motions_while_moving = sum(frame.motions for frame in moving)
    per_frame = motions_while_moving / len(moving) if moving else 0.0
    jumps = sum(
        ((b.pos[0] - a.pos[0]) / px_per_mm[0]) ** 2 + ((b.pos[1] - a.pos[1]) / px_per_mm[1]) ** 2 >= JUMP_MM**2
        for a, b in zip(frames, frames[1:])
    )
    button_down = sum(frame.button_down for frame in frames)
    pressed_while_moving = sum(frame.pressed for frame in moving) / len(moving) if moving else 0.0
    finger_down = sum(frame.finger_down for frame in frames)
    enough = len(moving) >= MIN_MOVING_FRAMES
    return {
        "seconds": round(duration, 2),
        "loop_hz": round(len(intervals) / duration, 1) if duration > 0 else 0.0,
        "loop_interval_median_ms": round(1000.0 * statistics.median(intervals), 2) if intervals else 0.0,
        "frames_with_movement": len(moving),
        "motion_events_per_moving_frame": round(per_frame, 2),
        "motion_events_per_second_while_moving": round(motions_while_moving / moving_s, 1) if moving_s > 0 else 0.0,
        "frames_by_motion_events": {
            label: sum(1 for frame in moving if pick(frame.motions))
            for label, pick in (("0", lambda n: n == 0), ("1", lambda n: n == 1), ("2", lambda n: n == 2), ("3+", lambda n: n >= 3))
        },
        "button_down_events": button_down,
        "button_up_events": sum(frame.button_up for frame in frames),
        "button_held_share_while_moving": round(pressed_while_moving, 2),
        "finger_down_events": finger_down,
        "finger_up_events": sum(frame.finger_up for frame in frames),
        "finger_motion_events": sum(frame.finger_motion for frame in frames),
        "jumps_over_10_mm": int(jumps),
        "enough_movement": enough,
        # More than one event per frame on average means reports are not being merged into one.
        "every_report_arrives": per_frame > 1.2 if enough else None,
        "contact_as_mouse_button": button_down >= 1 and pressed_while_moving > 0.8 if enough else None,
        "contact_as_touch_events": finger_down >= 1 if enough else None,
    }


def report(summary: dict[str, object]) -> str:
    counts = summary["frames_by_motion_events"]
    yes = lambda flag: "UNKNOWN" if flag is None else ("YES" if flag else "NO")  # noqa: E731
    warning = (
        ()
        if summary["enough_movement"]
        else (
            "",
            "!! Too little finger movement reached the program, so the answers below are UNKNOWN, not NO.",
            "!! Check that the red dot followed your finger on the touch screen, then run the check again.",
        )
    )
    return "\n".join(
        (
            f"Loop: {summary['loop_hz']} Hz (median interval {summary['loop_interval_median_ms']} ms), "
            f"{summary['seconds']} s, {summary['frames_with_movement']} frames with movement",
            f"Position reports while moving: {summary['motion_events_per_second_while_moving']} per second, "
            f"{summary['motion_events_per_moving_frame']} per frame (frames with 0/1/2/3+ events: "
            f"{counts['0']}/{counts['1']}/{counts['2']}/{counts['3+']})",
            f"Mouse button: {summary['button_down_events']} down, {summary['button_up_events']} up, "
            f"held in {100 * summary['button_held_share_while_moving']:.0f}% of moving frames",
            f"Touch events: {summary['finger_down_events']} down, {summary['finger_up_events']} up, "
            f"{summary['finger_motion_events']} motion",
            f"Jumps of 10 mm or more between two frames: {summary['jumps_over_10_mm']}",
            "",
            f"1. Does every position report reach the program?   {yes(summary['every_report_arrives'])}",
            f"2. Is contact reported as a mouse button press?     {yes(summary['contact_as_mouse_button'])}",
            f"3. Is contact reported as touch events?             {yes(summary['contact_as_touch_events'])}",
            *warning,
        )
    )


def run(seconds: float, *, fullscreen: bool = True, out_dir: Path | None = DATA_DIR) -> dict[str, object]:
    """Record what arrives for ``seconds`` while the operator slides, lifts and touches again."""
    import pygame

    from . import display
    from .calibration import make_configured_ir_frame_calibration
    from .input_keys import is_exit_key

    pygame.init()
    screen = display.init_window(fullscreen=fullscreen)
    calibration = make_configured_ir_frame_calibration(screen.get_size())
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 26)
    frames: list[Frame] = []
    start = time.perf_counter()
    done = False
    try:
        while not done and time.perf_counter() - start < seconds:
            counts = dict.fromkeys(("motions", "button_down", "button_up", "finger_down", "finger_up", "finger_motion"), 0)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and is_exit_key(event.key, pygame)):
                    done = True
                elif event.type == pygame.MOUSEMOTION:
                    counts["motions"] += 1
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    counts["button_down"] += 1
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    counts["button_up"] += 1
                elif event.type == pygame.FINGERDOWN:
                    counts["finger_down"] += 1
                elif event.type == pygame.FINGERUP:
                    counts["finger_up"] += 1
                elif event.type == pygame.FINGERMOTION:
                    counts["finger_motion"] += 1
            now = time.perf_counter()
            frames.append(Frame(now, pygame.mouse.get_pos(), bool(pygame.mouse.get_pressed()[0]), **counts))
            screen.fill(display.BACKGROUND)
            lines = (
                "Input check: slide one finger on the touch surface, lift it and touch again several times.",
                "The red dot must follow your finger; if it does not, input is not reaching this window.",
                f"{max(0.0, seconds - (now - start)):4.1f} s left     ESC ends early     moving frames so far: "
                f"{sum(1 for a, b in zip(frames, frames[1:]) if a.pos != b.pos)}",
            )
            for row, text in enumerate(lines):
                screen.blit(font.render(text, True, display.TEXT), (40, 40 + 40 * row))
            pygame.draw.circle(screen, display.TOUCH_POINT, frames[-1].pos, 10)
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()
    if len(frames) < 2:
        raise RuntimeError("the check ended before two frames were recorded")
    summary = summarize(frames, (calibration.px_per_mm_x, calibration.px_per_mm_y))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"input_probe_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(summary, indent=2))
        summary["saved_to"] = str(path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.input_probe",
        description="Check what the touch frame delivers: position reports per second and contact. No generator.",
    )
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--windowed", action="store_true", help="Use a window instead of fullscreen.")
    args = parser.parse_args(argv)
    summary = run(args.seconds, fullscreen=not args.windowed)
    print(report(summary))
    print(f"\nSaved: {summary.get('saved_to')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
