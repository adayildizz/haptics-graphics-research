"""Rig check: the IR frame's contact and position and the force sensor's load, recorded side by side.

Run it once per task (still, slide, lifts, hover, palm, ...) with a ``--label``. It first asks for the hands to
be off the screen for ``--bias-s`` seconds: the gauge voltages in the quietest part of that time are the unloaded
zero (the bias), and their spread is the noise the contact threshold is set from. Then it records every frame of the 60 Hz loop (position,
mouse button, touch events, newest force sample) for ``--seconds``, while the force reader process stores every
1 kHz sample. The screen shows the IR state and the load live.

Files, in ``--data-dir``, named ``contact_<label>_<time>``: ``_frames.csv`` (one row per frame),
``_force.sqlite3`` (every force sample), ``_probe.json`` (settings, bias, calibration matrix, live summary).
``python -m analysis.contact_check`` reads them.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict
from functools import partial
from pathlib import Path

from .ati_calibration import AtiCalibration, load_ati_calibration
from .contact_fusion import STATES, HysteresisContact, contact_state, load_of, thresholds_from_noise
from .force_daq import CHANNELS, FakeForceSource, ForceReader, NidaqmxForceSource

DATA_DIR = Path(__file__).resolve().parent / "data"
FRAME_FIELDS = (
    "t_s",
    "phase",
    "x_px",
    "y_px",
    "x_mm",
    "y_mm",
    "pressed",
    "motions",
    "button_down",
    "button_up",
    "finger_down",
    "finger_up",
    "finger_motion",
    "force_index",
    "force_t_s",
    "load",
)


def make_source(args: argparse.Namespace):
    if args.fake:
        return partial(FakeForceSource, channels=CHANNELS)
    return partial(
        NidaqmxForceSource,
        args.device,
        first_channel=args.first_channel,
        terminal=args.terminal,
        range_v=args.range_v,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    import pygame

    from . import display
    from .calibration import make_configured_ir_frame_calibration

    calibration: AtiCalibration | None = load_ati_calibration(args.cal) if args.cal else None
    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"contact_{args.label}_{time.strftime('%Y%m%d_%H%M%S')}"
    reader = ForceReader(stem.with_name(stem.name + "_force.sqlite3"), make_source(args)).start()
    rows: list[dict[str, object]] = []
    bias_volts: list[tuple[float, ...]] = []
    try:
        pygame.init()
        screen = display.init_window(fullscreen=not args.windowed)
        ir = make_configured_ir_frame_calibration(screen.get_size())
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 26)
        start = time.perf_counter()
        record_start = start + args.bias_s
        end = record_start + args.seconds
        bias: tuple[float, ...] | None = None
        detector: HysteresisContact | None = None
        unit = calibration.force_units if calibration is not None else "V (uncalibrated)"
        done = False
        while not done:
            now = time.perf_counter()
            if now >= end:
                break
            counts = dict.fromkeys(("motions", "button_down", "button_up", "finger_down", "finger_up", "finger_motion"), 0)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
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
            sample = reader.latest()
            phase = "bias" if now < record_start else "record"
            if phase == "bias" and sample is not None:
                bias_volts.append(sample.volts)
            if phase == "record" and bias is None:
                if len(bias_volts) < 2:
                    raise RuntimeError("no force samples arrived while measuring the zero level")
                quiet = _quietest(bias_volts)  # a hand that comes back early must not set the zero
                bias = tuple(statistics.fmean(v[c] for v in quiet) for c in range(len(quiet[0])))
                noise = [load_of(v, bias, calibration) for v in quiet]
                limits = thresholds_from_noise(noise)
                detector = HysteresisContact(args.on if args.on is not None else limits.on, args.off if args.off is not None else limits.off)
            load = load_of(sample.volts, bias, calibration) if (sample is not None and bias is not None) else None
            pos = pygame.mouse.get_pos()
            pressed = bool(pygame.mouse.get_pressed()[0])
            rows.append(
                {
                    "t_s": now,
                    "phase": phase,
                    "x_px": pos[0],
                    "y_px": pos[1],
                    "x_mm": (pos[0] - ir.active_left_px) / ir.px_per_mm_x,
                    "y_mm": (pos[1] - ir.active_top_px) / ir.px_per_mm_y,
                    "pressed": int(pressed),
                    **counts,
                    "force_index": sample.index if sample is not None else "",
                    "force_t_s": sample.t_s if sample is not None else "",
                    "load": load if load is not None else "",
                }
            )
            force_down = detector.update(load) if (detector is not None and load is not None) else False
            screen.fill(display.BACKGROUND)
            if phase == "bias":
                lines = (
                    f"Hands OFF the screen: measuring the zero level ({max(0.0, record_start - now):3.1f} s)",
                    f"Next: {args.label} for {args.seconds:g} s. ESC ends early.",
                )
            else:
                lines = (
                    f"{args.label}: {max(0.0, end - now):4.1f} s left     ESC ends early",
                    f"IR: {'DOWN' if pressed else 'up'}      force: {'DOWN' if force_down else 'up'}      "
                    f"load {load if load is not None else float('nan'):.4f} {unit}   (on at {detector.on:.4f})",
                    f"state: {contact_state(pressed, force_down)}",
                )
            for row_index, text in enumerate(lines):
                screen.blit(font.render(text, True, display.TEXT), (40, 40 + 40 * row_index))
            if phase == "record" and detector is not None and load is not None and detector.on > 0:
                width = min(1.0, load / (4 * detector.on)) * 600
                pygame.draw.rect(screen, display.MUTED, (40, 170, 600, 24), 2)
                pygame.draw.rect(screen, display.TOUCH_POINT if force_down else display.MUTED, (40, 170, width, 24))
            pygame.draw.circle(screen, display.TOUCH_POINT, pos, 10)
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()
        summary = reader.stop()
    if bias is None or detector is None:
        raise RuntimeError("the check ended before recording started")

    with stem.with_name(stem.name + "_frames.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    recorded = [row for row in rows if row["phase"] == "record" and row["load"] != ""]
    live = HysteresisContact(detector.on, detector.off)
    seconds = dict.fromkeys(STATES, 0.0)
    for a, b in zip(recorded, recorded[1:]):
        seconds[contact_state(bool(a["pressed"]), live.update(float(a["load"])))] += float(b["t_s"]) - float(a["t_s"])
    meta: dict[str, object] = {
        "label": args.label,
        "seconds": args.seconds,
        "bias_s": args.bias_s,
        "record_start_s": record_start,
        "source": "fake" if args.fake else {"device": args.device, "first_channel": args.first_channel, "terminal": args.terminal, "range_v": args.range_v},
        "calibration": asdict(calibration) if calibration is not None else None,
        "calibration_file": str(args.cal) if args.cal else None,
        "load_unit": unit,
        "bias_volts": list(bias),
        "noise": {"mean": limits.noise_mean, "sd": limits.noise_sd, "samples": len(bias_volts)},
        "thresholds": {"on": detector.on, "off": detector.off},
        "force_samples": summary.samples,
        "force_error": summary.error,
        "clock_offset_s": summary.clock_offset_s,
        "frames": len(rows),
        "live_seconds_by_state": {k: round(v, 3) for k, v in seconds.items()},
    }
    stem.with_name(stem.name + "_probe.json").write_text(json.dumps(meta, indent=2))
    meta["stem"] = str(stem)
    return meta


def _quietest(samples: list[tuple[float, ...]], share: float = 0.3) -> list[tuple[float, ...]]:
    """The run of consecutive samples, ``share`` of them, whose gauge voltages vary least."""
    n = max(2, int(len(samples) * share))
    def spread(start: int) -> float:
        chunk = samples[start : start + n]
        return sum(statistics.pstdev(v[c] for v in chunk) for c in range(len(chunk[0])))
    best = min(range(0, len(samples) - n + 1), key=spread)
    return samples[best : best + n]


def report(meta: dict[str, object]) -> str:
    noise = meta["noise"]
    limits = meta["thresholds"]
    by_state = meta["live_seconds_by_state"]
    return "\n".join(
        (
            f"{meta['label']}: {meta['frames']} frames, {meta['force_samples']} force samples"
            + (f"  FORCE ERROR: {meta['force_error']}" if meta["force_error"] else ""),
            f"Zero level noise: mean {noise['mean']:.5f}, SD {noise['sd']:.5f} {meta['load_unit']} "
            f"({noise['samples']} samples) -> on at {limits['on']:.5f}, off below {limits['off']:.5f}",
            "Seconds by state (IR / force): "
            + ", ".join(f"{state} {by_state[state]:.1f}" for state in STATES),
            f"Saved: {meta['stem']}_frames.csv / _force.sqlite3 / _probe.json",
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.contact_probe",
        description="Record the IR frame's contact and the force sensor's load side by side. No generator.",
    )
    parser.add_argument("--label", required=True, help="What this run is: still, slide, lifts, hover, palm, ...")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--bias-s", type=float, default=3.0, help="Hands-off time first, for the zero level.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--device", default="Dev1", help="NI-DAQmx device name, as NI MAX shows it.")
    parser.add_argument("--first-channel", type=int, default=0, help="First of the six analog inputs (ai<N>).")
    parser.add_argument("--terminal", default="diff", choices=("diff", "rse", "nrse", "default"))
    parser.add_argument("--range-v", type=float, default=10.0)
    parser.add_argument("--cal", type=Path, default=None, help="The sensor's ATI .cal file; without it, volts.")
    parser.add_argument("--on", type=float, default=None, help="Override the 'loaded' threshold.")
    parser.add_argument("--off", type=float, default=None, help="Override the 'unloaded' threshold.")
    parser.add_argument("--fake", action="store_true", help="Use the fake force source (no card).")
    parser.add_argument("--windowed", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    meta = run(parse_args(argv))
    print(report(meta))
    return 1 if meta["force_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
