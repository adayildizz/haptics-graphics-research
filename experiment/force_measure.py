"""Force sensor check on the console: the zero, the noise, live forces and the drift of the zero.

    python -m experiment.force_measure --label weights --cal FT05411.cal
    python -m experiment.force_measure --fake          (no card: the fake source, to try the tool)

The ATI Nano17 under the glass is read by ``force_daq.ForceReader``, a process of its own that stores every raw
sample (the six gauge voltages, 1000 per second) in its own SQLite file. This tool starts it once and walks
through three steps, saying on the console what to do:

1. Hands off the screen for ``--zero-s`` seconds. The zero (ATI calls it the bias: the gauge voltages with no
   load) is each gauge's mean over the quietest 0.5 s of that time, so hands that come back early do not spoil
   it. The spread of the values there is the noise.
2. Recording for ``--seconds``. A few times per second a line shows the press force Fn = |Fz| and the lateral
   force Ft = sqrt(Fx² + Fy²), averaged over the last 0.2 s so the numbers can be read, then Fx, Fy, Fz with
   their signs. A known weight on the glass checks them: 50 g presses with 0.49 N.
3. Hands off again: a second zero. End zero minus start zero is how far the zero drifted during the run.

Forces need the sensor's calibration file (``--cal``). Without it every value is the change of the six gauge
voltages from their zero, in volts: it rises with any load, but it is not a force.

Files in ``--data-dir``, named ``force_<label>_<YYYYmmdd_HHMMSS>``:

- ``_force.sqlite3``: every raw sample, exactly as the reader wrote it (``force_daq.load_force`` reads it back);
- ``_summary.json``: the settings, the calibration file, both zeros (volts), the noise, the drift, the start
  and end of each step on the samples' clock, the sample count and any reader error.

The samples stay raw volts; a calibration is applied when they are read back.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import statistics
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ati_calibration import AXES, AtiCalibration, load_ati_calibration
from .data_logger import DATA_DIR
from .force_daq import (
    CHANNELS,
    RATE_HZ,
    ForceReader,
    ForceRecording,
    ForceSample,
    add_card_arguments,
    load_force,
    source_factory,
)

QUIET_WINDOW_S = 0.5  # each zero comes from the quietest stretch of this length in its hands-off step
LIVE_AVERAGE_S = 0.2  # each live line is the mean of the samples of this last stretch
PRINT_EVERY_S = 0.25  # a live line four times per second
POLL_S = 0.005  # how often the newest sample is fetched while recording
FILE_WAIT_S = 2.0  # the reader writes to disk every 0.25 s; the first zero waits at most this long for it
GAUGES = tuple(f"g{k}" for k in range(CHANNELS))  # names of the six gauge voltages, used without a calibration


class ReaderStopped(Exception):
    """The reader process ended during a step; ``ForceReader.stop`` then reports why."""


# --- computations: pure functions, tested without a card ---------------------------------------


def quietest_window(volts: Sequence[Sequence[float]], rate_hz: float, window_s: float = QUIET_WINDOW_S) -> slice:
    """The stretch of ``window_s`` in which the gauge voltages vary least.

    "Vary" is the sum of the six gauges' standard deviations. A hand on the glass, or one being taken away,
    makes the voltages swing, so the quietest stretch is the one least touched by hands. With fewer samples
    than the window holds, all of them are used.
    """
    volts = np.asarray(volts, dtype=float)
    if len(volts) < 2:
        raise ValueError("at least two samples are needed to find the quietest stretch")
    n = min(len(volts), max(2, round(window_s * rate_hz)))
    stride = max(1, n // 50)  # a candidate start every 10 ms at 1 kHz: fine enough, and quick
    best = min(range(0, len(volts) - n + 1, stride), key=lambda start: volts[start : start + n].std(axis=0).sum())
    return slice(best, best + n)


def reading_names(calibration: AtiCalibration | None) -> tuple[str, ...]:
    """Fx, Fy, Fz, Tx, Ty, Tz with a calibration; g0 to g5 (the gauges) without."""
    return AXES if calibration is not None else GAUGES


def units(calibration: AtiCalibration | None) -> dict[str, str]:
    """The unit of every reading (and of Ft), as the calibration file states them; volts without one."""
    if calibration is None:
        return dict.fromkeys(GAUGES, "V")
    force, torque = calibration.force_units, calibration.torque_units
    return {**dict.fromkeys(AXES[:3], force), **dict.fromkeys(AXES[3:], torque), "Ft": force}


def readings(volts: Sequence[float], zero: Sequence[float], calibration: AtiCalibration | None) -> tuple[float, ...]:
    """One sample as six readings: Fx..Tz with a calibration, else each gauge's voltage minus its zero."""
    if calibration is None:
        return tuple(float(v - z) for v, z in zip(volts, zero))
    return calibration.apply(volts, zero)


def press_and_lateral(fx: float, fy: float, fz: float) -> tuple[float, float]:
    """Fn = |Fz|, how hard the glass is pressed down, and Ft = sqrt(Fx² + Fy²), how hard it is pushed sideways."""
    return abs(fz), math.hypot(fx, fy)


@dataclass(frozen=True)
class Zero:
    """The unloaded gauge voltages of one hands-off step and how much the readings wobbled there."""

    volts: tuple[float, ...]  # each gauge's mean over the quietest stretch: forces are measured from these
    noise_sd: dict[str, float]  # SD of each reading over that stretch (and of Ft, with a calibration)
    quiet_start_s: float  # first and last sample of the quietest stretch, on the samples' clock
    quiet_end_s: float
    samples: int


def measure_zero(
    t_s: Sequence[float],
    volts: Sequence[Sequence[float]],
    start_s: float,
    end_s: float,
    rate_hz: float,
    calibration: AtiCalibration | None,
) -> Zero:
    """The zero of the hands-off step from ``start_s`` to ``end_s``, from the quietest 0.5 s of it."""
    first, stop = bisect.bisect_left(t_s, start_s), bisect.bisect_right(t_s, end_s)  # samples are in time order
    if stop - first < 2:
        raise ValueError(f"only {stop - first} force samples between {start_s:.3f} s and {end_s:.3f} s")
    phase = np.asarray(volts[first:stop], dtype=float)
    quiet = quietest_window(phase, rate_hz)
    zero = tuple(float(v) for v in phase[quiet].mean(axis=0))
    values = np.array([readings(v, zero, calibration) for v in phase[quiet]])
    noise = {name: float(values[:, k].std(ddof=1)) for k, name in enumerate(reading_names(calibration))}
    if calibration is not None:
        noise["Ft"] = float(np.hypot(values[:, 0], values[:, 1]).std(ddof=1))
    return Zero(zero, noise, t_s[first + quiet.start], t_s[first + quiet.stop - 1], quiet.stop - quiet.start)


def drift(start: Zero, end: Zero, calibration: AtiCalibration | None) -> dict[str, float]:
    """How far the zero moved from the first hands-off step to the last (end minus start), per reading.

    A calibration is linear, so the drift in N is the calibration applied to the change of the gauge voltages.
    """
    return dict(zip(reading_names(calibration), readings(end.volts, start.volts, calibration)))


def mean_volts(samples: Sequence[ForceSample]) -> tuple[float, ...]:
    """Each gauge's mean over ``samples``."""
    return tuple(statistics.fmean(gauge) for gauge in zip(*(sample.volts for sample in samples)))


def live_line(elapsed_s: float, volts: Sequence[float], zero: Sequence[float], calibration: AtiCalibration | None) -> str:
    """One line of the live display for the (averaged) gauge voltages ``volts``."""
    values = readings(volts, zero, calibration)
    if calibration is None:
        change = math.sqrt(sum(v * v for v in values))
        return f"{elapsed_s:6.2f} s   change {change:.4f} V   ({_listing(dict(zip(GAUGES, values)), 4)} V)"
    fx, fy, fz = values[:3]
    fn, ft = press_and_lateral(fx, fy, fz)
    u = calibration.force_units
    return f"{elapsed_s:6.2f} s   Fn {fn:6.3f} {u}   Ft {ft:6.3f} {u}   (Fx {fx:+.3f}  Fy {fy:+.3f}  Fz {fz:+.3f} {u})"


def _listing(values: dict[str, float], digits: int, signed: bool = True) -> str:
    sign = "+" if signed else ""
    return "  ".join(f"{name} {value:{sign}.{digits}f}" for name, value in values.items())


# --- the three steps ---------------------------------------------------------------------------


def _check(reader: ForceReader) -> None:
    if not reader.running:
        raise ReaderStopped("the force reader process has ended")


def hands_off(reader: ForceReader, seconds: float, title: str) -> tuple[float, float]:
    """A step without hands on the screen, with a countdown; returns its start and end time."""
    print(f"\n{title}: hands OFF the screen for {seconds:g} s ...", end="", flush=True)
    start = time.perf_counter()
    end = start + seconds
    shown = None
    while (now := time.perf_counter()) < end:
        _check(reader)
        left = math.ceil(end - now)
        if left != shown:
            print(f" {left}", end="", flush=True)
            shown = left
        time.sleep(min(0.05, end - now))
    print(" done.", flush=True)
    return start, now


def record(reader: ForceReader, seconds: float, zero: Zero, calibration: AtiCalibration | None) -> tuple[float, float]:
    """The recording step: a live line every PRINT_EVERY_S; returns its start and end time.

    Each line averages the newest samples seen during the last LIVE_AVERAGE_S. On Windows a short sleep can
    last about 15 ms, so a line may average only a dozen samples instead of 200: enough to read the number.
    The force file keeps every sample regardless.
    """
    start = time.perf_counter()
    end = start + seconds
    recent: deque[ForceSample] = deque()  # samples seen lately, oldest first
    next_line = start + PRINT_EVERY_S
    while (now := time.perf_counter()) < end:
        _check(reader)
        sample = reader.latest()
        if sample is not None and (not recent or sample.index != recent[-1].index):
            recent.append(sample)
        while recent and recent[0].t_s < now - LIVE_AVERAGE_S:
            recent.popleft()
        if now >= next_line:
            next_line += PRINT_EVERY_S
            if recent:
                print("  " + live_line(now - start, mean_volts(recent), zero.volts, calibration), flush=True)
        time.sleep(POLL_S)
    return start, now


def read_back(path: Path, until_s: float, timeout_s: float = FILE_WAIT_S) -> ForceRecording:
    """The force file as written so far, once it holds the samples up to ``until_s``.

    The reader process writes to disk every quarter second, so the newest samples reach the file a moment late.
    """
    give_up_s = time.perf_counter() + timeout_s
    while True:
        try:
            recording = load_force(path)
            if recording.t_s and recording.t_s[-1] >= until_s:
                return recording
        except sqlite3.OperationalError:  # the file is not set up yet, or busy for a moment
            pass
        if time.perf_counter() > give_up_s:
            raise ReaderStopped(f"the force file had no samples up to {until_s:.3f} s after waiting {timeout_s:g} s")
        time.sleep(0.05)


def _print_zero(zero: Zero, calibration: AtiCalibration | None) -> None:
    print(f"  Zero (quietest {QUIET_WINDOW_S:g} s, {zero.samples} samples): {_listing(dict(zip(GAUGES, zero.volts)), 5)} V")
    unit = units(calibration)
    shown = ("Fz", "Ft") if calibration is not None else GAUGES
    noise = {name: zero.noise_sd[name] for name in shown}
    print(f"  Noise there (SD): {_listing(noise, 4, signed=False)} {unit[shown[0]]}")


def _print_header(args: argparse.Namespace, calibration: AtiCalibration | None) -> None:
    if args.fake:
        print(f'Force sensor check "{args.label}" with the FAKE force source (no card).')
    else:
        last = args.first_channel + CHANNELS - 1
        print(
            f'Force sensor check "{args.label}": card {args.device}, inputs ai{args.first_channel}-ai{last}, '
            f"{args.terminal}, +-{args.range_v:g} V."
        )
    if calibration is None:
        print("No calibration file (--cal): every value below is UNCALIBRATED gauge volts, not a force.")
    else:
        print(
            f"Calibration {Path(args.cal).name}: sensor {calibration.serial}, {calibration.part_number}, "
            f"forces in {calibration.force_units}."
        )


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    card = {"device": args.device, "first_channel": args.first_channel, "terminal": args.terminal, "range_v": args.range_v}
    return {
        "seconds": args.seconds,
        "zero_s": args.zero_s,
        "source": "fake" if args.fake else "nidaqmx",
        "card": None if args.fake else card,
        "rate_hz": RATE_HZ,
        "quiet_window_s": QUIET_WINDOW_S,
        "live_average_s": LIVE_AVERAGE_S,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Starts the reader, runs the three steps, stops the reader, and writes and returns the summary."""
    calibration = load_ati_calibration(args.cal) if args.cal else None
    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"force_{args.label}_{time.strftime('%Y%m%d_%H%M%S')}"
    force_path = out_dir / f"{stem}_force.sqlite3"
    summary_path = out_dir / f"{stem}_summary.json"
    phases: dict[str, dict[str, float]] = {}  # start and end of each step, on the samples' clock
    summary: dict[str, Any] = {
        "label": args.label,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": _settings(args),
        "calibration": None,
        "units": units(calibration),
        "force_file": force_path.name,
        "phases": phases,
        "zero_start": None,
        "zero_end": None,
        "drift": None,
        "force_samples": 0,
        "force_reads": 0,
        "clock_offset_s": None,
        "reader_error": None,
    }

    if calibration is not None:
        summary["calibration"] = {
            "file": Path(args.cal).name,
            "serial": calibration.serial,
            "part_number": calibration.part_number,
            "force_units": calibration.force_units,
            "torque_units": calibration.torque_units,
        }
    _print_header(args, calibration)

    reader = ForceReader(force_path, source_factory(args, fake=args.fake))
    print("Starting the force reader ...", end=" ", flush=True)
    try:
        reader.start()  # if it fails, it has already stopped its process
    except RuntimeError as exc:
        print("it could not start.")
        summary["reader_error"] = str(exc)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\nFORCE READER ERROR:\n{exc}")
        return summary
    print("running.")

    start_zero: Zero | None = None
    try:
        start_s, end_s = hands_off(reader, args.zero_s, "Step 1 of 3")
        phases["zero_start"] = {"start_s": start_s, "end_s": end_s}
        so_far = read_back(force_path, end_s)
        start_zero = measure_zero(so_far.t_s, so_far.volts, start_s, end_s, float(so_far.metadata["rate_hz"]), calibration)
        summary["zero_start"] = asdict(start_zero)
        _print_zero(start_zero, calibration)

        print(f"\nStep 2 of 3: recording for {args.seconds:g} s. Press the glass, or put a known weight on it.")
        if calibration is not None:
            print(
                f"  Fn = |Fz| presses down, Ft = sqrt(Fx^2 + Fy^2) pushes sideways (50 g = 0.49 N); "
                f"each line is the mean of the last {LIVE_AVERAGE_S:g} s."
            )
        else:
            print(
                f"  'change' is the size of the six gauges' change from the zero (uncalibrated volts); "
                f"each line is the mean of the last {LIVE_AVERAGE_S:g} s."
            )
        start_s, end_s = record(reader, args.seconds, start_zero, calibration)
        phases["record"] = {"start_s": start_s, "end_s": end_s}

        start_s, end_s = hands_off(reader, args.zero_s, "Step 3 of 3")
        phases["zero_end"] = {"start_s": start_s, "end_s": end_s}
    except ReaderStopped as exc:
        print(f"\nStopped before the end: {exc}.")
    finally:
        stats = reader.stop()  # waits until every sample is on disk
    summary.update(
        force_samples=stats.samples, force_reads=stats.reads, clock_offset_s=stats.clock_offset_s, reader_error=stats.error
    )

    if start_zero is not None and "zero_end" in phases:
        recording = load_force(force_path)
        end = phases["zero_end"]
        end_zero = measure_zero(
            recording.t_s, recording.volts, end["start_s"], end["end_s"], float(recording.metadata["rate_hz"]), calibration
        )
        summary["zero_end"] = asdict(end_zero)
        summary["drift"] = drift(start_zero, end_zero, calibration)
        _print_zero(end_zero, calibration)
        unit = units(calibration)
        shown = AXES[:3] if calibration is not None else GAUGES
        moved = {name: summary["drift"][name] for name in shown}
        print(f"  Drift of the zero over the run (end minus start): {_listing(moved, 4)} {unit[shown[0]]}")

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {force_path} ({stats.samples} samples)\n  and {summary_path}")
    if stats.error is not None:
        print(f"\nFORCE READER ERROR:\n{stats.error}")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.force_measure",
        description="Check the force sensor on the console: the zero and its noise with hands off, live forces "
        "while recording, then the drift of the zero. No generator, no window.",
    )
    parser.add_argument("--label", default="check", help="What this run is; part of the file names (default check).")
    parser.add_argument("--seconds", type=float, default=30.0, help="How long to record (default 30).")
    parser.add_argument("--zero-s", type=float, default=3.0, help="Length of each hands-off step (default 3).")
    parser.add_argument("--cal", type=Path, help="The sensor's ATI calibration file (.cal); without it, volts.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help=f"Where the files go (default {DATA_DIR}).")
    parser.add_argument("--fake", action="store_true", help="Use the fake force source: no card is opened.")
    add_card_arguments(parser)
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.zero_s < QUIET_WINDOW_S:
        parser.error(f"--zero-s must be at least {QUIET_WINDOW_S:g} s, the stretch the zero is taken from")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    return 0 if summary["reader_error"] is None and summary["drift"] is not None else 1


if __name__ == "__main__":  # needed: on Windows the reader process imports this module again and must not run it
    raise SystemExit(main())
