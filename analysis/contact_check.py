"""IR contact against force contact, from ``experiment.contact_probe`` recordings.

    python -m analysis.contact_check FOLDER_OR_FILES [--on X --off Y] [--hold-ms 150] [--jump-mm 20] [--out DIR]

For every run it measures the unloaded noise from all 1 kHz samples of the hands-off phase, sets the force
thresholds from it (unless given), and then, frame by frame, puts each moment of the recording in one of the
four states of ``experiment.contact_fusion``. It reports:

- how long the recording spent in each state, and how long each episode of a state lasted;
- every IR lift, sorted into false (the force stayed loaded the whole time the IR said up) and real (the force
  let go at some point of it; the two sources rarely switch in the same frame) ones;
- every IR position jump longer than ``--jump-mm`` between two frames with contact, and whether the force
  stayed loaded across it;
- the load while both sources say down, against the threshold: the margin for the lightest touches;
- what ``ContactFusion`` with ``--hold-ms`` and ``--jump-mm`` would have done: IR-up time it bridges, dropouts
  longer than the hold it gives up on, jumps it holds back.

With ``--out`` it writes one CSV of episodes per run and a figure: the load over time, the thresholds, and the
IR contact underneath.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiment.ati_calibration import AtiCalibration
from experiment.contact_fusion import (
    STATES,
    ContactFusion,
    HysteresisContact,
    contact_state,
    episodes,
    load_of,
    thresholds_from_noise,
)
from experiment.force_daq import load_force


@dataclass
class Run:
    stem: Path
    meta: dict
    frame_t: np.ndarray
    pressed: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray
    force_t: np.ndarray
    load: np.ndarray  # one per force sample


def find_runs(paths: Sequence[Path]) -> list[Path]:
    """Stems (path without the ``_probe.json`` suffix) of every probe run in the given files or folders."""
    stems: list[Path] = []
    for path in paths:
        candidates = sorted(path.glob("*_probe.json")) if path.is_dir() else [path]
        for candidate in candidates:
            name = candidate.name
            for suffix in ("_probe.json", "_frames.csv", "_force.sqlite3"):
                if name.endswith(suffix):
                    stem = candidate.with_name(name[: -len(suffix)])
                    if stem not in stems:
                        stems.append(stem)
    return stems


def load_run(stem: Path) -> Run:
    meta = json.loads(stem.with_name(stem.name + "_probe.json").read_text())
    calibration = None
    if meta.get("calibration"):
        c = meta["calibration"]
        calibration = AtiCalibration(
            c["serial"], c["part_number"], c["force_units"], c["torque_units"],
            tuple(tuple(row) for row in c["matrix"]), tuple(c["maxima"]),
        )  # fmt: skip
    with stem.with_name(stem.name + "_frames.csv").open() as file:
        frames = [row for row in csv.DictReader(file) if row["phase"] == "record"]
    recording = load_force(stem.with_name(stem.name + "_force.sqlite3"))
    bias = meta["bias_volts"]
    return Run(
        stem,
        meta,
        np.array([float(r["t_s"]) for r in frames]),
        np.array([r["pressed"] == "1" for r in frames]),
        np.array([float(r["x_mm"]) for r in frames]),
        np.array([float(r["y_mm"]) for r in frames]),
        np.array(recording.t_s),
        np.array([load_of(v, bias, calibration) for v in recording.volts]),
    )


def force_down_at_frames(run: Run, on: float, off: float) -> np.ndarray:
    """The hysteresis contact run over every force sample, read at each frame's time."""
    detector = HysteresisContact(on, off)
    states = np.array([detector.update(float(v)) for v in run.load])
    index = np.searchsorted(run.force_t, run.frame_t, side="right") - 1
    return np.where(index >= 0, states[np.clip(index, 0, None)], False)


def _stats_ms(durations_s: Sequence[float]) -> dict[str, float]:
    if not durations_s:
        return {"n": 0, "median_ms": 0.0, "p90_ms": 0.0, "max_ms": 0.0}
    ms = 1000.0 * np.asarray(durations_s)
    return {"n": len(ms), "median_ms": float(np.median(ms)), "p90_ms": float(np.percentile(ms, 90)), "max_ms": float(ms.max())}


def analyse(run: Run, on: float | None = None, off: float | None = None, hold_ms: float = 150.0, jump_mm: float = 20.0) -> dict:
    unloaded = run.load[run.force_t < run.meta["record_start_s"]]
    limits = thresholds_from_noise(unloaded.tolist())
    on = limits.on if on is None else on
    off = limits.off if off is None else off
    force = force_down_at_frames(run, on, off)
    t = run.frame_t
    states = [contact_state(bool(i), bool(f)) for i, f in zip(run.pressed, force)]
    runs = episodes(t.tolist(), states)
    total = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    seconds = {s: sum(e.duration_s for e in runs if e.state == s) for s in STATES}

    # IR lifts: each stretch with IR up between two stretches with IR down.
    lifts = {"false": [], "real": []}
    k = 1
    while k < len(t):
        if run.pressed[k - 1] and not run.pressed[k]:
            j = k
            while j < len(t) and not run.pressed[j]:
                j += 1
            if j < len(t):
                held = force[k:j]
                kind = "false" if held.all() else "real"
                lifts[kind].append(float(t[j] - t[k]))
            k = j
        k += 1

    jumps = {"force_held": 0, "force_released": 0}
    for k in range(1, len(t)):
        if run.pressed[k] and run.pressed[k - 1] and np.hypot(run.x_mm[k] - run.x_mm[k - 1], run.y_mm[k] - run.y_mm[k - 1]) > jump_mm:
            jumps["force_held" if force[k] and force[k - 1] else "force_released"] += 1

    fusion = ContactFusion(hold_s=hold_ms / 1000.0, jump_mm=jump_mm)
    fused = [fusion.update(float(tt), bool(i), bool(f), float(x), float(y)) for tt, i, f, x, y in zip(t, run.pressed, force, run.x_mm, run.y_mm)]
    dt = np.diff(t, append=t[-1])
    bridged_s = float(sum(d for d, frame, ir in zip(dt, fused, run.pressed) if frame.render_down and not ir))
    gave_up = sum(1 for e in runs if e.state == "force_only" and e.duration_s > hold_ms / 1000.0)
    both_load = run.load[np.isin(np.searchsorted(t, run.force_t, side="right") - 1, np.flatnonzero(np.array(states) == "both"))]
    return {
        "label": run.meta["label"],
        "stem": str(run.stem),
        "unit": run.meta["load_unit"],
        "recorded_s": total,
        "noise": {"mean": limits.noise_mean, "sd": limits.noise_sd, "samples": int(len(unloaded))},
        "thresholds": {"on": on, "off": off},
        "seconds_by_state": seconds,
        "episodes_by_state": {s: _stats_ms([e.duration_s for e in runs if e.state == s]) for s in STATES},
        "ir_lifts": {kind: _stats_ms(values) for kind, values in lifts.items()},
        "ir_jumps": jumps,
        "load_while_both": {
            "median": float(np.median(both_load)) if len(both_load) else float("nan"),
            "p5": float(np.percentile(both_load, 5)) if len(both_load) else float("nan"),
        },
        "fusion": {
            "hold_ms": hold_ms,
            "jump_mm": jump_mm,
            "ir_up_time_bridged_s": bridged_s,
            "force_only_longer_than_hold": gave_up,
            "jumps_held_back": sum(frame.teleport for frame in fused),
        },
        "episodes": runs,
    }


def report(result: dict) -> str:
    unit = result["unit"]
    noise, limits = result["noise"], result["thresholds"]
    lines = [
        f"{result['label']}  ({result['recorded_s']:.1f} s)  {Path(result['stem']).name}",
        f"  noise {noise['mean']:.5f} +- {noise['sd']:.5f} {unit} ({noise['samples']} samples); on {limits['on']:.5f}, off {limits['off']:.5f}",
        "  seconds:  " + "  ".join(f"{s} {result['seconds_by_state'][s]:.1f}" for s in STATES),
    ]
    for state in STATES:
        e = result["episodes_by_state"][state]
        if e["n"]:
            lines.append(f"  {state:10s} episodes {e['n']:3d}  median {e['median_ms']:6.0f} ms  p90 {e['p90_ms']:6.0f}  max {e['max_ms']:6.0f}")
    lifts = result["ir_lifts"]
    lines.append(
        f"  IR lifts: false {lifts['false']['n']} (median {lifts['false']['median_ms']:.0f} ms, max {lifts['false']['max_ms']:.0f}), "
        f"real {lifts['real']['n']}"
    )
    jumps = result["ir_jumps"]
    lines.append(f"  IR jumps with contact: {jumps['force_held']} with the force held, {jumps['force_released']} with it released")
    load = result["load_while_both"]
    margin = load["p5"] / limits["on"] if limits["on"] > 0 else float("nan")
    lines.append(f"  load while both down: median {load['median']:.4f}, lightest 5% {load['p5']:.4f} {unit} ({margin:.1f} x the on threshold)")
    f = result["fusion"]
    lines.append(
        f"  fusion (hold {f['hold_ms']:g} ms, jump {f['jump_mm']:g} mm): bridges {f['ir_up_time_bridged_s']:.2f} s of IR-up time, "
        f"gives up on {f['force_only_longer_than_hold']} longer dropout(s), holds back {f['jumps_held_back']} jump frame(s)"
    )
    return "\n".join(lines)


def write_outputs(run: Run, result: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(result["stem"]).name
    with (out_dir / f"{name}_episodes.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["state", "start_s", "end_s", "duration_ms"])
        t0 = run.frame_t[0]
        for e in result["episodes"]:
            writer.writerow([e.state, f"{e.start_s - t0:.4f}", f"{e.end_s - t0:.4f}", f"{1000 * e.duration_s:.1f}"])
    fig, ax = plt.subplots(figsize=(14, 4))
    t0 = run.frame_t[0]
    ax.plot(run.force_t - t0, run.load, color="#2a78d6", linewidth=0.6, label=f"load ({result['unit']})")
    ax.axhline(result["thresholds"]["on"], color="#e34948", linestyle="--", linewidth=0.8, label="on")
    ax.axhline(result["thresholds"]["off"], color="#e34948", linestyle=":", linewidth=0.8, label="off")
    top = float(np.nanmax(run.load)) if len(run.load) else 1.0
    for start, stop in _true_stretches(run.frame_t - t0, run.pressed):
        ax.axvspan(start, stop, ymin=0, ymax=0.04, color="#1baf7a", linewidth=0)
    ax.set_xlim(left=min(0.0, float(run.force_t[0] - t0)))
    ax.set_ylim(bottom=0, top=top * 1.05 if top > 0 else 1.0)
    ax.set_xlabel("time from the start of recording (s); green strip at the bottom = IR says down")
    ax.set_title(f"{result['label']}: force load and IR contact")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_load.png", dpi=120)
    plt.close(fig)


def _true_stretches(t: np.ndarray, flags: np.ndarray) -> list[tuple[float, float]]:
    stretches, start = [], None
    for k, flag in enumerate(flags):
        if flag and start is None:
            start = t[k]
        elif not flag and start is not None:
            stretches.append((float(start), float(t[k])))
            start = None
    if start is not None:
        stretches.append((float(start), float(t[-1])))
    return stretches


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m analysis.contact_check", description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", type=Path, nargs="+", help="Probe files or folders holding them.")
    parser.add_argument("--on", type=float, default=None)
    parser.add_argument("--off", type=float, default=None)
    parser.add_argument("--hold-ms", type=float, default=150.0)
    parser.add_argument("--jump-mm", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=None, help="Folder for the episode CSVs and load figures.")
    args = parser.parse_args(argv)
    stems = find_runs(args.paths)
    if not stems:
        print("no contact_probe runs found")
        return 2
    for stem in stems:
        run = load_run(stem)
        result = analyse(run, args.on, args.off, args.hold_ms, args.jump_mm)
        print(report(result))
        print()
        if args.out is not None:
            write_outputs(run, result, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
