from __future__ import annotations

import csv
import json
import sqlite3

import numpy as np
import pytest

from analysis.contact_check import analyse, find_runs, load_run, main
from experiment.contact_probe import FRAME_FIELDS
from experiment.force_daq import _create_schema

RATE = 1000.0
T0 = 100.0  # perf_counter at the start of the hands-off phase
BIAS_S = 1.0


def _load_at(t: float) -> float:
    """The scripted press, in volts above the zero: on 1.0-2.5 and 3.0-4.0, off otherwise."""
    pressed = (1.0 <= t < 2.5) or (3.0 <= t < 4.0)
    return 0.5 if pressed else 0.0


def _ir_at(t: float) -> tuple[bool, float, float]:
    """IR contact and position: a false lift at 1.5-1.6 (force stays), a real lift with the force at 2.5-3.0,
    a ghost jump at 3.5 (one frame 40 mm away, force held), and hovering in the beams at 4.2-4.5."""
    if 1.5 <= t < 1.6:
        return False, 80.0, 100.0  # cursor frozen
    if 1.0 <= t < 2.5 or 3.0 <= t < 4.0:
        x = 80.0 + 10.0 * (t - 1.0)
        if 3.5 <= t < 3.5 + 1 / 60:
            x += 40.0
        return True, x, 100.0
    if 4.2 <= t < 4.5:
        return True, 120.0, 60.0
    return False, 80.0, 100.0


def _write_run(folder, rng):
    stem = folder / "contact_test_20261030_120000"
    n = int((BIAS_S + 5.0) * RATE)
    noise = rng.normal(0.0, 0.002, size=(n, 6))
    rows = []
    for k in range(n):
        t = k / RATE - BIAS_S  # seconds from the start of recording
        load = _load_at(t)
        rows.append((k, *(0.1 + load * (1.0 if c == 2 else 0.0) + noise[k, c] for c in range(6))))
    connection = sqlite3.connect(stem.with_name(stem.name + "_force.sqlite3"))
    _create_schema(connection, 6)
    connection.executemany("INSERT INTO force_metadata VALUES (?, ?)", [("rate_hz", str(RATE)), ("channels", "6")])
    connection.executemany(f"INSERT INTO force_samples VALUES ({', '.join('?' * 7)})", rows)
    connection.executemany(
        "INSERT INTO force_reads VALUES (?, ?)", [(total, T0 + (total - 1) / RATE + 0.0005) for total in range(2, n + 1, 2)]
    )
    connection.commit()
    connection.close()
    record_start = T0 + BIAS_S
    with stem.with_name(stem.name + "_frames.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        for k in range(int(5.0 * 60)):
            t = k / 60.0
            down, x, y = _ir_at(t)
            writer.writerow({"t_s": record_start + t, "phase": "record", "x_px": 0, "y_px": 0, "x_mm": x, "y_mm": y,
                             "pressed": int(down), "load": ""})  # fmt: skip
    meta = {"label": "test", "record_start_s": record_start, "bias_volts": [0.1] * 6, "calibration": None,
            "load_unit": "V (uncalibrated)"}  # fmt: skip
    stem.with_name(stem.name + "_probe.json").write_text(json.dumps(meta))
    return stem


def test_the_check_sorts_ir_lifts_and_jumps_by_what_the_force_did(tmp_path):
    stem = _write_run(tmp_path, np.random.default_rng(3))
    assert find_runs([tmp_path]) == [stem]
    result = analyse(load_run(stem), hold_ms=150.0, jump_mm=20.0)
    assert result["noise"]["samples"] == 500  # the quietest half second of the hands-off phase
    assert 0.0 < result["thresholds"]["off"] < result["thresholds"]["on"] < 0.5
    lifts = result["ir_lifts"]
    assert lifts["false"]["n"] == 1 and lifts["false"]["median_ms"] == pytest.approx(100, abs=20)
    assert lifts["real"]["n"] == 2  # 2.5-3.0, and 4.0-4.2 before the finger hovers in the beams
    assert result["ir_jumps"]["force_held"] == 2  # out to the ghost and back
    assert result["seconds_by_state"]["ir_only"] == pytest.approx(0.3, abs=0.05)
    fusion = result["fusion"]
    # The false lift (0.1 s), plus the frame at each real lift where the IR is already up and the force not yet down.
    assert fusion["ir_up_time_bridged_s"] == pytest.approx(0.1 + 2 / 60, abs=0.01)
    assert fusion["force_only_longer_than_hold"] == 0 and fusion["jumps_held_back"] == 1
    assert result["load_while_both"]["p5"] > result["thresholds"]["on"]


def test_the_command_line_writes_episodes_and_a_figure(tmp_path, capsys):
    _write_run(tmp_path, np.random.default_rng(4))
    out = tmp_path / "out"
    assert main([str(tmp_path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "IR lifts: false 1" in text and "fusion (hold 150 ms" in text
    assert len(list(out.glob("*_episodes.csv"))) == 1 and len(list(out.glob("*_load.png"))) == 1
    assert main([str(tmp_path / "nothing_here")]) == 2


def test_the_probe_records_both_sources_with_the_fake_card(monkeypatch, tmp_path):
    from experiment import contact_probe

    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    args = contact_probe.parse_args(
        ["--label", "smoke", "--seconds", "0.4", "--bias-s", "0.3", "--fake", "--windowed", "--data-dir", str(tmp_path)]
    )
    meta = contact_probe.run(args)
    assert meta["force_samples"] > 300 and meta["force_error"] is None
    (stem,) = find_runs([tmp_path])
    assert load_run(stem).frame_t.size > 5
    assert "Seconds by state" in contact_probe.report(meta)


def test_the_zero_comes_from_the_quietest_part_of_the_hands_off_phase(tmp_path):
    from analysis.contact_check import quietest_window

    rng = np.random.default_rng(5)
    volts = np.r_[rng.normal(0, 0.003, (500, 6)), rng.normal(0.4, 0.2, (500, 6))]  # a hand comes back early
    window = quietest_window(volts, RATE)
    assert window.stop <= 550 and window.stop - window.start == 500

