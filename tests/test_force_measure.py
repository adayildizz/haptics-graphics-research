from __future__ import annotations

import json
import random
import re
from functools import partial

import numpy as np
import pytest

from experiment import force_measure
from experiment.ati_calibration import AXES, load_ati_calibration
from experiment.force_daq import FakeForceSource, ForceSample, fake_volts, load_force
from experiment import stimulus
from experiment.force_measure import (
    Zero,
    drift,
    ea_on_at,
    live_line,
    mean_volts,
    measure_zero,
    press_and_lateral,
    quietest_window,
    readings,
    units,
)
from tests.test_ati_calibration import CAL  # Fz = 0.5 g0 + 0.5 g1 + 2 g2; Fx = g0; Fy = g1

SUMMARY_KEYS = {
    "label", "started", "settings", "calibration", "units", "force_file", "phases", "zero_start", "zero_end",
    "drift", "force_samples", "force_reads", "clock_offset_s", "reader_error", "ea", "interrupted",
}  # fmt: skip


@pytest.fixture
def cal_path(tmp_path):
    path = tmp_path / "FT99999.cal"
    path.write_text(CAL)
    return path


def quiet_volts(n: int, bias, sd: float, seed: int = 1) -> list[tuple[float, ...]]:
    rng = random.Random(seed)
    return [tuple(b + rng.gauss(0.0, sd) for b in bias) for _ in range(n)]


# --- quietest window ---------------------------------------------------------------------------


def test_the_quietest_window_avoids_hands_that_leave_late_or_come_back_early():
    rng = random.Random(2)
    hands = [tuple(rng.uniform(-0.5, 0.5) for _ in range(6)) for _ in range(400)]  # still on the glass
    quiet = quiet_volts(900, [0.1] * 6, 0.009)  # 0.4 s to 1.3 s: nobody touches
    back = [tuple(0.3 + rng.uniform(-0.2, 0.2) for _ in range(6)) for _ in range(1700)]  # back early
    window = quietest_window(hands + quiet + back, 1000.0)
    assert window.stop - window.start == 500
    assert 400 <= window.start and window.stop <= 1300


def test_with_fewer_samples_than_the_window_all_are_used_and_one_sample_is_not_enough():
    assert quietest_window(quiet_volts(120, [0.0] * 6, 0.01), 1000.0) == slice(0, 120)
    with pytest.raises(ValueError):
        quietest_window([(0.0,) * 6], 1000.0)


# --- volts to forces ---------------------------------------------------------------------------


def test_volts_become_newtons_through_the_calibration(cal_path):
    cal = load_ati_calibration(cal_path)
    fx, fy, fz, *_ = readings([1.1, 0.2, 0.3, 0, 0, 0], [0.1, 0.2, 0.1, 0, 0, 0], cal)
    assert (fx, fy, fz) == pytest.approx((1.0, 0.0, 0.5 * 1.0 + 0.5 * 0.0 + 2 * 0.2))
    assert press_and_lateral(3.0, -4.0, -0.49) == pytest.approx((0.49, 5.0))
    line = live_line(1.5, [1.1, 0.2, 0.3, 0, 0, 0], [0.1, 0.2, 0.1, 0, 0, 0], cal)
    assert "Fn  0.900 N" in line and "Ft  1.000 N" in line and "Fz +0.900 N" in line
    assert units(cal) == {"Fx": "N", "Fy": "N", "Fz": "N", "Tx": "N-mm", "Ty": "N-mm", "Tz": "N-mm", "Ft": "N"}


def test_without_a_calibration_every_value_is_a_gauge_voltage_minus_its_zero():
    assert readings([0.5, 0.2, 0, 0, 0, 0], [0.1, 0.2, 0, 0, 0, 0], None) == pytest.approx((0.4, 0, 0, 0, 0, 0))
    assert units(None) == dict.fromkeys(("g0", "g1", "g2", "g3", "g4", "g5"), "V")
    line = live_line(1.5, [0.4, 0.3, 0, 0, 0, 0], [0.0] * 6, None)
    assert "change 0.5000 V" in line and "g0 +0.4000" in line and "Fn" not in line


def test_the_live_value_is_each_gauges_mean():
    samples = [ForceSample(k, 0.001 * k, 0.001 * k, (float(k), 1.0, 0, 0, 0, -float(k))) for k in range(5)]
    assert mean_volts(samples) == pytest.approx((2.0, 1.0, 0, 0, 0, -2.0))


# --- zero, noise, drift ------------------------------------------------------------------------


def test_the_zero_is_the_mean_of_the_quietest_half_second_inside_the_step_and_the_noise_is_in_newtons(cal_path):
    cal = load_ati_calibration(cal_path)
    bias = (0.11, -0.22, 0.33, 0.0, 0.05, -0.05)
    before = [bias] * 1000  # perfectly still, but before the step: must not be used
    step = [tuple(b + 0.4 for b in bias)] * 300 + quiet_volts(2000, bias, 0.01)  # hands still on for 0.3 s
    volts = before + step
    t_s = [10.0 + k / 1000.0 for k in range(len(volts))]
    zero = measure_zero(t_s, volts, 11.0, 13.3, 1000.0, cal)
    assert zero.samples == 500 and zero.quiet_start_s >= 11.3
    assert zero.volts == pytest.approx(bias, abs=0.003)
    # Fz = 0.5 g0 + 0.5 g1 + 2 g2 of independent noise with SD 0.01: SD sqrt(0.25 + 0.25 + 4) * 0.01 = 0.0212
    assert zero.noise_sd["Fz"] == pytest.approx(0.0212, rel=0.15)
    assert zero.noise_sd["Fx"] == pytest.approx(0.01, rel=0.15)
    assert set(zero.noise_sd) == {*AXES, "Ft"}
    with pytest.raises(ValueError, match="only 0 force samples"):
        measure_zero(t_s, volts, 20.0, 21.0, 1000.0, cal)


def test_the_drift_is_the_change_of_the_zero_in_newtons_or_in_volts(cal_path):
    cal = load_ati_calibration(cal_path)
    start = Zero((0.1, 0.2, 0.3, 0.0, 0.0, 0.0), {}, 0.0, 0.5, 500)
    end = Zero((0.13, 0.2, 0.32, 0.0, 0.0, 0.01), {}, 9.0, 9.5, 500)
    moved = drift(start, end, cal)
    assert list(moved) == list(AXES)
    assert moved["Fx"] == pytest.approx(0.03) and moved["Fy"] == pytest.approx(0.0)
    assert moved["Fz"] == pytest.approx(0.5 * 0.03 + 2 * 0.02)
    assert moved["Tz"] == pytest.approx(0.01)
    assert drift(start, end, None)["g2"] == pytest.approx(0.02)


# --- the whole tool with the fake card ---------------------------------------------------------


def run_tool(tmp_path, *extra: str) -> tuple[int, dict, object]:
    code = force_measure.main(
        ["--fake", "--label", "test", "--seconds", "0.6", "--zero-s", "0.5", "--data-dir", str(tmp_path / "out"), *extra]
    )
    summaries = list((tmp_path / "out").glob("*_summary.json"))
    assert len(summaries) == 1 and re.fullmatch(r"force_test_\d{8}_\d{6}_summary\.json", summaries[0].name)
    summary = json.loads(summaries[0].read_text())
    force_file = summaries[0].with_name(summary["force_file"])
    assert force_file.name == summaries[0].name.replace("_summary.json", "_force.sqlite3") and force_file.exists()
    return code, summary, load_force(force_file)


def test_a_fake_run_keeps_every_raw_sample_and_a_summary_that_the_file_confirms(tmp_path, cal_path, capsys):
    code, summary, recording = run_tool(tmp_path, "--cal", str(cal_path))
    out = capsys.readouterr().out
    assert code == 0 and summary["reader_error"] is None
    assert set(summary) == SUMMARY_KEYS
    assert summary["calibration"] == {
        "file": "FT99999.cal", "serial": "FT99999", "part_number": "SI-12-0.12", "force_units": "N", "torque_units": "N-mm",
    }  # fmt: skip
    assert summary["settings"]["source"] == "fake" and summary["settings"]["card"] is None
    assert summary["force_samples"] == len(recording.volts) > 1500
    assert recording.volts == [fake_volts(k) for k in range(len(recording.volts))]  # raw volts, untouched

    phases = summary["phases"]
    order = [phases[p][edge] for p in ("zero_start", "record", "zero_end") for edge in ("start_s", "end_s")]
    assert order == sorted(order)
    # The reader is stopped right after the last step; its newest sample is then a few ms old.
    assert recording.t_s[0] <= phases["zero_start"]["end_s"] and recording.t_s[-1] >= phases["zero_end"]["end_s"] - 0.02

    # Each zero is the mean of the file's samples over the stretch the summary names.
    t = np.array(recording.t_s)
    volts = np.array(recording.volts)
    for name in ("zero_start", "zero_end"):
        zero = summary[name]
        inside = (t >= zero["quiet_start_s"]) & (t <= zero["quiet_end_s"])
        assert inside.sum() == pytest.approx(zero["samples"], abs=2)
        assert volts[inside].mean(axis=0) == pytest.approx(zero["volts"], abs=1e-3)
    cal = load_ati_calibration(cal_path)
    expected = cal.apply(summary["zero_end"]["volts"], summary["zero_start"]["volts"])
    assert [summary["drift"][axis] for axis in AXES] == pytest.approx(expected)

    for text in ("Step 1 of 3", "Noise there (SD): Fz", "Fn ", "Step 3 of 3", "Drift of the zero", "Saved"):
        assert text in out


def test_without_a_calibration_the_run_says_so_and_keeps_volts(tmp_path, capsys):
    code, summary, recording = run_tool(tmp_path)
    out = capsys.readouterr().out
    assert code == 0 and "UNCALIBRATED" in out and "change" in out
    assert summary["calibration"] is None and set(summary["units"].values()) == {"V"}
    assert set(summary["drift"]) == {"g0", "g1", "g2", "g3", "g4", "g5"}
    assert summary["force_samples"] == len(recording.volts)


def test_a_reader_that_fails_midway_is_stopped_and_its_error_is_in_the_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(force_measure, "source_factory", lambda args, fake: partial(FakeForceSource, fail_after=700))
    code, summary, recording = run_tool(tmp_path, "--seconds", "3")
    assert code == 1
    assert "failed while reading" in summary["reader_error"]
    assert summary["drift"] is None and "zero_end" not in summary["phases"]
    assert summary["force_samples"] == len(recording.volts) <= 700
    assert "FORCE READER ERROR" in capsys.readouterr().out


def test_a_card_that_cannot_be_opened_still_leaves_a_summary_with_the_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(force_measure, "source_factory", lambda args, fake: partial(FakeForceSource, fail_at_start=True))
    code, summary, recording = run_tool(tmp_path)
    assert code == 1
    assert "could not be opened" in summary["reader_error"]
    assert summary["phases"] == {} and summary["force_samples"] == 0 and recording.volts == []
    assert "it could not start" in capsys.readouterr().out


def test_the_hands_off_step_must_hold_the_quiet_window():
    with pytest.raises(SystemExit):
        force_measure.parse_args(["--zero-s", "0.2"])


# --- the EA signal switched by the tool (--ea-every) -------------------------------------------


def test_the_ea_schedule_starts_off_and_then_alternates():
    assert [ea_on_at(t, 10.0) for t in (0.0, 9.99, 10.0, 19.99, 20.0, 30.0, 45.0, 55.0)] == [
        False, False, True, True, False, True, False, True,
    ]  # fmt: skip


def check_switches(summary: dict, every_s: float) -> list[dict]:
    """The switches alternate, start with ON, end with OFF, and all fall in the recording step."""
    switches = summary["ea"]["switches"]
    assert [s["on"] for s in switches] == [k % 2 == 0 for k in range(len(switches))] and not switches[-1]["on"]
    record = summary["phases"]["record"]
    for s in switches:
        assert record["start_s"] <= s["asked_s"] <= s["done_s"] <= summary["phases"]["zero_end"]["start_s"]
    assert switches[0]["asked_s"] - record["start_s"] == pytest.approx(every_s, abs=0.03)
    return switches


def test_with_a_fake_generator_the_signal_follows_the_schedule_and_every_switch_is_noted(tmp_path, cal_path, capsys):
    code, summary, _ = run_tool(tmp_path, "--cal", str(cal_path), "--ea-every", "0.2", "--fake-generator")
    out = capsys.readouterr().out
    assert code == 0 and summary["ea"]["every_s"] == 0.2 and summary["ea"]["generator"] == "fake"
    assert len(check_switches(summary, 0.2)) == 2  # on at 0.2 s, off at 0.4 s; the recording ends at 0.6 s
    assert "EA signal ON" in out and "EA signal OFF" in out


def test_a_signal_still_on_when_the_recording_ends_is_switched_off_before_the_last_zero(tmp_path, capsys):
    code, summary, _ = run_tool(tmp_path, "--seconds", "0.3", "--ea-every", "0.2", "--fake-generator")
    assert code == 0
    assert len(check_switches(summary, 0.2)) == 2  # on at 0.2 s, off when the recording ends at 0.3 s


class FakeInstrument:
    def query(self, command: str) -> str:
        return "FAKE,GENERATOR"


def test_the_real_generator_is_set_up_as_the_experiment_does_and_closed_with_the_signal_off(tmp_path, monkeypatch):
    calls: list[str] = []
    instrument = FakeInstrument()
    monkeypatch.setattr(stimulus, "connect_hardware", lambda cfg: calls.append(f"connect {cfg.carrier_freq_hz:g}") or instrument)
    monkeypatch.setattr(stimulus, "signal_on", lambda inst: calls.append("on") if inst is instrument else None)
    monkeypatch.setattr(stimulus, "signal_off", lambda inst: calls.append("off") if inst is instrument else None)
    monkeypatch.setattr(stimulus, "close_hardware", lambda inst: calls.append("close") if inst is instrument else None)
    code, summary, _ = run_tool(tmp_path, "--ea-every", "0.2")
    assert code == 0
    assert calls == ["connect 125", "on", "off", "close"]
    assert summary["ea"]["generator"] == "FAKE,GENERATOR" and summary["ea"]["carrier_hz"] == 125.0
    check_switches(summary, 0.2)


def test_a_generator_that_cannot_be_reached_stops_the_run_and_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(stimulus, "connect_hardware", lambda cfg: None)
    code, summary, _ = run_tool(tmp_path, "--ea-every", "0.2")
    assert code == 1
    assert summary["ea"]["error"] == "the signal generator could not be reached" and summary["drift"] is None
    assert "GENERATOR ERROR" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [["--ea-every", "0"], ["--ea-every", "30"], ["--fake-generator"]])
def test_an_ea_schedule_that_could_never_switch_on_is_refused(extra):
    with pytest.raises(SystemExit):
        force_measure.parse_args(["--seconds", "30", *extra])


def test_ctrl_c_while_recording_still_writes_the_summary_with_the_switches_so_far(tmp_path, monkeypatch, capsys):
    real_record = force_measure.record

    def record_then_ctrl_c(reader, seconds, zero, calibration, ea=None):
        real_record(reader, 0.3, zero, calibration, ea)  # the signal comes on at 0.2 s, then the operator stops
        ea.set(True, force_measure.time.perf_counter())
        raise KeyboardInterrupt

    monkeypatch.setattr(force_measure, "record", record_then_ctrl_c)
    code, summary, recording = run_tool(tmp_path, "--seconds", "3", "--ea-every", "0.2", "--fake-generator")
    assert code == 1 and summary["interrupted"] is True
    assert summary["zero_start"] is not None and summary["drift"] is None and "zero_end" not in summary["phases"]
    assert summary["phases"]["record"]["interrupted"] is True
    assert summary["force_samples"] == len(recording.volts) > 0
    switches = summary["ea"]["switches"]
    assert [s["on"] for s in switches][-1] is False  # switched off on the way out
    assert "Stopped with Ctrl+C" in capsys.readouterr().out
