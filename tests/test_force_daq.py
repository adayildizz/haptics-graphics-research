from __future__ import annotations

import os
import random
import statistics
import subprocess
import sys
import threading
import time
from functools import partial
from pathlib import Path

import pytest

from experiment import force_daq
from experiment.force_daq import (
    FakeForceSource,
    ForceReader,
    LiveClock,
    ReadStamp,
    _StoreWriter,
    fake_volts,
    fit_sample_times,
    load_force,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeTime:
    """A clock that only moves when someone sleeps, so the fake card can be read without real waiting."""

    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def fake_card(**kwargs) -> tuple[FakeForceSource, FakeTime]:
    timer = FakeTime()
    source = FakeForceSource(clock=timer.clock, sleep=timer.sleep, **kwargs)
    source.start()
    return source, timer


def wait_until(condition, timeout_s: float = 2.0) -> bool:
    end = time.perf_counter() + timeout_s
    while time.perf_counter() < end:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


# --- fake source ------------------------------------------------------------------------------


def test_the_fake_card_waits_for_the_samples_asked_for_and_then_returns_every_new_one_in_order():
    source, timer = fake_card()
    assert source.read(2, 1.0) == [fake_volts(0), fake_volts(1)]  # only sample 0 existed; it waited for sample 1
    assert timer.now == pytest.approx(100.001)
    timer.now += 0.0105
    rows = source.read(2, 1.0)
    assert rows == [fake_volts(i) for i in range(2, 12)]
    assert len(rows[0]) == 6


def test_a_fast_card_clock_delivers_more_samples_per_second_of_the_computers_clock():
    source, timer = fake_card(drift_ppm=1000.0)
    timer.now += 1.0005
    assert len(source.read(1, 1.0)) == 1002  # sample 0 at start, then 1001 more within one second
    assert source.sample_time(1001) == pytest.approx(101.0, abs=1e-9)


def test_the_fake_card_times_out_when_too_few_samples_arrive():
    source, timer = fake_card()
    with pytest.raises(TimeoutError):
        source.read(50, 0.010)
    assert timer.now == pytest.approx(100.010)


def test_the_fake_card_must_be_started_before_reading():
    with pytest.raises(RuntimeError):
        FakeForceSource().read(1, 0.1)


# --- clock mapping ----------------------------------------------------------------------------


def reads(seconds: float, *, every_s: float, drift_ppm: float = 0.0, latency_ms=(0.05, 1.5), seed: int = 1):
    """Read stamps of a card started at t=50 s, with a random delay between a sample arriving and the read returning."""
    rng = random.Random(seed)
    period = 1.0 / (1000.0 * (1.0 + 1e-6 * drift_ppm))
    stamps = []
    t = 50.0
    while t < 50.0 + seconds:
        t += every_s
        total = int((t - 50.0) / period) + 1
        stamps.append(ReadStamp(50.0 + (total - 1) * period + rng.uniform(*latency_ms) / 1000.0, total))
    return stamps, (lambda k: 50.0 + k * period)


def test_live_times_are_never_later_than_the_truth_by_more_than_the_smallest_read_delay():
    stamps, truth = reads(3.0, every_s=0.002)
    clock = LiveClock(1000.0)
    errors = []
    for stamp in stamps:
        clock.add(stamp)
        errors.append(clock.time_of(stamp.total - 1) - truth(stamp.total - 1))
    assert all(-1e-9 < error < 0.0002 for error in errors[100:])


def test_the_live_window_keeps_clock_drift_out_of_the_times():
    # A slow card clock: the earliest reads would anchor ever older starts if they were never dropped.
    stamps, truth = reads(60.0, every_s=0.002, drift_ppm=-100.0)
    windowed, unbounded = LiveClock(1000.0), LiveClock(1000.0, window_s=1e9)
    for stamp in stamps:
        windowed.add(stamp)
        unbounded.add(stamp)
    last = stamps[-1].total - 1
    assert abs(windowed.time_of(last) - truth(last)) < 0.0005
    assert abs(unbounded.time_of(last) - truth(last)) > 0.004  # 100 ppm over a minute is 6 ms


def test_the_offline_fit_times_a_half_hour_run_to_within_half_a_millisecond_despite_drift():
    stamps, truth = reads(1800.0, every_s=0.05, drift_ppm=50.0, latency_ms=(0.1, 3.0))
    t0, period = fit_sample_times(stamps, 1000.0)
    last = stamps[-1].total - 1
    worst = max(abs(t0 + k * period - truth(k)) for k in range(0, last, 997))
    assert worst < 0.0005
    nominal = abs(stamps[0].t_s + (last - stamps[0].total + 1) / 1000.0 - truth(last))
    assert nominal > 0.08  # what the card's nominal rate alone would get wrong by the end: about 90 ms


def test_the_fit_needs_at_least_one_read_and_falls_back_to_the_nominal_rate_with_one():
    with pytest.raises(ValueError):
        fit_sample_times([], 1000.0)
    assert fit_sample_times([ReadStamp(10.0, 5)], 1000.0) == (pytest.approx(9.996), 0.001)


# --- storage ----------------------------------------------------------------------------------


def test_stored_samples_load_back_with_their_values_and_times(tmp_path):
    path = tmp_path / "force.sqlite3"
    writer = _StoreWriter(path, 6, {"rate_hz": 1000.0, "channels": 6, "source": "test"})
    total = 0
    for chunk in range(50):
        rows = [fake_volts(total + i) for i in range(3)]
        writer.put(total, rows, ReadStamp(20.0 + (total + 2) / 1000.0 + 0.0003, total + 3))
        total += 3
    writer.close()
    assert writer.error is None
    recording = load_force(path)
    assert recording.metadata["schema_version"] == "1"
    assert recording.metadata["source"] == "test"
    assert recording.volts == [fake_volts(k) for k in range(150)]
    assert recording.t_s[0] == pytest.approx(20.0003, abs=1e-6)
    assert recording.t_s[149] - recording.t_s[0] == pytest.approx(0.149, abs=1e-6)


# --- reader process ---------------------------------------------------------------------------


def poll(reader: ForceReader, seconds: float) -> list:
    samples = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        time.sleep(1 / 60)
        sample = reader.latest()
        if sample is not None:
            samples.append((time.perf_counter(), sample))
    return samples


def test_the_reader_process_publishes_fresh_samples_and_stores_every_one(tmp_path, capsys):
    path = tmp_path / "force.sqlite3"
    reader = ForceReader(path, partial(FakeForceSource)).start()
    try:
        assert abs(reader.clock_offset_s) < 0.001  # both processes read the same clock
        seen = poll(reader, 0.6)
    finally:
        summary = reader.stop()
    assert summary.error is None
    ages = sorted(1000.0 * (now - sample.t_s) for now, sample in seen)
    with capsys.disabled():
        print(f"\n  force value age at 60 Hz polling: median {statistics.median(ages):.2f} ms, max {ages[-1]:.2f} ms")
    # Loose bound so a busy test machine does not fail the suite; ``python -m experiment.force_daq`` reports the real figure.
    assert ages[int(0.9 * len(ages))] < 10.0
    indices = [sample.index for _, sample in seen]
    assert indices == sorted(indices) and len(set(indices)) > 20
    assert all(sample.volts == fake_volts(sample.index) for _, sample in seen)

    recording = load_force(path)
    assert summary.samples == len(recording.volts) > 500
    assert recording.volts == [fake_volts(k) for k in range(summary.samples)]
    spacing = (recording.t_s[-1] - recording.t_s[0]) / (summary.samples - 1)
    assert spacing == pytest.approx(0.001, rel=0.01)
    _, last = seen[-1]
    assert recording.t_s[last.index] == pytest.approx(last.t_s, abs=0.001)  # live and offline times agree


def test_a_source_that_cannot_start_is_reported_by_start(tmp_path):
    reader = ForceReader(tmp_path / "force.sqlite3", partial(FakeForceSource, fail_at_start=True))
    with pytest.raises(RuntimeError, match="could not be opened"):
        reader.start()
    assert not reader.running


def test_a_source_with_a_different_channel_count_is_refused(tmp_path):
    reader = ForceReader(tmp_path / "force.sqlite3", partial(FakeForceSource, channels=7))
    with pytest.raises(RuntimeError, match="7 channels"):
        reader.start()


def test_a_failure_while_reading_stops_the_process_and_is_reported_by_stop(tmp_path):
    path = tmp_path / "force.sqlite3"
    reader = ForceReader(path, partial(FakeForceSource, fail_after=100)).start()
    assert wait_until(lambda: not reader.running)
    summary = reader.stop()
    assert "failed while reading" in summary.error
    assert 0 < summary.samples <= 100
    assert len(load_force(path).volts) == summary.samples
    assert reader.stop() is summary


def test_latest_gives_up_quickly_while_the_reader_holds_the_lock(tmp_path):
    reader = ForceReader(tmp_path / "force.sqlite3", partial(FakeForceSource)).start()
    try:
        assert wait_until(lambda: reader.latest() is not None)
        # The lock is reentrant, so it has to be held from another thread to stand in for the publishing reader.
        held, release = threading.Event(), threading.Event()

        def hold() -> None:
            with reader._shared.get_lock():
                held.set()
                release.wait(5.0)

        holder = threading.Thread(target=hold)
        holder.start()
        assert held.wait(5.0)
        before = reader._last
        try:
            start = time.perf_counter()
            during = reader.latest()
            waited = time.perf_counter() - start
        finally:
            release.set()
            holder.join()
        assert during is before
        assert waited < force_daq.LATEST_WAIT_S + 0.005
        assert wait_until(lambda: reader.latest().index > before.index)
    finally:
        reader.stop()


ORPHAN_SCRIPT = """
import os, sys, time
from functools import partial
from experiment.force_daq import FakeForceSource, ForceReader

if __name__ == "__main__":
    reader = ForceReader(sys.argv[1], partial(FakeForceSource)).start()
    print(reader._process.pid, flush=True)
    time.sleep(0.2)
    os._exit(0)  # no cleanup at all, as if the experiment crashed
"""


def test_the_reader_process_ends_by_itself_when_the_experiment_dies(tmp_path):
    script = tmp_path / "orphan.py"
    script.write_text(ORPHAN_SCRIPT)
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "force.sqlite3")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )  # fmt: skip
    child_pid = int(result.stdout.split()[0])

    def gone() -> bool:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    assert wait_until(gone, timeout_s=5.0)
    assert len(load_force(tmp_path / "force.sqlite3").volts) > 100


def test_the_self_check_reports_its_figures(capsys):
    code = force_daq.main(["--seconds", "0.5"])
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert "age of the newest value" in out and ("PASS" in out or "FAIL" in out)


def _fake_nidaqmx(monkeypatch, *, newer_names: bool = True, available: int = 5):
    """A stand-in for the ``nidaqmx`` package that records what the source asks of it."""
    import sys
    import types

    calls: dict = {}

    class TerminalConfiguration:
        pass

    for name in (("DIFF", "RSE", "NRSE", "DEFAULT") if newer_names else ("DIFFERENTIAL", "RSE", "NRSE", "DEFAULT")):
        setattr(TerminalConfiguration, name, name)

    class AcquisitionType:
        CONTINUOUS = "CONTINUOUS"

    class Task:
        def __init__(self):
            self.ai_channels = types.SimpleNamespace(add_ai_voltage_chan=self._add)
            self.timing = types.SimpleNamespace(cfg_samp_clk_timing=self._timing)
            self.in_stream = types.SimpleNamespace(avail_samp_per_chan=available)
            calls["task"] = self

        def _add(self, physical, **kwargs):
            calls["channels"] = (physical, kwargs)

        def _timing(self, rate, **kwargs):
            calls["timing"] = (rate, kwargs)

        def start(self):
            calls["started"] = True

        def read(self, number_of_samples_per_channel, timeout):
            calls["read"] = number_of_samples_per_channel
            return [[float(c * 10 + k) for k in range(number_of_samples_per_channel)] for c in range(6)]

        def stop(self):
            calls["stopped"] = True

        def close(self):
            calls["closed"] = True

    package = types.ModuleType("nidaqmx")
    package.Task = Task
    constants = types.ModuleType("nidaqmx.constants")
    constants.TerminalConfiguration = TerminalConfiguration
    constants.AcquisitionType = AcquisitionType
    package.constants = constants
    monkeypatch.setitem(sys.modules, "nidaqmx", package)
    monkeypatch.setitem(sys.modules, "nidaqmx.constants", constants)
    return calls


@pytest.mark.parametrize("newer_names", [True, False])
def test_the_nidaqmx_source_opens_six_differential_inputs_and_returns_samples_in_order(monkeypatch, newer_names):
    from experiment.force_daq import NidaqmxForceSource

    calls = _fake_nidaqmx(monkeypatch, newer_names=newer_names)
    source = NidaqmxForceSource("Dev2", first_channel=8)
    source.start()
    physical, kwargs = calls["channels"]
    assert physical == "Dev2/ai8:13"
    assert kwargs["terminal_config"] in ("DIFF", "DIFFERENTIAL") and kwargs["min_val"] == -10.0
    assert calls["timing"][0] == 1000.0 and calls["timing"][1]["sample_mode"] == "CONTINUOUS"
    rows = source.read(2, 0.5)  # 5 samples are waiting: all of them come back, oldest first
    assert calls["read"] == 5 and len(rows) == 5
    assert rows[0] == (0.0, 10.0, 20.0, 30.0, 40.0, 50.0) and rows[4][0] == 4.0
    source.close()
    assert calls["stopped"] and calls["closed"]
    with pytest.raises(ValueError):
        NidaqmxForceSource(terminal="pseudo")
