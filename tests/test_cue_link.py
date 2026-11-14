"""The Haptuator cue: the Bluetooth link against a fake transport, and the trial loop sending top-edge cues."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from render_harness import polyline, run_scripted_trial

from experiment import data_logger
from experiment import main as main_module
from experiment import trial_render
from experiment.calibration import make_configured_ir_frame_calibration
from experiment.config import ExperimentConfig, load_experiment_config
from experiment.constant_stimuli import TrialSpec
from experiment.cue_link import CueLink, CueRecord, message_for

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "experiment" / "configs" / "default.yaml"
CALIBRATION = make_configured_ir_frame_calibration((1920, 1080))
# Left bar (reference, 10 mm): x 67-77 mm, top edge at y = 135 mm. Right bar (12 mm): x 117-127 mm, top at 133 mm.
SPEC = TrialSpec(level_pct=0.2, comparison_height_mm=12.0, reference_height_mm=10.0, reference_side="left")


class FakeTransport:
    """Stands in for Bluetooth: counts connections and keeps what is written."""

    def __init__(self, fail_connects: int = 0, fail_writes: int = 0, write_delay_s: float = 0.0) -> None:
        self.connects = 0
        self.written: list[bytes] = []
        self.fail_connects = fail_connects
        self.fail_writes = fail_writes
        self.write_delay_s = write_delay_s
        self.disconnected = False
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self.connects += 1
        if self.connects <= self.fail_connects:
            raise ConnectionError("no device")
        self._connected = True

    async def write(self, data: bytes) -> None:
        if self.write_delay_s:
            await asyncio.sleep(self.write_delay_s)
        if self.fail_writes:
            self.fail_writes -= 1
            self._connected = False
            raise OSError("link lost")
        self.written.append(data)

    async def disconnect(self) -> None:
        self._connected = False
        self.disconnected = True


class FakeLink:
    """Stands in for the CueLink in a trial: every message counts as written at once."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self._records: list[CueRecord] = []

    def send(self, message: str, now_s: float) -> None:
        self.sent.append(message)
        self._records.append(CueRecord(message, now_s, now_s, now_s))

    def flush(self, timeout_s: float = 1.0) -> bool:
        return True

    def drain(self) -> list[CueRecord]:
        records, self._records = self._records, []
        return records


def wait_until(condition, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


@pytest.fixture
def make_link():
    links: list[CueLink] = []

    def make(transport: FakeTransport, **kwargs) -> CueLink:
        link = CueLink(transport, retry_s=0.01, **kwargs).start()
        links.append(link)
        return link

    yield make
    for link in links:
        link.close()


def config(**overrides) -> ExperimentConfig:
    return replace(load_experiment_config(CONFIG_PATH), **overrides)


def test_messages_name_the_direction_and_the_bar():
    assert message_for("left", "entering") == "ENTER_LEFT"
    assert message_for("left", "leaving") == "EXIT_LEFT"
    assert message_for("right", "entering") == "ENTER_RIGHT"
    assert message_for("right", "leaving") == "EXIT_RIGHT"


def test_a_message_goes_out_as_one_text_line(make_link):
    transport = FakeTransport()
    link = make_link(transport)
    assert link.wait_connected(2.0)
    link.send("EXIT_LEFT", time.perf_counter())
    assert link.flush(2.0)
    (record,) = link.drain()
    assert transport.written == [b"EXIT_LEFT\n"]
    assert record.message == "EXIT_LEFT" and record.error is None
    assert record.queued_s <= record.write_start_s <= record.write_end_s


def test_a_message_while_not_connected_is_recorded_and_never_sent_later(make_link):
    transport = FakeTransport(fail_connects=10**6)
    link = make_link(transport)
    link.send("ENTER_RIGHT", 1.0)
    (record,) = link.drain()
    assert record.error == "not connected" and record.write_start_s is None
    wait_until(lambda: transport.connects >= 2)  # it keeps trying in the background
    assert link.last_error.startswith("connect failed")
    assert transport.written == []


def test_after_a_failed_write_the_link_connects_again(make_link):
    transport = FakeTransport(fail_writes=1)
    link = make_link(transport)
    assert link.wait_connected(2.0)
    link.send("ENTER_LEFT", time.perf_counter())
    assert link.flush(2.0)
    (failed,) = link.drain()
    assert failed.error.startswith("write failed")
    wait_until(lambda: transport.connects == 2 and link.connected)
    link.send("EXIT_LEFT", time.perf_counter())
    assert link.flush(2.0)
    assert [record.error for record in link.drain()] == [None]
    assert transport.written == [b"EXIT_LEFT\n"]


def test_a_message_that_waits_too_long_is_dropped_not_sent_late(make_link):
    transport = FakeTransport(write_delay_s=0.15)
    link = make_link(transport, max_age_s=0.1)
    assert link.wait_connected(2.0)
    link.send("ENTER_LEFT", time.perf_counter())
    link.send("EXIT_LEFT", time.perf_counter())  # waits behind the first, slow write
    assert link.flush(2.0)
    assert [(r.message, r.error) for r in link.drain()] == [("ENTER_LEFT", None), ("EXIT_LEFT", "too late")]
    assert transport.written == [b"ENTER_LEFT\n"]


def test_close_stops_the_thread_and_disconnects():
    transport = FakeTransport()
    link = CueLink(transport, retry_s=0.01).start()
    assert link.wait_connected(2.0)
    link.close()
    assert not any(t.name == "cue-link" and t.is_alive() for t in threading.enumerate())
    assert transport.disconnected
    link.send("EXIT_LEFT", 1.0)
    assert link.drain()[0].error == "not connected"


def test_a_top_edge_crossing_down_and_up_sends_enter_and_exit(monkeypatch):
    link = FakeLink()
    monkeypatch.setattr(trial_render, "CUE_LINK", link)
    # From 15 mm above the left bar's top edge down into the bar, then back up.
    scenario = dict(path=polyline([(0, 72, 120), (20, 72, 140), (40, 72, 120)], 41), end="left")
    _, attempt, _, _ = run_scripted_trial(config(cue_enabled=True), CALIBRATION, SPEC, scenario)
    assert link.sent == ["ENTER_LEFT", "EXIT_LEFT"]
    crossings = [payload for _, kind, payload in attempt.events if kind == "cue_crossing"]
    assert [(c["bar"], c["direction"], c["outcome"]) for c in crossings] == [
        ("left", "entering", "pulse"),
        ("left", "leaving", "pulse"),
    ]
    sent = [payload for _, kind, payload in attempt.events if kind == "cue_sent"]
    assert [s["message"] for s in sent] == ["ENTER_LEFT", "EXIT_LEFT"]
    assert all(s["write_end_us"] >= s["write_start_us"] >= s["queued_us"] for s in sent)


def test_leaving_a_bar_through_its_side_sends_nothing(monkeypatch):
    link = FakeLink()
    monkeypatch.setattr(trial_render, "CUE_LINK", link)
    scenario = dict(path=polyline([(0, 72, 140), (20, 90, 140)], 21), end="left")
    _, attempt, _, _ = run_scripted_trial(config(cue_enabled=True), CALIBRATION, SPEC, scenario)
    assert link.sent == []
    assert not [kind for _, kind, _ in attempt.events if kind.startswith("cue_")]


def test_with_the_cue_off_nothing_is_sent_even_across_the_top_edge(monkeypatch):
    link = FakeLink()
    monkeypatch.setattr(trial_render, "CUE_LINK", link)
    scenario = dict(path=polyline([(0, 72, 120), (20, 72, 140), (40, 72, 120)], 41), end="left")
    _, attempt, _, _ = run_scripted_trial(config(), CALIBRATION, SPEC, scenario)
    assert link.sent == []
    assert not [kind for _, kind, _ in attempt.events if kind.startswith("cue_")]


def test_the_cue_settings_are_checked():
    assert config().cue_enabled is False  # the default config leaves the cue off
    with pytest.raises(ValueError, match="cue_min_interval_ms"):
        config(cue_min_interval_ms=0.0)
    with pytest.raises(ValueError, match="cue_device_name"):
        config(cue_enabled=True, cue_device_name=" ")


def test_main_opens_the_cue_link_names_it_on_the_start_screen_and_closes_it(monkeypatch, tmp_path):
    from experiment import cue_link, signal_scheduler, stimulus

    cue_config = tmp_path / "cue.yaml"
    cue_config.write_text(CONFIG_PATH.read_text().replace("cue_enabled: false", "cue_enabled: true"))
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setattr("sys.argv", ["experiment.main", "--config", str(cue_config), "--participant", "T1", "--windowed"])
    monkeypatch.setattr(stimulus, "connect_hardware", lambda _cfg: None)
    monkeypatch.setattr(data_logger, "ensure_data_dir", lambda: tmp_path)
    monkeypatch.setattr(signal_scheduler, "exit_cleanly_on_sigterm", lambda: None)
    transport = FakeTransport()
    monkeypatch.setattr(cue_link, "BleakTransport", lambda _name: transport)
    shown: list[str] = []
    monkeypatch.setattr(main_module, "wait_for_continue_or_exit", lambda _screen, message: shown.append(message) or False)

    assert main_module.run() == 0

    assert "Cue: connected." in shown[0]
    assert transport.connects == 1 and transport.disconnected
    assert not any(t.name == "cue-link" and t.is_alive() for t in threading.enumerate())
    assert trial_render.CUE_LINK is None
