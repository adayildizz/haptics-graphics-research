from __future__ import annotations

import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from experiment.render_logic import Plan, Transition
from experiment.signal_scheduler import RecordingDevice, SignalScheduler, lateness_ms, run_benchmark

REPO_ROOT = Path(__file__).resolve().parent.parent


def plan(on: bool, *transitions: Transition, t: float | None = None) -> Plan:
    return Plan(time.perf_counter() if t is None else t, on, "left" if on else None, "top", transitions, 72.0, 135.0, 1.5, -40.0)


def wait_until(condition, timeout_s: float = 1.0) -> bool:
    end = time.perf_counter() + timeout_s
    while time.perf_counter() < end:
        if condition():
            return True
        time.sleep(0.001)
    return condition()


@pytest.fixture
def rig():
    device = RecordingDevice()
    scheduler = SignalScheduler(device).start()
    yield scheduler, device
    scheduler.close()


def states(device: RecordingDevice) -> list[bool]:
    return [on for _, on in device.states]


def test_a_plan_sets_the_state_at_once_and_only_writes_on_a_change(rig):
    scheduler, device = rig
    scheduler.submit(plan(True))
    assert wait_until(lambda: states(device) == [True])
    for _ in range(5):
        scheduler.submit(plan(True))
        time.sleep(0.005)
    assert states(device) == [True]
    scheduler.submit(plan(False))
    assert wait_until(lambda: states(device) == [True, False])
    kinds = [(r.kind, r.on) for r in scheduler.drain()]
    assert kinds == [("plan", True), ("plan", False)]


def test_a_transition_runs_at_its_planned_time_and_is_logged_with_edge_position_and_velocity(rig):
    scheduler, device = rig
    due = time.perf_counter() + 0.020
    scheduler.submit(plan(False, Transition(due, True, "right", "top")))
    assert wait_until(lambda: states(device) == [True])
    (record,) = scheduler.drain()
    assert (record.kind, record.on, record.bar, record.edge) == ("transition", True, "right", "top")
    assert record.planned_s == due
    assert 0.0 <= record.write_start_s - due < 0.005
    assert record.write_end_s >= record.write_start_s
    assert (record.x_mm, record.y_mm, record.vx_mm_s, record.vy_mm_s) == (72.0, 135.0, 1.5, -40.0)


def test_entry_and_exit_planned_together_both_run_in_order(rig):
    scheduler, device = rig
    now = time.perf_counter()
    scheduler.submit(plan(False, Transition(now + 0.010, True, "left", "top"), Transition(now + 0.015, False, "left", "right")))
    assert wait_until(lambda: states(device) == [True, False])
    on_time, off_time = (t for t, _ in device.states)
    assert off_time - on_time == pytest.approx(0.005, abs=0.003)
    assert [r.edge for r in scheduler.drain()] == ["top", "right"]


def test_a_newer_plan_cancels_what_was_pending(rig):
    scheduler, device = rig
    scheduler.submit(plan(False, Transition(time.perf_counter() + 0.030, True, "left", "left")))
    time.sleep(0.005)
    scheduler.submit(plan(False))
    time.sleep(0.060)
    assert states(device) == [] and scheduler.drain() == []


def test_watchdog_switches_off_when_plans_stop_coming_and_logs_it():
    device = RecordingDevice()
    scheduler = SignalScheduler(device, watchdog_s=0.100).start()
    try:
        scheduler.submit(plan(True))
        submitted = time.perf_counter()
        assert wait_until(lambda: states(device) == [True, False], timeout_s=0.5)
        assert device.states[1][0] - submitted == pytest.approx(0.100, abs=0.020)
        records = scheduler.drain()
        assert [(r.kind, r.on) for r in records] == [("plan", True), ("watchdog", False)]
        assert not scheduler.is_on and scheduler.error is None
    finally:
        scheduler.close()


def test_watchdog_stays_quiet_while_plans_keep_arriving():
    device = RecordingDevice()
    scheduler = SignalScheduler(device, watchdog_s=0.050).start()
    try:
        for _ in range(25):  # a finger resting inside a bar: the same plan every frame for 0.4 s
            scheduler.submit(plan(True))
            time.sleep(1 / 60)
        assert states(device) == [True]
    finally:
        scheduler.close()
    assert states(device) == [True, False]


def test_watchdog_also_drops_transitions_that_were_still_pending():
    device = RecordingDevice()
    scheduler = SignalScheduler(device, watchdog_s=0.030).start()
    try:
        scheduler.submit(plan(False, Transition(time.perf_counter() + 0.080, True, "left", "left")))
        time.sleep(0.150)
        assert states(device) == []
    finally:
        scheduler.close()


class FailingDevice(RecordingDevice):
    """Raises on the second attempt to switch on, like a dropped VISA connection."""

    def __init__(self) -> None:
        super().__init__()
        self.on_calls = 0

    def on(self) -> None:
        self.on_calls += 1
        if self.on_calls == 2:
            raise OSError("VI_ERROR_CONN_LOST")
        super().on()


def test_a_device_error_switches_off_reports_itself_and_stops_the_scheduler():
    device = FailingDevice()
    scheduler = SignalScheduler(device).start()
    try:
        scheduler.submit(plan(True))
        assert wait_until(lambda: states(device) == [True])
        scheduler.submit(plan(False))
        assert wait_until(lambda: states(device) == [True, False])
        scheduler.submit(plan(True))
        assert wait_until(lambda: scheduler.error is not None)
        assert isinstance(scheduler.error, OSError)
        assert states(device) == [True, False, False]  # the failed "on" was followed by one more "off"
        assert not scheduler.is_on
        scheduler.submit(plan(True))  # ignored from now on
        time.sleep(0.030)
        assert device.on_calls == 2
        started = time.perf_counter()
        records = scheduler.stop_trial()
        assert time.perf_counter() - started < 0.1
        assert [r.kind for r in records] == ["plan", "plan", "error"]
    finally:
        scheduler.close()


def test_stop_trial_switches_off_waits_for_it_and_returns_every_record(rig):
    scheduler, device = rig
    scheduler.submit(plan(True, Transition(time.perf_counter() + 5.0, False, "left", "top")))
    assert wait_until(lambda: states(device) == [True])
    records = scheduler.stop_trial("response")
    assert states(device) == [True, False]  # already off when stop_trial returns
    assert [(r.kind, r.on) for r in records] == [("plan", True), ("response", False)]
    assert scheduler.stop_trial("response") == [] and states(device) == [True, False]


def test_close_switches_off_stops_the_thread_and_can_be_repeated():
    device = RecordingDevice()
    scheduler = SignalScheduler(device).start()
    switch_interval = sys.getswitchinterval()
    scheduler.submit(plan(True))
    assert wait_until(lambda: states(device) == [True])
    scheduler.close()
    scheduler.close()
    assert states(device) == [True, False]
    assert not any(t.name == "signal-scheduler" and t.is_alive() for t in threading.enumerate())
    assert sys.getswitchinterval() >= switch_interval
    scheduler.submit(plan(True))
    time.sleep(0.02)
    assert states(device) == [True, False]


def _draw_loop_frame():
    """The trial loop's own per-frame work, on SDL's dummy video driver."""
    import pygame

    from experiment import display
    from experiment.calibration import make_configured_ir_frame_calibration

    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((1920, 1080))
    clock = pygame.time.Clock()
    layout = display.make_trial_layout(screen, make_configured_ir_frame_calibration(screen.get_size()), 10.0, 10.0, 12.0, True, 40.0)

    def frame() -> None:
        pygame.event.get()
        display.draw_trial(screen, layout, bar_width_mm=10.0, trial_index=1, is_practice=False, touch_pos=(900, 800), remaining_time_s=20.0)
        pygame.display.flip()
        clock.tick(60)

    return frame


@pytest.mark.parametrize("load", ["idle", "draw loop"])
def test_timed_transitions_are_on_time_on_this_machine(load, capsys):
    if load == "draw loop":
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        frame = _draw_loop_frame()
    else:
        frame = lambda: time.sleep(1 / 60)  # noqa: E731
    late, frames, spin_s = run_benchmark(3.0, frame)
    ordered = sorted(late)
    p95 = ordered[int(0.95 * len(ordered))]
    with capsys.disabled():
        print(
            f"\n  scheduler lateness, main loop {load}: n={len(late)} median {statistics.median(late):.3f} ms, "
            f"p95 {p95:.3f} ms, max {max(late):.3f} ms (busy-wait margin {1000 * spin_s:.1f} ms, "
            f"frame median {statistics.median(frames):.1f} ms)"
        )
    assert len(late) > 100
    # Loose bounds so a busy test machine does not fail the suite; the benchmark command reports the real figure.
    assert statistics.median(late) < 1.0 and p95 < 3.0


EXIT_SCRIPT = """
import os, signal, sys, time
from experiment.render_logic import Plan
from experiment.signal_scheduler import SignalScheduler, exit_cleanly_on_sigterm

class FileDevice:
    def on(self): open(sys.argv[1], "a").write("on\\n")
    def off(self): open(sys.argv[1], "a").write("off\\n")

scheduler = SignalScheduler(FileDevice(), watchdog_s=30.0).start()   # the watchdog must not be what saves us
scheduler.submit(Plan(time.perf_counter(), True, "left", "top", (), 0.0, 0.0))
time.sleep(0.1)
how = sys.argv[2]
if how == "exception":
    raise RuntimeError("crash in the trial loop")
if how == "sys_exit":
    sys.exit(3)
if how == "keyboard_interrupt":
    raise KeyboardInterrupt
if how == "sigterm":
    exit_cleanly_on_sigterm()
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(5)
"""


@pytest.mark.parametrize("how", ["falls_off_the_end", "exception", "sys_exit", "keyboard_interrupt", "sigterm"])
def test_signal_is_off_however_the_program_ends(how, tmp_path):
    if how == "sigterm" and sys.platform == "win32":
        pytest.skip("Windows cannot deliver a catchable SIGTERM")
    log = tmp_path / "device.log"
    subprocess.run([sys.executable, "-c", EXIT_SCRIPT, str(log), how], cwd=REPO_ROOT, capture_output=True, timeout=30)
    assert log.read_text().split() == ["on", "off"]


def test_lateness_only_counts_timed_transitions(rig):
    scheduler, _ = rig
    scheduler.submit(plan(True, Transition(time.perf_counter() + 0.010, False, "left", "top")))
    time.sleep(0.05)
    records = scheduler.stop_trial()
    assert [r.kind for r in records] == ["plan", "transition"]
    assert len(lateness_ms(records)) == 1
