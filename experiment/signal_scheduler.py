"""One thread owns the signal generator and switches it at the planned instants."""

from __future__ import annotations

import argparse
import atexit
import signal
import statistics
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .render_logic import Plan, Transition

WATCHDOG_S = 0.1
GIL_SWITCH_INTERVAL_S = 0.0005
SPIN_RANGE_S = (0.001, 0.008)
NAP_S = 0.002  # long timed waits overshoot by several ms on some systems; short ones do not


class SignalDevice(Protocol):
    def on(self) -> None: ...

    def off(self) -> None: ...


class NullDevice:
    """No generator attached: the scheduler still runs, so its timing can be logged and tested."""

    def on(self) -> None:
        pass

    def off(self) -> None:
        pass


@dataclass(frozen=True)
class WriteRecord:
    """One switch of the output. Times are on the scheduler's clock (``time.perf_counter``)."""

    kind: str  # "plan" | "transition" | "watchdog" | "error" | the reason given to ``stop_trial``
    on: bool
    planned_s: float
    write_start_s: float
    write_end_s: float
    bar: str | None = None
    edge: str = ""
    x_mm: float | None = None
    y_mm: float | None = None
    vx_mm_s: float | None = None
    vy_mm_s: float | None = None


def measure_wait_overshoot(waits: int = 40, wait_s: float = NAP_S) -> float:
    """Worst lateness of a timed wait on this machine right now, in seconds."""
    condition = threading.Condition()
    worst = 0.0
    with condition:
        for _ in range(waits):
            start = time.perf_counter()
            condition.wait(wait_s)
            worst = max(worst, time.perf_counter() - start - wait_s)
    return worst


def exit_cleanly_on_sigterm() -> None:
    """Turn SIGTERM into a normal exit so ``finally`` blocks and ``atexit`` still switch the signal off."""
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(143))


class SignalScheduler:
    """Executes the latest ``Plan``: its state now, then its timed transitions, with a watchdog."""

    def __init__(
        self,
        device: SignalDevice | None,
        *,
        watchdog_s: float = WATCHDOG_S,
        spin_s: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._device: SignalDevice = device if device is not None else NullDevice()
        self.watchdog_s = watchdog_s
        self.spin_s = spin_s
        self._clock = clock
        self._cond = threading.Condition()
        self._generation = 0
        self._applied = 0
        self._new_plan: Plan | None = None
        self._plan: Plan | None = None
        self._pending: list[Transition] = []
        self._last_plan_s = 0.0
        self._stop_reason: str | None = None
        self._stopped = threading.Event()
        self._closing = False
        self._closed = False
        self._on = False
        self._records: list[WriteRecord] = []
        self._error: BaseException | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="signal-scheduler", daemon=True)
        self._old_switch_interval: float | None = None

    @property
    def error(self) -> BaseException | None:
        return self._error

    @property
    def is_on(self) -> bool:
        return self._on

    def start(self) -> SignalScheduler:
        """Starts the thread, measures the busy-wait margin if none was given, and registers a safe exit."""
        self._old_switch_interval = sys.getswitchinterval()
        # A shorter GIL switch interval bounds how long the main loop can keep this thread waiting.
        sys.setswitchinterval(min(self._old_switch_interval, GIL_SWITCH_INTERVAL_S))
        self._thread.start()
        self._ready.wait()
        atexit.register(self.close)
        return self

    def submit(self, plan: Plan) -> None:
        """Replace whatever was planned with ``plan``; also the heartbeat that keeps the watchdog quiet."""
        with self._cond:
            if self._error is not None or self._closing:
                return
            self._new_plan = plan
            self._last_plan_s = self._clock()
            self._generation += 1
            self._cond.notify()

    def flush(self, timeout_s: float = 1.0) -> bool:
        """Wait until the latest submitted plan's state has been written; True if it has."""
        with self._cond:
            return self._cond.wait_for(lambda: self._applied == self._generation or self._closing, timeout_s)

    def stop_trial(self, reason: str = "trial_end", timeout_s: float = 1.0) -> list[WriteRecord]:
        """Switch off, drop the plan, wait until that is done, and hand back every write not yet collected."""
        if self._thread.is_alive() and self._error is None:
            with self._cond:
                self._stopped.clear()
                self._stop_reason = reason
                self._new_plan = None
                self._generation += 1
                self._cond.notify()
            self._stopped.wait(timeout_s)
        return self.drain()

    def drain(self) -> list[WriteRecord]:
        with self._cond:
            records, self._records = self._records, []
        return records

    def close(self) -> None:
        """Signal off, thread stopped. Safe to call more than once and from ``atexit``."""
        if self._closed:
            return
        self._closed = True
        self.stop_trial("shutdown")
        with self._cond:
            self._closing = True
            self._generation += 1
            self._cond.notify()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._old_switch_interval is not None:
            sys.setswitchinterval(self._old_switch_interval)
        atexit.unregister(self.close)

    # --- scheduler thread -------------------------------------------------------------------

    def _run(self) -> None:
        if self.spin_s is None:
            low, high = SPIN_RANGE_S
            self.spin_s = min(max(measure_wait_overshoot() + 0.0005, low), high)
        self._ready.set()
        try:
            while self._step():
                pass
        except BaseException as exc:  # noqa: BLE001 -- nothing may leave the output on
            self._fail(exc)
        finally:
            self._stopped.set()

    def _step(self) -> bool:
        """Wait for the next thing to do and do it; False once the scheduler is closing."""
        with self._cond:
            while True:
                if self._closing:
                    return False
                generation = self._generation
                if self._stop_reason is not None or self._new_plan is not None:
                    due_s, action = 0.0, "now"
                    break
                due_s, action = self._next_deadline()
                if action is None:
                    self._cond.wait()
                    continue
                # Only an edge needs the exact instant; the watchdog can simply wake up on time.
                margin = (self.spin_s or 0.0) if action == "transition" else 0.0
                remaining = due_s - self._clock() - margin
                if remaining <= 0:
                    break
                self._cond.wait(min(remaining, NAP_S) if action == "transition" else remaining)
            stop_reason, plan = self._stop_reason, self._new_plan
            self._stop_reason = self._new_plan = None
            if stop_reason is not None or plan is not None:
                self._pending = []
            if plan is not None:
                self._plan, self._pending = plan, list(plan.transitions)

        if stop_reason is not None:
            if self._on:
                self._write(False, stop_reason, self._clock(), None)
            self._stopped.set()
        elif plan is not None:
            if plan.on != self._on:
                self._write(plan.on, "plan", plan.t_s, plan, plan.bar, plan.edge)
        if stop_reason is not None or plan is not None:
            with self._cond:
                self._applied = generation
                self._cond.notify_all()
        elif action == "transition":
            self._run_transition(generation, due_s)
        elif action == "watchdog":
            self._run_watchdog(generation)
        return True

    def _next_deadline(self) -> tuple[float, str | None]:
        """The earlier of the next planned transition and the watchdog; called with the lock held."""
        due_s, action = 0.0, None
        if self._pending:
            due_s, action = self._pending[0].t_s, "transition"
        if self._on or self._pending:
            watchdog_due = self._last_plan_s + self.watchdog_s
            if action is None or watchdog_due < due_s:
                due_s, action = watchdog_due, "watchdog"
        return due_s, action

    def _spin_until(self, due_s: float, generation: int) -> bool:
        """Busy-wait to the exact instant; False if a newer plan or a stop arrived meanwhile."""
        while self._clock() < due_s:
            if self._generation != generation:
                return False
        return self._generation == generation

    def _run_transition(self, generation: int, due_s: float) -> None:
        if not self._spin_until(due_s, generation):
            return
        with self._cond:
            if self._generation != generation or not self._pending:
                return
            transition = self._pending.pop(0)
        if transition.on != self._on:
            self._write(transition.on, "transition", transition.t_s, self._plan, transition.bar, transition.edge)

    def _run_watchdog(self, generation: int) -> None:
        with self._cond:
            if self._generation != generation:
                return
            self._pending = []
            due_s = self._last_plan_s + self.watchdog_s
        if self._on:
            self._write(False, "watchdog", due_s, self._plan)

    def _write(
        self, on: bool, kind: str, planned_s: float, plan: Plan | None, bar: str | None = None, edge: str = ""
    ) -> None:
        start = self._clock()
        try:
            self._device.on() if on else self._device.off()
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc)
            return
        end = self._clock()
        self._on = on
        self._record(WriteRecord(kind, on, planned_s, start, end, bar, edge, *self._where(plan)))

    @staticmethod
    def _where(plan: Plan | None) -> tuple[float | None, float | None, float | None, float | None]:
        if plan is None:
            return None, None, None, None
        return plan.x_mm, plan.y_mm, plan.vx_mm_s, plan.vy_mm_s

    def _record(self, record: WriteRecord) -> None:
        with self._cond:
            self._records.append(record)

    def _fail(self, exc: BaseException) -> None:
        """Remember the error, try once more to switch off, and stop accepting plans."""
        start = self._clock()
        try:
            self._device.off()
        except BaseException:  # noqa: BLE001 -- the device is already failing
            pass
        with self._cond:
            self._error = exc
            self._pending = []
            self._new_plan = self._stop_reason = None
            self._closing = True
        self._on = False
        self._record(WriteRecord("error", False, start, start, self._clock()))
        self._stopped.set()


# --- timing benchmark (fake device; never touches the generator) ----------------------------


class RecordingDevice:
    """Fake generator that only notes when it was switched."""

    def __init__(self) -> None:
        self.states: list[tuple[float, bool]] = []

    def on(self) -> None:
        self.states.append((time.perf_counter(), True))

    def off(self) -> None:
        self.states.append((time.perf_counter(), False))


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def lateness_ms(records: Sequence[WriteRecord]) -> list[float]:
    """Planned-to-write-start delay of every timed transition, in ms."""
    return [1000.0 * (r.write_start_s - r.planned_s) for r in records if r.kind == "transition"]


def run_benchmark(seconds: float, frame: Callable[[], None]) -> tuple[list[float], list[float], float]:
    """Feed the scheduler one plan per ``frame()`` call, as the trial loop does; returns lateness and frame times."""
    scheduler = SignalScheduler(RecordingDevice()).start()
    records: list[WriteRecord] = []
    frames: list[float] = []
    on = False
    end = time.perf_counter() + seconds
    last = time.perf_counter()
    try:
        while time.perf_counter() < end:
            now = time.perf_counter()
            # A switch 4-14 ms ahead, as when the finger is about to cross an edge.
            ahead = 0.004 + 0.010 * ((len(frames) * 7919) % 100) / 100.0
            on = not on
            scheduler.submit(Plan(now, not on, None, "", (Transition(now + ahead, on, "left", "top"),), 0.0, 0.0))
            frame()
            records.extend(scheduler.drain())
            frames.append(1000.0 * (time.perf_counter() - last))
            last = time.perf_counter()
    finally:
        records.extend(scheduler.stop_trial())
        spin_s = scheduler.spin_s or 0.0
        scheduler.close()
    return lateness_ms(records), frames, spin_s


def _report(label: str, late: Sequence[float], frames: Sequence[float], spin_s: float) -> bool:
    if not late:
        print(f"{label:28s} no timed transition was executed")
        return False
    p95 = _percentile(late, 0.95)
    print(
        f"{label:28s} n={len(late):4d}  lateness median {statistics.median(late):6.3f}  p95 {p95:6.3f}  "
        f"max {max(late):6.3f} ms   busy-wait margin {1000 * spin_s:.1f} ms   "
        f"frame median {statistics.median(frames):5.1f} ms"
    )
    return p95 <= 1.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.signal_scheduler",
        description="Measure how late the scheduler thread switches a FAKE device. No generator is contacted.",
    )
    parser.add_argument("--seconds", type=float, default=10.0, help="Duration of each measurement (default 10).")
    parser.add_argument("--no-window", action="store_true", help="Skip the measurement under the drawing loop.")
    args = parser.parse_args(argv)

    import platform

    print(f"{platform.platform()} · Python {platform.python_version()}")
    ok = _report("main loop idle (60 Hz)", *run_benchmark(args.seconds, lambda: time.sleep(1 / 60)))
    if not args.no_window:
        import pygame

        from . import display
        from .calibration import make_configured_ir_frame_calibration

        pygame.init()
        screen = display.init_window(fullscreen=False, size=(1280, 720))
        clock = pygame.time.Clock()
        calibration = make_configured_ir_frame_calibration(screen.get_size())
        layout = display.make_trial_layout(screen, calibration, 10.0, 10.0, 12.0, True, 40.0)

        def draw_frame() -> None:
            pygame.event.get()
            display.draw_trial(
                screen, layout, bar_width_mm=10.0, trial_index=1, is_practice=False,
                touch_pos=pygame.mouse.get_pos(), remaining_time_s=20.0,
            )  # fmt: skip
            pygame.display.flip()
            clock.tick(60)

        ok = _report("under the trial's draw loop", *run_benchmark(args.seconds, draw_frame)) and ok
        pygame.quit()
    print("PASS: p95 <= 1 ms" if ok else "FAIL: p95 > 1 ms -- report this before running a session")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
