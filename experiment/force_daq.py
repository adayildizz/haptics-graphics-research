"""The force sensor is read in a process of its own; the experiment only ever takes the newest value.

ATI's DAQ F/T setup turns the Nano17's six strain-gauge signals into voltages that a DAQ card samples on its
own clock. Reading the card from a thread of the experiment would make it compete with the signal scheduler
for Python's interpreter lock, so a separate process does it: it reads a few samples at a time, publishes the
newest one in shared memory and writes every sample to its own SQLite file. The experiment calls
``ForceReader.latest()``, which at most waits the few microseconds the reader needs to publish a sample.

Times are on ``time.perf_counter``, the clock the scheduler and the trace use. No sample is timed directly:
the card's clock fixes the spacing, and every read fixes a latest possible time ("this many samples had
arrived by now"). ``LiveClock`` turns that into times while running; ``fit_sample_times`` does it for a whole
file afterwards and also removes the slow drift between the card's clock and the computer's.

Only a fake source exists so far. The NI-DAQmx source follows once the card model, the channel wiring and
the sensor's .cal file are known. Nothing in the experiment imports this module yet.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from .signal_scheduler import GIL_SWITCH_INTERVAL_S

RATE_HZ = 1000.0
CHANNELS = 6  # Nano17 through DAQ F/T: one voltage per strain gauge (ai0-ai5)
READ_BLOCK = 2  # samples per read; at 1 kHz the published value is then about 2 ms old at most
READ_TIMEOUT_S = 0.5
CLOCK_WINDOW_S = 2.0  # reads that anchor the live times; short enough that 50 ppm of drift stays below 0.1 ms
COMMIT_INTERVAL_S = 0.25
PARENT_CHECK_S = 0.5
HANDSHAKE_ROUNDS = 20
START_TIMEOUT_S = 10.0  # a spawned process imports Python afresh, which takes a while on a cold machine
LATEST_WAIT_S = 0.002
FORCE_SCHEMA_VERSION = 1

# Shared-memory layout of the newest sample.
_SEQ, _INDEX, _T, _PUBLISHED, _VOLTS = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class ForceSample:
    """The newest sample the reader process has published."""

    index: int  # 0 is the first sample of the run
    t_s: float  # when the card took it, on ``time.perf_counter``
    published_s: float  # when the reader process made it available
    volts: tuple[float, ...]


@dataclass(frozen=True)
class ReadStamp:
    """By ``t_s`` the card had delivered ``total`` samples, so sample ``total - 1`` was taken no later than ``t_s``."""

    t_s: float
    total: int


@dataclass(frozen=True)
class ReaderSummary:
    samples: int
    reads: int
    clock_offset_s: float
    path: Path
    error: str | None = None


@dataclass(frozen=True)
class ForceRecording:
    metadata: dict[str, str]
    t_s: list[float]
    volts: list[tuple[float, ...]]


class ForceSource(Protocol):
    rate_hz: float
    channels: int

    def start(self) -> None: ...

    def read(self, min_samples: int, timeout_s: float) -> list[tuple[float, ...]]:
        """Block until at least ``min_samples`` new samples exist; return every new sample, oldest first."""
        ...

    def close(self) -> None: ...


def fake_volts(index: int, rate_hz: float = RATE_HZ, channels: int = CHANNELS) -> tuple[float, ...]:
    """What the fake card reads as sample ``index``: a slow sine per channel, so every stored sample can be checked."""
    t = index / rate_hz
    return tuple(0.5 * math.sin(2.0 * math.pi * 0.5 * t + channel) for channel in range(channels))


class FakeForceSource:
    """Stands in for the DAQ card: samples appear on its own clock, which may run ``drift_ppm`` fast or slow."""

    def __init__(
        self,
        rate_hz: float = RATE_HZ,
        channels: int = CHANNELS,
        *,
        drift_ppm: float = 0.0,
        fail_at_start: bool = False,
        fail_after: int | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rate_hz = rate_hz
        self.channels = channels
        self.drift_ppm = drift_ppm
        self._fail_at_start = fail_at_start
        self._fail_after = fail_after
        self._clock = clock
        self._sleep = sleep
        self._start_s: float | None = None
        self._delivered = 0

    def start(self) -> None:
        if self._fail_at_start:
            raise RuntimeError("fake DAQ card could not be opened")
        self._start_s = self._clock()

    def sample_time(self, index: int) -> float:
        """When the fake card takes sample ``index``; the truth that the clock mapping should recover."""
        if self._start_s is None:
            raise RuntimeError("the fake source has not been started")
        return self._start_s + index / (self.rate_hz * (1.0 + 1e-6 * self.drift_ppm))

    def read(self, min_samples: int, timeout_s: float) -> list[tuple[float, ...]]:
        if self._start_s is None:
            raise RuntimeError("the fake source has not been started")
        rate = self.rate_hz * (1.0 + 1e-6 * self.drift_ppm)
        needed = self._delivered + max(1, min_samples)
        deadline = self._clock() + timeout_s
        while True:
            now = self._clock()
            arrived = int((now - self._start_s) * rate) + 1  # sample 0 is taken at start
            if arrived >= needed:
                break
            if now >= deadline:
                raise TimeoutError(f"the fake card delivered no {min_samples} samples within {timeout_s} s")
            self._sleep(max(0.0, min(self.sample_time(needed - 1) - now, deadline - now)))
        if self._fail_after is not None and arrived > self._fail_after:
            raise RuntimeError("fake DAQ card failed while reading")
        rows = [fake_volts(index, self.rate_hz, self.channels) for index in range(self._delivered, arrived)]
        self._delivered = arrived
        return rows

    def close(self) -> None:
        self._start_s = None


class LiveClock:
    """Sample times while running: the card's spacing, anchored at the earliest start the recent reads allow."""

    def __init__(self, rate_hz: float, window_s: float = CLOCK_WINDOW_S) -> None:
        self.period_s = 1.0 / rate_hz
        self.window_s = window_s
        # (read time, start time it implies), kept increasing in the implied start: a sliding-window minimum.
        self._anchors: deque[tuple[float, float]] = deque()

    def add(self, stamp: ReadStamp) -> None:
        implied_start = stamp.t_s - (stamp.total - 1) * self.period_s
        while self._anchors and self._anchors[-1][1] >= implied_start:
            self._anchors.pop()
        self._anchors.append((stamp.t_s, implied_start))
        while self._anchors[0][0] < stamp.t_s - self.window_s:
            self._anchors.popleft()

    def time_of(self, index: int) -> float:
        if not self._anchors:
            raise ValueError("no read has been added yet")
        return self._anchors[0][1] + index * self.period_s


def fit_sample_times(stamps: Sequence[ReadStamp], rate_hz: float) -> tuple[float, float]:
    """``(t0, period)``, with sample ``k`` taken at ``t0 + k * period``, from all reads of a run.

    The period comes from a least-squares line through the reads, which takes out the drift between the card's
    clock and the computer's. The line is then lowered until no read comes before a sample it returned.
    """
    if not stamps:
        raise ValueError("no reads to fit")
    xs = [stamp.total - 1 for stamp in stamps]
    ys = [stamp.t_s for stamp in stamps]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    period = 1.0 / rate_hz
    if sxx > 0.0:
        period = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    t0 = min(y - period * x for x, y in zip(xs, ys))
    return t0, period


# --- storage ---------------------------------------------------------------------------------


def _create_schema(connection: sqlite3.Connection, channels: int) -> None:
    columns = ", ".join(f"v{channel} REAL NOT NULL" for channel in range(channels))
    connection.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS force_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS force_samples (sample_index INTEGER PRIMARY KEY, {columns});
        CREATE TABLE IF NOT EXISTS force_reads (total INTEGER NOT NULL, t_s REAL NOT NULL);
        """
    )


class _StoreWriter:
    """Writes samples and read stamps from a thread of the reader process, so a slow disk never delays a read."""

    def __init__(self, path: Path, channels: int, metadata: dict[str, Any]) -> None:
        self._path = path
        self._channels = channels
        self._metadata = {key: str(value) for key, value in metadata.items()}
        self._queue: queue.SimpleQueue[tuple[int, list[tuple[float, ...]], ReadStamp] | None] = queue.SimpleQueue()
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="force-store", daemon=True)
        self._thread.start()

    def put(self, first_index: int, rows: list[tuple[float, ...]], stamp: ReadStamp) -> None:
        self._queue.put((first_index, rows, stamp))

    def close(self, timeout_s: float = 5.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout_s)

    def _run(self) -> None:
        connection = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self._path)
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            _create_schema(connection, self._channels)
            metadata = {"schema_version": str(FORCE_SCHEMA_VERSION), **self._metadata}
            connection.executemany("INSERT OR REPLACE INTO force_metadata VALUES (?, ?)", metadata.items())
            connection.commit()
            placeholders = ", ".join("?" * (self._channels + 1))
            insert_samples = f"INSERT INTO force_samples VALUES ({placeholders})"
            last_commit = time.perf_counter()
            while True:
                item = self._queue.get()
                if item is None:
                    break
                first_index, rows, stamp = item
                connection.executemany(insert_samples, ((first_index + i, *row) for i, row in enumerate(rows)))
                connection.execute("INSERT INTO force_reads VALUES (?, ?)", (stamp.total, stamp.t_s))
                if time.perf_counter() - last_commit >= COMMIT_INTERVAL_S:
                    connection.commit()
                    last_commit = time.perf_counter()
            connection.commit()
        except BaseException as exc:  # noqa: BLE001 -- reported by the reader process
            self.error = exc
        finally:
            if connection is not None:
                connection.close()


def load_force(path: str | Path) -> ForceRecording:
    """A whole force file, every sample timed on ``time.perf_counter`` by ``fit_sample_times``."""
    connection = sqlite3.connect(Path(path))
    try:
        metadata = dict(connection.execute("SELECT key, value FROM force_metadata"))
        stamps = [ReadStamp(t_s, total) for total, t_s in connection.execute("SELECT total, t_s FROM force_reads")]
        rows = connection.execute("SELECT * FROM force_samples ORDER BY sample_index").fetchall()
    finally:
        connection.close()
    if not rows:
        return ForceRecording(metadata, [], [])
    t0, period = fit_sample_times(stamps, float(metadata["rate_hz"]))
    return ForceRecording(metadata, [t0 + row[0] * period for row in rows], [tuple(row[1:]) for row in rows])


# --- reader process --------------------------------------------------------------------------


def _child_main(
    conn: Any,
    shared: Any,
    stop: Any,
    path: str,
    make_source: Callable[[], ForceSource],
    read_block: int,
) -> None:
    """Entry point of the reader process."""
    sys.setswitchinterval(GIL_SWITCH_INTERVAL_S)  # the store thread must not hold up a read for long
    offset = 0.0
    while True:  # clock handshake: the parent measures how this process's clock relates to its own
        message = conn.recv()
        if message[0] == "ping":
            conn.send(time.perf_counter())
        else:
            offset = message[1]
            break

    def now() -> float:
        return time.perf_counter() - offset

    source: ForceSource | None = None
    writer: _StoreWriter | None = None
    started = False
    total = reads = 0
    error: str | None = None
    try:
        source = make_source()
        if source.channels != len(shared) - _VOLTS:
            raise ValueError(f"the source has {source.channels} channels, the reader was set up for {len(shared) - _VOLTS}")
        clock = LiveClock(source.rate_hz)
        writer = _StoreWriter(
            Path(path),
            source.channels,
            {
                "rate_hz": source.rate_hz,
                "channels": source.channels,
                "source": type(source).__name__,
                "clock_offset_s": offset,
                "read_block": read_block,
            },
        )
        source.start()
        conn.send(("started", offset))
        started = True
        raw = shared.get_obj()
        lock = shared.get_lock()
        parent = mp.parent_process()
        next_parent_check = now() + PARENT_CHECK_S
        while not stop.is_set():
            rows = source.read(read_block, READ_TIMEOUT_S)
            stamp = ReadStamp(now(), total + len(rows))
            clock.add(stamp)
            newest = stamp.total - 1
            packet = [float(reads + 1), float(newest), clock.time_of(newest), now(), *rows[-1]]
            with lock:  # one slice assignment, so the lock is held for microseconds
                raw[:] = packet
            writer.put(total, rows, stamp)
            total, reads = stamp.total, reads + 1
            if writer.error is not None:
                raise writer.error
            if stamp.t_s >= next_parent_check:
                next_parent_check = stamp.t_s + PARENT_CHECK_S
                if parent is not None and not parent.is_alive():
                    break
    except BaseException:  # noqa: BLE001 -- everything is reported to the parent
        error = traceback.format_exc()
    finally:
        if source is not None:
            try:
                source.close()
            except BaseException:  # noqa: BLE001
                pass
        if writer is not None:
            writer.close()
            if writer.error is not None and error is None:
                exc = writer.error
                error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        try:
            if started:
                conn.send(("stopped", {"samples": total, "reads": reads, "error": error}))
            else:
                conn.send(("error", error or "the reader process ended before the source started"))
        except (BrokenPipeError, OSError):
            pass


class ForceReader:
    """Runs the reader process; ``latest()`` hands the experiment the newest sample without waiting."""

    def __init__(
        self,
        path: str | Path,
        make_source: Callable[[], ForceSource],
        *,
        channels: int = CHANNELS,
        read_block: int = READ_BLOCK,
    ) -> None:
        """``make_source`` runs inside the reader process, so it must be picklable (a class or a ``functools.partial``)."""
        self.path = Path(path)
        self.channels = channels
        self._ctx = mp.get_context("spawn")  # the only method Windows has; the same everywhere keeps tests honest
        self._conn, child_conn = self._ctx.Pipe()
        self._shared = self._ctx.Array("d", _VOLTS + channels)
        self._raw = self._shared.get_obj()
        self._stop = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_child_main,
            args=(child_conn, self._shared, self._stop, str(self.path), make_source, read_block),
            name="force-reader",
            daemon=True,
        )
        self._last: ForceSample | None = None
        self._last_seq = 0.0
        self.clock_offset_s = 0.0
        self.handshake_rtt_s = math.inf
        self._summary: ReaderSummary | None = None

    def start(self, timeout_s: float = START_TIMEOUT_S) -> ForceReader:
        """Starts the process, measures its clock offset, and returns once the source is reading."""
        self._process.start()
        best_rtt, best_offset = math.inf, 0.0
        for _ in range(HANDSHAKE_ROUNDS):
            sent = time.perf_counter()
            self._conn.send(("ping",))
            if not self._conn.poll(timeout_s):
                self._abort()
                raise RuntimeError("the force reader process did not answer")
            child_s = self._conn.recv()
            received = time.perf_counter()
            if received - sent < best_rtt:
                best_rtt, best_offset = received - sent, child_s - (sent + received) / 2.0
        self.clock_offset_s, self.handshake_rtt_s = best_offset, best_rtt
        self._conn.send(("go", best_offset))
        if not self._conn.poll(timeout_s):
            self._abort()
            raise RuntimeError("the force source did not start in time")
        kind, payload = self._conn.recv()
        if kind != "started":
            self._abort()
            raise RuntimeError(f"the force source could not start:\n{payload}")
        return self

    def latest(self) -> ForceSample | None:
        """The newest published sample, or None before the first.

        Waits only while the reader process is writing a sample, which takes microseconds; after
        ``LATEST_WAIT_S`` it gives up and returns the sample from the previous call.
        """
        lock = self._shared.get_lock()
        give_up_s = None
        while not lock.acquire(False):
            if give_up_s is None:
                give_up_s = time.perf_counter() + LATEST_WAIT_S
            elif time.perf_counter() > give_up_s:
                return self._last
        try:
            values = self._raw[:]
        finally:
            lock.release()
        if values[_SEQ] != self._last_seq:
            self._last_seq = values[_SEQ]
            self._last = ForceSample(int(values[_INDEX]), values[_T], values[_PUBLISHED], tuple(values[_VOLTS:]))
        return self._last

    @property
    def running(self) -> bool:
        return self._process.is_alive() and not self._stop.is_set()

    def stop(self, timeout_s: float = 5.0) -> ReaderSummary:
        """Stops reading, waits until every sample is on disk, and says how many there were. Safe to repeat."""
        if self._summary is not None:
            return self._summary
        self._stop.set()
        stats: dict[str, Any] = {"samples": 0, "reads": 0, "error": "the force reader process did not report back"}
        if self._process.pid is not None and self._conn.poll(timeout_s):
            kind, payload = self._conn.recv()
            stats = payload if kind == "stopped" else {"samples": 0, "reads": 0, "error": payload}
        self._abort()
        self._summary = ReaderSummary(stats["samples"], stats["reads"], self.clock_offset_s, self.path, stats["error"])
        return self._summary

    def _abort(self) -> None:
        self._stop.set()
        if self._process.pid is None:
            return
        self._process.join(2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(2.0)


# --- self-check (fake source; never opens a DAQ card) ----------------------------------------


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def run_check(seconds: float, path: Path, poll_hz: float = 60.0) -> tuple[ReaderSummary, list[float], list[float]]:
    """Read a fake card for ``seconds`` while polling as the trial loop would; returns ages and publish lags in ms."""
    reader = ForceReader(path, partial(FakeForceSource)).start()
    ages: list[float] = []
    lags: list[float] = []
    end = time.perf_counter() + seconds
    try:
        while time.perf_counter() < end:
            time.sleep(1.0 / poll_hz)
            sample = reader.latest()
            if sample is not None:
                ages.append(1000.0 * (time.perf_counter() - sample.t_s))
                lags.append(1000.0 * (sample.published_s - sample.t_s))
    finally:
        summary = reader.stop()
    return summary, ages, lags


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.force_daq",
        description="Read a FAKE force source in its own process and measure how fresh the newest value is. "
        "No DAQ card is opened.",
    )
    parser.add_argument("--seconds", type=float, default=10.0, help="Duration of the measurement (default 10).")
    parser.add_argument("--out", type=Path, help="Keep the force file here instead of a temporary folder.")
    args = parser.parse_args(argv)

    import platform

    print(f"{platform.platform()} · Python {platform.python_version()}")
    with tempfile.TemporaryDirectory() as folder:
        path = args.out or Path(folder) / "force_check.sqlite3"
        summary, ages, lags = run_check(args.seconds, path)
        recording = load_force(path)
    if summary.error is not None or not ages:
        print(f"FAIL: {summary.error or 'no sample was published'}")
        return 1
    span_s = recording.t_s[-1] - recording.t_s[0] if len(recording.t_s) > 1 else 0.0
    rate = (len(recording.t_s) - 1) / span_s if span_s > 0 else 0.0
    p95 = _percentile(ages, 0.95)
    print(
        f"samples {summary.samples} in {summary.reads} reads, rate on the computer's clock {rate:.2f} Hz, "
        f"clock offset between the processes {1e6 * summary.clock_offset_s:.0f} us"
    )
    print(
        f"age of the newest value when the loop looked: median {statistics.median(ages):.2f}  p95 {p95:.2f}  "
        f"max {max(ages):.2f} ms   published after being taken: median {statistics.median(lags):.2f}  "
        f"max {max(lags):.2f} ms"
    )
    ok = p95 <= 5.0 and summary.samples == len(recording.t_s)
    print("PASS: p95 age <= 5 ms, every sample stored" if ok else "FAIL -- report this before using the force sensor")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
