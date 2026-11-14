"""Bluetooth LE link to the ESP32 that plays the top-edge cue on the Haptuator (firmware/esp32_cue)."""

# One thread owns the link and runs its own asyncio loop, as bleak needs one. The trial loop only hands messages
# over and never waits for Bluetooth. bleak is imported only when a session actually connects.

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from typing import Protocol

DEVICE_NAME = "EA-CUE"
CUE_UUID = "03c34c2a-b29f-4040-b812-5778ed2507cd"  # the ESP32's CUE characteristic; see firmware/esp32_cue/README.md
SCAN_TIMEOUT_S = 10.0
RETRY_S = 2.0
MAX_AGE_S = 0.1  # a message still unsent after this long is dropped: its pulse would be felt in the wrong place
POLL_S = 0.5  # how often a quiet link checks that it is still connected


def message_for(bar: str, direction: str) -> str:
    """The text the ESP32 expects for a top-edge crossing: ENTER_LEFT, EXIT_LEFT, ENTER_RIGHT or EXIT_RIGHT."""
    verb = {"entering": "ENTER", "leaving": "EXIT"}[direction]
    return f"{verb}_{bar.upper()}"


@dataclass(frozen=True)
class CueRecord:
    """One message the trial loop handed over, and what became of it."""

    message: str  # without the newline
    queued_s: float  # the caller's clock (perf_counter) when it was handed over
    write_start_s: float | None  # perf_counter; None if it was never written
    write_end_s: float | None
    error: str | None = None  # None: written to the link


class Transport(Protocol):
    @property
    def is_connected(self) -> bool: ...

    async def connect(self) -> None: ...

    async def write(self, data: bytes) -> None: ...

    async def disconnect(self) -> None: ...


class BleakTransport:
    """The ESP32 over Bluetooth LE, found by its advertised name."""

    def __init__(self, device_name: str = DEVICE_NAME, scan_timeout_s: float = SCAN_TIMEOUT_S) -> None:
        self.device_name = device_name
        self.scan_timeout_s = scan_timeout_s
        self._client = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected

    async def connect(self) -> None:
        from bleak import BleakClient, BleakScanner

        device = await BleakScanner.find_device_by_name(self.device_name, timeout=self.scan_timeout_s)
        if device is None:
            raise ConnectionError(f"no Bluetooth device named {self.device_name!r} found")
        client = BleakClient(device)
        await client.connect()
        self._client = client

    async def write(self, data: bytes) -> None:
        # Without response: the PC does not wait for the ESP32 to acknowledge each message.
        await self._client.write_gatt_char(CUE_UUID, data, response=False)

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.disconnect()


class CueLink:
    """Sends cue messages from its own thread, so the trial loop never waits for Bluetooth.

    Connects in the background and connects again after a drop. A message handed over while the link is down is
    not kept for later, and neither is one that could not be written within ``max_age_s``: a late pulse would be
    felt in the wrong place. Both are recorded with the reason instead.
    """

    def __init__(self, transport: Transport, *, retry_s: float = RETRY_S, max_age_s: float = MAX_AGE_S) -> None:
        self._transport = transport
        self._retry_s = retry_s
        self._max_age_s = max_age_s
        self._loop = asyncio.new_event_loop()
        self._queue: asyncio.Queue | None = None
        self._stop: asyncio.Event | None = None
        self._connected = threading.Event()
        self._connected_since = 0.0
        self._cond = threading.Condition()
        self._records: list[CueRecord] = []
        self._pending = 0
        self._closed = False
        self.last_error: str | None = None
        self._thread = threading.Thread(target=self._run, name="cue-link", daemon=True)

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def start(self) -> CueLink:
        self._thread.start()
        return self

    def wait_connected(self, timeout_s: float) -> bool:
        return self._connected.wait(timeout_s)

    def send(self, message: str, now_s: float) -> None:
        """Hand one message over (without the newline); returns at once."""
        with self._cond:
            if self._closed or not self._connected.is_set():
                self._records.append(CueRecord(message, now_s, None, None, "not connected"))
                return
            self._pending += 1
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, (message, now_s, time.perf_counter()))
        except RuntimeError:  # the loop has already closed
            self._record(CueRecord(message, now_s, None, None, "closed"))

    def flush(self, timeout_s: float = 1.0) -> bool:
        """Wait until every message handed over has been written or dropped."""
        with self._cond:
            return self._cond.wait_for(lambda: self._pending == 0, timeout_s)

    def drain(self) -> list[CueRecord]:
        with self._cond:
            records, self._records = self._records, []
        return records

    def close(self, timeout_s: float = 3.0) -> None:
        with self._cond:
            self._closed = True
        if self._thread.is_alive():
            try:
                self._loop.call_soon_threadsafe(self._request_stop)
            except RuntimeError:
                pass
            self._thread.join(timeout_s)

    def _request_stop(self) -> None:
        if self._stop is not None:
            self._stop.set()

    def _record(self, record: CueRecord) -> None:
        with self._cond:
            self._records.append(record)
            self._pending -= 1
            self._cond.notify_all()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._loop.close()

    async def _main(self) -> None:
        self._queue = asyncio.Queue()
        self._stop = asyncio.Event()
        while not self._stop.is_set() and not self._closed:
            try:
                if not await self._until_stop(self._transport.connect()):
                    break
            except Exception as exc:
                self.last_error = f"connect failed: {exc}"
                await self._until_stop(asyncio.sleep(self._retry_s))
                continue
            self.last_error = None
            self._connected_since = time.perf_counter()
            self._connected.set()
            try:
                await self._serve()
            finally:
                self._connected.clear()
                self._drop_queued("link dropped")
        try:
            await self._transport.disconnect()
        except Exception:
            pass
        self._drop_queued("closed")

    async def _until_stop(self, awaitable) -> bool:
        """Run ``awaitable`` unless a stop comes first; False if it was stopped."""
        task = asyncio.ensure_future(awaitable)
        stop = asyncio.ensure_future(self._stop.wait())
        done, _ = await asyncio.wait({task, stop}, return_when=asyncio.FIRST_COMPLETED)
        stop.cancel()
        if task not in done:
            task.cancel()
            return False
        task.result()  # raises what the awaitable raised
        return True

    async def _serve(self) -> None:
        stop = asyncio.ensure_future(self._stop.wait())
        try:
            while self._transport.is_connected:
                get = asyncio.ensure_future(self._queue.get())
                done, _ = await asyncio.wait({get, stop}, timeout=POLL_S, return_when=asyncio.FIRST_COMPLETED)
                if get not in done:
                    get.cancel()
                    if stop in done:
                        return
                    continue
                message, queued_s, handed_s = get.result()
                if handed_s < self._connected_since:
                    self._record(CueRecord(message, queued_s, None, None, "link dropped"))
                    continue
                if time.perf_counter() - handed_s > self._max_age_s:
                    self._record(CueRecord(message, queued_s, None, None, "too late"))
                    continue
                start_s = time.perf_counter()
                try:
                    await self._transport.write(f"{message}\n".encode("ascii"))
                except Exception as exc:
                    self.last_error = f"write failed: {exc}"
                    self._record(CueRecord(message, queued_s, start_s, time.perf_counter(), self.last_error))
                    return
                self._record(CueRecord(message, queued_s, start_s, time.perf_counter()))
        finally:
            stop.cancel()

    def _drop_queued(self, reason: str) -> None:
        while self._queue is not None and not self._queue.empty():
            message, queued_s, _ = self._queue.get_nowait()
            self._record(CueRecord(message, queued_s, None, None, reason))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.cue_link",
        description="Send test cues to the ESP32 over Bluetooth. The Haptuator should pulse once per message.",
    )
    parser.add_argument("--device", default=DEVICE_NAME, help="the ESP32's Bluetooth name")
    parser.add_argument("--count", type=int, default=10, help="messages to send")
    parser.add_argument("--interval-ms", type=float, default=500.0, help="time between messages")
    parser.add_argument("--connect-timeout-s", type=float, default=20.0)
    args = parser.parse_args(argv)

    link = CueLink(BleakTransport(args.device)).start()
    try:
        print(f"Looking for {args.device} ...")
        if not link.wait_connected(args.connect_timeout_s):
            print(f"FAIL: not connected ({link.last_error or 'timed out'})")
            return 1
        print(f"Connected. Sending {args.count} messages; the Haptuator should pulse once for each.")
        cycle = [message_for(bar, direction) for bar in ("left", "right") for direction in ("entering", "leaving")]
        for i in range(args.count):
            link.send(cycle[i % len(cycle)], time.perf_counter())
            print(f"  {i + 1:>3} {cycle[i % len(cycle)]}")
            time.sleep(args.interval_ms / 1000.0)
        link.flush(2.0)
        records = link.drain()
        written = [r for r in records if r.error is None]
        failed = [r for r in records if r.error is not None]
        if written:
            delays_ms = [1000.0 * (r.write_end_s - r.queued_s) for r in written]
            print(
                f"Written: {len(written)}/{args.count}. Hand-over to written on the PC: "
                f"median {statistics.median(delays_ms):.1f} ms, max {max(delays_ms):.1f} ms."
            )
        for record in failed:
            print(f"  not sent: {record.message} ({record.error})")
        print("This is the PC side only. The delay to the pulse itself is measured with the oscilloscope.")
        return 0 if len(written) == args.count else 1
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
