"""Non-blocking trace recorder for the main experiment loop."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .trace_store import AttemptDefinition, AttemptOutcome, CursorSample, Side, TraceStore

SAMPLE_BATCH_SIZE = 256


@dataclass(frozen=True)
class _Command:
    kind: str
    payload: tuple[Any, ...]


class AsyncTraceRecorder:
    """Send trace writes to a dedicated SQLite worker thread.

    The experiment thread only appends samples to an in-memory list and places
    completed batches on an unbounded in-memory queue. It never waits for disk
    I/O while a trial is running.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._origin_ns = time.perf_counter_ns()
        self._queue: queue.SimpleQueue[_Command] = queue.SimpleQueue()
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="experiment-trace-writer",
            daemon=False,
        )
        self._thread.start()

    @property
    def error(self) -> BaseException | None:
        return self._error

    def elapsed_us(self) -> int:
        return (time.perf_counter_ns() - self._origin_ns) // 1_000

    def create_session(
        self,
        *,
        session_id: str,
        participant_id: str,
        started_at: str,
        config: dict[str, Any],
        calibration: dict[str, Any],
        app_version: str,
    ) -> None:
        self._submit(
            "create_session",
            session_id,
            participant_id,
            started_at,
            config,
            calibration,
            app_version,
        )

    def start_attempt(self, attempt_key: str, definition: AttemptDefinition) -> AttemptTraceBuffer:
        self._submit("start_attempt", attempt_key, definition)
        return AttemptTraceBuffer(self, attempt_key)

    def set_metadata(self, key: str, value: str) -> None:
        """A session-level fact for ``trace_metadata``; written by the worker thread like everything else."""
        self._submit("metadata", key, value)

    def _set_trial_start(self, attempt_key: str, trial_start_s: float) -> None:
        self._submit("trial_start", attempt_key, trial_start_s)

    def _append_samples(self, attempt_key: str, samples: list[CursorSample]) -> None:
        self._submit("samples", attempt_key, samples)

    def _append_event(
        self,
        attempt_key: str,
        t_us: int,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self._submit("event", attempt_key, t_us, event_type, payload)

    def _finish_attempt(
        self,
        attempt_key: str,
        *,
        ended_us: int,
        outcome: AttemptOutcome,
        response: Side | None,
        response_time_us: int | None,
    ) -> None:
        self._submit("finish", attempt_key, ended_us, outcome, response, response_time_us)

    def _submit(self, kind: str, *payload: Any) -> None:
        if self._closed:
            return
        self._queue.put(_Command(kind, payload))

    def _writer_loop(self) -> None:
        store: TraceStore | None = None
        attempt_ids: dict[str, int] = {}
        try:
            store = TraceStore(self.path)
            while True:
                command = self._queue.get()
                if command.kind == "stop":
                    break
                if self._error is not None:
                    continue
                try:
                    if command.kind == "create_session":
                        session_id, participant_id, started_at, config, calibration, app_version = command.payload
                        store.create_session(
                            session_id=session_id,
                            participant_id=participant_id,
                            started_at=started_at,
                            config=config,
                            calibration=calibration,
                            app_version=app_version,
                        )
                    elif command.kind == "start_attempt":
                        attempt_key, definition = command.payload
                        attempt_ids[attempt_key] = store.start_attempt(definition)
                    elif command.kind == "trial_start":
                        attempt_key, trial_start_s = command.payload
                        store.set_trial_start(attempt_ids[attempt_key], trial_start_s)
                    elif command.kind == "metadata":
                        key, value = command.payload
                        store.set_metadata(key, value)
                    elif command.kind == "samples":
                        attempt_key, samples = command.payload
                        store.append_samples(attempt_ids[attempt_key], samples)
                    elif command.kind == "event":
                        attempt_key, t_us, event_type, payload = command.payload
                        store.append_event(
                            attempt_ids[attempt_key],
                            t_us=t_us,
                            event_type=event_type,
                            payload=payload,
                        )
                    elif command.kind == "finish":
                        attempt_key, ended_us, outcome, response, response_time_us = command.payload
                        store.finish_attempt(
                            attempt_ids[attempt_key],
                            ended_us=ended_us,
                            outcome=outcome,
                            response=response,
                            response_time_us=response_time_us,
                        )
                        del attempt_ids[attempt_key]
                    else:
                        raise ValueError(f"unknown trace command: {command.kind}")
                except BaseException as exc:  # keep draining without affecting the experiment
                    self._error = exc
        except BaseException as exc:
            self._error = exc
            while self._queue.get().kind != "stop":
                pass
        finally:
            if store is not None:
                store.close()

    def close(self) -> BaseException | None:
        if self._closed:
            return self._error
        self._closed = True
        self._queue.put(_Command("stop", ()))
        self._thread.join()
        return self._error


class AttemptTraceBuffer:
    """Small per-attempt memory buffer used by the real-time trial loop."""

    def __init__(self, recorder: AsyncTraceRecorder, attempt_key: str) -> None:
        self._recorder = recorder
        self.attempt_key = attempt_key
        self._samples: list[CursorSample] = []
        self._closed = False

    def add_sample(self, sample: CursorSample) -> None:
        if self._closed:
            return
        self._samples.append(sample)
        if len(self._samples) >= SAMPLE_BATCH_SIZE:
            self.flush()

    def session_elapsed_us(self) -> int:
        return self._recorder.elapsed_us()

    def set_trial_start(self, trial_start_s: float) -> None:
        """The trial's ``time.perf_counter()`` start, from which every ``t_us`` of this attempt counts."""
        if self._closed:
            return
        self._recorder._set_trial_start(self.attempt_key, trial_start_s)

    def flush(self) -> None:
        if not self._samples:
            return
        batch = self._samples
        self._samples = []
        self._recorder._append_samples(self.attempt_key, batch)

    def add_event(self, *, t_us: int, event_type: str, payload: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        self._recorder._append_event(self.attempt_key, t_us, event_type, payload or {})

    def finish(
        self,
        *,
        ended_us: int,
        outcome: AttemptOutcome,
        response: Side | None = None,
        response_time_us: int | None = None,
    ) -> None:
        if self._closed:
            return
        self.flush()
        self._recorder._finish_attempt(
            self.attempt_key,
            ended_us=ended_us,
            outcome=outcome,
            response=response,
            response_time_us=response_time_us,
        )
        self._closed = True
