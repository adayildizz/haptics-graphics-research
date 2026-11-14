"""The force sensor during an experiment session (``force_sensor: true``).

``main.run`` uses one ``ForceSession`` per session, in four steps:

1. ``start()``, while the session is set up and before "Ready to begin": reads the calibration file once (a
   session never runs with one that cannot be read), starts ``force_daq.ForceReader`` and notes the session's
   force facts in the trace. If anything fails it raises ``ForceSensorError`` and the session does not start.
2. ``add_hands_off(block, start_s, end_s)``: before the practice block and before the main block the screen
   asks for the hands off the glass for ``force_zero_s`` seconds (``main.hands_off_step``); this notes when.
   The zero itself is computed afterwards from these windows (``analysis.force_sync``).
3. ``check()``, between trials: if the reader has stopped, it warns once and notes it in the trace. The
   session goes on; only its force data end there.
4. ``stop()``, in the session's ``finally`` after the signal is off: stops the reader, which writes its last
   samples to disk, and notes its summary in the trace and on the console.

The trial loop never touches the sensor: the reader is a process of its own that writes every raw sample (six
gauge voltages, 1000 per second) to ``<session_id>_force.sqlite3`` next to the trace, with times on
``time.perf_counter``. The trace keeps every trial's start on that clock (``attempts.trial_start_s``), which is
all that is needed to line the two up afterwards.

The session's force facts are rows of the trace's ``trace_metadata`` table (values are JSON where they are not
plain text): ``force_file``, ``force_cal_file``, ``force_card``, ``force_hands_off``, and, when they happen,
``force_reader_problem`` and ``force_reader_summary``. All times in them are ``time.perf_counter`` seconds.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .ati_calibration import load_ati_calibration
from .config import ExperimentConfig
from .force_daq import RATE_HZ, ForceReader, ForceSource, ReaderSummary, source_factory

if TYPE_CHECKING:
    from .trace_recorder import AsyncTraceRecorder

REPO_ROOT = Path(__file__).resolve().parent.parent
FORCE_FILE_SUFFIX = "_force.sqlite3"
STOPPED_REPORT_WAIT_S = 1.0  # how long check() waits for a stopped reader's own report of why it stopped


class ForceSensorError(RuntimeError):
    """The force sensor could not be started; the session must not start without it."""


def cal_file_path(name: str) -> Path:
    """The calibration file as the config names it: relative to the repository root, or absolute."""
    path = Path(name)
    return path if path.is_absolute() else REPO_ROOT / path


def force_file_path(trace_dir: Path, session_id: str) -> Path:
    """The session's force file: next to the trace and the other session files."""
    return trace_dir / f"{session_id}{FORCE_FILE_SUFFIX}"


def make_source(cfg: ExperimentConfig) -> Callable[[], ForceSource]:
    """What the reader process builds its source with: the NI card of the config, or the fake card."""
    card = argparse.Namespace(
        device=cfg.force_device,
        first_channel=cfg.force_first_channel,
        terminal=cfg.force_terminal,
        range_v=cfg.force_range_v,
    )
    return source_factory(card, fake=cfg.force_fake)


def card_facts(cfg: ExperimentConfig) -> dict[str, Any]:
    """The card settings as noted in the trace."""
    if cfg.force_fake:
        return {"source": "fake", "rate_hz": RATE_HZ}
    return {
        "source": "nidaqmx",
        "device": cfg.force_device,
        "first_channel": cfg.force_first_channel,
        "terminal": cfg.force_terminal,
        "range_v": cfg.force_range_v,
        "rate_hz": RATE_HZ,
    }


class ForceSession:
    """The force reader of one session and what the trace says about it."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        trace: AsyncTraceRecorder,
        path: Path,
        source: Callable[[], ForceSource] | None = None,
    ) -> None:
        """``source`` replaces the config's card (tests); like any reader source it must be picklable."""
        self.cfg = cfg
        self.path = Path(path)
        self.zero_s = cfg.force_zero_s
        self._trace = trace
        self._source = source or make_source(cfg)
        self._reader: ForceReader | None = None
        self.hands_off: list[dict[str, Any]] = []  # one window per block: {"block", "start_s", "end_s"}
        self.problem: dict[str, Any] | None = None  # set if the reader could not start or stopped during the session
        self.summary: ReaderSummary | None = None

    def start(self) -> None:
        """Reads the calibration file and starts the reader; raises ``ForceSensorError`` saying why if it cannot."""
        cal_path = cal_file_path(self.cfg.force_cal_file)
        try:
            calibration = load_ati_calibration(cal_path)
        except Exception as exc:  # noqa: BLE001 -- whatever it is, the file cannot be used
            raise self._refused(f"the calibration file {cal_path} could not be read: {exc}") from exc
        try:
            reader = ForceReader(self.path, self._source).start()  # a few seconds on Windows: a fresh Python
        except Exception as exc:  # noqa: BLE001 -- start() has already stopped its process
            raise self._refused(str(exc)) from exc
        self._reader = reader
        self._note("force_file", self.path.name)
        self._note("force_cal_file", self.cfg.force_cal_file)
        self._note("force_card", {**card_facts(self.cfg), "calibration_serial": calibration.serial})
        print(f"Force sensor running ({calibration.serial}); raw samples go to {self.path}")

    def add_hands_off(self, block: str, start_s: float, end_s: float) -> None:
        """Notes one hands-off window (``time.perf_counter`` seconds) before ``block`` ("practice" or "main")."""
        self.hands_off.append({"block": block, "start_s": start_s, "end_s": end_s})
        self._note("force_hands_off", self.hands_off)
        print(f"Hands-off step before the {block} block: {end_s - start_s:.1f} s")

    def check(self) -> None:
        """Between trials only: if the reader has stopped, warn once and note it in the trace.

        The session goes on either way; only its force data end where the reader stopped.
        """
        if self._reader is None or self.problem is not None or self._reader.running:
            return
        # The reader has already stopped; this collects its own report of why (or gives up after a second).
        self.summary = self._reader.stop(STOPPED_REPORT_WAIT_S)
        reason = self.summary.error or "the reader process ended without saying why"
        self._problem(reason, samples=self.summary.samples)
        print(
            "WARNING: the force reader stopped during the session; the session goes on, but without force data "
            f"from here ({self.summary.samples} samples were recorded).\n{reason}"
        )

    def stop(self) -> None:
        """At the session's end, after the signal is off: stops the reader and notes its summary.

        Never raises, so the rest of the session's clean-up (closing the trace above all) still runs.
        """
        if self._reader is None:
            return
        try:
            self.summary = self._reader.stop()  # waits until every sample is on disk; safe after check() stopped it
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: the force reader could not be stopped cleanly ({exc!r}); see {self.path}")
            return
        self._note(
            "force_reader_summary",
            {
                "samples": self.summary.samples,
                "reads": self.summary.reads,
                "clock_offset_s": self.summary.clock_offset_s,
                "error": self.summary.error,
            },
        )
        print(f"Force data saved: {self.path} ({self.summary.samples} samples)")
        if self.summary.error is not None and self.problem is None:
            print(f"WARNING: the force reader reported an error:\n{self.summary.error}")

    def _refused(self, message: str) -> ForceSensorError:
        """Notes in the trace why the sensor could not start, and returns the error to raise."""
        self._problem(message, samples=0)
        return ForceSensorError(message)

    def _problem(self, error: str, samples: int) -> None:
        self.problem = {"noticed_s": time.perf_counter(), "samples": samples, "error": error}
        self._note("force_reader_problem", self.problem)

    def _note(self, key: str, value: Any) -> None:
        self._trace.set_metadata(key, value if isinstance(value, str) else json.dumps(value))
