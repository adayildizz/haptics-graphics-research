"""Electroadhesion signal helpers for the height-JND experiment."""

from __future__ import annotations

import time
from typing import Any

from .config import (
    DISABLE_OUTPUT_WHEN_OFF,
    MAX_SIGNAL_DURATION_S,
    MIN_SPEED_MM_S,
    MIN_VOLTAGE,
    OFFSET_V,
    VISA_ADDRESS,
    WAVE_SQUARE,
    ExperimentConfig,
)

_current_waveform: str | None = None
_current_frequency: float | None = None
_current_voltage: float | None = None
_output_enabled = False
_voltage_peak: float = MIN_VOLTAGE


def _reset_cached_state() -> None:
    global _current_waveform, _current_frequency, _current_voltage, _output_enabled
    _current_waveform = None
    _current_frequency = None
    _current_voltage = None
    _output_enabled = False


def _write_waveform(instrument: Any, waveform: str, *, force: bool = False) -> None:
    global _current_waveform
    if force or waveform != _current_waveform:
        instrument.write(f"FUNC {waveform}")
        _current_waveform = waveform


def _write_frequency(instrument: Any, frequency: float, *, force: bool = False) -> None:
    global _current_frequency
    if force or _current_frequency is None or int(frequency) != int(_current_frequency):
        instrument.write(f"FREQ {int(frequency)}")
        _current_frequency = frequency


def _write_voltage(instrument: Any, voltage: float, *, force: bool = False) -> None:
    global _current_voltage
    clamped_voltage = max(MIN_VOLTAGE, min(voltage, _voltage_peak))
    if force or _current_voltage is None or abs(clamped_voltage - _current_voltage) > 0.05:
        instrument.write(f"VOLT {clamped_voltage:.2f}")
        _current_voltage = clamped_voltage


def _write_output(instrument: Any, enabled: bool, *, force: bool = False) -> None:
    global _output_enabled
    if force or enabled != _output_enabled:
        instrument.write(f"OUTP {'ON' if enabled else 'OFF'}")
        _output_enabled = enabled


def connect_hardware(
    cfg: ExperimentConfig,
    address: str = VISA_ADDRESS,
    *,
    max_attempts: int = 6,
    retry_delay_s: float = 1.0,
) -> Any | None:
    """Open and configure the VISA signal generator using ``cfg``'s carrier/voltage.

    The VXI-11 handshake to this instrument is flaky over LAN and often needs
    a few tries before it succeeds, so the connection attempt is retried up
    to ``max_attempts`` times before giving up.

    Returns ``None`` when PyVISA or the instrument is unavailable so demo mode
    can run without hardware.
    """
    global _voltage_peak
    _voltage_peak = cfg.voltage_peak

    try:
        import pyvisa
    except ModuleNotFoundError as exc:
        print(f"Hardware unavailable ({exc}). Running without electroadhesion output.")
        return None

    rm = pyvisa.ResourceManager()
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            _reset_cached_state()
            instrument = rm.open_resource(address)
            instrument.write_termination = "\n"
            instrument.read_termination = "\n"
            instrument.write("*RST")
            instrument.write("VOLT:UNIT VPP")
            instrument.write("OUTP:LOAD INF")
            instrument.write(f"VOLT:OFFS {OFFSET_V}")
            _write_waveform(instrument, WAVE_SQUARE, force=True)
            instrument.write("FUNC:SQU:DCYC 50")
            _write_frequency(instrument, cfg.carrier_freq_hz, force=True)
            _write_voltage(instrument, MIN_VOLTAGE, force=True)
            _write_output(instrument, not DISABLE_OUTPUT_WHEN_OFF, force=True)
            print(f"Connected to signal generator: {instrument.query('*IDN?').strip()}")
            return instrument
        except Exception as exc:
            last_exc = exc
            print(f"Connection attempt {attempt}/{max_attempts} failed ({exc}); retrying...")
            if attempt < max_attempts:
                time.sleep(retry_delay_s)

    print(f"Hardware unavailable ({last_exc}). Running without electroadhesion output.")
    return None


def signal_on(instrument: Any | None) -> None:
    """Activate the bar-interior signal. Bar interior = ON, exterior = OFF."""
    if instrument is None:
        return
    _write_waveform(instrument, WAVE_SQUARE)
    _write_voltage(instrument, _voltage_peak)
    _write_output(instrument, True)


def signal_off(instrument: Any | None) -> None:
    """Deactivate the signal for bar-exterior regions."""
    if instrument is None:
        return
    if DISABLE_OUTPUT_WHEN_OFF:
        _write_output(instrument, False)
    else:
        _write_output(instrument, True)
        _write_voltage(instrument, MIN_VOLTAGE)


def close_hardware(instrument: Any | None) -> None:
    """Turn the signal off and close the VISA resource."""
    if instrument is None:
        return
    try:
        _write_voltage(instrument, MIN_VOLTAGE, force=True)
        _write_output(instrument, False, force=True)
        instrument.close()
    except Exception:
        pass


def stimulus_duration(bar_width_mm: float, finger_speed_mm_s: float) -> float:
    """Compute timed rendering duration from bar width and finger speed."""
    speed = max(abs(finger_speed_mm_s), MIN_SPEED_MM_S)
    return min(bar_width_mm / speed, MAX_SIGNAL_DURATION_S)


class GeneratorDevice:
    """The scheduler's handle on the generator. During a session nothing else may switch the output."""

    def __init__(self, instrument: Any) -> None:
        self._instrument = instrument

    def on(self) -> None:
        signal_on(self._instrument)

    def off(self) -> None:
        signal_off(self._instrument)


def device_for(instrument: Any | None) -> GeneratorDevice | None:
    """None without hardware, so the scheduler falls back to its no-op device."""
    return None if instrument is None else GeneratorDevice(instrument)
