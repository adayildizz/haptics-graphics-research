"""ATI F/T calibration files (``FTxxxxx.cal``): gauge voltages to forces and torques.

A ``.cal`` file is XML. It names the sensor (``FTSensor Serial``), the calibration
(``Calibration PartNumber``, ``ForceUnits``, ``TorqueUnits``) and gives one row of the calibration matrix per
axis (``UserAxis Name="Fx" values="..."``; older files use ``Axis``). Forces and torques are that matrix times
the gauge voltages minus their unloaded values (the bias). Standard library only.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

AXES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


@dataclass(frozen=True)
class AtiCalibration:
    serial: str
    part_number: str
    force_units: str
    torque_units: str
    matrix: tuple[tuple[float, ...], ...]  # one row per axis in AXES, one column per gauge
    maxima: tuple[float, ...]  # rated range per axis, in the file's units

    @property
    def gauges(self) -> int:
        return len(self.matrix[0])

    def apply(self, volts: Sequence[float], bias: Sequence[float]) -> tuple[float, ...]:
        """Fx, Fy, Fz, Tx, Ty, Tz for one sample of gauge voltages."""
        if len(volts) != self.gauges or len(bias) != self.gauges:
            raise ValueError(f"expected {self.gauges} gauge voltages, got {len(volts)} (bias {len(bias)})")
        loaded = [v - b for v, b in zip(volts, bias)]
        return tuple(sum(c * x for c, x in zip(row, loaded)) for row in self.matrix)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def load_ati_calibration(path: str | Path) -> AtiCalibration:
    """Read a ``.cal`` file; raises ValueError naming what is missing if it is not one."""
    root = ET.parse(Path(path)).getroot()
    elements = list(root.iter())
    by_tag: dict[str, list[ET.Element]] = {}
    for element in elements:
        by_tag.setdefault(_local(element.tag), []).append(element)
    rows = by_tag.get("useraxis") or by_tag.get("axis") or []
    named = {row.get("Name", ""): row for row in rows}
    missing = [axis for axis in AXES if axis not in named]
    if missing:
        raise ValueError(f"{path}: no matrix row for {missing}; elements found: {sorted(by_tag)}")
    matrix = tuple(tuple(float(v) for v in named[axis].get("values", "").split()) for axis in AXES)
    if len({len(row) for row in matrix}) != 1 or not matrix[0]:
        raise ValueError(f"{path}: the matrix rows do not all have the same number of gauges")
    calibration = (by_tag.get("calibration") or [ET.Element("Calibration")])[0]
    sensor = (by_tag.get("ftsensor") or [root])[0]
    return AtiCalibration(
        serial=sensor.get("Serial", ""),
        part_number=calibration.get("PartNumber", ""),
        force_units=calibration.get("ForceUnits", ""),
        torque_units=calibration.get("TorqueUnits", ""),
        matrix=matrix,
        maxima=tuple(float(named[axis].get("max", "nan")) for axis in AXES),
    )
