from __future__ import annotations

import pytest

from experiment.ati_calibration import AXES, load_ati_calibration

CAL = """<?xml version="1.0" encoding="utf-8"?>
<FTSensor Serial="FT99999" BodyStyle="Nano17" NumGages="6">
  <Calibration PartNumber="SI-12-0.12" ForceUnits="N" TorqueUnits="N-mm" OutputRange="20">
    <BasicTransform Dx="0" Dy="0" Dz="0" Rx="0" Ry="0" Rz="0"/>
    <UserAxis Name="Fx" values="1 0 0 0 0 0" max="12"/>
    <UserAxis Name="Fy" values="0 1 0 0 0 0" max="12"/>
    <UserAxis Name="Fz" values="0.5 0.5 2 0 0 0" max="17"/>
    <UserAxis Name="Tx" values="0 0 0 1 0 0" max="120"/>
    <UserAxis Name="Ty" values="0 0 0 0 1 0" max="120"/>
    <UserAxis Name="Tz" values="0 0 0 0 0 1" max="120"/>
  </Calibration>
</FTSensor>
"""


def test_a_cal_file_gives_the_matrix_units_and_ranges(tmp_path):
    path = tmp_path / "FT99999.cal"
    path.write_text(CAL)
    cal = load_ati_calibration(path)
    assert (cal.serial, cal.part_number, cal.force_units, cal.torque_units) == ("FT99999", "SI-12-0.12", "N", "N-mm")
    assert cal.gauges == 6 and cal.maxima[2] == 17.0
    forces = cal.apply([1.1, 0.2, 0.3, 0, 0, 0], [0.1, 0.2, 0.1, 0, 0, 0])
    assert forces[0] == pytest.approx(1.0) and forces[2] == pytest.approx(0.5 * 1.0 + 0.5 * 0.0 + 2 * 0.2)


def test_older_files_with_axis_rows_load_too_and_a_wrong_file_says_what_is_missing(tmp_path):
    older = tmp_path / "old.cal"
    older.write_text(CAL.replace("UserAxis", "Axis"))
    assert load_ati_calibration(older).matrix[1] == (0, 1, 0, 0, 0, 0)
    wrong = tmp_path / "wrong.cal"
    wrong.write_text("<Root><Something/></Root>")
    with pytest.raises(ValueError, match="no matrix row"):
        load_ati_calibration(wrong)
    assert AXES == ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
