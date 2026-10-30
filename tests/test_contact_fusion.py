from __future__ import annotations

import pytest

from experiment.ati_calibration import AtiCalibration
from experiment.contact_fusion import (
    ContactFusion,
    HysteresisContact,
    contact_state,
    episodes,
    load_of,
    thresholds_from_noise,
)


def test_the_four_states():
    assert contact_state(True, True) == "both"
    assert contact_state(False, True) == "force_only"
    assert contact_state(True, False) == "ir_only"
    assert contact_state(False, False) == "neither"


def test_hysteresis_needs_the_on_level_to_load_and_the_off_level_to_unload():
    detector = HysteresisContact(on=0.05, off=0.03)
    assert [detector.update(v) for v in (0.01, 0.04, 0.05, 0.04, 0.031, 0.029, 0.04)] == [
        False, False, True, True, True, False, False
    ]  # fmt: skip
    with pytest.raises(ValueError):
        HysteresisContact(on=0.01, off=0.02)


def test_thresholds_sit_above_the_unloaded_noise():
    noise = [0.010, 0.012, 0.008, 0.011, 0.009]
    limits = thresholds_from_noise(noise)
    assert limits.noise_mean == pytest.approx(0.01)
    assert limits.on > limits.off > limits.noise_mean


def test_load_is_fz_with_a_calibration_and_the_voltage_change_without():
    identity = tuple(tuple(1.0 if r == c else 0.0 for c in range(6)) for r in range(6))
    cal = AtiCalibration("FT1", "SI", "N", "N-mm", identity, (1.0,) * 6)
    volts, bias = (0.0, 0.0, -0.3, 0.0, 0.0, 0.0), (0.0,) * 6
    assert load_of(volts, bias, cal) == pytest.approx(0.3)
    assert load_of((0.3, 0.4, 0, 0, 0, 0), bias, None) == pytest.approx(0.5)


def test_an_ir_dropout_with_the_force_loaded_is_held_then_given_up():
    fusion = ContactFusion(hold_s=0.15, jump_mm=20.0)
    assert fusion.update(0.00, True, True, 70.0, 100.0).render_down
    held = fusion.update(0.05, False, True, 70.0, 100.0)
    assert held.state == "force_only" and held.render_down and held.held and not held.position_valid
    assert fusion.update(0.15, False, True, 70.0, 100.0).render_down
    late = fusion.update(0.25, False, True, 70.0, 100.0)
    assert not late.render_down and not late.held


def test_a_real_lift_switches_off_at_once_and_the_next_touchdown_lands_anywhere():
    fusion = ContactFusion()
    fusion.update(0.00, True, True, 70.0, 100.0)
    assert not fusion.update(0.02, False, False, 70.0, 100.0).render_down
    landed = fusion.update(0.30, True, True, 150.0, 40.0)  # 90 mm away, after a real lift: fine
    assert landed.position_valid and not landed.teleport


def test_a_jump_with_the_force_held_throughout_is_a_teleport_until_confirmed():
    fusion = ContactFusion(jump_mm=20.0)
    fusion.update(0.000, True, True, 70.0, 100.0)
    fusion.update(0.016, False, True, 70.0, 100.0)  # IR dropout, finger still loaded
    ghost = fusion.update(0.090, True, True, 110.0, 100.0)  # IR comes back 40 mm away
    assert ghost.teleport and not ghost.position_valid and ghost.render_down
    back = fusion.update(0.106, True, True, 71.0, 100.0)  # next frame is near the old place again
    assert back.position_valid and not back.teleport
    fusion.update(0.122, True, True, 120.0, 100.0)  # a real fast stroke...
    confirmed = fusion.update(0.138, True, True, 122.0, 100.0)  # ...is believed once a second frame agrees
    assert confirmed.position_valid


def test_hovering_in_the_beams_keeps_the_signal_switchable():
    frame = ContactFusion().update(0.0, True, False, 70.0, 100.0)
    assert frame.state == "ir_only" and frame.render_down and frame.position_valid


def test_episodes_are_runs_of_one_state():
    runs = episodes([0.0, 0.1, 0.2, 0.3, 0.4], ["both", "both", "force_only", "both", "both"])
    assert [(r.state, r.start_s, r.end_s) for r in runs] == [
        ("both", 0.0, 0.2), ("force_only", 0.2, 0.3), ("both", 0.3, 0.4)
    ]  # fmt: skip
