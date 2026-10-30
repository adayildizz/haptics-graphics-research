"""The IR glitch filter: short false lifts and jumps to a second object, ignored while the finger stays on the glass."""

from __future__ import annotations

import pytest

from experiment.render_logic import IrGlitchFilter

HZ = 60.0

def feed(ir: IrGlitchFilter, reports: list[tuple[float, float, bool]]) -> tuple[list[tuple[float, float, bool]], list]:
    """(x, y, down) the renderer uses for each 60 Hz (x, y, down) report, and the glitches the filter ignored."""
    used, glitches = [], []
    for frame, (x, y, down) in enumerate(reports):
        reading = ir.step(frame / HZ, x, y, down)
        glitches += ir.glitches
        used.append((reading.x_mm, reading.y_mm, reading.down))
    return used, glitches


def test_a_short_lift_is_ignored_and_the_finger_taken_as_down_where_it_was():
    ir = IrGlitchFilter()
    # Down at 70, the cursor moves on to 71, the report says "up" for 2 frames (33 ms) at a frozen 72, down at 72.
    used, glitches = feed(ir, [(70, 140, True), (71, 140, True), (72, 140, False), (72, 140, False), (72, 140, True)])
    assert used == [(70, 140, True), (71, 140, True), (71, 140, True), (71, 140, True), (72, 140, True)]
    (glitch,) = glitches
    assert glitch.kind == "ir_lift_ignored" and glitch.duration_s == pytest.approx(2 / HZ)


def test_a_lift_that_lasts_the_debounce_is_real_from_then_on():
    ir = IrGlitchFilter(lift_debounce_s=0.09)  # not a whole number of frames, so no frame sits on the limit
    reports = [(70, 140, True)] + [(70 + k, 140, False) for k in range(1, 10)] + [(100, 140, True)]
    used, glitches = feed(ir, reports)
    # Frames 1-6 have been up for less than 90 ms: still down at 70. From frame 7 (100 ms up) the lift is real.
    assert used[1:7] == [(70, 140, True)] * 6
    assert used[7:10] == [(77, 140, False), (78, 140, False), (79, 140, False)]
    assert used[10] == (100, 140, True)  # the touchdown is accepted wherever it is
    assert glitches == []


def test_a_jump_is_rejected_until_the_report_comes_back():
    ir = IrGlitchFilter(jump_mm=20.0)
    reports = [(70, 140, True), (71, 140, True), (111, 120, True), (112, 121, True), (72, 140, True), (73, 140, True)]
    used, glitches = feed(ir, reports)
    assert used == [(70, 140, True), (71, 140, True), (71, 140, True), (71, 140, True), (72, 140, True), (73, 140, True)]
    (glitch,) = glitches
    assert (glitch.kind, glitch.ended) == ("ir_jump_rejected", "return")
    assert glitch.duration_s == pytest.approx(2 / HZ) and glitch.distance_mm == pytest.approx(500**0.5 * 2)


def test_a_step_of_exactly_the_limit_is_accepted():
    ir = IrGlitchFilter(jump_mm=20.0)
    assert feed(ir, [(70, 140, True), (90, 140, True)]) == ([(70, 140, True), (90, 140, True)], [])


def test_the_first_position_after_an_ignored_lift_is_checked_for_a_jump():
    ir = IrGlitchFilter()
    # The finger stays at 72; the frame drops it for one frame and comes back on the palm, 40 mm away.
    reports = [(72, 140, True), (72, 140, False), (112, 140, True), (112, 140, True), (72, 140, True)]
    used, glitches = feed(ir, reports)
    assert used == [(72, 140, True)] * 5
    assert [glitch.kind for glitch in glitches] == ["ir_lift_ignored", "ir_jump_rejected"]


def test_a_rejection_that_lasts_the_reanchor_time_accepts_the_report():
    ir = IrGlitchFilter(jump_mm=20.0, reanchor_s=0.99)  # not a whole number of frames, as above
    # The finger really moved 40 mm (or the frame keeps seeing the second object): taken as true after ~1 s.
    reports = [(70, 140, True)] + [(110, 140, True)] * 70
    used, glitches = feed(ir, reports)
    assert used[1:61] == [(70, 140, True)] * 60  # frames 1-60: rejected for less than 0.99 s
    assert used[61:] == [(110, 140, True)] * 10
    (glitch,) = glitches
    assert glitch.kind == "ir_jump_reanchored"
    assert glitch.duration_s == pytest.approx(1.0) and glitch.distance_mm == pytest.approx(40.0)
    # From the new position on, it is measured from there: back at the old one is now the jump.
    assert ir.step(71 / HZ, 70, 140, True).held


def test_a_real_lift_ends_the_rejection_and_the_next_touchdown_is_accepted_anywhere():
    ir = IrGlitchFilter(lift_debounce_s=0.09)
    reports = [(70, 140, True), (110, 140, True)] + [(110, 140, False)] * 7 + [(150, 140, True)]
    used, glitches = feed(ir, reports)
    assert used[1] == (70, 140, True)  # the jump, rejected
    assert used[2:8] == [(70, 140, True)] * 6  # the lift, not real yet
    assert used[8] == (110, 140, False)  # real: as reported
    assert used[9] == (150, 140, True)
    (glitch,) = glitches
    assert (glitch.kind, glitch.ended) == ("ir_jump_rejected", "lift")


def test_a_rejection_still_open_at_the_end_of_the_trial_is_reported():
    ir = IrGlitchFilter()
    assert feed(ir, [(70, 140, True), (110, 140, True), (110, 140, True)])[1] == []
    glitch = ir.finish(3 / HZ)
    assert glitch is not None and (glitch.kind, glitch.ended) == ("ir_jump_rejected", "end")
    assert glitch.duration_s == pytest.approx(2 / HZ)
    assert ir.finish(4 / HZ) is None


def test_a_trial_that_starts_lifted_is_lifted_until_the_touchdown():
    ir = IrGlitchFilter()
    used, glitches = feed(ir, [(10, 10, False), (12, 10, False), (72, 140, True), (73, 140, True)])
    assert used == [(10, 10, False), (12, 10, False), (72, 140, True), (73, 140, True)]

