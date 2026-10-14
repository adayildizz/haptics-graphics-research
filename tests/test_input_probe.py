from __future__ import annotations

import json

import pytest

from experiment.input_probe import Frame, report, run, summarize

PX_PER_MM = (1920 / 249.0, 1080 / 187.0)
HZ = 60.0


def sliding(n: int, motions, *, pressed: bool = False, step_px: int = 5, **first_frame) -> list[Frame]:
    """A finger sliding right at 60 Hz; ``motions(k)`` is how many motion events frame k received."""
    frames = [Frame(10.0, (100, 500), pressed, **first_frame)]
    frames += [Frame(10.0 + k / HZ, (100 + step_px * k, 500), pressed, motions=motions(k)) for k in range(1, n)]
    return frames


def test_reports_arriving_one_by_one_are_recognised():
    # A 100 Hz frame read by a 60 Hz loop: two events in two of every three frames.
    summary = summarize(sliding(301, lambda k: 1 if k % 3 == 0 else 2), PX_PER_MM)
    assert summary["every_report_arrives"] is True
    assert summary["motion_events_per_second_while_moving"] == pytest.approx(100.0, abs=1.0)
    assert summary["frames_by_motion_events"] == {"0": 0, "1": 100, "2": 200, "3+": 0}
    assert summary["loop_hz"] == pytest.approx(60.0, abs=0.1)


def test_reports_merged_into_one_per_frame_are_recognised():
    summary = summarize(sliding(301, lambda k: 1), PX_PER_MM)
    assert summary["every_report_arrives"] is False
    assert summary["motion_events_per_second_while_moving"] == pytest.approx(60.0, abs=1.0)


def test_contact_as_a_mouse_button_needs_a_press_and_the_button_held_while_moving():
    touching = sliding(120, lambda k: 1, pressed=True, button_down=1)
    assert summarize(touching, PX_PER_MM)["contact_as_mouse_button"] is True
    hovering = sliding(120, lambda k: 1, pressed=False)  # the cursor moves but no button is ever reported
    summary = summarize(hovering, PX_PER_MM)
    assert summary["contact_as_mouse_button"] is False and summary["button_held_share_while_moving"] == 0.0
    assert summary["contact_as_touch_events"] is False


def test_touch_events_and_jumps_are_counted_and_resting_frames_are_left_out():
    frames = sliding(60, lambda k: 1, finger_down=1)
    last = frames[-1]
    frames += [Frame(last.t_s + k / HZ, last.pos, False) for k in range(1, 31)]  # half a second at rest
    frames.append(Frame(frames[-1].t_s + 1 / HZ, (last.pos[0] + 200, last.pos[1]), False, motions=1))  # 26 mm at once
    summary = summarize(frames, PX_PER_MM)
    assert summary["contact_as_touch_events"] is True and summary["finger_down_events"] == 1
    assert summary["jumps_over_10_mm"] == 1
    assert summary["frames_with_movement"] == 60  # 59 sliding frames and the jump; the 30 at rest do not count
    text = report(summary)
    assert "3. Is contact reported as touch events?             YES" in text
    assert "1. Does every position report reach the program?   NO" in text


def test_run_records_frames_and_saves_the_summary_without_a_real_window(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    summary = run(0.3, fullscreen=False, out_dir=tmp_path)
    assert summary["loop_hz"] > 10 and summary["frames_with_movement"] == 0
    (saved,) = tmp_path.glob("input_probe_*.json")
    assert json.loads(saved.read_text())["every_report_arrives"] is False
