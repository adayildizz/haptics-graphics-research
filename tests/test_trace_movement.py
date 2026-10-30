from __future__ import annotations

import pytest

from analysis.trace_movement import animate_attempt, bar_geometry, select_attempts
from experiment.demo_trace import ensure_demo_trace
from experiment.replay_data import load_trace


@pytest.fixture
def demo_session(tmp_path):
    return load_trace(ensure_demo_trace(tmp_path / "demo_replay_trace.sqlite3"))


def test_bar_geometry_matches_live_layout(demo_session):
    attempt = demo_session.attempts[0]  # level -18%, reference left -> comparison right, shorter
    geometry = bar_geometry(demo_session, attempt)

    assert geometry.surface_width_mm == pytest.approx(194.0)
    assert geometry.baseline_y == pytest.approx(145.0)
    # 40 mm gap between the bars' inner edges, both 10 mm wide, centred.
    assert geometry.right_x0 - geometry.left_x1 == pytest.approx(40.0)
    assert (geometry.left_x0 + geometry.right_x1) / 2 == pytest.approx(97.0)
    assert geometry.left_is_comparison is False
    assert geometry.left_height_mm == pytest.approx(10.0)
    assert geometry.right_height_mm == pytest.approx(8.2)
    assert geometry.taller_side == "left"


def test_selection_defaults_to_main_block(demo_session):
    assert all(not a.is_practice for a in select_attempts(demo_session))
    assert [a.trial_index for a in select_attempts(demo_session, trial=2)] == [2, 2]


def test_animation_renders_gif(demo_session, tmp_path):
    attempt = demo_session.attempts[0]
    out = animate_attempt(demo_session, attempt, tmp_path / "trial.gif", fps=10, speed=4.0, max_frames=4)
    assert out.exists() and out.stat().st_size > 0
