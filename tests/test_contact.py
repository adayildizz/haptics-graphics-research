from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from render_harness import polyline, run_scripted_trial

from analysis.exploration_maps.run import run as run_exploration_maps
from experiment import trial_render
from experiment.calibration import make_configured_ir_frame_calibration
from experiment.config import load_experiment_config
from experiment.constant_stimuli import TrialSpec
from experiment.render_logic import ContactGate
from experiment.trace_store import SCHEMA_VERSION, AttemptDefinition, CursorSample, TraceStore

CONFIG_PATH = Path(__file__).resolve().parent.parent / "experiment" / "configs" / "default.yaml"
CALIBRATION = make_configured_ir_frame_calibration((1920, 1080))
SPEC = TrialSpec(level_pct=0.2, comparison_height_mm=12.0, reference_height_mm=10.0, reference_side="left")
# Down at the start, lifted for frames 30-49 while the cursor stays put, down again 25 mm further right.
LIFT = dict(
    path=polyline([(0, 55, 141), (29, 72, 141), (49, 72, 141), (50, 97, 141), (89, 130, 141)], 90),
    touch={0: True, 30: False, 50: True},
    end="left",
)


def config(**overrides):
    # These pin the lift rule itself, which the IR glitch filter only delays: with the filter off it must be as before.
    return replace(load_experiment_config(CONFIG_PATH), **{"ir_filter": False, **overrides})


def test_gate_trusts_contact_only_after_the_device_has_reported_a_touch():
    gate = ContactGate()
    assert gate.finger_down(False) is True and not gate.device_reports_touch  # never reported: assume down
    assert gate.finger_down(True) is True and gate.device_reports_touch
    assert gate.finger_down(False) is False  # from now on a released button means a lifted finger


def test_samples_carry_contact_and_lifts_become_contact_events():
    _, attempt, _, _ = run_scripted_trial(config(render_mode="position"), CALIBRATION, SPEC, LIFT)
    contact = [sample.contact for sample in attempt.samples]
    assert contact[:30] == [True] * 30 and contact[30:50] == [False] * 20 and all(contact[50:])
    events = [(kind, payload) for _, kind, payload in attempt.events if kind.startswith("contact_")]
    assert [kind for kind, _ in events] == ["contact_up", "contact_down"]
    (stats,) = [payload for _, kind, payload in attempt.events if kind == "sample_stats"]
    assert stats["device_reports_touch"] is True


def test_a_device_that_never_reports_touch_records_contact_as_unknown_and_warns_once(capsys):
    scenario = dict(path=polyline([(0, 55, 141), (40, 90, 141)], 41), end="left")
    for _ in range(2):
        _, attempt, _, _ = run_scripted_trial(config(render_mode="position"), CALIBRATION, SPEC, scenario)
        assert all(sample.contact for sample in attempt.samples)
        assert not [kind for _, kind, _ in attempt.events if kind.startswith("contact_")]
    assert capsys.readouterr().out.count("has not reported a finger touching the surface") == 1
    assert not trial_render.CONTACT.device_reports_touch


def test_schema_v4_writes_contact_and_what_was_used_and_upgrades_older_files_in_place(tmp_path):
    assert SCHEMA_VERSION == 4
    path = tmp_path / "old_trace.sqlite3"
    # A schema-1 file: no contact, no is_practice / practice_round.
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE trace_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY, participant_id TEXT NOT NULL, started_at TEXT NOT NULL,
            config_json TEXT NOT NULL, calibration_json TEXT NOT NULL, app_version TEXT NOT NULL);
        CREATE TABLE attempts (attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            trial_index INTEGER NOT NULL, attempt_index INTEGER NOT NULL, started_us INTEGER NOT NULL, ended_us INTEGER,
            level_pct REAL NOT NULL, comparison_height_mm REAL NOT NULL, reference_height_mm REAL NOT NULL,
            bar_width_mm REAL NOT NULL, reference_side TEXT NOT NULL, is_catch INTEGER NOT NULL, outcome TEXT,
            response TEXT, response_time_us INTEGER);
        CREATE TABLE cursor_samples (attempt_id INTEGER NOT NULL, sequence INTEGER NOT NULL, t_us INTEGER NOT NULL,
            frame_dt_us INTEGER NOT NULL, x_px INTEGER NOT NULL, y_px INTEGER NOT NULL, x_mm REAL NOT NULL,
            y_mm REAL NOT NULL, speed_mm_s REAL NOT NULL, in_active_area INTEGER NOT NULL, active_side TEXT,
            signal_on INTEGER NOT NULL, PRIMARY KEY (attempt_id, sequence));
        INSERT INTO cursor_samples VALUES (99, 0, 0, 0, 1, 2, 0.1, 0.2, 0.0, 1, NULL, 0);
        """
    )
    connection.commit()
    connection.close()

    with TraceStore(path) as store:
        store.create_session(session_id="S", participant_id="P", started_at="t", config={}, calibration={}, app_version="x")
        attempt_id = store.start_attempt(AttemptDefinition("S", 1, 1, 0, 0.1, 11.0, 10.0, 10.0, "left", False))
        sample = CursorSample(0, 0, 16_667, 5, 6, 1.0, 2.0, 0.0, True, None, False, contact=False)
        store.append_samples(attempt_id, [replace(sample, used_x_mm=0.5, used_y_mm=2.5, used_contact=True)])

    connection = sqlite3.connect(path)
    columns = "attempt_id, contact, used_x_mm, used_y_mm, used_contact"
    rows = {row[0]: row[1:] for row in connection.execute(f"SELECT {columns} FROM cursor_samples")}
    version = connection.execute("SELECT value FROM trace_metadata WHERE key = 'schema_version'").fetchone()[0]
    practice = connection.execute("SELECT is_practice, practice_round FROM attempts").fetchone()
    connection.close()
    # The old sample keeps the implied "finger down"; what the renderer used was not recorded then.
    assert rows == {99: (1, None, None, None), attempt_id: (0, 0.5, 2.5, 1)}
    assert version == "4" and practice == (0, 0)


def test_heat_maps_use_the_recorded_contact_instead_of_guessing(tmp_path):
    from experiment.trace_recorder import AsyncTraceRecorder

    import experiment.main as main_module

    cfg = config(render_mode="position")
    trace_path = tmp_path / "P11_touch_trace.sqlite3"
    recorder = AsyncTraceRecorder(trace_path)
    recorder.create_session(
        session_id="P11_touch", participant_id="P11", started_at="t", config=main_module.config_to_dict(cfg),
        calibration=CALIBRATION.__dict__, app_version="test",
    )  # fmt: skip
    buffer = recorder.start_attempt("1:1", AttemptDefinition("P11_touch", 1, 1, 0, 0.2, 12.0, 10.0, 10.0, "left", False))

    class Forward:
        """Hands the scripted trial's samples and events to the real recorder."""

        add_sample = buffer.add_sample
        add_event = buffer.add_event
        session_elapsed_us = buffer.session_elapsed_us
        finish = buffer.finish

    import render_harness

    original = render_harness.FakeAttempt
    render_harness.FakeAttempt = Forward
    try:
        run_scripted_trial(cfg, CALIBRATION, SPEC, LIFT)
    finally:
        render_harness.FakeAttempt = original
        assert recorder.close() is None

    out = tmp_path / "maps"
    assert run_exploration_maps(trace_path, "all", out) == 0
    with (out / "P11" / "P11_touch" / "trials.csv").open() as file:
        header, row = (line.strip().split(",") for line in file.readlines())
    summary = dict(zip(header, row))
    assert summary["contact_method"] == "contact flag"
    # 20 frames off the glass, plus the step into the first and out of the last of them.
    assert float(summary["no_contact_ms"]) == pytest.approx(21 * 1000 / 60, abs=25)


def _writes(attempt):
    return [payload for _, kind, payload in attempt.events if kind == "signal_write"]


@pytest.mark.parametrize("mode", ["predictive", "position", "timed_legacy"])
def test_a_lifted_finger_switches_the_signal_off_in_every_mode(mode):
    # During frames 30-49 the cursor sits inside the left bar, but the finger is off the glass.
    _, attempt, _, _ = run_scripted_trial(config(render_mode=mode), CALIBRATION, SPEC, LIFT)
    on = [sample.signal_on for sample in attempt.samples]
    assert all(on[22:30]) and not any(on[30:50])
    assert not any(on[50:58])  # touched down again between the bars
    assert any(on[58:])  # and the right bar is felt as usual afterwards
    if mode != "predictive":  # with a scripted clock only the immediate writes are deterministic
        lift = [w for w in _writes(attempt) if w["edge"] == "lift"]
        assert len(lift) == 1 and lift[0]["on"] is False and lift[0]["bar"] == "left"


@pytest.mark.parametrize("mode", ["predictive", "position", "timed_legacy"])
def test_touching_down_inside_a_bar_switches_on_at_once_and_is_labelled(mode):
    scenario = dict(
        path=polyline([(0, 72, 141), (9, 72, 141), (10, 122, 140), (30, 123, 140)], 31),
        touch={0: True, 5: False, 10: True},
        end="left",
    )
    _, attempt, _, _ = run_scripted_trial(config(render_mode=mode), CALIBRATION, SPEC, scenario)
    on = [sample.signal_on for sample in attempt.samples]
    assert all(on[:5]) and not any(on[5:10]) and all(on[10:])
    touch = [w for w in _writes(attempt) if w["edge"] == "touch"]
    assert len(touch) == 1 and touch[0]["on"] is True and touch[0]["bar"] == "right"


def test_without_touch_reports_a_still_cursor_inside_a_bar_keeps_the_signal_on():
    scenario = dict(path=LIFT["path"], end="left")  # same path, but the device never reports touch
    _, attempt, _, _ = run_scripted_trial(config(render_mode="position"), CALIBRATION, SPEC, scenario)
    on = [sample.signal_on for sample in attempt.samples]
    assert all(on[22:50])
    assert not [w for w in _writes(attempt) if w["edge"] in ("lift", "touch")]


def test_once_the_device_has_reported_touch_the_next_trial_trusts_it_from_its_first_frame():
    run_scripted_trial(config(render_mode="position"), CALIBRATION, SPEC, LIFT)
    hovering = dict(path=polyline([(0, 60, 141), (40, 90, 141)], 41), end="left")  # button never pressed
    _, attempt, _, _ = run_scripted_trial(config(render_mode="position"), CALIBRATION, SPEC, hovering)
    assert not any(sample.signal_on for sample in attempt.samples)
    assert not any(sample.contact for sample in attempt.samples)
