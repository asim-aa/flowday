from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
import streamlit as st
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st_h
from streamlit.testing.v1 import AppTest

from flowday.app.inputs import parse_events, parse_tasks
from flowday.app.session import PlanningSession, compare_estimators, diff_schedules, persona_history
from flowday.eventlog import EventLog
from flowday.scheduler import Event, Task, build_schedule, hhmm, validate_schedule
from tests.conftest import SMALL

APP_SCRIPT = str(Path(__file__).resolve().parents[1] / "flowday" / "app" / "main.py")

# --- parsing the editable tables -------------------------------------------------------------------


def test_parse_events_accepts_good_rows_and_skips_blank_ones():
    df = pd.DataFrame({"title": ["Standup", None], "start": ["09:30", None], "end": ["10:00", None]})
    events, errors = parse_events(df)
    assert errors == [] and [(e.title, e.start, e.end) for e in events] == [("Standup", 570, 600)]


def test_parse_events_reports_bad_times_reversed_ranges_and_overlaps():
    df = pd.DataFrame(
        {
            "title": ["a", "b", "c", "d"],
            "start": ["9am", "11:00", "10:00", "10:30"],
            "end": ["10:00", "10:00", "11:00", "11:30"],
        }
    )
    _, errors = parse_events(df)
    assert any("Event row 1" in e and "9am" in e for e in errors)
    assert any("Event row 2" in e for e in errors)  # ends before it starts
    assert any("overlap" in e for e in errors)  # c (10:00-11:00) and d (10:30-11:30)


def test_parse_tasks():
    df = pd.DataFrame(
        {
            "title": ["Write", "Mail", None],
            "type": ["writing", "email", None],
            "estimate": [90, 15, None],
            "deadline": ["15:00", "", None],
            "priority": [5, None, None],
        }
    )
    tasks, errors = parse_tasks(df)
    assert errors == []
    assert [(t.title, t.estimated_minutes, t.deadline, t.priority) for t in tasks] == [
        ("Write", 90, hhmm("15:00"), 5),
        ("Mail", 15, None, 3),  # a blank priority defaults to 3
    ]
    assert len({t.id for t in tasks}) == 2


@pytest.mark.parametrize(
    "row, fragment",
    [
        ({"estimate": 0}, "positive"),
        ({"estimate": None}, "Task row 1"),
        ({"priority": 9}, "priority"),
        ({"deadline": "25:00"}, "not a time"),
    ],
)
def test_parse_tasks_reports_readable_errors(row, fragment):
    base = {"title": "x", "type": "email", "estimate": 30, "deadline": "", "priority": 3}
    _, errors = parse_tasks(pd.DataFrame([{**base, **row}]))
    assert len(errors) == 1 and fragment in errors[0]


# --- the planning session --------------------------------------------------------------------------

EVENTS = [Event("e1", hhmm("12:00"), hhmm("13:00"), "Lunch")]
TASKS = [
    Task("t1", "writing", 60, hhmm("12:00"), 5, title="Proposal"),
    Task("t2", "email", 20, None, 3, title="Email"),
    Task("t3", "coding", 60, hhmm("17:00"), 4, title="Bug"),
]
WINDOW = [(hhmm("09:00"), hhmm("18:00"))]


def start(estimators, name="trust", log=None, history=None):
    return PlanningSession.start(name, estimators[name], EVENTS, TASKS, WINDOW, history, log)


def test_session_overrun_replans_and_explains(fitted):
    s = start(fitted)
    first = s.next_task()
    assert first is not None and first.task_id == "t1"
    before = s.schedule
    s.overrun(30)
    assert [c.interval for c in s.schedule.completed] == [(first.start, first.end + 30)]
    assert s.now == first.end + 30
    assert s.schedule.events == before.events
    assert validate_schedule(s.schedule) == []
    assert all(p.start >= s.now for p in s.schedule.placements)
    assert s.changes and any("moved" in c for c in s.changes)


def test_overrun_into_a_fixed_event_is_called_out_and_the_event_stays_put(fitted):
    events = [Event("e1", hhmm("10:00"), hhmm("10:30"), "Standup")]
    tasks = [Task("t1", "email", 60, None, 3, title="Inbox")]
    s = PlanningSession.start("trust", fitted["trust"], events, tasks, WINDOW)
    assert s.next_task() is not None and s.next_task().end == hhmm("10:00")  # type: ignore[union-attr]
    s.overrun(20)
    assert any("20 min into **Standup**" in c for c in s.changes)
    assert not any(c.startswith("Nothing else") for c in s.changes)
    assert s.schedule.events == tuple(events)


def test_session_mark_done_advances_the_clock_and_clears_the_change_log(fitted):
    s = start(fitted)
    first = s.next_task()
    assert first is not None
    s.mark_done()
    assert s.now == first.end and s.changes == []
    assert [c.task_id for c in s.schedule.completed] == [first.task_id]


def test_session_refuses_to_act_with_nothing_left(fitted):
    s = start(fitted)
    while s.next_task():
        s.mark_done()
    with pytest.raises(ValueError):
        s.overrun(5)


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    st_h.lists(st_h.integers(0, 90), min_size=1, max_size=8), st_h.sampled_from(["trust", "gbm", "gbm_q80"])
)
def test_a_day_of_clicks_keeps_every_invariant(fitted, extras, name):
    s = start(fitted, name)
    events = s.schedule.events
    for extra in extras:
        if s.next_task() is None:
            break
        done_before = s.schedule.completed
        s.overrun(extra) if extra else s.mark_done()
        assert validate_schedule(s.schedule) == []
        assert s.schedule.events == events
        assert s.schedule.completed[: len(done_before)] == done_before or set(done_before) <= set(
            s.schedule.completed
        )


def test_session_writes_plans_and_completions_to_the_log(fitted):
    log = EventLog()
    s = start(fitted, "gbm", log)
    s.overrun(25)
    s.mark_done()
    assert log.count("plans") == 2 and log.count("completions") == 2
    assert log.db.execute("SELECT trigger FROM plans ORDER BY plan_id").fetchall() == [
        ("initial",),
        ("overrun",),
    ]
    assert log.db.execute("SELECT DISTINCT estimator FROM plans").fetchall() == [("gbm",)]
    actual = log.db.execute(
        "SELECT actual_minutes, planned_minutes FROM completions ORDER BY completion_id"
    ).fetchone()
    assert actual[0] == actual[1] + 25  # the logged actual includes the overrun
    assert log.history_frame(source="demo").shape[0] == 2


def test_ml_reserves_more_time_than_trusting_the_user_for_underestimated_types(fitted):
    trust, ml = start(fitted, "trust").schedule, start(fitted, "gbm").schedule
    assert sum(t.duration for t in ml.tasks) > sum(t.duration for t in trust.tasks)


def test_diff_schedules_describes_moves_and_new_drops():
    tasks = [
        Task("a", "x", 60, title="Alpha"),
        Task("b", "x", 60, deadline=hhmm("10:30"), priority=1, title="Beta"),
    ]
    old = build_schedule([], tasks, WINDOW)
    new = build_schedule([], tasks, WINDOW, now=hhmm("09:45"))
    notes = diff_schedules(old, new)
    assert any("Alpha" in n and "moved" in n for n in notes)
    assert any("Beta" in n and "can no longer fit" in n for n in notes)
    assert diff_schedules(old, old) == ["Nothing else needed to move."]


def test_compare_estimators_plans_identical_inputs_under_each_estimator(fitted):
    table, schedules = compare_estimators(fitted, EVENTS, TASKS, WINDOW)
    assert set(schedules) == set(fitted)
    assert len(table) == len(fitted)
    assert all(validate_schedule(s) == [] for s in schedules.values())
    reserved = dict(zip(schedules, table["Reserved minutes"], strict=True))
    assert reserved["trust"] == sum(t.estimated_minutes for t in TASKS)


def test_persona_history_returns_features_for_every_task(split):
    features, (ratio, n) = persona_history(SMALL, split["test"][0], TASKS)
    assert set(features) == {t.id for t in TASKS}
    assert n > 100 and 0.5 < ratio < 3.0
    assert {"user_lr_mean", "user_n", "user_type_lr_mean", "user_type_n"} == set(features["t1"])


# --- the Streamlit app itself, headless --------------------------------------------------------------


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fitted, split) -> AppTest:
    keep = ["trust", "global_multiplier", "type_median", "gbm", "gbm_q80"]
    joblib.dump(
        {"estimators": {k: fitted[k] for k in keep}, "config": SMALL, "split": split},
        tmp_path / "models.joblib",
    )
    monkeypatch.setenv("FLOWDAY_ARTIFACT", str(tmp_path / "models.joblib"))
    monkeypatch.setenv("FLOWDAY_DB", str(tmp_path / "flowday.db"))
    st.cache_resource.clear()
    return AppTest.from_file(APP_SCRIPT, default_timeout=120)


def test_app_plans_a_day_and_replans_on_overrun(app: AppTest, tmp_path: Path):
    at = app.run()
    assert not at.exception
    at.sidebar.radio(key="estimator").set_value("gbm")
    at.button(key="plan").click().run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.estimator_name == "gbm" and session.schedule.placements
    first = session.next_task()

    at.number_input(key="extra").set_value(20)
    at.button(key="overrun").click().run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.schedule.completed[0].end == first.end + 20
    assert any("Replanned" in info.value for info in at.info)

    at.button(key="done").click().run()
    assert len(at.session_state["session"].schedule.completed) == 2
    with EventLog(tmp_path / "flowday.db") as log:
        assert log.count("completions") == 2 and log.count("plans") == 2


def test_app_without_a_trained_model_still_works_with_baseline_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FLOWDAY_ARTIFACT", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("FLOWDAY_DB", str(tmp_path / "flowday.db"))
    st.cache_resource.clear()
    at = AppTest.from_file(APP_SCRIPT, default_timeout=60).run()
    assert not at.exception
    assert any("No trained model" in w.value for w in at.sidebar.warning)
    at.button(key="plan").click().run()
    assert not at.exception and at.session_state["session"].estimator_name == "trust"


def test_app_compare_tab_shows_every_estimator(app: AppTest):
    at = app.run()
    assert not at.exception
    assert len(at.selectbox(key="cmp_left").options) == 5
    assert at.selectbox(key="cmp_left").value == "trust" and at.selectbox(key="cmp_right").value == "gbm"
