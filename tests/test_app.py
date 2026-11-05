from __future__ import annotations

import math
import sqlite3
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import pytest
import streamlit as st
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st_h
from streamlit.testing.v1 import AppTest

from flowday.app.inputs import parse_events, parse_routines, parse_tasks, parse_weekdays
from flowday.app.session import (
    APP_USER,
    ESTIMATOR_LABELS,
    RECLAIM_MIN_GAIN,
    PlanningSession,
    compare_estimators,
    diff_schedules,
    log_history,
    persona_history,
)
from flowday.eventlog import EventLog, EventLogObserver
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


def start(estimators, name="trust", log=None, history=None, **kwargs):
    return PlanningSession.start(name, estimators[name], EVENTS, TASKS, WINDOW, history, log, **kwargs)


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


def test_closing_a_session_releases_its_log(fitted):
    log = EventLog()
    s = start(fitted, "trust", log)
    s.close()
    with pytest.raises(sqlite3.ProgrammingError):
        log.count("plans")
    start(fitted).close()  # a session without a log closes quietly


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
    # "done as planned" is only assumed, so just the overrun is usable training data
    assert log.history_frame(source="demo").shape[0] == 1
    assert log.history_frame(source="demo", measured_only=False).shape[0] == 2
    assert log.db.execute("SELECT measurement FROM completions ORDER BY completion_id").fetchall() == [
        ("measured",),
        ("assumed",),
    ]


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
    keep = ["trust", "global_multiplier", "type_median", "gbm", "gbm_q60", "gbm_q70", "gbm_q80", "gbm_q90"]
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


def test_hosted_mode_trains_its_own_model_and_keeps_nothing_on_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fitted
):
    monkeypatch.setenv("FLOWDAY_HOSTED", "1")
    monkeypatch.setenv("FLOWDAY_ARTIFACT", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("FLOWDAY_DB", str(tmp_path / "flowday.db"))
    monkeypatch.setattr("flowday.models.train.quick_fit", lambda cfg: fitted)  # skip the real training
    st.cache_resource.clear()
    at = AppTest.from_file(APP_SCRIPT, default_timeout=60).run()
    assert not at.exception
    assert ESTIMATOR_LABELS["gbm"] in at.sidebar.radio(key="estimator").options  # ML without any clicking
    assert not any("No trained model" in w.value for w in at.sidebar.warning)
    assert [b.key for b in at.sidebar.button] == [] and [c.key for c in at.sidebar.checkbox] == []
    at.sidebar.radio(key="estimator").set_value("gbm")
    at.button(key="plan").click().run()
    at.button(key="overrun").click().run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.observer is None and session.schedule.completed
    assert not (tmp_path / "flowday.db").exists()
    assert any("keeps no event log" in i.value for i in at.info)


def test_session_carry_over_returns_the_unfinished_tasks_ready_for_tomorrow(fitted):
    s = start(fitted, "trust")
    first = s.next_task()
    assert first is not None
    s.mark_done()
    left = s.carry_over(1)
    assert s.unfinished == len(left) == len(TASKS) - 1
    assert first.task_id not in {t.id.removeprefix("d1-") for t in left}
    assert all(t.id.startswith("d1-") and t.carried == 1 and t.deadline is None for t in left)


def test_a_task_carried_a_second_time_keeps_its_id_and_counts_another_day(fitted):
    first = start(fitted, "trust").carry_over(1)
    tomorrow = PlanningSession.start("trust", fitted["trust"], EVENTS, first, WINDOW)
    again = tomorrow.carry_over(2)
    assert {t.id for t in again} == {t.id for t in first} and all(t.carried == 2 for t in again)


def test_carried_tasks_join_tomorrows_plan_and_carry_again_with_unique_ids(app: AppTest):
    at = app.run()
    at.sidebar.selectbox(key="weekday").set_value(1)  # Tuesday: only the daily routine applies
    at.button(key="plan").click().run()
    at.button(key="done").click().run()
    unfinished = at.session_state["session"].unfinished
    at.button(key="carry").click().run()
    assert not at.exception
    carried = at.session_state["carried"]
    assert len(carried) == unfinished and all(c.carried == 1 and c.id.startswith("d1-") for c in carried)
    assert "session" not in at.session_state and any(
        "carried over from yesterday" in i.value for i in at.info
    )

    at.button(key="plan").click().run()  # tomorrow: its own tasks plus the carried ones
    assert not at.exception
    tomorrow = at.session_state["session"].schedule
    assert {c.id for c in carried} <= {t.id for t in tomorrow.tasks} and validate_schedule(tomorrow) == []
    at.button(key="carry").click().run()  # carry everything again, untouched
    again = at.session_state["carried"]
    assert len({t.id for t in again}) == len(again)  # old ones keep d1-, tomorrow's own get d2-
    assert {t.carried for t in again if t.id.startswith("d1-")} == {2}
    assert {t.carried for t in again if t.id.startswith("d2-")} == {1}


def test_carried_tasks_can_be_dropped(app: AppTest):
    at = app.run()
    at.button(key="plan").click().run()
    at.button(key="carry").click().run()
    assert at.session_state["carried"]
    at.button(key="drop_carried").click().run()
    assert not at.exception and "carried" not in at.session_state


def test_app_compare_tab_shows_every_estimator(app: AppTest):
    at = app.run()
    assert not at.exception
    assert len(at.selectbox(key="cmp_left").options) == 8
    assert at.selectbox(key="cmp_left").value == "trust" and at.selectbox(key="cmp_right").value == "gbm"


# --- routines -----------------------------------------------------------------------------------------


def test_parse_weekdays():
    assert parse_weekdays("daily") == frozenset(range(7)) and parse_weekdays("") == frozenset(range(7))
    assert parse_weekdays("weekdays") == frozenset(range(5)) and parse_weekdays("weekends") == frozenset(
        {5, 6}
    )
    assert parse_weekdays("Mon, Wed Fri") == frozenset({0, 2, 4})
    assert parse_weekdays("Tuesday,thu") == frozenset({1, 3})
    with pytest.raises(ValueError, match="unknown day"):
        parse_weekdays("Mon, Funday")


def test_parse_routines_fixed_and_flexible():
    df = pd.DataFrame(
        {
            "title": ["Gym", "Study", None],
            "kind": ["fixed", "flexible", None],
            "time": ["17:15", "", None],
            "minutes": [30, 60, None],
            "type": ["exercise", "study", None],
            "priority": [3, None, None],
            "days": ["Mon, Wed", "daily", None],
        }
    )
    routines, errors = parse_routines(df)
    assert errors == []
    gym, study = routines
    assert (gym.kind, gym.start_minute, gym.duration, gym.weekdays) == (
        "fixed",
        hhmm("17:15"),
        30,
        frozenset({0, 2}),
    )
    assert (study.kind, study.start_minute, study.deadline_minute, study.priority) == (
        "flexible",
        None,
        None,
        3,
    )
    assert len({r.id for r in routines}) == 2


@pytest.mark.parametrize(
    "row, fragment",
    [
        ({"kind": "fixed", "time": ""}, "needs a start time"),
        ({"kind": "sometimes"}, "kind must be"),
        ({"minutes": 0}, "duration must be positive"),
        ({"days": "Someday"}, "unknown day"),
        ({"time": "9am"}, "not a time"),
    ],
)
def test_parse_routines_reports_readable_errors(row, fragment):
    base = {
        "title": "x",
        "kind": "fixed",
        "time": "09:00",
        "minutes": 30,
        "type": "admin",
        "priority": 3,
        "days": "daily",
    }
    _, errors = parse_routines(pd.DataFrame([{**base, **row}]))
    assert len(errors) == 1 and fragment in errors[0] and "Routine row 1" in errors[0]


# --- real durations: measured, early, assumed -----------------------------------------------------------


def test_finishing_early_records_a_measured_duration_and_pulls_the_day_forward(fitted):
    log = EventLog()
    s = start(fitted, "trust", log)
    first, second = sorted(s.schedule.placements, key=lambda p: p.start)[:2]
    s.finish_early(10)
    done = s.schedule.completed[0]
    assert (done.task_id, done.end) == (first.task_id, first.end - 10) and s.now == first.end - 10
    assert min(p.start for p in s.schedule.placements) >= s.now
    new_second = s.schedule.placement_of(second.task_id)
    assert new_second is not None and new_second.start <= second.start  # nothing was pushed later
    assert any("moved" in c for c in s.changes) or s.changes == ["Nothing else needed to move."]
    assert validate_schedule(s.schedule) == []
    assert log.db.execute("SELECT trigger FROM plans ORDER BY plan_id").fetchall() == [
        ("initial",),
        ("early",),
    ]
    assert log.db.execute("SELECT measurement FROM completions").fetchall() == [("measured",)]


def test_a_small_early_finish_keeps_the_plan_and_a_large_one_pulls_the_day_forward(fitted):
    log = EventLog()
    small = start(fitted, "trust", log)
    before = {p.task_id: p.start for p in small.schedule.placements}
    small.finish_early(RECLAIM_MIN_GAIN - 1)
    after = {p.task_id: p.start for p in small.schedule.placements}
    assert not small.replanned and all(after[t] == before[t] for t in after)
    assert len(small.changes) == 1 and "stays as planned" in small.changes[0]
    assert log.db.execute("SELECT trigger FROM plans").fetchall() == [("initial",)]  # measured, but no replan

    big = start(fitted, "trust")
    big.finish_early(RECLAIM_MIN_GAIN)
    assert big.replanned and min(p.start for p in big.schedule.placements) >= big.now

    eager = start(fitted, "trust", reclaim_min_gain=1)
    eager.finish_early(RECLAIM_MIN_GAIN - 1)
    assert eager.replanned


def test_done_as_planned_is_assumed_but_overrun_and_early_are_measured(fitted):
    log = EventLog()
    s = start(fitted, "trust", log)
    s.mark_done()
    s.overrun(5)
    s.finish_early(5)
    rows = log.db.execute("SELECT measurement FROM completions ORDER BY completion_id").fetchall()
    assert rows == [("assumed",), ("measured",), ("measured",)]
    assert len(log.history_frame(source="demo")) == 2


def test_finishing_early_never_leaves_a_zero_length_task(fitted):
    s = start(fitted, "trust")
    first = s.next_task()
    assert first is not None
    s.finish_early(10_000)
    assert (
        s.schedule.completed[0].end - s.schedule.completed[0].start == 1
        and first.task_id == s.schedule.completed[0].task_id
    )


# --- history from the user's own log ---------------------------------------------------------------------


def test_log_history_uses_only_the_app_users_earlier_days():
    rows = pd.DataFrame(
        {
            "source": "demo",
            "user_id": [APP_USER, APP_USER, "someone-else"],
            "day": [date(2026, 1, 5).toordinal(), date(2026, 1, 6).toordinal(), date(2026, 1, 5).toordinal()],
            "task_id": ["a", "b", "c"],
            "task_type": ["coding", "coding", "coding"],
            "estimate": [30, 30, 30],
            "hour": 10.0,
            "actual": [60, 45, 600],
        }
    )
    feats, (ratio, n) = log_history(rows, [Task("t1", "coding", 30)], today=date(2026, 1, 7))
    assert n == 2 and ratio == pytest.approx(
        math.sqrt(2.0 * 1.5)
    )  # geometric mean of 2.0 and 1.5; "someone-else" ignored
    assert feats["t1"]["user_type_n"] == 2
    none_yet, (r0, n0) = log_history(rows, [Task("t1", "coding", 30)], today=date(2026, 1, 5))
    assert n0 == 0 and r0 == 1.0 and none_yet["t1"]["user_n"] == 0


# --- the app with routines, early finishes and a retrained model ---------------------------------------


def test_routines_apply_only_on_their_weekdays(app: AppTest):
    at = app.run()
    at.sidebar.selectbox(key="weekday").set_value(0)  # Monday: the gym routine applies
    at.button(key="plan").click().run()
    assert not at.exception
    monday = at.session_state["session"].schedule
    assert "Gym" in [e.title for e in monday.events]
    assert "Deep-work study block" in [t.title for t in monday.tasks]

    at.sidebar.selectbox(key="weekday").set_value(1)  # Tuesday: study block (daily) yes, gym no
    at.button(key="plan").click().run()
    tuesday = at.session_state["session"].schedule
    assert "Gym" not in [e.title for e in tuesday.events]
    assert "Deep-work study block" in [t.title for t in tuesday.tasks]


def test_app_finished_early_button_logs_a_measured_duration(app: AppTest, tmp_path: Path):
    at = app.run()
    at.button(key="plan").click().run()
    at.number_input(key="saved").set_value(10)
    at.button(key="early").click().run()
    assert not at.exception
    assert len(at.session_state["session"].schedule.completed) == 1
    with EventLog(tmp_path / "flowday.db") as log:
        assert log.db.execute("SELECT measurement FROM completions").fetchall() == [("measured",)]
        assert log.db.execute("SELECT trigger FROM plans ORDER BY plan_id").fetchall()[-1] in {
            ("early",),
            ("initial",),
        }


def test_sidebar_threshold_decides_whether_a_small_early_finish_replans(app: AppTest):
    at = app.run()
    assert at.sidebar.number_input(key="reclaim_gain").value == RECLAIM_MIN_GAIN
    at.sidebar.number_input(key="reclaim_gain").set_value(1).run()
    at.button(key="plan").click().run()
    at.number_input(key="saved").set_value(5)
    at.button(key="early").click().run()
    assert not at.exception and at.session_state["session"].replanned


def test_app_can_retrain_on_the_users_own_log(app: AppTest, tmp_path: Path):
    from datetime import timedelta

    from flowday.simulator import TrustPlanner, generate_timeline, sample_users, simulate_day

    with EventLog(tmp_path / "flowday.db") as log:  # a pre-existing log with enough measured history
        for day in generate_timeline(sample_users(1, 9)[0], 25, 9):
            iso = (date.today() - timedelta(days=30 - day.index)).isoformat()
            simulate_day(day, TrustPlanner(), EventLogObserver(log, APP_USER, iso, "trust", source="demo"))
    at = app.run()
    assert not at.exception
    assert at.sidebar.button(key="retrain").disabled is False
    mine = {ESTIMATOR_LABELS["mine_gbm"], ESTIMATOR_LABELS["mine_gbm_q80"]}
    assert not mine & set(at.sidebar.radio(key="estimator").options)  # only offered after retraining
    at.sidebar.button(key="retrain").click().run()
    assert not at.exception
    assert mine <= set(at.sidebar.radio(key="estimator").options)
    at.sidebar.selectbox(key="persona").set_value(-1)  # plan for "me", using my own history
    at.sidebar.radio(key="estimator").set_value("mine_gbm")
    at.button(key="plan").click().run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.estimator_name == "mine_gbm" and session.schedule.placements
    assert any("typically take" in c.value for c in at.caption)


def test_retrain_button_is_disabled_without_enough_measured_history(app: AppTest):
    at = app.run()
    assert at.sidebar.button(key="retrain").disabled is True


def test_the_caution_slider_picks_the_planning_percentile(app: AppTest):
    at = app.run()
    assert not at.sidebar.select_slider  # no slider until a cautious ML estimate is chosen
    at.sidebar.radio(key="estimator").set_value("gbm_q80").run()
    slider = at.sidebar.select_slider(key="caution")
    assert slider.value == 80
    assert slider.options == ["60th percentile", "70th percentile", "80th percentile", "90th percentile"]
    slider.set_value(90).run()
    at.button(key="plan").click().run()
    assert not at.exception
    assert at.session_state["session"].estimator_name == "gbm_q90"
    # the extra percentiles are reached through the slider, not as separate radio options
    assert len(at.sidebar.radio(key="estimator").options) == 5
