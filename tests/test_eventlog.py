from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flowday.eventlog import EventLog, EventLogObserver
from flowday.models import GBMEstimator, add_history_features
from flowday.planning import EstimatorPlanner, task_features
from flowday.scheduler import Event, Task, build_schedule, complete
from flowday.simulator import TrustPlanner, generate_timeline, sample_users, simulate_day


def simulated_log(n_users: int = 4, n_days: int = 6) -> EventLog:
    log = EventLog()
    for u in sample_users(n_users, 5):
        for day in generate_timeline(u, n_days, 5):
            obs = EventLogObserver(log, f"synthetic-{u.user_id}", f"2026-01-{5 + day.index:02d}", "trust")
            simulate_day(day, TrustPlanner(), obs)
    return log


def test_schema_enforces_its_constraints():
    log = EventLog()
    with pytest.raises(sqlite3.IntegrityError):
        log.add_user("u", "scraped-from-somewhere")  # source must be one of the known labels
    log.add_user("u", "demo")
    s = build_schedule([], [Task("a", "x", 30)], [(540, 1080)])
    pid = log.log_plan("u", "2026-01-05", s, "trust")
    with pytest.raises(sqlite3.IntegrityError):
        log.log_completion("u", "missing-task", 540, 570)  # foreign key to tasks
    log.log_completion("u", "a", 540, 570, planned_minutes=30, plan_id=pid)
    with pytest.raises(sqlite3.IntegrityError):
        log.log_completion("u", "a", 540, 570)  # one completion per task
    with pytest.raises(sqlite3.IntegrityError):
        log.log_completion("u", "a", 600, 600)  # must have positive length
    assert log.db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == ("2",)


def test_plan_snapshot_records_events_tasks_completed_and_drops():
    log = EventLog()
    log.add_user("u", "demo")
    tasks = [Task("a", "x", 60), Task("b", "x", 600, deadline=700)]  # b can never fit
    s = build_schedule([Event("m", 600, 660, "meeting")], tasks, [(540, 1080)])
    s = complete(s, "a", 540, 600)
    pid = log.log_plan("u", "2026-01-05", s, "gbm", trigger="overrun")
    kinds = dict(log.db.execute("SELECT item_id, kind FROM plan_items WHERE plan_id=?", (pid,)).fetchall())
    assert kinds == {"m": "event", "a": "completed", "b": "dropped"}
    reason = log.db.execute("SELECT drop_reason FROM plan_items WHERE item_id='b'").fetchone()
    assert reason == ("no_fit_alone",)
    assert log.count("tasks") == 2 and log.count("plans") == 1


def test_a_simulated_day_is_fully_logged_with_a_replan_chain():
    log = simulated_log(2, 4)
    assert log.count("users") == 2
    # every plan except each day's first points at the plan it replaced
    rows = log.db.execute("SELECT trigger, parent_plan_id FROM plans").fetchall()
    assert all((t == "initial") == (parent is None) for t, parent in rows)
    assert any(t == "overrun" for t, _ in rows)
    # completions never reference a plan from another user
    bad = log.db.execute(
        "SELECT COUNT(*) FROM completions c JOIN plans p USING (plan_id) WHERE p.user_id != c.user_id"
    ).fetchone()
    assert bad == (0,)


def test_history_frame_matches_what_was_simulated():
    users = sample_users(2, 8)
    log = EventLog()
    truth = []
    for u in users:
        for day in generate_timeline(u, 3, 8):
            simulate_day(
                day,
                TrustPlanner(),
                EventLogObserver(log, f"u{u.user_id}", f"2026-02-{1 + day.index:02d}", "trust"),
            )
            out = simulate_day(day, TrustPlanner())
            truth += [
                (f"u{u.user_id}", c.task_id, c.end - c.start, (c.start % 1440) / 60)
                for c in out.schedule.completed
            ]
    df = log.history_frame().sort_values("task_id").reset_index(drop=True)
    expected = (
        pd.DataFrame(truth, columns=["user_id", "task_id", "actual", "hour"])
        .sort_values("task_id")
        .reset_index(drop=True)
    )
    assert df["task_id"].tolist() == expected["task_id"].tolist()
    assert df["actual"].tolist() == expected["actual"].tolist()
    np.testing.assert_allclose(df["hour"], expected["hour"])
    assert set(df["source"]) == {"synthetic"}
    assert log.history_frame("real").empty


def test_logged_data_can_train_a_model_as_real_data_later_would():
    log = simulated_log(12, 10)
    df = add_history_features(log.history_frame())
    model = GBMEstimator(0.8, 0, max_iter=20, min_samples_leaf=5).fit(df)
    pred = model.predict(df.head(20))
    assert np.isfinite(pred).all() and (pred > 0).all()
    assert {"user_lr_mean", "user_type_n"} <= set(df.columns)  # string user ids and ordinal days work


def test_log_persists_to_disk_and_reopens(tmp_path: Path):
    path = tmp_path / "sub" / "flowday.db"
    with EventLog(path) as log:
        log.add_user("u", "real")
    with EventLog(path) as again:
        assert again.count("users") == 1


def test_logging_does_not_change_simulation_results():
    u = sample_users(1, 2)[0]
    for day in generate_timeline(u, 5, 2):
        plain = simulate_day(day, TrustPlanner())
        logged = simulate_day(day, TrustPlanner(), EventLogObserver(EventLog(), "x", "2026-01-05", "trust"))
        assert plain.as_row() == logged.as_row()


def test_estimator_planner_plans_are_logged_with_the_estimator_name():
    log = EventLog()
    day = generate_timeline(sample_users(1, 4)[0], 1, 4)[0]

    class Double:
        name, uses_hour = "double", False

        def predict(self, df: pd.DataFrame) -> np.ndarray:
            return df["estimate"].to_numpy(dtype=float) * 2

    simulate_day(
        day,
        EstimatorPlanner(Double(), task_features(day.tasks)),
        EventLogObserver(log, "u", "2026-01-05", "double"),
    )
    assert log.db.execute("SELECT DISTINCT estimator FROM plans").fetchall() == [("double",)]
    planned = log.db.execute("SELECT planned_minutes FROM plan_items WHERE kind='task' LIMIT 1").fetchone()[0]
    assert planned is not None and planned % 2 == 0


# --- measured vs assumed durations, and schema migration -----------------------------------------------


def test_assumed_durations_are_kept_but_excluded_from_training_data():
    log = EventLog()
    log.add_user("u", "demo")
    s = build_schedule([], [Task("a", "x", 30), Task("b", "x", 30)], [(540, 1080)])
    log.log_plan("u", "2026-01-05", s, "trust")
    log.log_completion("u", "a", 540, 570, 30, measured=False)  # "done as planned": actual is just the plan
    log.log_completion("u", "b", 570, 615, 30, measured=True)
    assert log.count("completions") == 2
    assert log.history_frame()["task_id"].tolist() == ["b"]
    assert sorted(log.history_frame(measured_only=False)["task_id"]) == ["a", "b"]
    with pytest.raises(sqlite3.IntegrityError):
        log.db.execute("UPDATE completions SET measurement = 'guess' WHERE task_id = 'a'")


def test_plan_triggers_are_constrained():
    log = EventLog()
    log.add_user("u", "demo")
    s = build_schedule([], [Task("a", "x", 30)], [(540, 1080)])
    for trigger in ("initial", "overrun", "early", "manual"):
        log.log_plan("u", "2026-01-05", s, "trust", trigger=trigger)
    with pytest.raises(sqlite3.IntegrityError):
        log.log_plan("u", "2026-01-05", s, "trust", trigger="whenever")


V1_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT INTO meta VALUES ('schema_version', '1');
CREATE TABLE users (user_id TEXT PRIMARY KEY, source TEXT NOT NULL
    CHECK (source IN ('synthetic', 'demo', 'real')), created_at TEXT NOT NULL);
CREATE TABLE tasks (user_id TEXT NOT NULL REFERENCES users(user_id), task_id TEXT NOT NULL, day TEXT NOT NULL,
    task_type TEXT NOT NULL, estimated_minutes INTEGER NOT NULL CHECK (estimated_minutes > 0),
    deadline_min INTEGER, priority INTEGER NOT NULL, title TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, task_id));
CREATE TABLE plans (plan_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL REFERENCES users(user_id),
    day TEXT NOT NULL, created_at TEXT NOT NULL,
    trigger TEXT NOT NULL CHECK (trigger IN ('initial', 'overrun', 'manual')), estimator TEXT NOT NULL,
    parent_plan_id INTEGER REFERENCES plans(plan_id), now_min INTEGER NOT NULL);
CREATE TABLE plan_items (plan_id INTEGER NOT NULL REFERENCES plans(plan_id), kind TEXT NOT NULL
    CHECK (kind IN ('event', 'task', 'completed', 'dropped')), item_id TEXT NOT NULL, start_min INTEGER,
    end_min INTEGER, planned_minutes INTEGER, drop_reason TEXT, drop_detail TEXT,
    PRIMARY KEY (plan_id, kind, item_id));
CREATE TABLE completions (completion_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
    task_id TEXT NOT NULL, started_min INTEGER NOT NULL, ended_min INTEGER NOT NULL CHECK (ended_min > started_min),
    actual_minutes INTEGER NOT NULL, planned_minutes INTEGER, plan_id INTEGER REFERENCES plans(plan_id),
    logged_at TEXT NOT NULL, FOREIGN KEY (user_id, task_id) REFERENCES tasks(user_id, task_id),
    UNIQUE (user_id, task_id));
INSERT INTO users VALUES ('demo-u', 'demo', 't'), ('syn-u', 'synthetic', 't');
INSERT INTO tasks VALUES ('demo-u', 'a', '2026-01-05', 'coding', 30, NULL, 3, '', 't'),
                         ('syn-u', 'b', '2026-01-05', 'coding', 30, NULL, 3, '', 't');
INSERT INTO plans VALUES (1, 'demo-u', '2026-01-05', 't', 'initial', 'trust', NULL, 540),
                         (2, 'demo-u', '2026-01-05', 't', 'overrun', 'trust', 1, 600);
INSERT INTO plan_items VALUES (1, 'task', 'a', 540, 570, 30, NULL, NULL), (2, 'completed', 'a', 540, 600, 30, NULL, NULL);
INSERT INTO completions VALUES (1, 'demo-u', 'a', 540, 600, 60, 30, 2, 't'), (2, 'syn-u', 'b', 540, 590, 50, 30, NULL, 't');
"""


def test_a_v1_database_is_migrated_in_place_without_losing_data(tmp_path: Path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript(V1_SCHEMA)
    db.commit()
    db.close()
    with EventLog(path) as log:
        assert log.db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == ("2",)
        assert [log.count(t) for t in ("users", "tasks", "plans", "plan_items", "completions")] == [
            2,
            2,
            2,
            2,
            2,
        ]
        by_user = dict(log.db.execute("SELECT user_id, measurement FROM completions").fetchall())
        assert by_user == {"demo-u": "assumed", "syn-u": "measured"}  # v1 demo rows could not say
        assert log.db.execute("SELECT parent_plan_id FROM plans WHERE plan_id = 2").fetchone() == (1,)
        assert log.db.execute("PRAGMA foreign_key_check").fetchall() == []
        s = build_schedule([], [Task("n", "x", 30)], [(540, 1080)])
        log.log_plan("demo-u", "2026-01-06", s, "trust", trigger="early")  # new trigger now allowed
        assert log.history_frame()["task_id"].tolist() == ["b"]  # assumed row excluded from training
    with EventLog(path) as again:  # reopening a migrated file is a no-op
        assert again.count("completions") == 2


def test_an_unknown_schema_version_is_refused(tmp_path: Path):
    path = tmp_path / "future.db"
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL); INSERT INTO meta VALUES ('schema_version', '99');"
    )
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="unknown schema_version"):
        EventLog(path)
