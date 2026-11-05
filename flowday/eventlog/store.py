"""Local SQLite event log: plans, completions and actual durations.

The schema is meant to take real user data later. Rows say where they came from via
`users.source` ('synthetic' | 'demo' | 'real'); nothing else distinguishes them, so a model trained
on `EventLog.history_frame()` does not care. Times are integer minutes since midnight of the
row's `day` (an ISO date); a plan spanning several days simply uses minutes beyond 1440.

    users 1--* tasks 1--0..1 completions
    users 1--* plans 1--* plan_items         (a plan is one snapshot: initial, or after a replan)

Each completion says whether its duration was `measured` (the user reported or a system observed how
long the task took) or `assumed` (e.g. "done as planned" was clicked, so the actual is just the plan).
Training uses measured rows only: assumed durations would teach a model its own predictions back.

Schema history: v1 (initial); v2 adds `completions.measurement` and the 'early' plan trigger.
Older files are migrated in place when opened.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from flowday.scheduler import MINUTES_PER_DAY, Schedule

SCHEMA_VERSION = "2"
SOURCES = ("synthetic", "demo", "real")
TRIGGERS = ("initial", "overrun", "early", "manual")
MEASUREMENTS = ("measured", "assumed")


def _plans_ddl(name: str, if_not_exists: bool = False) -> str:
    return f"""
CREATE TABLE {"IF NOT EXISTS " if if_not_exists else ""}{name} (
    plan_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        TEXT NOT NULL REFERENCES users(user_id),
    day            TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    trigger        TEXT NOT NULL CHECK (trigger IN ({", ".join(repr(t) for t in TRIGGERS)})),
    estimator      TEXT NOT NULL,                         -- which duration estimator sized the tasks
    parent_plan_id INTEGER REFERENCES plans(plan_id),     -- the plan this one replaced
    now_min        INTEGER NOT NULL                       -- nothing is planned before this minute
);
"""


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT INTO meta VALUES ('schema_version', '{SCHEMA_VERSION}') ON CONFLICT(key) DO NOTHING;

CREATE TABLE IF NOT EXISTS users (
    user_id    TEXT PRIMARY KEY,
    source     TEXT NOT NULL CHECK (source IN ({", ".join(repr(s) for s in SOURCES)})),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    user_id           TEXT NOT NULL REFERENCES users(user_id),
    task_id           TEXT NOT NULL,
    day               TEXT NOT NULL,                      -- ISO date the task was planned for
    task_type         TEXT NOT NULL,
    estimated_minutes INTEGER NOT NULL CHECK (estimated_minutes > 0),
    deadline_min      INTEGER,                            -- minutes since midnight of `day`; NULL = none
    priority          INTEGER NOT NULL,
    title             TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL,
    PRIMARY KEY (user_id, task_id)
);

{_plans_ddl("plans", if_not_exists=True)}
CREATE TABLE IF NOT EXISTS plan_items (
    plan_id         INTEGER NOT NULL REFERENCES plans(plan_id),
    kind            TEXT NOT NULL CHECK (kind IN ('event', 'task', 'completed', 'dropped')),
    item_id         TEXT NOT NULL,
    start_min       INTEGER,
    end_min         INTEGER,
    planned_minutes INTEGER,
    drop_reason     TEXT,
    drop_detail     TEXT,
    PRIMARY KEY (plan_id, kind, item_id)
);

CREATE TABLE IF NOT EXISTS completions (
    completion_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    started_min     INTEGER NOT NULL,
    ended_min       INTEGER NOT NULL CHECK (ended_min > started_min),
    actual_minutes  INTEGER NOT NULL,
    planned_minutes INTEGER,
    plan_id         INTEGER REFERENCES plans(plan_id),
    logged_at       TEXT NOT NULL,
    measurement     TEXT NOT NULL DEFAULT 'measured'
                    CHECK (measurement IN ({", ".join(repr(m) for m in MEASUREMENTS)})),
    FOREIGN KEY (user_id, task_id) REFERENCES tasks(user_id, task_id),
    UNIQUE (user_id, task_id)
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _migrate(db: sqlite3.Connection) -> None:
    """Bring an existing database up to SCHEMA_VERSION. A fresh (empty) database is left alone."""
    has_meta = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone()
    if not has_meta:
        return
    row = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    version = row[0] if row else None
    if version == SCHEMA_VERSION:
        return
    if version != "1":
        raise ValueError(
            f"event log has unknown schema_version {version!r}; this code expects {SCHEMA_VERSION}"
        )
    # v1 -> v2. CHECK constraints cannot be altered in place, so rebuild `plans`.
    db.execute("PRAGMA foreign_keys = OFF")
    db.executescript(
        f"""
BEGIN;
{_plans_ddl("plans_v2")}
INSERT INTO plans_v2 SELECT * FROM plans;
DROP TABLE plans;
ALTER TABLE plans_v2 RENAME TO plans;
ALTER TABLE completions ADD COLUMN measurement TEXT NOT NULL DEFAULT 'measured'
    CHECK (measurement IN ({", ".join(repr(m) for m in MEASUREMENTS)}));
-- v1 demo-app completions could not say whether the duration was measured; treat them as assumed
UPDATE completions SET measurement = 'assumed'
    WHERE user_id IN (SELECT user_id FROM users WHERE source = 'demo');
UPDATE meta SET value = '{SCHEMA_VERSION}' WHERE key = 'schema_version';
COMMIT;
"""
    )


class EventLog:
    """Thin wrapper over a SQLite file (or ":memory:")."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)  # Streamlit may rerun on another thread
        _migrate(self.db)
        self.db.executescript(SCHEMA)
        self.db.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def add_user(self, user_id: str, source: str) -> None:
        self.db.execute(
            "INSERT INTO users VALUES (?, ?, ?) ON CONFLICT(user_id) DO NOTHING", (user_id, source, _now())
        )
        self.db.commit()

    def log_plan(
        self,
        user_id: str,
        day: str,
        schedule: Schedule,
        estimator: str,
        trigger: str = "initial",
        parent_plan_id: int | None = None,
    ) -> int:
        """Snapshot `schedule` (and register any tasks not yet logged). Returns the plan id."""
        cur = self.db.cursor()
        for t in schedule.tasks:
            cur.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(user_id, task_id) DO NOTHING",
                (user_id, t.id, day, t.type, t.estimated_minutes, t.deadline, t.priority, t.title, _now()),
            )
        cur.execute(
            "INSERT INTO plans (user_id, day, created_at, trigger, estimator, parent_plan_id, now_min)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, day, _now(), trigger, estimator, parent_plan_id, schedule.now),
        )
        plan_id = cur.lastrowid
        assert plan_id is not None
        rows: list[tuple[object, ...]] = [
            (plan_id, "event", e.id, e.start, e.end, None, None, None) for e in schedule.events
        ]
        rows += [
            (plan_id, "task", p.task_id, p.start, p.end, schedule.task(p.task_id).duration, None, None)
            for p in schedule.placements
        ]
        rows += [
            (plan_id, "completed", p.task_id, p.start, p.end, schedule.task(p.task_id).duration, None, None)
            for p in schedule.completed
        ]
        rows += [
            (
                plan_id,
                "dropped",
                d.task_id,
                None,
                None,
                schedule.task(d.task_id).duration,
                d.reason.value,
                d.detail,
            )
            for d in schedule.dropped
        ]
        cur.executemany("INSERT INTO plan_items VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        self.db.commit()
        return plan_id

    def log_completion(
        self,
        user_id: str,
        task_id: str,
        started_min: int,
        ended_min: int,
        planned_minutes: int | None = None,
        plan_id: int | None = None,
        measured: bool = True,
    ) -> None:
        """Record that a task ran [started_min, ended_min). `measured=False` marks a duration that was
        assumed (e.g. "done as planned") rather than observed; such rows are not used for training."""
        self.db.execute(
            "INSERT INTO completions (user_id, task_id, started_min, ended_min, actual_minutes,"
            " planned_minutes, plan_id, logged_at, measurement) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                task_id,
                started_min,
                ended_min,
                ended_min - started_min,
                planned_minutes,
                plan_id,
                _now(),
                "measured" if measured else "assumed",
            ),
        )
        self.db.commit()

    def history_frame(self, source: str | None = None, measured_only: bool = True) -> pd.DataFrame:
        """Completed tasks as a training table (user_id, day, task_type, estimate, hour, actual).

        `day` is the ISO date's ordinal, so it sorts correctly; this is exactly what
        `flowday.models.add_history_features` and the estimators' `fit` expect. Rows whose duration
        was only assumed (not measured) are left out unless `measured_only=False`.
        """
        query = (
            "SELECT t.user_id, t.day, t.task_id, t.task_type, t.estimated_minutes AS estimate,"
            " c.started_min, c.actual_minutes AS actual, u.source"
            " FROM completions c JOIN tasks t USING (user_id, task_id) JOIN users u USING (user_id)"
        )
        conditions: list[str] = []
        params: tuple[str, ...] = ()
        if source is not None:
            conditions.append("u.source = ?")
            params = (source,)
        if measured_only:
            conditions.append("c.measurement = 'measured'")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        df = pd.read_sql_query(query + " ORDER BY t.day, c.started_min", self.db, params=params)
        df["day"] = df["day"].map(lambda d: date.fromisoformat(d).toordinal())
        df["hour"] = (df.pop("started_min") % MINUTES_PER_DAY) / 60
        return df[["source", "user_id", "day", "task_id", "task_type", "estimate", "hour", "actual"]]

    def count(self, table: str) -> int:
        assert table in {"users", "tasks", "plans", "plan_items", "completions"}
        row = self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0])


class EventLogObserver:
    """Feeds a simulated (or live) day into an EventLog. Implements `simulator.Observer`."""

    def __init__(
        self, log: EventLog, user_id: str, day: str, estimator: str, source: str = "synthetic"
    ) -> None:
        self.log, self.user_id, self.day, self.estimator = log, user_id, day, estimator
        self.plan_id: int | None = None
        log.add_user(user_id, source)

    def on_plan(self, trigger: str, schedule: Schedule) -> None:
        self.plan_id = self.log.log_plan(
            self.user_id, self.day, schedule, self.estimator, trigger, parent_plan_id=self.plan_id
        )

    def on_completion(
        self, task_id: str, start: int, end: int, planned_minutes: int, measured: bool = True
    ) -> None:
        self.log.log_completion(self.user_id, task_id, start, end, planned_minutes, self.plan_id, measured)
