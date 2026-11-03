"""App logic with no Streamlit in it: plan a day, mark tasks done, report overruns, explain changes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from flowday.dataset import DataConfig
from flowday.eventlog import EventLog, EventLogObserver
from flowday.models import Estimator, history_features
from flowday.planning import EstimatorPlanner, task_features
from flowday.scheduler import (
    Event,
    Interval,
    Overrun,
    Placement,
    Schedule,
    Task,
    collisions,
    complete,
    fmt_time,
)
from flowday.simulator import generate_timeline, history_frame, sample_users

APP_USER = "app-demo"

ESTIMATOR_LABELS = {
    "trust": "Trust my estimates (baseline)",
    "global_multiplier": "Global multiplier (baseline)",
    "type_median": "Per-type median ratio (baseline)",
    "gbm": "ML: gradient boosting (median)",
    "gbm_q80": "ML: gradient boosting, cautious (80th percentile)",
}


def title_of(schedule: Schedule, task_id: str) -> str:
    return schedule.task(task_id).title or task_id


def diff_schedules(old: Schedule, new: Schedule) -> list[str]:
    """Plain-English list of what a replan changed for the remaining tasks."""
    notes: list[str] = []
    old_at = {p.task_id: p for p in old.placements}
    new_at = {p.task_id: p for p in new.placements}
    for task_id, p in sorted(new_at.items(), key=lambda kv: kv[1].start):
        before = old_at.get(task_id)
        name = title_of(new, task_id)
        if before is None:
            notes.append(f"**{name}** is back in the plan at {fmt_time(p.start)}")
        elif before.start != p.start:
            notes.append(f"**{name}** moved {fmt_time(before.start)} → {fmt_time(p.start)}")
    old_dropped = {d.task_id for d in old.dropped}
    for d in new.dropped:
        if d.task_id not in old_dropped:
            notes.append(f"**{title_of(new, d.task_id)}** can no longer fit: {d.detail}")
    return notes or ["Nothing else needed to move."]


@dataclass
class PlanningSession:
    """One planned day being lived: the current schedule plus the clock."""

    estimator_name: str
    planner: EstimatorPlanner
    schedule: Schedule
    now: int
    observer: EventLogObserver | None = None
    changes: list[str] = field(default_factory=list)

    @classmethod
    def start(
        cls,
        estimator_name: str,
        estimator: Estimator,
        events: Sequence[Event],
        tasks: Sequence[Task],
        windows: Sequence[Interval],
        history: Mapping[str, Mapping[str, float]] | None = None,
        log: EventLog | None = None,
    ) -> PlanningSession:
        planner = EstimatorPlanner(estimator, task_features(tasks, history))
        schedule = planner.plan(events, tasks, windows)
        observer = None
        if log is not None:
            observer = EventLogObserver(
                log, APP_USER, date.today().isoformat(), estimator_name, source="demo"
            )
            observer.on_plan("initial", schedule)
        return cls(estimator_name, planner, schedule, schedule.now, observer)

    def next_task(self) -> Placement | None:
        return min(self.schedule.placements, key=lambda p: p.start, default=None)

    def mark_done(self) -> None:
        """The next task finished exactly as planned."""
        p = self._require_next()
        self.schedule = complete(self.schedule, p.task_id, p.start, p.end)
        self.now = p.end
        self.changes = []
        if self.observer:
            self.observer.on_completion(p.task_id, p.start, p.end, p.end - p.start)

    def overrun(self, extra_minutes: int) -> None:
        """The next task took `extra_minutes` longer than planned: log it and replan the rest."""
        p = self._require_next()
        end = p.end + extra_minutes
        before = self.schedule
        self.schedule = self.planner.replan(before, end, Overrun(p.task_id, extra_minutes))
        self.now = end
        self.changes = diff_schedules(before, self.schedule)
        spills = [
            f"⚠ **{title_of(self.schedule, p.task_id)}** now runs {minutes} min into "
            f"**{event.title or event.id}**; fixed events never move"
            for task_id, event_id, minutes in collisions(self.schedule)
            if task_id == p.task_id
            for event in self.schedule.events
            if event.id == event_id
        ]
        if spills:
            self.changes = spills + [c for c in self.changes if not c.startswith("Nothing else")]
        if self.observer:
            self.observer.on_completion(p.task_id, p.start, end, p.end - p.start)
            self.observer.on_plan("overrun", self.schedule)

    def _require_next(self) -> Placement:
        p = self.next_task()
        if p is None:
            raise ValueError("nothing left to do")
        return p


def persona_history(
    cfg: DataConfig, user_id: int, tasks: Sequence[Task]
) -> tuple[dict[str, dict[str, float]], tuple[float, int]]:
    """History features for `tasks` as if they belonged to synthetic user `user_id`, who has
    `cfg.eval_start` days of logged history. Also returns (typical actual/estimate ratio, rows)."""
    profile = next(u for u in sample_users(cfg.n_users, cfg.seed) if u.user_id == user_id)
    hist = history_frame(generate_timeline(profile, cfg.eval_start, cfg.seed), cfg.seed)
    queries = pd.DataFrame(
        {"user_id": user_id, "day": cfg.eval_start, "task_type": [t.type for t in tasks]},
        index=[t.id for t in tasks],
    )
    feats = history_features(hist, queries)
    summary = (
        (math.exp(float(feats["user_lr_mean"].iloc[0])), int(feats["user_n"].iloc[0]))
        if len(feats)
        else (1.0, 0)
    )
    return feats.to_dict(orient="index"), summary  # type: ignore[return-value]


def compare_estimators(
    estimators: Mapping[str, Estimator],
    events: Sequence[Event],
    tasks: Sequence[Task],
    windows: Sequence[Interval],
    history: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Schedule]]:
    """Plan the same inputs under each estimator. Returns (summary table, schedules)."""
    features = task_features(tasks, history)
    schedules = {
        name: EstimatorPlanner(est, features).plan(events, tasks, windows) for name, est in estimators.items()
    }
    rows = []
    for name, s in schedules.items():
        rows.append(
            {
                "Estimator": ESTIMATOR_LABELS.get(name, name),
                "Reserved minutes": sum(t.duration for t in s.tasks),
                "Scheduled": len(s.placements),
                "Dropped": ", ".join(title_of(s, d.task_id) for d in s.dropped) or "none",
                "Last task ends": fmt_time(max((p.end for p in s.placements), default=s.now)),
            }
        )
    return pd.DataFrame(rows), schedules
