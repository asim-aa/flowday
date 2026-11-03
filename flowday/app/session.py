"""App logic with no Streamlit in it: plan a day, mark tasks done, report overruns, explain changes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from flowday.dataset import DataConfig, make_users
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
from flowday.simulator import generate_timeline, history_frame

APP_USER = "app-demo"
# An early finish pulls the rest of the day forward only if it frees at least this many minutes. Chosen at
# the knee of the sweep in reports/sensitivity.md ("Reclaim threshold"): for the cautious planner it keeps
# about 80% of the idle-time saving of reclaiming everything and halves the extra start-time churn.
RECLAIM_MIN_GAIN = 10

ESTIMATOR_LABELS = {
    "trust": "Trust my estimates (baseline)",
    "global_multiplier": "Global multiplier (baseline)",
    "type_median": "Per-type median ratio (baseline)",
    "gbm": "ML: gradient boosting (median)",
    "gbm_q80": "ML: gradient boosting, cautious (80th percentile)",
    "gbm_q60": "ML: gradient boosting, 60th percentile",
    "gbm_q70": "ML: gradient boosting, 70th percentile",
    "gbm_q90": "ML: gradient boosting, 90th percentile (very cautious)",
    "mine_gbm": "ML trained on my own log (median)",
    "mine_gbm_q80": "ML trained on my own log, cautious (80th percentile)",
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
    replanned: bool = False  # did the last completion change the plan (as opposed to a note about keeping it)
    reclaim_min_gain: int = RECLAIM_MIN_GAIN

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
        reclaim_min_gain: int = RECLAIM_MIN_GAIN,
    ) -> PlanningSession:
        planner = EstimatorPlanner(estimator, task_features(tasks, history))
        schedule = planner.plan(events, tasks, windows)
        observer = None
        if log is not None:
            observer = EventLogObserver(
                log, APP_USER, date.today().isoformat(), estimator_name, source="demo"
            )
            observer.on_plan("initial", schedule)
        return cls(
            estimator_name, planner, schedule, schedule.now, observer, reclaim_min_gain=reclaim_min_gain
        )

    def close(self) -> None:
        """Release the event-log connection, if any."""
        if self.observer:
            self.observer.log.close()

    def next_task(self) -> Placement | None:
        return min(self.schedule.placements, key=lambda p: p.start, default=None)

    def mark_done(self) -> None:
        """The next task finished about as planned. The duration is *assumed*, not measured, so it is
        logged as such and never used to train a model."""
        p = self._require_next()
        self._finish(p.end - p.start, measured=False)

    def overrun(self, extra_minutes: int) -> None:
        """The next task took `extra_minutes` longer than planned: record it and replan the rest."""
        p = self._require_next()
        self._finish(p.end - p.start + extra_minutes, measured=True)

    def finish_early(self, saved_minutes: int) -> None:
        """The next task took `saved_minutes` less than planned: record it and pull the rest forward."""
        p = self._require_next()
        self._finish(max(1, p.end - p.start - saved_minutes), measured=True)

    def _finish(self, actual_minutes: int, measured: bool) -> None:
        p = self._require_next()
        end = p.start + actual_minutes
        before = self.schedule
        trigger = None
        kept: list[str] = []
        if end > p.end:
            self.schedule = self.planner.replan(before, end, Overrun(p.task_id, end - p.end))
            trigger = "overrun"
        else:
            self.schedule = complete(before, p.task_id, p.start, end)
            freed = p.end - end
            if freed >= max(1, self.reclaim_min_gain) and self.schedule.placements:  # reclaim the freed time
                self.schedule = self.planner.replan(self.schedule, end, None)
                trigger = "early"
            elif freed > 0 and self.schedule.placements:
                kept = [
                    f"Finished {freed} min early, under the {self.reclaim_min_gain}-minute threshold, "
                    "so the rest of the day stays as planned."
                ]
        self.now = end
        self.changes = self._explain(before, p, trigger) or kept
        self.replanned = trigger is not None
        if self.observer:
            self.observer.on_completion(p.task_id, p.start, end, p.end - p.start, measured)
            if trigger:
                self.observer.on_plan(trigger, self.schedule)

    def _explain(self, before: Schedule, p: Placement, trigger: str | None) -> list[str]:
        if trigger is None:
            return []
        completed = {c.task_id for c in self.schedule.completed}
        old = Schedule(
            events=before.events,
            tasks=before.tasks,
            windows=before.windows,
            placements=tuple(q for q in before.placements if q.task_id not in completed),
            dropped=before.dropped,
        )
        notes = diff_schedules(old, self.schedule)
        spills = [
            f"⚠ **{title_of(self.schedule, p.task_id)}** now runs {minutes} min into "
            f"**{event.title or event.id}**; fixed events never move"
            for task_id, event_id, minutes in collisions(self.schedule)
            if task_id == p.task_id
            for event in self.schedule.events
            if event.id == event_id
        ]
        return spills + [n for n in notes if not (spills and n.startswith("Nothing else"))]

    def _require_next(self) -> Placement:
        p = self.next_task()
        if p is None:
            raise ValueError("nothing left to do")
        return p


def features_from_history(
    hist: pd.DataFrame, user_id: object, day: int, tasks: Sequence[Task]
) -> tuple[dict[str, dict[str, float]], tuple[float, int]]:
    """History features for `tasks` planned on `day` for `user_id`, using only rows of `hist` from
    earlier days. Also returns (the user's typical actual/estimate ratio, number of rows behind it)."""
    queries = pd.DataFrame(
        {"user_id": [user_id] * len(tasks), "day": day, "task_type": [t.type for t in tasks]},
        index=[t.id for t in tasks],
    )
    feats = history_features(hist, queries)
    summary = (1.0, 0)
    if len(feats) and feats["user_n"].iloc[0] > 0:
        summary = (math.exp(float(feats["user_lr_mean"].iloc[0])), int(feats["user_n"].iloc[0]))
    return feats.to_dict(orient="index"), summary  # type: ignore[return-value]


def persona_history(
    cfg: DataConfig, user_id: int, tasks: Sequence[Task]
) -> tuple[dict[str, dict[str, float]], tuple[float, int]]:
    """History features for `tasks` as if they belonged to synthetic user `user_id`, who has
    `cfg.eval_start` days of logged history."""
    profile = next(u for u in make_users(cfg) if u.user_id == user_id)
    hist = history_frame(generate_timeline(profile, cfg.eval_start, cfg.seed), cfg.seed)
    return features_from_history(hist, user_id, cfg.eval_start, tasks)


def log_history(
    log_frame: pd.DataFrame, tasks: Sequence[Task], today: date | None = None
) -> tuple[dict[str, dict[str, float]], tuple[float, int]]:
    """History features for the app user from their own event log (`EventLog.history_frame`)."""
    when = (today or date.today()).toordinal()
    return features_from_history(log_frame[log_frame["user_id"] == APP_USER], APP_USER, when, tasks)


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
