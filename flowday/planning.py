"""Glue between the deterministic scheduler and a duration estimator.

The scheduler only sees `Task.planned_minutes`. This module fills that in from an `Estimator`.

Durations depend on *when* a task runs (time of day), and when a task runs depends on the
durations. We resolve the loop by iterating: size tasks assuming they start at the beginning of
the day, schedule, re-size each task for the slot it landed in, schedule again. Estimators that
ignore the hour need only one pass.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Protocol

import pandas as pd

from flowday.models import HISTORY_FEATURES, Estimator
from flowday.scheduler import (
    MINUTES_PER_DAY,
    Event,
    Interval,
    Overrun,
    Schedule,
    Task,
    build_schedule,
    replan,
    with_planned_minutes,
)


class DayLike(Protocol):
    """What `EstimatorPlanner.initial` needs from a day (satisfied by `simulator.SyntheticDay`)."""

    @property
    def events(self) -> Sequence[Event]: ...
    @property
    def tasks(self) -> Sequence[Task]: ...
    @property
    def windows(self) -> Sequence[Interval]: ...


FEATURE_COLUMNS = ["task_type", "estimate", *HISTORY_FEATURES]


def task_features(
    tasks: Sequence[Task], history: Mapping[str, Mapping[str, float]] | None = None
) -> pd.DataFrame:
    """Per-task estimator inputs (indexed by task id). `history` optionally maps a task id to its
    user-history features; tasks without an entry are treated as a new user (NaN / 0)."""
    rows = []
    for t in tasks:
        h = (history or {}).get(t.id, {})
        row = {"task_type": t.type, "estimate": t.estimated_minutes}
        row.update({f: h.get(f, 0.0 if f.endswith("_n") else float("nan")) for f in HISTORY_FEATURES})
        rows.append(row)
    return pd.DataFrame(rows, index=[t.id for t in tasks], columns=FEATURE_COLUMNS)


class EstimatorPlanner:
    """Plans and replans a day, sizing tasks with `estimator`. Implements `simulator.Planner`."""

    def __init__(self, estimator: Estimator, features: pd.DataFrame, refine_iters: int = 2) -> None:
        self.estimator = estimator
        self.features = features[FEATURE_COLUMNS] if set(FEATURE_COLUMNS) <= set(features) else features
        self.refine_iters = refine_iters

    def predict_minutes(self, tasks: Sequence[Task], hours: Mapping[str, float]) -> dict[str, int]:
        ids = [t.id for t in tasks]
        if not ids:
            return {}
        frame = self.features.loc[ids].copy()
        frame["hour"] = [hours[i] for i in ids]
        pred = self.estimator.predict(frame)
        return {i: max(1, round(float(p))) for i, p in zip(ids, pred, strict=True)}

    def _refine(
        self,
        tasks: Sequence[Task],
        hours: dict[str, float],
        place: Callable[[Mapping[str, int]], Schedule],
    ) -> Schedule:
        passes = self.refine_iters if self.estimator.uses_hour else 1
        for _ in range(passes):
            schedule = place(self.predict_minutes(tasks, hours))
            hours.update({p.task_id: (p.start % MINUTES_PER_DAY) / 60 for p in schedule.placements})
        return schedule

    def plan(self, events: Sequence[Event], tasks: Sequence[Task], windows: Sequence[Interval]) -> Schedule:
        first = (windows[0][0] % MINUTES_PER_DAY) / 60 if windows else 9.0
        return self._refine(
            tasks,
            {t.id: first for t in tasks},
            lambda planned: build_schedule(
                events, [replace(t, planned_minutes=planned[t.id]) for t in tasks], windows
            ),
        )

    def initial(self, day: DayLike) -> Schedule:
        return self.plan(day.events, day.tasks, day.windows)

    def replan(self, schedule: Schedule, now: int, overrun: Overrun) -> Schedule:
        remaining = [
            t for t in schedule.tasks if t.id not in schedule.completed_ids and t.id != overrun.task_id
        ]
        hours = {t.id: (now % MINUTES_PER_DAY) / 60 for t in remaining}
        hours.update({p.task_id: (p.start % MINUTES_PER_DAY) / 60 for p in schedule.placements})
        return self._refine(
            remaining, hours, lambda planned: replan(with_planned_minutes(schedule, planned), now, overrun)
        )
