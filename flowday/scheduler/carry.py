"""Carrying unfinished tasks over to the next day."""

from __future__ import annotations

from dataclasses import replace

from .models import Schedule, Task

MAX_PRIORITY = 5


def carry_over(schedule: Schedule, age_priority: bool = True) -> tuple[Task, ...]:
    """Every task of `schedule` that was not completed, as it should enter the next day's plan.

    Dropped tasks and tasks that were planned but never done are both carried. Each comes back one
    day older (`carried + 1`), without its deadline (that day has passed) and without a reserved
    duration (tomorrow's estimator sizes it afresh; the user's own estimate is kept). With
    `age_priority` its priority also rises one step, capped at 5, so that something postponed again
    and again is not starved by newer work. That aging rule is a design choice, not a measured one.
    """
    done = schedule.completed_ids
    return tuple(
        replace(
            t,
            carried=t.carried + 1,
            priority=min(MAX_PRIORITY, t.priority + 1) if age_priority else t.priority,
            deadline=None,
            planned_minutes=None,
            release=None,
        )
        for t in schedule.tasks
        if t.id not in done
    )
