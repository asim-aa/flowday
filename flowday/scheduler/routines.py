"""Expand recurring routines into concrete fixed events and flexible tasks."""

from __future__ import annotations

from collections.abc import Sequence

from .models import Event, Routine, Task
from .timeutil import MINUTES_PER_DAY


def expand_routines(
    routines: Sequence[Routine], num_days: int, first_weekday: int = 0
) -> tuple[list[Event], list[Task]]:
    """Instantiate each routine on every day in `range(num_days)` whose weekday it runs on.

    Day `d` starts at minute `d * 1440` and has weekday `(first_weekday + d) % 7`.
    Fixed routines become Events with id `"<routine>@<day>"`; flexible routines become Tasks
    (type = the routine's task_type, estimate = its duration) with the same id scheme, a
    deadline of `deadline_minute` on that day if given, and a release time at the start of that day
    so an instance is never planned on an earlier day.
    """
    events: list[Event] = []
    tasks: list[Task] = []
    for day in range(num_days):
        origin = day * MINUTES_PER_DAY
        weekday = (first_weekday + day) % 7
        for r in routines:
            if weekday not in r.weekdays:
                continue
            ident = f"{r.id}@{day}"
            if r.kind == "fixed":
                assert r.start_minute is not None  # enforced by Routine
                start = origin + r.start_minute
                events.append(Event(ident, start, start + r.duration, r.title))
            else:
                deadline = None if r.deadline_minute is None else origin + r.deadline_minute
                tasks.append(
                    Task(ident, r.task_type, r.duration, deadline, r.priority, title=r.title, release=origin)
                )
    return events, tasks
