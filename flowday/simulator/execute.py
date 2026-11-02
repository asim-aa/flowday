"""Execute a synthetic day: tasks take their true duration, overruns trigger replanning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from flowday.scheduler import (
    Overrun,
    Placement,
    Schedule,
    build_schedule,
    collisions,
    complete,
    replan,
    subtract_intervals,
)

from .days import SyntheticDay


class Planner(Protocol):
    """Whatever turns a day into a Schedule and repairs it after an overrun."""

    def initial(self, day: SyntheticDay) -> Schedule: ...

    def replan(self, schedule: Schedule, now: int, overrun: Overrun) -> Schedule: ...


class TrustPlanner:
    """Plan with the user's own estimates (the baseline every estimator is compared to)."""

    def initial(self, day: SyntheticDay) -> Schedule:
        return build_schedule(day.events, day.tasks, day.windows)

    def replan(self, schedule: Schedule, now: int, overrun: Overrun) -> Schedule:
        return replan(schedule, now, overrun)


@dataclass(frozen=True)
class DayOutcome:
    """What happened when the day was lived. Minutes are integers.

    deadlines_missed:      completed tasks that finished after their deadline.
    replans:               replans triggered by a task finishing later than planned.
    idle_minutes:          unoccupied work-hours minutes before the last task ended (slack that
                           over-long plans leave unused; early finishes are not reclaimed).
    tasks_dropped:         tasks the scheduler could not fit (never started).
    event_overlap_minutes: minutes of fixed events spent still working on an overrunning task.
    overtime_minutes:      work done after the end of the work window.
    """

    n_tasks: int
    tasks_completed: int
    deadlines_missed: int
    replans: int
    idle_minutes: int
    tasks_dropped: int
    event_overlap_minutes: int
    overtime_minutes: int
    schedule: Schedule

    def as_row(self) -> dict[str, int]:
        return {
            "n_tasks": self.n_tasks,
            "tasks_completed": self.tasks_completed,
            "deadlines_missed": self.deadlines_missed,
            "replans": self.replans,
            "idle_minutes": self.idle_minutes,
            "tasks_dropped": self.tasks_dropped,
            "event_overlap_minutes": self.event_overlap_minutes,
            "overtime_minutes": self.overtime_minutes,
        }


def simulate_day(day: SyntheticDay, planner: Planner) -> DayOutcome:
    """Run `day` with `planner`.

    Tasks run back to back at their planned start. A task takes its true duration (which
    depends on when it actually starts). If it finishes after its planned end, the planner is
    asked to replan from that moment; if it finishes early or on time, the plan is kept and the
    next task waits for its planned start (early finishes are not reclaimed).
    """
    schedule = planner.initial(day)
    replans = 0
    while schedule.placements:
        nxt = min(schedule.placements, key=lambda p: p.start)
        end = nxt.start + day.true_minutes(nxt.task_id, nxt.start)
        if end > nxt.end:
            schedule = planner.replan(schedule, end, Overrun(nxt.task_id, end - nxt.end))
            replans += 1
        else:
            schedule = complete(schedule, nxt.task_id, nxt.start, end)
    return _outcome(day, schedule, replans)


def _outcome(day: SyntheticDay, schedule: Schedule, replans: int) -> DayOutcome:
    deadlines = {t.id: t.deadline for t in day.tasks}
    done = list(schedule.completed)
    missed = sum(1 for p in done if (d := deadlines[p.task_id]) is not None and p.end > d)
    window_end = day.windows[0][1]
    return DayOutcome(
        n_tasks=len(day.tasks),
        tasks_completed=len(done),
        deadlines_missed=missed,
        replans=replans,
        idle_minutes=_idle_minutes(day, done),
        tasks_dropped=len(schedule.dropped),
        event_overlap_minutes=sum(m for _, _, m in collisions(schedule)),
        overtime_minutes=sum(max(0, p.end - max(p.start, window_end)) for p in done),
        schedule=schedule,
    )


def _idle_minutes(day: SyntheticDay, done: list[Placement]) -> int:
    if not done:
        return 0
    window_start, window_end = day.windows[0]
    span = [(window_start, min(window_end, max(p.end for p in done)))]
    busy = [e.interval for e in day.events] + [p.interval for p in done]
    return sum(b - a for a, b in subtract_intervals(span, busy))
