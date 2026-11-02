"""Deterministic scheduler: priority-ordered admission + exact (budgeted) deadline search.

Problem: place non-preemptible tasks into the free gaps left by fixed events inside work
windows so every placed task meets its deadline. Even with gaps this is NP-hard in general,
so:

1. Tasks are admitted one at a time, highest priority first (ties: earlier deadline, shorter).
2. A task is admitted only if the admitted set *plus that task* can still be sequenced with
   every deadline met. That check is a depth-first search over task orders (earliest-deadline
   first by default) with memoisation, a per-deadline capacity bound and a node budget.
3. A task that cannot be admitted is dropped, with a reason.

With enough budget the check is exact, so "everything fits" implies nothing is dropped.
If the budget is hit the result is still valid (never misses a deadline) but may drop more
than necessary; `Schedule.search_truncated` says so.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace

from .models import Dropped, DropReason, Event, Overrun, Placement, Schedule, Task
from .timeutil import (
    Interval,
    fmt_time,
    merge_intervals,
    overlap_minutes,
    subtract_intervals,
)

DEFAULT_NODE_BUDGET = 20_000
_INF = 10**9


class _BudgetExceeded(Exception):
    pass


def earliest_fit(free: Sequence[Interval], after: int, duration: int, deadline: int | None) -> int | None:
    """Earliest start >= `after` such that [start, start+duration) lies in one free interval
    and ends by `deadline`; None if there is no such start."""
    limit = _INF if deadline is None else deadline
    for a, b in free:
        start = max(a, after)
        if start + duration <= b:
            return start if start + duration <= limit else None
        if start >= limit:
            return None
    return None


def _dl(task: Task) -> int:
    return _INF if task.deadline is None else task.deadline


def _capacity(free: Sequence[Interval], start: int, end: int) -> int:
    return sum(overlap_minutes(iv, (start, end)) for iv in free)


def _sequence(
    tasks: Sequence[Task], free: Sequence[Interval], now: int, budget: int
) -> tuple[list[Placement] | None, bool]:
    """Find an order for `tasks` in which each is placed at its earliest fit after the previous
    one and meets its deadline. Returns (placements | None, budget_exceeded)."""
    order = sorted(tasks, key=lambda t: (_dl(t), -t.priority, t.id))
    n = len(order)
    dur = [t.duration for t in order]
    dl = [t.deadline for t in order]
    full = (1 << n) - 1
    failed: set[tuple[int, int]] = set()
    nodes = 0
    chosen: list[Placement] = []

    def viable(mask: int, t: int) -> bool:
        """Cheap necessary conditions on the tasks still to place."""
        remaining = [i for i in range(n) if not mask >> i & 1]
        for i in remaining:
            if earliest_fit(free, t, dur[i], dl[i]) is None:
                return False
        demand = 0
        for i in remaining:  # `order` is sorted by deadline, so demand accumulates in deadline order
            demand += dur[i]
            deadline = dl[i]
            if deadline is not None and demand > _capacity(free, t, deadline):
                return False
        return True

    def dfs(mask: int, t: int) -> bool:
        nonlocal nodes
        if mask == full:
            return True
        if (mask, t) in failed:
            return False
        nodes += 1
        if nodes > budget:
            raise _BudgetExceeded
        if viable(mask, t):
            for i in range(n):
                if mask >> i & 1:
                    continue
                start = earliest_fit(free, t, dur[i], dl[i])
                if start is None:
                    continue
                chosen.append(Placement(order[i].id, start, start + dur[i]))
                if dfs(mask | 1 << i, start + dur[i]):
                    return True
                chosen.pop()
        failed.add((mask, t))
        return False

    try:
        found = dfs(0, now)
    except _BudgetExceeded:
        return None, True
    return (list(chosen) if found else None), False


def _longest_gap(free: Sequence[Interval], start: int, end: int | None) -> int:
    hi = _INF if end is None else end
    return max((max(0, min(b, hi) - max(a, start)) for a, b in free), default=0)


def _plan(
    events: Sequence[Event],
    tasks: Sequence[Task],
    windows: Sequence[Interval],
    locked: Iterable[Interval],
    now: int,
    budget: int,
) -> tuple[list[Placement], list[Dropped], bool]:
    """Place `tasks` around `events`, `locked` intervals and outside `windows`, from `now`."""
    blocked = [e.interval for e in events] + list(locked)
    free = [(max(a, now), b) for a, b in subtract_intervals(windows, blocked) if b > now]

    admitted: list[Task] = []
    placements: list[Placement] = []
    dropped: list[Dropped] = []
    truncated = False

    for task in sorted(tasks, key=lambda t: (-t.priority, _dl(t), t.duration, t.id)):
        if earliest_fit(free, now, task.duration, task.deadline) is None:
            dropped.append(Dropped(task.id, DropReason.NO_FIT_ALONE, _no_fit_detail(task, free, now)))
            continue
        result, over = _sequence([*admitted, task], free, now, budget)
        truncated = truncated or over
        if result is None:
            rivals = sorted(a.id for a in admitted if _dl(a) <= _dl(task))
            detail = "no room alongside higher-priority tasks" + (f" ({', '.join(rivals)})" if rivals else "")
            if over:
                detail += "; search budget exhausted, so a better packing may exist"
            dropped.append(Dropped(task.id, DropReason.CAPACITY, detail))
            continue
        admitted.append(task)
        placements = result

    placements.sort(key=lambda p: (p.start, p.task_id))
    return placements, dropped, truncated


def _no_fit_detail(task: Task, free: Sequence[Interval], now: int) -> str:
    if task.deadline is not None and task.deadline <= now:
        return f"deadline {fmt_time(task.deadline)} has already passed"
    gap = _longest_gap(free, now, task.deadline)
    where = "" if task.deadline is None else f" before {fmt_time(task.deadline)}"
    return f"needs {task.duration} min but the longest free gap{where} is {gap} min"


def _event_conflicts(events: Sequence[Event]) -> tuple[tuple[str, str], ...]:
    ordered = sorted(events, key=lambda e: (e.start, e.end, e.id))
    return tuple((a.id, b.id) for i, a in enumerate(ordered) for b in ordered[i + 1 :] if b.start < a.end)


def build_schedule(
    events: Sequence[Event],
    tasks: Sequence[Task],
    windows: Sequence[Interval],
    now: int | None = None,
    node_budget: int = DEFAULT_NODE_BUDGET,
) -> Schedule:
    """Plan `tasks` around fixed `events` within `windows`, starting no earlier than `now`."""
    _require_unique(task.id for task in tasks)
    merged = merge_intervals(windows)
    if now is None:
        now = merged[0][0] if merged else 0
    placements, dropped, truncated = _plan(events, tasks, merged, (), now, node_budget)
    return Schedule(
        events=tuple(sorted(events, key=lambda e: (e.start, e.end, e.id))),
        tasks=tuple(tasks),
        windows=tuple(merged),
        placements=tuple(placements),
        dropped=tuple(dropped),
        now=now,
        event_conflicts=_event_conflicts(events),
        search_truncated=truncated,
    )


def complete(schedule: Schedule, task_id: str, start: int, end: int) -> Schedule:
    """Record that `task_id` actually ran [start, end). It becomes locked, completed work.

    The interval may not overlap work that is still planned; call `replan` with an
    `Overrun` instead when a task ran past its slot.
    """
    if task_id in schedule.completed_ids:
        raise ValueError(f"task {task_id!r} is already completed")
    if not any(t.id == task_id for t in schedule.tasks):
        raise KeyError(task_id)
    if start >= end:
        raise ValueError("completed interval must have start < end")
    for p in schedule.placements:
        if p.task_id != task_id and overlap_minutes((start, end), p.interval):
            raise ValueError(f"completion of {task_id!r} overlaps planned task {p.task_id!r}; replan instead")
    return replace(
        schedule,
        placements=tuple(p for p in schedule.placements if p.task_id != task_id),
        completed=tuple(sorted((*schedule.completed, Placement(task_id, start, end)), key=lambda p: p.start)),
        dropped=tuple(d for d in schedule.dropped if d.task_id != task_id),
    )


def with_planned_minutes(schedule: Schedule, planned: Mapping[str, int]) -> Schedule:
    """Copy of `schedule` whose not-yet-completed tasks use the given planned durations.

    Placements are left as they were, so the result is only a valid input to `replan`.
    """
    done = schedule.completed_ids
    tasks = tuple(
        replace(t, planned_minutes=planned[t.id]) if t.id in planned and t.id not in done else t
        for t in schedule.tasks
    )
    return replace(schedule, tasks=tasks)


def replan(
    schedule: Schedule,
    now: int,
    overrun: Overrun | None = None,
    node_budget: int = DEFAULT_NODE_BUDGET,
) -> Schedule:
    """Re-place all remaining flexible tasks from `now`.

    Fixed events and completed work are never moved. If `overrun` is given, that task is
    first recorded as completed over [its planned start, its planned end + extra]; nothing
    else is placed before that new end. Tasks that were planned earlier than `now` but never
    completed are treated as not done and are re-placed too, as are previously dropped tasks.
    """
    if now < schedule.now:
        raise ValueError(f"cannot replan into the past (now={now} < {schedule.now})")
    completed = list(schedule.completed)
    floor = now
    if overrun is not None:
        planned = schedule.placement_of(overrun.task_id)
        if planned is None:
            raise ValueError(f"overrun task {overrun.task_id!r} is not currently planned")
        actual = Placement(planned.task_id, planned.start, planned.end + overrun.extra_minutes)
        completed.append(actual)
        floor = max(now, actual.end)
    done = {p.task_id for p in completed}
    remaining = [t for t in schedule.tasks if t.id not in done]
    placements, dropped, truncated = _plan(
        schedule.events, remaining, schedule.windows, [p.interval for p in completed], floor, node_budget
    )
    return replace(
        schedule,
        placements=tuple(placements),
        completed=tuple(sorted(completed, key=lambda p: p.start)),
        dropped=tuple(dropped),
        now=floor,
        search_truncated=truncated,
    )


def collisions(schedule: Schedule) -> list[tuple[str, str, int]]:
    """(task_id, event_id, minutes) for completed work that ran into a fixed event.

    Planned work never overlaps events; completed work can, when a task overran into one.
    """
    return [
        (p.task_id, e.id, m)
        for p in schedule.completed
        for e in schedule.events
        if (m := overlap_minutes(p.interval, e.interval))
    ]


def validate_schedule(schedule: Schedule) -> list[str]:
    """Return a list of invariant violations (empty means the schedule is valid)."""
    problems: list[str] = []
    tasks = {t.id: t for t in schedule.tasks}
    placed = [p.task_id for p in schedule.placements]
    done = [p.task_id for p in schedule.completed]
    drop = [d.task_id for d in schedule.dropped]
    everything = placed + done + drop
    if sorted(everything) != sorted(tasks):
        problems.append("every task must be exactly one of placed, completed or dropped")

    # (kind, label, interval). Events may overlap each other (reported in event_conflicts) and
    # completed work may have hit an event (reality); anything else must be disjoint.
    items = [("planned", f"task {p.task_id}", p.interval) for p in schedule.placements]
    items += [("event", f"event {e.id}", e.interval) for e in schedule.events]
    items += [("done", f"completed {p.task_id}", p.interval) for p in schedule.completed]
    for i, (kind_a, name_a, iv_a) in enumerate(items):
        for kind_b, name_b, iv_b in items[i + 1 :]:
            if {kind_a, kind_b} in ({"event"}, {"event", "done"}):
                continue
            if overlap_minutes(iv_a, iv_b):
                problems.append(f"{name_a} overlaps {name_b}")
    return problems


def _require_unique(ids: Iterable[str]) -> None:
    seen: set[str] = set()
    for i in ids:
        if i in seen:
            raise ValueError(f"duplicate task id {i!r}")
        seen.add(i)
