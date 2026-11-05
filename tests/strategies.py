"""Hypothesis strategies shared by the scheduler property tests."""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from flowday.scheduler import MINUTES_PER_DAY, Event, Task, daily_windows, subtract_intervals
from flowday.scheduler.timeutil import Interval


@dataclass(frozen=True)
class Scenario:
    events: list[Event]
    tasks: list[Task]
    windows: list[Interval]


@st.composite
def windows_strategy(draw: st.DrawFn) -> list[Interval]:
    n_days = draw(st.integers(1, 2))
    start = draw(st.sampled_from([0, 360, 480, 540, 600]))
    length = draw(st.sampled_from([120, 240, 360, 480, 540]))
    return daily_windows(n_days, start, start + length)


@st.composite
def events_strategy(draw: st.DrawFn, windows: list[Interval], max_per_window: int = 3) -> list[Event]:
    """Non-overlapping events, placed mostly inside the work windows."""
    events: list[Event] = []
    for w_start, w_end in windows:
        k = draw(st.integers(0, max_per_window))
        points = draw(
            st.lists(st.integers(w_start // 5, w_end // 5), min_size=2 * k, max_size=2 * k, unique=True)
        )
        points.sort()
        for j in range(k):
            events.append(Event(f"e{len(events)}", points[2 * j] * 5, points[2 * j + 1] * 5))
    return events


@st.composite
def tasks_strategy(draw: st.DrawFn, windows: list[Interval], max_tasks: int = 7) -> list[Task]:
    horizon = windows[-1][1]
    n = draw(st.integers(0, max_tasks))
    tasks = []
    for i in range(n):
        planned = draw(st.one_of(st.none(), st.integers(1, 180)))
        tasks.append(
            Task(
                id=f"t{i}",
                type=draw(st.sampled_from(["writing", "email", "coding"])),
                estimated_minutes=draw(st.integers(1, 150)),
                deadline=draw(st.one_of(st.none(), st.integers(windows[0][0], horizon))),
                priority=draw(st.integers(1, 5)),
                planned_minutes=planned,
                release=draw(st.one_of(st.none(), st.none(), st.integers(windows[0][0], horizon))),
            )
        )
    return tasks


@st.composite
def scenarios(draw: st.DrawFn, max_tasks: int = 7) -> Scenario:
    windows = draw(windows_strategy())
    return Scenario(
        events=draw(events_strategy(windows)),
        tasks=draw(tasks_strategy(windows, max_tasks)),
        windows=windows,
    )


assert MINUTES_PER_DAY == 1440


@st.composite
def planted_scenarios(draw: st.DrawFn, max_tasks: int = 7) -> Scenario:
    """A scenario with a known feasible solution that uses *all* the free time exactly.

    The free gaps (work windows minus events) are cut into pieces; each piece becomes a task whose
    length is the piece and whose deadline is its planted end (often with no slack). Demand equals
    capacity at the deadlines, so pruning rules that are off by one are exposed.
    """
    windows = draw(windows_strategy())
    events = draw(events_strategy(windows))
    free = subtract_intervals(windows, [e.interval for e in events])
    pieces: list[tuple[int, int]] = []
    for a, b in free:
        if b - a >= 2 and draw(st.booleans()):
            cut = draw(st.integers(a + 1, b - 1))
            pieces += [(a, cut), (cut, b)]
        else:
            pieces.append((a, b))
    pieces = pieces[:max_tasks]
    tasks = []
    for i, (a, b) in enumerate(pieces):
        slack = draw(st.one_of(st.just(0), st.integers(0, 40)))
        tasks.append(
            Task(
                id=f"t{i}",
                type="x",
                estimated_minutes=b - a,
                deadline=b + slack if draw(st.booleans()) else None,
                priority=draw(st.integers(1, 5)),
                release=draw(
                    st.one_of(st.none(), st.integers(max(0, a - 60), a))
                ),  # never after the planted start
            )
        )
    order = draw(st.permutations(tasks))
    return Scenario(events=events, tasks=list(order), windows=windows)
