"""Plain data types for the scheduler. No ML or I/O dependencies."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .timeutil import MINUTES_PER_DAY, Interval

ALL_WEEKDAYS: frozenset[int] = frozenset(range(7))


@dataclass(frozen=True, slots=True)
class Event:
    """A fixed commitment. The scheduler never moves or shrinks it."""

    id: str
    start: int
    end: int
    title: str = ""

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError(f"event {self.id!r}: start must be before end")

    @property
    def interval(self) -> Interval:
        return (self.start, self.end)


@dataclass(frozen=True, slots=True)
class Task:
    """A flexible task. Higher `priority` means more important (1-5 by convention).

    `estimated_minutes` is what the user said. `planned_minutes` is what the scheduler
    should reserve; an estimator layer fills it in, otherwise the user's estimate is used.
    """

    id: str
    type: str
    estimated_minutes: int
    deadline: int | None = None
    priority: int = 3
    planned_minutes: int | None = None
    title: str = ""

    def __post_init__(self) -> None:
        if self.estimated_minutes <= 0:
            raise ValueError(f"task {self.id!r}: estimated_minutes must be positive")
        if self.planned_minutes is not None and self.planned_minutes <= 0:
            raise ValueError(f"task {self.id!r}: planned_minutes must be positive")

    @property
    def duration(self) -> int:
        return self.planned_minutes if self.planned_minutes is not None else self.estimated_minutes


@dataclass(frozen=True, slots=True)
class Placement:
    task_id: str
    start: int
    end: int

    @property
    def interval(self) -> Interval:
        return (self.start, self.end)


class DropReason(enum.Enum):
    NO_FIT_ALONE = "no_fit_alone"  # cannot be placed even on an otherwise empty calendar
    CAPACITY = "capacity"  # fits alone, but not alongside higher-priority tasks


@dataclass(frozen=True, slots=True)
class Dropped:
    task_id: str
    reason: DropReason
    detail: str


@dataclass(frozen=True, slots=True)
class Overrun:
    """The running task `task_id` finishes `extra_minutes` later than it was planned to."""

    task_id: str
    extra_minutes: int

    def __post_init__(self) -> None:
        if self.extra_minutes < 0:
            raise ValueError("extra_minutes must be non-negative")


@dataclass(frozen=True, slots=True)
class Routine:
    """A recurring daily pattern, expanded into fixed events or flexible tasks.

    kind="fixed":    an Event at `start_minute` (minutes since midnight) on each matching day.
    kind="flexible": a Task with an optional `deadline_minute` (minutes since midnight).
    `weekdays` uses 0=Monday .. 6=Sunday.
    """

    id: str
    title: str
    duration: int
    kind: str = "fixed"
    start_minute: int | None = None
    deadline_minute: int | None = None
    task_type: str = "routine"
    priority: int = 3
    weekdays: frozenset[int] = ALL_WEEKDAYS

    def __post_init__(self) -> None:
        if self.kind not in ("fixed", "flexible"):
            raise ValueError(f"routine {self.id!r}: kind must be 'fixed' or 'flexible'")
        if self.duration <= 0:
            raise ValueError(f"routine {self.id!r}: duration must be positive")
        if self.kind == "fixed":
            if self.start_minute is None:
                raise ValueError(f"routine {self.id!r}: fixed routines need start_minute")
            if self.start_minute + self.duration > MINUTES_PER_DAY:
                raise ValueError(f"routine {self.id!r}: must end by midnight")
        elif self.start_minute is not None:
            raise ValueError(f"routine {self.id!r}: flexible routines take deadline_minute, not start_minute")


@dataclass(frozen=True)
class Schedule:
    """The result of planning. Immutable; `replan` and `complete` return new Schedules.

    events:      fixed events, sorted by start, exactly as given.
    tasks:       every flexible task known to this schedule (placed, dropped or completed).
    windows:     work-hours windows tasks may be placed in.
    placements:  planned, not-yet-done work, sorted by start.
    completed:   work that actually happened (locked: replanning never changes it).
    dropped:     tasks that could not be scheduled, each with a reason.
    now:         earliest time anything in `placements` may start.
    event_conflicts:  pairs of fixed-event ids that overlap each other (input problem; reported, not fixed).
    search_truncated: True if the exact search hit its node budget, so drops may be sub-optimal.
    """

    events: tuple[Event, ...]
    tasks: tuple[Task, ...]
    windows: tuple[Interval, ...]
    placements: tuple[Placement, ...]
    completed: tuple[Placement, ...] = ()
    dropped: tuple[Dropped, ...] = ()
    now: int = 0
    event_conflicts: tuple[tuple[str, str], ...] = ()
    search_truncated: bool = False

    def task(self, task_id: str) -> Task:
        return next(t for t in self.tasks if t.id == task_id)

    def placement_of(self, task_id: str) -> Placement | None:
        return next((p for p in self.placements if p.task_id == task_id), None)

    @property
    def completed_ids(self) -> frozenset[str]:
        return frozenset(p.task_id for p in self.completed)
