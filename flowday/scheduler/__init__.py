"""Deterministic scheduling core. Depends only on the standard library."""

from .carry import MAX_PRIORITY, carry_over
from .core import (
    DEFAULT_NODE_BUDGET,
    build_schedule,
    collisions,
    complete,
    earliest_fit,
    replan,
    validate_schedule,
    with_planned_minutes,
)
from .models import (
    ALL_WEEKDAYS,
    Dropped,
    DropReason,
    Event,
    Overrun,
    Placement,
    Routine,
    Schedule,
    Task,
)
from .routines import expand_routines
from .timeutil import (
    MINUTES_PER_DAY,
    Interval,
    daily_windows,
    fmt_time,
    hhmm,
    merge_intervals,
    overlap_minutes,
    subtract_intervals,
)

__all__ = [
    "ALL_WEEKDAYS",
    "DEFAULT_NODE_BUDGET",
    "MAX_PRIORITY",
    "MINUTES_PER_DAY",
    "DropReason",
    "Dropped",
    "Event",
    "Interval",
    "Overrun",
    "Placement",
    "Routine",
    "Schedule",
    "Task",
    "build_schedule",
    "carry_over",
    "collisions",
    "complete",
    "daily_windows",
    "earliest_fit",
    "expand_routines",
    "fmt_time",
    "hhmm",
    "merge_intervals",
    "overlap_minutes",
    "replan",
    "subtract_intervals",
    "validate_schedule",
    "with_planned_minutes",
]
