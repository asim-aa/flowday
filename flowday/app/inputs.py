"""Turn the app's editable tables into scheduler inputs, collecting readable errors."""

from __future__ import annotations

import re

import pandas as pd

from flowday.scheduler import ALL_WEEKDAYS, Event, Routine, Task, fmt_time, hhmm


def _clock(value: object) -> int | None:
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value).strip()
    if not text:
        return None
    try:
        h, m = text.split(":")
        if not (0 <= int(h) < 24 and 0 <= int(m) < 60):
            raise ValueError
    except ValueError:
        raise ValueError(f"{text!r} is not a time like 09:30") from None
    return hhmm(text)


def parse_events(df: pd.DataFrame) -> tuple[list[Event], list[str]]:
    events: list[Event] = []
    errors: list[str] = []
    for i, row in df.iterrows():
        if row.isna().all():
            continue
        try:
            start, end = _clock(row["start"]), _clock(row["end"])
            if start is None or end is None:
                raise ValueError("start and end are required")
            events.append(Event(f"e{len(events) + 1}", start, end, str(row["title"] or "")))
        except ValueError as exc:
            errors.append(f"Event row {int(str(i)) + 1}: {exc}")
    for a, b in zip(
        sorted(events, key=lambda e: e.start), sorted(events, key=lambda e: e.start)[1:], strict=False
    ):
        if b.start < a.end:
            first = f"{a.title or a.id} ({fmt_time(a.start)}-{fmt_time(a.end)})"
            errors.append(f"Events overlap: {first} and {b.title or b.id}")
    return events, errors


def parse_tasks(df: pd.DataFrame) -> tuple[list[Task], list[str]]:
    tasks: list[Task] = []
    errors: list[str] = []
    for i, row in df.iterrows():
        if row.isna().all():
            continue
        try:
            estimate = int(row["estimate"])
            priority = int(row["priority"]) if not pd.isna(row["priority"]) else 3
            if not 1 <= priority <= 5:
                raise ValueError("priority must be 1-5 (5 = most important)")
            tasks.append(
                Task(
                    id=f"t{len(tasks) + 1}",
                    type=str(row["type"]),
                    estimated_minutes=estimate,
                    deadline=_clock(row["deadline"]),
                    priority=priority,
                    title=str(row["title"] or f"Task {len(tasks) + 1}"),
                )
            )
        except (ValueError, TypeError) as exc:
            errors.append(f"Task row {int(str(i)) + 1}: {exc}")
    return tasks, errors


DAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def parse_weekdays(text: object) -> frozenset[int]:
    """ "daily", "weekdays", "weekends", or a list like "Mon, Wed, Fri" -> weekday numbers (0 = Monday)."""
    t = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text).strip().lower()
    if t in ("", "daily", "every day", "all"):
        return ALL_WEEKDAYS
    if t == "weekdays":
        return frozenset(range(5))
    if t == "weekends":
        return frozenset({5, 6})
    days: set[int] = set()
    for token in re.split(r"[,\s]+", t):
        if token:
            if token[:3] not in DAY_NAMES:
                raise ValueError(f"unknown day {token!r}")
            days.add(DAY_NAMES.index(token[:3]))
    return frozenset(days)


def parse_routines(df: pd.DataFrame) -> tuple[list[Routine], list[str]]:
    """Rows of (title, kind, time, minutes, type, priority, days) -> routines.

    For a fixed routine `time` is its start (required); for a flexible one it is an optional deadline.
    """
    routines: list[Routine] = []
    errors: list[str] = []
    for i, row in df.iterrows():
        if row.isna().all():
            continue
        try:
            kind = str(row["kind"]).strip().lower()
            when = _clock(row["time"])
            if kind == "fixed" and when is None:
                raise ValueError("a fixed routine needs a start time")
            priority = int(row["priority"]) if not pd.isna(row["priority"]) else 3
            routines.append(
                Routine(
                    id=f"r{len(routines) + 1}",
                    title=str(row["title"] or f"Routine {len(routines) + 1}"),
                    duration=int(row["minutes"]),
                    kind=kind,
                    start_minute=when if kind == "fixed" else None,
                    deadline_minute=when if kind == "flexible" else None,
                    task_type=str(row["type"]) if not pd.isna(row["type"]) else "routine",
                    priority=priority,
                    weekdays=parse_weekdays(row["days"]),
                )
            )
        except (ValueError, TypeError) as exc:
            errors.append(f"Routine row {int(str(i)) + 1}: {exc}")
    return routines, errors
