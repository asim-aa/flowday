"""Time helpers. All scheduler times are integer minutes from an arbitrary origin (day 0, 00:00)."""

from __future__ import annotations

from collections.abc import Iterable

MINUTES_PER_DAY = 1440

Interval = tuple[int, int]  # half-open [start, end)


def hhmm(text: str) -> int:
    """Parse "09:30" into minutes since midnight."""
    h, m = text.split(":")
    return int(h) * 60 + int(m)


def fmt_time(minute: int) -> str:
    """Format absolute minutes as "09:30", or "+1d 09:30" past the first day."""
    day, rem = divmod(minute, MINUTES_PER_DAY)
    stamp = f"{rem // 60:02d}:{rem % 60:02d}"
    return stamp if day == 0 else f"+{day}d {stamp}"


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Sort and merge overlapping or touching intervals; drops empty ones."""
    out: list[Interval] = []
    for start, end in sorted(i for i in intervals if i[0] < i[1]):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def subtract_intervals(base: Iterable[Interval], blocked: Iterable[Interval]) -> list[Interval]:
    """Return the parts of `base` not covered by `blocked`, sorted and disjoint."""
    holes = merge_intervals(blocked)
    out: list[Interval] = []
    for start, end in merge_intervals(base):
        cursor = start
        for h_start, h_end in holes:
            if h_end <= cursor:
                continue
            if h_start >= end:
                break
            if h_start > cursor:
                out.append((cursor, h_start))
            cursor = max(cursor, h_end)
        if cursor < end:
            out.append((cursor, end))
    return out


def overlap_minutes(a: Interval, b: Interval) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def daily_windows(num_days: int, start: int, end: int) -> list[Interval]:
    """Work-hours window repeated on each of `num_days` days; start/end are minutes since midnight."""
    if not 0 <= start < end <= MINUTES_PER_DAY:
        raise ValueError(f"work hours must satisfy 0 <= start < end <= 1440, got {start}, {end}")
    return [(d * MINUTES_PER_DAY + start, d * MINUTES_PER_DAY + end) for d in range(num_days)]
