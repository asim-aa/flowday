"""Generate synthetic days (fixed events + flexible tasks) and task-history tables.

All output is synthetic. Times inside a day are minutes since midnight; `day` is just an index
used to order a user's history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from flowday.scheduler import Event, Interval, Task, hhmm

from .profiles import TASK_TYPES, UserProfile

SOURCE = "synthetic"
HISTORY_COLUMNS = ["source", "user_id", "day", "task_id", "task_type", "estimate", "hour", "actual"]


@dataclass(frozen=True)
class DayConfig:
    work_start: int = hhmm("09:00")
    work_end: int = hhmm("18:00")
    min_events: int = 1
    max_events: int = 3
    min_tasks: int = 4
    max_tasks: int = 8
    deadline_prob: float = 0.55
    max_estimate: int = 240


DEFAULT_CONFIG = DayConfig()


@dataclass(frozen=True)
class SyntheticDay:
    """One synthetic day for one user, plus the hidden noise that decides true durations."""

    user: UserProfile
    index: int
    events: tuple[Event, ...]
    tasks: tuple[Task, ...]
    windows: tuple[Interval, ...]
    noise: dict[str, float]

    def true_minutes(self, task_id: str, start: int) -> int:
        """How long `task_id` really takes if started at minute `start` of the day."""
        task = next(t for t in self.tasks if t.id == task_id)
        return self.user.true_minutes(task.type, task.estimated_minutes, start / 60.0, self.noise[task_id])


def _round_to(value: float, step: int, floor: int, cap: int) -> int:
    return int(min(cap, max(floor, step * round(value / step))))


def _draw_noise(rng: np.random.Generator, user: UserProfile) -> float:
    if user.tail_df is None:
        return float(rng.standard_normal())
    return float(rng.standard_t(user.tail_df) / math.sqrt(user.tail_df / (user.tail_df - 2.0)))


def _gen_events(rng: np.random.Generator, cfg: DayConfig, day_key: str) -> tuple[Event, ...]:
    events: list[Event] = []
    target = int(rng.integers(cfg.min_events, cfg.max_events + 1))
    budget = (cfg.work_end - cfg.work_start) * 0.45
    for _ in range(40):  # rejection sampling; the window is wide so this terminates quickly
        if len(events) >= target:
            break
        length = int(rng.choice([30, 45, 60, 90]))
        start = cfg.work_start + 15 * int(rng.integers(0, (cfg.work_end - cfg.work_start - length) // 15 + 1))
        candidate = Event(f"{day_key}e{len(events)}", start, start + length, "meeting")
        if any(candidate.start < e.end and e.start < candidate.end for e in events):
            continue
        if sum(e.end - e.start for e in events) + length > budget:
            continue
        events.append(candidate)
    return tuple(sorted(events, key=lambda e: e.start))


def generate_day(
    user: UserProfile, index: int, rng: np.random.Generator, cfg: DayConfig = DEFAULT_CONFIG
) -> SyntheticDay:
    """Sample one synthetic day for `user`."""
    day_key = f"u{user.user_id}d{index}"
    events = _gen_events(rng, cfg, day_key)
    names = list(TASK_TYPES)
    weights = np.array([TASK_TYPES[n].weight for n in names])
    weights = weights / weights.sum()
    window_len = cfg.work_end - cfg.work_start

    tasks: list[Task] = []
    noise: dict[str, float] = {}
    for i in range(int(rng.integers(cfg.min_tasks, cfg.max_tasks + 1))):
        ttype = str(rng.choice(names, p=weights))
        spec = TASK_TYPES[ttype]
        estimate = _round_to(
            spec.median_estimate * math.exp(spec.estimate_sigma * rng.standard_normal()),
            user.rounding,
            user.rounding,
            cfg.max_estimate,
        )
        deadline = None
        if rng.random() < cfg.deadline_prob:
            slack = int(rng.integers(0, max(1, window_len - estimate)))
            deadline = min(cfg.work_end, cfg.work_start + 15 * math.ceil((estimate + slack) / 15))
        task_id = f"{day_key}t{i}"
        tasks.append(
            Task(
                id=task_id,
                type=ttype,
                estimated_minutes=estimate,
                deadline=deadline,
                priority=int(rng.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.30, 0.20, 0.10])),
            )
        )
        noise[task_id] = _draw_noise(rng, user)
    return SyntheticDay(user, index, events, tuple(tasks), ((cfg.work_start, cfg.work_end),), noise)


def generate_timeline(
    user: UserProfile, n_days: int, seed: int, cfg: DayConfig = DEFAULT_CONFIG
) -> list[SyntheticDay]:
    """`n_days` consecutive synthetic days for one user; reproducible from (seed, user_id). With
    `user.drift_per_day` the user's bias on day d is `offset + drift_per_day * d`."""
    rng = np.random.default_rng([seed, user.user_id, 1])
    return [
        generate_day(
            replace(user, offset=user.offset + user.drift_per_day * d) if user.drift_per_day else user,
            d,
            rng,
            cfg,
        )
        for d in range(n_days)
    ]


def history_frame(days: list[SyntheticDay], seed: int, cfg: DayConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Task-level history table: what a log of past completions would look like.

    Each task is assigned a random start time within work hours (history is not produced by
    running the scheduler), and its actual duration follows the generative model at that hour.
    """
    rows = []
    for day in days:
        rng = np.random.default_rng([seed, day.user.user_id, 2, day.index])
        for task in day.tasks:
            start = int(rng.uniform(cfg.work_start, cfg.work_end - 30))
            rows.append(
                {
                    "source": SOURCE,
                    "user_id": day.user.user_id,
                    "day": day.index,
                    "task_id": task.id,
                    "task_type": task.type,
                    "estimate": task.estimated_minutes,
                    "hour": start / 60.0,
                    "actual": day.true_minutes(task.id, start),
                }
            )
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)
