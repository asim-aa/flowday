"""Synthetic users, days and task histories, plus a day-execution engine.

All data produced here is SYNTHETIC. See `profiles.py` for the documented bias model.
"""

from .days import (
    DEFAULT_CONFIG,
    HISTORY_COLUMNS,
    SOURCE,
    DayConfig,
    SyntheticDay,
    generate_day,
    generate_timeline,
    history_frame,
)
from .execute import DayOutcome, Observer, Planner, TrustPlanner, simulate_day
from .profiles import TASK_TYPES, TypeSpec, UserProfile, sample_users, tod_shape

__all__ = [
    "DEFAULT_CONFIG",
    "HISTORY_COLUMNS",
    "SOURCE",
    "TASK_TYPES",
    "DayConfig",
    "DayOutcome",
    "Observer",
    "Planner",
    "SyntheticDay",
    "TrustPlanner",
    "TypeSpec",
    "UserProfile",
    "generate_day",
    "generate_timeline",
    "history_frame",
    "sample_users",
    "simulate_day",
    "tod_shape",
]
