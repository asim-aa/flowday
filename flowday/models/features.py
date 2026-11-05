"""Feature engineering and the by-user data split. No dependency on the simulator.

A history table has one row per completed task with columns:
    user_id, day, task_type, estimate (minutes the user said), hour (start, hours since midnight),
    actual (minutes it really took).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T", int, str)

HISTORY_FEATURES = ["user_lr_mean", "user_n", "user_type_lr_mean", "user_type_n"]
BASE_COLUMNS = ["task_type", "estimate", "hour"]


def history_features(history: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """Per-user bias features for each query row, using only history from *strictly earlier days*.

    For a query (user_id, day, task_type) this returns, over that user's history rows with
    `history.day < queries.day`:
      user_lr_mean:       mean of log(actual / estimate) over all task types (NaN if none yet)
      user_n:             how many rows that mean is based on
      user_type_lr_mean:  the same, restricted to rows of the query's task_type
      user_type_n:        how many rows that is based on
    Rows of the same day are excluded because their outcomes are unknown when a day is planned.
    Returns a frame indexed like `queries`.
    """
    hist = history[["user_id", "day", "task_type"]].copy()
    hist["_lr"] = np.log(history["actual"] / history["estimate"])
    out = pd.DataFrame(index=queries.index)
    for prefix, keys in (("user", ["user_id"]), ("user_type", ["user_id", "task_type"])):
        daily: pd.DataFrame = hist.groupby([*keys, "day"], as_index=False).agg(
            s=("_lr", "sum"), n=("_lr", "count")
        )
        daily = daily.sort_values("day")
        daily["cs"] = daily.groupby(keys)["s"].cumsum()
        daily["cn"] = daily.groupby(keys)["n"].cumsum()
        left = queries[[*keys, "day"]].copy()
        left["_row"] = np.arange(len(left))
        merged = pd.merge_asof(
            left.sort_values("day"),
            daily[[*keys, "day", "cs", "cn"]],
            on="day",
            by=keys,
            allow_exact_matches=False,
        ).sort_values("_row")
        n = merged["cn"].fillna(0).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, merged["cs"].to_numpy() / np.where(n > 0, n, 1), np.nan)
        out[f"{prefix}_lr_mean"] = mean
        out[f"{prefix}_n"] = n
    return out[HISTORY_FEATURES]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach past-only history features to a table of users' own timelines."""
    return pd.concat([df, history_features(df, df)], axis=1)


def split_users(
    user_ids: Iterable[T], seed: int, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> dict[str, list[T]]:
    """Randomly assign whole users (int or str ids) to train / val / test. Rows of one user never
    straddle sets."""
    ids = sorted(set(user_ids))
    order = np.random.default_rng([seed, 0x5917]).permutation(len(ids))
    shuffled = [ids[i] for i in order]
    n_train = round(len(ids) * fractions[0])
    n_val = round(len(ids) * fractions[1])
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }
