"""Retrain the estimators from an event log, closing the loop: log -> model.

    python -m flowday.models.retrain --db data/flowday.db [--source real] [--out PATH]

Only `measured` completions are used (see `eventlog`). Before the final fit on all rows, a held-out
check reports MAE against the baselines, split by user when the log has enough users and
chronologically otherwise, so a bad retrain is visible. Logs are small compared with the synthetic
training set, so the gradient-boosted models use smaller, more conservative settings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from flowday.eventlog import EventLog
from flowday.models import (
    Estimator,
    GBMEstimator,
    GlobalMultiplier,
    TrustEstimate,
    TypeMedianRatio,
    add_history_features,
    mae,
    split_users,
)

MIN_ROWS = 60  # fewer measured completions than this is not enough to fit anything useful
MIN_USERS_FOR_USER_SPLIT = 10


def _params(n_rows: int) -> dict[str, float | int]:
    return {
        "max_iter": 100,
        "learning_rate": 0.1,
        "max_leaf_nodes": 8,
        "min_samples_leaf": max(5, n_rows // 30),
    }


def _fit(frame: pd.DataFrame, seed: int) -> dict[str, Estimator]:
    params = _params(len(frame))
    return {
        "trust": TrustEstimate().fit(frame),
        "global_multiplier": GlobalMultiplier().fit(frame),
        "type_median": TypeMedianRatio().fit(frame),
        "gbm": GBMEstimator(None, seed, **params).fit(frame),
        "gbm_q80": GBMEstimator(0.8, seed, **params).fit(frame),
    }


def _prepare(history: pd.DataFrame) -> pd.DataFrame:
    if len(history) < MIN_ROWS:
        raise ValueError(
            f"only {len(history)} measured completions in the log; need at least {MIN_ROWS} to retrain"
        )
    return add_history_features(history)


def fit_from_log(history: pd.DataFrame, seed: int = 0) -> dict[str, Estimator]:
    """Fit all estimators on a log's `history_frame()` (raises if it has too few rows)."""
    return _fit(_prepare(history), seed)


def holdout_mae(history: pd.DataFrame, seed: int = 0) -> dict[str, float | str]:
    """Held-out MAE (minutes) of each estimator trained on part of the log. The split is by user when
    the log has enough users, otherwise the last 20% of days is held out."""
    frame = _prepare(history)
    users = sorted(frame["user_id"].unique())
    if len(users) >= MIN_USERS_FOR_USER_SPLIT:
        split = split_users(users, seed, (0.8, 0.0, 0.2))
        test_mask = frame["user_id"].isin(split["test"])
        how = "held-out users"
    else:
        test_mask = frame["day"] >= frame["day"].quantile(0.8)
        how = "last 20% of days"
    train, test = frame[~test_mask], frame[test_mask]
    models = _fit(train, seed)
    y = test["actual"].to_numpy(dtype=float)
    out: dict[str, float | str] = {name: mae(y, m.predict(test)) for name, m in models.items()}
    out.update(split=how, train_rows=float(len(train)), test_rows=float(len(test)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=Path("data/flowday.db"))
    ap.add_argument("--source", choices=["synthetic", "demo", "real"], default=None)
    ap.add_argument("--out", type=Path, default=Path("artifacts/models_from_log.joblib"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    with EventLog(args.db) as log:
        history = log.history_frame(args.source)
    report = holdout_mae(history, args.seed)
    print(
        f"Held-out MAE in minutes ({report.pop('split')}; {int(report.pop('train_rows'))} train rows, "
        f"{int(report.pop('test_rows'))} test rows):"
    )
    for name, value in report.items():
        print(f"  {name:18s} {float(value):6.2f}")
    models = fit_from_log(history, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted(history["source"].unique())
    joblib.dump({"estimators": models, "trained_on": {"rows": len(history), "sources": sources}}, args.out)
    print(f"\nTrained on {len(history)} measured completions (sources: {', '.join(sources)}) -> {args.out}")
    if "synthetic" in sources:
        print("Note: this log contains synthetic data; the models describe the simulator, not real people.")


if __name__ == "__main__":
    main()
