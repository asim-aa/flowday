"""End-to-end evaluation: run the scheduler in the simulator with each estimator.

    python -m flowday.evaluation.run [--workers N] [--max-users K]

Every estimator sees exactly the same synthetic days and the same hidden noise (common random
numbers), so differences between estimators are paired. Uncertainty comes from resampling whole
test users, because the days of one user are correlated. All data is synthetic.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from flowday.dataset import DataConfig
from flowday.models import Estimator, add_history_features, cluster_bootstrap_ci
from flowday.planning import FEATURE_COLUMNS, EstimatorPlanner
from flowday.simulator import (
    SyntheticDay,
    UserProfile,
    generate_timeline,
    history_frame,
    sample_users,
    simulate_day,
)

ARTIFACT = Path("artifacts/models.joblib")
REPORT_DIR = Path("reports")
DATA_DIR = Path("data")

# order matters: it fixes each estimator's colour in every figure
ESTIMATORS = ["trust", "global_multiplier", "type_median", "gbm", "gbm_q80", "oracle"]
LABELS = {
    "trust": "Trust estimate",
    "global_multiplier": "Global multiplier",
    "type_median": "Per-type median",
    "gbm": "GBM (ML)",
    "gbm_q80": "GBM q80 (ML, cautious)",
    "oracle": "Oracle (true durations)",
}
METRICS = {
    "deadlines_missed": "Deadlines missed / day",
    "replans": "Replans / day",
    "idle_minutes": "Idle minutes / day",
    "tasks_dropped": "Tasks dropped / day",
    "event_overlap_minutes": "Minutes spilling into fixed events / day",
    "overtime_minutes": "Overtime minutes / day",
    "tasks_completed": "Tasks completed / day",
}


class OracleEstimator:
    """Cheating upper bound: knows each task's true duration for the slot it is placed in."""

    name = "oracle"
    uses_hour = True

    def __init__(self, day: SyntheticDay) -> None:
        self.day = day

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(
            [self.day.true_minutes(str(i), int(h * 60)) for i, h in zip(df.index, df["hour"], strict=True)],
            dtype=float,
        )


def evaluate_user(
    user: UserProfile, cfg: DataConfig, models: dict[str, Estimator], estimators: list[str] = ESTIMATORS
) -> list[dict[str, Any]]:
    """Run every estimator over one user's evaluation-period days. Returns one row per (day, estimator)."""
    days = generate_timeline(user, cfg.n_days, cfg.seed)
    table = add_history_features(history_frame(days, cfg.seed))
    # estimator inputs only: the outcome column `actual` is deliberately not passed on
    features = table.set_index("task_id")[FEATURE_COLUMNS]
    rows: list[dict[str, Any]] = []
    for day in days[cfg.eval_start :]:
        day_features = features.loc[[t.id for t in day.tasks]]
        for name in estimators:
            if name == "oracle":
                planner = EstimatorPlanner(OracleEstimator(day), day_features, refine_iters=4)
            else:
                planner = EstimatorPlanner(models[name], day_features)
            outcome = simulate_day(day, planner)
            rows.append({"user_id": user.user_id, "day": day.index, "estimator": name, **outcome.as_row()})
    return rows


_STATE: dict[str, Any] = {}


def _init_worker(models: dict[str, Estimator], cfg: DataConfig, users: dict[int, UserProfile]) -> None:
    threadpool_limits(1)  # tiny predict() calls: OpenMP threading only adds overhead
    _STATE.update(models=models, cfg=cfg, users=users)


def _job(user_id: int) -> list[dict[str, Any]]:
    return evaluate_user(_STATE["users"][user_id], _STATE["cfg"], _STATE["models"])


def run_simulations(
    models: dict[str, Estimator], cfg: DataConfig, user_ids: list[int], workers: int
) -> pd.DataFrame:
    users = {u.user_id: u for u in sample_users(cfg.n_users, cfg.seed)}
    if workers <= 1:
        _init_worker(models, cfg, users)
        results = [_job(u) for u in user_ids]
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(models, cfg, users)) as pool:
            results = list(pool.map(_job, user_ids, chunksize=2))
    return pd.DataFrame([row for rows in results for row in rows])


def summarize(rows: pd.DataFrame, baseline: str = "trust") -> dict[str, Any]:
    """Per-estimator mean per day with 95% CIs (resampling users), plus paired differences vs `baseline`."""
    out: dict[str, Any] = {
        "n_users": int(rows["user_id"].nunique()),
        "n_days": int(rows.groupby("user_id")["day"].nunique().iloc[0]),
    }
    base = rows[rows["estimator"] == baseline].set_index(["user_id", "day"])
    for name in ESTIMATORS:
        sub = rows[rows["estimator"] == name].set_index(["user_id", "day"])
        entry: dict[str, Any] = {}
        for metric in METRICS:
            mean, lo, hi = cluster_bootstrap_ci(
                sub[metric].to_numpy(), sub.index.get_level_values(0).to_numpy()
            )
            entry[metric] = {"mean": mean, "ci95": [lo, hi]}
            if name != baseline:
                diff = (sub[metric] - base[metric].reindex(sub.index)).to_numpy()
                d, dlo, dhi = cluster_bootstrap_ci(diff, sub.index.get_level_values(0).to_numpy())
                entry[metric]["diff_vs_" + baseline] = {"mean": d, "ci95": [dlo, dhi]}
        out[name] = entry
    return out


def markdown_table(summary: dict[str, Any], metrics: list[str] | None = None) -> str:
    metrics = metrics or list(METRICS)
    head = "| Estimator | " + " | ".join(METRICS[m].replace(" / day", "") for m in metrics) + " |"
    lines = [head, "|" + "---|" * (len(metrics) + 1)]
    for name in ESTIMATORS:
        cells = []
        for m in metrics:
            e = summary[name][m]
            cells.append(f"{e['mean']:.2f} [{e['ci95'][0]:.2f}, {e['ci95'][1]:.2f}]")
        lines.append(f"| {LABELS[name]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(workers: int, max_users: int | None = None) -> dict[str, Any]:
    bundle = joblib.load(ARTIFACT)
    cfg: DataConfig = bundle["config"]
    test_users: list[int] = bundle["split"]["test"][:max_users]
    rows = run_simulations(bundle["estimators"], cfg, test_users, workers)
    summary = summarize(rows)
    summary["synthetic"] = True
    REPORT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    rows.to_csv(DATA_DIR / "schedule_eval_rows.csv", index=False)
    (REPORT_DIR / "schedule_results.json").write_text(json.dumps(summary, indent=2))
    (REPORT_DIR / "schedule_results.md").write_text(markdown_table(summary) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-users", type=int, default=None, help="evaluate only the first K test users")
    args = ap.parse_args()
    summary = run(args.workers, args.max_users)
    print(markdown_table(summary))
    from flowday.evaluation.plots import make_figures

    make_figures()


if __name__ == "__main__":
    main()
