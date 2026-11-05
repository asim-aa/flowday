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

from flowday.dataset import DataConfig, make_users
from flowday.models import Estimator, add_history_features, cluster_bootstrap_ci
from flowday.planning import FEATURE_COLUMNS, EstimatorPlanner
from flowday.simulator import (
    SyntheticDay,
    UserProfile,
    generate_timeline,
    history_frame,
    simulate_day,
    simulate_days,
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
    "replan_churn": "Tasks moved by replans / day",
    "pull_forwards": "Early-finish pull-forwards / day",
}


# extra columns of the multi-day (carry-over) runs; every row has them (zero when nothing is carried)
CARRY_METRICS = {
    "carried_in": "Tasks carried in from yesterday / day",
    "carry_days_completed": "Days of carry-over delay on tasks done / day",
}
CARRY_TABLE = [
    "carried_in",
    "tasks_completed",
    "tasks_dropped",
    "carry_days_completed",
    "deadlines_missed",
    "event_overlap_minutes",
    "overtime_minutes",
]


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
    user: UserProfile,
    cfg: DataConfig,
    models: dict[str, Estimator],
    estimators: list[str] = ESTIMATORS,
    reclaim_early: bool = False,
    reclaim_min_gain: int = 0,
    carry: bool = False,
    age_priority: bool = True,
) -> list[dict[str, Any]]:
    """Run every estimator over one user's evaluation-period days. Returns one row per (day, estimator).
    With `carry`, the days run in sequence and what a day cannot finish is carried into the next."""
    days = generate_timeline(user, cfg.n_days, cfg.seed)
    table = add_history_features(history_frame(days, cfg.seed))
    # estimator inputs only: the outcome column `actual` is deliberately not passed on
    features = table.set_index("task_id")[FEATURE_COLUMNS]

    def planner_for(name: str, day: SyntheticDay) -> EstimatorPlanner:
        day_features = features.loc[[t.id for t in day.tasks]]
        if name == "oracle":
            return EstimatorPlanner(OracleEstimator(day), day_features, refine_iters=4)
        return EstimatorPlanner(models[name], day_features)

    rows: list[dict[str, Any]] = []
    if carry:
        eval_days = days[cfg.eval_start :]
        for name in estimators:
            outcomes = simulate_days(
                eval_days,
                lambda day, name=name: planner_for(name, day),  # type: ignore[misc]
                reclaim_early=reclaim_early,
                reclaim_min_gain=reclaim_min_gain,
                age_priority=age_priority,
            )
            rows += [
                {"user_id": user.user_id, "day": d.index, "estimator": name, **o.as_row()}
                for d, o in zip(eval_days, outcomes, strict=True)
            ]
        return rows
    for day in days[cfg.eval_start :]:
        for name in estimators:
            outcome = simulate_day(
                day,
                planner_for(name, day),
                reclaim_early=reclaim_early,
                reclaim_min_gain=reclaim_min_gain,
            )
            rows.append({"user_id": user.user_id, "day": day.index, "estimator": name, **outcome.as_row()})
    return rows


_STATE: dict[str, Any] = {}


def _init_worker(
    models: dict[str, Estimator],
    cfg: DataConfig,
    users: dict[int, UserProfile],
    reclaim_early: bool,
    reclaim_min_gain: int,
    estimators: list[str],
    carry: bool,
    age_priority: bool,
) -> None:
    threadpool_limits(1)  # tiny predict() calls: OpenMP threading only adds overhead
    _STATE.update(
        models=models,
        cfg=cfg,
        users=users,
        reclaim_early=reclaim_early,
        reclaim_min_gain=reclaim_min_gain,
        estimators=estimators,
        carry=carry,
        age_priority=age_priority,
    )


def _job(user_id: int) -> list[dict[str, Any]]:
    return evaluate_user(
        _STATE["users"][user_id],
        _STATE["cfg"],
        _STATE["models"],
        _STATE["estimators"],
        reclaim_early=_STATE["reclaim_early"],
        reclaim_min_gain=_STATE["reclaim_min_gain"],
        carry=_STATE["carry"],
        age_priority=_STATE["age_priority"],
    )


def run_simulations(
    models: dict[str, Estimator],
    cfg: DataConfig,
    user_ids: list[int],
    workers: int,
    reclaim_early: bool = False,
    estimators: list[str] | None = None,
    reclaim_min_gain: int = 0,
    carry: bool = False,
    age_priority: bool = True,
) -> pd.DataFrame:
    names = list(estimators or ESTIMATORS)
    users = {u.user_id: u for u in make_users(cfg)}
    if workers <= 1:
        _init_worker(models, cfg, users, reclaim_early, reclaim_min_gain, names, carry, age_priority)
        results = [_job(u) for u in user_ids]
    else:
        with ProcessPoolExecutor(
            workers,
            initializer=_init_worker,
            initargs=(models, cfg, users, reclaim_early, reclaim_min_gain, names, carry, age_priority),
        ) as pool:
            results = list(pool.map(_job, user_ids, chunksize=2))
    return pd.DataFrame([row for rows in results for row in rows])


def summarize(
    rows: pd.DataFrame, baseline: str = "trust", metrics: list[str] | None = None
) -> dict[str, Any]:
    """Per-estimator mean per day with 95% CIs (resampling users), plus paired differences vs `baseline`."""
    out: dict[str, Any] = {
        "n_users": int(rows["user_id"].nunique()),
        "n_days": int(rows.groupby("user_id")["day"].nunique().iloc[0]),
    }
    base = rows[rows["estimator"] == baseline].set_index(["user_id", "day"])
    for name in dict.fromkeys(rows["estimator"]):  # in order of appearance
        sub = rows[rows["estimator"] == name].set_index(["user_id", "day"])
        entry: dict[str, Any] = {}
        for metric in metrics or METRICS:
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
    labels = {**METRICS, **CARRY_METRICS}
    head = "| Estimator | " + " | ".join(labels[m].replace(" / day", "") for m in metrics) + " |"
    lines = [head, "|" + "---|" * (len(metrics) + 1)]
    for name in [n for n in summary if isinstance(summary[n], dict) and metrics[0] in summary[n]]:
        cells = []
        for m in metrics:
            e = summary[name][m]
            cells.append(f"{e['mean']:.2f} [{e['ci95'][0]:.2f}, {e['ci95'][1]:.2f}]")
        lines.append(f"| {LABELS.get(name, name)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(workers: int, max_users: int | None = None, reclaim: bool = False) -> dict[str, Any]:
    """Evaluate every estimator on the test users. With `reclaim`, early finishes also trigger a
    replan (pull the rest of the day forward); results then go to `*_reclaim` files."""
    bundle = joblib.load(ARTIFACT)
    cfg: DataConfig = bundle["config"]
    test_users: list[int] = bundle["split"]["test"][:max_users]
    rows = run_simulations(bundle["estimators"], cfg, test_users, workers, reclaim_early=reclaim)
    summary = summarize(rows)
    summary["synthetic"] = True
    summary["reclaim_early"] = reclaim
    suffix = "_reclaim" if reclaim else ""
    REPORT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    rows.to_csv(DATA_DIR / f"schedule_eval_rows{suffix}.csv", index=False)
    (REPORT_DIR / f"schedule_results{suffix}.json").write_text(json.dumps(summary, indent=2))
    (REPORT_DIR / f"schedule_results{suffix}.md").write_text(markdown_table(summary) + "\n")
    return summary


def run_carry(workers: int, max_users: int | None = None, age_priority: bool = True) -> dict[str, Any]:
    """Multi-day evaluation: each test user's evaluation days run in sequence and whatever a day cannot
    finish is carried into the next. With `age_priority` a task's priority rises one step per day carried."""
    bundle = joblib.load(ARTIFACT)
    cfg: DataConfig = bundle["config"]
    test_users: list[int] = bundle["split"]["test"][:max_users]
    rows = run_simulations(
        bundle["estimators"], cfg, test_users, workers, carry=True, age_priority=age_priority
    )
    summary = summarize(rows, metrics=CARRY_TABLE)
    last = rows[rows["day"] == rows["day"].max()]
    backlog: dict[str, Any] = {}
    delay: dict[str, float] = {}
    for name in dict.fromkeys(rows["estimator"]):
        final = last[last["estimator"] == name]
        mean, lo, hi = cluster_bootstrap_ci(final["tasks_dropped"].to_numpy(), final["user_id"].to_numpy())
        backlog[name] = {"mean": mean, "ci95": [lo, hi]}
        mine = rows[rows["estimator"] == name]
        delay[name] = float(mine["carry_days_completed"].sum() / mine["tasks_completed"].sum())
    summary.update(
        synthetic=True, age_priority=age_priority, backlog_end=backlog, delay_days_per_task_done=delay
    )
    suffix = "_carry" if age_priority else "_carry_noaging"
    REPORT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    rows.to_csv(DATA_DIR / f"schedule_eval_rows{suffix}.csv", index=False)
    extra = [
        "",
        "| Estimator | Backlog on the last day (tasks waiting) | Carry-over delay per task done (days) |",
        "|---|---|---|",
    ]
    for name in backlog:
        b = backlog[name]
        interval = f"{b['mean']:.2f} [{b['ci95'][0]:.2f}, {b['ci95'][1]:.2f}]"
        extra.append(f"| {LABELS.get(name, name)} | {interval} | {delay[name]:.3f} |")
    text = markdown_table(summary, CARRY_TABLE) + "\n" + "\n".join(extra) + "\n"
    (REPORT_DIR / f"schedule_results{suffix}.json").write_text(json.dumps(summary, indent=2))
    (REPORT_DIR / f"schedule_results{suffix}.md").write_text(text)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-users", type=int, default=None, help="evaluate only the first K test users")
    ap.add_argument(
        "--mode",
        choices=["both", "no-reclaim", "reclaim", "carry"],
        default="both",
        help="no-reclaim: early finishes are not reclaimed (the headline setting); reclaim: they are; "
        "carry: run the days in sequence with unfinished tasks carried over, with and without priority aging",
    )
    args = ap.parse_args()
    if args.mode == "carry":
        for aging in (True, False):
            print(f"\n== unfinished tasks carried over, priority aging {'on' if aging else 'off'} ==")
            print(markdown_table(run_carry(args.workers, args.max_users, aging), CARRY_TABLE))
        return
    for reclaim in {"both": (False, True), "no-reclaim": (False,), "reclaim": (True,)}[args.mode]:
        print(f"\n== early finishes {'reclaimed' if reclaim else 'not reclaimed'} ==")
        print(markdown_table(run(args.workers, args.max_users, reclaim)))
    from flowday.evaluation.plots import make_figures

    make_figures()


if __name__ == "__main__":
    main()
