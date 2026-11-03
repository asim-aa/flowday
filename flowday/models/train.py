"""Train, tune and evaluate the duration models on synthetic data.

    python -m flowday.models.train [--users 500 --days 90 --seed 0]

Users are split into train / validation / test as whole users. Hyper-parameters are chosen on
validation users; every number reported in `reports/model_metrics.json` is from test users.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from flowday.dataset import DataConfig, Dataset, build_dataset
from flowday.models import (
    Estimator,
    GBMEstimator,
    GlobalMultiplier,
    TrustEstimate,
    TypeMedianRatio,
    cluster_bootstrap_ci,
    coverage,
    mae,
    pinball,
    reliability_table,
    split_users,
)
from flowday.models.metrics import bias

GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves} for lr, leaves in itertools.product([0.05, 0.1], [8, 24])
]
QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.9)  # 0.5 and 0.8 are the headline pair; the rest feed the caution frontier
ARTIFACT_DIR = Path("artifacts")
REPORT_DIR = Path("reports")
DATA_DIR = Path("data")


def _tune(
    make: Any, train: pd.DataFrame, val: pd.DataFrame, score: Any
) -> tuple[Estimator, dict[str, Any], float]:
    best: tuple[Estimator, dict[str, Any], float] | None = None
    for params in GRID:
        model = make(**params).fit(train)
        s = score(val["actual"].to_numpy(), model.predict(val))
        if best is None or s < best[2]:
            best = (model, params, float(s))
    assert best is not None
    return best


def fit_models(
    train: pd.DataFrame, val: pd.DataFrame, seed: int
) -> tuple[dict[str, Estimator], dict[str, Any]]:
    """Fit baselines on train and tune each GBM on val. Returns (estimators, tuning log)."""
    models: dict[str, Estimator] = {
        "trust": TrustEstimate().fit(train),
        "global_multiplier": GlobalMultiplier().fit(train),
        "type_median": TypeMedianRatio().fit(train),
    }
    tuning: dict[str, Any] = {}
    model, params, score = _tune(lambda **p: GBMEstimator(None, seed, **p), train, val, mae)
    models["gbm"] = model
    tuning["gbm"] = {"params": params, "val_mae": score}
    for q in QUANTILES:
        model, params, score = _tune(
            lambda q=q, **p: GBMEstimator(q, seed, **p), train, val, lambda y, pred, q=q: pinball(y, pred, q)
        )
        models[model.name] = model
        tuning[model.name] = {"params": params, "val_pinball": score}
    return models, tuning


def quick_fit(cfg: DataConfig) -> dict[str, Estimator]:
    """Fit the estimators on every user with default hyper-parameters (no tuning, no evaluation).
    For demos that need a model fast; `run` is the real, evaluated pipeline."""
    frame = build_dataset(cfg).frame
    return {
        "trust": TrustEstimate().fit(frame),
        "global_multiplier": GlobalMultiplier().fit(frame),
        "type_median": TypeMedianRatio().fit(frame),
        "gbm": GBMEstimator(None, cfg.seed).fit(frame),
        **{f"gbm_q{q}": GBMEstimator(q / 100, cfg.seed).fit(frame) for q in (60, 70, 80, 90)},
    }


def _slice_metrics(test: pd.DataFrame, models: dict[str, Estimator], oracle: np.ndarray) -> dict[str, Any]:
    y = test["actual"].to_numpy(dtype=float)
    users = test["user_id"].to_numpy()
    out: dict[str, Any] = {"n_rows": len(test), "n_users": int(test["user_id"].nunique())}
    errs = {name: np.abs(y - m.predict(test)) for name, m in models.items()}
    errs["oracle_median"] = np.abs(y - oracle)
    for name, e in errs.items():
        m, lo, hi = cluster_bootstrap_ci(e, users)
        entry: dict[str, Any] = {"mae": m, "ci95": [lo, hi]}
        if name in models:
            entry["bias"] = bias(y, models[name].predict(test))
        out[name] = entry
    for a, b in [("gbm", "type_median"), ("gbm", "trust"), ("type_median", "trust")]:
        d, lo, hi = cluster_bootstrap_ci(errs[a] - errs[b], users)
        out[f"mae_diff_{a}_minus_{b}"] = {"mean": d, "ci95": [lo, hi]}
    return out


def _calibration(test: pd.DataFrame, models: dict[str, Estimator]) -> dict[str, Any]:
    y = test["actual"].to_numpy(dtype=float)
    users = test["user_id"].to_numpy()
    hour_bucket = pd.cut(
        test["hour"], [0, 11, 14, 17, 24], labels=["<11h", "11-14h", "14-17h", "17h+"], right=False
    )
    out: dict[str, Any] = {}
    for q in QUANTILES:
        name = f"gbm_q{round(q * 100)}"
        pred = models[name].predict(test)
        covered = (y <= pred).astype(float)
        cov, lo, hi = cluster_bootstrap_ci(covered, users)
        table = reliability_table(y, pred, pred / test["estimate"].to_numpy(dtype=float), q)
        out[name] = {
            "nominal": q,
            "coverage": cov,
            "coverage_ci95": [lo, hi],
            "pinball": pinball(y, pred, q),
            "by_task_type": {
                str(k): coverage(y[idx], pred[idx]) for k, idx in test.groupby("task_type").indices.items()
            },
            "by_hour": {
                str(k): coverage(y[idx], pred[idx])
                for k, idx in test.groupby(hour_bucket, observed=True).indices.items()
            },
            "reliability": table.round(4).to_dict(orient="records"),
        }
    # baselines are point estimates of the median ratio: show how (mis)calibrated they are as quantiles
    for name in ("trust", "global_multiplier", "type_median"):
        out[f"{name}_as_q50"] = {"coverage": coverage(y, models[name].predict(test))}
    return out


def evaluate(ds: Dataset, models: dict[str, Estimator], test_users: list[int]) -> dict[str, Any]:
    test = ds.frame[ds.frame["user_id"].isin(test_users)].reset_index(drop=True)
    oracle = np.array(
        [
            ds.timelines[u][d].user.median_minutes(t, e, h)  # the day's own profile: habits may drift
            for u, d, t, e, h in zip(
                test["user_id"], test["day"], test["task_type"], test["estimate"], test["hour"], strict=True
            )
        ]
    )
    cold = test["day"] < 7
    warm = test["day"] >= ds.config.eval_start
    return {
        "mae_minutes": {
            "all_days": _slice_metrics(test, models, oracle),
            "cold_start_first_7_days": _slice_metrics(test[cold], models, oracle[cold.to_numpy()]),
            "eval_period": _slice_metrics(test[warm], models, oracle[warm.to_numpy()]),
        },
        "quantile_calibration": _calibration(test, models),
    }


STRESS_SHIFT = (
    0.30  # log-scale: shifted users take ~35% longer relative to their estimate than trained-on users
)


def stress_test(cfg: DataConfig, models: dict[str, Estimator]) -> dict[str, Any]:
    """Score the trained models on a population that underestimates more than anything seen in
    training (every user's log-bias shifted by STRESS_SHIFT). Nothing is retrained."""
    ds = build_dataset(replace(cfg, n_users=100, seed=cfg.seed + 1000, offset_shift=STRESS_SHIFT))
    out: dict[str, Any] = {"offset_shift": STRESS_SHIFT, "n_users": 100}
    for label, rows in (
        ("cold_start_first_7_days", ds.frame[ds.frame["day"] < 7]),
        ("eval_period", ds.frame[ds.frame["day"] >= cfg.eval_start]),
    ):
        y = rows["actual"].to_numpy(dtype=float)
        maes = {
            n: mae(y, models[n].predict(rows)) for n in ("trust", "global_multiplier", "type_median", "gbm")
        }
        covs = {n: coverage(y, models[n].predict(rows)) for n in ("gbm_q50", "gbm_q80")}
        out[label] = {"mae": maes, "coverage": covs}
    return out


def markdown_summary(metrics: dict[str, Any]) -> str:
    names = ["trust", "global_multiplier", "type_median", "gbm", "gbm_q50", "gbm_q80", "oracle_median"]
    slices = ("all_days", "cold_start_first_7_days", "eval_period")
    lines = ["| Estimator | MAE, all test days | MAE, first 7 days | MAE, eval period |", "|---|---|---|---|"]
    for n in names:
        cells = []
        for sl in slices:
            e = metrics["mae_minutes"][sl][n]
            cells.append(f"{e['mae']:.1f} [{e['ci95'][0]:.1f}, {e['ci95'][1]:.1f}]")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    st = metrics["distribution_shift"]
    lines += ["", f"Shifted population (log-bias {STRESS_SHIFT:+.2f}), models not retrained:", ""]
    lines += ["| Slice | MAE trust | MAE type-median | MAE GBM | q50 coverage (0.50) | q80 coverage (0.80) |"]
    lines += ["|---|---|---|---|---|---|"]
    for sl in ("cold_start_first_7_days", "eval_period"):
        e = st[sl]
        lines.append(
            f"| {sl} | {e['mae']['trust']:.1f} | {e['mae']['type_median']:.1f} | {e['mae']['gbm']:.1f} |"
            f" {e['coverage']['gbm_q50']:.2f} | {e['coverage']['gbm_q80']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def run(
    cfg: DataConfig, out_dir: Path = ARTIFACT_DIR, report_dir: Path = REPORT_DIR, save_csv: bool = True
) -> dict[str, Any]:
    ds = build_dataset(cfg)
    split = split_users(list(ds.users), cfg.seed)
    parts = {k: ds.frame[ds.frame["user_id"].isin(v)].reset_index(drop=True) for k, v in split.items()}
    assert not (set(split["train"]) & set(split["val"])) and not (set(split["train"]) & set(split["test"]))
    models, tuning = fit_models(parts["train"], parts["val"], cfg.seed)
    metrics = {
        "data": {
            "synthetic": True,
            "n_users": cfg.n_users,
            "n_days": cfg.n_days,
            "eval_period_days": [cfg.eval_start, cfg.n_days - 1],
            "seed": cfg.seed,
            "rows": {k: len(v) for k, v in parts.items()},
            "users": {k: len(v) for k, v in split.items()},
        },
        "tuning": tuning,
        **evaluate(ds, models, split["test"]),
        "distribution_shift": stress_test(cfg, models),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"estimators": models, "config": cfg, "split": split, "synthetic": True}, out_dir / "models.joblib"
    )
    (report_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    (report_dir / "model_summary.md").write_text(markdown_summary(metrics))
    if save_csv:
        DATA_DIR.mkdir(exist_ok=True)
        ds.frame.to_csv(DATA_DIR / "synthetic_history.csv", index=False)
    return metrics


def _print_summary(metrics: dict[str, Any]) -> None:
    for slice_name, sl in metrics["mae_minutes"].items():
        print(f"\nMAE (minutes), test users, {slice_name} [{sl['n_rows']} rows]")
        for name in (
            "trust",
            "global_multiplier",
            "type_median",
            "gbm",
            "gbm_q50",
            "gbm_q80",
            "oracle_median",
        ):
            e = sl[name]
            print(f"  {name:18s} {e['mae']:6.2f}  95% CI [{e['ci95'][0]:.2f}, {e['ci95'][1]:.2f}]")
    print("\nQuantile calibration on test users (coverage should equal nominal)")
    for name, c in metrics["quantile_calibration"].items():
        if "nominal" in c:
            line = f"nominal {c['nominal']:.2f}  coverage {c['coverage']:.3f}  pinball {c['pinball']:.2f}"
            print(f"  {name}: {line}")


def _print_shift(metrics: dict[str, Any]) -> None:
    print(f"\nDistribution shift (+{STRESS_SHIFT} log-bias, no retraining)")
    for sl in ("cold_start_first_7_days", "eval_period"):
        e = metrics["distribution_shift"][sl]
        mae_s = "  ".join(f"{k} {v:.1f}" for k, v in e["mae"].items())
        cov_s = "  ".join(f"{k} {v:.3f}" for k, v in e["coverage"].items())
        print(f"  {sl}: MAE {mae_s}\n    coverage {cov_s}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--users", type=int, default=DataConfig.n_users)
    ap.add_argument("--days", type=int, default=DataConfig.n_days)
    ap.add_argument("--seed", type=int, default=DataConfig.seed)
    args = ap.parse_args()
    metrics = run(DataConfig(n_users=args.users, n_days=args.days, seed=args.seed))
    _print_summary(metrics)
    _print_shift(metrics)


if __name__ == "__main__":
    main()
