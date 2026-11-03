"""Sensitivity analyses: how much do the headline results depend on the choices behind them?

    python -m flowday.evaluation.sensitivity [--workers N] [--only ablation,sweep,frontier]

* ablation: retrain the GBM with features blanked out, to see which ones earn their keep.
* sweep:    regenerate the synthetic world with different noise levels and bias strengths, and with
            different random seeds, retrain, and re-run the end-to-end evaluation each time.
* frontier: end-to-end results as the planning quantile moves from the median to the 90th percentile,
            with and without reclaiming early finishes.

Everything is synthetic. These analyses probe the simulator's own assumptions; they cannot tell us
anything about real users. Results go to `reports/sensitivity.json` and `reports/sensitivity.md`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from functools import partial
from typing import Any

import joblib
import numpy as np

from flowday.dataset import DataConfig, build_dataset
from flowday.evaluation.run import ARTIFACT, LABELS, REPORT_DIR, run_simulations, summarize
from flowday.models import (
    HISTORY_FEATURES,
    GBMEstimator,
    cluster_bootstrap_ci,
    split_users,
)
from flowday.models.train import evaluate, fit_models

ABLATIONS: dict[str, tuple[str, ...]] = {
    "full model": (),
    "no history features": tuple(HISTORY_FEATURES),
    "no per-user history": ("user_lr_mean", "user_n"),
    "no per-user-and-type history": ("user_type_lr_mean", "user_type_n"),
    "no start hour": ("hour",),
    "no task type": ("type_code",),
    "estimate only": ("type_code", "hour", *HISTORY_FEATURES),
}

SWEEP_USERS = 250  # smaller than the headline run so seven full retrain-and-evaluate runs stay quick
SWEEP: list[tuple[str, dict[str, Any]]] = [
    ("base, seed 0", {}),
    ("base, seed 1", {"seed": 1}),
    ("base, seed 2", {"seed": 2}),
    ("noise x0.5", {"noise_scale": 0.5}),
    ("noise x2", {"noise_scale": 2.0}),
    ("bias x0.5", {"bias_scale": 0.5}),
    ("bias x1.5", {"bias_scale": 1.5}),
]
FRONTIER = ["trust", "type_median", "gbm_q50", "gbm_q60", "gbm_q70", "gbm_q80", "gbm_q90", "oracle"]
KEY_METRICS = [
    "deadlines_missed",
    "replans",
    "idle_minutes",
    "tasks_dropped",
    "event_overlap_minutes",
    "tasks_completed",
]


def ablation() -> dict[str, Any]:
    """Held-out MAE (minutes) of the point GBM with each feature group removed, on the test users."""
    bundle = joblib.load(ARTIFACT)
    cfg: DataConfig = bundle["config"]
    split = bundle["split"]
    params = bundle["estimators"]["gbm"].params
    ds = build_dataset(cfg)
    train = ds.frame[ds.frame["user_id"].isin(split["train"])]
    test = ds.frame[ds.frame["user_id"].isin(split["test"])]
    slices = {
        "all_days": test,
        "first_7_days": test[test["day"] < 7],
        "eval_period": test[test["day"] >= cfg.eval_start],
    }
    errors: dict[str, dict[str, np.ndarray]] = {}
    for name, drop in ABLATIONS.items():
        model = GBMEstimator(None, cfg.seed, drop=drop, **params).fit(train)
        errors[name] = {
            sl: np.abs(frame["actual"].to_numpy(dtype=float) - model.predict(frame))
            for sl, frame in slices.items()
        }
    out: dict[str, Any] = {}
    for name in ABLATIONS:
        entry: dict[str, Any] = {"dropped": list(ABLATIONS[name])}
        for sl, frame in slices.items():
            users = frame["user_id"].to_numpy()
            mean, lo, hi = cluster_bootstrap_ci(errors[name][sl], users)
            d, dlo, dhi = cluster_bootstrap_ci(errors[name][sl] - errors["full model"][sl], users)
            entry[sl] = {"mae": mean, "ci95": [lo, hi], "diff_vs_full": {"mean": d, "ci95": [dlo, dhi]}}
        out[name] = entry
    return out


def run_variant(cfg: DataConfig, workers: int) -> dict[str, Any]:
    """Regenerate data under `cfg`, retrain (with validation tuning), evaluate models and the scheduler."""
    ds = build_dataset(cfg)
    split = split_users(list(ds.users), cfg.seed)
    parts = {k: ds.frame[ds.frame["user_id"].isin(v)].reset_index(drop=True) for k, v in split.items()}
    models, _ = fit_models(parts["train"], parts["val"], cfg.seed)
    model_metrics = evaluate(ds, models, split["test"])
    rows = run_simulations(models, cfg, split["test"], workers)
    mae = model_metrics["mae_minutes"]["eval_period"]
    return {
        "config": {
            "n_users": cfg.n_users,
            "seed": cfg.seed,
            "noise_scale": cfg.noise_scale,
            "bias_scale": cfg.bias_scale,
        },
        "median_actual_over_estimate": float((ds.frame["actual"] / ds.frame["estimate"]).median()),
        "mae_eval_period": {
            n: mae[n]["mae"] for n in ("trust", "global_multiplier", "type_median", "gbm", "oracle_median")
        },
        "gbm_minus_type_median_mae": mae["mae_diff_gbm_minus_type_median"],
        "coverage": {n: model_metrics["quantile_calibration"][n]["coverage"] for n in ("gbm_q50", "gbm_q80")},
        "schedule": summarize(rows),
    }


def sweep(workers: int) -> dict[str, Any]:
    base = DataConfig(n_users=SWEEP_USERS, n_days=90, eval_days=30, seed=0)
    out: dict[str, Any] = {}
    for label, overrides in SWEEP:
        print(f"  sweep: {label}", flush=True)
        out[label] = run_variant(replace(base, **overrides), workers)
    return out


def frontier(workers: int) -> dict[str, Any]:
    """End-to-end results as the planning quantile rises, with and without reclaiming early finishes."""
    bundle = joblib.load(ARTIFACT)
    cfg: DataConfig = bundle["config"]
    models = bundle["estimators"]
    missing = [n for n in FRONTIER if n not in models and n != "oracle"]
    if missing:
        raise SystemExit(f"model artifact lacks {missing}; run `make train` first")
    out: dict[str, Any] = {}
    for reclaim in (False, True):
        rows = run_simulations(models, cfg, bundle["split"]["test"], workers, reclaim, FRONTIER)
        out["reclaim" if reclaim else "no_reclaim"] = summarize(rows)
    return out


def _mean(summary: dict[str, Any], name: str, metric: str) -> float:
    return float(summary[name][metric]["mean"])


def _cell(e: dict[str, Any], digits: int = 2) -> str:
    """ "mean [low, high]" for a {mean | mae, ci95} entry."""
    mean = e["mean"] if "mean" in e else e["mae"]
    return f"{mean:.{digits}f} [{e['ci95'][0]:.{digits}f}, {e['ci95'][1]:.{digits}f}]"


def markdown(results: dict[str, Any]) -> str:
    parts: list[str] = []
    if "ablation" in results:
        parts += ["### Feature ablation (GBM point model, held-out MAE in minutes)", ""]
        parts += [
            "| Variant | All test days | First 7 days | Evaluation period | Change vs full (all days) |",
            "|---|---|---|---|---|",
        ]
        for name, e in results["ablation"].items():
            d = e["all_days"]["diff_vs_full"]
            change = (
                "" if name == "full model" else f"{d['mean']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
            )
            parts.append(
                f"| {name} | {_cell(e['all_days'], 1)} | {_cell(e['first_7_days'], 1)} | {_cell(e['eval_period'], 1)} | {change} |"
            )
        parts.append("")
    if "sweep" in results:
        sw = results["sweep"]
        parts += ["### Generator sweep and seeds (250 users each; evaluation period)", ""]
        parts += [
            "| World | median actual/estimate | MAE trust | MAE per-type median | MAE GBM | q50 / q80 coverage |",
            "|---|---|---|---|---|---|",
        ]
        for label, v in sw.items():
            m = v["mae_eval_period"]
            parts.append(
                f"| {label} | {v['median_actual_over_estimate']:.2f} | {m['trust']:.1f} | {m['type_median']:.1f} |"
                f" {m['gbm']:.1f} | {v['coverage']['gbm_q50']:.2f} / {v['coverage']['gbm_q80']:.2f} |"
            )
        parts += ["", "Per simulated day (means):", ""]
        parts += [
            "| World | missed: trust / type-median / GBM / q80 | replans: trust / GBM / q80 | spill min: trust / GBM / q80 | idle min: trust / GBM / q80 |",
            "|---|---|---|---|---|",
        ]
        for label, v in sw.items():
            s = v["schedule"]
            g = partial(_mean, s)
            parts.append(
                f"| {label} | {g('trust', 'deadlines_missed'):.2f} / {g('type_median', 'deadlines_missed'):.2f} /"
                f" {g('gbm', 'deadlines_missed'):.2f} / {g('gbm_q80', 'deadlines_missed'):.2f}"
                f" | {g('trust', 'replans'):.2f} / {g('gbm', 'replans'):.2f} / {g('gbm_q80', 'replans'):.2f}"
                f" | {g('trust', 'event_overlap_minutes'):.1f} / {g('gbm', 'event_overlap_minutes'):.1f} / {g('gbm_q80', 'event_overlap_minutes'):.1f}"
                f" | {g('trust', 'idle_minutes'):.0f} / {g('gbm', 'idle_minutes'):.0f} / {g('gbm_q80', 'idle_minutes'):.0f} |"
            )
        parts.append("")
    if "frontier" in results:
        for mode, title in (
            ("no_reclaim", "early finishes not reclaimed"),
            ("reclaim", "early finishes reclaimed"),
        ):
            summary = results["frontier"][mode]
            parts += [f"### Caution frontier, {title} (per simulated day, 95% CI)", ""]
            heads = [m.replace("_", " ") for m in KEY_METRICS]
            parts += ["| Estimator | " + " | ".join(heads) + " |", "|" + "---|" * (len(KEY_METRICS) + 1)]
            for name in FRONTIER:
                label = LABELS.get(name, name.replace("gbm_q", "GBM q"))
                parts.append(
                    f"| {label} | " + " | ".join(_cell(summary[name][m]) for m in KEY_METRICS) + " |"
                )
            parts.append("")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--only", default="ablation,sweep,frontier", help="comma-separated subset to run")
    args = ap.parse_args()
    wanted = set(args.only.split(","))
    path = REPORT_DIR / "sensitivity.json"
    results: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    results["synthetic"] = True
    if "ablation" in wanted:
        print("ablation", flush=True)
        results["ablation"] = ablation()
    if "frontier" in wanted:
        print("frontier", flush=True)
        results["frontier"] = frontier(args.workers)
    if "sweep" in wanted:
        print("sweep", flush=True)
        results["sweep"] = sweep(args.workers)
    REPORT_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    (REPORT_DIR / "sensitivity.md").write_text(markdown(results))
    print(markdown(results))
    from flowday.evaluation.plots import make_figures

    make_figures()


if __name__ == "__main__":
    main()
