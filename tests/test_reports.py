"""Training entry point, evaluation CLI pieces and figures, on a tiny synthetic dataset."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

import flowday.evaluation.run as run_mod
from flowday.dataset import DataConfig
from flowday.evaluation.plots import make_figures
from flowday.evaluation.run import ESTIMATORS, markdown_table, run_simulations, summarize
from flowday.models.train import _print_shift, _print_summary
from flowday.models.train import run as train_run
from flowday.simulator import sample_users

TINY = DataConfig(n_users=30, n_days=30, eval_days=10, seed=2)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("trained")
    cwd = Path.cwd()
    import os

    os.chdir(root)  # train writes data/ relative to the cwd
    try:
        metrics = train_run(TINY, root / "artifacts", root / "reports")
    finally:
        os.chdir(cwd)
    return root, metrics


def test_train_writes_artifacts_with_disjoint_user_splits(trained):
    root, metrics = trained
    bundle = joblib.load(root / "artifacts" / "models.joblib")
    split = bundle["split"]
    assert not set(split["train"]) & set(split["test"]) and not set(split["val"]) & set(split["test"])
    assert not set(split["train"]) & set(split["val"])
    assert bundle["synthetic"] is True and metrics["data"]["synthetic"] is True
    assert set(bundle["estimators"]) >= {
        "trust",
        "global_multiplier",
        "type_median",
        "gbm",
        "gbm_q50",
        "gbm_q80",
    }
    saved = json.loads((root / "reports" / "model_metrics.json").read_text())
    assert set(saved["mae_minutes"]) == {"all_days", "cold_start_first_7_days", "eval_period"}
    assert set(saved["distribution_shift"]) >= {"cold_start_first_7_days", "eval_period", "offset_shift"}
    assert "Shifted population" in (root / "reports" / "model_summary.md").read_text()
    assert set(pd.read_csv(root / "data" / "synthetic_history.csv")["source"]) == {"synthetic"}


def test_console_summaries_print_every_estimator(trained, capsys):
    _, metrics = trained
    _print_summary(metrics)
    _print_shift(metrics)
    out = capsys.readouterr().out
    for name in ("trust", "global_multiplier", "type_median", "gbm", "oracle_median", "Distribution shift"):
        assert name in out


def test_parallel_and_serial_simulation_give_identical_results(trained):
    root, _ = trained
    bundle = joblib.load(root / "artifacts" / "models.joblib")
    ids = bundle["split"]["test"][:3]
    serial = run_simulations(bundle["estimators"], bundle["config"], ids, workers=1)
    parallel = run_simulations(bundle["estimators"], bundle["config"], ids, workers=2)
    key = ["user_id", "day", "estimator"]
    pd.testing.assert_frame_equal(
        serial.sort_values(key).reset_index(drop=True), parallel.sort_values(key).reset_index(drop=True)
    )
    assert set(serial["estimator"]) == set(ESTIMATORS)


def test_evaluation_run_writes_reports_and_figures(trained, monkeypatch: pytest.MonkeyPatch):
    root, _ = trained
    monkeypatch.setattr(run_mod, "ARTIFACT", root / "artifacts" / "models.joblib")
    monkeypatch.setattr(run_mod, "REPORT_DIR", root / "reports")
    monkeypatch.setattr(run_mod, "DATA_DIR", root / "data")
    summary = run_mod.run(workers=1, max_users=3)
    assert summary["synthetic"] is True and summary["n_users"] == 3
    results = json.loads((root / "reports" / "schedule_results.json").read_text())
    table = (root / "reports" / "schedule_results.md").read_text()
    assert results["trust"]["replans"]["mean"] > results["oracle"]["replans"]["mean"]
    assert table.startswith("| Estimator") and table.count("\n") == len(ESTIMATORS) + 2 - 1 + 1
    made = make_figures(root / "reports")
    assert len(made) == 3
    for path in made:
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 10_000, path


def test_markdown_table_and_summary_shape():
    rows = pd.DataFrame(
        [
            {"user_id": u, "day": d, "estimator": e, **{m: float(u + d) for m in run_mod.METRICS}}
            for u in range(4)
            for d in range(3)
            for e in ESTIMATORS
        ]
    )
    s = summarize(rows)
    t = markdown_table(s, ["replans", "idle_minutes"])
    assert t.splitlines()[0] == "| Estimator | Replans | Idle minutes |"
    assert len(t.splitlines()) == len(ESTIMATORS) + 2
    assert sample_users(1, 0)  # keep import used


# --- sensitivity analyses ----------------------------------------------------------------------------------

import flowday.evaluation.sensitivity as sens  # noqa: E402
from flowday.models import HISTORY_FEATURES, GBMEstimator  # noqa: E402


def test_gbm_drop_blanks_the_named_features(parts):
    train, test = parts["train"], parts["test"].head(300)
    full = GBMEstimator(None, 1, max_iter=30).fit(train)
    no_hour = GBMEstimator(None, 1, drop=("hour",), max_iter=30).fit(train)
    assert not (full.predict(test) == no_hour.predict(test)).all()
    only_estimate = GBMEstimator(None, 1, drop=("type_code", "hour", *HISTORY_FEATURES), max_iter=30).fit(
        train
    )
    a, b = test.iloc[[0]].copy(), test.iloc[[1]].copy()
    b["estimate"] = a["estimate"].iloc[0]
    b["task_type"], b["hour"] = "coding" if a["task_type"].iloc[0] != "coding" else "email", 16.5
    assert only_estimate.predict(a)[0] == only_estimate.predict(b)[0]  # nothing but the estimate matters
    with pytest.raises(ValueError, match="can only drop"):
        GBMEstimator(None, 1, drop=("log_estimate",))
    with pytest.raises(ValueError, match="can only drop"):
        GBMEstimator(None, 1, drop=("no_such_feature",))


@pytest.fixture(scope="module")
def sens_inputs(trained, tmp_path_factory):
    root, _ = trained
    mp = pytest.MonkeyPatch()
    mp.setattr(sens, "ARTIFACT", root / "artifacts" / "models.joblib")
    yield root
    mp.undo()


def test_ablation_reports_every_variant_against_the_full_model(sens_inputs):
    result = sens.ablation()
    assert list(result) == list(sens.ABLATIONS)
    for entry in result.values():
        assert set(entry) >= {"dropped", "all_days", "first_7_days", "eval_period"}
        assert entry["all_days"]["ci95"][0] <= entry["all_days"]["mae"] <= entry["all_days"]["ci95"][1]
    assert result["full model"]["all_days"]["diff_vs_full"]["mean"] == 0.0
    assert result["estimate only"]["dropped"] == ["type_code", "hour", *HISTORY_FEATURES]


def test_frontier_runs_both_modes_over_every_planning_quantile(sens_inputs):
    result = sens.frontier(workers=1)
    assert set(result) == {"no_reclaim", "reclaim"}
    for summary in result.values():
        assert set(sens.FRONTIER) <= set(summary)
        assert (
            summary["oracle"]["deadlines_missed"]["mean"]
            <= summary["trust"]["deadlines_missed"]["mean"] + 0.05
        )
    assert result["no_reclaim"]["trust"]["pull_forwards"]["mean"] == 0.0
    assert result["reclaim"]["trust"]["pull_forwards"]["mean"] > 0.0


def test_reclaim_threshold_trades_pull_forwards_for_idle_time(sens_inputs):
    result = sens.reclaim_threshold(workers=1)
    assert list(result) == ["never", *map(str, sens.RECLAIM_THRESHOLDS)]
    for name in sens.RECLAIM_ESTIMATORS:
        pulled = [result[k][name]["pull_forwards"]["mean"] for k in result]
        assert pulled[0] == 0.0 and pulled[1] > pulled[-1] >= 0.0
        assert result["1"][name]["idle_minutes"]["mean"] <= result["never"][name]["idle_minutes"]["mean"]


def test_drift_detection_flags_drifting_users_far_more_often(sens_inputs, monkeypatch):
    monkeypatch.setattr(sens, "DRIFT_SD", 0.03)  # strong enough to see in the tiny test world
    result = sens.drift_detection()
    stable, drifting = result["stable habits"], result["drifting habits"]
    assert stable["checked"] > 50 and stable["flagged"] <= 0.15 * stable["checked"]
    assert drifting["flagged"] > 3 * max(1, stable["flagged"])


def test_run_variant_applies_the_generator_knobs_and_reports_everything():
    cfg = DataConfig(n_users=30, n_days=30, eval_days=10, seed=4, noise_scale=0.5, bias_scale=1.5)
    v = sens.run_variant(cfg, workers=1)
    assert v["config"] == {"n_users": 30, "seed": 4, "noise_scale": 0.5, "bias_scale": 1.5}
    assert set(v["mae_eval_period"]) == {"trust", "global_multiplier", "type_median", "gbm", "oracle_median"}
    assert set(v["coverage"]) == {"gbm_q50", "gbm_q80"} and v["schedule"]["n_users"] == 6
    assert v["median_actual_over_estimate"] > 1.0


def test_markdown_and_figures_cover_all_sections(sens_inputs, tmp_path: Path):
    results = {
        "synthetic": True,
        "ablation": sens.ablation(),
        "sweep": {"world": sens.run_variant(DataConfig(n_users=30, n_days=30, eval_days=10, seed=5), 1)},
        "frontier": sens.frontier(workers=1),
        "reclaim": sens.reclaim_threshold(workers=1),
        "drift": sens.drift_detection(),
    }
    text = sens.markdown(results)
    for heading in (
        "Feature ablation",
        "Generator sweep",
        "Caution frontier, early finishes not reclaimed",
        "Caution frontier, early finishes reclaimed",
        "Reclaim threshold, GBM q80",
        "Drift check",
    ):
        assert heading in text
    assert "| full model |" in text and "| world |" in text
    (tmp_path / "sensitivity.json").write_text(json.dumps(results))
    made = make_figures(tmp_path)
    assert [p.name for p in made] == ["fig4_caution_frontier.png"]
    assert made[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and made[0].stat().st_size > 10_000
