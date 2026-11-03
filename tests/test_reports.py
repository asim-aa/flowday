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
