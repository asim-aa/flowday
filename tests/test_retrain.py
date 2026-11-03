"""The log -> model loop: synthetic log generation and retraining from logged completions."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pytest

from flowday.eventlog import EventLog, EventLogObserver
from flowday.eventlog.simulate import simulate_into_log
from flowday.models import retrain
from flowday.models.retrain import MIN_ROWS, fit_from_log, holdout_mae
from flowday.scheduler import Task, build_schedule
from flowday.simulator import TrustPlanner, generate_timeline, sample_users, simulate_day


@pytest.fixture(scope="module")
def many_user_log() -> EventLog:
    log = EventLog()
    simulate_into_log(log, n_users=30, n_days=30, seed=3)
    return log


def test_simulated_log_is_fully_labelled_and_consistent(many_user_log: EventLog):
    log = many_user_log
    assert log.count("users") == 30
    assert log.db.execute("SELECT DISTINCT source FROM users").fetchall() == [("synthetic",)]
    assert log.db.execute("SELECT COUNT(*) FROM completions WHERE measurement != 'measured'").fetchone() == (
        0,
    )
    frame = log.history_frame()
    assert (
        len(frame) == log.count("completions")
        and frame["user_id"].map(lambda u: str(u).startswith("synthetic-")).all()
    )


def test_retraining_needs_enough_measured_completions():
    log = EventLog()
    simulate_into_log(log, n_users=1, n_days=3, seed=1)
    assert len(log.history_frame()) < MIN_ROWS
    with pytest.raises(ValueError, match="need at least"):
        fit_from_log(log.history_frame())
    with pytest.raises(ValueError, match="need at least"):
        holdout_mae(log.history_frame())


def test_models_retrained_from_a_log_beat_the_baselines_on_held_out_users(many_user_log: EventLog):
    history = many_user_log.history_frame()
    report = holdout_mae(history)
    assert report["split"] == "held-out users"
    assert float(report["train_rows"]) + float(report["test_rows"]) == len(history)
    mae = {
        k: float(v)
        for k, v in report.items()
        if k in {"trust", "global_multiplier", "type_median", "gbm", "gbm_q80"}
    }
    assert mae["trust"] > mae["global_multiplier"] > mae["type_median"] > mae["gbm"]
    models = fit_from_log(history)
    assert set(models) == {"trust", "global_multiplier", "type_median", "gbm", "gbm_q80"}


def test_a_single_user_log_is_evaluated_chronologically():
    log = EventLog()
    user = sample_users(1, 5)[0]
    for day in generate_timeline(user, 90, 5):
        obs = EventLogObserver(
            log, "me", f"2026-{1 + day.index // 28:02d}-{1 + day.index % 28:02d}", "trust", "demo"
        )
        simulate_day(day, TrustPlanner(), obs)
    history = log.history_frame()
    assert len(history) >= MIN_ROWS
    report = holdout_mae(history)
    assert report["split"] == "last 20% of days"
    assert 0 < float(report["test_rows"]) < 0.35 * len(history)


def test_assumed_durations_never_reach_the_retrained_model():
    log = EventLog()
    log.add_user("u", "demo")
    tasks = [Task(f"t{i}", "coding", 30) for i in range(80)]
    s = build_schedule([], tasks, [(0, 10_000)])
    log.log_plan("u", "2026-01-05", s, "trust")
    for i, p in enumerate(sorted(s.placements, key=lambda p: p.start)):
        log.log_completion("u", p.task_id, p.start, p.start + 30, 30, measured=i < 40)  # half assumed
    assert len(log.history_frame()) == 40 and len(log.history_frame(measured_only=False)) == 80
    with pytest.raises(ValueError, match="need at least"):
        fit_from_log(log.history_frame())  # 40 measured rows is below the minimum, even though 80 exist


def test_cli_writes_a_model_bundle_and_warns_about_synthetic_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    db, out = tmp_path / "log.db", tmp_path / "m.joblib"
    with EventLog(db) as log:
        simulate_into_log(log, n_users=12, n_days=20, seed=2)
    monkeypatch.setattr(sys, "argv", ["retrain", "--db", str(db), "--out", str(out)])
    retrain.main()
    text = capsys.readouterr().out
    assert "Held-out MAE" in text and "synthetic data" in text
    bundle = joblib.load(out)
    assert set(bundle["estimators"]) >= {"gbm", "type_median"} and bundle["trained_on"]["sources"] == [
        "synthetic"
    ]
