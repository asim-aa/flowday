from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from flowday.dataset import Dataset
from flowday.evaluation.run import ESTIMATORS, METRICS, OracleEstimator, evaluate_user, summarize
from flowday.models import Estimator, TrustEstimate
from flowday.planning import FEATURE_COLUMNS, EstimatorPlanner, task_features
from flowday.scheduler import Overrun, Task, build_schedule, validate_schedule
from flowday.simulator import TrustPlanner, generate_timeline, sample_users, simulate_day

# --- the estimator-aware planner ---------------------------------------------------------------------


class Fixed:
    """Estimator that returns estimate * factor and records what it was shown."""

    uses_hour = False

    def __init__(self, factor: float) -> None:
        self.name = f"x{factor}"
        self.factor = factor
        self.seen_columns: set[str] = set()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self.seen_columns |= set(df.columns)
        return df["estimate"].to_numpy(dtype=float) * self.factor


def one_day():
    user = sample_users(1, 3)[0]
    return generate_timeline(user, 1, 3)[0]


def test_planner_fills_planned_minutes_from_the_estimator():
    day = one_day()
    planner = EstimatorPlanner(Fixed(2.0), task_features(day.tasks))
    s = planner.initial(day)
    for t in s.tasks:
        assert t.planned_minutes == max(1, round(2.0 * t.estimated_minutes))
    assert validate_schedule(s) == []


def test_trust_estimator_planner_matches_the_plain_scheduler_exactly():
    users = sample_users(5, 9)
    for u in users:
        for day in generate_timeline(u, 6, 9):
            via_planner = simulate_day(day, EstimatorPlanner(TrustEstimate(), task_features(day.tasks)))
            plain = simulate_day(day, TrustPlanner())
            assert via_planner.as_row() == plain.as_row()


def test_estimators_never_see_the_outcome_column():
    day = one_day()
    spy = Fixed(1.3)
    leaky = task_features(day.tasks).assign(actual=999.0)  # outcome column smuggled into the features
    simulate_day(day, EstimatorPlanner(spy, leaky))
    assert "actual" not in spy.seen_columns
    assert spy.seen_columns <= {*FEATURE_COLUMNS, "hour"}


def test_replan_resizes_only_remaining_tasks_and_keeps_completed_work():
    day = one_day()
    planner = EstimatorPlanner(Fixed(1.5), task_features(day.tasks))
    s = planner.initial(day)
    first = min(s.placements, key=lambda p: p.start)
    r = planner.replan(s, now=first.end + 20, overrun=Overrun(first.task_id, 20))
    assert [c.interval for c in r.completed] == [(first.start, first.end + 20)]
    assert validate_schedule(r) == []
    for p in r.placements:
        assert p.end - p.start == r.task(p.task_id).planned_minutes


def test_hour_dependent_estimators_trigger_a_second_pass():
    calls: list[list[float]] = []

    class HourAware(Fixed):
        uses_hour = True

        def predict(self, df: pd.DataFrame) -> np.ndarray:
            calls.append(df["hour"].tolist())
            return df["estimate"].to_numpy(dtype=float) * (1 + 0.05 * df["hour"].to_numpy(dtype=float))

    day = one_day()
    EstimatorPlanner(HourAware(1.0), task_features(day.tasks), refine_iters=2).initial(day)
    assert len(calls) == 2
    assert set(calls[0]) == {9.0}  # first guess: everything starts at the window opening
    assert len(set(calls[1])) > 1  # second pass sized each task for the slot it landed in


# --- evaluation ------------------------------------------------------------------------------------


def test_oracle_estimator_is_near_perfect(small: Dataset):
    users = sample_users(8, 21)
    missed = spill = days = 0
    for u in users:
        for day in generate_timeline(u, 8, 21):
            feats = task_features(day.tasks)
            out = simulate_day(day, EstimatorPlanner(OracleEstimator(day), feats, refine_iters=4))
            missed += out.deadlines_missed
            spill += out.event_overlap_minutes
            days += 1
    assert missed / days < 0.02 and spill / days < 1.0


def test_per_user_features_equal_the_training_table(small: Dataset):
    """evaluate_user rebuilds a user's timeline alone; its features must match the full dataset's."""
    from flowday.models import add_history_features
    from flowday.simulator import history_frame

    uid = next(iter(small.users))
    alone = add_history_features(
        history_frame(
            generate_timeline(small.users[uid], small.config.n_days, small.config.seed), small.config.seed
        )
    )
    full = small.frame[small.frame["user_id"] == uid].reset_index(drop=True)
    pd.testing.assert_frame_equal(alone, full)


def test_evaluate_user_covers_every_estimator_and_eval_day(
    small: Dataset, fitted: dict[str, Estimator], split
):
    uid = split["test"][0]
    rows = pd.DataFrame(evaluate_user(small.users[uid], small.config, fitted))
    n_days = small.config.eval_days
    assert len(rows) == n_days * len(ESTIMATORS)
    assert set(rows["estimator"]) == set(ESTIMATORS)
    assert rows["day"].min() == small.config.eval_start
    # paired design: every estimator saw the same tasks on the same days
    n_tasks = rows.pivot(index="day", columns="estimator", values="n_tasks")
    assert (n_tasks.nunique(axis=1) == 1).all()


def test_ml_beats_trusting_the_user_on_spillover_and_misses(
    small: Dataset, fitted: dict[str, Estimator], split
):
    frames = [pd.DataFrame(evaluate_user(small.users[u], small.config, fitted)) for u in split["test"][:10]]
    rows = pd.concat(frames)
    mean = rows.groupby("estimator")[["deadlines_missed", "event_overlap_minutes", "replans"]].mean()
    assert mean.loc["gbm", "event_overlap_minutes"] < mean.loc["trust", "event_overlap_minutes"]
    assert mean.loc["gbm", "deadlines_missed"] < mean.loc["trust", "deadlines_missed"]
    assert mean.loc["gbm_q80", "replans"] < mean.loc["gbm", "replans"] < mean.loc["trust", "replans"]
    # the oracle sizes tasks by iterating to a fixed point, which very rarely fails to converge
    assert mean.loc["oracle", "deadlines_missed"] < 0.02
    assert mean.loc["oracle", "deadlines_missed"] < mean.drop("oracle")["deadlines_missed"].min()


def test_summarize_means_and_paired_differences_by_hand():
    rows = []
    for user in range(6):
        for day in range(4):
            for name in ESTIMATORS:
                base = {m: 0.0 for m in METRICS}
                base["replans"] = 3.0 if name == "trust" else 1.0
                rows.append({"user_id": user, "day": day, "estimator": name, **base})
    s = summarize(pd.DataFrame(rows))
    assert s["trust"]["replans"]["mean"] == 3.0 and s["gbm"]["replans"]["mean"] == 1.0
    assert s["gbm"]["replans"]["diff_vs_trust"]["mean"] == -2.0
    lo, hi = s["gbm"]["replans"]["diff_vs_trust"]["ci95"]
    assert lo == hi == -2.0  # no variation across users -> degenerate interval
    assert "diff_vs_trust" not in s["trust"]["replans"]
    assert s["n_users"] == 6 and s["n_days"] == 4


def test_task_features_defaults_to_a_new_user():
    f = task_features([Task("a", "coding", 30)])
    assert f.loc["a", "user_n"] == 0 and np.isnan(f.loc["a", "user_lr_mean"])
    assert list(f.columns) == FEATURE_COLUMNS
    with pytest.raises(KeyError):
        EstimatorPlanner(TrustEstimate(), f).predict_minutes([replace(Task("b", "x", 5))], {"b": 9.0})


def test_planner_plan_works_without_a_simulated_day():
    tasks = [Task("a", "coding", 60, priority=5), Task("b", "email", 15)]
    s = EstimatorPlanner(Fixed(1.5), task_features(tasks)).plan([], tasks, [(540, 1080)])
    assert [p.end - p.start for p in s.placements] == [90, 22] or [p.end - p.start for p in s.placements] == [
        22,
        90,
    ]
    assert s == build_schedule(
        [], [replace(t, planned_minutes=m) for t, m in zip(tasks, (90, 22), strict=True)], [(540, 1080)]
    )
