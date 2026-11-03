"""Early-finish reclaiming, replan churn, and the generator knobs (all on hand-checkable days)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flowday.dataset import DataConfig, build_dataset, make_users
from flowday.eventlog import EventLog, EventLogObserver
from flowday.planning import EstimatorPlanner, task_features
from flowday.scheduler import Task, validate_schedule
from flowday.simulator import (
    SyntheticDay,
    TrustPlanner,
    UserProfile,
    generate_timeline,
    sample_users,
    simulate_day,
)


def crafted(offset: float, n: int = 2) -> SyntheticDay:
    """Deterministic day: 'email' (base factor 1.0), no noise, so true minutes = round(30 * exp(offset))."""
    user = UserProfile(
        0, offset=offset, size_slope=0.0, peak_hour=11.0, tod_scale=0.0, noise_sigma=0.0, rounding=5
    )
    tasks = tuple(Task(f"t{i}", "email", 30) for i in range(n))
    return SyntheticDay(user, 0, (), tasks, ((540, 1080),), {t.id: 0.0 for t in tasks})


def test_crafted_day_has_the_durations_we_think():
    assert crafted(0.5).true_minutes("t0", 540) == 49 and crafted(-0.5).true_minutes("t0", 540) == 18


def test_overruns_replan_and_the_churn_counts_tasks_that_moved():
    out = simulate_day(crafted(0.5), TrustPlanner())
    # t0 runs 540-589 against a plan of 540-570: t1 is pushed 570 -> 589, then t1 overruns too
    assert out.replans == 2 and out.replan_churn == 1 and out.pull_forwards == 0
    assert [(c.task_id, c.start, c.end) for c in out.schedule.completed] == [
        ("t0", 540, 589),
        ("t1", 589, 638),
    ]


def test_without_reclaiming_an_early_finish_leaves_idle_time():
    out = simulate_day(crafted(-0.5), TrustPlanner())
    assert out.idle_minutes == 12 and out.replans == 0 and out.pull_forwards == 0
    assert [(c.start, c.end) for c in out.schedule.completed] == [(540, 558), (570, 588)]


def test_reclaiming_pulls_the_next_task_forward_and_removes_the_idle_gap():
    out = simulate_day(crafted(-0.5), TrustPlanner(), reclaim_early=True)
    assert out.idle_minutes == 0 and out.pull_forwards == 1 and out.replan_churn == 1 and out.replans == 0
    assert [(c.start, c.end) for c in out.schedule.completed] == [(540, 558), (558, 576)]


def test_reclaiming_is_a_no_op_when_nothing_finishes_early_or_nothing_is_left():
    assert (
        simulate_day(crafted(0.5), TrustPlanner(), reclaim_early=True).as_row()
        == simulate_day(crafted(0.5), TrustPlanner()).as_row()
    )
    assert simulate_day(crafted(-0.5, n=1), TrustPlanner(), reclaim_early=True).pull_forwards == 0


def test_min_gain_skips_early_finishes_that_free_too_little_time():
    # each 30-minute task of crafted(-0.5) takes 18 minutes, freeing 12
    assert (
        simulate_day(crafted(-0.5), TrustPlanner(), reclaim_early=True, reclaim_min_gain=12).pull_forwards
        == 1
    )
    kept = simulate_day(crafted(-0.5), TrustPlanner(), reclaim_early=True, reclaim_min_gain=13)
    assert kept.as_row() == simulate_day(crafted(-0.5), TrustPlanner()).as_row() and kept.pull_forwards == 0


def test_an_unreachable_min_gain_is_the_same_as_not_reclaiming():
    for user in sample_users(6, 12):
        for day in generate_timeline(user, 4, 12):
            never = simulate_day(day, EstimatorPlanner(Cautious(), task_features(day.tasks)))
            gated = simulate_day(
                day,
                EstimatorPlanner(Cautious(), task_features(day.tasks)),
                reclaim_early=True,
                reclaim_min_gain=10_000,
            )
            assert gated.as_row() == never.as_row()


class Cautious:
    """Plans 1.6x the user's estimate, so most tasks finish early."""

    name, uses_hour = "cautious", False

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["estimate"].to_numpy(dtype=float) * 1.6


def test_reclaiming_cuts_idle_time_for_a_cautious_planner_and_keeps_every_invariant():
    idle: dict[bool, list[int]] = {False: [], True: []}
    for user in sample_users(12, 31):
        for day in generate_timeline(user, 6, 31):
            for reclaim in (False, True):
                planner = EstimatorPlanner(Cautious(), task_features(day.tasks))
                out = simulate_day(day, planner, reclaim_early=reclaim)
                assert validate_schedule(out.schedule) == []
                assert out.schedule.events == day.events
                assert out.tasks_completed + out.tasks_dropped == out.n_tasks
                assert (out.pull_forwards > 0) <= reclaim
                idle[reclaim].append(out.idle_minutes)
    assert np.mean(idle[True]) < np.mean(idle[False])


def test_early_replans_are_logged_with_their_own_trigger():
    log = EventLog()
    simulate_day(
        crafted(-0.5), TrustPlanner(), EventLogObserver(log, "u", "2026-01-05", "trust"), reclaim_early=True
    )
    assert log.db.execute("SELECT trigger FROM plans ORDER BY plan_id").fetchall() == [
        ("initial",),
        ("early",),
    ]


# --- generator knobs ---------------------------------------------------------------------------------


def test_knobs_are_neutral_at_their_defaults_and_scale_what_they_say():
    base = sample_users(20, 4)
    assert base == sample_users(20, 4, bias_scale=1.0, noise_scale=1.0, offset_shift=0.0)
    noisy = sample_users(20, 4, noise_scale=2.0)
    assert [n.noise_sigma for n in noisy] == [2.0 * b.noise_sigma for b in base]
    weak = sample_users(20, 4, bias_scale=0.0)
    for b, w in zip(base, weak, strict=True):
        assert (
            w.log_ratio_mean("coding", 60, 15.0) == w.offset
        )  # no systematic bias, only the personal offset
        assert b.log_ratio_mean("coding", 60, 15.0) != b.offset
    half = sample_users(1, 4, bias_scale=0.5)[0]
    full = base[0]
    expected = full.offset + 0.5 * (full.log_ratio_mean("writing", 45, 10.0) - full.offset)
    assert abs(half.log_ratio_mean("writing", 45, 10.0) - expected) < 1e-12


def test_dataset_config_carries_the_knobs_to_the_data():
    cfg = DataConfig(n_users=40, n_days=10, eval_days=3, seed=5)
    strong = build_dataset(cfg).frame
    none = build_dataset(DataConfig(n_users=40, n_days=10, eval_days=3, seed=5, bias_scale=0.0)).frame
    ratio = lambda f: float((f["actual"] / f["estimate"]).median())  # noqa: E731
    assert ratio(strong) > 1.15 and abs(ratio(none) - 1.0) < 0.08
    assert [u.bias_scale for u in make_users(DataConfig(bias_scale=0.7))][:3] == [0.7] * 3
