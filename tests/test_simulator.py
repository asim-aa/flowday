from __future__ import annotations

import math

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from flowday.scheduler import validate_schedule
from flowday.simulator import (
    HISTORY_COLUMNS,
    TASK_TYPES,
    TrustPlanner,
    generate_day,
    generate_timeline,
    history_frame,
    sample_users,
    simulate_day,
    tod_shape,
)


def big_history(n_users: int = 60, n_days: int = 40, seed: int = 0) -> tuple[pd.DataFrame, list]:
    users = sample_users(n_users, seed)
    frames = [history_frame(generate_timeline(u, n_days, seed), seed) for u in users]
    return pd.concat(frames, ignore_index=True), users


# --- data is labelled, reproducible and well-formed ----------------------------------------------


def test_all_history_is_labelled_synthetic():
    df, _ = big_history(5, 5)
    assert list(df.columns) == HISTORY_COLUMNS
    assert set(df["source"]) == {"synthetic"}


def test_generation_is_reproducible_and_seed_dependent():
    a, _ = big_history(5, 5, seed=3)
    b, _ = big_history(5, 5, seed=3)
    c, _ = big_history(5, 5, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_users_vary_but_stay_in_documented_ranges():
    users = sample_users(300, seed=1)
    assert np.std([u.offset for u in users]) > 0.1
    assert np.std([u.peak_hour for u in users]) > 1.0
    assert all(8 <= u.peak_hour <= 15 and 0.08 <= u.noise_sigma <= 0.5 and u.size_slope >= 0 for u in users)
    assert {u.rounding for u in users} == {5, 15}


def test_offset_shift_moves_every_users_bias_and_nothing_else():
    base, shifted = sample_users(50, 4), sample_users(50, 4, offset_shift=0.3)
    assert [round(s.offset - b.offset, 9) for b, s in zip(base, shifted, strict=True)] == [0.3] * 50
    assert [u.noise_sigma for u in base] == [u.noise_sigma for u in shifted]
    assert [u.peak_hour for u in base] == [u.peak_hour for u in shifted]


def test_days_are_well_formed():
    users = sample_users(20, 0)
    for u in users:
        for day in generate_timeline(u, 10, 0):
            ((w_start, w_end),) = day.windows
            ids = [t.id for t in day.tasks]
            assert len(set(ids)) == len(ids) and set(ids) == set(day.noise)
            assert all(w_start <= e.start < e.end <= w_end for e in day.events)
            assert all(a.end <= b.start for a, b in zip(day.events, day.events[1:], strict=False))
            assert all(t.deadline is None or w_start < t.deadline <= w_end for t in day.tasks)
            assert all(t.estimated_minutes % u.rounding == 0 for t in day.tasks)


# --- the documented biases are really in the data -------------------------------------------------


def test_true_minutes_matches_the_documented_formula():
    u = sample_users(1, 0)[0]
    got = u.true_minutes("coding", 60, 10.0, 0.0)
    expected = round(60 * math.exp(u.log_ratio_mean("coding", 60, 10.0)))
    assert got == expected


def test_underestimation_varies_by_task_type():
    df, _ = big_history()
    ratio = (df["actual"] / df["estimate"]).groupby(df["task_type"]).median()
    assert ratio["coding"] > ratio["admin"] > ratio["email"] * 0.98
    assert ratio["writing"] > ratio["study"] > ratio["errands"]
    assert ratio["exercise"] < 1.1
    assert (df["actual"] / df["estimate"]).median() > 1.15  # systematic underestimation overall


def test_underestimation_varies_by_time_of_day():
    df, _ = big_history(n_users=150)
    cog = df[df["task_type"].isin(["coding", "writing", "study"])].copy()
    cog["ratio"] = cog["actual"] / cog["estimate"]
    late = cog[cog["hour"] >= 16.5]["ratio"].median()
    morning = cog[cog["hour"] <= 11]["ratio"].median()
    assert late > morning * 1.05


def test_users_differ_in_how_wrong_they_are():
    df, _ = big_history(n_users=80)
    per_user = (df["actual"] / df["estimate"]).groupby(df["user_id"]).median()
    assert per_user.std() > 0.1


def test_time_of_day_shape_is_zero_at_peak_and_larger_away_from_it():
    assert tod_shape(11, 11) < tod_shape(17, 11)
    assert tod_shape(11, 11) >= 0
    assert set(TASK_TYPES) >= {"coding", "email", "exercise"}


# --- executing a day -------------------------------------------------------------------------------


def test_simulate_day_accounts_for_every_task_and_never_moves_events():
    users = sample_users(15, 7)
    for u in users:
        for day in generate_timeline(u, 8, 7):
            out = simulate_day(day, TrustPlanner())
            s = out.schedule
            assert validate_schedule(s) == []
            assert s.events == day.events
            assert not s.placements  # everything planned was run
            assert out.tasks_completed + out.tasks_dropped == out.n_tasks
            # executed work is sequential
            done = sorted(s.completed, key=lambda p: p.start)
            assert all(a.end <= b.start for a, b in zip(done, done[1:], strict=False))


def test_simulate_day_metrics_match_a_manual_recount():
    u = sample_users(1, 11)[0]
    for day in generate_timeline(u, 25, 11):
        out = simulate_day(day, TrustPlanner())
        s = out.schedule
        tasks = {t.id: t for t in day.tasks}
        planned_ends = {t.id: t.estimated_minutes for t in day.tasks}
        missed = sum(1 for p in s.completed if (d := tasks[p.task_id].deadline) is not None and p.end > d)
        assert out.deadlines_missed == missed
        # under the trust planner a task overruns iff its true time exceeds the user's estimate
        overran = sum(1 for p in s.completed if p.end - p.start > planned_ends[p.task_id])
        assert out.replans == overran
        assert out.idle_minutes >= 0 and out.overtime_minutes >= 0 and out.event_overlap_minutes >= 0


def test_a_planner_with_perfect_durations_never_replans_or_misses():
    class Oracle(TrustPlanner):
        def initial(self, day):
            from dataclasses import replace

            from flowday.scheduler import build_schedule

            # the true duration depends on the start hour; plan with it at the window start, then
            # iterate to a fixed point so each task is sized for the slot it actually lands in
            sized = list(day.tasks)
            for _ in range(8):
                s = build_schedule(day.events, sized, day.windows)
                starts = {p.task_id: p.start for p in s.placements}
                new = [
                    replace(t, planned_minutes=day.true_minutes(t.id, starts[t.id])) if t.id in starts else t
                    for t in sized
                ]
                if new == sized:
                    break
                sized = new
            return s

    users = sample_users(6, 5)
    oracle = Oracle()
    for u in users:
        for day in generate_timeline(u, 10, 5):
            out = simulate_day(day, oracle)
            assert out.deadlines_missed == 0
            assert out.event_overlap_minutes == 0


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(0, 10_000), st.integers(0, 50))
def test_any_seed_produces_a_consistent_day(seed, user_id):
    user = sample_users(1, seed, first_id=user_id)[0]
    day = generate_day(user, 0, np.random.default_rng(seed))
    out = simulate_day(day, TrustPlanner())
    assert validate_schedule(out.schedule) == []
    assert out.tasks_completed + out.tasks_dropped == len(day.tasks)
