"""Carrying unfinished tasks over to the next day."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flowday.scheduler import (
    MAX_PRIORITY,
    Task,
    build_schedule,
    carry_over,
    complete,
    daily_windows,
    hhmm,
    validate_schedule,
)
from tests.strategies import scenarios

DAY = daily_windows(1, hhmm("09:00"), hhmm("17:00"))


def test_a_dropped_task_comes_back_older_more_urgent_and_without_its_deadline():
    tasks = [
        Task("big", "coding", 480, priority=5),
        Task("small", "email", 30, deadline=hhmm("12:00"), priority=2),
    ]
    s = build_schedule([], tasks, DAY)
    assert [d.task_id for d in s.dropped] == ["small"]  # the big task takes the whole day
    carried = {t.id: t for t in carry_over(s)}
    small = carried["small"]
    assert (small.carried, small.priority, small.deadline, small.planned_minutes) == (1, 3, None, None)
    assert small.estimated_minutes == 30 and small.type == "email"  # what the user said is kept


def test_planned_but_unfinished_tasks_are_carried_too_and_finished_ones_are_not():
    tasks = [Task("a", "x", 60), Task("b", "x", 60), Task("c", "x", 60)]
    s = build_schedule([], tasks, DAY)
    first = s.placements[0]
    s = complete(s, first.task_id, first.start, first.end)
    assert {t.id for t in carry_over(s)} == {"a", "b", "c"} - {first.task_id}


def test_priority_is_capped_and_aging_can_be_switched_off():
    s = build_schedule([], [Task("top", "x", 600, priority=5), Task("low", "x", 60, priority=1)], DAY)
    assert {t.id: t.priority for t in carry_over(s)} == {"top": MAX_PRIORITY, "low": 2}
    assert {t.id: t.priority for t in carry_over(s, age_priority=False)} == {"top": 5, "low": 1}


def test_carrying_repeatedly_counts_the_days():
    t = Task("x", "x", 600)
    for day in range(1, 4):
        (t,) = carry_over(build_schedule([], [t], DAY))
        assert t.carried == day


def test_a_negative_carry_count_is_rejected():
    with pytest.raises(ValueError, match="carried"):
        Task("x", "x", 30, carried=-1)


@settings(max_examples=150, deadline=None)
@given(scenarios(), st.data())
def test_every_task_is_either_completed_or_carried_and_carrying_changes_only_what_it_says(scenario, data):
    s = build_schedule(scenario.events, scenario.tasks, scenario.windows)
    for p in list(s.placements)[: data.draw(st.integers(0, len(s.placements)))]:
        s = complete(s, p.task_id, p.start, p.end)
    carried = carry_over(s)
    assert {t.id for t in carried} | s.completed_ids == {t.id for t in s.tasks}
    assert not {t.id for t in carried} & s.completed_ids
    originals = {t.id: t for t in s.tasks}
    for t in carried:
        o = originals[t.id]
        assert (t.type, t.estimated_minutes, t.title) == (o.type, o.estimated_minutes, o.title)
        assert t.carried == o.carried + 1 and t.priority == min(MAX_PRIORITY, o.priority + 1)
        assert t.deadline is None and t.planned_minutes is None and t.release is None


@settings(max_examples=100, deadline=None)
@given(scenarios())
def test_carried_tasks_always_make_a_valid_next_day(scenario):
    s = build_schedule(scenario.events, scenario.tasks, scenario.windows)
    tomorrow = build_schedule(scenario.events, list(carry_over(s)), scenario.windows)
    assert validate_schedule(tomorrow) == []
    # without deadlines, anything that fits the empty calendar at all is placed or dropped for capacity
    assert len(tomorrow.placements) + len(tomorrow.dropped) == len(s.tasks)


# --- carrying across simulated days -------------------------------------------------------------------

import numpy as np  # noqa: E402

from flowday.simulator import (  # noqa: E402
    SyntheticDay,
    TrustPlanner,
    UserProfile,
    generate_timeline,
    sample_users,
    simulate_days,
)


def blank_day(index: int, tasks: list[Task]) -> SyntheticDay:
    """A day whose 'email' tasks take exactly their estimate (no bias, no noise)."""
    user = UserProfile(
        0, offset=0.0, size_slope=0.0, peak_hour=11.0, tod_scale=0.0, noise_sigma=0.0, rounding=5
    )
    return SyntheticDay(user, index, (), tuple(tasks), ((540, 1020),), {t.id: 0.0 for t in tasks})


def test_an_overloaded_day_carries_its_surplus_into_the_next_and_the_oldest_goes_first():
    # each day holds 480 minutes of work and gets 3 x 240 minutes
    d0 = blank_day(0, [Task(f"a{i}", "email", 240) for i in range(3)])
    d1 = blank_day(1, [Task(f"b{i}", "email", 240) for i in range(3)])
    out0, out1 = simulate_days([d0, d1], lambda day: TrustPlanner())
    assert (out0.tasks_completed, out0.tasks_dropped, out0.carried_in) == (2, 1, 0)
    assert (out1.carried_in, out1.n_tasks, out1.tasks_completed, out1.tasks_dropped) == (1, 4, 2, 2)
    done = {p.task_id for p in out1.schedule.completed}
    carried = next(t.id for t in out0.schedule.tasks if t.id in {d.task_id for d in out0.schedule.dropped})
    assert carried in done  # aged to a higher priority, so it beats the newer work
    assert out1.carry_days_completed == 1  # it waited one day


def test_without_aging_the_carried_task_gets_no_head_start():
    d0 = blank_day(0, [Task(f"a{i}", "email", 240) for i in range(3)])
    d1 = blank_day(1, [Task(f"b{i}", "email", 240) for i in range(3)])
    _, out1 = simulate_days([d0, d1], lambda day: TrustPlanner(), age_priority=False)
    assert out1.tasks_completed == 2 and out1.n_tasks == 4  # same load; which two run is now a tie-break


def test_a_carried_task_keeps_its_hidden_noise_so_it_takes_the_same_time_whenever_it_runs():
    noisy = UserProfile(
        0, offset=0.0, size_slope=0.0, peak_hour=11.0, tod_scale=0.0, noise_sigma=0.3, rounding=5
    )
    # the 30-minute task outranks the 480-minute one on the tie-break, so "big" is what gets carried
    d0 = SyntheticDay(
        noisy,
        0,
        (),
        (Task("big", "email", 480), Task("late", "email", 30)),
        ((540, 1020),),
        {"big": 0.7, "late": -0.4},
    )
    d1 = SyntheticDay(noisy, 1, (), (), ((540, 1020),), {})
    seen: list[SyntheticDay] = []

    def record(day):
        seen.append(day)
        return TrustPlanner()

    simulate_days([d0, d1], record)
    assert [t.id for t in seen[1].tasks] == ["big"] and seen[1].noise == {"big": 0.7}
    assert seen[1].true_minutes("big", 540) == d0.true_minutes("big", 540)


def test_no_task_is_ever_lost_across_days_and_the_per_day_books_balance():
    for user in sample_users(6, 41):
        days = generate_timeline(user, 8, 41)
        outcomes = simulate_days(days, lambda day: TrustPlanner())
        generated = sum(len(d.tasks) for d in days)
        completed = sum(o.tasks_completed for o in outcomes)
        backlog = outcomes[-1].tasks_dropped  # whatever the last day could not do is still waiting
        assert completed + backlog == generated
        for earlier, later in zip(outcomes, outcomes[1:], strict=False):
            assert later.carried_in == earlier.tasks_dropped
        assert all(o.tasks_completed + o.tasks_dropped == o.n_tasks for o in outcomes)
        assert np.sum([o.carry_days_completed for o in outcomes]) >= 0
