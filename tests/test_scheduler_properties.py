"""Property-based tests for the scheduler and replanner (hypothesis)."""

from __future__ import annotations

import itertools

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from flowday.scheduler import (
    DropReason,
    Event,
    Overrun,
    Schedule,
    Task,
    build_schedule,
    complete,
    replan,
    validate_schedule,
)

from .strategies import Scenario, scenarios

PROPS = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def run(sc: Scenario) -> Schedule:
    return build_schedule(sc.events, sc.tasks, sc.windows)


# --- an independent feasibility oracle: minute grid + brute force over every task order ----------


def brute_force_feasible(sc: Scenario) -> bool:
    horizon = sc.windows[-1][1]
    free = [False] * (horizon + 1)
    for a, b in sc.windows:
        for m in range(a, b):
            free[m] = True
    for e in sc.events:
        for m in range(e.start, min(e.end, horizon)):
            free[m] = False
    run_len = [0] * (horizon + 2)  # consecutive free minutes starting at m
    for m in range(horizon - 1, -1, -1):
        run_len[m] = run_len[m + 1] + 1 if free[m] else 0

    def earliest(after: int, task: Task) -> int | None:
        for s in range(after, horizon):
            if run_len[s] >= task.duration:
                return s if task.deadline is None or s + task.duration <= task.deadline else None
        return None

    start = sc.windows[0][0]
    for order in itertools.permutations(sc.tasks):
        t = start
        for task in order:
            s = earliest(t, task)
            if s is None:
                break
            t = s + task.duration
        else:
            return True
    return not sc.tasks


# --- scheduling invariants ------------------------------------------------------------------------


@PROPS
@given(scenarios(max_tasks=8))
def test_schedule_is_always_valid(sc):
    """No overlaps, nothing outside work hours, nothing past its deadline, everything accounted for."""
    s = run(sc)
    assert validate_schedule(s) == []


@PROPS
@given(scenarios(max_tasks=8))
def test_fixed_events_never_move(sc):
    s = run(sc)
    assert list(s.events) == sorted(sc.events, key=lambda e: (e.start, e.end, e.id))
    assert not s.event_conflicts  # the strategy generates non-overlapping events


@PROPS
@given(scenarios(max_tasks=5))
def test_nothing_is_dropped_when_a_feasible_schedule_exists(sc):
    s = run(sc)
    assert (not s.dropped) == brute_force_feasible(sc)
    assert not s.search_truncated


@PROPS
@given(scenarios(max_tasks=8))
def test_drop_reasons_are_truthful(sc):
    s = run(sc)
    for d in s.dropped:
        task = s.task(d.task_id)
        alone = build_schedule(sc.events, [task], sc.windows)
        if d.reason is DropReason.NO_FIT_ALONE:
            assert alone.dropped, "claimed it cannot fit alone, but it does"
        else:
            assert not alone.dropped, "claimed capacity conflict, but it cannot fit alone anyway"
        assert d.detail


@PROPS
@given(scenarios(max_tasks=8))
def test_scheduling_is_deterministic(sc):
    assert run(sc) == run(sc)


@PROPS
@given(scenarios(max_tasks=6), st.data())
def test_removing_a_task_never_causes_a_new_drop(sc, data):
    if not sc.tasks:
        return
    s = run(sc)
    victim = data.draw(st.sampled_from(sc.tasks))
    smaller = build_schedule(sc.events, [t for t in sc.tasks if t.id != victim.id], sc.windows)
    assert {d.task_id for d in smaller.dropped} <= {d.task_id for d in s.dropped}


# --- replanning invariants ------------------------------------------------------------------------


@st.composite
def replan_cases(draw):
    sc = draw(scenarios(max_tasks=7))
    s = run(sc)
    ordered = sorted(s.placements, key=lambda p: p.start)
    n_done = draw(st.integers(0, len(ordered)))
    for p in ordered[:n_done]:
        s = complete(s, p.task_id, p.start, p.end)
    rest = sorted(s.placements, key=lambda p: p.start)
    overrun = None
    horizon_end = max([s.now] + [c.end for c in s.completed])
    if rest and draw(st.booleans()):
        target = draw(st.sampled_from(rest))
        overrun = Overrun(target.task_id, draw(st.integers(0, 120)))
        horizon_end = max(horizon_end, target.end + overrun.extra_minutes)
    # `now` may fall inside completed work (e.g. completions logged ahead of the clock)
    now = draw(st.integers(s.now, horizon_end + 60))
    return sc, s, now, overrun


@PROPS
@given(replan_cases())
def test_replan_never_touches_fixed_events_or_completed_work(case):
    sc, before, now, overrun = case
    after = replan(before, now, overrun)
    assert validate_schedule(after) == []
    assert after.events == before.events
    previously_done = {c.task_id: c for c in before.completed}
    now_done = {c.task_id: c for c in after.completed}
    for task_id, c in previously_done.items():
        assert now_done[task_id] == c
    extra = set(now_done) - set(previously_done)
    assert extra == ({overrun.task_id} if overrun else set())


@PROPS
@given(replan_cases())
def test_replan_places_nothing_before_now_or_the_overrun_end(case):
    sc, before, now, overrun = case
    after = replan(before, now, overrun)
    floor = now
    if overrun:
        old = before.placement_of(overrun.task_id)
        assert old is not None
        done = next(c for c in after.completed if c.task_id == overrun.task_id)
        assert done.interval == (old.start, old.end + overrun.extra_minutes)
        floor = max(now, done.end)
    assert all(p.start >= floor for p in after.placements)


@PROPS
@given(scenarios(max_tasks=6), st.lists(st.tuples(st.integers(0, 90), st.booleans()), min_size=1, max_size=8))
def test_a_whole_day_of_overruns_keeps_every_invariant(sc, noise):
    """Run the schedule to the end, overrunning at arbitrary points, replanning each time."""
    s = run(sc)
    fixed_events: tuple[Event, ...] = s.events
    for extra, is_overrun in itertools.cycle(noise):
        if not s.placements:
            break
        first = min(s.placements, key=lambda p: p.start)
        if is_overrun and extra:
            s = replan(s, now=first.end, overrun=Overrun(first.task_id, extra))
        else:
            s = complete(s, first.task_id, first.start, first.end)
        assert validate_schedule(s) == []
        assert s.events == fixed_events
    ids = {t.id for t in s.tasks}
    assert {c.task_id for c in s.completed} | {d.task_id for d in s.dropped} == ids


def test_oracle_agrees_on_a_known_tight_case():
    sc = Scenario(
        events=[Event("e", 30, 40)],
        tasks=[Task("a", "x", 60, deadline=100), Task("b", "x", 30, deadline=100)],
        windows=[(0, 100)],
    )
    assert brute_force_feasible(sc) and not run(sc).dropped
    infeasible = Scenario(sc.events, [Task("a", "x", 61, deadline=100), sc.tasks[1]], sc.windows)
    assert not brute_force_feasible(infeasible) and run(infeasible).dropped
