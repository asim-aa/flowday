import pytest

from flowday.scheduler import (
    DropReason,
    Event,
    Task,
    build_schedule,
    daily_windows,
    hhmm,
    validate_schedule,
)

DAY = daily_windows(1, hhmm("09:00"), hhmm("17:00"))


def starts(schedule):
    return {p.task_id: (p.start, p.end) for p in schedule.placements}


def test_single_task_starts_at_window_open():
    s = build_schedule([], [Task("a", "x", 60)], DAY)
    assert starts(s) == {"a": (540, 600)}
    assert validate_schedule(s) == []


def test_tasks_route_around_fixed_event():
    events = [Event("m", hhmm("09:00"), hhmm("10:00"))]
    s = build_schedule(events, [Task("a", "x", 30)], DAY)
    assert starts(s)["a"] == (600, 630)
    assert s.events == tuple(events)


def test_task_uses_gap_between_events():
    events = [Event("m1", 540, 600), Event("m2", 660, 720)]
    s = build_schedule(events, [Task("a", "x", 60), Task("b", "x", 45)], DAY)
    assert starts(s)["a"] == (600, 660)  # exactly fills the gap
    assert s.dropped == ()
    assert validate_schedule(s) == []


def test_earliest_deadline_first():
    tasks = [Task("late", "x", 60, deadline=hhmm("17:00")), Task("soon", "x", 60, deadline=hhmm("10:00"))]
    s = build_schedule([], tasks, DAY)
    assert starts(s)["soon"][0] < starts(s)["late"][0]


def test_priority_breaks_deadline_ties():
    tasks = [Task("lo", "x", 30, priority=1), Task("hi", "x", 30, priority=5)]
    s = build_schedule([], tasks, DAY)
    assert starts(s)["hi"][0] < starts(s)["lo"][0]


def test_search_finds_packing_that_greedy_edf_misses():
    # Free gaps are [0, 30) and [40, 100). Taking the 60-min task first (earliest deadline
    # tie-break by id) leaves no room for the 30-min one; the other order fits both.
    windows = [(0, 100)]
    events = [Event("e", 30, 40)]
    tasks = [Task("a", "x", 60, deadline=100), Task("b", "x", 30, deadline=100)]
    s = build_schedule(events, tasks, windows)
    assert s.dropped == ()
    assert starts(s) == {"b": (0, 30), "a": (40, 100)}


def test_overload_drops_lowest_priority_with_reason():
    tasks = [
        Task("keep", "x", 240, deadline=hhmm("13:00"), priority=5),
        Task("drop", "x", 60, deadline=hhmm("13:00"), priority=1),
    ]
    s = build_schedule([], tasks, DAY)
    assert [p.task_id for p in s.placements] == ["keep"]
    (d,) = s.dropped
    assert (d.task_id, d.reason) == ("drop", DropReason.CAPACITY)
    assert "keep" in d.detail


def test_drop_reasons_name_tasks_by_title_when_they_have_one():
    tasks = [
        Task("a", "x", 240, deadline=hhmm("13:00"), priority=5, title="Write the proposal"),
        Task("b", "x", 60, deadline=hhmm("13:00"), priority=1, title="Tidy desk"),
    ]
    (d,) = build_schedule([], tasks, DAY).dropped
    assert d.task_id == "b" and "Write the proposal" in d.detail and "(a)" not in d.detail


def test_higher_priority_wins_even_with_later_deadline():
    tasks = [
        Task("urgent_low", "x", 300, deadline=hhmm("15:00"), priority=1),
        Task("important", "x", 300, deadline=hhmm("16:00"), priority=5),
    ]
    s = build_schedule([], tasks, DAY)
    assert [p.task_id for p in s.placements] == ["important"]
    assert s.dropped[0].task_id == "urgent_low"


def test_task_longer_than_any_gap_is_dropped_alone():
    events = [Event("m", hhmm("12:00"), hhmm("13:00"))]
    s = build_schedule(events, [Task("big", "x", 300)], DAY)  # gaps are 180 and 240
    (d,) = s.dropped
    assert d.reason is DropReason.NO_FIT_ALONE
    assert "300" in d.detail and "240" in d.detail


def test_deadline_before_work_starts_is_dropped():
    s = build_schedule([], [Task("a", "x", 30, deadline=hhmm("08:00"))], DAY)
    assert s.dropped[0].reason is DropReason.NO_FIT_ALONE
    assert s.placements == ()


def test_planned_minutes_override_the_estimate():
    s = build_schedule([], [Task("a", "x", 30, planned_minutes=90)], DAY)
    assert starts(s)["a"] == (540, 630)


def test_release_time_delays_a_task_and_can_make_it_impossible():
    s = build_schedule([], [Task("a", "x", 30, release=hhmm("11:00"))], DAY)
    assert starts(s)["a"] == (hhmm("11:00"), hhmm("11:30"))
    late = build_schedule([], [Task("b", "x", 30, deadline=hhmm("10:00"), release=hhmm("11:00"))], DAY)
    (d,) = late.dropped
    assert d.reason is DropReason.NO_FIT_ALONE and "cannot start before 11:00" in d.detail
    tight = build_schedule([], [Task("c", "x", 60, deadline=hhmm("12:00"), release=hhmm("11:30"))], DAY)
    assert tight.dropped and "longest free gap" in tight.dropped[0].detail


def test_other_tasks_fill_the_time_before_a_release():
    tasks = [Task("later", "x", 60, release=hhmm("13:00")), Task("now", "x", 60)]
    s = build_schedule([], tasks, DAY)
    assert starts(s) == {"now": (540, 600), "later": (780, 840)}


def test_now_clips_free_time():
    s = build_schedule([], [Task("a", "x", 30)], DAY, now=hhmm("11:00"))
    assert starts(s)["a"][0] == hhmm("11:00")


def test_overlapping_fixed_events_are_reported_not_hidden():
    events = [Event("a", 540, 600), Event("b", 570, 630)]
    s = build_schedule(events, [], DAY)
    assert s.event_conflicts == (("a", "b"),)


def test_duplicate_task_ids_rejected():
    with pytest.raises(ValueError):
        build_schedule([], [Task("a", "x", 10), Task("a", "x", 10)], DAY)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        Event("e", 10, 10)
    with pytest.raises(ValueError):
        Task("t", "x", 0)


def test_multi_day_horizon_spills_to_next_day():
    windows = daily_windows(2, hhmm("09:00"), hhmm("10:00"))
    s = build_schedule([], [Task("a", "x", 60), Task("b", "x", 60)], windows)
    assert starts(s)["b"][0] == 1440 + 540


def test_many_tasks_stay_valid_and_fast():
    tasks = [
        Task(f"t{i}", "x", 15 + (i * 7) % 40, deadline=540 + 60 * (1 + i % 8), priority=1 + i % 5)
        for i in range(30)
    ]
    s = build_schedule([Event("m", 600, 660)], tasks, DAY)
    assert validate_schedule(s) == []
    assert len(s.placements) + len(s.dropped) == 30
