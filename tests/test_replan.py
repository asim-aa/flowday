import pytest

from flowday.scheduler import (
    DropReason,
    Event,
    Overrun,
    Task,
    build_schedule,
    collisions,
    complete,
    daily_windows,
    hhmm,
    replan,
    validate_schedule,
    with_planned_minutes,
)

DAY = daily_windows(1, hhmm("09:00"), hhmm("17:00"))


def starts(schedule):
    return {p.task_id: (p.start, p.end) for p in schedule.placements}


def three_tasks():
    return [
        Task("a", "x", 60, deadline=hhmm("12:00")),
        Task("b", "x", 60, deadline=hhmm("15:00")),
        Task("c", "x", 60),
    ]


def test_overrun_pushes_remaining_work_later():
    s = build_schedule([], three_tasks(), DAY)
    assert starts(s) == {"a": (540, 600), "b": (600, 660), "c": (660, 720)}
    r = replan(s, now=600, overrun=Overrun("a", 30))
    assert r.completed[0].interval == (540, 630)  # a ran 30 min over
    assert starts(r) == {"b": (630, 690), "c": (690, 750)}
    assert validate_schedule(r) == []


def test_replan_leaves_events_and_completed_work_untouched():
    events = [Event("m", hhmm("11:00"), hhmm("12:00"))]
    s = complete(build_schedule(events, three_tasks(), DAY), "a", 540, 600)
    r = replan(s, now=600, overrun=Overrun("b", 20))
    assert r.events == s.events
    assert r.completed[:1] == s.completed
    assert [p.task_id for p in r.completed] == ["a", "b"]


def test_overrun_can_force_a_drop_with_reason():
    tasks = [Task("a", "x", 60), Task("b", "x", 60, deadline=hhmm("10:30"), priority=1)]
    s = build_schedule([], tasks, DAY)
    assert s.dropped == ()
    r = replan(s, now=600, overrun=Overrun("a", 45))
    (d,) = r.dropped
    assert d.task_id == "b"
    assert d.reason in (DropReason.NO_FIT_ALONE, DropReason.CAPACITY)


def test_overrun_into_event_is_recorded_as_collision_not_moved():
    events = [Event("m", hhmm("10:00"), hhmm("11:00"))]
    s = build_schedule(events, [Task("a", "x", 60), Task("b", "x", 30)], DAY)
    assert starts(s) == {"a": (540, 600), "b": (660, 690)}
    r = replan(s, now=hhmm("10:00"), overrun=Overrun("a", 45))
    assert r.events == s.events
    assert collisions(r) == [("a", "m", 45)]
    assert starts(r) == {"b": (660, 690)}  # waits for the event to end; the event did not move
    assert validate_schedule(r) == []


def test_stale_unfinished_placements_are_replaced_after_now():
    s = build_schedule([], three_tasks(), DAY)
    r = replan(s, now=hhmm("13:00"))  # nothing was completed; the morning is gone
    assert starts(r) == {"b": (780, 840), "c": (840, 900)}
    (d,) = r.dropped  # a's 12:00 deadline has passed
    assert d.task_id == "a" and "already passed" in d.detail


def test_previously_dropped_tasks_get_another_chance():
    s = build_schedule([], [Task("big", "x", 300, deadline=hhmm("12:00"))], DAY)
    assert s.dropped
    r = replan(s, now=s.now)
    assert r.dropped  # still cannot fit, and is still reported
    ok = replan(with_planned_minutes(s, {"big": 120}), now=s.now)
    assert not ok.dropped and starts(ok)["big"] == (540, 660)


def test_cannot_replan_into_the_past_or_overrun_unplanned_task():
    s = replan(build_schedule([], three_tasks(), DAY), now=600)
    with pytest.raises(ValueError):
        replan(s, now=500)
    with pytest.raises(ValueError):
        replan(s, now=600, overrun=Overrun("nope", 5))
    with pytest.raises(ValueError):
        Overrun("a", -1)


def test_complete_rejects_overlap_with_planned_work_and_double_completion():
    s = build_schedule([], three_tasks(), DAY)
    with pytest.raises(ValueError):
        complete(s, "a", 540, 640)  # runs into b's slot
    done = complete(s, "a", 540, 600)
    with pytest.raises(ValueError):
        complete(done, "a", 540, 600)
    with pytest.raises(KeyError):
        complete(s, "zzz", 540, 600)


def test_with_planned_minutes_does_not_touch_completed_tasks():
    s = complete(build_schedule([], three_tasks(), DAY), "a", 540, 600)
    s2 = with_planned_minutes(s, {"a": 5, "b": 90})
    assert s2.task("a").planned_minutes is None and s2.task("b").planned_minutes == 90
