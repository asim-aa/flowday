import pytest

from flowday.scheduler import (
    Routine,
    build_schedule,
    daily_windows,
    expand_routines,
    hhmm,
    validate_schedule,
)


def test_fixed_routine_becomes_daily_events():
    study = Routine("study", "Study block", 90, kind="fixed", start_minute=hhmm("19:00"))
    events, tasks = expand_routines([study], num_days=3)
    assert tasks == []
    assert [(e.id, e.start, e.end) for e in events] == [
        ("study@0", 1140, 1230),
        ("study@1", 1440 + 1140, 1440 + 1230),
        ("study@2", 2880 + 1140, 2880 + 1230),
    ]


def test_flexible_routine_becomes_tasks_with_daily_deadlines():
    r = Routine(
        "read", "Read", 45, kind="flexible", deadline_minute=hhmm("17:00"), task_type="study", priority=4
    )
    _, tasks = expand_routines([r], num_days=2)
    assert [(t.id, t.type, t.estimated_minutes, t.deadline, t.priority) for t in tasks] == [
        ("read@0", "study", 45, 1020, 4),
        ("read@1", "study", 45, 1440 + 1020, 4),
    ]


def test_weekday_filter_uses_first_weekday():
    gym = Routine("gym", "Gym", 60, kind="fixed", start_minute=hhmm("07:00"), weekdays=frozenset({0, 2, 4}))
    events, _ = expand_routines([gym], num_days=7, first_weekday=0)
    assert [e.id for e in events] == ["gym@0", "gym@2", "gym@4"]
    events, _ = expand_routines([gym], num_days=7, first_weekday=1)  # day 0 is Tuesday
    assert [e.id for e in events] == ["gym@1", "gym@3", "gym@6"]


def test_expanded_routines_feed_the_scheduler():
    routines = [
        Routine("standup", "Standup", 15, kind="fixed", start_minute=hhmm("09:30")),
        Routine("study", "Study", 60, kind="flexible", deadline_minute=hhmm("17:00"), task_type="study"),
    ]
    events, tasks = expand_routines(routines, num_days=2)
    s = build_schedule(events, tasks, daily_windows(2, hhmm("09:00"), hhmm("17:00")))
    assert validate_schedule(s) == [] and s.dropped == ()
    assert len(s.placements) == 2


def test_routine_validation():
    with pytest.raises(ValueError):
        Routine("x", "x", 30, kind="fixed")  # needs a start
    with pytest.raises(ValueError):
        Routine("x", "x", 30, kind="flexible", start_minute=60)
    with pytest.raises(ValueError):
        Routine("x", "x", 30, kind="sometimes")
    with pytest.raises(ValueError):
        Routine("x", "x", 120, kind="fixed", start_minute=1400)
