"""The validator is the oracle behind the property tests, so test that it really detects violations,
and test the search-budget path the README makes claims about."""

from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import HealthCheck, given, settings

from flowday.scheduler import (
    Dropped,
    DropReason,
    Event,
    Placement,
    Task,
    build_schedule,
    complete,
    validate_schedule,
)

from .strategies import Scenario, scenarios

WINDOWS = [(0, 100)]


def good() -> tuple[Event, ...]:
    return (Event("e", 40, 50),)


def base():
    tasks = [Task("a", "x", 30, deadline=100), Task("b", "x", 20)]
    s = build_schedule(good(), tasks, WINDOWS)
    assert validate_schedule(s) == []
    return s


@pytest.mark.parametrize(
    "corrupt, fragment",
    [
        (lambda s: replace(s, placements=(Placement("a", 0, 30), Placement("b", 20, 40))), "overlaps"),
        (
            lambda s: replace(s, placements=(Placement("a", 0, 30), Placement("b", 45, 65))),
            "overlaps event e",
        ),
        (lambda s: replace(s, placements=(Placement("a", 80, 110), s.placements[1])), "after deadline"),
        (lambda s: replace(s, placements=(Placement("a", 0, 25), s.placements[1])), "!= duration"),
        (
            lambda s: replace(s, placements=(Placement("a", 90, 120), s.placements[1])),
            "not inside a work window",
        ),
        (lambda s: replace(s, now=500), "starts before now"),
        (lambda s: replace(s, placements=(s.placements[0],)), "exactly one of"),
        (lambda s: replace(s, dropped=(Dropped("b", DropReason.CAPACITY, "x"),)), "exactly one of"),
        (lambda s: replace(s, completed=(Placement("a", 0, 10),)), "exactly one of"),
    ],
)
def test_validator_detects_each_kind_of_violation(corrupt, fragment):
    problems = validate_schedule(corrupt(base()))
    assert any(fragment in p for p in problems), problems


def test_validator_flags_planned_work_over_completed_work_and_completed_overlaps():
    s = complete(base(), "a", 0, 30)
    assert validate_schedule(s) == []
    over_completed = replace(s, placements=(Placement("b", 10, 30),))
    assert any("overlaps completed a" in p for p in validate_schedule(over_completed))
    two_completed = replace(
        complete(s, "b", 60, 80), completed=(Placement("a", 0, 30), Placement("b", 20, 40))
    )
    assert any("completed a overlaps completed b" in p for p in validate_schedule(two_completed))


def test_validator_allows_completed_work_that_ran_into_an_event_and_overlapping_events():
    s = complete(base(), "a", 35, 45)  # reality: the task ran into event e (40-50)
    assert validate_schedule(replace(s, placements=(Placement("b", 50, 70),))) == []


def test_complete_rejects_an_empty_interval():
    with pytest.raises(ValueError):
        complete(base(), "a", 10, 10)


# --- the node budget -------------------------------------------------------------------------------

TIGHT = Scenario(
    events=[Event("e", 30, 40)],
    tasks=[Task("a", "x", 60, deadline=100), Task("b", "x", 30, deadline=100)],
    windows=WINDOWS,
)


def test_exhausted_budget_stays_valid_flags_itself_and_explains_the_drop():
    cut = build_schedule(TIGHT.events, TIGHT.tasks, TIGHT.windows, node_budget=1)
    assert cut.search_truncated
    assert validate_schedule(cut) == []  # a worse packing, never a missed deadline
    (d,) = cut.dropped
    assert d.reason is DropReason.CAPACITY and "budget exhausted" in d.detail
    full = build_schedule(TIGHT.events, TIGHT.tasks, TIGHT.windows)
    assert not full.search_truncated and not full.dropped  # the default budget finds the packing


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenarios(max_tasks=6))
def test_small_budgets_never_break_validity_and_untruncated_results_are_exact(sc):
    exact = build_schedule(sc.events, sc.tasks, sc.windows)
    for budget in (1, 3, 10):
        s = build_schedule(sc.events, sc.tasks, sc.windows, node_budget=budget)
        assert validate_schedule(s) == []
        if not s.search_truncated:
            assert s == exact
