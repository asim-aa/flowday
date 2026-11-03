"""Inject known bugs into the scheduler and check that the test suite notices each one.

    make mutation

Each mutant edits `flowday/scheduler/core.py` in memory-then-disk, runs only the property tests that
are supposed to catch it, and the original file is always restored. A mutant that survives is a hole
in the tests, except those marked equivalent (the change cannot alter behaviour).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TARGET = Path("flowday/scheduler/core.py")
P = "tests/test_scheduler_properties.py"

# (name, old text, new text, tests that must fail, equivalent?)
MUTANTS: list[tuple[str, str, str, list[str], bool]] = [
    (
        "ignore deadlines when placing",
        "return start if start + duration <= limit else None",
        "return start",
        [f"{P}::test_schedule_is_always_valid"],
        False,
    ),
    (
        "forget fixed events when computing free time",
        "blocked = [e.interval for e in events] + list(locked)",
        "blocked = list(locked)",
        [f"{P}::test_schedule_is_always_valid", f"{P}::test_fixed_events_never_move"],
        False,
    ),
    (
        "treat all time as work time (ignore windows)",
        "free = [(max(a, now), b) for a, b in subtract_intervals(windows, blocked) if b > now]",
        "free = [(max(a, now), b) for a, b in subtract_intervals([(0, 10**6)], blocked) if b > now]",
        [f"{P}::test_schedule_is_always_valid"],
        False,
    ),
    (
        "forget completed work when replanning",
        "schedule.events, remaining, schedule.windows, [p.interval for p in completed], floor, node_budget",
        "schedule.events, remaining, schedule.windows, [], floor, node_budget",
        [f"{P}::test_replan_never_touches_fixed_events_or_completed_work"],
        False,
    ),
    (
        "overrun records the wrong actual end",
        "actual = Placement(planned.task_id, planned.start, planned.end + overrun.extra_minutes)",
        "actual = Placement(planned.task_id, planned.start, planned.end)",
        [f"{P}::test_replan_places_nothing_before_now_or_the_overrun_end"],
        False,
    ),
    (
        "off-by-one in the capacity pruning rule",
        "demand > _capacity(free, t, deadline)",
        "demand >= _capacity(free, t, deadline)",
        [f"{P}::test_a_planted_exact_packing_is_always_found"],
        False,
    ),
    (
        "admit low priority first",
        "for task in sorted(tasks, key=lambda t: (-t.priority, _dl(t), t.duration, t.id)):",
        "for task in sorted(tasks, key=lambda t: (t.priority, _dl(t), t.duration, t.id)):",
        [f"{P}::test_a_task_is_only_dropped_if_it_cannot_fit_alongside_higher_ranked_tasks"],
        False,
    ),
    (
        "do not clip free time at `now` (equivalent: the search already starts at `now`)",
        "free = [(max(a, now), b) for a, b in subtract_intervals(windows, blocked) if b > now]",
        "free = [(a, b) for a, b in subtract_intervals(windows, blocked)]",
        [f"{P}::test_replan_places_nothing_before_now_or_the_overrun_end"],
        True,
    ),
]


def main() -> int:
    original = TARGET.read_text()
    survivors = 0
    try:
        for name, old, new, tests, equivalent in MUTANTS:
            if old not in original:
                print(f"STALE   {name}: pattern not found, update the script")
                survivors += 1
                continue
            TARGET.write_text(original.replace(old, new))
            result = subprocess.run(
                ["uv", "run", "pytest", *tests, "-x", "-q", "-p", "no:cacheprovider"],
                capture_output=True,
                text=True,
            )
            caught = result.returncode != 0
            verdict = "CAUGHT " if caught else ("EQUIV  " if equivalent else "MISSED!")
            survivors += (not caught) and not equivalent
            print(f"{verdict} {name}")
    finally:
        TARGET.write_text(original)
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
