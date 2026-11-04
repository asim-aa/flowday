"""Export two real plans of the app's default day for the project page (docs/day.json).

    uv run python scripts/make_site_data.py      # needs `make train`

The same events and tasks as the app's defaults (flowday/app/main.py), planned three ways for a brand-new
user with no history: trusting the user's estimates, the median ML estimator, and the cautious (80th
percentile) one.
Nothing is simulated: this is only what the scheduler reserves, not how the day would go.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from flowday.app.inputs import parse_events, parse_tasks
from flowday.app.session import compare_estimators
from flowday.scheduler import Schedule

EVENTS = pd.DataFrame(
    {
        "title": ["Team standup", "Lunch", "1:1 with manager"],
        "start": ["09:30", "12:00", "14:00"],
        "end": ["10:00", "13:00", "14:30"],
    }
)
TASKS = pd.DataFrame(
    {
        "title": [
            "Write project proposal",
            "Reply to emails",
            "Fix login bug",
            "Prep for 1:1",
            "Expense report",
            "Study",
        ],
        "type": ["writing", "email", "coding", "meeting_prep", "admin", "study"],
        "estimate": [90, 20, 60, 30, 25, 60],
        "deadline": ["15:00", "", "17:00", "13:50", "", ""],
        "priority": [5, 3, 4, 4, 2, 3],
    }
)
WINDOW = (9 * 60, 18 * 60)
PLANS = {
    "trust": "Trusting your estimates",
    "gbm": "ML, median",
    "gbm_q80": "ML, cautious (80th percentile)",
}


def describe(schedule: Schedule) -> dict[str, object]:
    return {
        "placed": [
            {
                "title": schedule.task(p.task_id).title,
                "type": schedule.task(p.task_id).type,
                "start": p.start,
                "end": p.end,
                "estimate": schedule.task(p.task_id).estimated_minutes,
                "deadline": schedule.task(p.task_id).deadline,
            }
            for p in sorted(schedule.placements, key=lambda p: p.start)
        ],
        "dropped": [
            {
                "title": schedule.task(d.task_id).title,
                "reserved": schedule.task(d.task_id).duration,
                "reason": d.reason.value,
                "detail": d.detail,
            }
            for d in schedule.dropped
        ],
    }


def main() -> None:
    events, event_errors = parse_events(EVENTS)
    tasks, task_errors = parse_tasks(TASKS)
    assert not (event_errors or task_errors), (event_errors, task_errors)
    models = joblib.load("artifacts/models.joblib")["estimators"]
    _, schedules = compare_estimators({k: models[k] for k in PLANS}, events, tasks, [WINDOW])
    out = {
        "window": WINDOW,
        "events": [{"title": e.title, "start": e.start, "end": e.end} for e in events],
        "plans": {k: {"label": label, **describe(schedules[k])} for k, label in PLANS.items()},
    }
    Path("docs/day.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, plan in out["plans"].items():
        print(k, [(t["title"], t["end"] - t["start"]) for t in plan["placed"]], "dropped:", plan["dropped"])  # type: ignore[index]


if __name__ == "__main__":
    main()
