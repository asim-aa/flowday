"""Write synthetic days into an event log, so the log -> retrain path can be tried end to end.

    python -m flowday.eventlog.simulate --db data/synthetic_log.db [--users 20 --days 40]

Every row is labelled source=synthetic. Days are planned with the user's own estimates and executed
by the simulator, so start hours come from the scheduler rather than being random.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from flowday.simulator import TrustPlanner, generate_timeline, sample_users, simulate_day

from .store import EventLog, EventLogObserver

FIRST_DAY = date(2026, 1, 5)


def simulate_into_log(log: EventLog, n_users: int, n_days: int, seed: int = 0) -> int:
    """Simulate `n_days` days for `n_users` synthetic users into `log`. Returns tasks completed."""
    done = 0
    for user in sample_users(n_users, seed):
        for day in generate_timeline(user, n_days, seed):
            iso = (FIRST_DAY + timedelta(days=day.index)).isoformat()
            observer = EventLogObserver(log, f"synthetic-{user.user_id}", iso, "trust")
            done += simulate_day(day, TrustPlanner(), observer).tasks_completed
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=Path("data/synthetic_log.db"))
    ap.add_argument("--users", type=int, default=20)
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    with EventLog(args.db) as log:
        done = simulate_into_log(log, args.users, args.days, args.seed)
    print(f"Logged {done} synthetic completions for {args.users} users to {args.db}")


if __name__ == "__main__":
    main()
