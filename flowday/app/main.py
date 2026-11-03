"""FlowDay demo app.   Run with:  make app   (or: streamlit run flowday/app/main.py)

Enter fixed events and flexible tasks, plan the day, then report "ran over by N minutes" and
watch the rest of the day replan. Switch the duration estimator between baselines and the ML
model to compare. Everything the ML model knows came from SYNTHETIC data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from flowday.app.inputs import parse_events, parse_tasks
from flowday.app.session import (
    ESTIMATOR_LABELS,
    PlanningSession,
    compare_estimators,
    persona_history,
    title_of,
)
from flowday.dataset import DataConfig
from flowday.eventlog import EventLog
from flowday.models import Estimator, TrustEstimate
from flowday.models.train import quick_fit
from flowday.scheduler import Schedule, fmt_time
from flowday.simulator import TASK_TYPES

ARTIFACT = Path(os.environ.get("FLOWDAY_ARTIFACT", "artifacts/models.joblib"))
DB_PATH = Path(os.environ.get("FLOWDAY_DB", "data/flowday.db"))
ORIGIN = datetime(2026, 1, 5)  # arbitrary date; only the time of day matters
KIND_COLORS = {
    "Fixed event": "#8a8985",
    "Planned task": "#2a78d6",
    "Done": "#1baf7a",
    "Done (ran over)": "#eb6834",
}

DEFAULT_EVENTS = pd.DataFrame(
    {
        "title": ["Team standup", "Lunch", "1:1 with manager"],
        "start": ["09:30", "12:00", "14:00"],
        "end": ["10:00", "13:00", "14:30"],
    }
)
DEFAULT_TASKS = pd.DataFrame(
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


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


@st.cache_resource(show_spinner=False)
def load_estimators() -> tuple[dict[str, Estimator] | None, DataConfig | None, list[int]]:
    """Models trained by `make train`, if present: (estimators, data config, demo user ids)."""
    if not ARTIFACT.exists():
        return None, None, []
    bundle = joblib.load(ARTIFACT)
    names = ["trust", "global_multiplier", "type_median", "gbm", "gbm_q80"]
    return {n: bundle["estimators"][n] for n in names}, bundle["config"], bundle["split"]["test"][:8]


@st.cache_resource(show_spinner="Training a small demo model on synthetic data (about 20 s)...")
def quick_estimators() -> dict[str, Estimator]:
    return quick_fit(DataConfig(n_users=150, n_days=60, eval_days=20, seed=0))


def gantt(schedule: Schedule, height: int = 280) -> alt.Chart:
    rows = [(e.title or e.id, "Fixed event", e.start, e.end) for e in schedule.events]
    for p in schedule.completed:
        planned = schedule.task(p.task_id).duration
        kind = "Done (ran over)" if p.end - p.start > planned else "Done"
        rows.append((title_of(schedule, p.task_id), kind, p.start, p.end))
    rows += [(title_of(schedule, p.task_id), "Planned task", p.start, p.end) for p in schedule.placements]
    df = pd.DataFrame(rows, columns=["item", "kind", "start_min", "end_min"])
    df["start"] = ORIGIN + pd.to_timedelta(df["start_min"], unit="m")
    df["end"] = ORIGIN + pd.to_timedelta(df["end_min"], unit="m")
    df["span"] = df["start_min"].map(fmt_time) + "-" + df["end_min"].map(fmt_time)
    window_start = ORIGIN + pd.to_timedelta(schedule.windows[0][0], unit="m")
    window_end = ORIGIN + pd.to_timedelta(schedule.windows[-1][1], unit="m")
    chart: alt.Chart = (
        alt.Chart(df)
        .mark_bar(height=18, cornerRadius=3)
        .encode(
            x=alt.X(
                "start:T",
                axis=alt.Axis(format="%H:%M", title=None),
                scale=alt.Scale(domain=[window_start, window_end]),
            ),
            x2="end:T",
            y=alt.Y("item:N", sort=alt.EncodingSortField("start_min"), title=None),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=list(KIND_COLORS), range=list(KIND_COLORS.values())),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["item", "kind", "span"],
        )
        .properties(height=height)
    )
    return chart


def schedule_table(schedule: Schedule) -> pd.DataFrame:
    rows = []
    for p in schedule.placements:
        t = schedule.task(p.task_id)
        rows.append(
            {
                "Task": t.title or t.id,
                "Type": t.type,
                "Your estimate": t.estimated_minutes,
                "Reserved": t.duration,
                "Start": fmt_time(p.start),
                "End": fmt_time(p.end),
                "Deadline": "" if t.deadline is None else fmt_time(t.deadline),
                "Priority": t.priority,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class Settings:
    estimators: dict[str, Estimator]
    cfg: DataConfig
    name: str
    start: int
    end: int
    persona: int | None
    log_on: bool


def sidebar() -> Settings:
    st.sidebar.header("Settings")
    estimators, cfg, demo_users = load_estimators()
    if estimators is None:
        st.sidebar.warning("No trained model found. Run `make train`, or train a small demo model now.")
        if st.sidebar.button("Train small demo model", key="quick_train"):
            st.session_state["quick_models"] = quick_estimators()
            st.rerun()
        estimators = st.session_state.get("quick_models")
        cfg = DataConfig(n_users=150, n_days=60, eval_days=20, seed=0)
        demo_users = list(range(8))
    assert cfg is not None
    start = _minutes(st.sidebar.time_input("Work day starts", time(9, 0), key="work_start"))
    end = _minutes(st.sidebar.time_input("Work day ends", time(18, 0), key="work_end"))

    if estimators is None:  # no model yet: only the user's own estimates are available
        estimators = {"trust": TrustEstimate()}
    name = st.sidebar.radio(
        "Duration estimates",
        list(estimators),
        format_func=lambda k: ESTIMATOR_LABELS.get(k, k),
        key="estimator",
        help="How long the scheduler reserves for each task. Baselines use only your estimate; "
        "ML also uses task type, time of day and, for a persona, their history.",
    )
    persona_options: list[int | None] = [None, *demo_users]
    persona = st.sidebar.selectbox(
        "Who is the ML model planning for?",
        persona_options,
        format_func=lambda u: (
            "A new user (no history)" if u is None else f"Synthetic user #{u} (history on file)"
        ),
        key="persona",
    )
    log_on = st.sidebar.checkbox(f"Log plans and completions to {DB_PATH}", value=True, key="log_on")
    st.sidebar.caption("All demo data and the ML model are synthetic. Not evidence of real-world benefit.")
    return Settings(estimators, cfg, name, start, end, persona, log_on)


def main() -> None:
    st.set_page_config(page_title="FlowDay", layout="wide")
    st.title("FlowDay")
    st.caption(
        "A daily planner that learns how long your tasks really take. Demo model trained on **synthetic** data."
    )

    cfg = sidebar()
    estimators, name, window = cfg.estimators, cfg.name, [(cfg.start, cfg.end)]
    plan_tab, compare_tab, log_tab = st.tabs(["Plan and replan", "Compare estimators", "Event log"])

    with plan_tab:
        left, right = st.columns(2)
        with left:
            st.subheader("Fixed events")
            events_df = st.data_editor(
                DEFAULT_EVENTS,
                num_rows="dynamic",
                key="events",
                column_config={
                    "start": st.column_config.TextColumn("start (HH:MM)", validate=r"^\d{1,2}:\d{2}$"),
                    "end": st.column_config.TextColumn("end (HH:MM)", validate=r"^\d{1,2}:\d{2}$"),
                },
            )
        with right:
            st.subheader("Tasks")
            tasks_df = st.data_editor(
                DEFAULT_TASKS,
                num_rows="dynamic",
                key="tasks",
                column_config={
                    "type": st.column_config.SelectboxColumn("type", options=list(TASK_TYPES), required=True),
                    "estimate": st.column_config.NumberColumn("your estimate (min)", min_value=1, step=5),
                    "deadline": st.column_config.TextColumn(
                        "deadline (HH:MM, optional)", validate=r"^(\d{1,2}:\d{2})?$"
                    ),
                    "priority": st.column_config.NumberColumn(
                        "priority (5 = most important)", min_value=1, max_value=5, step=1
                    ),
                },
            )
        events, event_errors = parse_events(events_df)
        tasks, task_errors = parse_tasks(tasks_df)
        for message in [*event_errors, *task_errors]:
            st.error(message)
        if cfg.start >= cfg.end:
            st.error("The work day must end after it starts.")
        inputs_ok = not (event_errors or task_errors) and bool(tasks) and cfg.start < cfg.end

        history = None
        if cfg.persona is not None and tasks:
            history, (ratio, n) = persona_history(cfg.cfg, cfg.persona, tasks)
            if name in ("gbm", "gbm_q80"):
                st.caption(
                    f"From their history this user typically takes **x{ratio:.2f}** their estimate (based on {n} logged tasks)."
                )

        if st.button("Plan my day", type="primary", key="plan", disabled=not inputs_ok):
            previous: PlanningSession | None = st.session_state.get("session")
            if previous is not None:
                previous.close()
            log = EventLog(DB_PATH) if cfg.log_on else None
            st.session_state["session"] = PlanningSession.start(
                name, estimators[name], events, tasks, window, history, log
            )

        session: PlanningSession | None = st.session_state.get("session")
        if session is not None:
            show_session(session)

    with compare_tab:
        if not inputs_ok:
            st.info("Fix the inputs on the first tab to compare estimators.")
        else:
            table, schedules = compare_estimators(estimators, events, tasks, window, history)
            st.dataframe(table, hide_index=True, width="stretch")
            st.caption("Same events and tasks; only the reserved duration of each task differs.")
            names = list(schedules)
            for col, default, key in zip(
                st.columns(2),
                ("trust", "gbm" if "gbm" in schedules else names[-1]),
                ("cmp_left", "cmp_right"),
                strict=True,
            ):
                with col:
                    pick = st.selectbox(
                        "Estimator",
                        names,
                        index=names.index(default),
                        format_func=lambda k: ESTIMATOR_LABELS.get(k, k),
                        key=key,
                    )
                    st.altair_chart(gantt(schedules[pick]), width="stretch")

    with log_tab:
        if not DB_PATH.exists():
            st.info("Nothing logged yet. Plan a day with logging enabled.")
        else:
            with EventLog(DB_PATH) as log:
                st.write({t: log.count(t) for t in ("users", "tasks", "plans", "plan_items", "completions")})
                st.dataframe(log.history_frame().tail(20), hide_index=True, width="stretch")


def show_session(session: PlanningSession) -> None:
    s = session.schedule
    st.divider()
    st.subheader(f"Your day, planned with: {ESTIMATOR_LABELS[session.estimator_name]}")
    st.altair_chart(gantt(s), width="stretch")

    nxt = session.next_task()
    if nxt is None:
        done = len(s.completed)
        st.success(f"Day complete: {done} task(s) done, {len(s.dropped)} dropped.")
    else:
        st.markdown(
            f"**Next up: {title_of(s, nxt.task_id)}**, planned {fmt_time(nxt.start)}-{fmt_time(nxt.end)}"
        )
        c1, c2, c3 = st.columns([1, 1, 1])
        c1.button("Done as planned", key="done", on_click=session.mark_done)
        c2.number_input("Ran over by (minutes)", min_value=1, max_value=480, value=15, step=5, key="extra")
        c3.write("")
        # read the number at click time: `args=` would capture whatever it was when the page rendered
        c3.button(
            "Task ran over",
            key="overrun",
            type="primary",
            on_click=lambda: session.overrun(int(st.session_state["extra"])),
        )
    if session.changes:
        st.info("Replanned. " + "  \n".join(session.changes))

    if s.placements:
        st.dataframe(schedule_table(s), hide_index=True, width="stretch")
    if s.dropped:
        st.warning(
            "Could not schedule:  \n"
            + "  \n".join(f"**{title_of(s, d.task_id)}**: {d.detail}" for d in s.dropped)
        )


main()
