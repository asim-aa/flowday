# FlowDay

An ML-informed daily scheduler. It plans a day around fixed events and flexible tasks, learns how long
tasks *actually* take compared with what the user estimated, and replans when a task overruns.

> **Everything in this repository that involves people is synthetic.** The users, task histories, biases
> and every number below come from a simulator written by the author. The results show how the method
> behaves *when its assumptions hold*. They are not evidence that it would help real people. See
> [Why these results do not prove real-world benefit](#why-these-results-do-not-prove-real-world-benefit).

![Schedule outcomes per estimator](reports/figures/fig3_schedule_outcomes.png)

## Contents

1. [The problem](#the-problem)
2. [Quick start](#quick-start)
3. [How it fits together](#how-it-fits-together)
4. [The scheduler: algorithm and tradeoffs](#the-scheduler-algorithm-and-tradeoffs)
5. [Synthetic data, and why results do not prove real-world benefit](#synthetic-data)
6. [Duration models](#duration-models)
7. [End-to-end evaluation](#end-to-end-evaluation)
8. [Event log](#event-log)
9. [The app](#the-app)
10. [Tests and quality](#tests-and-quality)
11. [Limitations and next steps](#limitations-and-next-steps)

## The problem

People plan their day from their own time estimates, and those estimates are biased: usually low, more so
for some kinds of work (writing and coding more than email), more for some people, and more at some times
of day. A planner that believes the estimates over-promises, so the day collapses into a chain of
overruns, missed deadlines and meetings that start while you are still working.

FlowDay separates two jobs:

1. **A deterministic scheduler** that, given durations, produces a conflict-free plan and an explanation
   for anything it could not fit. It contains no ML and can be reasoned about and tested exhaustively.
2. **A duration estimator** that turns "I think this takes 60 minutes" into a better number (a median, or a
   cautious 80th percentile), learned from what the user's past tasks really took.

The scheduler only ever sees a number of minutes per task, so estimators are swappable, which is what
makes the comparison in this README possible.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

| Command | What it does | Time (laptop) |
|---|---|---|
| `make install` | create the virtualenv and install dependencies | |
| `make test` | run the full pytest suite (unit + property-based + headless app tests) | ~50 s |
| `make check` | ruff + strict mypy + tests | ~1 min |
| `make train` | generate synthetic data, tune and train models, write `artifacts/` and `reports/model_metrics.json` | ~40 s |
| `make evaluate` | run the scheduler end to end with every estimator; write `reports/` and the figures | ~40 s |
| `make figures` | regenerate the figures from the JSON reports | seconds |
| `make app` | launch the Streamlit app (run `make train` first for the ML estimators) | |

Everything is seeded; re-training and re-running the evaluation reproduced the committed
`reports/schedule_results.json` exactly (checked by diffing the JSON). Generated data (`data/`) and the trained model (`artifacts/`) are git-ignored.

## How it fits together

```
flowday/
  scheduler/    deterministic core: Event, Task, Schedule, build_schedule, replan, complete,
                expand_routines, validate_schedule. Standard library only.
  simulator/    synthetic users, days and histories; simulate_day executes a plan with true durations
  models/       features, 3 baselines, GBM point + quantile models, metrics, train CLI
  planning.py   glue: sizes tasks with an estimator, then schedules / replans
  dataset.py    builds the synthetic dataset
  evaluation/   end-to-end runner (parallel, paired, cluster-bootstrap CIs) and the three figures
  eventlog/     SQLite schema + observer that logs plans, completions and actual durations
  app/          Streamlit app; pure logic lives in session.py / inputs.py so it is unit-tested
tests/          pytest + hypothesis
reports/        committed results: JSON, markdown tables and figures
```

The dependency direction is `scheduler` ← `simulator` ← `models` ← `planning` ← `evaluation` / `app`.
`scheduler` never imports anything else in the project.

## The scheduler: algorithm and tradeoffs

**Model.** Times are integer minutes from an arbitrary origin. Work happens inside *windows* (one per
day, so multi-day horizons work). *Fixed events* block time. *Tasks* have a type, a duration, an optional
hard deadline and a priority (1 to 5, 5 most important). Tasks are non-preemptible: each gets one
contiguous slot inside a single free gap.

**Algorithm.**

1. Compute free time: windows minus fixed events minus already-completed work, clipped to `now`.
2. Walk the tasks from highest priority down (ties: earlier deadline, then shorter, then id).
3. *Admit* a task only if the already-admitted tasks plus this one can still all be sequenced with every
   deadline met. That feasibility check is an exhaustive depth-first search over task orders, with
   earliest-deadline-first as the first thing it tries, memoisation on (set placed, time), a cheap
   per-deadline capacity bound, and a node budget (20,000).
4. Otherwise *drop* the task and say why: `no_fit_alone` (not even placeable on an otherwise empty
   calendar, with the longest free gap quoted) or `capacity` (fits alone, but not beside the
   higher-priority tasks, which are named).

The search is complete: if any sequencing meets every deadline, some earliest-fit-after-the-previous-task
sequencing does (shifting any task left to its earliest fit cannot hurt a deadline), so with the budget
not hit, **nothing is dropped whenever everything fits.**

**`replan(schedule, now, overrun)`** records the overrunning task as completed over its *actual* interval
(planned start to planned end plus the overrun), then re-runs the algorithm on everything not completed,
from `max(now, that end)`. It never changes fixed events or completed work. Tasks that were planned before
`now` but never completed are treated as not done, and previously dropped tasks get another chance.
Fixed events are never moved even if the overrunning task ran into one: that collision is reported
(`collisions()`), not hidden.

**Recurring routines** (`Routine`, `expand_routines`) expand a daily or per-weekday pattern into fixed
events (a study block at 19:00) or flexible tasks (study for 60 minutes before 21:00).

**Tradeoffs, stated honestly.**

- *Not globally optimal.* Priority-ordered admission is a greedy rule on top of an exact feasibility
  check. It guarantees a task is never dropped to make room for something that ranks lower (a property
  test checks this against a brute-force oracle), but it can drop more total value than the best possible
  selection. It is also not monotone: removing one task can admit a larger one that then crowds out a
  third. A property-based test found exactly this while I was writing the suite; it is a consequence of
  the rule, not a bug.
- *Exponential worst case.* The problem is NP-hard even with gaps. In practice the first branch (EDF) or
  the capacity bound settles typical days in a handful of nodes. If the budget is exhausted the result is
  still valid (no deadline is ever missed) but may drop more than necessary, and
  `Schedule.search_truncated` says so. In an instrumented run of 4,500 simulated days (73,497 feasibility
  searches) the budget was never reached.
- *No splitting, no dependencies, no release times, no travel time or buffers.* Each is a reasonable
  extension; none is modelled.
- *Placement is earliest-fit, left-packed.* Good for deadlines, indifferent to what a person would rather
  do at which hour.
- *Deadlines are hard.* There is no "late but still worth doing" option.

**Evidence it is correct** (all in `tests/`, run by `make test`):

| Property | How it is checked |
|---|---|
| No two scheduled items overlap; nothing outside work hours; every placed task ends by its deadline; every task is exactly one of placed / completed / dropped | hypothesis, 200 random scenarios, `validate_schedule` |
| Fixed events never move | hypothesis, events compared before and after, including through whole days of replans |
| Nothing is dropped when a feasible schedule exists, and vice versa | hypothesis against an independent brute-force oracle (every permutation, minute-grid representation) |
| A task is dropped only if it cannot fit beside the tasks that outrank it | same oracle |
| Drop reasons are truthful | each `no_fit_alone` / `capacity` claim is re-checked by scheduling the task alone |
| Replanning never touches completed work, places nothing before `now` or before the overrun ends | hypothesis over random completion prefixes, overruns and clock values, including a clock *inside* completed work |
| Deterministic | same input, equal output |

I also checked that these tests can fail: I injected four bugs into the scheduler (ignore deadlines,
forget fixed events, forget completed work in replans, an over-aggressive pruning rule). The first run
caught three; the fourth survived because my replan strategy never put `now` inside completed work. I
widened the strategy and all four are now caught.

## Synthetic data

There is no real data in this project. `flowday/simulator/profiles.py` defines the generative model, and
every row carries `source = "synthetic"`.

```
log(actual / estimate) = log(type_factor[type])                      systematic bias by task type
                       + user.offset                                 per-user bias, ~ N(0, 0.15)
                       + user.size_slope * log(estimate / 30)        bigger tasks are underestimated more
                       + user.tod_scale * type_tod[type] * tod(hour) slower away from the user's peak hour
                       + user.noise_sigma * z,  z ~ N(0, 1)          noise (sigma ~ 0.22, varies by user)
```

`tod(hour)` is zero at the user's chronotype peak (drawn around 11:00), rises to one twelve hours away,
plus a post-lunch dip around 14:00.

| Task type | email | admin | errands | meeting prep | exercise | study | writing | coding |
|---|---|---|---|---|---|---|---|---|
| base actual/estimate | 1.00 | 1.10 | 1.10 | 1.15 | 0.95 | 1.20 | 1.35 | 1.40 |
| time-of-day sensitivity | 0.10 | 0.10 | 0.05 | 0.20 | 0.05 | 0.30 | 0.35 | 0.30 |

After all effects, the median actual/estimate in the generated data is 1.23 overall (mean 1.32): coding
1.66, writing 1.60, study 1.40, meeting prep 1.23, admin 1.12, errands 1.11, email 1.00, exercise 1.00. Each synthetic day has 1 to 3
meetings and 4 to 8 tasks; about 55% of tasks have a same-day deadline. Hidden noise is drawn once per
task, so every estimator is evaluated on exactly the same days and the same "true" durations (common
random numbers).

### Why these results do not prove real-world benefit

- **The model was told the answer's shape.** The generator is log-linear in a handful of features the
  estimator is given. A gradient-boosted model recovering it to within a few minutes of the theoretical
  noise floor (below) says the pipeline works, not that human durations are this learnable.
- **The biases are my assumptions**, loosely inspired by the planning-fallacy literature, not measured.
  Magnitudes, the type ranking and the time-of-day curve could all be wrong in reality.
- **Nothing is non-stationary.** Real people change habits, have good and bad weeks, get interrupted, and
  change their estimates *because* a planner exists. Here each user's parameters are frozen.
- **Plans are followed perfectly.** No one skips a task, reorders, takes a call, or ignores the app.
- **Tasks are independent and clean.** Real task types are messy free text, not eight clean labels.
- **One seed, one population, one set of design choices.** Intervals reflect sampling noise over simulated
  users, not uncertainty about the simulator itself.
- **Calibration is in-distribution.** The stress test below shows how quickly that breaks.

The first line of defence is the oracle and the noise floor, which stop me from reading too much into
small gaps. The second is a **distribution-shift test**: score the already-trained models on a population
in which every user underestimates 0.30 more (on the log scale) than anyone in training, with no
retraining.

| Shifted population | MAE trust | MAE per-type median | MAE GBM | q50 coverage (nominal 0.50) | q80 coverage (nominal 0.80) |
|---|---|---|---|---|---|
| first 7 days of a user's history | 44.5 | 29.0 | 19.5 | **0.35** | **0.68** |
| days 60 to 89 | 43.6 | 28.2 | 16.1 | 0.48 | 0.80 |

The point predictions stay best, but the quantiles are badly miscalibrated for new users and only recover
once the model has personal history to correct for (it sees each user's past actual/estimate ratios).
That is a feature of how this particular model uses history, and it is the kind of failure to expect from
real data too.

## Duration models

All models predict *actual minutes* from `(task type, user's estimate, start hour, user history)`.

| Model | What it is |
|---|---|
| `trust` | (a) believe the user's estimate |
| `global_multiplier` | (b) estimate x the median actual/estimate over all training tasks |
| `type_median` | (c) estimate x the median actual/estimate for that task type |
| `gbm` | gradient boosting (scikit-learn `HistGradientBoostingRegressor`) on `log(actual / estimate)`; point estimate |
| `gbm_q50`, `gbm_q80` | the same model with pinball loss for the 50th and 80th percentile. Quantiles are unchanged by the log transform, so `exp(prediction)` is a quantile of actual duration |

I used scikit-learn rather than LightGBM to avoid a native OpenMP dependency on macOS.

**Features.** log estimate, task type, start hour, and four *past-only* history features per user: the mean
log(actual/estimate) over all their earlier tasks and over earlier tasks of the same type, with counts.
"Past-only" means strictly earlier days, because outcomes of the same day are unknown when it is planned.
Leakage is the easy way to fool yourself here, so it is tested directly: changing the outcome of any row
must not change the features of any row on the same or an earlier day (a hypothesis test), and the
estimators are shown a spy-checked frame that contains no outcome column. Missing history (a brand-new
user) is handled natively, so cold-start users still get a prediction.

**Split.** 500 synthetic users, 90 days each (about 270,000 tasks), split **by user**: 300 train, 100
validation, 100 test. No user appears in two sets (asserted in code and in a test). Hyper-parameters are
chosen on validation users (a four-point grid); every number below is from test users. Days 60 to 89 of
each timeline are the evaluation period used in the end-to-end run, and days 0 to 6 are the cold-start
slice.

**Accuracy** (mean absolute error in minutes on the 100 test users; 95% CI from resampling whole users,
since one user's tasks are correlated):

| Estimator | All test days (54,044 tasks) | First 7 days | Evaluation period |
|---|---|---|---|
| trust | 23.2 [21.6, 24.9] | 22.9 [21.0, 24.7] | 23.1 [21.4, 24.9] |
| global_multiplier | 18.5 [17.3, 19.8] | 18.1 [16.8, 19.6] | 18.4 [17.2, 19.7] |
| type_median | 14.6 [13.6, 15.6] | 14.4 [13.3, 15.6] | 14.6 [13.6, 15.7] |
| **gbm** | **11.8 [11.1, 12.6]** | **12.6 [11.6, 13.6]** | **11.7 [10.9, 12.5]** |
| gbm_q50 | 11.9 [11.1, 12.6] | 12.7 [11.8, 13.7] | 11.7 [10.9, 12.5] |
| *noise floor: true conditional median* | *11.4 [10.6, 12.1]* | *11.4 [10.6, 12.3]* | *11.3 [10.5, 12.1]* |

The GBM beats the per-type median by 2.7 minutes overall (paired 95% CI 2.1 to 3.4), which is the
increment attributable to time of day and per-user history beyond task type. Trusting the user
under-predicts by 20.5 minutes per task on average (bias), the per-type median by 3.1, the GBM by 2.1. The
noise floor is the MAE of an estimator that knows each synthetic user's true parameters, so the GBM being
close to it is a statement about the simulator's simplicity as much as about the model.

![MAE by estimator](reports/figures/fig1_estimation_error.png)

**Quantile calibration** (test users): the 50th-percentile model was exceeded by 49.5% of actual
durations (CI 48.9% to 50.1%) and the 80th by 80.6% (CI 79.5% to 81.8%), across task types within 0.49 to
0.51 and 0.80 to 0.83 and across times of day within 0.01. Reading the baselines as if they were
medians: trusting the user covers only 30% of outcomes.

![Quantile calibration](reports/figures/fig2_quantile_calibration.png)

## End-to-end evaluation

For each of the 100 test users and each of their 30 evaluation days (3,000 days), the scheduler plans the
day with an estimator and the simulator lives it: tasks start at their planned time and take their *true*
duration (which depends on the hour they start). **Whenever a task finishes later than planned, the day is
replanned from that moment.** A task that finishes early is not rewarded: the next one waits for its
planned start, so unused slack shows up as idle time. Six estimators are compared; "oracle" knows the true
durations and is an unattainable reference. Because durations depend on start time and start time depends
on durations, hour-dependent estimators size tasks, schedule, re-size each task for the slot it landed in,
and schedule again (two passes).

Per simulated day, mean with 95% CI (resampling users):

| Estimator | Deadlines missed | Replans | Idle min | Tasks dropped | Min spilling into fixed events | Overtime min | Tasks completed |
|---|---|---|---|---|---|---|---|
| Trust estimate | 0.17 [0.15, 0.20] | 3.54 [3.37, 3.70] | 27.0 [25.4, 28.6] | 0.77 [0.72, 0.81] | 11.2 [10.0, 12.4] | 6.4 [5.1, 7.8] | 5.28 [5.22, 5.34] |
| Global multiplier | 0.11 [0.09, 0.12] | 2.16 [2.00, 2.32] | 44.7 [41.8, 47.8] | 0.95 [0.91, 1.00] | 6.6 [5.7, 7.6] | 3.7 [2.9, 4.6] | 5.09 [5.04, 5.14] |
| Per-type median | 0.06 [0.04, 0.07] | 2.13 [1.92, 2.34] | 49.8 [46.3, 53.6] | 1.02 [0.97, 1.06] | 3.4 [2.6, 4.2] | 1.6 [1.2, 2.0] | 5.03 [4.98, 5.08] |
| GBM (ML) | 0.05 [0.04, 0.06] | 2.39 [2.33, 2.45] | 41.1 [39.7, 42.6] | 0.98 [0.91, 1.04] | 2.6 [2.1, 3.0] | 0.9 [0.6, 1.1] | 5.07 [5.00, 5.14] |
| GBM q80 (ML, cautious) | 0.02 [0.01, 0.02] | 0.84 [0.77, 0.90] | 65.1 [63.5, 66.7] | 1.25 [1.18, 1.32] | 0.9 [0.6, 1.2] | 0.3 [0.2, 0.5] | 4.80 [4.72, 4.87] |
| Oracle (true durations) | 0.00 [0.00, 0.00] | 0.02 [0.02, 0.03] | 28.6 [27.5, 29.7] | 0.91 [0.85, 0.97] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] | 5.14 [5.08, 5.21] |

(Full numbers and paired differences against "trust" are in `reports/schedule_results.json`; a day has
about six tasks.)

What this shows, and does not:

- **Better durations make days more reliable.** Against trusting the user, the ML median plan cuts missed
  deadlines from 0.17 to 0.05 per day (paired difference -0.12, CI -0.14 to -0.11), minutes spilling into
  fixed events from 11.2 to 2.6, and overtime from 6.4 to 0.9.
- **There is no free lunch; the gain is paid for in throughput and idle time.** Trusting the user
  *completes the most tasks* (5.28 per day, more than the oracle's 5.14) because it over-commits and then
  gets them done by spilling into meetings, working overtime and missing deadlines. ML planning drops about
  0.2 more tasks per day, up front and with a reason, instead of failing them later. Whether that is the
  right trade depends on what an overrun costs a person, which this study cannot measure.
- **Planning at the median still overruns about half the time.** Replans fall only from 3.5 to 2.4 per day
  for the median GBM, because roughly half of tasks exceed their median by construction. The cautious
  80th-percentile plan is what cuts replans (0.84 per day) and misses (0.02), at the cost of 24 more idle
  minutes per day than the median GBM plan (65 vs 41; 38 more than trusting the user, CI +36 to +40) and
  0.27 more dropped tasks per day (1.25 vs 0.98). It trades reliability for slack, and the slack is not
  reclaimed in this simulator, so this overstates the idle cost a real person would pay.
- **The per-type median is a strong baseline.** It is within noise of the GBM on missed deadlines
  (0.06 vs 0.05), has slightly fewer replans (2.1 vs 2.4), and is far better than the global multiplier.
  The GBM's clearest edge is overtime (1.6 to 0.9, non-overlapping CIs); its edge on minutes spilling into
  fixed events (3.4 to 2.6) is within noise. With no real data to validate on, the per-type ratio is the
  lowest-risk piece to ship: it is simple and captures most of the benefit.
- **The oracle is not perfectly zero** (0.001 missed deadlines and 0.02 replans per day): sizing tasks for
  the slot they land in is a fixed-point iteration that very rarely fails to converge in four passes.
  This is an approximation in the reference, not in the scheduler.

![Schedule outcomes](reports/figures/fig3_schedule_outcomes.png)

## Event log

`flowday/eventlog` writes plans, completions and actual durations to a local SQLite file (default
`data/flowday.db`). The same schema is used by synthetic simulations and by the app, and is intended to
take real data later: `users.source` is `synthetic`, `demo` or `real`, and `EventLog.history_frame()`
returns exactly the table the models train on (user, day, task type, estimate, start hour, actual).

```
users        user_id PK, source CHECK IN (synthetic, demo, real), created_at
tasks        (user_id, task_id) PK, day (ISO date), task_type, estimated_minutes > 0,
             deadline_min, priority, title, created_at
plans        plan_id PK, user_id, day, trigger CHECK IN (initial, overrun, manual), estimator,
             parent_plan_id -> plans, now_min, created_at          (one row per plan snapshot)
plan_items   (plan_id, kind, item_id) PK, kind IN (event, task, completed, dropped),
             start_min, end_min, planned_minutes, drop_reason, drop_detail
completions  completion_id PK, (user_id, task_id) UNIQUE -> tasks, started_min, ended_min > started_min,
             actual_minutes, planned_minutes, plan_id -> plans, logged_at
```

Times are minutes since midnight of the row's `day`. Foreign keys and CHECK constraints are enforced and
tested (including a bug the tests caught: `INSERT OR IGNORE` silently swallowed CHECK violations, so
invalid sources would have vanished instead of failing). A test trains a model directly from a logged
database to show the path from logged data to a fitted estimator works.

## The app

`make app` starts a Streamlit app. Edit the fixed events and tasks, choose how durations are estimated
(your estimate, three baselines, or ML median / cautious q80), and press **Plan my day**. The timeline
shows fixed events, planned and finished tasks; tasks that did not fit are listed with the reason. For
the next task, press **Done as planned**, or enter minutes and press **Task ran over**: the rest of the
day replans, the app says in plain English what moved or no longer fits, and warns when the overrun ran
into a fixed event. **Compare estimators** plans the same inputs under every estimator side by side. The
**ML persona** selector chooses between a brand-new user and one of eight synthetic users whose history
the model uses. With no trained model present, the app falls back to the user's own estimates and offers
to train a small demo model. Plans and completions are logged to the event log unless the sidebar box is
unticked.

The app has no auth, hosted database, payments or LLM by design. I checked it in a real browser and
headlessly with Streamlit's `AppTest`; one bug surfaced that way (Streamlit binds a button's `args` when
the page renders, so the overrun minutes must be read when the button is clicked).

## Tests and quality

- **116 tests**: unit, hypothesis property-based (scheduler, replanning, feature leakage, whole days of
  clicks through the app session), headless Streamlit, and a mini end-to-end check that ML beats trusting
  the user on a small simulated population.
- `make check` runs `ruff`, **strict** `mypy` on the package, and the tests.
- Bugs the tests found in my own code while building this: baselines computed a median on the log scale
  instead of the ratio scale (even counts differed); `INSERT OR IGNORE` hid constraint violations; the
  stale-`args` button bug; and several of my own over-strong assertions (priority-greedy admission is not
  monotone; the oracle is not exactly perfect), which I corrected rather than hid.

## Limitations and next steps

**Limitations**

- Synthetic only. See above; this is the dominant limitation.
- Early finishes are not reclaimed (no "pull the next task forward"), which inflates idle time for
  conservative estimators.
- Non-preemptible tasks: an overrun cannot be split around a meeting; it collides instead.
- One day of tasks is planned at a time in the simulator and app (the scheduler supports multi-day
  windows and routines; the experiments do not exercise them).
- The 80th-percentile estimator is trained to be calibrated, not to be optimal for the scheduler's
  objective; there is no tuning of the quantile against the cost of an overrun versus a drop.
- History features are not updated from simulated executions during an evaluation day (they come from
  pre-generated history), which keeps estimators comparable but is not how a deployed system would learn.
- Greedy admission can drop more value than an optimal selection would; no solver comparison was done.

**Next steps, in rough order of value**

1. Run on real logged data (the event log is ready for it) before believing any number here; start by
   checking whether the type ratios and the cold-start behaviour look anything like the simulator's.
2. Reclaim early finishes (replan on any large deviation, not only overruns).
3. Choose the planning quantile per task from the cost of an overrun (deadline, event collision) versus
   a drop, rather than a global q50 or q80.
4. Task splitting and dependencies; release times; user preferences for when to do which work.
5. Online learning and drift detection, since real habits change.
6. Compare against an optimal solver (MILP / CP-SAT) on small instances to quantify the greedy gap.
