"""Synthetic users and the documented generative model of how long tasks *really* take.

EVERYTHING in this package is synthetic. The biases below are plausible stand-ins for the
"planning fallacy", chosen by the author, not measured from real people. Results obtained
on this data say how the *method* behaves when its assumptions hold, not how it would do
on real users.

Generative model (log scale, per task):

    log(actual / estimate) = log(type_factor[type])                        systematic, by task type
                           + user.offset                                   per-user bias
                           + user.size_slope * log(estimate / 30)          big tasks are underestimated more
                           + user.tod_scale * type_tod[type] * tod(hour)   time-of-day slowdown
                           + user.noise_sigma * z,   z ~ N(0, 1)           irreducible noise

    tod(hour) = 0.5 * (1 - cos(2*pi*(hour - user.peak_hour) / 24))   0 at the user's peak, 1 twelve hours away
              + 0.35 * exp(-0.5 * ((hour - 14) / 1.2) ** 2)          post-lunch dip for everyone
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TypeSpec:
    median_estimate: float  # minutes: typical user estimate for this kind of task
    estimate_sigma: float  # log-scale spread of estimates
    factor: float  # typical actual/estimate ratio (before user, size and time-of-day effects)
    tod_sensitivity: float  # how much time of day slows this type (cognitive work > chores)
    weight: float  # relative frequency


TASK_TYPES: dict[str, TypeSpec] = {
    "email": TypeSpec(15, 0.5, 1.00, 0.10, 0.20),
    "admin": TypeSpec(25, 0.5, 1.10, 0.10, 0.15),
    "errands": TypeSpec(30, 0.4, 1.10, 0.05, 0.08),
    "meeting_prep": TypeSpec(30, 0.4, 1.15, 0.20, 0.10),
    "exercise": TypeSpec(45, 0.3, 0.95, 0.05, 0.07),  # people slightly over-budget this one
    "study": TypeSpec(60, 0.4, 1.20, 0.30, 0.14),
    "writing": TypeSpec(60, 0.45, 1.35, 0.35, 0.13),
    "coding": TypeSpec(75, 0.45, 1.40, 0.30, 0.13),
}

POST_LUNCH_DIP = 0.35
NOISE_FLOOR_MINUTES = 1


def tod_shape(hour: float, peak_hour: float) -> float:
    circadian = 0.5 * (1.0 - math.cos(2.0 * math.pi * (hour - peak_hour) / 24.0))
    dip = POST_LUNCH_DIP * math.exp(-0.5 * ((hour - 14.0) / 1.2) ** 2)
    return circadian + dip


@dataclass(frozen=True)
class UserProfile:
    """One synthetic person. Parameters are drawn once and never change."""

    user_id: int
    offset: float  # log-scale personal bias, ~N(0, 0.15)
    size_slope: float  # extra log-bias per log-unit of estimate size, >= 0
    peak_hour: float  # hour of day at which this user is fastest (chronotype)
    tod_scale: float  # multiplier on time-of-day sensitivity
    noise_sigma: float  # log-scale noise
    rounding: int  # the user rounds estimates to multiples of this many minutes

    def log_ratio_mean(self, task_type: str, estimate: float, hour: float) -> float:
        spec = TASK_TYPES[task_type]
        return (
            math.log(spec.factor)
            + self.offset
            + self.size_slope * math.log(max(estimate, 1.0) / 30.0)
            + self.tod_scale * spec.tod_sensitivity * tod_shape(hour, self.peak_hour)
        )

    def median_minutes(self, task_type: str, estimate: float, hour: float) -> float:
        """The true conditional median duration. Not available to any real estimator; used only
        as a noise-floor reference when reporting model error."""
        return estimate * math.exp(self.log_ratio_mean(task_type, estimate, hour))

    def true_minutes(self, task_type: str, estimate: float, hour: float, z: float) -> int:
        """Actual duration for a task given its pre-drawn noise `z` (common random numbers)."""
        log_ratio = self.log_ratio_mean(task_type, estimate, hour) + self.noise_sigma * z
        return max(NOISE_FLOOR_MINUTES, round(estimate * math.exp(log_ratio)))


def sample_users(n: int, seed: int, first_id: int = 0, offset_shift: float = 0.0) -> list[UserProfile]:
    """Draw `n` synthetic users deterministically from `seed`.

    `offset_shift` adds a constant to every user's log-bias, to build populations that underestimate
    more (or less) than the training population for distribution-shift tests."""
    rng = np.random.default_rng([seed, 0xF10D])
    users = []
    for i in range(n):
        users.append(
            UserProfile(
                user_id=first_id + i,
                offset=float(rng.normal(0.0, 0.15)) + offset_shift,
                size_slope=float(np.clip(rng.normal(0.08, 0.04), 0.0, None)),
                peak_hour=float(np.clip(rng.normal(11.0, 2.0), 8.0, 15.0)),
                tod_scale=float(rng.lognormal(0.0, 0.3)),
                noise_sigma=float(np.clip(rng.lognormal(math.log(0.22), 0.25), 0.08, 0.5)),
                rounding=int(rng.choice([5, 15], p=[0.7, 0.3])),
            )
        )
    return users
