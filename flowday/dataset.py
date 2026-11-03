"""Build the synthetic dataset used for training and evaluation. ALL OF IT IS SYNTHETIC."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from flowday.models import add_history_features
from flowday.simulator import SyntheticDay, UserProfile, generate_timeline, history_frame, sample_users


@dataclass(frozen=True)
class DataConfig:
    n_users: int = 500
    n_days: int = 90
    eval_days: int = 30  # the last `eval_days` of each timeline are the end-to-end evaluation period
    seed: int = 0
    # stress / sensitivity knobs (neutral at the defaults); see simulator.sample_users
    offset_shift: float = 0.0
    bias_scale: float = 1.0
    noise_scale: float = 1.0
    drift_sd: float = 0.0  # per-user habit drift, log-units per day (sd across users)
    tail_df: float | None = None  # heavy-tailed noise: Student-t degrees of freedom

    @property
    def eval_start(self) -> int:
        return self.n_days - self.eval_days


@dataclass
class Dataset:
    config: DataConfig
    users: dict[int, UserProfile]
    timelines: dict[int, list[SyntheticDay]]
    frame: pd.DataFrame  # one row per task with past-only history features attached


def make_users(cfg: DataConfig) -> list[UserProfile]:
    """The synthetic users described by `cfg` (the single place the generator knobs are applied)."""
    return sample_users(
        cfg.n_users,
        cfg.seed,
        offset_shift=cfg.offset_shift,
        bias_scale=cfg.bias_scale,
        noise_scale=cfg.noise_scale,
        drift_sd=cfg.drift_sd,
        tail_df=cfg.tail_df,
    )


def build_dataset(cfg: DataConfig) -> Dataset:
    users = {u.user_id: u for u in make_users(cfg)}
    timelines = {uid: generate_timeline(u, cfg.n_days, cfg.seed) for uid, u in users.items()}
    frame = pd.concat([history_frame(days, cfg.seed) for days in timelines.values()], ignore_index=True)
    return Dataset(cfg, users, timelines, add_history_features(frame))
