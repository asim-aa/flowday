"""Shared fixtures: one small synthetic dataset, split by user, with models fitted on its train users."""

from __future__ import annotations

import pandas as pd
import pytest

from flowday.dataset import DataConfig, Dataset, build_dataset
from flowday.models import (
    Estimator,
    GBMEstimator,
    GlobalMultiplier,
    TrustEstimate,
    TypeMedianRatio,
    split_users,
)

SMALL = DataConfig(n_users=100, n_days=60, eval_days=20, seed=1)


@pytest.fixture(scope="session")
def small() -> Dataset:
    return build_dataset(SMALL)


@pytest.fixture(scope="session")
def split(small: Dataset) -> dict[str, list[int]]:
    return split_users(list(small.users), SMALL.seed)


@pytest.fixture(scope="session")
def parts(small: Dataset, split: dict[str, list[int]]) -> dict[str, pd.DataFrame]:
    return {k: small.frame[small.frame["user_id"].isin(v)].reset_index(drop=True) for k, v in split.items()}


@pytest.fixture(scope="session")
def fitted(parts: dict[str, pd.DataFrame]) -> dict[str, Estimator]:
    train = parts["train"]
    return {
        "trust": TrustEstimate().fit(train),
        "global_multiplier": GlobalMultiplier().fit(train),
        "type_median": TypeMedianRatio().fit(train),
        "gbm": GBMEstimator(None, SMALL.seed).fit(train),
        "gbm_q50": GBMEstimator(0.5, SMALL.seed).fit(train),
        "gbm_q80": GBMEstimator(0.8, SMALL.seed).fit(train),
    }
