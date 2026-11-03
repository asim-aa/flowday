"""Duration estimators: three baselines and gradient-boosted models (point + quantile).

Every estimator maps a frame with columns task_type, estimate, hour (plus the history features
for the GBMs) to predicted *actual minutes*.
"""

from __future__ import annotations

from typing import Protocol, Self

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import HISTORY_FEATURES


class Estimator(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def uses_hour(self) -> bool:
        """Does the prediction depend on when the task will run?"""
        ...

    def predict(self, df: pd.DataFrame) -> np.ndarray: ...


def _ratio(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df["actual"] / df["estimate"], index=df.index)


def _log_ratio(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.log(_ratio(df)), index=df.index)


class TrustEstimate:
    """(a) Believe the user."""

    name = "trust"
    uses_hour = False

    def fit(self, df: pd.DataFrame) -> Self:
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["estimate"].to_numpy(dtype=float)


class GlobalMultiplier:
    """(b) One multiplier for everyone: the median of actual / estimate in the training data."""

    name = "global_multiplier"
    uses_hour = False

    def __init__(self) -> None:
        self.ratio = 1.0

    def fit(self, df: pd.DataFrame) -> Self:
        self.ratio = float(_ratio(df).median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["estimate"].to_numpy(dtype=float) * self.ratio


class TypeMedianRatio:
    """(c) The median actual / estimate for the task's type; the global median for unseen types."""

    name = "type_median"
    uses_hour = False

    def __init__(self) -> None:
        self.ratio_by_type: dict[str, float] = {}
        self.fallback = 1.0

    def fit(self, df: pd.DataFrame) -> Self:
        ratio = _ratio(df)
        self.fallback = float(ratio.median())
        self.ratio_by_type = {str(k): float(v) for k, v in ratio.groupby(df["task_type"]).median().items()}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ratio = df["task_type"].map(self.ratio_by_type).fillna(self.fallback).to_numpy(dtype=float)
        return df["estimate"].to_numpy(dtype=float) * ratio


GBM_FEATURES = ["log_estimate", "type_code", "hour", *HISTORY_FEATURES]


class GBMEstimator:
    """Gradient boosting (sklearn HistGradientBoosting) on the log of actual / estimate.

    quantile=None  -> squared error on the log ratio (a point estimate; ~ the median when the
                      noise is symmetric on the log scale).
    quantile=q     -> pinball loss, i.e. the q-th conditional quantile. Quantiles are unchanged by
                      the monotone log transform, so exp() of the prediction is the q-th quantile of
                      the actual duration.
    Handles missing history features natively, so brand-new users (no history) still get a
    prediction. Task types never seen in training are treated as missing.
    """

    uses_hour = True

    def __init__(self, quantile: float | None = None, seed: int = 0, **params: float | int) -> None:
        self.quantile = quantile
        self.seed = seed
        self.params: dict[str, float | int] = {
            "max_iter": 200,
            "learning_rate": 0.1,
            "max_leaf_nodes": 16,
            "min_samples_leaf": 100,
            "l2_regularization": 1.0,
            **params,
        }
        self.types_: list[str] = []
        self.model: HistGradientBoostingRegressor | None = None

    @property
    def name(self) -> str:
        return "gbm" if self.quantile is None else f"gbm_q{round(self.quantile * 100)}"

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        codes = df["task_type"].map({t: i for i, t in enumerate(self.types_)}).to_numpy(dtype=float)
        cols = {
            "log_estimate": np.log(df["estimate"].to_numpy(dtype=float)),
            "type_code": codes,
            "hour": df["hour"].to_numpy(dtype=float),
        }
        for f in HISTORY_FEATURES:
            cols[f] = df[f].to_numpy(dtype=float) if f in df else np.full(len(df), np.nan)
        return np.column_stack([cols[c] for c in GBM_FEATURES])

    def fit(self, df: pd.DataFrame) -> Self:
        self.types_ = sorted(str(t) for t in df["task_type"].unique())
        kwargs: dict[str, object] = dict(self.params)
        if self.quantile is None:
            kwargs["loss"] = "squared_error"
        else:
            kwargs.update(loss="quantile", quantile=self.quantile)
        self.model = HistGradientBoostingRegressor(
            categorical_features=[GBM_FEATURES.index("type_code")],
            early_stopping=False,
            random_state=self.seed,
            **kwargs,
        )
        self.model.fit(self._matrix(df), _log_ratio(df).to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() first")
        log_ratio: np.ndarray = self.model.predict(self._matrix(df))
        return df["estimate"].to_numpy(dtype=float) * np.exp(log_ratio)
