"""Tried and not adopted: anchor the GBM on the user's own history mean.

    uv run python scripts/anchor_experiment.py [k ...]      # needs `make train`

Trees cannot extrapolate. Users whose habits are further from the training population than anything seen
in training get predictions stuck at the edge of what the model learned, which is one reason the quantiles
miscalibrate under the distribution-shift test. The idea tried here: train on log(actual / estimate) minus
the user's shrunk history mean, mean * n / (n + k), and add it back at prediction time, so the part that
must extrapolate is linear.

Result (see the printed table): in-distribution error (within 0.02 minutes) and coverage are unchanged, and
under the +0.30 shift the q80 coverage in a user's first week only moves from 0.68 to about 0.72 (target
0.80). Most of that gap is tasks with no history at all (14% of first-week tasks, coverage 0.41), which
nothing built from history can fix, so the change was not adopted and no headline number depends on it.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from flowday.dataset import build_dataset
from flowday.models import GBMEstimator, coverage, mae
from flowday.models.estimators import GBM_FEATURES, _log_ratio
from flowday.models.train import STRESS_SHIFT


class Anchored(GBMEstimator):
    def __init__(self, quantile: float | None, seed: int, k: float, **params: float | int) -> None:
        super().__init__(quantile, seed, **params)
        self.k = k

    def _anchor(self, df: pd.DataFrame) -> np.ndarray:
        n = df["user_n"].to_numpy(dtype=float)
        return df["user_lr_mean"].fillna(0.0).to_numpy(dtype=float) * n / (n + self.k)

    def fit(self, df: pd.DataFrame) -> Anchored:
        self.types_ = sorted(str(t) for t in df["task_type"].unique())
        kwargs: dict[str, Any] = dict(self.params)
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
        self.model.fit(self._matrix(df), _log_ratio(df).to_numpy() - self._anchor(df))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        log_ratio = self.model.predict(self._matrix(df)) + self._anchor(df)
        return np.asarray(df["estimate"].to_numpy(dtype=float) * np.exp(log_ratio))


def main() -> None:
    ks = [float(a) for a in sys.argv[1:]] or [2.0, 5.0, 10.0]
    bundle = joblib.load("artifacts/models.joblib")
    cfg, base, split = bundle["config"], bundle["estimators"], bundle["split"]
    frame = build_dataset(cfg).frame
    train = frame[frame["user_id"].isin(split["train"])]
    test = frame[frame["user_id"].isin(split["test"])]
    shifted = build_dataset(replace(cfg, n_users=100, seed=cfg.seed + 1000, offset_shift=STRESS_SHIFT)).frame
    slices = {
        "test, eval period": test[test["day"] >= cfg.eval_start],
        "shifted, first 7 days": shifted[shifted["day"] < 7],
        "shifted, eval period": shifted[shifted["day"] >= cfg.eval_start],
    }
    week = slices["shifted, first 7 days"]
    blank = week["user_n"] == 0
    y, pred = week["actual"].to_numpy(dtype=float), base["gbm_q80"].predict(week)
    print(
        f"first shifted week: {blank.mean():.0%} of tasks have no history; the trained q80 covers "
        f"{coverage(y[blank.to_numpy()], pred[blank.to_numpy()]):.2f} of those and "
        f"{coverage(y[~blank.to_numpy()], pred[~blank.to_numpy()]):.2f} of the rest"
    )
    for quantile, key in ((None, "gbm"), (0.5, "gbm_q50"), (0.8, "gbm_q80")):
        print(f"\n{key}: " + ("MAE in minutes" if quantile is None else f"coverage (nominal {quantile})"))
        models = {"as trained": base[key]} | {
            f"anchored, k={k:g}": Anchored(quantile, cfg.seed, k, **base[key].params).fit(train) for k in ks
        }
        for label, model in models.items():
            cells = []
            for name, rows in slices.items():
                y, pred = rows["actual"].to_numpy(dtype=float), model.predict(rows)
                cells.append(
                    f"{name}: " + (f"{mae(y, pred):.2f}" if quantile is None else f"{coverage(y, pred):.3f}")
                )
            print(f"  {label:18s} " + "   ".join(cells))


if __name__ == "__main__":
    main()
