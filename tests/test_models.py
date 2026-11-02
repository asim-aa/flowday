from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from flowday.dataset import DataConfig, Dataset, build_dataset
from flowday.models import (
    GBMEstimator,
    GlobalMultiplier,
    TrustEstimate,
    TypeMedianRatio,
    add_history_features,
    cluster_bootstrap_ci,
    coverage,
    history_features,
    mae,
    pinball,
    reliability_table,
    split_users,
)
from flowday.models.train import run as train_run

# --- history features: past-only, per user, per type ---------------------------------------------


def table(rows: list[tuple[int, int, str, float, float]]) -> pd.DataFrame:
    """rows: (user_id, day, task_type, estimate, actual)"""
    return pd.DataFrame(rows, columns=["user_id", "day", "task_type", "estimate", "actual"]).assign(hour=10.0)


def test_history_features_hand_computed():
    df = table(
        [
            (1, 0, "a", 10, 20),  # ratio 2
            (1, 0, "b", 10, 10),  # ratio 1
            (1, 1, "a", 10, 40),  # ratio 4
            (1, 2, "a", 10, 10),  # query row: should see days 0-1 only
            (2, 2, "a", 10, 10),  # a different user with no history
        ]
    )
    f = add_history_features(df)
    last = f.iloc[3]
    assert last["user_n"] == 3 and last["user_type_n"] == 2
    assert last["user_lr_mean"] == pytest.approx((math.log(2) + math.log(1) + math.log(4)) / 3)
    assert last["user_type_lr_mean"] == pytest.approx((math.log(2) + math.log(4)) / 2)


def test_cold_start_is_nan_with_zero_count_and_same_day_is_excluded():
    df = table([(1, 0, "a", 10, 20), (1, 0, "a", 10, 30), (1, 1, "a", 10, 10), (2, 0, "a", 10, 99)])
    f = add_history_features(df)
    assert f.loc[[0, 1, 3], "user_n"].tolist() == [0, 0, 0]  # day 0 sees nothing, not even day-0 siblings
    assert f.loc[[0, 1, 3], "user_lr_mean"].isna().all()
    assert f.loc[2, "user_n"] == 2  # day 1 sees both day-0 rows of user 1 and nothing of user 2


def test_new_type_has_overall_but_no_type_history():
    df = table([(1, 0, "a", 10, 20), (1, 1, "z", 10, 10)])
    f = add_history_features(df)
    assert f.loc[1, "user_n"] == 1 and f.loc[1, "user_type_n"] == 0
    assert math.isnan(f.loc[1, "user_type_lr_mean"])


def test_queries_can_come_from_a_later_day_than_all_history():
    hist = table([(1, 0, "a", 10, 20), (1, 1, "a", 10, 20)])
    q = pd.DataFrame({"user_id": [1, 7], "day": [50, 50], "task_type": ["a", "a"]})
    f = history_features(hist, q)
    assert f["user_n"].tolist() == [2, 0] and f["user_lr_mean"].iloc[0] == pytest.approx(math.log(2))


rows_strategy = st.lists(
    st.tuples(
        st.integers(0, 3),
        st.integers(0, 6),
        st.sampled_from(["a", "b", "c"]),
        st.integers(5, 60),
        st.integers(5, 120),
    ),
    min_size=2,
    max_size=40,
)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(rows_strategy, st.data())
def test_features_never_depend_on_the_current_or_future_days(rows, data):
    """Changing the outcome of any row on day >= d must not change the features of rows on day d."""
    df = table(rows)
    base = add_history_features(df)
    i = data.draw(st.integers(0, len(df) - 1))
    tampered = df.copy()
    tampered.loc[i, "actual"] = tampered.loc[i, "actual"] * 7 + 3
    after = add_history_features(tampered)
    unaffected = df["day"] <= df.loc[i, "day"]  # rows on the same day or earlier must be identical
    cols = ["user_lr_mean", "user_n", "user_type_lr_mean", "user_type_n"]
    pd.testing.assert_frame_equal(base.loc[unaffected, cols], after.loc[unaffected, cols])


# --- splitting by user -----------------------------------------------------------------------------


def test_split_is_by_user_disjoint_exhaustive_and_deterministic():
    ids = list(range(200))
    s = split_users(ids, seed=3)
    assert set(s["train"]).isdisjoint(s["val"]) and set(s["train"]).isdisjoint(s["test"])
    assert set(s["val"]).isdisjoint(s["test"])
    assert sorted(s["train"] + s["val"] + s["test"]) == ids
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (120, 40, 40)
    assert split_users(ids, 3) == s and split_users(ids, 4) != s


# --- metrics ---------------------------------------------------------------------------------------


def test_metrics_by_hand():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    pred = np.array([12.0, 18.0, 30.0, 50.0])
    assert mae(y, pred) == pytest.approx(3.5)
    assert coverage(y, pred) == 0.75  # 10<=12, 20>18, 30<=30, 40<=50
    assert pinball(y, pred, 0.8) == pytest.approx(np.mean([0.2 * 2, 0.8 * 2, 0.0, 0.2 * 10]))
    assert pinball(y, y, 0.8) == 0.0


def test_reliability_table_covers_every_row():
    rng = np.random.default_rng(0)
    y = rng.uniform(5, 50, 400)
    pred = y * rng.uniform(0.5, 1.5, 400)
    t = reliability_table(y, pred, pred / 20, 0.8, n_bins=5)
    assert t["n"].sum() == 400 and (t["nominal"] == 0.8).all()


def test_cluster_bootstrap_ci_contains_mean_and_widens_with_clustering():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(40), 25)
    iid = rng.normal(0, 1, 1000)
    shared = np.repeat(rng.normal(0, 1, 40), 25)  # whole cluster shares one value
    m1, lo1, hi1 = cluster_bootstrap_ci(iid, clusters)
    m2, lo2, hi2 = cluster_bootstrap_ci(shared, clusters)
    assert lo1 <= m1 <= hi1 and lo2 <= m2 <= hi2
    assert (hi2 - lo2) > 3 * (hi1 - lo1)  # row-level resampling would badly understate this


# --- estimators ------------------------------------------------------------------------------------


def test_baselines_by_hand():
    train = table([(1, 0, "a", 10, 20), (1, 0, "a", 10, 30), (1, 0, "b", 10, 10), (1, 0, "b", 10, 10)])
    q = pd.DataFrame({"task_type": ["a", "b", "new"], "estimate": [10, 10, 10], "hour": 10.0})
    assert TrustEstimate().fit(train).predict(q).tolist() == [10, 10, 10]
    g = GlobalMultiplier().fit(train)  # ratios 2, 3, 1, 1 -> median 1.5
    assert g.predict(q).tolist() == pytest.approx([15, 15, 15])
    t = TypeMedianRatio().fit(train)  # a: median 2.5, b: 1.0, unseen -> global 1.5
    assert t.predict(q).tolist() == pytest.approx([25, 10, 15])


@pytest.fixture(scope="module")
def small() -> Dataset:
    return build_dataset(DataConfig(n_users=100, n_days=60, eval_days=20, seed=1))


@pytest.fixture(scope="module")
def parts(small: Dataset) -> dict[str, pd.DataFrame]:
    split = split_users(list(small.users), 1)
    return {k: small.frame[small.frame["user_id"].isin(v)].reset_index(drop=True) for k, v in split.items()}


@pytest.fixture(scope="module")
def fitted(parts):
    train = parts["train"]
    return {
        "trust": TrustEstimate().fit(train),
        "global": GlobalMultiplier().fit(train),
        "type": TypeMedianRatio().fit(train),
        "gbm": GBMEstimator(None, 1).fit(train),
        "q50": GBMEstimator(0.5, 1).fit(train),
        "q80": GBMEstimator(0.8, 1).fit(train),
    }


def test_models_beat_baselines_on_unseen_users(parts, fitted):
    test = parts["test"]
    y = test["actual"].to_numpy(dtype=float)
    err = {k: mae(y, m.predict(test)) for k, m in fitted.items()}
    assert err["trust"] > err["global"] > err["type"] > err["gbm"]
    assert err["q50"] < err["type"]


def test_quantile_models_are_roughly_calibrated_on_unseen_users(parts, fitted):
    test = parts["test"]
    y = test["actual"].to_numpy(dtype=float)
    assert abs(coverage(y, fitted["q50"].predict(test)) - 0.5) < 0.06
    assert abs(coverage(y, fitted["q80"].predict(test)) - 0.8) < 0.06
    assert np.mean(fitted["q80"].predict(test) >= fitted["q50"].predict(test)) > 0.95


def test_gbm_handles_unseen_types_and_users_without_history(fitted):
    q = pd.DataFrame({"task_type": ["never_seen", "coding"], "estimate": [30, 30], "hour": [10.0, 15.0]})
    for name in ("gbm", "q50", "q80"):
        p = fitted[name].predict(q)  # no history columns at all (brand-new user)
        assert np.isfinite(p).all() and (p > 0).all()


def test_gbm_is_deterministic_and_survives_serialisation(parts, fitted, tmp_path: Path):
    test = parts["test"].head(200)
    again = GBMEstimator(0.8, 1).fit(parts["train"])
    np.testing.assert_allclose(again.predict(test), fitted["q80"].predict(test))
    joblib.dump(fitted["q80"], tmp_path / "m.joblib")
    np.testing.assert_allclose(joblib.load(tmp_path / "m.joblib").predict(test), fitted["q80"].predict(test))


# --- the training entry point ----------------------------------------------------------------------


def test_train_run_writes_artifacts_with_disjoint_user_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)  # train writes data/ relative to the cwd
    cfg = DataConfig(n_users=30, n_days=30, eval_days=10, seed=2)
    metrics = train_run(cfg, tmp_path / "artifacts", tmp_path / "reports")
    bundle = joblib.load(tmp_path / "artifacts" / "models.joblib")
    split = bundle["split"]
    assert not set(split["train"]) & set(split["test"]) and not set(split["val"]) & set(split["test"])
    assert not set(split["train"]) & set(split["val"])
    assert bundle["synthetic"] is True and metrics["data"]["synthetic"] is True
    assert set(bundle["estimators"]) == {
        "trust",
        "global_multiplier",
        "type_median",
        "gbm",
        "gbm_q50",
        "gbm_q80",
    }
    saved = json.loads((tmp_path / "reports" / "model_metrics.json").read_text())
    assert set(saved["mae_minutes"]) == {"all_days", "cold_start_first_7_days", "eval_period"}
    assert {"gbm_q50", "gbm_q80"} <= set(saved["quantile_calibration"])
    assert set(pd.read_csv(tmp_path / "data" / "synthetic_history.csv")["source"]) == {"synthetic"}
