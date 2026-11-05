"""Error and calibration metrics, plus a cluster bootstrap that resamples whole users."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def bias(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean signed error (pred - y); negative means under-prediction."""
    return float(np.mean(pred - y))


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of outcomes at or below the predicted quantile. Calibrated q-quantile -> ~q."""
    return float(np.mean(y <= pred))


def reliability_table(
    y: np.ndarray, pred: np.ndarray, ratio: np.ndarray, q: float, n_bins: int = 8
) -> pd.DataFrame:
    """Coverage within bins of the predicted quantile (binned by predicted/estimate ratio)."""
    bins = pd.qcut(ratio, n_bins, duplicates="drop")
    frame = pd.DataFrame({"covered": y <= pred, "ratio": ratio, "bin": bins})
    grouped = frame.groupby("bin", observed=True)
    out = grouped.agg(n=("covered", "size"), coverage=("covered", "mean"), mean_ratio=("ratio", "mean"))
    out["nominal"] = q
    return out.reset_index(drop=True)


def cluster_bootstrap_ci(
    values: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean of `values` with a percentile CI from resampling whole clusters (e.g. users).

    Rows within a cluster are correlated, so resampling rows would give intervals that are too
    narrow. Returns (mean, low, high); the mean is the pooled row-level mean.
    """
    values = np.asarray(values, dtype=float)
    _, inv = np.unique(clusters, return_inverse=True)
    n_clusters = int(np.max(inv)) + 1
    sums = np.bincount(inv, weights=values, minlength=n_clusters)
    counts = np.bincount(inv, minlength=n_clusters).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(values.mean()), float(lo), float(hi)
