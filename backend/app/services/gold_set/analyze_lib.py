"""Agreement and weighted-estimate helpers for the gold set."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size < 2 or len(np.unique(np.concatenate([y_true, y_pred]))) < 2:
        try:
            return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        except ValueError:
            return None
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def unweighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights=None))
    except ValueError:
        return None


def percent_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return None
    return float(np.mean(y_true == y_pred))


def bootstrap_metric_ci(
    y_a: np.ndarray,
    y_b: np.ndarray,
    fn,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float | None]:
    y_a = np.asarray(y_a)
    y_b = np.asarray(y_b)
    n = y_a.size
    observed = fn(y_a, y_b)
    if n < 2 or observed is None:
        return {"value": observed, "ci_low": None, "ci_high": None, "n": int(n)}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        v = fn(y_a[idx], y_b[idx])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"value": float(observed), "ci_low": None, "ci_high": None, "n": int(n)}
    arr = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {
        "value": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[mask], weights[mask]
    if values.size == 0:
        return None
    return float(np.sum(weights * values) / np.sum(weights))
