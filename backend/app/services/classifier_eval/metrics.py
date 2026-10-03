"""Classifier metrics, baselines, McNemar, paired-bootstrap gains."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


LABELS = ("No Fit", "Potential Fit", "Good Fit")


def _finite_pair(y: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return y, yhat


def quadratic_weighted_kappa(y: np.ndarray, yhat: np.ndarray) -> float | None:
    try:
        return float(cohen_kappa_score(y, yhat, weights="quadratic"))
    except ValueError:
        return None


def three_class_metrics(
    y: np.ndarray,
    yhat: np.ndarray,
    *,
    proba: np.ndarray | None = None,
) -> dict[str, Any]:
    y, yhat = _finite_pair(y, yhat)
    report = classification_report(
        y, yhat, labels=[0, 1, 2], target_names=list(LABELS), output_dict=True, zero_division=0
    )
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, yhat, labels=[0, 1, 2], zero_division=0
    )
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "macro_f1": float(f1_score(y, yhat, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, yhat, average="weighted", zero_division=0)),
        "qwk": quadratic_weighted_kappa(y, yhat),
        "per_class": {
            LABELS[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
            }
            for i in range(3)
        },
        "confusion_matrix": confusion_matrix(y, yhat, labels=[0, 1, 2]).tolist(),
        "classification_report": report,
    }
    if proba is not None and proba.shape[1] >= 3 and len(np.unique(y)) > 1:
        try:
            out["ovr_auc_macro"] = float(
                roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=[0, 1, 2])
            )
        except ValueError:
            out["ovr_auc_macro"] = None
    else:
        out["ovr_auc_macro"] = None
    return out


def binary_metrics(y_bin: np.ndarray, yhat_bin: np.ndarray, scores: np.ndarray | None = None) -> dict[str, Any]:
    out = {
        "accuracy": float(accuracy_score(y_bin, yhat_bin)),
        "balanced_accuracy": float(balanced_accuracy_score(y_bin, yhat_bin)),
        "macro_f1": float(f1_score(y_bin, yhat_bin, average="macro", zero_division=0)),
        "auc": None,
    }
    if scores is not None and len(np.unique(y_bin)) == 2:
        try:
            out["auc"] = float(roc_auc_score(y_bin, scores))
        except ValueError:
            out["auc"] = None
    return out


def majority_baseline(y: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y)
    counts = np.bincount(y, minlength=3)
    maj = int(np.argmax(counts))
    yhat = np.full_like(y, maj)
    return {"majority_class": maj, **three_class_metrics(y, yhat)}


def chance_baseline(y: np.ndarray, *, seed: int = 42, n: int = 200) -> dict[str, Any]:
    """Expected metrics under uniform-random predictions (Monte Carlo)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    accs, f1s, bas = [], [], []
    for _ in range(n):
        yhat = rng.integers(0, 3, size=y.size)
        accs.append(accuracy_score(y, yhat))
        f1s.append(f1_score(y, yhat, average="macro", zero_division=0))
        bas.append(balanced_accuracy_score(y, yhat))
    return {
        "accuracy": float(np.mean(accs)),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(bas)),
    }


def stratified_random_baseline(y: np.ndarray, *, seed: int = 42, n: int = 200) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.bincount(y, minlength=3).astype(float)
    p = p / p.sum()
    accs, f1s, bas = [], [], []
    for _ in range(n):
        yhat = rng.choice(3, size=y.size, p=p)
        accs.append(accuracy_score(y, yhat))
        f1s.append(f1_score(y, yhat, average="macro", zero_division=0))
        bas.append(balanced_accuracy_score(y, yhat))
    return {
        "accuracy": float(np.mean(accs)),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(bas)),
    }


def bootstrap_metric_ci(
    y: np.ndarray,
    yhat: np.ndarray,
    fn,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    extra=None,
) -> dict[str, float | None]:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    n = y.size
    observed = fn(y, yhat) if extra is None else fn(y, yhat, extra)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if extra is None:
            v = fn(y[idx], yhat[idx])
        else:
            v = fn(y[idx], yhat[idx], extra[idx] if hasattr(extra, "__getitem__") else extra)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return {"value": None if observed is None else float(observed), "ci_low": None, "ci_high": None}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"value": float(observed), "ci_low": float(lo), "ci_high": float(hi)}


def paired_bootstrap_gain(
    y: np.ndarray,
    yhat_a: np.ndarray,
    yhat_b: np.ndarray,
    fn,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float | None]:
    """Gain of b minus a (e.g. variant − V0)."""
    y = np.asarray(y)
    a = np.asarray(yhat_a)
    b = np.asarray(yhat_b)
    n = y.size
    obs = fn(y, b) - fn(y, a)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        vals[i] = fn(y[idx], b[idx]) - fn(y[idx], a[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        "gain": float(obs),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "improves": bool(lo > 0),
    }


def mcnemar_accuracy(y: np.ndarray, yhat_a: np.ndarray, yhat_b: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y)
    a_ok = np.asarray(yhat_a) == y
    b_ok = np.asarray(yhat_b) == y
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    from scipy.stats import binomtest

    # Exact McNemar (binomial on discordant pairs)
    n_disc = n01 + n10
    if n_disc == 0:
        p = 1.0
    else:
        p = float(binomtest(n10, n_disc, 0.5, alternative="two-sided").pvalue)
    return {"n01_v0_correct_var_wrong": n01, "n10_v0_wrong_var_correct": n10, "p_value": p}


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    adj: list[float | None] = [None] * len(p_values)
    if not indexed:
        return adj
    m = len(indexed)
    order = sorted(indexed, key=lambda t: t[1])
    running = 0.0
    for rank, (i, p) in enumerate(order):
        val = min(1.0, (m - rank) * p)
        val = max(val, running)
        running = val
        adj[i] = float(val)
    return adj
