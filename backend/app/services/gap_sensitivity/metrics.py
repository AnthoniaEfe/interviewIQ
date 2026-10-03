"""Statistics for gap-sensitivity analysis (bootstrap, Wilcoxon, transitions)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from app.schemas.domain import MatchLabel
from app.services.gap_sensitivity.scoring import is_gap_label, three_category

LABEL_ORDER = (
    MatchLabel.STRONG.value,
    MatchLabel.PARTIAL.value,
    MatchLabel.WEAK.value,
    MatchLabel.MISSING.value,
)

THREE_ORDER = ("strength", "partial", "gap")


def label_counts(labels: list[MatchLabel | str]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for lab in labels:
        key = lab.value if isinstance(lab, MatchLabel) else str(lab)
        c[key] += 1
    return {k: int(c.get(k, 0)) for k in LABEL_ORDER}


def three_category_counts(labels: list[MatchLabel | str]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for lab in labels:
        ml = lab if isinstance(lab, MatchLabel) else MatchLabel(lab)
        c[three_category(ml)] += 1
    return {k: int(c.get(k, 0)) for k in THREE_ORDER}


def gap_share(labels: list[MatchLabel | str]) -> float | None:
    if not labels:
        return None
    n_gap = 0
    for lab in labels:
        ml = lab if isinstance(lab, MatchLabel) else MatchLabel(lab)
        if is_gap_label(ml):
            n_gap += 1
    return n_gap / len(labels)


def per_pair_gap_rates(pair_labels: list[list[MatchLabel | str]]) -> np.ndarray:
    rates = []
    for labels in pair_labels:
        g = gap_share(labels)
        rates.append(float(g) if g is not None else float("nan"))
    return np.asarray(rates, dtype=np.float64)


def summarise_rates(rates: np.ndarray) -> dict[str, float | None]:
    clean = rates[np.isfinite(rates)]
    if clean.size == 0:
        return {
            "mean": None,
            "median": None,
            "iqr": None,
            "q1": None,
            "q3": None,
            "n": 0,
        }
    q1, q3 = np.percentile(clean, [25, 75])
    return {
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "iqr": float(q3 - q1),
        "q1": float(q1),
        "q3": float(q3),
        "n": int(clean.size),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | None]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, clean.size, size=(n_resamples, clean.size))
    means = clean[idx].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(clean.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(clean.size),
    }


def bootstrap_pooled_gap_share_ci(
    pair_labels: list[list[MatchLabel | str]],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | None]:
    """Bootstrap over pairs for the pooled gap share."""
    n = len(pair_labels)
    if n == 0:
        return {"share": None, "ci_low": None, "ci_high": None, "n_pairs": 0}
    rng = np.random.default_rng(seed)
    observed_gap = 0
    observed_total = 0
    for labels in pair_labels:
        for lab in labels:
            ml = lab if isinstance(lab, MatchLabel) else MatchLabel(lab)
            observed_total += 1
            if is_gap_label(ml):
                observed_gap += 1
    shares = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idxs = rng.integers(0, n, size=n)
        g = t = 0
        for i in idxs:
            for lab in pair_labels[i]:
                ml = lab if isinstance(lab, MatchLabel) else MatchLabel(lab)
                t += 1
                if is_gap_label(ml):
                    g += 1
        shares[b] = g / t if t else float("nan")
    clean = shares[np.isfinite(shares)]
    low, high = np.percentile(clean, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "share": (observed_gap / observed_total) if observed_total else None,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_pairs": n,
        "n_requirements": observed_total,
    }


def matched_pairs_rank_biserial(diffs: np.ndarray) -> float | None:
    """Matched-pairs rank-biserial correlation from paired differences (x - y)."""
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs != 0]
    if diffs.size == 0:
        return None
    ranks = np.empty(diffs.size, dtype=np.float64)
    order = np.argsort(np.abs(diffs))
    ranks[order] = np.arange(1, diffs.size + 1, dtype=np.float64)
    pos = ranks[diffs > 0].sum()
    neg = ranks[diffs < 0].sum()
    denom = pos + neg
    if denom == 0:
        return None
    return float((pos - neg) / denom)


def wilcoxon_paired(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Wilcoxon signed-rank test on paired samples (x vs y).

    Returns p-value, effect size (matched-pairs rank-biserial on x-y), n.
    Uses ``zero_method='wilcox'`` (drop zeros). If fewer than 1 non-zero
    difference, returns null p/effect.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(x.size)
    diffs = x - y
    nonzero = int(np.sum(diffs != 0))
    if nonzero < 1:
        return {
            "n_pairs": n,
            "n_nonzero": nonzero,
            "statistic": None,
            "p_value": None,
            "rank_biserial": None,
        }
    # SciPy requires at least one non-zero; for very small n it still runs.
    try:
        result = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        stat = float(result.statistic)
        p = float(result.pvalue)
    except ValueError:
        stat, p = None, None
    return {
        "n_pairs": n,
        "n_nonzero": nonzero,
        "statistic": stat,
        "p_value": p,
        "rank_biserial": matched_pairs_rank_biserial(diffs),
    }


def transition_matrix(
    labels_from: list[MatchLabel | str],
    labels_to: list[MatchLabel | str],
) -> dict[str, dict[str, int]]:
    if len(labels_from) != len(labels_to):
        raise ValueError("Transition matrices require aligned label lists")
    mat = {a: {b: 0 for b in LABEL_ORDER} for a in LABEL_ORDER}
    for a, b in zip(labels_from, labels_to, strict=True):
        ka = a.value if isinstance(a, MatchLabel) else str(a)
        kb = b.value if isinstance(b, MatchLabel) else str(b)
        mat[ka][kb] += 1
    return mat


def spearman_dose_response(
    evidence_sizes: np.ndarray,
    gap_rates: np.ndarray,
) -> dict[str, float | None]:
    x = np.asarray(evidence_sizes, dtype=np.float64)
    y = np.asarray(gap_rates, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"rho": None, "p_value": None, "n": int(x.size)}
    rho, p = spearmanr(x, y)
    return {
        "rho": float(rho) if np.isfinite(rho) else None,
        "p_value": float(p) if np.isfinite(p) else None,
        "n": int(x.size),
    }
