"""Statistics for the fit-label control (JT, MWU, AUC, OLS)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from app.services.gap_sensitivity.metrics import bootstrap_mean_ci, summarise_rates

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover
    sm = None


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    """Holm step-down adjustment. None stays None."""
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


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float | None:
    """Cliff's delta: P(x>y) - P(x<y). Positive => x tends to be larger."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return None
    gt = lt = 0
    for a in x:
        gt += int(np.sum(a > y))
        lt += int(np.sum(a < y))
    denom = x.size * y.size
    if denom == 0:
        return None
    return float((gt - lt) / denom)


def bootstrap_cliffs_delta(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | None]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    observed = cliffs_delta(x, y)
    if x.size == 0 or y.size == 0 or observed is None:
        return {"delta": None, "ci_low": None, "ci_high": None, "n_x": int(x.size), "n_y": int(y.size)}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        xs = x[rng.integers(0, x.size, size=x.size)]
        ys = y[rng.integers(0, y.size, size=y.size)]
        d = cliffs_delta(xs, ys)
        stats[b] = d if d is not None else np.nan
    clean = stats[np.isfinite(stats)]
    low, high = np.percentile(clean, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "delta": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_x": int(x.size),
        "n_y": int(y.size),
    }


def jonckheere_terpstra_statistic(groups: list[np.ndarray]) -> float:
    """JT statistic: sum of pairwise MW counts that later groups are larger."""
    stat = 0.0
    for i, gi in enumerate(groups):
        gi = np.asarray(gi, dtype=np.float64)
        gi = gi[np.isfinite(gi)]
        for gj in groups[i + 1 :]:
            gj = np.asarray(gj, dtype=np.float64)
            gj = gj[np.isfinite(gj)]
            for a in gi:
                stat += float(np.sum(gj > a) + 0.5 * np.sum(gj == a))
    return float(stat)


def jonckheere_terpstra(
    groups: list[np.ndarray],
    *,
    n_permutations: int = 10000,
    seed: int = 42,
    alternative: str = "greater",
) -> dict[str, Any]:
    """Permutation JT test. ``alternative='greater'`` = increasing group order."""
    clean = [np.asarray(g, dtype=np.float64)[np.isfinite(g)] for g in groups]
    sizes = [int(g.size) for g in clean]
    pooled = np.concatenate(clean) if all(s > 0 for s in sizes) else np.asarray([])
    n = int(pooled.size)
    observed = jonckheere_terpstra_statistic(clean) if n else None
    if n < 3 or any(s < 1 for s in sizes) or observed is None:
        return {
            "statistic": observed,
            "p_value": None,
            "n_permutations": n_permutations,
            "seed": seed,
            "alternative": alternative,
            "group_sizes": sizes,
        }
    rng = np.random.default_rng(seed)
    if alternative == "two-sided":
        null_stats = np.empty(n_permutations, dtype=np.float64)
        for b in range(n_permutations):
            perm = rng.permutation(pooled)
            start = 0
            fake = []
            for s in sizes:
                fake.append(perm[start : start + s])
                start += s
            null_stats[b] = jonckheere_terpstra_statistic(fake)
        obs_dev = abs(observed - float(null_stats.mean()))
        count = int(np.sum(np.abs(null_stats - null_stats.mean()) >= obs_dev - 1e-12))
        p = (count + 1) / (n_permutations + 1)
    else:
        count = 0
        for _ in range(n_permutations):
            perm = rng.permutation(pooled)
            start = 0
            fake = []
            for s in sizes:
                fake.append(perm[start : start + s])
                start += s
            stat = jonckheere_terpstra_statistic(fake)
            if alternative == "greater" and stat >= observed:
                count += 1
            elif alternative == "less" and stat <= observed:
                count += 1
        p = (count + 1) / (n_permutations + 1)
    return {
        "statistic": float(observed),
        "p_value": float(p),
        "n_permutations": n_permutations,
        "seed": seed,
        "alternative": alternative,
        "group_sizes": sizes,
    }


def mann_whitney(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 1 or y.size < 1:
        return {"statistic": None, "p_value": None, "n_x": int(x.size), "n_y": int(y.size)}
    try:
        res = mannwhitneyu(x, y, alternative=alternative)
        return {
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "n_x": int(x.size),
            "n_y": int(y.size),
        }
    except ValueError:
        return {"statistic": None, "p_value": None, "n_x": int(x.size), "n_y": int(y.size)}


def kruskal_wallis(groups: list[np.ndarray]) -> dict[str, Any]:
    clean = [np.asarray(g, dtype=np.float64)[np.isfinite(g)] for g in groups]
    clean = [g for g in clean if g.size]
    if len(clean) < 2:
        return {"statistic": None, "p_value": None, "n_groups": len(clean)}
    try:
        res = kruskal(*clean)
        return {
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "n_groups": len(clean),
            "group_sizes": [int(g.size) for g in clean],
        }
    except ValueError:
        return {"statistic": None, "p_value": None, "n_groups": len(clean)}


def spearman_ordinal(ordinal: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(ordinal, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
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


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if y.size < 2 or len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return None


def roc_curve_points(y_true: np.ndarray, scores: np.ndarray) -> dict[str, list[float]]:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if y.size < 2 or len(np.unique(y)) < 2:
        return {"fpr": [], "tpr": []}
    fpr, tpr, _ = roc_curve(y, s)
    return {"fpr": [float(v) for v in fpr], "tpr": [float(v) for v in tpr]}


def bootstrap_auc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    observed = roc_auc(y, s)
    n = int(y.size)
    if observed is None or n < 2:
        return {"auc": observed, "ci_low": None, "ci_high": None, "n": n}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        a = roc_auc(y[idx], s[idx])
        if a is not None:
            vals.append(a)
    if not vals:
        return {"auc": float(observed), "ci_low": None, "ci_high": None, "n": n}
    arr = np.asarray(vals, dtype=np.float64)
    low, high = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "auc": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": n,
        "n_valid_resamples": int(arr.size),
    }


def ols_hc3(
    y: np.ndarray,
    design: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    """Secondary OLS with HC3 robust SE. Returns nulls if statsmodels missing."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    y, x = y[mask], x[mask]
    if sm is None:
        return {"available": False, "error": "statsmodels not installed", "n": int(y.size)}
    if y.size < x.shape[1] + 2:
        return {"available": True, "error": "too few rows for OLS", "n": int(y.size)}
    try:
        model = sm.OLS(y, x, hasconst=True).fit(cov_type="HC3")
        ci = model.conf_int(alpha=0.05)
        coefs = []
        for i, name in enumerate(names):
            coefs.append(
                {
                    "name": name,
                    "coef": float(model.params[i]),
                    "se_hc3": float(model.bse[i]),
                    "p_value": float(model.pvalues[i]),
                    "ci_low": float(ci[i, 0]),
                    "ci_high": float(ci[i, 1]),
                }
            )
        return {
            "available": True,
            "n": int(y.size),
            "rsquared": float(model.rsquared),
            "coefficients": coefs,
            "note": "Secondary, low-powered OLS of per-pair gap rate; HC3 robust SE.",
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "error": f"{type(exc).__name__}: {exc}", "n": int(y.size)}


def group_descriptives(values: np.ndarray, *, seed: int, n_resamples: int) -> dict[str, Any]:
    rates = np.asarray(values, dtype=np.float64)
    summary = summarise_rates(rates)
    boot = bootstrap_mean_ci(rates, n_resamples=n_resamples, seed=seed)
    return {**summary, "bootstrap": boot}
