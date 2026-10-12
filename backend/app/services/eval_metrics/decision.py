"""Win/tie/loss and the frozen SCALE / DIAGNOSE decision rule."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.services.eval_metrics.constants import (
    BOOTSTRAP_RESAMPLES,
    FAVOURABLE_SIGN,
    PRIMARY_METRICS,
    SEED,
)
from app.services.fit_label_control.stats import holm_adjust
from app.services.gap_sensitivity.metrics import (
    bootstrap_mean_ci,
    matched_pairs_rank_biserial,
    wilcoxon_paired,
)


def favourable_differences(
    treatment: np.ndarray,
    baseline: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Positive = favourable to treatment under the frozen sign convention."""
    t = np.asarray(treatment, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    sign = FAVOURABLE_SIGN[metric]
    # sign=+1: higher better → t - b; sign=-1: lower better → b - t
    return sign * (t - b)


def bootstrap_diff_ci(
    diffs: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
) -> dict[str, float | None]:
    return bootstrap_mean_ci(diffs, n_resamples=n_resamples, seed=seed)


def win_tie_loss(ci: dict[str, float | None]) -> str:
    low, high = ci.get("ci_low"), ci.get("ci_high")
    if low is None or high is None:
        return "tie"
    if low > 0:
        return "win"
    if high < 0:
        return "lose"
    return "tie"


def compare_metric(
    treatment: np.ndarray,
    baseline: np.ndarray,
    metric: str,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
) -> dict[str, Any]:
    t = np.asarray(treatment, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(b)
    t, b = t[mask], b[mask]
    diffs = favourable_differences(t, b, metric)
    ci = bootstrap_diff_ci(diffs, n_resamples=n_resamples, seed=seed)
    # Wilcoxon on raw values with direction matching favourable?
    # Spec: Wilcoxon signed-rank p on the comparison; we use treatment vs baseline
    # for higher-is-better and baseline vs treatment conceptually via diffs.
    # Use wilcoxon on favourable series vs zeros via paired t/b with sign flip:
    if FAVOURABLE_SIGN[metric] > 0:
        w = wilcoxon_paired(t, b)
    else:
        w = wilcoxon_paired(b, t)
    outcome = win_tie_loss(ci)
    return {
        "metric": metric,
        "n_pairs": int(t.size),
        "mean_favourable_diff": ci.get("mean"),
        "ci_low": ci.get("ci_low"),
        "ci_high": ci.get("ci_high"),
        "wilcoxon_p": w.get("p_value"),
        "rank_biserial": w.get("rank_biserial")
        if w.get("rank_biserial") is not None
        else matched_pairs_rank_biserial(diffs),
        "outcome": outcome,
    }


def compare_conditions(
    treatment_by_metric: dict[str, np.ndarray],
    baseline_by_metric: dict[str, np.ndarray],
    baseline_name: str,
) -> dict[str, Any]:
    comparisons = []
    for m in PRIMARY_METRICS:
        if m not in treatment_by_metric or m not in baseline_by_metric:
            comparisons.append(
                {
                    "metric": m,
                    "baseline": baseline_name,
                    "n_pairs": 0,
                    "mean_favourable_diff": None,
                    "ci_low": None,
                    "ci_high": None,
                    "wilcoxon_p": None,
                    "rank_biserial": None,
                    "outcome": "tie",
                }
            )
            continue
        row = compare_metric(
            treatment_by_metric[m],
            baseline_by_metric[m],
            m,
        )
        row["baseline"] = baseline_name
        comparisons.append(row)
    pvals = [c.get("wilcoxon_p") for c in comparisons]
    # Holm over all 12 is applied externally; here per-baseline raw p only
    return {"baseline": baseline_name, "comparisons": comparisons, "raw_p": pvals}


def apply_holm_across(comparisons_12: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pvals = [c.get("wilcoxon_p") for c in comparisons_12]
    adj = holm_adjust(pvals)
    out = []
    for c, p_adj in zip(comparisons_12, adj, strict=True):
        row = dict(c)
        row["wilcoxon_p_holm"] = p_adj
        out.append(row)
    return out


def decision_rule(
    *,
    outcomes_vs_b1: dict[str, str],
    outcomes_vs_b2: dict[str, str],
    outcomes_vs_b0: dict[str, str],
    projected_cost_cny: float | None,
    scale_budget_cny: float,
) -> dict[str, Any]:
    """Apply the frozen SCALE / DIAGNOSE rule. Returns literal clause outcomes."""

    def count(outcomes: dict[str, str]) -> tuple[int, int]:
        wins = sum(1 for m in PRIMARY_METRICS if outcomes.get(m) == "win")
        losses = sum(1 for m in PRIMARY_METRICS if outcomes.get(m) == "lose")
        return wins, losses

    w1, l1 = count(outcomes_vs_b1)
    clause1 = {
        "id": 1,
        "text": "vs B1: >=2 wins among four primaries and 0 losses",
        "wins": w1,
        "losses": l1,
        "holds": w1 >= 2 and l1 == 0,
    }

    w2_target = sum(
        1 for m in ("M2.P", "M3.P") if outcomes_vs_b2.get(m) == "win"
    )
    _, l2 = count(outcomes_vs_b2)
    clause2 = {
        "id": 2,
        "text": "vs B2: >=1 win among {M2.P, M3.P} and 0 losses among four",
        "wins_M2_M3": w2_target,
        "losses": l2,
        "holds": w2_target >= 1 and l2 == 0,
    }

    w0, l0 = count(outcomes_vs_b0)
    clause3 = {
        "id": 3,
        "text": "vs B0: >=3 wins and 0 losses",
        "wins": w0,
        "losses": l0,
        "holds": w0 >= 3 and l0 == 0,
    }

    if projected_cost_cny is None:
        clause4 = {
            "id": 4,
            "text": "projected 150-pair cost within scale budget (CNY)",
            "projected_cost_cny": None,
            "scale_budget_cny": scale_budget_cny,
            "holds": False,
            "note": "cost unavailable",
        }
        budget_blocked = False
    else:
        within = float(projected_cost_cny) <= float(scale_budget_cny)
        clause4 = {
            "id": 4,
            "text": "projected 150-pair cost within scale budget (CNY)",
            "projected_cost_cny": float(projected_cost_cny),
            "scale_budget_cny": float(scale_budget_cny),
            "holds": within,
        }
        budget_blocked = not within

    scientific = clause1["holds"] and clause2["holds"] and clause3["holds"]
    if scientific and budget_blocked:
        verdict = "SCALE_BLOCKED_BY_BUDGET"
    elif scientific and clause4["holds"]:
        verdict = "SCALE"
    else:
        verdict = "DIAGNOSE"

    return {
        "verdict": verdict,
        "clauses": [clause1, clause2, clause3, clause4],
        "advisory": True,
        "scientific_scale_criteria_met": scientific,
    }
