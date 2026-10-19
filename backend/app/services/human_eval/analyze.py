"""Pre-registered human-rating analysis. Refuses demo, pilot, and incomplete exports."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import binomtest, wilcoxon

from app.services.classifier_eval.metrics import holm_adjust

MIN_COMPLETION = 0.90
CRITERIA = ("relevance", "coverage", "non_redundancy")


class HumanEvalAnalysisError(RuntimeError):
    pass


def krippendorff_alpha_ordinal(matrix: list[list[float | None]]) -> float | None:
    """Ordinal Krippendorff's alpha. Columns are items, rows are raters."""
    columns: list[list[float]] = []
    for col in range(0 if not matrix else len(matrix[0])):
        values = []
        for row in matrix:
            value = row[col] if col < len(row) else None
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                values.append(float(value))
        if len(values) >= 2:
            columns.append(values)
    if len(columns) < 2:
        return None
    levels = sorted({value for column in columns for value in column})
    index = {level: i for i, level in enumerate(levels)}
    coincidence = np.zeros((len(levels), len(levels)), dtype=float)
    for column in columns:
        counts: dict[float, int] = {}
        for value in column:
            counts[value] = counts.get(value, 0) + 1
        m = len(column)
        for left, n_left in counts.items():
            for right, n_right in counts.items():
                pairs = n_left * (n_right - 1 if left == right else n_right)
                coincidence[index[left], index[right]] += pairs / (m - 1)
    observed = float(coincidence.sum())
    if observed <= 1:
        return None
    marginal = coincidence.sum(axis=1)
    metric = np.zeros_like(coincidence)
    for i in range(len(levels)):
        for j in range(len(levels)):
            if i == j:
                continue
            lo, hi = (i, j) if i < j else (j, i)
            span = float(marginal[lo : hi + 1].sum())
            span -= (marginal[i] + marginal[j]) / 2.0
            metric[i, j] = span ** 2
    do = float(np.sum(coincidence * metric) / observed)
    de = float(np.sum(np.outer(marginal, marginal) * metric) / (observed * (observed - 1)))
    if de == 0:
        return 1.0 if do == 0 else None
    return float(1.0 - (do / de))


def _holm_ready(p_values: list[float]) -> list[float]:
    return holm_adjust(p_values)


def completion_rate(ratings: list[dict[str, Any]], n_items: int) -> dict[str, float]:
    by_rater: dict[str, set[str]] = defaultdict(set)
    for row in ratings:
        if _complete(row):
            by_rater[str(row["rater_id"])].add(str(row["item_id"]))
    if not by_rater or n_items <= 0:
        return {}
    return {rater: len(items) / n_items for rater, items in by_rater.items()}


def _complete(row: dict[str, Any]) -> bool:
    for side in ("x", "y"):
        for criterion in CRITERIA:
            value = row.get(f"{criterion}_{side}")
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= int(value) <= 5:
                return False
    return row.get("preference") in {"X", "Y", "no_preference"}


def analyze_ratings(
    ratings: list[dict[str, Any]],
    condition_key: dict[str, dict[str, str]],
    *,
    mode: str,
    n_items: int,
) -> dict[str, Any]:
    if mode != "main":
        raise HumanEvalAnalysisError(
            f"main analysis refuses mode {mode!r}. Pilot and demo stores are excluded."
        )
    if any(row.get("is_demo") or row.get("mode") == "demo" for row in ratings):
        raise HumanEvalAnalysisError("demo ratings are mixed into the export")
    rates = completion_rate(ratings, n_items)
    if not rates or any(rate < MIN_COMPLETION for rate in rates.values()):
        raise HumanEvalAnalysisError(
            f"completion below {MIN_COMPLETION:.0%} for at least one rater: {rates or 'no ratings'}"
        )
    agreement = _agreement(ratings, n_items)
    effects = _condition_effects(ratings, condition_key)
    raters = sorted({str(row["rater_id"]) for row in ratings})
    sensitivity = []
    if len(raters) >= 3:
        for held_out in raters:
            subset = [row for row in ratings if str(row["rater_id"]) != held_out]
            sensitivity.append(
                {
                    "held_out": held_out,
                    "effects": _condition_effects(subset, condition_key),
                }
            )
    return {
        "mode": mode,
        "min_completion": MIN_COMPLETION,
        "completion_by_rater": rates,
        "agreement": agreement,
        "effects": effects,
        "leave_one_rater_out": sensitivity,
    }


def _agreement(ratings: list[dict[str, Any]], n_items: int) -> dict[str, Any]:
    raters = sorted({str(row["rater_id"]) for row in ratings})
    items = sorted({str(row["item_id"]) for row in ratings})
    by_key = {(str(row["rater_id"]), str(row["item_id"])): row for row in ratings}
    out: dict[str, Any] = {}
    for criterion in CRITERIA:
        for side, label in (("x", "X"), ("y", "Y")):
            matrix = []
            for rater in raters:
                matrix.append(
                    [
                        _as_float(by_key.get((rater, item_id), {}).get(f"{criterion}_{side}"))
                        for item_id in items
                    ]
                )
            out[f"{criterion}_{label}"] = krippendorff_alpha_ordinal(matrix)
    out["n_raters"] = len(raters)
    out["n_items_seen"] = len(items)
    out["n_items_assigned"] = n_items
    return out


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return float(value)


def _condition_effects(
    ratings: list[dict[str, Any]],
    condition_key: dict[str, dict[str, str]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ratings:
        grouped[(str(row["item_id"]), str(row["rater_id"]))].append(row)
    # latest row wins; caller should already have collapsed, but be safe
    latest = {key: rows[-1] for key, rows in grouped.items()}
    pair_means: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {"T": defaultdict(list), "B1": defaultdict(list)}
    )
    preferences: list[str] = []
    for (item_id, _rater), row in latest.items():
        sides = condition_key.get(item_id)
        if not sides:
            raise HumanEvalAnalysisError(f"condition key missing item {item_id}")
        for side_name, prefix in (("X", "x"), ("Y", "y")):
            condition = sides.get(side_name)
            if condition not in {"T", "B1"}:
                raise HumanEvalAnalysisError(f"bad condition label for {item_id}")
            for criterion in CRITERIA:
                value = row.get(f"{criterion}_{prefix}")
                pair_means[item_id][condition][criterion].append(float(value))
        winner = _preference_condition(row.get("preference"), sides)
        if winner is not None:
            preferences.append(winner)
    criteria_out = {}
    p_values = []
    names = []
    for criterion in CRITERIA:
        diffs = []
        for item_id, sides in pair_means.items():
            t_vals = sides["T"][criterion]
            b_vals = sides["B1"][criterion]
            if not t_vals or not b_vals:
                continue
            diffs.append(float(np.mean(t_vals) - np.mean(b_vals)))
        stat = _wilcoxon(diffs)
        criteria_out[criterion] = stat
        p_values.append(stat["p_value"] if stat["p_value"] is not None else 1.0)
        names.append(criterion)
    holm = _holm_ready(p_values)
    for name, adjusted in zip(names, holm, strict=True):
        criteria_out[name]["p_holm"] = adjusted
    n_pref = len(preferences)
    n_t = sum(1 for winner in preferences if winner == "T")
    if n_pref == 0:
        preference = {"n": 0, "n_t": 0, "p_value": None, "note": "no decisive preferences"}
    else:
        test = binomtest(n_t, n_pref, 0.5, alternative="two-sided")
        preference = {"n": n_pref, "n_t": n_t, "n_b1": n_pref - n_t, "p_value": float(test.pvalue)}
    return {"criteria": criteria_out, "preference": preference}


def _preference_condition(preference: Any, sides: dict[str, str]) -> str | None:
    if preference == "no_preference" or preference in (None, ""):
        return None
    if preference not in {"X", "Y"}:
        return None
    return sides[preference]


def _wilcoxon(diffs: list[float]) -> dict[str, Any]:
    array = np.asarray(diffs, dtype=float)
    if array.size == 0:
        return {"n": 0, "mean_diff": None, "p_value": None, "rank_biserial": None}
    mean = float(array.mean())
    nonzero = array[array != 0]
    if nonzero.size == 0:
        return {"n": int(array.size), "mean_diff": mean, "p_value": 1.0, "rank_biserial": 0.0}
    result = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
    # Rank-biserial from the signed-rank statistic: r = 1 - (2W)/(n(n+1)/ something)
    # Use the common form r_rb = Z / sqrt(N) is not rank-biserial.
    # Kerby simple difference: proportion of positive pairs minus negative, for matched pairs
    # of the ranks is heavier. Report mean diff and the sign-based rank-biserial:
    n = int(nonzero.size)
    # scipy statistic is the smaller of the two rank sums when method is auto in older scipy.
    # Use a direct rank-biserial: (n_pos - n_neg) / n as a simple effect, plus mean.
    n_pos = int(np.sum(nonzero > 0))
    n_neg = int(np.sum(nonzero < 0))
    rank_biserial = (n_pos - n_neg) / n
    return {
        "n": int(array.size),
        "n_nonzero": n,
        "mean_diff_T_minus_B1": mean,
        "p_value": float(result.pvalue),
        "rank_biserial_sign": float(rank_biserial),
    }
