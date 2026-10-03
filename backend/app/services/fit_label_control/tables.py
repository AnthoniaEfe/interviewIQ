"""Group metrics, hypothesis tests, decision rule, OLS confound audit."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from app.services.fit_label_control.constants import (
    BOOTSTRAP_RESAMPLES,
    JT_PERMUTATIONS,
    LABEL_ORDER,
    ORDINAL,
    SEED,
)
from app.services.fit_label_control.stats import (
    bootstrap_auc_ci,
    bootstrap_cliffs_delta,
    group_descriptives,
    holm_adjust,
    jonckheere_terpstra,
    kruskal_wallis,
    mann_whitney,
    ols_hc3,
    roc_auc,
    roc_curve_points,
    spearman_ordinal,
)


def _by_label(rows: list[dict], field: str) -> dict[str, np.ndarray]:
    buckets: dict[str, list[float]] = {k: [] for k in LABEL_ORDER}
    for row in rows:
        lab = row.get("fit_label")
        if lab not in buckets:
            continue
        val = row.get(field)
        if val is None:
            continue
        buckets[str(lab)].append(float(val))
    return {k: np.asarray(v, dtype=np.float64) for k, v in buckets.items()}


def label_descriptives(rows: list[dict], *, seed: int = SEED) -> list[dict[str, Any]]:
    out = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        lab = row.get("fit_label")
        if lab:
            grouped[str(lab)].append(row)
    for lab in LABEL_ORDER:
        items = grouped.get(lab, [])
        n_req = sum(int(r["n_requirements"]) for r in items)
        n_strong = sum(int(r["n_strong"]) for r in items)
        n_partial = sum(int(r["n_partial"]) for r in items)
        n_weak = sum(int(r["n_weak"]) for r in items)
        n_missing = sum(int(r["n_missing"]) for r in items)
        rates = np.asarray([r["gap_rate"] for r in items], dtype=np.float64)
        desc = group_descriptives(rates, seed=seed, n_resamples=BOOTSTRAP_RESAMPLES)
        pw = [r["priority_weighted_gap_share"] for r in items if r.get("priority_weighted_gap_share") is not None]
        out.append(
            {
                "fit_label": lab,
                "n_pairs": len(items),
                "n_requirements": n_req,
                "strong_share": (n_strong / n_req) if n_req else None,
                "partial_share_req": (n_partial / n_req) if n_req else None,
                "weak_share": (n_weak / n_req) if n_req else None,
                "missing_share": (n_missing / n_req) if n_req else None,
                "strength_share": (n_strong / n_req) if n_req else None,
                "partial_share": (n_partial / n_req) if n_req else None,
                "gap_share": ((n_weak + n_missing) / n_req) if n_req else None,
                "mean_per_pair_gap_rate": desc["mean"],
                "median_per_pair_gap_rate": desc["median"],
                "iqr_per_pair_gap_rate": desc["iqr"],
                "gap_rate_ci_low": desc["bootstrap"]["ci_low"],
                "gap_rate_ci_high": desc["bootstrap"]["ci_high"],
                "mean_combined": _mean_field(items, "mean_combined"),
                "mean_semantic": _mean_field(items, "mean_semantic"),
                "mean_tfidf": _mean_field(items, "mean_tfidf"),
                "mean_bm25_norm": _mean_field(items, "mean_bm25_norm"),
                "priority_weighted_gap_share": float(np.mean(pw)) if pw else None,
                "mean_requirements_per_pair": _mean_field(items, "n_requirements"),
                "mean_evidence_strings": _mean_field(items, "evidence_strings"),
            }
        )
    return out


def _mean_field(items: list[dict], field: str) -> float | None:
    vals = [r[field] for r in items if r.get(field) is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def gradient_tests(rows: list[dict], field: str = "gap_rate") -> dict[str, Any]:
    by = _by_label(rows, field)
    groups = [by[k] for k in LABEL_ORDER]
    pairs = [
        ("No Fit", "Potential Fit"),
        ("No Fit", "Good Fit"),
        ("Potential Fit", "Good Fit"),
    ]
    raw = []
    for a, b in pairs:
        # For gap rate, expect a > b when a is worse fit. MW two-sided.
        mw = mann_whitney(by[a], by[b], alternative="two-sided")
        cd = bootstrap_cliffs_delta(
            by[a], by[b], n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
        )
        mean_a = float(np.mean(by[a])) if by[a].size else None
        mean_b = float(np.mean(by[b])) if by[b].size else None
        direction_ok = (
            mean_a is not None
            and mean_b is not None
            and mean_a > mean_b
            if field == "gap_rate"
            else mean_a is not None and mean_b is not None and mean_a < mean_b
        )
        raw.append(
            {
                "comparison": f"{a} vs {b}",
                "mann_whitney": mw,
                "cliffs_delta": cd,
                "mean_left": mean_a,
                "mean_right": mean_b,
                "direction_ok_for_h1": direction_ok,
            }
        )
    pvals = [r["mann_whitney"]["p_value"] for r in raw]
    adj = holm_adjust(pvals)
    for r, p_adj in zip(raw, adj, strict=True):
        r["holm_p"] = p_adj
    # JT: Good < Potential < No Fit on gap rate
    jt = jonckheere_terpstra(
        groups,
        n_permutations=JT_PERMUTATIONS,
        seed=SEED,
        alternative="greater" if field == "gap_rate" else "less",
    )
    ordinal = []
    values = []
    for row in rows:
        lab = row.get("fit_label")
        if lab not in ORDINAL or row.get(field) is None:
            continue
        ordinal.append(ORDINAL[str(lab)])
        values.append(float(row[field]))
    sp = spearman_ordinal(np.asarray(ordinal), np.asarray(values))
    adjacent_ok = all(
        r["direction_ok_for_h1"]
        for r in raw
        if r["comparison"] in {"No Fit vs Potential Fit", "Potential Fit vs Good Fit"}
    )
    jt_sig = jt.get("p_value") is not None and jt["p_value"] < 0.05
    return {
        "kruskal_wallis": kruskal_wallis(groups),
        "pairwise": raw,
        "jonckheere_terpstra": jt,
        "spearman_ordinal": sp,
        "h1_monotonic_claim_allowed": bool(jt_sig and adjacent_ok),
    }


def discrimination(rows: list[dict], field: str) -> dict[str, Any]:
    nf, good, pot = [], [], []
    y_ng, s_ng = [], []
    y_np, s_np = [], []
    for row in rows:
        lab = row.get("fit_label")
        val = row.get(field)
        if val is None or lab not in ORDINAL:
            continue
        v = float(val)
        # higher gap -> No Fit; invert combined/strength
        score = v
        if field in {"mean_combined", "strength_share", "mean_semantic"}:
            score = -v
        if lab == "No Fit":
            nf.append(v)
            y_ng.append(1)
            s_ng.append(score)
            y_np.append(1)
            s_np.append(score)
        elif lab == "Good Fit":
            good.append(v)
            y_ng.append(0)
            s_ng.append(score)
        if lab in {"Good Fit", "Potential Fit"}:
            y_np.append(0)
            s_np.append(score)
            if lab == "Potential Fit":
                pot.append(v)
    return {
        "field": field,
        "no_fit_vs_good": {
            **bootstrap_auc_ci(
                np.asarray(y_ng), np.asarray(s_ng), n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
            ),
            "roc": roc_curve_points(np.asarray(y_ng), np.asarray(s_ng)),
        },
        "no_fit_vs_fit_positive": {
            **bootstrap_auc_ci(
                np.asarray(y_np), np.asarray(s_np), n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
            ),
            "roc": roc_curve_points(np.asarray(y_np), np.asarray(s_np)),
        },
    }


def confound_audit(rows: list[dict]) -> dict[str, Any]:
    fields = {
        "jd_length": "JD length (chars)",
        "n_requirements": "requirements per pair",
        "share_required": "share required vs preferred",
        "cv_length": "CV length (chars)",
        "evidence_strings": "evidence strings per pair",
    }
    tests = {}
    for field, title in fields.items():
        by = _by_label(rows, field)
        tests[field] = {
            "title": title,
            "kruskal_wallis": kruskal_wallis([by[k] for k in LABEL_ORDER]),
            "means": {
                k: (float(np.mean(by[k])) if by[k].size else None) for k in LABEL_ORDER
            },
        }
    mm = _by_label(rows, "mismatch_gap_rate")
    tests["mismatch_gap_rate"] = {
        "title": "random-candidate (mismatched) gap rate",
        "kruskal_wallis": kruskal_wallis([mm[k] for k in LABEL_ORDER]),
        "means": {k: (float(np.mean(mm[k])) if mm[k].size else None) for k in LABEL_ORDER},
        "note": (
            "With mismatched evidence the pair fit label is meaningless. "
            "A significant Kruskal-Wallis indicates a job-side confound."
        ),
    }
    margins = _by_label(rows, "true_minus_mismatch_gap")
    tests["random_candidate_margin"] = {
        "title": "true minus random-candidate gap share",
        "means": {
            k: (float(np.mean(margins[k])) if margins[k].size else None)
            for k in LABEL_ORDER
        },
    }

    y = []
    design = []
    for row in rows:
        lab = row.get("fit_label")
        if lab not in ORDINAL or row.get("gap_rate") is None:
            continue
        if (
            row.get("n_requirements") is None
            or row.get("evidence_strings") is None
            or row.get("jd_length") is None
        ):
            continue
        y.append(float(row["gap_rate"]))
        # intercept, Potential dummy, No Fit dummy, n_req, evidence, jd_len
        design.append(
            [
                1.0,
                1.0 if lab == "Potential Fit" else 0.0,
                1.0 if lab == "No Fit" else 0.0,
                float(row["n_requirements"]),
                float(row["evidence_strings"]),
                float(row["jd_length"]),
            ]
        )
    ols = ols_hc3(
        np.asarray(y),
        np.asarray(design) if design else np.zeros((0, 6)),
        [
            "intercept_GoodFit",
            "PotentialFit_vs_Good",
            "NoFit_vs_Good",
            "n_requirements",
            "evidence_strings",
            "jd_length",
        ],
    )
    return {"kruskal": tests, "ols_hc3_secondary": ols}


def apply_decision_rule(
    primary_grad: dict,
    holdout_grad: dict,
    primary_disc: dict,
    holdout_disc: dict,
) -> dict[str, Any]:
    def nf_vs_good(grad: dict, disc: dict) -> dict[str, Any]:
        pair = next(
            (p for p in grad["pairwise"] if p["comparison"] == "No Fit vs Good Fit"),
            None,
        )
        holm = pair["holm_p"] if pair else None
        auc = disc["no_fit_vs_good"].get("auc")
        sig = holm is not None and holm < 0.05 and auc is not None and auc >= 0.65
        return {"holm_p": holm, "auc": auc, "significant_coarse": sig}

    p = nf_vs_good(primary_grad, primary_disc)
    h = nf_vs_good(holdout_grad, holdout_disc)
    if p["significant_coarse"] and h["significant_coarse"]:
        outcome = "the gap matrix tracks human fit labels at the coarse level"
    elif p["significant_coarse"] or h["significant_coarse"]:
        outcome = "partial/inconsistent support"
    else:
        outcome = (
            "the gap matrix does not separate fit labels; possible reasons: "
            "label noise, extraction, scoring; do not claim validation"
        )
    h1_ok = bool(
        primary_grad.get("h1_monotonic_claim_allowed")
        and holdout_grad.get("h1_monotonic_claim_allowed")
    )
    return {
        "primary": p,
        "holdout": h,
        "outcome": outcome,
        "h1_monotonic_claimed": False,
        "h1_monotonic_allowed_both_splits": h1_ok,
        "h1_note": (
            "H1 monotonicity is claimed only if JT and both adjacent differences "
            "point the right way. This flag is computed; the report must not claim "
            "H1 unless allowed."
        ),
    }


def bundle_condition(rows: list[dict]) -> dict[str, Any]:
    desc = label_descriptives(rows)
    grad_gap = gradient_tests(rows, "gap_rate")
    disc = {
        "gap_rate": discrimination(rows, "gap_rate"),
        "strength_share": discrimination(rows, "strength_share"),
        "mean_combined": discrimination(rows, "mean_combined"),
    }
    return {
        "descriptives": desc,
        "gradient_gap_rate": grad_gap,
        "discrimination": disc,
        "confound": confound_audit(rows),
        "n_pairs": len(rows),
    }
