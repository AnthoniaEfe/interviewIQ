"""G1–G4 calculations for gold-set phase 2.

Rules are the frozen statements in docs/gold_set_protocol.md. The search grid
below is the analysis specification used for G4. It is fixed in this module
and is not revised after the gold-set summary is written.

G4 direction, matching the rest of this project: tune on primary-30, confirm
on holdout-30. The reverse direction is stored as sensitivity and is not an
adoption test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from app.services.gold_set.analyze_lib import (
    bootstrap_metric_ci,
    quadratic_weighted_kappa,
)
from app.services.gold_set.constants import BOOTSTRAP, STATS_SEED

PROD_WEIGHTS = (0.25, 0.25, 0.50)
PROD_PARTIAL = 0.40
PROD_STRONG = 0.65
WEIGHT_STEP = 0.25
PARTIAL_GRID = (0.30, 0.35, 0.40, 0.45, 0.50)
STRONG_GRID = (0.55, 0.60, 0.65, 0.70, 0.75)
MIN_SPLIT_N = 8


def landis_koch(kappa: float | None) -> str | None:
    """Descriptive convention only. Not a decision threshold."""
    if kappa is None or not np.isfinite(kappa):
        return None
    if kappa < 0:
        return "poor"
    if kappa <= 0.20:
        return "slight"
    if kappa <= 0.40:
        return "fair"
    if kappa <= 0.60:
        return "moderate"
    if kappa <= 0.80:
        return "substantial"
    return "almost_perfect"


def weight_grid(step: float = WEIGHT_STEP) -> list[tuple[float, float, float]]:
    ticks = [round(i * step, 10) for i in range(int(round(1 / step)) + 1)]
    grid: list[tuple[float, float, float]] = []
    for tfidf in ticks:
        for bm25 in ticks:
            semantic = round(1.0 - tfidf - bm25, 10)
            if semantic < -1e-9:
                continue
            if abs(semantic / step - round(semantic / step)) > 1e-6:
                continue
            grid.append((tfidf, bm25, max(0.0, semantic)))
    if PROD_WEIGHTS not in grid:
        grid.append(PROD_WEIGHTS)
    return grid


def combine(tfidf: float, bm25: float, semantic: float, weights: tuple[float, float, float]) -> float:
    return weights[0] * tfidf + weights[1] * bm25 + weights[2] * semantic


def three_category(score: float, partial_at: float, strong_at: float) -> int:
    if score >= strong_at:
        return 2
    if score >= partial_at:
        return 1
    return 0


def production_three(score: float) -> int:
    return three_category(score, PROD_PARTIAL, PROD_STRONG)


def adoption_decision(delta: float | None, ci_low: float | None, ci_high: float | None) -> bool:
    """Adopt only when the untouched-split improvement CI lies entirely above 0."""
    if delta is None or ci_low is None or ci_high is None:
        return False
    return bool(delta > 0 and ci_low > 0)


def _spearman_rho(score: np.ndarray, human: np.ndarray) -> float | None:
    if score.size < 3 or np.unique(human).size < 2 or np.unique(score).size < 2:
        return None
    from scipy.stats import spearmanr

    rho, _ = spearmanr(score, human)
    if rho is None or not np.isfinite(rho):
        return None
    return float(rho)


def _spearman_p(score: np.ndarray, human: np.ndarray) -> float | None:
    if score.size < 3 or np.unique(human).size < 2 or np.unique(score).size < 2:
        return None
    from scipy.stats import spearmanr

    rho, p = spearmanr(score, human)
    if p is None or not np.isfinite(p) or rho is None or not np.isfinite(rho):
        return None
    return float(p)


def _auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if y.size < 2 or len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return None


def _roc_points(y: np.ndarray, score: np.ndarray) -> dict[str, list[float]]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if y.size < 2 or len(np.unique(y)) < 2:
        return {"fpr": [], "tpr": []}
    fpr, tpr, _ = roc_curve(y, score)
    return {"fpr": [float(v) for v in fpr], "tpr": [float(v) for v in tpr]}


def _ci_excludes_zero_above(ci: dict[str, Any] | None) -> bool:
    if not ci:
        return False
    lo = ci.get("ci_low")
    val = ci.get("value")
    return lo is not None and val is not None and lo > 0 and val > 0


def _labels_for(
    rows: list[dict],
    weights: tuple[float, float, float],
    partial_at: float,
    strong_at: float,
) -> np.ndarray:
    scores = np.asarray(
        [combine(r["tfidf"], r["bm25_norm"], r["semantic"], weights) for r in rows],
        dtype=float,
    )
    return np.asarray([three_category(s, partial_at, strong_at) for s in scores], dtype=int)


def _config_distance(
    weights: tuple[float, float, float],
    partial_at: float,
    strong_at: float,
) -> float:
    return (
        abs(weights[0] - PROD_WEIGHTS[0])
        + abs(weights[1] - PROD_WEIGHTS[1])
        + abs(weights[2] - PROD_WEIGHTS[2])
        + abs(partial_at - PROD_PARTIAL)
        + abs(strong_at - PROD_STRONG)
    )


def _select_config(rows: list[dict]) -> dict[str, Any]:
    human = np.asarray([r["consensus"] for r in rows], dtype=int)
    best: dict[str, Any] | None = None
    n_evaluated = 0
    for weights in weight_grid():
        for partial_at in PARTIAL_GRID:
            for strong_at in STRONG_GRID:
                if strong_at <= partial_at:
                    continue
                pred = _labels_for(rows, weights, partial_at, strong_at)
                try:
                    qwk = quadratic_weighted_kappa(human, pred)
                except ValueError:
                    qwk = None
                if qwk is not None and not np.isfinite(qwk):
                    qwk = None
                n_evaluated += 1
                cand = {
                    "weights": {"tfidf": weights[0], "bm25_norm": weights[1], "semantic": weights[2]},
                    "partial_threshold": partial_at,
                    "strong_threshold": strong_at,
                    "qwk_tune": qwk,
                    "distance_to_production": _config_distance(weights, partial_at, strong_at),
                    "on_grid_boundary": (
                        partial_at in {PARTIAL_GRID[0], PARTIAL_GRID[-1]}
                        or strong_at in {STRONG_GRID[0], STRONG_GRID[-1]}
                        or min(weights) <= 0.0
                        or max(weights) >= 1.0
                    ),
                }
                if qwk is None:
                    continue
                if best is None:
                    best = cand
                    continue
                better = qwk > float(best["qwk_tune"]) + 1e-12
                tied = abs(qwk - float(best["qwk_tune"])) <= 1e-12
                closer = cand["distance_to_production"] < best["distance_to_production"] - 1e-12
                if better or (tied and closer):
                    best = cand
    if best is None:
        best = {
            "weights": {"tfidf": PROD_WEIGHTS[0], "bm25_norm": PROD_WEIGHTS[1], "semantic": PROD_WEIGHTS[2]},
            "partial_threshold": PROD_PARTIAL,
            "strong_threshold": PROD_STRONG,
            "qwk_tune": None,
            "distance_to_production": 0.0,
            "on_grid_boundary": False,
        }
    best["n_configs_evaluated"] = n_evaluated
    best["is_production"] = (
        abs(best["weights"]["tfidf"] - PROD_WEIGHTS[0]) < 1e-9
        and abs(best["weights"]["bm25_norm"] - PROD_WEIGHTS[1]) < 1e-9
        and abs(best["weights"]["semantic"] - PROD_WEIGHTS[2]) < 1e-9
        and abs(best["partial_threshold"] - PROD_PARTIAL) < 1e-9
        and abs(best["strong_threshold"] - PROD_STRONG) < 1e-9
    )
    return best


def _qwk_delta(human: np.ndarray, production: np.ndarray, candidate: np.ndarray) -> float | None:
    base = quadratic_weighted_kappa(human, production)
    alt = quadratic_weighted_kappa(human, candidate)
    if base is None or alt is None:
        return None
    return float(alt - base)


def _bootstrap_delta(
    human: np.ndarray,
    production: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    observed = _qwk_delta(human, production, candidate)
    n = int(human.size)
    point = {
        "value": observed,
        "ci_low": None,
        "ci_high": None,
        "n": n,
        "qwk_production": quadratic_weighted_kappa(human, production),
        "qwk_candidate": quadratic_weighted_kappa(human, candidate),
    }
    if observed is None or n < 2:
        return point
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        # A resample that collapses a label makes kappa undefined; skip it.
        if np.unique(human[idx]).size < 2:
            continue
        delta = _qwk_delta(human[idx], production[idx], candidate[idx])
        if delta is not None and np.isfinite(delta):
            vals.append(delta)
    if not vals:
        return point
    lo, hi = np.percentile(np.asarray(vals, dtype=float), [2.5, 97.5])
    point["ci_low"] = float(lo)
    point["ci_high"] = float(hi)
    point["n_valid_resamples"] = len(vals)
    return point


def _signal_block(rows: list[dict], *, seed: int) -> dict[str, Any]:
    human = np.asarray([r["consensus"] for r in rows], dtype=int)
    binary = (human >= 1).astype(int)
    columns = {
        "combined": np.asarray([r["combined"] for r in rows], dtype=float),
        "tfidf": np.asarray([r["tfidf"] for r in rows], dtype=float),
        "bm25_norm": np.asarray([r["bm25_norm"] for r in rows], dtype=float),
        "semantic": np.asarray([r["semantic"] for r in rows], dtype=float),
    }
    aucs: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    for i, (name, score) in enumerate(columns.items()):
        aucs[name] = bootstrap_metric_ci(
            binary,
            score,
            lambda y, s: _auc(y, s),
            n_resamples=BOOTSTRAP,
            seed=seed + i,
        )
        curves[name] = _roc_points(binary, score)
    hybrid = columns["combined"]
    beats: dict[str, Any] = {}
    y = binary
    for i, name in enumerate(("tfidf", "bm25_norm", "semantic")):
        signal = columns[name]
        a0 = _auc(y, hybrid)
        b0 = _auc(y, signal)
        observed = None if a0 is None or b0 is None else float(a0 - b0)
        rng = np.random.default_rng(seed + 20 + i)
        vals: list[float] = []
        n = int(y.size)
        if observed is not None and n >= 2:
            for _ in range(BOOTSTRAP):
                idx = rng.integers(0, n, size=n)
                if np.unique(y[idx]).size < 2:
                    continue
                a = _auc(y[idx], hybrid[idx])
                b = _auc(y[idx], columns[name][idx])
                if a is not None and b is not None:
                    vals.append(a - b)
        ci_low = ci_high = None
        if vals:
            ci_low, ci_high = [float(v) for v in np.percentile(np.asarray(vals), [2.5, 97.5])]
        beats[name] = {
            "delta_auc": observed,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "beats": observed is not None and ci_low is not None and observed > 0 and ci_low > 0,
        }
    hybrid_ci = aucs["combined"]
    return {
        "n": int(binary.size),
        "n_positive": int(binary.sum()),
        "n_negative": int(binary.size - binary.sum()),
        "auc": aucs,
        "roc": curves,
        "hybrid_separates": _ci_excludes_zero_above(
            {
                "value": None if hybrid_ci.get("value") is None else hybrid_ci["value"] - 0.5,
                "ci_low": None if hybrid_ci.get("ci_low") is None else hybrid_ci["ci_low"] - 0.5,
            }
        ),
        "hybrid_beats": beats,
    }


def _false_gaps(rows: list[dict]) -> dict[str, Any]:
    gaps = [r for r in rows if r["system3"] == 0 and r["consensus"] >= 1]
    missed = [r for r in rows if r["system3"] >= 1 and r["consensus"] == 0]
    not_extracted = []
    scored_low = []
    for row in gaps:
        evidence = str(row.get("evidence_string") or "").strip()
        if evidence:
            scored_low.append(row)
        else:
            not_extracted.append(row)
    n = len(rows)
    n_system_gap = sum(1 for r in rows if r["system3"] == 0)

    def _brief(row: dict) -> dict[str, Any]:
        text = str(row.get("requirement_text") or "").replace("\n", " ").strip()
        return {
            "item_id": row["item_id"],
            "split": row["split"],
            "system_label": row["system_label"],
            "human": row["consensus"],
            "combined": row["combined"],
            "tfidf": row["tfidf"],
            "bm25_norm": row["bm25_norm"],
            "semantic": row["semantic"],
            "evidence_string": row.get("evidence_string") or "",
            "requirement_text": text[:180],
        }

    return {
        "n_usable": n,
        "n_system_gap": n_system_gap,
        "false_gap_n": len(gaps),
        "false_gap_rate_among_usable": (len(gaps) / n) if n else None,
        "false_gap_rate_among_system_gaps": (len(gaps) / n_system_gap) if n_system_gap else None,
        "cause_evidence_not_extracted": len(not_extracted),
        "cause_matched_but_scored_low": len(scored_low),
        "false_gap_requirement_phrase_in_cv": sum(1 for r in gaps if r.get("requirement_in_cv")),
        "missed_gap_n": len(missed),
        "missed_gap_rate_among_usable": (len(missed) / n) if n else None,
        "examples": [_brief(r) for r in gaps],
    }


def _g4_one_way(
    tune_rows: list[dict],
    confirm_rows: list[dict],
    *,
    tune_name: str,
    confirm_name: str,
    seed: int,
) -> dict[str, Any]:
    evaluable = len(tune_rows) >= MIN_SPLIT_N and len(confirm_rows) >= MIN_SPLIT_N
    if not evaluable:
        return {
            "evaluable": False,
            "tune_split": tune_name,
            "confirm_split": confirm_name,
            "n_tune": len(tune_rows),
            "n_confirm": len(confirm_rows),
            "selected": None,
            "improvement": None,
            "adopt": False,
            "reason": f"Need at least {MIN_SPLIT_N} consensus items on each split.",
        }
    selected = _select_config(tune_rows)
    weights = (
        selected["weights"]["tfidf"],
        selected["weights"]["bm25_norm"],
        selected["weights"]["semantic"],
    )
    human = np.asarray([r["consensus"] for r in confirm_rows], dtype=int)
    production = _labels_for(confirm_rows, PROD_WEIGHTS, PROD_PARTIAL, PROD_STRONG)
    candidate = _labels_for(
        confirm_rows,
        weights,
        selected["partial_threshold"],
        selected["strong_threshold"],
    )
    improvement = _bootstrap_delta(human, production, candidate, seed=seed)
    adopt = adoption_decision(
        improvement.get("value"),
        improvement.get("ci_low"),
        improvement.get("ci_high"),
    )
    reason = (
        "Untouched-split QWK gain has a bootstrap 95% CI entirely above 0."
        if adopt
        else "Untouched-split QWK gain CI does not lie entirely above 0. Keep production weights and thresholds."
    )
    if selected["is_production"]:
        adopt = False
        reason = "The tune split selected the production configuration. Keep it."
    return {
        "evaluable": True,
        "tune_split": tune_name,
        "confirm_split": confirm_name,
        "n_tune": len(tune_rows),
        "n_confirm": len(confirm_rows),
        "selected": selected,
        "improvement": improvement,
        "adopt": adopt,
        "reason": reason,
    }


def _g1_block(rows: list[dict], *, seed: int) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "supported": False}
    human = np.asarray([r["consensus"] for r in rows], dtype=int)
    system = np.asarray([r["system3"] for r in rows], dtype=int)
    score = np.asarray([r["combined"] for r in rows], dtype=float)
    qwk = bootstrap_metric_ci(
        human,
        system,
        quadratic_weighted_kappa,
        n_resamples=BOOTSTRAP,
        seed=seed,
    )
    rho = _spearman_rho(score, human)
    rho_p = _spearman_p(score, human)
    rho_ci = bootstrap_metric_ci(
        score,
        human.astype(float),
        lambda a, b: _spearman_rho(a, b),
        n_resamples=BOOTSTRAP,
        seed=seed + 1,
    )
    reconstructed = np.asarray([production_three(float(s)) for s in score], dtype=int)
    qwk_beyond = _ci_excludes_zero_above(qwk)
    spearman_beyond = (
        rho is not None
        and rho > 0
        and rho_p is not None
        and rho_p < 0.05
        and _ci_excludes_zero_above(rho_ci)
    )
    return {
        "n": int(human.size),
        "qwk": qwk,
        "qwk_landis_koch_descriptive": landis_koch(qwk.get("value")),
        "spearman": {"rho": rho, "p_value": rho_p, **{k: rho_ci.get(k) for k in ("ci_low", "ci_high")}},
        "qwk_beyond_chance": qwk_beyond,
        "spearman_beyond_chance": spearman_beyond,
        "supported": bool(qwk_beyond and spearman_beyond),
        "production_label_reconstruction_match_rate": float(np.mean(reconstructed == system)),
    }


def evaluate_hypotheses(usable: list[dict]) -> dict[str, Any]:
    primary = [r for r in usable if r["split"] == "primary-30"]
    holdout = [r for r in usable if r["split"] == "holdout-30"]
    g1 = {
        "pooled": _g1_block(usable, seed=STATS_SEED),
        "primary-30": _g1_block(primary, seed=STATS_SEED + 10),
        "holdout-30": _g1_block(holdout, seed=STATS_SEED + 20),
        "decision_rule": (
            "Supported only if pooled quadratic-weighted kappa has a bootstrap 95% CI "
            "entirely above 0 and Spearman rho of combined score vs human label is "
            "positive, two-sided p < 0.05, and its bootstrap 95% CI is entirely above 0."
        ),
    }
    g2 = {
        "pooled": _signal_block(usable, seed=STATS_SEED + 100),
        "primary-30": _signal_block(primary, seed=STATS_SEED + 140),
        "holdout-30": _signal_block(holdout, seed=STATS_SEED + 180),
        "decision_rule": (
            "Combined score separates met/partial from not-evidenced when its ROC AUC "
            "bootstrap 95% CI lies entirely above 0.5. It beats a single signal when the "
            "paired AUC difference CI lies entirely above 0."
        ),
    }
    g3 = {
        "pooled": _false_gaps(usable),
        "primary-30": _false_gaps(primary),
        "holdout-30": _false_gaps(holdout),
        "cause_rule": (
            "evidence_not_extracted when the stored matched evidence string is empty; "
            "matched_but_scored_low otherwise. Annotator snippet fields are not used, "
            "because empty snippets would mis-label every row."
        ),
    }
    # Drop bulky examples from the split copies' contribution to summary size is fine.
    g4_primary = _g4_one_way(
        primary,
        holdout,
        tune_name="primary-30",
        confirm_name="holdout-30",
        seed=STATS_SEED + 300,
    )
    g4_reverse = _g4_one_way(
        holdout,
        primary,
        tune_name="holdout-30",
        confirm_name="primary-30",
        seed=STATS_SEED + 400,
    )
    g4_reverse["role"] = "sensitivity_not_an_adoption_test"
    return {
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g4": {
            "grid": {
                "weight_step": WEIGHT_STEP,
                "partial_thresholds": list(PARTIAL_GRID),
                "strong_thresholds": list(STRONG_GRID),
                "production": {
                    "weights": {"tfidf": 0.25, "bm25_norm": 0.25, "semantic": 0.50},
                    "partial_threshold": PROD_PARTIAL,
                    "strong_threshold": PROD_STRONG,
                },
                "tie_break": "higher tune QWK, then smaller L1 distance to production",
            },
            "adoption_test": g4_primary,
            "reverse_sensitivity": g4_reverse,
            "adopt_recalibration": bool(g4_primary.get("adopt")),
        },
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(block: dict[str, Any] | None) -> str:
    if not block:
        return "n/a"
    return f"{_fmt(block.get('value'))} [{_fmt(block.get('ci_low'))}, {_fmt(block.get('ci_high'))}]"


def g1_may_be_interpreted(kappa_quadratic: dict[str, Any] | None) -> bool:
    """Round-2 rule: interpret G1 only when main-pair quadratic kappa has a 95% CI entirely above 0."""
    if not kappa_quadratic:
        return False
    lo = kappa_quadratic.get("ci_low")
    val = kappa_quadratic.get("value")
    return lo is not None and val is not None and float(lo) > 0 and float(val) > 0


def render_report(payload: dict[str, Any]) -> str:
    g1 = payload["g1"]["pooled"]
    g2 = payload["g2"]["pooled"]
    g3 = payload["g3"]["pooled"]
    g4 = payload["g4"]["adoption_test"]
    gate = payload.get("g1_interpretation") or {}
    lines = [
        "# Gold-set report",
        "",
        f"Analysis round: `{payload.get('analysis_round', 'round1_human_labels')}`.",
        "",
        "Frozen hypotheses: `docs/gold_set_protocol.md` (G1–G4). This file is the phase-2 result. It does not edit the protocol.",
        "",
        f"G1 interpretation: **{gate.get('interpreted')}**. {gate.get('reason', '')}",
        "",
        "Human labels are the five files in `experiments/gold_set/annotator_key/` (annotators A–E), copied unchanged into `experiments/gold_set/annotations/` for the runner. No label was imputed. Disagreements without `adjudicated.csv` are excluded from consensus. That exclusion can make agreement with the system look better than it is, so per-annotator quadratic-weighted kappa is reported beside the consensus test.",
        "",
        f"Assigned main items: {payload.get('n_assigned_main')}. Usable consensus: {payload.get('n_usable_consensus')}. Disposition: `{payload.get('consensus_disposition')}`.",
        "",
        "## Inter-annotator agreement",
        "",
        "Main dual-coded pairs (quadratic-weighted kappa, bootstrap 95% CI):",
        "",
        f"- n pairs = {payload['agreement_main'].get('n')}",
        f"- percent agreement = {_fmt(payload['agreement_main'].get('percent_agreement'))}",
        f"- unweighted kappa = {_ci(payload['agreement_main'].get('kappa_unweighted'))}",
        f"- quadratic kappa = {_ci(payload['agreement_main'].get('kappa_quadratic'))} ({landis_koch((payload['agreement_main'].get('kappa_quadratic') or {}).get('value'))}; Landis–Koch is descriptive only)",
        "",
        "Calibration, all annotator pairs on the 10 shared items:",
        "",
        f"- n pairs = {payload['agreement_calibration'].get('n')}",
        f"- percent agreement = {_fmt(payload['agreement_calibration'].get('percent_agreement'))}",
        f"- quadratic kappa = {_ci(payload['agreement_calibration'].get('kappa_quadratic'))}",
        "",
    ]
    cal_adj = payload.get("calibration_adjudication") or {}
    if cal_adj:
        label_bits = ", ".join(f"{iid}={cal_adj['labels'][iid]}" for iid in cal_adj["labels"])
        lines += [
            "Adjudicated calibration labels, recorded after the discussion. Independent round-1 labels are unchanged. This set is not G1:",
            "",
            f"- {label_bits}",
            "",
        ]
        pooled = cal_adj.get("pooled_independent_vs_adjudicated") or {}
        lines.append(
            f"- Independent ratings versus the adjudicated label: n = {pooled.get('n')}, "
            f"percent agreement = {_fmt(pooled.get('percent_agreement'))}, "
            f"quadratic kappa = {_ci(pooled.get('qwk'))}"
        )
        lines.append("")
        for name, block in (cal_adj.get("annotator_vs_adjudicated") or {}).items():
            lines.append(
                f"- {name}: {block.get('n_match')}/{block.get('n')} match "
                f"(agreement {_fmt(block.get('percent_agreement'))}, QWK {_fmt(block.get('qwk'))})"
            )
        system = cal_adj.get("system_vs_adjudicated") or {}
        lines += [
            "",
            f"- System versus adjudicated calibration: n = {system.get('n')}, "
            f"accuracy = {_fmt(system.get('accuracy'))}, QWK = {_fmt(system.get('qwk'))}. "
            "Descriptive only. Not G1.",
            "",
        ]
    lines += [
        "Main-item label counts (0 not evidenced, 1 partial, 2 met). Different use of the scale lowers kappa even when raters are not guessing:",
        "",
    ]
    for name, counts in (payload.get("label_margins_main") or {}).items():
        lines.append(f"- {name}: `{counts}`")
    lines += [
        "",
        "## G1 — system labels vs human consensus",
        "",
        payload["g1"]["decision_rule"],
        "",
        f"- n = {g1.get('n')}",
        f"- quadratic-weighted kappa = {_ci(g1.get('qwk'))} ({g1.get('qwk_landis_koch_descriptive')}; descriptive band only)",
        f"- Spearman combined score vs human = {_fmt((g1.get('spearman') or {}).get('rho'))} (p = {_fmt((g1.get('spearman') or {}).get('p_value'))}, CI [{_fmt((g1.get('spearman') or {}).get('ci_low'))}, {_fmt((g1.get('spearman') or {}).get('ci_high'))}])",
        f"- beyond chance: kappa {g1.get('qwk_beyond_chance')}, Spearman {g1.get('spearman_beyond_chance')}",
        f"- **G1 supported: {g1.get('supported')}**",
        f"- Stored three-category labels match the 0.40/0.65 reconstruction on {_fmt(g1.get('production_label_reconstruction_match_rate'), 3)} of consensus items.",
        "",
        "Per-annotator quadratic kappa against the stored system label (sensitivity; disagreements are kept):",
        "",
    ]
    for name, block in (payload.get("annotator_vs_system") or {}).items():
        lines.append(
            f"- {name}: n = {block.get('n')}, QWK = {_ci(block.get('qwk'))}"
        )
    lines += [
        "",
        "Split consensus QWK:",
        "",
        f"- primary-30: {_ci(payload['g1']['primary-30'].get('qwk'))} (n = {payload['g1']['primary-30'].get('n')})",
        f"- holdout-30: {_ci(payload['g1']['holdout-30'].get('qwk'))} (n = {payload['g1']['holdout-30'].get('n')})",
        "",
        "## G2 — does the combined score separate met/partial from not evidenced?",
        "",
        payload["g2"]["decision_rule"],
        "",
        f"Pooled n = {g2.get('n')} (positive {g2.get('n_positive')}, negative {g2.get('n_negative')}).",
        "",
        "| Signal | AUC [95% CI] |",
        "| --- | --- |",
    ]
    for name in ("combined", "tfidf", "bm25_norm", "semantic"):
        lines.append(f"| {name} | {_ci(g2['auc'][name])} |")
    lines += [
        "",
        f"Hybrid separates (CI entirely above 0.5): **{g2.get('hybrid_separates')}**.",
        "",
    ]
    for name, block in g2.get("hybrid_beats", {}).items():
        lines.append(
            f"- hybrid minus {name}: delta AUC {_fmt(block.get('delta_auc'))} "
            f"[{_fmt(block.get('ci_low'))}, {_fmt(block.get('ci_high'))}], beats = {block.get('beats')}"
        )
    lines += [
        "",
        "## G3 — false gaps",
        "",
        payload["g3"]["cause_rule"],
        "",
        f"- false gaps = {g3.get('false_gap_n')} / {g3.get('n_usable')} usable "
        f"(rate {_fmt(g3.get('false_gap_rate_among_usable'))})",
        f"- among items the system called gap: {_fmt(g3.get('false_gap_rate_among_system_gaps'))} "
        f"({g3.get('false_gap_n')} / {g3.get('n_system_gap')})",
        f"- evidence not extracted: {g3.get('cause_evidence_not_extracted')}",
        f"- matched but scored low: {g3.get('cause_matched_but_scored_low')}",
        f"- of those false gaps, requirement phrase is a literal substring of the CV: {g3.get('false_gap_requirement_phrase_in_cv')}",
        f"- complementary error (human not evidenced, system partial or strong): "
        f"{g3.get('missed_gap_n')} (rate {_fmt(g3.get('missed_gap_rate_among_usable'))})",
        "",
        "## G4 — recalibration",
        "",
        "Grid: signal weights in steps of 0.25 summing to 1; partial threshold in {0.30, 0.35, 0.40, 0.45, 0.50}; strong threshold in {0.55, 0.60, 0.65, 0.70, 0.75}. Objective on the tune split is quadratic-weighted kappa. Ties prefer the configuration closer to production (0.25 / 0.25 / 0.50, thresholds 0.40 / 0.65).",
        "",
        f"Adoption test tunes on {g4.get('tune_split')} (n = {g4.get('n_tune')}) and confirms on {g4.get('confirm_split')} (n = {g4.get('n_confirm')}). The search runs only when both splits have at least 8 consensus items.",
        "",
        f"- evaluable: {g4.get('evaluable')}",
        f"- selected: `{g4.get('selected')}`",
        f"- untouched improvement (candidate QWK − production QWK) = {_ci(g4.get('improvement'))}",
        f"- production QWK on untouched = {_fmt((g4.get('improvement') or {}).get('qwk_production'))}",
        f"- candidate QWK on untouched = {_fmt((g4.get('improvement') or {}).get('qwk_candidate'))}",
        f"- **adopt recalibration: {payload['g4'].get('adopt_recalibration')}**",
        f"- reason: {g4.get('reason')}",
        "",
        "Reverse split (tune on holdout, confirm on primary) is a sensitivity. It is not an adoption test.",
        "",
        f"- reverse adopt flag (ignored): {payload['g4']['reverse_sensitivity'].get('adopt')}",
        f"- reverse improvement = {_ci(payload['g4']['reverse_sensitivity'].get('improvement'))}",
        "",
        "## Figures",
        "",
        "![Figure 26: system vs human confusion](figures/fig26_gold_confusion.png)",
        "",
        "![Figure 27: ROC of gap signals](figures/fig27_gold_roc.png)",
        "",
        "![Figure 28: false-gap causes](figures/fig28_gold_false_gaps.png)",
        "",
        "![Figure 29: recalibration QWK](figures/fig29_gold_recalibration.png)",
        "",
        "## What this does to question generation",
        "",
        payload.get("aqg_implication") or "",
        "",
    ]
    return "\n".join(lines) + "\n"


def aqg_implication(hypotheses: dict[str, Any]) -> str:
    """Short consequence for interview-question generation. Written from the flags only."""
    g1_ok = bool(hypotheses["g1"]["pooled"].get("supported"))
    separates = bool(hypotheses["g2"]["pooled"].get("hybrid_separates"))
    beats = hypotheses["g2"]["pooled"].get("hybrid_beats") or {}
    any_beat = any(bool(v.get("beats")) for v in beats.values())
    g3 = hypotheses["g3"]["pooled"]
    adopt = bool(hypotheses["g4"].get("adopt_recalibration"))
    false_rate = g3.get("false_gap_rate_among_system_gaps")
    extracted = g3.get("cause_evidence_not_extracted") or 0
    scored_low = g3.get("cause_matched_but_scored_low") or 0
    parts = []
    if not g1_ok:
        parts.append(
            "G1 is not supported. The production gap label is not a validated map of what a human finds in the CV. "
            "Questions aimed at those gap labels are not yet personalization. Keep the production scorer, and do not "
            "describe competency-weighted MMR as gap-validated."
        )
    else:
        parts.append(
            "G1 is supported on the consensus items. Gap labels carry human-aligned signal, so conditioning a question "
            "pool on the gap matrix is a defensible personalization step."
        )
    if separates and any_beat:
        parts.append(
            "The hybrid score separates met/partial from not-evidenced and beats at least one single signal on the "
            "pre-registered AUC contrast. Keep the hybrid rather than dropping to one retrieval channel."
        )
    elif separates:
        parts.append(
            "The hybrid score separates the human binary, but it does not beat every single signal under the CI rule. "
            "Keep the hybrid as the production score. Do not claim it dominates TF-IDF, BM25, and semantic similarity."
        )
    else:
        parts.append(
            "The combined score does not separate human met/partial from not-evidenced under the CI rule. "
            "Targeting questions at low combined scores is not a reliable way to find un-evidenced requirements."
        )
    if false_rate is not None:
        parts.append(
            f"On the consensus subset, {g3.get('false_gap_n')} of {g3.get('n_system_gap')} system-gap items "
            f"were judged met or partial ({extracted} with no stored evidence string, {scored_low} with a stored neighbor). "
            f"The requirement phrase is a literal substring of the CV in {g3.get('false_gap_requirement_phrase_in_cv')} of those false gaps. "
            "A stored neighbor that is a different competency, plus an absent phrase, is not the same failure as finding the right evidence and scoring it low. "
            "The 6-item count is also not a population rate, because items without rater consensus are excluded."
        )
    if adopt:
        parts.append(
            "G4 adopts the tune-split configuration because the holdout QWK gain CI excludes 0. "
            "Later scoring should use that locked configuration. This run does not rewrite stored gap artifacts."
        )
    else:
        parts.append(
            "G4 does not adopt a new configuration. Production weights 0.25/0.25/0.50 and thresholds 0.65/0.40/0.20 stay. "
            "The next gain in question quality is selection balance on the pools already generated, not another threshold search on these items."
        )
    return " ".join(parts)
