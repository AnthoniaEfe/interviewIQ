"""Diagnostic playbook D1–D8 (local, zero LLM cost)."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.services.embeddings.service import EmbeddingService
from app.services.eval_metrics.assignment import score_with_assignment
from app.services.eval_metrics.compute import compute_full_metrics, embed_texts
from app.services.eval_metrics.constants import (
    FINAL_N,
    GAP_LABELS,
    MMR_LAMBDAS,
    POOL_SIZE_KS,
    SEED,
)
from app.services.eval_metrics.decision import compare_metric, win_tie_loss
from app.services.eval_metrics.loaders import question_texts
from app.services.mmr.select import competency_weighted_mmr
from app.schemas.domain import InterviewQuestion


def _req_arrays(
    embeddings: EmbeddingService,
    requirements: list[dict],
) -> np.ndarray:
    return embed_texts(
        embeddings, [str(r.get("job_competency") or "") for r in requirements]
    )


def oracle_greedy_selection(
    *,
    pool_texts: list[str],
    pool_emb: np.ndarray,
    req_emb: np.ndarray,
    requirements: list[dict],
    tau: float,
    metric: str,
    k: int = FINAL_N,
) -> list[int]:
    """Greedy select k questions maximising M2.P or M3.P."""
    n = len(pool_texts)
    selected: list[int] = []
    remaining = set(range(n))
    while remaining and len(selected) < k:
        best_i = None
        best_score = -1e18
        for i in remaining:
            trial = selected + [i]
            E = pool_emb[trial]
            scored = score_with_assignment(
                question_embeddings=E,
                req_embeddings=req_emb,
                requirements=requirements,
                tau=tau,
                n_questions=len(trial),
            )
            # Normalise to /10 framing for partial sets: use rate definitions with n_questions=len(trial)
            val = scored.get(metric)
            if val is None:
                continue
            if val > best_score:
                best_score = val
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


def d1_pool_vs_selection(
    *,
    embeddings: EmbeddingService,
    pool_record: dict,
    selected_record: dict,
    requirements: list[dict],
    tau: float,
) -> dict[str, Any]:
    pool_texts = question_texts(pool_record)
    sel_texts = question_texts(selected_record)
    if not pool_texts or not requirements:
        return {"available": False}
    pool_emb = embed_texts(embeddings, pool_texts)
    req_emb = _req_arrays(embeddings, requirements)
    pool_m = compute_full_metrics(
        pool_texts,
        embeddings,
        requirements=requirements,
        tau=tau,
        precomputed_q=pool_emb,
        precomputed_r=req_emb,
    )
    # For pool, n_questions is len(pool); recompute M2/M3 with actual n
    pool_m2m3 = score_with_assignment(
        question_embeddings=pool_emb,
        req_embeddings=req_emb,
        requirements=requirements,
        tau=tau,
        n_questions=len(pool_texts),
    )
    sel_m = compute_full_metrics(
        sel_texts,
        embeddings,
        requirements=requirements,
        tau=tau,
        precomputed_r=req_emb,
    )
    oracle_m3_idx = oracle_greedy_selection(
        pool_texts=pool_texts,
        pool_emb=pool_emb,
        req_emb=req_emb,
        requirements=requirements,
        tau=tau,
        metric="M3.P",
    )
    oracle_m2_idx = oracle_greedy_selection(
        pool_texts=pool_texts,
        pool_emb=pool_emb,
        req_emb=req_emb,
        requirements=requirements,
        tau=tau,
        metric="M2.P",
    )
    oracle_m3 = score_with_assignment(
        question_embeddings=pool_emb[oracle_m3_idx],
        req_embeddings=req_emb,
        requirements=requirements,
        tau=tau,
        n_questions=FINAL_N,
    )
    oracle_m2 = score_with_assignment(
        question_embeddings=pool_emb[oracle_m2_idx],
        req_embeddings=req_emb,
        requirements=requirements,
        tau=tau,
        n_questions=FINAL_N,
    )
    return {
        "available": True,
        "pool": {"M2.P": pool_m2m3.get("M2.P"), "M3.P": pool_m2m3.get("M3.P"), "M1.P": pool_m.get("M1.P"), "M4.P": pool_m.get("M4.P"), "n": len(pool_texts)},
        "selected_T": {"M2.P": sel_m.get("M2.P"), "M3.P": sel_m.get("M3.P"), "M1.P": sel_m.get("M1.P"), "M4.P": sel_m.get("M4.P")},
        "oracle_M3.P": {"M2.P": oracle_m3.get("M2.P"), "M3.P": oracle_m3.get("M3.P"), "indices": oracle_m3_idx},
        "oracle_M2.P": {"M2.P": oracle_m2.get("M2.P"), "M3.P": oracle_m2.get("M3.P"), "indices": oracle_m2_idx},
    }


def d2_weight_audit(pool_record: dict, selected_record: dict, requirements: list[dict]) -> dict[str, Any]:
    def _weights(rec: dict) -> dict[str, Any]:
        qs = rec.get("questions") or rec.get("valid_questions") or []
        weights = []
        # Pool records may store parallel weight arrays
        stored_weights = rec.get("weights") or []
        fallback = 0
        probes = 0
        by_cat = {"gap": 0, "partial": 0, "strength": 0, "profile_probe": 0, "other": 0}
        by_name = {r.get("job_competency"): r for r in requirements}
        outcomes = rec.get("weight_outcomes") or []
        for i, q in enumerate(qs):
            if not isinstance(q, dict):
                continue
            w = q.get("weight")
            if w is None and i < len(stored_weights):
                w = stored_weights[i]
            if w is not None:
                weights.append(float(w))
            outcome = outcomes[i] if i < len(outcomes) else None
            if (
                q.get("weight_fallback")
                or q.get("fallback_weight")
                or (isinstance(outcome, dict) and outcome.get("fallback"))
                or outcome == "fallback"
            ):
                fallback += 1
            tg = q.get("target_gap")
            if tg == "profile_probe":
                probes += 1
                by_cat["profile_probe"] += 1
            elif tg in by_name:
                lab = str(by_name[tg].get("match_label"))
                if lab in GAP_LABELS:
                    by_cat["gap"] += 1
                elif lab == "partial_match":
                    by_cat["partial"] += 1
                elif lab == "strong_match":
                    by_cat["strength"] += 1
                else:
                    by_cat["other"] += 1
            else:
                by_cat["other"] += 1
        n = len(qs) or None
        # Prefer recorded rates on pool summaries when present
        fb_rate = rec.get("fallback_rate")
        if fb_rate is None:
            fb_rate = (fallback / len(qs)) if qs else None
        pp_rate = rec.get("profile_probe_rate")
        if pp_rate is None:
            pp_rate = (probes / len(qs)) if qs else None
        return {
            "mean_weight": float(np.mean(weights)) if weights else None,
            "fallback_rate": fb_rate,
            "profile_probe_rate": pp_rate,
            "gap_category_counts": by_cat,
            "n": n,
        }

    return {"pool": _weights(pool_record), "selected": _weights(selected_record)}


def d3_lambda_sensitivity(
    *,
    embeddings: EmbeddingService,
    pool_record: dict,
    requirements: list[dict],
    tau: float,
    weights: list[float] | None = None,
) -> dict[str, Any]:
    texts = question_texts(pool_record)
    if len(texts) < FINAL_N:
        return {"available": False}
    qs_raw = (
        pool_record.get("questions")
        or pool_record.get("valid_questions")
        or []
    )
    stored_w = pool_record.get("weights") or []
    questions = []
    for i, t in enumerate(texts):
        qdict = qs_raw[i] if i < len(qs_raw) and isinstance(qs_raw[i], dict) else {}
        questions.append(
            InterviewQuestion(
                question=t,
                competency=str(qdict.get("competency") or "x"),
                target_gap=str(qdict.get("target_gap") or "profile_probe"),
                priority=float(qdict.get("priority") or 0.5),
            )
        )
    sim = embeddings.similarity_matrix(texts, texts)
    if weights is None:
        weights = []
        for i, q in enumerate(questions):
            qdict = qs_raw[i] if i < len(qs_raw) and isinstance(qs_raw[i], dict) else {}
            w = qdict.get("weight")
            if w is None and i < len(stored_w):
                w = stored_w[i]
            weights.append(float(w if w is not None else (q.priority or 1.0)))
    req_emb = _req_arrays(embeddings, requirements)
    out = {"lambdas": list(MMR_LAMBDAS), "T_weighted": {}, "B2_priority": {}}
    for lam in MMR_LAMBDAS:
        sel_w = competency_weighted_mmr(
            questions, sim, weights, lambda_=float(lam), k=FINAL_N
        )
        sel_p = competency_weighted_mmr(
            questions, sim, [1.0] * len(questions), lambda_=float(lam), k=FINAL_N
        )
        for label, sel in (("T_weighted", sel_w), ("B2_priority", sel_p)):
            idxs = sel.selected_indices
            E = embed_texts(embeddings, [texts[i] for i in idxs])
            scored = score_with_assignment(
                question_embeddings=E,
                req_embeddings=req_emb,
                requirements=requirements,
                tau=tau,
            )
            m1 = compute_full_metrics(
                [texts[i] for i in idxs], embeddings, requirements=None, tau=None
            )
            out[label][str(lam)] = {
                "M1.P": m1.get("M1.P"),
                "M2.P": scored.get("M2.P"),
                "M3.P": scored.get("M3.P"),
                "M4.P": m1.get("M4.P"),
            }
    return out


def d4_pool_size_sensitivity(
    *,
    embeddings: EmbeddingService,
    pool_record: dict,
    requirements: list[dict],
    tau: float,
    seed: int = SEED,
) -> dict[str, Any]:
    texts = question_texts(pool_record)
    rng = np.random.default_rng(seed)
    req_emb = _req_arrays(embeddings, requirements)
    out = {}
    for k in POOL_SIZE_KS:
        if len(texts) < k:
            out[str(k)] = None
            continue
        idx = rng.choice(len(texts), size=k, replace=False)
        sub = [texts[i] for i in idx]
        # select top by simple oracle M3 on subset then take 10 if k>10 else all
        E = embed_texts(embeddings, sub)
        if k > FINAL_N:
            sel = oracle_greedy_selection(
                pool_texts=sub,
                pool_emb=E,
                req_emb=req_emb,
                requirements=requirements,
                tau=tau,
                metric="M3.P",
            )
            E2 = E[sel]
        else:
            E2 = E
        scored = score_with_assignment(
            question_embeddings=E2,
            req_embeddings=req_emb,
            requirements=requirements,
            tau=tau,
        )
        out[str(k)] = {"M2.P": scored.get("M2.P"), "M3.P": scored.get("M3.P")}
    return out


def d5_failures(pool_records: list[dict]) -> dict[str, Any]:
    n = len(pool_records)
    if n == 0:
        return {"n": 0}
    val_fail_rates = []
    trunc = 0
    backfill = 0
    for r in pool_records:
        n_raw = r.get("n_raw")
        n_failed = r.get("n_failed")
        if n_raw and n_failed is not None:
            val_fail_rates.append(float(n_failed) / float(n_raw))
        elif r.get("validation_failed") or r.get("pool_validation_failed"):
            val_fail_rates.append(1.0)
        else:
            fails = r.get("validation_failures") or []
            if fails:
                val_fail_rates.append(1.0)
            else:
                val_fail_rates.append(0.0)
        if r.get("truncated"):
            trunc += 1
        if r.get("backfill_used") or r.get("backfill_rate"):
            backfill += 1
        prov = r.get("provenance") or r.get("generation_meta") or {}
        if prov.get("backfill_used"):
            backfill += 1
    return {
        "n": n,
        "validation_failure_rate": float(np.mean(val_fail_rates)) if val_fail_rates else 0.0,
        "truncation_rate": trunc / n,
        "backfill_rate": backfill / n,
    }


def d6_robustness(
    *,
    treatment: dict[str, np.ndarray],
    baselines: dict[str, dict[str, np.ndarray]],
    sensitivity_series: dict[str, dict[str, dict[str, np.ndarray]]],
) -> dict[str, Any]:
    """Repeat win/tie/loss under assignment variants."""
    flips = []
    primary_outcomes = {}
    for bname, bmetrics in baselines.items():
        primary_outcomes[bname] = {}
        for m in ("M1.P", "M2.P", "M3.P", "M4.P"):
            if m not in treatment or m not in bmetrics:
                continue
            row = compare_metric(treatment[m], bmetrics[m], m)
            primary_outcomes[bname][m] = row["outcome"]

    for variant, series in sensitivity_series.items():
        for bname, bmetrics in series.items():
            for m in ("M2.P", "M3.P"):
                if m not in treatment or bname not in baselines:
                    continue
                # compare variant treatment vs variant baseline if available
                t_arr = series.get("T", {}).get(m)
                b_arr = bmetrics.get(m)
                if t_arr is None or b_arr is None:
                    continue
                row = compare_metric(t_arr, b_arr, m)
                prim = primary_outcomes.get(bname, {}).get(m)
                if prim and row["outcome"] != prim:
                    flips.append(
                        {
                            "variant": variant,
                            "baseline": bname,
                            "metric": m,
                            "primary": prim,
                            "variant_outcome": row["outcome"],
                        }
                    )
    return {"primary_outcomes": primary_outcomes, "flips": flips, "any_flip": bool(flips)}


def d7_strata(rows_by_condition: dict[str, list[dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cond, rows in rows_by_condition.items():
        good = [r for r in rows if r.get("fit_label") == "Good Fit"]
        pot = [r for r in rows if r.get("fit_label") == "Potential Fit"]
        few = [r for r in rows if (r.get("n_requirements") or 99) < 10]
        many = [r for r in rows if (r.get("n_requirements") or 0) >= 10]

        def mean_m(rs, m):
            xs = [r.get(m) for r in rs if r.get(m) is not None]
            return float(np.mean(xs)) if xs else None

        out[cond] = {
            "Good Fit": {m: mean_m(good, m) for m in ("M1.P", "M2.P", "M3.P", "M4.P")},
            "Potential Fit": {m: mean_m(pot, m) for m in ("M1.P", "M2.P", "M3.P", "M4.P")},
            "lt_10_requirements": {
                "n": len(few),
                **{m: mean_m(few, m) for m in ("M1.P", "M2.P", "M3.P", "M4.P")},
            },
            "ge_10_requirements": {
                "n": len(many),
                **{m: mean_m(many, m) for m in ("M1.P", "M2.P", "M3.P", "M4.P")},
            },
        }
    return out


def d8_attribution(d1_agg: dict, d2_agg: dict, d6: dict, fit_label_null: bool = True) -> dict[str, Any]:
    notes = []
    if d1_agg.get("oracle_vs_T_M3") is not None:
        gap = d1_agg["oracle_vs_T_M3"]
        if gap is not None and gap < 0.05:
            notes.append("Oracle ≈ T on M3.P → generation (pool) likely bottleneck.")
        elif gap is not None and gap >= 0.05:
            notes.append("Oracle ≫ T on M3.P → selection likely bottleneck.")
    if fit_label_null:
        notes.append(
            "Fit-label control was null; gap-matrix quality may limit M2/M3 ceiling."
        )
    if d6.get("any_flip"):
        notes.append("Some win/tie/loss outcomes flip under assignment sensitivity.")
    else:
        notes.append("Primary win/tie/loss stable under reported assignment variants.")
    notes.append(
        "M2/M3 advantage for T vs B1 is partly by construction (T sees the gap matrix)."
    )
    return {"likely_components": notes}
