"""Core metric computations M1–M4 (frozen v1 definitions)."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.services.eval_metrics.constants import (
    FINAL_N,
    GAP_LABELS,
    NEAR_DUP_SENSITIVITY,
    NEAR_DUP_THRESHOLD,
)
from app.services.eval_metrics.vendi import vendi_score_from_embeddings


def pairwise_similarities(sim: np.ndarray) -> np.ndarray:
    """Upper-triangle pairwise similarities (i < j)."""
    n = sim.shape[0]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(float(sim[i, j]))
    return np.asarray(vals, dtype=np.float64)


def m1_redundancy(
    question_embeddings: np.ndarray,
    *,
    near_dup_threshold: float = NEAR_DUP_THRESHOLD,
    sensitivity_threshold: float = NEAR_DUP_SENSITIVITY,
) -> dict[str, float | None]:
    """M1 redundancy metrics. Lower primary is better."""
    E = np.asarray(question_embeddings, dtype=np.float64)
    if E.ndim != 2 or E.shape[0] < 2:
        return {
            "M1.P": None,
            "M1.S1": None,
            "M1.S2": None,
            "n_pairs": 0,
            "n_questions": int(E.shape[0]) if E.ndim == 2 else 0,
        }
    norms = np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    En = E / norms
    sim = En @ En.T
    pairs = pairwise_similarities(sim)
    n_pairs = int(pairs.size)
    return {
        "M1.P": float(pairs.mean()) if n_pairs else None,
        "M1.S1": float(np.mean(pairs >= near_dup_threshold)) if n_pairs else None,
        "M1.S2": float(np.mean(pairs >= sensitivity_threshold)) if n_pairs else None,
        "n_pairs": n_pairs,
        "n_questions": int(E.shape[0]),
    }


def m4_diversity(question_embeddings: np.ndarray) -> dict[str, float | None]:
    """M4 diversity (Vendi). Higher primary is better."""
    vs = vendi_score_from_embeddings(question_embeddings)
    return {"M4.P": vs, "M4.S": None}


def dimension_entropy(labels: list[str], n_dimensions: int = 5) -> float | None:
    if not labels:
        return None
    from collections import Counter

    c = Counter(labels)
    probs = np.asarray([c[k] / len(labels) for k in c], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(probs > 0, np.log(probs), 0.0)
    return float(-np.sum(probs * logs))


def _is_gap(label: str) -> bool:
    return str(label) in GAP_LABELS


def m2_m3_from_assignment(
    *,
    linked: list[bool],
    r_star_indices: list[int | None],
    requirements: list[dict[str, Any]],
    n_questions: int = FINAL_N,
) -> dict[str, Any]:
    """Compute M2/M3 given per-question assignment results."""
    n_req = len(requirements)
    if n_questions <= 0:
        return _empty_m2_m3(n_req)

    gap_indices = [i for i, r in enumerate(requirements) if _is_gap(r.get("match_label", ""))]
    n_gap = len(gap_indices)
    required_indices = [
        i for i, r in enumerate(requirements) if str(r.get("priority", "")).lower() == "required"
    ]
    n_required = len(required_indices)

    linked_gap_q = 0
    covered_reqs: set[int] = set()
    covered_gaps: set[int] = set()
    covered_required: set[int] = set()
    unlinked = 0
    cat_counts = {"strength": 0, "partial": 0, "gap": 0, "other": 0}

    for i in range(min(len(linked), n_questions)):
        if not linked[i]:
            unlinked += 1
            continue
        j = r_star_indices[i]
        if j is None or j < 0 or j >= n_req:
            unlinked += 1
            continue
        covered_reqs.add(j)
        lab = str(requirements[j].get("match_label", ""))
        if _is_gap(lab):
            linked_gap_q += 1
            covered_gaps.add(j)
            cat_counts["gap"] += 1
        elif lab == "partial_match":
            cat_counts["partial"] += 1
        elif lab == "strong_match":
            cat_counts["strength"] += 1
        else:
            cat_counts["other"] += 1
        if j in required_indices:
            covered_required.add(j)

    # M2.P
    m2p = linked_gap_q / n_questions

    # M2.S1 capped gap coverage
    denom_gap = min(n_questions, n_gap) if n_gap > 0 else 0
    m2s1 = (len(covered_gaps) / denom_gap) if denom_gap else None

    # M2.S2 weight captured
    gap_weights = sorted(
        (float(requirements[i].get("weight") or 0.0) for i in gap_indices),
        reverse=True,
    )
    top_k = gap_weights[: min(n_questions, n_gap)] if n_gap else []
    denom_w = sum(top_k)
    covered_w = sum(float(requirements[i].get("weight") or 0.0) for i in covered_gaps)
    m2s2 = (covered_w / denom_w) if denom_w > 0 else None

    # M3.P
    denom_req = min(n_questions, n_req) if n_req > 0 else 0
    m3p = (len(covered_reqs) / denom_req) if denom_req else None

    # M3.S1
    if n_required == 0:
        m3s1 = None
        m3s1_undefined = True
    else:
        denom_reqd = min(n_questions, n_required)
        m3s1 = len(covered_required) / denom_reqd if denom_reqd else None
        m3s1_undefined = False

    n_linked = n_questions - unlinked
    return {
        "M2.P": float(m2p),
        "M2.S1": None if m2s1 is None else float(m2s1),
        "M2.S2": None if m2s2 is None else float(m2s2),
        "M3.P": None if m3p is None else float(m3p),
        "M3.S1": None if m3s1 is None else float(m3s1),
        "M3.S1_undefined": m3s1_undefined,
        "M3.S2": float(unlinked / n_questions),
        "n_requirements": n_req,
        "n_gap_requirements": n_gap,
        "n_required_requirements": n_required,
        "n_linked": int(n_linked),
        "n_unlinked": int(unlinked),
        "linked_category_shares": {
            k: (v / n_linked if n_linked else None) for k, v in cat_counts.items()
        },
    }


def _empty_m2_m3(n_req: int) -> dict[str, Any]:
    return {
        "M2.P": None,
        "M2.S1": None,
        "M2.S2": None,
        "M3.P": None,
        "M3.S1": None,
        "M3.S1_undefined": True,
        "M3.S2": None,
        "n_requirements": n_req,
        "n_gap_requirements": 0,
        "n_required_requirements": 0,
        "n_linked": 0,
        "n_unlinked": 0,
        "linked_category_shares": {
            "strength": None,
            "partial": None,
            "gap": None,
            "other": None,
        },
    }


def lexical_specificity_proxy(texts: list[str]) -> float | None:
    """Exploratory: mean fraction of tokens with length >= 6 (not frozen)."""
    if not texts:
        return None
    rates = []
    for t in texts:
        toks = [w for w in t.lower().split() if w.isalpha()]
        if not toks:
            rates.append(0.0)
        else:
            rates.append(sum(1 for w in toks if len(w) >= 6) / len(toks))
    return float(np.mean(rates))
