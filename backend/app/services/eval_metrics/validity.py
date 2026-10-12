"""Metric validity checks V1–V5 (Part A, before freeze)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from app.services.eval_metrics.assignment import (
    calibrate_tau_p,
    sample_assignment_audit_rows,
    score_with_assignment,
)
from app.services.eval_metrics.constants import (
    DUPLICATE_INJECT_N,
    FINAL_N,
    NULL_PERMUTATIONS,
    SEED,
)
from app.services.eval_metrics.metrics_core import m1_redundancy, m4_diversity
from app.services.eval_metrics.vendi import vendi_score_from_embeddings


def v1_known_answer_checks() -> dict[str, Any]:
    """Unit-style known-answer checks (also covered by pytest)."""
    # Identical embeddings → Vendi 1
    ident = np.ones((10, 8), dtype=np.float64)
    vs_ident = vendi_score_from_embeddings(ident)
    # Orthogonal-ish: identity rows in high-dim orthonormal basis
    ortho = np.eye(10, dtype=np.float64)
    vs_ortho = vendi_score_from_embeddings(ortho)

    # M1.S1: zero near-dups vs one near-dup pair
    rng = np.random.default_rng(0)
    base = rng.normal(size=(10, 32))
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    m1_zero = m1_redundancy(base, near_dup_threshold=0.99)
    # Force one near-duplicate
    dup = base.copy()
    dup[1] = dup[0] + 1e-9
    dup[1] = dup[1] / np.linalg.norm(dup[1])
    m1_one = m1_redundancy(dup, near_dup_threshold=0.99)

    # Hand-built M2/M3
    q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    # pad to 10 with near-orthogonal noise
    extra = np.zeros((7, 2), dtype=np.float64)
    extra[:, 0] = 0.01
    Q = np.vstack([q, extra])
    R = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    reqs = [
        {"job_competency": "A", "match_label": "missing", "weight": 0.9, "priority": "required"},
        {"job_competency": "B", "match_label": "strong_match", "weight": 0.1, "priority": "preferred"},
    ]
    scored = score_with_assignment(
        question_embeddings=Q,
        req_embeddings=R,
        requirements=reqs,
        tau=0.5,
        n_questions=FINAL_N,
    )
    return {
        "vendi_identical": vs_ident,
        "vendi_orthogonal": vs_ortho,
        "vendi_identical_ok": vs_ident is not None and abs(vs_ident - 1.0) < 1e-6,
        "vendi_orthogonal_ok": vs_ortho is not None and abs(vs_ortho - 10.0) < 1e-6,
        "M1.S1_zero": m1_zero["M1.S1"],
        "M1.S1_one_pair": m1_one["M1.S1"],
        "M2.P_hand": scored["M2.P"],
        "M3.P_hand": scored["M3.P"],
        "passed": bool(
            vs_ident is not None
            and abs(vs_ident - 1.0) < 1e-6
            and vs_ortho is not None
            and abs(vs_ortho - 10.0) < 1e-6
            and m1_zero["M1.S1"] == 0.0
            and m1_one["M1.S1"] is not None
            and m1_one["M1.S1"] > 0
        ),
    }


def v2_null_control(
    *,
    pair_ids: list[str],
    question_emb_by_pair: dict[str, np.ndarray],
    req_emb_by_pair: dict[str, np.ndarray],
    requirements_by_pair: dict[str, list[dict]],
    tau_by_pair: dict[str, float],
    n_perm: int = NULL_PERMUTATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Score each B0 set against a different pair's requirements."""
    usable = [
        p
        for p in pair_ids
        if p in question_emb_by_pair
        and p in req_emb_by_pair
        and p in requirements_by_pair
        and p in tau_by_pair
    ]
    rng = np.random.default_rng(seed)
    true_m2, true_m3 = [], []
    for p in usable:
        s = score_with_assignment(
            question_embeddings=question_emb_by_pair[p],
            req_embeddings=req_emb_by_pair[p],
            requirements=requirements_by_pair[p],
            tau=tau_by_pair[p],
        )
        if s["M2.P"] is not None:
            true_m2.append(s["M2.P"])
        if s["M3.P"] is not None:
            true_m3.append(s["M3.P"])

    null_m2_all: list[float] = []
    null_m3_all: list[float] = []
    for _ in range(n_perm):
        perm = usable.copy()
        rng.shuffle(perm)
        # derangement-ish: ensure no self-match when possible
        for i, p in enumerate(usable):
            donor = perm[i]
            if donor == p and len(usable) > 1:
                donor = perm[(i + 1) % len(usable)]
            s = score_with_assignment(
                question_embeddings=question_emb_by_pair[p],
                req_embeddings=req_emb_by_pair[donor],
                requirements=requirements_by_pair[donor],
                tau=tau_by_pair[donor],
            )
            if s["M2.P"] is not None:
                null_m2_all.append(float(s["M2.P"]))
            if s["M3.P"] is not None:
                null_m3_all.append(float(s["M3.P"]))

    def _mean(xs: list[float]) -> float | None:
        return float(np.mean(xs)) if xs else None

    return {
        "n_pairs": len(usable),
        "n_permutations": n_perm,
        "true_M2.P_mean": _mean(true_m2),
        "null_M2.P_mean": _mean(null_m2_all),
        "gap_M2.P": (
            None
            if not true_m2 or not null_m2_all
            else float(np.mean(true_m2) - np.mean(null_m2_all))
        ),
        "true_M3.P_mean": _mean(true_m3),
        "null_M3.P_mean": _mean(null_m3_all),
        "gap_M3.P": (
            None
            if not true_m3 or not null_m3_all
            else float(np.mean(true_m3) - np.mean(null_m3_all))
        ),
    }


def v3_duplicate_injection(
    *,
    question_emb_by_pair: dict[str, np.ndarray],
    n_replace: int = DUPLICATE_INJECT_N,
    seed: int = SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rise = 0
    fall = 0
    n = 0
    for _pid, E in question_emb_by_pair.items():
        E = np.asarray(E, dtype=np.float64)
        if E.shape[0] < n_replace + 1:
            continue
        m1_before = m1_redundancy(E)["M1.P"]
        m4_before = m4_diversity(E)["M4.P"]
        inj = E.copy()
        # Replace last n_replace with copies of earlier questions
        sources = rng.choice(E.shape[0] - n_replace, size=n_replace, replace=False)
        for k, src in enumerate(sources):
            inj[E.shape[0] - 1 - k] = E[int(src)]
        m1_after = m1_redundancy(inj)["M1.P"]
        m4_after = m4_diversity(inj)["M4.P"]
        n += 1
        if m1_before is not None and m1_after is not None and m1_after > m1_before:
            rise += 1
        if m4_before is not None and m4_after is not None and m4_after < m4_before:
            fall += 1
    return {
        "n_sets": n,
        "fraction_M1P_rises": (rise / n) if n else None,
        "fraction_M4P_falls": (fall / n) if n else None,
    }


def v4_noise_floor(
    *,
    replicate_metrics: dict[str, list[dict[str, float | None]]],
) -> dict[str, Any]:
    """Per-pair SD and between-replicate paired diffs on primary-30.

    replicate_metrics[pair_id] = list of {M1.P, M2.P, M3.P, M4.P} per replicate.
    """
    out: dict[str, Any] = {}
    for metric in ("M1.P", "M2.P", "M3.P", "M4.P"):
        sds = []
        paired_abs = []
        for _pid, reps in replicate_metrics.items():
            vals = [r.get(metric) for r in reps if r.get(metric) is not None]
            vals_f = [float(v) for v in vals if v is not None]
            if len(vals_f) >= 2:
                sds.append(float(np.std(vals_f, ddof=1)))
                for i in range(len(vals_f)):
                    for j in range(i + 1, len(vals_f)):
                        paired_abs.append(abs(vals_f[i] - vals_f[j]))
        out[metric] = {
            "mean_per_pair_sd": float(np.mean(sds)) if sds else None,
            "median_per_pair_sd": float(np.median(sds)) if sds else None,
            "mean_abs_replicate_diff": float(np.mean(paired_abs)) if paired_abs else None,
            "smallest_distinguishable_paired_mean_diff": (
                float(np.mean(paired_abs)) if paired_abs else None
            ),
            "n_pairs_with_reps": len(sds),
        }
    return out


def write_assignment_audit_csv(
    path,
    rows: list[dict[str, Any]],
) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id",
        "question_text",
        "requirement_text",
        "rank",
        "cosine",
        "linked",
        "human_judges_related",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "pair_id": r.get("pair_id"),
                    "question_text": r.get("question_text"),
                    "requirement_text": r.get("requirement_text"),
                    "rank": r.get("rank"),
                    "cosine": r.get("cosine"),
                    "linked": r.get("linked"),
                    "human_judges_related": "",
                    "notes": "",
                }
            )
