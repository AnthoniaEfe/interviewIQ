"""Question→requirement assignment with leave-one-pair-out calibrated thresholds."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.services.eval_metrics.constants import (
    FINAL_N,
    FIXED_ASSIGNMENT_THRESHOLDS,
    SEED,
    TAU_PERCENTILE,
)
from app.services.eval_metrics.metrics_core import m2_m3_from_assignment


def calibrate_tau_p(
    *,
    req_embeddings: np.ndarray,
    other_question_embeddings: np.ndarray,
    percentile: float = TAU_PERCENTILE,
) -> float:
    """tau_p = percentile of max_r cos(q', r) over questions from other pairs."""
    R = np.asarray(req_embeddings, dtype=np.float64)
    Q = np.asarray(other_question_embeddings, dtype=np.float64)
    if R.size == 0 or Q.size == 0:
        return 1.0
    Rn = R / np.clip(np.linalg.norm(R, axis=1, keepdims=True), 1e-12, None)
    Qn = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12, None)
    sim = Qn @ Rn.T  # (nq_other, n_req)
    maxima = sim.max(axis=1)
    return float(np.percentile(maxima, percentile))


def assign_questions(
    *,
    question_embeddings: np.ndarray,
    req_embeddings: np.ndarray,
    requirements: list[dict[str, Any]],
    tau: float | None,
    threshold_free: bool = False,
) -> dict[str, Any]:
    """Link each question to argmax requirement if cos >= tau (unless threshold-free)."""
    Q = np.asarray(question_embeddings, dtype=np.float64)
    R = np.asarray(req_embeddings, dtype=np.float64)
    n_q = Q.shape[0] if Q.ndim == 2 else 0
    n_r = R.shape[0] if R.ndim == 2 else 0
    if n_q == 0 or n_r == 0:
        return {
            "linked": [False] * n_q,
            "r_star_indices": [None] * n_q,
            "cosine_to_r_star": [None] * n_q,
            "tau": tau,
        }
    Qn = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12, None)
    Rn = R / np.clip(np.linalg.norm(R, axis=1, keepdims=True), 1e-12, None)
    sim = Qn @ Rn.T
    linked: list[bool] = []
    idxs: list[int | None] = []
    cosines: list[float | None] = []
    for i in range(n_q):
        j = int(np.argmax(sim[i]))
        c = float(sim[i, j])
        if threshold_free or (tau is not None and c >= tau):
            linked.append(True)
            idxs.append(j)
            cosines.append(c)
        else:
            linked.append(False)
            idxs.append(j)  # still record argmax for audit
            cosines.append(c)
    return {
        "linked": linked,
        "r_star_indices": idxs,
        "cosine_to_r_star": cosines,
        "tau": tau,
        "similarity_matrix": sim,
    }


def score_with_assignment(
    *,
    question_embeddings: np.ndarray,
    req_embeddings: np.ndarray,
    requirements: list[dict[str, Any]],
    tau: float | None,
    threshold_free: bool = False,
    n_questions: int = FINAL_N,
) -> dict[str, Any]:
    asg = assign_questions(
        question_embeddings=question_embeddings,
        req_embeddings=req_embeddings,
        requirements=requirements,
        tau=tau,
        threshold_free=threshold_free,
    )
    metrics = m2_m3_from_assignment(
        linked=asg["linked"],
        r_star_indices=asg["r_star_indices"],
        requirements=requirements,
        n_questions=n_questions,
    )
    metrics["assignment"] = {
        "linked": asg["linked"],
        "r_star_indices": asg["r_star_indices"],
        "cosine_to_r_star": asg["cosine_to_r_star"],
        "tau": asg["tau"],
        "threshold_free": threshold_free,
    }
    return metrics


def sensitivity_assignment_scores(
    *,
    question_embeddings: np.ndarray,
    req_embeddings: np.ndarray,
    requirements: list[dict[str, Any]],
    tau_primary: float,
    n_questions: int = FINAL_N,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    primary = score_with_assignment(
        question_embeddings=question_embeddings,
        req_embeddings=req_embeddings,
        requirements=requirements,
        tau=tau_primary,
        n_questions=n_questions,
    )
    out["calibrated"] = {k: primary[k] for k in primary if k != "assignment"}
    out["calibrated"]["tau"] = tau_primary
    free = score_with_assignment(
        question_embeddings=question_embeddings,
        req_embeddings=req_embeddings,
        requirements=requirements,
        tau=None,
        threshold_free=True,
        n_questions=n_questions,
    )
    out["threshold_free"] = {k: free[k] for k in free if k != "assignment"}
    for thr in FIXED_ASSIGNMENT_THRESHOLDS:
        scored = score_with_assignment(
            question_embeddings=question_embeddings,
            req_embeddings=req_embeddings,
            requirements=requirements,
            tau=float(thr),
            n_questions=n_questions,
        )
        out[f"fixed_{thr}"] = {k: scored[k] for k in scored if k != "assignment"}
    out["primary_assignment"] = primary.get("assignment")
    return out


def hash_tau_map(tau_map: dict[str, float]) -> str:
    import hashlib
    import json

    payload = json.dumps(tau_map, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_assignment_audit_rows(
    candidates: list[dict[str, Any]],
    *,
    n: int = 40,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Stratified linked/unlinked sample for human audit sheet (blank judge columns)."""
    rng = np.random.default_rng(seed)
    linked = [c for c in candidates if c.get("linked")]
    unlinked = [c for c in candidates if not c.get("linked")]
    half = n // 2
    rows: list[dict[str, Any]] = []
    for pool, k in ((linked, half), (unlinked, n - half)):
        if not pool:
            continue
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        for i in idx:
            rows.append(pool[int(i)])
    # pad if one stratum empty
    if len(rows) < n and candidates:
        need = n - len(rows)
        remaining = [c for c in candidates if c not in rows]
        if remaining:
            idx = rng.choice(len(remaining), size=min(need, len(remaining)), replace=False)
            for i in idx:
                rows.append(remaining[int(i)])
    rng.shuffle(rows)
    return rows[:n]
