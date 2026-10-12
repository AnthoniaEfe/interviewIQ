"""Shared embedding + per-set metric computation."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.services.embeddings.service import EmbeddingService
from app.services.eval_metrics.assignment import sensitivity_assignment_scores
from app.services.eval_metrics.constants import FINAL_N
from app.services.eval_metrics.metrics_core import (
    lexical_specificity_proxy,
    m1_redundancy,
    m4_diversity,
)


def embed_texts(embeddings: EmbeddingService, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float64)
    return np.asarray(embeddings.embed_batch(texts), dtype=np.float64)


def compute_text_only_metrics(
    texts: list[str],
    embeddings: EmbeddingService,
    *,
    precomputed: np.ndarray | None = None,
) -> dict[str, Any]:
    E = precomputed if precomputed is not None else embed_texts(embeddings, texts)
    m1 = m1_redundancy(E)
    m4 = m4_diversity(E)
    return {
        **m1,
        **m4,
        "lexical_specificity_proxy": lexical_specificity_proxy(texts),
        "n_questions": len(texts),
        "_embeddings": E,
    }


def compute_full_metrics(
    texts: list[str],
    embeddings: EmbeddingService,
    *,
    requirements: list[dict[str, Any]] | None,
    tau: float | None,
    precomputed_q: np.ndarray | None = None,
    precomputed_r: np.ndarray | None = None,
) -> dict[str, Any]:
    base = compute_text_only_metrics(texts, embeddings, precomputed=precomputed_q)
    E = base.pop("_embeddings")
    out: dict[str, Any] = {k: v for k, v in base.items()}
    if not requirements or tau is None:
        out.update(
            {
                "M2.P": None,
                "M2.S1": None,
                "M2.S2": None,
                "M3.P": None,
                "M3.S1": None,
                "M3.S1_undefined": True,
                "M3.S2": None,
                "assignment_available": False,
            }
        )
        return out
    req_texts = [str(r.get("job_competency") or "") for r in requirements]
    R = (
        precomputed_r
        if precomputed_r is not None
        else embed_texts(embeddings, req_texts)
    )
    sens = sensitivity_assignment_scores(
        question_embeddings=E,
        req_embeddings=R,
        requirements=requirements,
        tau_primary=float(tau),
        n_questions=FINAL_N,
    )
    cal = sens["calibrated"]
    out.update(
        {
            "M2.P": cal.get("M2.P"),
            "M2.S1": cal.get("M2.S1"),
            "M2.S2": cal.get("M2.S2"),
            "M3.P": cal.get("M3.P"),
            "M3.S1": cal.get("M3.S1"),
            "M3.S1_undefined": cal.get("M3.S1_undefined"),
            "M3.S2": cal.get("M3.S2"),
            "n_requirements": cal.get("n_requirements"),
            "n_gap_requirements": cal.get("n_gap_requirements"),
            "linked_category_shares": cal.get("linked_category_shares"),
            "assignment_available": True,
            "tau": tau,
            "sensitivity": {
                k: v
                for k, v in sens.items()
                if k != "primary_assignment"
            },
            "primary_assignment": sens.get("primary_assignment"),
        }
    )
    return out
