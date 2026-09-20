from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.schemas.domain import InterviewQuestion


@dataclass
class MMRSelection:
    selected_indices: list[int]
    relevance_scores: list[float]
    redundancy_scores: list[float]
    mmr_scores: list[float]
    competency_weights: list[float]


def competency_weighted_mmr(
    questions: list[InterviewQuestion],
    similarity_matrix: np.ndarray,
    weights: list[float],
    *,
    lambda_: float,
    k: int,
) -> MMRSelection:
    """Select k questions balancing relevance, diversity, and competency priority.

    MMR = λ * relevance - (1-λ) * max_similarity_to_selected
    relevance incorporates competency weight.
    """
    n = len(questions)
    if n == 0 or k <= 0:
        return MMRSelection([], [], [], [], [])

    relevance = np.array(
        [q.priority * (weights[i] if i < len(weights) else 1.0) for i, q in enumerate(questions)],
        dtype=float,
    )
    selected: list[int] = []
    remaining = set(range(n))
    recorded_rel: list[float] = []
    recorded_red: list[float] = []
    recorded_mmr: list[float] = []
    recorded_w: list[float] = []

    while remaining and len(selected) < k:
        best_i = None
        best_score = -1e18
        best_red = 0.0
        for i in remaining:
            redundancy = 0.0
            if selected:
                redundancy = float(max(similarity_matrix[i, j] for j in selected))
            score = lambda_ * float(relevance[i]) - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_i = i
                best_red = redundancy
        assert best_i is not None
        selected.append(best_i)
        remaining.remove(best_i)
        recorded_rel.append(float(relevance[best_i]))
        recorded_red.append(best_red)
        recorded_mmr.append(best_score)
        recorded_w.append(weights[best_i] if best_i < len(weights) else 1.0)

    return MMRSelection(
        selected_indices=selected,
        relevance_scores=recorded_rel,
        redundancy_scores=recorded_red,
        mmr_scores=recorded_mmr,
        competency_weights=recorded_w,
    )


def redundant_pairs(similarity_matrix: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    n = similarity_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            score = float(similarity_matrix[i, j])
            if score >= threshold:
                pairs.append((i, j, score))
    return pairs
