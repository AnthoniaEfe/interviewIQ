"""Selector T-prime. Does not modify weighted MMR, plain MMR, or profile-aware generation.

The score and tie-break match ``competency_weighted_mmr`` in
``app.services.mmr.select``: lambda 0.7, relevance = priority * weight,
redundancy = max cosine to the selected set, strict ``>`` while walking
``set(range(n))`` in insertion order. No random draw. The seed is recorded
and unused.

Parameters are the values fixed in ``docs/selector_tprime_prereg.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.schemas.domain import InterviewQuestion
from app.schemas.pool import POOL_DIMENSIONS

K = 10
LAMBDA = 0.7
RESERVED_NON_GAP_SLOTS = 2
QUOTA = 1
NON_GAP_LABELS = frozenset({"partial_match", "strong_match"})
SEED = 42


@dataclass(frozen=True)
class Relaxation:
    phase: str
    reason: str
    dimension: str | None = None
    reserved_index: int | None = None


@dataclass
class TPrimeSelection:
    selected_indices: list[int]
    dimension_counts: dict[str, int]
    relaxations: list[Relaxation] = field(default_factory=list)
    quota_met: bool = False
    seed: int = SEED
    k: int = K
    lambda_: float = LAMBDA
    reserved_slots: int = RESERVED_NON_GAP_SLOTS


def _eligible_non_gap(
    target_gaps: list[str],
    match_label_by_competency: dict[str, str],
) -> list[bool]:
    flags: list[bool] = []
    for target in target_gaps:
        label = match_label_by_competency.get(target)
        flags.append(label in NON_GAP_LABELS)
    return flags


def _step(
    *,
    remaining: set[int],
    allowed: set[int],
    selected: list[int],
    relevance: np.ndarray,
    similarity: np.ndarray,
) -> int | None:
    """One MMR step. Returns None when ``allowed`` is empty."""
    if not allowed:
        return None
    best_i = None
    best_score = -1e18
    for i in remaining:
        if i not in allowed:
            continue
        redundancy = 0.0
        if selected:
            redundancy = float(max(similarity[i, j] for j in selected))
        score = LAMBDA * float(relevance[i]) - (1.0 - LAMBDA) * redundancy
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def select_tprime(
    questions: list[InterviewQuestion],
    similarity_matrix: np.ndarray,
    weights: list[float],
    dimensions: list[str],
    target_gaps: list[str],
    match_label_by_competency: dict[str, str],
    *,
    use_weights: bool = True,
) -> TPrimeSelection:
    """Select k questions. ``use_weights=False`` is the pre-registered ablation."""
    n = len(questions)
    if not (len(weights) == len(dimensions) == len(target_gaps) == n):
        raise ValueError("weights, dimensions, and target_gaps must match the pool length")
    sim = np.asarray(similarity_matrix, dtype=float)
    if sim.shape != (n, n):
        raise ValueError(f"similarity_matrix shape {sim.shape} != {(n, n)}")
    relevance = np.array(
        [
            float(questions[i].priority) * (float(weights[i]) if use_weights else 1.0)
            for i in range(n)
        ],
        dtype=float,
    )
    non_gap = _eligible_non_gap(target_gaps, match_label_by_competency)
    pool_counts: dict[str, int] = {dimension: 0 for dimension in POOL_DIMENSIONS}
    for dimension in dimensions:
        if dimension in pool_counts:
            pool_counts[dimension] += 1

    selected: list[int] = []
    remaining = set(range(n))
    relaxations: list[Relaxation] = []
    target = min(K, n)
    if n < K:
        relaxations.append(Relaxation(phase="length", reason="pool_shorter_than_k"))

    for dimension in POOL_DIMENSIONS:
        if len(selected) >= target:
            break
        allowed = {i for i in remaining if dimensions[i] == dimension}
        if not allowed:
            reason = "zero_availability" if pool_counts[dimension] == 0 else "exhausted"
            relaxations.append(Relaxation(phase="quota", reason=reason, dimension=dimension))
            continue
        choice = _step(
            remaining=remaining,
            allowed=allowed,
            selected=selected,
            relevance=relevance,
            similarity=sim,
        )
        if choice is None:
            relaxations.append(Relaxation(phase="quota", reason="exhausted", dimension=dimension))
            continue
        selected.append(choice)
        remaining.remove(choice)

    for slot in range(RESERVED_NON_GAP_SLOTS):
        if len(selected) >= target:
            break
        allowed = {i for i in remaining if non_gap[i]}
        if not allowed:
            relaxations.append(
                Relaxation(
                    phase="reserved",
                    reason="no_partial_or_strong_remaining",
                    reserved_index=slot,
                )
            )
            continue
        choice = _step(
            remaining=remaining,
            allowed=allowed,
            selected=selected,
            relevance=relevance,
            similarity=sim,
        )
        if choice is None:
            relaxations.append(
                Relaxation(
                    phase="reserved",
                    reason="no_partial_or_strong_remaining",
                    reserved_index=slot,
                )
            )
            continue
        selected.append(choice)
        remaining.remove(choice)

    while remaining and len(selected) < target:
        choice = _step(
            remaining=remaining,
            allowed=set(remaining),
            selected=selected,
            relevance=relevance,
            similarity=sim,
        )
        if choice is None:
            break
        selected.append(choice)
        remaining.remove(choice)

    counts = {dimension: 0 for dimension in POOL_DIMENSIONS}
    for index in selected:
        dimension = dimensions[index]
        if dimension in counts:
            counts[dimension] += 1
    quota_met = all(counts[dimension] >= QUOTA for dimension in POOL_DIMENSIONS)
    return TPrimeSelection(
        selected_indices=selected,
        dimension_counts=counts,
        relaxations=relaxations,
        quota_met=quota_met,
    )
