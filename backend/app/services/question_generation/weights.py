"""Question-to-weight mapping for competency-weighted MMR (frozen rule)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.schemas.domain import GapMatrix, ProfileGap
from app.schemas.pool import PROFILE_PROBE
from app.services.embeddings.service import EmbeddingService


@dataclass
class WeightMappingOutcome:
    weight: float
    rule: str  # exact | profile_probe | embedding_fallback
    matched_competency: str | None
    cosine: float | None = None


def mean_gap_weight(gaps: list[ProfileGap]) -> float:
    if not gaps:
        return 1.0
    return float(sum(g.weight for g in gaps) / len(gaps))


def map_question_weight(
    target_gap: str | None,
    gap_matrix: GapMatrix,
    embeddings: EmbeddingService,
    *,
    job_competency_vectors: dict[str, np.ndarray] | None = None,
) -> WeightMappingOutcome:
    """Frozen weight-mapping rule (treatment pilot protocol §0.3 item 2)."""
    gaps = gap_matrix.gaps
    by_name = {g.job_competency: g for g in gaps}
    mean_w = mean_gap_weight(gaps)

    if target_gap is None or not str(target_gap).strip():
        return WeightMappingOutcome(
            weight=mean_w,
            rule="embedding_fallback",
            matched_competency=None,
            cosine=None,
        )

    key = str(target_gap).strip()
    if key == PROFILE_PROBE:
        return WeightMappingOutcome(
            weight=mean_w,
            rule="profile_probe",
            matched_competency=PROFILE_PROBE,
            cosine=None,
        )

    if key in by_name:
        return WeightMappingOutcome(
            weight=float(by_name[key].weight),
            rule="exact",
            matched_competency=key,
            cosine=1.0,
        )

    if not gaps:
        return WeightMappingOutcome(
            weight=1.0,
            rule="embedding_fallback",
            matched_competency=None,
            cosine=None,
        )

    query_vec = embeddings.embed_text(key)
    best_name: str | None = None
    best_cos = -1.0
    for name in by_name:
        if job_competency_vectors is not None and name in job_competency_vectors:
            vec = job_competency_vectors[name]
        else:
            vec = embeddings.embed_text(name)
        denom = float(np.linalg.norm(query_vec) * np.linalg.norm(vec))
        cos = float(np.dot(query_vec, vec) / denom) if denom > 0 else 0.0
        if cos > best_cos:
            best_cos = cos
            best_name = name
    assert best_name is not None
    return WeightMappingOutcome(
        weight=float(by_name[best_name].weight),
        rule="embedding_fallback",
        matched_competency=best_name,
        cosine=best_cos,
    )


def map_weights_for_questions(
    target_gaps: list[str | None],
    gap_matrix: GapMatrix,
    embeddings: EmbeddingService,
) -> tuple[list[float], list[WeightMappingOutcome]]:
    names = [g.job_competency for g in gap_matrix.gaps]
    vectors: dict[str, np.ndarray] = {}
    if names:
        mat = embeddings.embed_batch(names)
        for i, name in enumerate(names):
            vectors[name] = mat[i]
    outcomes = [
        map_question_weight(tg, gap_matrix, embeddings, job_competency_vectors=vectors)
        for tg in target_gaps
    ]
    return [o.weight for o in outcomes], outcomes
